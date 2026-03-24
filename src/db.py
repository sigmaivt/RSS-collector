"""SQLite database access layer (async, WAL mode)."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator, Optional

import aiosqlite

from models import ChannelJob, JobState, RssItem, TelegramOutboxEntry


_DB_PATH: str = "./data/pipeline.db"


def set_db_path(path: str) -> None:
    global _DB_PATH
    _DB_PATH = path


@asynccontextmanager
async def get_conn() -> AsyncIterator[aiosqlite.Connection]:
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        yield conn


async def init_db(schema_path: str | None = None) -> None:
    if schema_path is None:
        schema_path = str(Path(__file__).parent / "schema.sql")
    async with get_conn() as conn:
        with open(schema_path, encoding="utf-8") as f:
            await conn.executescript(f.read())
        await conn.commit()


def make_item_id(source_id: str, guid: str) -> str:
    return hashlib.sha256(f"{source_id}:{guid}".encode()).hexdigest()


# ── items ──────────────────────────────────────────────────────────────────

async def upsert_item(item: RssItem) -> bool:
    """Insert item; return True if new, False if already existed."""
    async with get_conn() as conn:
        cursor = await conn.execute(
            """INSERT OR IGNORE INTO items
               (item_id, source_id, guid, link, title, raw_content, clean_text, published, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                item.item_id,
                item.source_id,
                item.guid,
                item.link,
                item.title,
                item.raw_content,
                item.clean_text,
                item.published.isoformat() if item.published else None,
                (item.fetched_at or datetime.utcnow()).isoformat(),
            ),
        )
        await conn.commit()
        return cursor.rowcount == 1


async def get_item(item_id: str) -> Optional[RssItem]:
    async with get_conn() as conn:
        async with conn.execute(
            "SELECT * FROM items WHERE item_id = ?", (item_id,)
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return RssItem(**dict(row))


# ── channel_jobs ───────────────────────────────────────────────────────────

async def create_job(item_id: str, channel_id: str) -> bool:
    """Create NEW job; return True if created, False if already exists."""
    async with get_conn() as conn:
        cursor = await conn.execute(
            """INSERT OR IGNORE INTO channel_jobs (item_id, channel_id, state, created_at, updated_at)
               VALUES (?, ?, 'NEW', ?, ?)""",
            (item_id, channel_id, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
        )
        await conn.commit()
        return cursor.rowcount == 1


async def acquire_jobs(
    channel_id: str,
    state: JobState,
    limit: int,
    lease_minutes: int,
) -> list[ChannelJob]:
    """Fetch and lease jobs for processing (atomic lease pattern)."""
    async with get_conn() as conn:
        stale_threshold = (
            datetime.utcnow() - timedelta(minutes=lease_minutes)
        ).isoformat()
        leased_rows: list[aiosqlite.Row] = []
        await conn.execute("BEGIN IMMEDIATE")
        try:
            async with conn.execute(
                """SELECT id FROM channel_jobs
                   WHERE channel_id = ? AND state = ?
                   AND (leased_at IS NULL OR leased_at < ?)
                   ORDER BY created_at
                   LIMIT ?""",
                (channel_id, state.value, stale_threshold, limit),
            ) as cur:
                rows = await cur.fetchall()

            ids = [row["id"] for row in rows]
            if ids:
                now = datetime.utcnow().isoformat()
                placeholders = ",".join("?" * len(ids))
                await conn.execute(
                    f"""UPDATE channel_jobs
                        SET leased_at = ?, updated_at = ?
                        WHERE id IN ({placeholders})
                          AND (leased_at IS NULL OR leased_at < ?)""",
                    [now, now, *ids, stale_threshold],
                )
                async with conn.execute(
                    f"""SELECT * FROM channel_jobs
                        WHERE id IN ({placeholders})
                        ORDER BY created_at""",
                    ids,
                ) as cur:
                    leased_rows = await cur.fetchall()
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    return [_job_from_row(dict(r)) for r in leased_rows]


async def update_job(job: ChannelJob) -> None:
    async with get_conn() as conn:
        await conn.execute(
            """UPDATE channel_jobs
               SET state=?, classifier_result=?, summary=?, attempts=?,
                   last_error=?, leased_at=?, updated_at=?
               WHERE id=?""",
            (
                job.state.value,
                job.classifier_result,
                job.summary,
                job.attempts,
                job.last_error,
                job.leased_at.isoformat() if job.leased_at else None,
                datetime.utcnow().isoformat(),
                job.id,
            ),
        )
        await conn.commit()


async def update_job_state_by_item_channel(
    item_id: str,
    channel_id: str,
    state: JobState,
    last_error: str | None = None,
) -> None:
    async with get_conn() as conn:
        await conn.execute(
            """UPDATE channel_jobs
               SET state=?, last_error=?, leased_at=NULL, updated_at=?
               WHERE item_id=? AND channel_id=?""",
            (
                state.value,
                last_error,
                datetime.utcnow().isoformat(),
                item_id,
                channel_id,
            ),
        )
        await conn.commit()


async def move_to_dead_letter(job: ChannelJob, error: str) -> None:
    async with get_conn() as conn:
        await conn.execute(
            """INSERT INTO dead_letter_jobs
               (item_id, channel_id, original_state, error, attempts, moved_at)
               VALUES (?,?,?,?,?,?)""",
            (
                job.item_id,
                job.channel_id,
                job.state.value,
                error,
                job.attempts,
                datetime.utcnow().isoformat(),
            ),
        )
        await conn.execute("DELETE FROM channel_jobs WHERE id = ?", (job.id,))
        await conn.commit()


# ── telegram_outbox ────────────────────────────────────────────────────────

async def enqueue_outbox(entry: TelegramOutboxEntry) -> bool:
    async with get_conn() as conn:
        cursor = await conn.execute(
            """INSERT OR IGNORE INTO telegram_outbox
               (item_id, channel_id, chat_id, message_text, created_at)
               VALUES (?,?,?,?,?)""",
            (
                entry.item_id,
                entry.channel_id,
                entry.chat_id,
                entry.message_text,
                datetime.utcnow().isoformat(),
            ),
        )
        await conn.commit()
        return cursor.rowcount == 1


async def get_pending_outbox(limit: int = 20) -> list[TelegramOutboxEntry]:
    async with get_conn() as conn:
        async with conn.execute(
            "SELECT * FROM telegram_outbox WHERE sent = 0 ORDER BY created_at LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [TelegramOutboxEntry(**dict(r)) for r in rows]


async def mark_outbox_sent(entry_id: int, tg_message_id: int) -> None:
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE telegram_outbox SET sent=1, tg_message_id=?, sent_at=? WHERE id=?",
            (tg_message_id, datetime.utcnow().isoformat(), entry_id),
        )
        await conn.commit()


async def increment_outbox_attempt(entry_id: int, error: str) -> int:
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE telegram_outbox SET attempts=attempts+1, last_error=? WHERE id=?",
            (error, entry_id),
        )
        async with conn.execute(
            "SELECT attempts FROM telegram_outbox WHERE id=?",
            (entry_id,),
        ) as cur:
            row = await cur.fetchone()
        await conn.commit()
        return int(row["attempts"]) if row else 0


async def mark_outbox_failed(entry_id: int, error: str) -> None:
    async with get_conn() as conn:
        await conn.execute(
            """UPDATE telegram_outbox
               SET sent=1, last_error=?, sent_at=?
               WHERE id=?""",
            (error, datetime.utcnow().isoformat(), entry_id),
        )
        await conn.commit()


# ── feed_poll_log ──────────────────────────────────────────────────────────

async def log_poll(
    source_id: str,
    items_found: int,
    items_new: int,
    error: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> None:
    async with get_conn() as conn:
        await conn.execute(
            """INSERT INTO feed_poll_log
               (source_id, polled_at, items_found, items_new, error, duration_ms)
               VALUES (?,?,?,?,?,?)""",
            (
                source_id,
                datetime.utcnow().isoformat(),
                items_found,
                items_new,
                error,
                duration_ms,
            ),
        )
        await conn.commit()


# ── metrics ────────────────────────────────────────────────────────────────

async def record_metric(name: str, value: float, tags: Optional[dict] = None) -> None:
    async with get_conn() as conn:
        await conn.execute(
            "INSERT INTO pipeline_metrics (ts, metric_name, metric_value, tags) VALUES (?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                name,
                value,
                json.dumps(tags) if tags else None,
            ),
        )
        await conn.commit()


async def get_daily_stats(channel_id: str) -> dict:
    async with get_conn() as conn:
        today = datetime.utcnow().date().isoformat()
        async with conn.execute(
            """SELECT
               COUNT(*) as total,
               SUM(CASE WHEN state='SENT' THEN 1 ELSE 0 END) as sent,
               SUM(CASE WHEN state='REJECTED' THEN 1 ELSE 0 END) as rejected,
               SUM(CASE WHEN state='SEND_FAILED' THEN 1 ELSE 0 END) as failed
               FROM channel_jobs
               WHERE channel_id = ? AND DATE(created_at) = ?""",
            (channel_id, today),
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else {}


# -- proactive reports ---------------------------------------------------------

_PROACTIVE_INCLUDED_STATES = (
    JobState.VALIDATED.value,
    JobState.SUMMARIZING.value,
    JobState.READY_TO_SEND.value,
    JobState.OUTBOX_PENDING.value,
    JobState.SENT.value,
    JobState.SEND_FAILED.value,
)


async def get_validated_items_since(
    channel_id: str,
    hours: int = 24,
    limit: int = 200,
) -> list[dict]:
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    placeholders = ",".join(["?"] * len(_PROACTIVE_INCLUDED_STATES))
    query = f"""
        SELECT
            i.item_id,
            i.source_id,
            i.title,
            i.link,
            i.clean_text,
            cj.summary,
            cj.updated_at
        FROM channel_jobs cj
        JOIN items i ON i.item_id = cj.item_id
        WHERE cj.channel_id = ?
          AND cj.updated_at >= ?
          AND cj.state IN ({placeholders})
        ORDER BY cj.updated_at DESC
        LIMIT ?
    """
    params: list = [channel_id, since, *_PROACTIVE_INCLUDED_STATES, limit]
    async with get_conn() as conn:
        async with conn.execute(query, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def check_proactive_report_exists(
    channel_id: str,
    report_type: str,
    report_date: str,
) -> bool:
    async with get_conn() as conn:
        async with conn.execute(
            """SELECT 1 FROM proactive_reports
               WHERE channel_id=? AND report_type=? AND report_date=?
                 AND status IN ('created', 'sent')
               LIMIT 1""",
            (channel_id, report_type, report_date),
        ) as cur:
            row = await cur.fetchone()
    return row is not None


async def check_recent_proactive_report(
    channel_id: str,
    report_type: str,
    hours: int = 24,
) -> bool:
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    async with get_conn() as conn:
        async with conn.execute(
            """SELECT 1 FROM proactive_reports
               WHERE channel_id=?
                 AND report_type=?
                 AND status IN ('created', 'sent')
                 AND created_at >= ?
               LIMIT 1""",
            (channel_id, report_type, since),
        ) as cur:
            row = await cur.fetchone()
    return row is not None


async def save_proactive_report(
    channel_id: str,
    report_type: str,
    report_date: str,
    status: str = "created",
) -> int:
    async with get_conn() as conn:
        now = datetime.utcnow().isoformat()
        await conn.execute(
            """INSERT OR IGNORE INTO proactive_reports
               (channel_id, report_type, report_date, status, created_at, error_message, skipped_reason, sent_at)
               VALUES (?,?,?,?,?,?,?,NULL)""",
            (
                channel_id,
                report_type,
                report_date,
                status,
                now,
                None,
                None,
            ),
        )
        await conn.execute(
            """UPDATE proactive_reports
               SET status=?, created_at=?, error_message=NULL, skipped_reason=NULL, sent_at=NULL
               WHERE channel_id=? AND report_type=? AND report_date=?
                 AND status IN ('failed', 'skipped')""",
            (
                status,
                now,
                channel_id,
                report_type,
                report_date,
            ),
        )
        async with conn.execute(
            """SELECT id FROM proactive_reports
               WHERE channel_id=? AND report_type=? AND report_date=?
               LIMIT 1""",
            (channel_id, report_type, report_date),
        ) as cur:
            row = await cur.fetchone()
        await conn.commit()
    return int(row["id"])


async def update_proactive_report_status(
    report_id: int,
    status: str,
    error_message: Optional[str] = None,
    skipped_reason: Optional[str] = None,
    mark_sent: bool = False,
) -> None:
    sent_at = datetime.utcnow().isoformat() if mark_sent else None
    async with get_conn() as conn:
        await conn.execute(
            """UPDATE proactive_reports
               SET status=?, sent_at=COALESCE(?, sent_at),
                   error_message=?, skipped_reason=?
               WHERE id=?""",
            (
                status,
                sent_at,
                error_message,
                skipped_reason,
                report_id,
            ),
        )
        await conn.commit()


async def count_validated_items_since(channel_id: str, hours: int = 24) -> int:
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    placeholders = ",".join(["?"] * len(_PROACTIVE_INCLUDED_STATES))
    query = f"""
        SELECT COUNT(*) AS c
        FROM channel_jobs
        WHERE channel_id=?
          AND updated_at >= ?
          AND state IN ({placeholders})
    """
    params: list = [channel_id, since, *_PROACTIVE_INCLUDED_STATES]
    async with get_conn() as conn:
        async with conn.execute(query, params) as cur:
            row = await cur.fetchone()
    return int(row["c"]) if row else 0


# ── helpers ────────────────────────────────────────────────────────────────

def _job_from_row(row: dict) -> ChannelJob:
    row["state"] = JobState(row["state"])
    return ChannelJob(**row)
