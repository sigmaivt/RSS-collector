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
        async with conn.execute(
            """SELECT * FROM channel_jobs
               WHERE channel_id = ? AND state = ?
               AND (leased_at IS NULL OR leased_at < ?)
               ORDER BY created_at
               LIMIT ?""",
            (channel_id, state.value, stale_threshold, limit),
        ) as cur:
            rows = await cur.fetchall()

        now = datetime.utcnow().isoformat()
        ids = [row["id"] for row in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            await conn.execute(
                f"UPDATE channel_jobs SET leased_at = ? WHERE id IN ({placeholders})",
                [now, *ids],
            )
            await conn.commit()

    return [_job_from_row(dict(r)) for r in rows]


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


async def increment_outbox_attempt(entry_id: int, error: str) -> None:
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE telegram_outbox SET attempts=attempts+1, last_error=? WHERE id=?",
            (error, entry_id),
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


# ── helpers ────────────────────────────────────────────────────────────────

def _job_from_row(row: dict) -> ChannelJob:
    row["state"] = JobState(row["state"])
    return ChannelJob(**row)
