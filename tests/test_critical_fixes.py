from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import db  # noqa: E402
import main as main_module  # noqa: E402
from models import ChannelConfig, JobState, RssItem  # noqa: E402
from telegram_sender import TelegramSender  # noqa: E402


class _DummyWorker:
    def __init__(self) -> None:
        self.calls = 0

    async def run_batch(self, channel: ChannelConfig) -> int:
        self.calls += 1
        return 0


class _DummySender:
    async def enqueue_job(self, job, channel) -> None:
        return None

    async def flush_outbox(self) -> int:
        return 0


class CriticalFixesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "pipeline.db")
        db.set_db_path(self.db_path)
        await db.init_db()

    async def asyncTearDown(self) -> None:
        self._tmp.cleanup()

    async def _seed_job(
        self,
        channel: ChannelConfig,
        state: JobState,
        *,
        summary: str | None = None,
    ) -> tuple[str, int]:
        item_id = f"{channel.id}-item-1"
        item = RssItem(
            item_id=item_id,
            source_id=f"{channel.id}-source",
            guid="guid-1",
            link="https://example.com/article",
            title="Title",
            raw_content="content",
            clean_text="clean",
        )
        await db.upsert_item(item)
        await db.create_job(item_id, channel.id)
        jobs = await db.acquire_jobs(channel.id, JobState.NEW, 1, 10)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        job.state = state
        job.summary = summary
        job.leased_at = None
        await db.update_job(job)
        return item_id, int(job.id)

    async def _job_state(self, item_id: str, channel_id: str) -> str:
        async with db.get_conn() as conn:
            async with conn.execute(
                "SELECT state FROM channel_jobs WHERE item_id=? AND channel_id=?",
                (item_id, channel_id),
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        return row["state"]

    async def _outbox_row(self, item_id: str, channel_id: str):
        async with db.get_conn() as conn:
            async with conn.execute(
                """SELECT sent, attempts, last_error
                   FROM telegram_outbox
                   WHERE item_id=? AND channel_id=?""",
                (item_id, channel_id),
            ) as cur:
                row = await cur.fetchone()
        return row

    async def test_pipeline_tick_respects_channel_schedule_and_marks_outbox_pending(self) -> None:
        channel = ChannelConfig(
            id="ai_models",
            name="AI Models",
            description="desc",
            classifier_prompt_file="classifier_ai_models.txt",
            summarizer_prompt_file="summarizer_ru.txt",
            telegram_chat_id="10001",
            schedule_minutes=60,
            enabled=True,
            sources=[],
        )
        item_id, _ = await self._seed_job(channel, JobState.READY_TO_SEND, summary="summary")

        classifier = _DummyWorker()
        summarizer = _DummyWorker()
        sender = _DummySender()
        settings = SimpleNamespace(max_text_length=5000, job_lease_minutes=10)
        channel_last_run = {channel.id: datetime.utcnow()}

        with patch.object(main_module, "poll_all_sources", new=AsyncMock(return_value=None)):
            await main_module.pipeline_tick(
                [channel],
                classifier,
                summarizer,
                sender,
                settings,
                channel_last_run,
            )
            self.assertEqual(classifier.calls, 0)
            self.assertEqual(summarizer.calls, 0)

            channel_last_run[channel.id] = datetime.utcnow() - timedelta(minutes=61)
            await main_module.pipeline_tick(
                [channel],
                classifier,
                summarizer,
                sender,
                settings,
                channel_last_run,
            )

        self.assertEqual(classifier.calls, 1)
        self.assertEqual(summarizer.calls, 1)
        self.assertEqual(
            await self._job_state(item_id, channel.id),
            JobState.OUTBOX_PENDING.value,
        )

    async def test_flush_outbox_sets_job_sent_on_success(self) -> None:
        channel = ChannelConfig(
            id="ai_coding",
            name="AI Coding",
            description="desc",
            classifier_prompt_file="classifier_ai_coding.txt",
            summarizer_prompt_file="summarizer_ru.txt",
            telegram_chat_id="10002",
            enabled=True,
            sources=[],
        )
        item_id, job_id = await self._seed_job(
            channel,
            JobState.OUTBOX_PENDING,
            summary="short summary",
        )

        settings = SimpleNamespace(tg_bot_token="token", max_send_retries=2)
        sender = TelegramSender(settings)

        jobs = await db.acquire_jobs(channel.id, JobState.OUTBOX_PENDING, 1, 10)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        job.id = job_id
        await sender.enqueue_job(job, channel)

        with patch.object(sender, "_send_message", new=AsyncMock(return_value=42)):
            sent = await sender.flush_outbox(max_retries=2)

        self.assertEqual(sent, 1)
        self.assertEqual(await self._job_state(item_id, channel.id), JobState.SENT.value)
        outbox = await self._outbox_row(item_id, channel.id)
        self.assertIsNotNone(outbox)
        self.assertEqual(outbox["sent"], 1)

    async def test_flush_outbox_marks_send_failed_after_retry_limit(self) -> None:
        channel = ChannelConfig(
            id="erp_mes",
            name="ERP/MES",
            description="desc",
            classifier_prompt_file="classifier_erp_mes.txt",
            summarizer_prompt_file="summarizer_ru.txt",
            telegram_chat_id="10003",
            enabled=True,
            sources=[],
        )
        item_id, job_id = await self._seed_job(
            channel,
            JobState.OUTBOX_PENDING,
            summary="summary",
        )

        settings = SimpleNamespace(tg_bot_token="token", max_send_retries=2)
        sender = TelegramSender(settings)

        jobs = await db.acquire_jobs(channel.id, JobState.OUTBOX_PENDING, 1, 10)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        job.id = job_id
        await sender.enqueue_job(job, channel)

        pending = await db.get_pending_outbox(limit=1)
        self.assertEqual(len(pending), 1)
        await db.increment_outbox_attempt(pending[0].id, "first failure")

        with patch.object(sender, "_send_message", new=AsyncMock(side_effect=RuntimeError("boom"))):
            sent = await sender.flush_outbox(max_retries=2)

        self.assertEqual(sent, 0)
        self.assertEqual(
            await self._job_state(item_id, channel.id),
            JobState.SEND_FAILED.value,
        )
        outbox = await self._outbox_row(item_id, channel.id)
        self.assertIsNotNone(outbox)
        self.assertEqual(outbox["sent"], 1)
        self.assertGreaterEqual(outbox["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
