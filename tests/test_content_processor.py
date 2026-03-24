from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import db  # noqa: E402
from content_processor import ContentProcessor  # noqa: E402
from models import ChannelConfig, JobState, RssItem  # noqa: E402
from prompt_manager import PromptManager  # noqa: E402


class ContentProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db.set_db_path(str(Path(self._tmp.name) / "pipeline.db"))
        await db.init_db()

        self.settings = SimpleNamespace(
            lm_studio_url="http://localhost:1234/v1",
            lm_studio_model="model",
            llm_semaphore=2,
            proactive_enabled=True,
            proactive_digest_hour=9,
            proactive_min_items_24h=2,
            proactive_temperature=0.3,
            proactive_max_tokens=700,
            tg_proactive_chat_id="-100123456",
        )
        self.sender = SimpleNamespace(send_text=AsyncMock(return_value=None))
        self.alert_mgr = SimpleNamespace(alert=AsyncMock(return_value=None))
        self.processor = ContentProcessor(
            self.settings,
            PromptManager(PROJECT_ROOT / "prompts"),
            self.sender,
            self.alert_mgr,
        )
        self.channel = ChannelConfig(
            id="ai_models",
            name="AI Models",
            description="desc",
            classifier_prompt_file="classifier_ai_models.txt",
            summarizer_prompt_file="summarizer_ru.txt",
            telegram_chat_id="-100200",
            enabled=True,
            sources=[],
        )

    async def asyncTearDown(self) -> None:
        self._tmp.cleanup()

    async def _seed_validated_item(self, suffix: int) -> None:
        item_id = f"item-{suffix}"
        item = RssItem(
            item_id=item_id,
            source_id="src1",
            guid=f"guid-{suffix}",
            link=f"https://example.com/{suffix}",
            title=f"Title {suffix}",
            clean_text="Some clean content",
        )
        await db.upsert_item(item)
        await db.create_job(item_id, self.channel.id)
        jobs = await db.acquire_jobs(self.channel.id, JobState.NEW, 10, 10)
        for job in jobs:
            if job.item_id == item_id:
                job.state = JobState.VALIDATED
                job.summary = "Summary"
                job.leased_at = None
                await db.update_job(job)
                break

    async def _proactive_rows(self) -> list[dict]:
        async with db.get_conn() as conn:
            async with conn.execute(
                "SELECT * FROM proactive_reports ORDER BY id"
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def test_skip_when_not_enough_items(self) -> None:
        await self._seed_validated_item(1)
        await self.processor.run_daily_report(self.channel)

        rows = await self._proactive_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "skipped")
        self.sender.send_text.assert_not_called()

    async def test_dedup_daily_report(self) -> None:
        await self._seed_validated_item(1)
        await self._seed_validated_item(2)

        with patch.object(
            self.processor, "analyze_trends", new=AsyncMock(return_value="analysis")
        ), patch.object(
            self.processor,
            "provide_recommendations",
            new=AsyncMock(return_value="recommendations"),
        ):
            await self.processor.run_daily_report(self.channel)
            await self.processor.run_daily_report(self.channel)

        rows = await self._proactive_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "sent")
        self.sender.send_text.assert_awaited_once()

    async def test_failed_report_when_llm_errors(self) -> None:
        await self._seed_validated_item(1)
        await self._seed_validated_item(2)

        with patch.object(
            self.processor,
            "analyze_trends",
            new=AsyncMock(side_effect=RuntimeError("llm down")),
        ):
            await self.processor.run_daily_report(self.channel)

        rows = await self._proactive_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("llm down", rows[0]["error_message"])

    async def test_compose_report_fits_telegram_limit(self) -> None:
        text = "A" * 10000
        report = self.processor.compose_markdown_report(
            channel=self.channel,
            analysis=text,
            recommendations=text,
            item_count=123,
        )
        self.assertLessEqual(len(report), 3800)

    async def test_run_daily_reports_integration_single_send(self) -> None:
        await self._seed_validated_item(1)
        await self._seed_validated_item(2)

        with patch.object(
            self.processor, "analyze_trends", new=AsyncMock(return_value="analysis")
        ), patch.object(
            self.processor,
            "provide_recommendations",
            new=AsyncMock(return_value="recommendations"),
        ):
            await self.processor.run_daily_reports([self.channel])

        rows = await self._proactive_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "sent")
        self.sender.send_text.assert_awaited_once()

    async def test_telegram_error_marked_failed_without_crash(self) -> None:
        await self._seed_validated_item(1)
        await self._seed_validated_item(2)
        self.sender.send_text = AsyncMock(side_effect=RuntimeError("telegram failed"))
        self.processor = ContentProcessor(
            self.settings,
            PromptManager(PROJECT_ROOT / "prompts"),
            self.sender,
            self.alert_mgr,
        )
        with patch.object(
            self.processor, "analyze_trends", new=AsyncMock(return_value="analysis")
        ), patch.object(
            self.processor,
            "provide_recommendations",
            new=AsyncMock(return_value="recommendations"),
        ):
            await self.processor.run_daily_reports([self.channel])

        rows = await self._proactive_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
