"""Proactive daily reports: trends analysis + recommendations."""

from __future__ import annotations

import asyncio
import html
import re
import time
from datetime import datetime

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

import db
from config import Settings
from models import ChannelConfig
from prompt_manager import PromptManager

log = structlog.get_logger(__name__)


class ContentProcessor:
    def __init__(
        self,
        settings: Settings,
        prompt_manager: PromptManager,
        sender,
        alert_manager=None,
    ) -> None:
        self._settings = settings
        self._pm = prompt_manager
        self._sender = sender
        self._alerts = alert_manager
        self._sem = asyncio.Semaphore(settings.llm_semaphore)
        self._failure_streak: dict[str, int] = {}
        self._max_tg_message_chars = 3800

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_llm(self, system_prompt: str, text: str) -> str:
        url = f"{self._settings.lm_studio_url}/chat/completions"
        payload = {
            "model": self._settings.lm_studio_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self._settings.proactive_temperature,
            "max_tokens": self._settings.proactive_max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def build_dataset(self, channel_id: str, hours: int = 24) -> list[dict]:
        return await db.get_validated_items_since(channel_id, hours=hours)

    def _format_dataset_prompt(self, dataset: list[dict]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(dataset, start=1):
            title = (item.get("title") or "").strip()
            source = (item.get("source_id") or "").strip()
            link = (item.get("link") or "").strip()
            body = item.get("summary") or item.get("clean_text") or ""
            body = re.sub(r"\s+", " ", str(body)).strip()
            if len(body) > 700:
                body = body[:700].rsplit(" ", 1)[0] + "..."
            lines.append(
                f"[{idx}] TITLE: {title}\nSOURCE: {source}\nLINK: {link}\nTEXT: {body}"
            )
        return "\n\n".join(lines)

    async def analyze_trends(self, dataset: list[dict]) -> str:
        analyzer_prompt = self._pm.get("analyzer_ru.txt")
        input_text = self._format_dataset_prompt(dataset)
        async with self._sem:
            return await self._call_llm(analyzer_prompt, input_text)

    async def provide_recommendations(self, analysis: str, channel_id: str) -> str:
        recommender_prompt = self._pm.get("recommender_ru.txt")
        user_text = (
            f"Канал: {channel_id}\n"
            f"На основе этого анализа сформируй рекомендации.\n\n{analysis}"
        )
        async with self._sem:
            return await self._call_llm(recommender_prompt, user_text)

    def _fit_sections(self, sections: list[tuple[str, str]], max_chars: int) -> str:
        parts: list[str] = []
        for title, content in sections:
            chunk = f"{title}\n{content}".strip()
            candidate = "\n\n".join([*parts, chunk]).strip()
            if len(candidate) <= max_chars:
                parts.append(chunk)
                continue

            remaining = max_chars - len("\n\n".join(parts)) - len(title) - 8
            if remaining > 80:
                trimmed = content[:remaining].rsplit(" ", 1)[0] + "..."
                parts.append(f"{title}\n{trimmed}")
            parts.append("<i>Сообщение сокращено из-за лимита Telegram.</i>")
            break
        result = "\n\n".join(parts).strip()
        if len(result) <= max_chars:
            return result
        safe_limit = max(40, max_chars - len("..."))
        return result[:safe_limit].rsplit(" ", 1)[0] + "..."

    def compose_markdown_report(
        self,
        channel: ChannelConfig,
        analysis: str,
        recommendations: str,
        item_count: int,
    ) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            ("<b>🤖 Proactive Report</b>", ""),
            (f"<b>Канал:</b> {html.escape(channel.name)}", ""),
            (f"<b>Материалов за 24ч:</b> {item_count}", ""),
            ("<b>📊 Анализ трендов</b>", html.escape(analysis)),
            ("<b>💡 Рекомендации</b>", html.escape(recommendations)),
            (f"<i>Сформировано: {timestamp}</i>", ""),
        ]
        return self._fit_sections(sections, self._max_tg_message_chars)

    async def run_daily_report(self, channel: ChannelConfig) -> None:
        if not self._settings.proactive_enabled:
            return
        if not self._settings.tg_proactive_chat_id:
            log.warning("proactive_chat_missing")
            return

        started_at = time.monotonic()
        report_date = datetime.utcnow().date().isoformat()
        report_type = "daily"
        tags = {"channel": channel.id, "type": report_type}

        if await db.check_proactive_report_exists(channel.id, report_type, report_date):
            log.info("proactive_skip_duplicate", channel=channel.id, report_date=report_date)
            return

        report_id = await db.save_proactive_report(
            channel.id,
            report_type,
            report_date,
            status="created",
        )
        await db.record_metric("proactive_run_started", 1.0, tags)

        dataset = await self.build_dataset(channel.id, hours=24)
        if len(dataset) < self._settings.proactive_min_items_24h:
            reason = (
                f"not enough validated items: {len(dataset)} < "
                f"{self._settings.proactive_min_items_24h}"
            )
            await db.update_proactive_report_status(
                report_id,
                status="skipped",
                skipped_reason=reason,
            )
            await db.record_metric("proactive_run_skipped_not_enough_items", 1.0, tags)
            self._failure_streak[channel.id] = 0
            log.info("proactive_skip_not_enough_items", channel=channel.id, count=len(dataset))
            return

        try:
            analysis = await self.analyze_trends(dataset)
            recommendations = await self.provide_recommendations(analysis, channel.id)
            report = self.compose_markdown_report(
                channel=channel,
                analysis=analysis,
                recommendations=recommendations,
                item_count=len(dataset),
            )
            await self._sender.send_text(self._settings.tg_proactive_chat_id, report)
            await db.update_proactive_report_status(
                report_id,
                status="sent",
                mark_sent=True,
            )
            await db.record_metric("proactive_run_sent", 1.0, tags)
            self._failure_streak[channel.id] = 0
            log.info("proactive_sent", channel=channel.id, count=len(dataset))
        except Exception as exc:
            error_text = str(exc)
            await db.update_proactive_report_status(
                report_id,
                status="failed",
                error_message=error_text[:1000],
            )
            await db.record_metric("proactive_run_failed", 1.0, tags)
            self._failure_streak[channel.id] = self._failure_streak.get(channel.id, 0) + 1
            log.error("proactive_failed", channel=channel.id, error=error_text)

            if self._alerts and self._failure_streak[channel.id] >= 2:
                safe_error = html.escape(error_text[:300])
                await self._alerts.alert(
                    f"proactive_failed_{channel.id}",
                    f"Proactive daily report keeps failing for <b>{html.escape(channel.name)}</b>: {safe_error}",
                )
        finally:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            await db.record_metric("proactive_run_duration_ms", float(duration_ms), tags)

    async def run_daily_reports(self, channels: list[ChannelConfig]) -> None:
        if not self._settings.proactive_enabled:
            return
        for channel in channels:
            if not channel.enabled:
                continue
            try:
                await self.run_daily_report(channel)
            except Exception as exc:
                log.error("proactive_channel_unhandled_error", channel=channel.id, error=str(exc))
