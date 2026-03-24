"""Proactive daily reports: trends analysis + recommendations."""

from __future__ import annotations

import asyncio
import html
import re
import time
from collections import Counter
from datetime import datetime
from typing import Literal

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
                {
                    "role": "system",
                    "content": (
                        "/no_think\n"
                        + system_prompt
                        + "\n\nВерни только финальный ответ. "
                        + "Не показывай рассуждения, chain-of-thought, Thinking Process."
                    ),
                },
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
            return self._sanitize_model_output(content)

    def _sanitize_model_output(self, content: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        cleaned = cleaned.replace("```markdown", "").replace("```", "").strip()

        banned_markers = [
            "thinking process",
            "analyze the request",
            "**role:**",
            "**input:**",
            "**task:**",
            "**requirements:**",
            "output language",
            "format:",
        ]

        kept_lines: list[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            low = line.lower()
            if any(marker in low for marker in banned_markers):
                continue
            if re.match(r"^\d+\.\s*\*\*.*(request|requirements|analysis).*", low):
                continue
            kept_lines.append(raw_line)

        result = "\n".join(kept_lines).strip()
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result or cleaned

    def _looks_like_prompt_echo(self, text: str) -> bool:
        low = text.lower()
        markers = [
            "analyze the request",
            "analyze the input data",
            "thinking process",
            "identify trends",
            "requirements",
            "structure:",
            "output language",
            "**role:**",
            "**input:**",
            "**task:**",
        ]
        hits = sum(1 for m in markers if m in low)
        english_chars = len(re.findall(r"[A-Za-z]", text))
        cyrillic_chars = len(re.findall(r"[А-Яа-яЁё]", text))
        mostly_english = english_chars > (cyrillic_chars * 2 + 120)
        return hits >= 2 or mostly_english

    def _fallback_analysis(self, dataset: list[dict]) -> str:
        source_counts = Counter((item.get("source_id") or "unknown") for item in dataset)
        words: Counter[str] = Counter()
        for item in dataset:
            title = (item.get("title") or "").lower()
            for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", title):
                if w in {"with", "from", "that", "this", "open", "source"}:
                    continue
                words[w] += 1

        top_sources = source_counts.most_common(3)
        top_words = words.most_common(6)
        lines = ["1) Главные темы в потоке:"]
        if top_words:
            lines.append("- Часто встречающиеся темы: " + ", ".join(w for w, _ in top_words))
        if top_sources:
            lines.append(
                "- Наиболее активные источники: "
                + ", ".join(f"{src} ({cnt})" for src, cnt in top_sources)
            )
        lines.append(
            "2) Почему это важно: поток фокусируется на AI-инструментах разработки, "
            "инфраструктуре и практическом применении моделей."
        )
        lines.append(
            "3) Ключевые выводы:\n"
            "- Растёт доля практических инструментов для разработчиков.\n"
            "- Ускоряется выпуск OSS/AI-решений для продакшена.\n"
            "- Важно фильтровать сигналы по повторяемости темы в разных источниках."
        )
        return "\n".join(lines)

    def _fallback_recommendations(self, channel_id: str) -> str:
        return (
            "Что изучить прямо сейчас:\n"
            "1) Практика с agentic coding инструментами и workflow.\n"
            "2) Оценка LLM-инфраструктуры: стоимость, latency, качество.\n"
            "3) DevSecOps для AI-проектов (сканирование, supply-chain).\n\n"
            "Что углублять в ближайший месяц:\n"
            "1) Оркестрация пайплайнов и наблюдаемость.\n"
            "2) Архитектуры мультимодальных/reasoning-моделей.\n"
            "3) Продуктовые метрики ценности AI-функций.\n\n"
            "Какие навыки развивать:\n"
            "1) Системный дизайн AI-сервисов.\n"
            "2) Промпт-инжиниринг с валидацией результата.\n"
            "3) Быстрая проверка гипотез на данных.\n\n"
            "Риски и чего избегать:\n"
            "1) Публикация выводов без верификации источников.\n"
            "2) Слепое доверие одному источнику/одной модели.\n"
            f"3) Игнорирование контекста канала ({channel_id})."
        )

    def _fallback_posts(self, analysis: str) -> dict[str, str]:
        twitter = (
            "1) AI-кодинг смещается к agentic workflow: важна не модель сама по себе, "
            "а воспроизводимый пайплайн вокруг нее. #AI #DevTools\n"
            "2) Главный тренд недели: практичность и стоимость AI-решений важнее хайпа. "
            "#MLOps #Engineering\n"
            "3) Для команды сейчас критично: наблюдаемость, тестирование и безопасный rollout "
            "AI-фич. #Platform #DevSecOps"
        )
        linkedin = (
            "Пост 1\n"
            "Заголовок: AI-инструменты разработки переходят из экспериментов в production.\n"
            "Ключевая идея: ценность дает не единичный инструмент, а целостная инженерная система "
            "с метриками, наблюдаемостью и контролем рисков.\n"
            "Практический вывод: строить повторяемый workflow от ingestion данных до релизов.\n\n"
            "Пост 2\n"
            "Заголовок: Как команде не потеряться в потоке AI-новостей.\n"
            "Ключевая идея: фильтруйте сигналы по применимости, стоимости и поддерживаемости.\n"
            "Практический вывод: раз в неделю фиксировать 2-3 гипотезы и проверять их на данных."
        )
        return {"twitter": twitter, "linkedin": linkedin}

    async def build_dataset(self, channel_id: str, hours: int = 24) -> list[dict]:
        return await db.get_validated_items_since(channel_id, hours=hours, limit=25)

    def _format_dataset_prompt(self, dataset: list[dict]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(dataset, start=1):
            title = (item.get("title") or "").strip()
            source = (item.get("source_id") or "").strip()
            link = (item.get("link") or "").strip()
            body = item.get("summary") or item.get("clean_text") or ""
            body = re.sub(r"\s+", " ", str(body)).strip()
            if len(body) > 320:
                body = body[:320].rsplit(" ", 1)[0] + "..."
            lines.append(
                f"[{idx}] TITLE: {title}\nSOURCE: {source}\nLINK: {link}\nTEXT: {body}"
            )
        return "\n\n".join(lines)

    async def analyze_trends(self, dataset: list[dict]) -> str:
        analyzer_prompt = self._pm.get("analyzer_ru.txt")
        input_text = self._format_dataset_prompt(dataset)
        async with self._sem:
            result = await self._call_llm(analyzer_prompt, input_text)
        if self._looks_like_prompt_echo(result):
            log.warning("proactive_analysis_low_quality_fallback")
            return self._fallback_analysis(dataset)
        return result

    async def provide_recommendations(self, analysis: str, channel_id: str) -> str:
        recommender_prompt = self._pm.get("recommender_ru.txt")
        user_text = (
            f"Канал: {channel_id}\n"
            f"На основе этого анализа сформируй рекомендации.\n\n{analysis}"
        )
        async with self._sem:
            result = await self._call_llm(recommender_prompt, user_text)
        if self._looks_like_prompt_echo(result):
            log.warning("proactive_reco_low_quality_fallback", channel=channel_id)
            return self._fallback_recommendations(channel_id)
        return result

    async def generate_posts(self, analysis: str, channel_id: str) -> dict[str, str]:
        twitter_prompt = self._pm.get("generator_twitter_ru.txt")
        linkedin_prompt = self._pm.get("generator_linkedin_ru.txt")
        base_text = f"Канал: {channel_id}\n\nАнализ:\n{analysis}"

        try:
            async with self._sem:
                twitter = await self._call_llm(twitter_prompt, base_text)
            async with self._sem:
                linkedin = await self._call_llm(linkedin_prompt, base_text)
            if self._looks_like_prompt_echo(twitter) or self._looks_like_prompt_echo(linkedin):
                log.warning("proactive_posts_low_quality_fallback", channel=channel_id)
                return self._fallback_posts(analysis)
            return {"twitter": twitter, "linkedin": linkedin}
        except Exception:
            log.warning("proactive_posts_fallback_on_error", channel=channel_id)
            return self._fallback_posts(analysis)

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
        posts: dict[str, str] | None = None,
        report_type: str = "daily",
    ) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        report_label = "Daily" if report_type == "daily" else "Triggered"
        sections = [
            (f"<b>🤖 Proactive Report ({report_label})</b>", ""),
            (f"<b>Канал:</b> {html.escape(channel.name)}", ""),
            (f"<b>Материалов за 24ч:</b> {item_count}", ""),
            ("<b>📊 Анализ трендов</b>", html.escape(analysis)),
        ]
        if posts:
            sections.append(("<b>📝 Идеи постов (X/LinkedIn)</b>", html.escape(
                "X/Twitter:\n" + posts.get("twitter", "") + "\n\nLinkedIn:\n" + posts.get("linkedin", "")
            )))
        sections.extend([
            ("<b>💡 Рекомендации</b>", html.escape(recommendations)),
            (f"<i>Сформировано: {timestamp}</i>", ""),
        ])
        return self._fit_sections(sections, self._max_tg_message_chars)

    async def run_proactive_report(
        self,
        channel: ChannelConfig,
        report_type: Literal["daily", "triggered"] = "daily",
    ) -> None:
        if not self._settings.proactive_enabled:
            return
        if not self._settings.tg_proactive_chat_id:
            log.warning("proactive_chat_missing")
            return

        started_at = time.monotonic()
        report_date = datetime.utcnow().date().isoformat()
        tags = {"channel": channel.id, "type": report_type}

        if report_type == "daily":
            exists = await db.check_proactive_report_exists(channel.id, report_type, report_date)
        else:
            exists = await db.check_recent_proactive_report(channel.id, report_type, hours=24)
        if exists:
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
            posts = await self.generate_posts(analysis, channel.id)
            recommendations = await self.provide_recommendations(analysis, channel.id)
            report = self.compose_markdown_report(
                channel=channel,
                analysis=analysis,
                recommendations=recommendations,
                item_count=len(dataset),
                posts=posts,
                report_type=report_type,
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
        await self.run_proactive_reports(channels, report_type="daily")

    async def run_daily_report(self, channel: ChannelConfig) -> None:
        await self.run_proactive_report(channel, report_type="daily")

    async def should_trigger_proactive(self, channel_id: str) -> bool:
        count = await db.count_validated_items_since(channel_id, hours=24)
        if count < self._settings.proactive_trigger_threshold:
            return False
        already_sent = await db.check_recent_proactive_report(
            channel_id=channel_id,
            report_type="triggered",
            hours=24,
        )
        return not already_sent

    async def run_proactive_reports(
        self,
        channels: list[ChannelConfig],
        report_type: Literal["daily", "triggered"] = "daily",
    ) -> None:
        if not self._settings.proactive_enabled:
            return
        for channel in channels:
            if not channel.enabled:
                continue
            try:
                await self.run_proactive_report(channel, report_type=report_type)
            except Exception as exc:
                log.error("proactive_channel_unhandled_error", channel=channel.id, error=str(exc))
