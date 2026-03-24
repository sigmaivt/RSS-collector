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
        # Keep proactive prompts compact for local 4k-context models.
        self._dataset_limit = 12
        self._dataset_text_chars = 160

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_local_llm(self, system_prompt: str, text: str) -> str:
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
            # Guard against LM Studio context overflows.
            "max_tokens": min(self._settings.proactive_max_tokens, 320),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return self._sanitize_model_output(content)

    def _openrouter_available(self) -> bool:
        return bool(
            getattr(self._settings, "openrouter_enabled", False)
            and getattr(self._settings, "openrouter_api_key", None)
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
    async def _call_openrouter_llm(self, system_prompt: str, text: str) -> str:
        if not self._openrouter_available():
            raise RuntimeError("openrouter_not_configured")

        base_url = self._settings.openrouter_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self._settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Верни только финальный ответ без рассуждений.\n"
                        + system_prompt
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": self._settings.proactive_temperature,
            "max_tokens": min(self._settings.proactive_max_tokens, 500),
        }
        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_site_url:
            headers["HTTP-Referer"] = self._settings.openrouter_site_url
        if self._settings.openrouter_app_name:
            headers["X-Title"] = self._settings.openrouter_app_name

        timeout = int(getattr(self._settings, "openrouter_timeout_sec", 120))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = self._extract_message_content(data)
            return self._sanitize_model_output(content)

    def _extract_message_content(self, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    text = chunk.get("text") or chunk.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(p for p in parts if p).strip()
        return str(content)

    def _is_empty_or_null_output(self, text: str) -> bool:
        value = (text or "").strip().lower()
        return value in {"", "none", "null", "n/a", "нет данных"}

    async def _call_llm(self, system_prompt: str, text: str) -> str:
        try:
            return await self._call_local_llm(system_prompt, text)
        except Exception as exc:
            if not self._openrouter_available():
                raise
            log.warning("proactive_local_llm_failed_fallback_openrouter", error=str(exc))
            return await self._call_openrouter_llm(system_prompt, text)

    async def _repair_with_openrouter(
        self, system_prompt: str, text: str, reason: str
    ) -> str | None:
        if not self._openrouter_available():
            return None
        try:
            repaired = await self._call_openrouter_llm(system_prompt, text)
            if self._looks_like_prompt_echo_v2(repaired):
                log.warning("proactive_openrouter_low_quality", reason=reason)
                return None
            log.info("proactive_openrouter_used", reason=reason)
            return repaired
        except Exception as exc:
            log.warning("proactive_openrouter_failed", reason=reason, error=str(exc))
            return None

    def _sanitize_model_output(self, content: str) -> str:
        if not isinstance(content, str):
            content = str(content)
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
        return await db.get_validated_items_since(
            channel_id,
            hours=hours,
            limit=self._dataset_limit,
        )

    def _format_dataset_prompt(self, dataset: list[dict]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(dataset, start=1):
            title = (item.get("title") or "").strip()
            source = (item.get("source_id") or "").strip()
            link = (item.get("link") or "").strip()
            body = item.get("summary") or item.get("clean_text") or ""
            body = re.sub(r"\s+", " ", str(body)).strip()
            if len(body) > self._dataset_text_chars:
                body = body[: self._dataset_text_chars].rsplit(" ", 1)[0] + "..."
            lines.append(
                f"[{idx}] TITLE: {title}\nSOURCE: {source}\nLINK: {link}\nTEXT: {body}"
            )
        return "\n\n".join(lines)

    async def analyze_trends(self, dataset: list[dict]) -> str:
        analyzer_prompt = self._pm.get("analyzer_ru.txt")
        input_text = self._format_dataset_prompt(dataset)
        try:
            async with self._sem:
                result = await self._call_llm(analyzer_prompt, input_text)
        except Exception:
            log.warning("proactive_analysis_fallback_on_error")
            return self._fallback_analysis_v2(dataset)
        if self._is_empty_or_null_output(result):
            repaired = await self._repair_with_openrouter(
                analyzer_prompt,
                input_text,
                reason="analysis_empty_or_null",
            )
            if repaired and not self._is_empty_or_null_output(repaired):
                return repaired
            return self._fallback_analysis_v2(dataset)
        if self._looks_like_prompt_echo_v2(result):
            repaired = await self._repair_with_openrouter(
                analyzer_prompt,
                input_text,
                reason="analysis_low_quality",
            )
            if repaired:
                return repaired
            log.warning("proactive_analysis_low_quality_fallback")
            return self._fallback_analysis_v2(dataset)
        return result

    async def provide_recommendations(
        self, analysis: str, channel_id: str, dataset: list[dict]
    ) -> str:
        recommender_prompt = self._pm.get("recommender_ru.txt")
        user_text = (
            f"Канал: {channel_id}\n"
            f"На основе этого анализа сформируй рекомендации.\n\n{analysis}"
        )
        try:
            async with self._sem:
                result = await self._call_llm(recommender_prompt, user_text)
        except Exception:
            log.warning("proactive_reco_fallback_on_error", channel=channel_id)
            return self._fallback_recommendations_v2(channel_id, dataset)
        if self._is_empty_or_null_output(result):
            repaired = await self._repair_with_openrouter(
                recommender_prompt,
                user_text,
                reason=f"recommendations_empty_or_null_{channel_id}",
            )
            if repaired and not self._is_empty_or_null_output(repaired):
                return repaired
            return self._fallback_recommendations_v2(channel_id, dataset)
        if self._looks_like_prompt_echo_v2(result):
            repaired = await self._repair_with_openrouter(
                recommender_prompt,
                user_text,
                reason=f"recommendations_low_quality_{channel_id}",
            )
            if repaired:
                return repaired
            log.warning("proactive_reco_low_quality_fallback", channel=channel_id)
            return self._fallback_recommendations_v2(channel_id, dataset)
        return result

    async def generate_posts(
        self, analysis: str, channel_id: str, dataset: list[dict]
    ) -> dict[str, str]:
        twitter_prompt = self._pm.get("generator_twitter_ru.txt")
        linkedin_prompt = self._pm.get("generator_linkedin_ru.txt")
        base_text = f"Канал: {channel_id}\n\nАнализ:\n{analysis}"

        try:
            async with self._sem:
                twitter = await self._call_llm(twitter_prompt, base_text)
            async with self._sem:
                linkedin = await self._call_llm(linkedin_prompt, base_text)
            if self._looks_like_prompt_echo_v2(twitter) or self._looks_like_prompt_echo_v2(
                linkedin
            ):
                repaired_twitter = await self._repair_with_openrouter(
                    twitter_prompt,
                    base_text,
                    reason=f"creative_twitter_low_quality_{channel_id}",
                )
                repaired_linkedin = await self._repair_with_openrouter(
                    linkedin_prompt,
                    base_text,
                    reason=f"creative_linkedin_low_quality_{channel_id}",
                )
                if repaired_twitter and repaired_linkedin:
                    return {"twitter": repaired_twitter, "linkedin": repaired_linkedin}
                log.warning("proactive_posts_low_quality_fallback", channel=channel_id)
                return self._fallback_posts_v2(channel_id, dataset)
            return {"twitter": twitter, "linkedin": linkedin}
        except Exception:
            log.warning("proactive_posts_fallback_on_error", channel=channel_id)
            return self._fallback_posts_v2(channel_id, dataset)

    def _looks_like_prompt_echo_v2(self, text: str) -> bool:
        low = text.lower()
        markers = [
            "thinking process",
            "analyze the request",
            "analyze the input",
            "identify 3-5 trends",
            "requirements",
            "structure:",
            "**role:**",
            "**input:**",
            "**task:**",
        ]
        hits = sum(1 for marker in markers if marker in low)
        english_chars = len(re.findall(r"[A-Za-z]", text))
        cyrillic_chars = len(re.findall(r"[\u0400-\u04FF]", text))
        mostly_english = english_chars > (cyrillic_chars * 2 + 80)
        return hits >= 2 or mostly_english

    def _extract_topic_groups_v2(self, dataset: list[dict]) -> list[tuple[str, int, list[str]]]:
        topic_keywords: list[tuple[str, set[str]]] = [
            ("LLM и модели", {"llm", "model", "reasoning", "open-weight", "multimodal"}),
            ("AI-инфраструктура", {"nvidia", "cloud", "gpu", "inference", "latency"}),
            ("Инструменты разработки", {"github", "tool", "dev", "coder", "workflow", "git"}),
            ("Безопасность и DevSecOps", {"security", "vuln", "scanner", "supply", "injection"}),
            ("OSS и платформы", {"open source", "oss", "release", "linux", "database", "filesystem"}),
            ("Медиа и мультимодальность", {"video", "audio", "tts", "vision", "diffusion"}),
            ("Агенты и автоматизация", {"agent", "agentic", "autonomous", "orchestration"}),
            ("Право и этика", {"lawsuit", "ethics", "regulation", "legal", "policy"}),
        ]

        buckets: dict[str, list[str]] = {}
        for item in dataset:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            low = title.lower()
            matched = False
            for topic, keywords in topic_keywords:
                if any(k in low for k in keywords):
                    buckets.setdefault(topic, []).append(title)
                    matched = True
                    break
            if not matched:
                buckets.setdefault("Прочее", []).append(title)

        groups: list[tuple[str, int, list[str]]] = []
        for topic, titles in buckets.items():
            groups.append((topic, len(titles), titles[:2]))
        groups.sort(key=lambda x: x[1], reverse=True)
        return groups[:5]

    def _channel_profile_v2(self, channel_id: str) -> dict[str, str]:
        if channel_id == "ai_coding":
            return {
                "study": "agentic coding workflow и инструменты разработчика",
                "deepen": "качество кода, CI, безопасный rollout AI-фич",
                "skill": "архитектура dev-пайплайнов с LLM в контуре",
                "risk": "внедрение без тестов/наблюдаемости",
                "tag": "#AICoding",
            }
        if channel_id == "ai_models":
            return {
                "study": "оценка моделей: качество, latency, стоимость",
                "deepen": "архитектуры reasoning/multimodal и бенчмарки",
                "skill": "эксперименты и валидация гипотез по моделям",
                "risk": "выбор модели по хайпу без метрик",
                "tag": "#AIModels",
            }
        return {
            "study": "прикладные AI-сценарии для enterprise/ERP/MES",
            "deepen": "интеграция в бизнес-процессы и данные",
            "skill": "продуктовая аналитика ценности AI-функций",
            "risk": "неучтенные ограничения процессов и данных",
            "tag": "#ERPMES",
        }

    def _fallback_analysis_v2(self, dataset: list[dict]) -> str:
        groups = self._extract_topic_groups_v2(dataset)
        sources = Counter((item.get("source_id") or "unknown") for item in dataset).most_common(3)

        lines = ["1) Главные темы за 24ч:"]
        if groups:
            for idx, (topic, count, examples) in enumerate(groups, start=1):
                example_text = "; ".join(examples) if examples else "без примеров"
                lines.append(
                    f"- [{idx}] {topic}: подтверждений ~{count}. Примеры: {example_text}."
                )
        else:
            lines.append("- Недостаточно данных для уверенной тематической группировки.")
        if sources:
            lines.append(
                "- Наиболее активные источники: "
                + ", ".join(f"{source} ({count})" for source, count in sources)
            )
        lines.append(
            "2) Почему это важно: поток смещается в сторону прикладных AI-решений, "
            "где важны эксплуатация, цена и воспроизводимость результата."
        )
        lines.append(
            "3) Ключевые выводы:\n"
            "- Приоритет у практических инструментов и сценариев внедрения.\n"
            "- Устойчивость пайплайна важнее разовых демо.\n"
            "- Для решения нужны сигналы, подтвержденные несколькими источниками."
        )
        return "\n".join(lines)

    def _fallback_recommendations_v2(self, channel_id: str, dataset: list[dict]) -> str:
        groups = self._extract_topic_groups_v2(dataset)
        profile = self._channel_profile_v2(channel_id)
        top = [topic for topic, _, _ in groups[:3]]
        while len(top) < 3:
            top.append("AI-инженерия")
        examples = []
        for _, _, ex in groups:
            examples.extend(ex)
        ex1 = examples[0] if len(examples) > 0 else "без яркого примера"
        ex2 = examples[1] if len(examples) > 1 else "без второго примера"
        return (
            "Что изучить прямо сейчас:\n"
            f"1) {profile['study']}: начните с кейса «{ex1}».\n"
            f"2) Быстрые эксперименты по теме «{top[1]}» с замером эффекта на примере «{ex2}».\n"
            f"3) Чек-лист внедрения по теме «{top[2]}» (качество, стоимость, риски).\n\n"
            "Что углублять в ближайший месяц:\n"
            f"1) {profile['deepen']}.\n"
            "2) Архитектура fallback: где локальная модель, где внешняя.\n"
            "3) Процесс валидации: критерии полезности для аудитории.\n\n"
            "Какие навыки развивать:\n"
            f"1) {profile['skill']}.\n"
            "2) Prompt engineering + пост-валидация результата.\n"
            "3) Приоритизация задач по impact/effort.\n\n"
            "Риски и чего избегать:\n"
            "1) Выводы без проверки по нескольким источникам.\n"
            "2) Слишком длинные промпты и избыточная параллельность на локальной LLM.\n"
            f"3) {profile['risk']} для канала ({channel_id})."
        )

    def _fallback_posts_v2(self, channel_id: str, dataset: list[dict]) -> dict[str, str]:
        groups = self._extract_topic_groups_v2(dataset)
        profile = self._channel_profile_v2(channel_id)
        top = [topic for topic, _, _ in groups[:3]]
        while len(top) < 3:
            top.append("AI-инженерия")
        examples = []
        for _, _, ex in groups:
            examples.extend(ex)
        ex1 = examples[0] if len(examples) > 0 else "последний дайджест"
        ex2 = examples[1] if len(examples) > 1 else "второй кейс из ленты"

        twitter = (
            f"1) {profile['tag']} Тема дня: «{top[0]}» на примере «{ex1}». В приоритете измеримый эффект, не демо. #AI #Engineering\n"
            f"2) По теме «{top[1]}» на кейсе «{ex2}» критичны latency/стоимость и понятный fallback-план. #MLOps #LLM\n"
            f"3) Для {profile['tag']} «{top[2]}» лучше внедрять короткими итерациями: гипотеза -> пилот -> решение. #DevTools #Product"
        )
        linkedin = (
            "Пост 1\n"
            f"Заголовок: Как каналу {channel_id} превратить тренд «{top[0]}» в рабочий результат.\n"
            f"Опорный кейс из ленты: «{ex1}».\n"
            "В AI-проектах ценность создается процессом: критерии качества, мониторинг, управление рисками.\n"
            "Практический вывод: запускать маленький пилот и масштабировать только после измеримого эффекта.\n\n"
            "Пост 2\n"
            f"Заголовок: Как отбирать сигналы по темам «{top[1]}» и «{top[2]}» в {channel_id}.\n"
            f"Второй опорный кейс: «{ex2}».\n"
            "В решение должны попадать только идеи, которые подтверждаются разными источниками и применимы к вашей среде.\n"
            "Рабочая рамка: impact, стоимость внедрения, устойчивость эксплуатации.\n"
            "Практический вывод: вести недельный backlog гипотез и закрывать цикл от идеи до решения."
        )
        return {"twitter": twitter, "linkedin": linkedin}

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
        sections.extend([
            ("<b>💡 Рекомендации</b>", html.escape(recommendations)),
            (f"<i>Сформировано: {timestamp}</i>", ""),
        ])
        return self._fit_sections(sections, self._max_tg_message_chars)

    def compose_creative_report(
        self,
        channel: ChannelConfig,
        posts: dict[str, str],
        report_type: str = "daily",
    ) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        report_label = "Daily" if report_type == "daily" else "Triggered"
        creative = (
            "X/Twitter:\n"
            + posts.get("twitter", "")
            + "\n\nLinkedIn:\n"
            + posts.get("linkedin", "")
        )
        sections = [
            (f"<b>📝 Creative Pack ({report_label})</b>", ""),
            (f"<b>Канал:</b> {html.escape(channel.name)}", ""),
            ("<b>Идеи постов</b>", html.escape(creative)),
            (f"<i>Сформировано: {timestamp}</i>", ""),
        ]
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
            posts = await self.generate_posts(analysis, channel.id, dataset)
            recommendations = await self.provide_recommendations(
                analysis, channel.id, dataset
            )
            report = self.compose_markdown_report(
                channel=channel,
                analysis=analysis,
                recommendations=recommendations,
                item_count=len(dataset),
                report_type=report_type,
            )
            await self._sender.send_text(self._settings.tg_proactive_chat_id, report)
            creative_chat_id = self._settings.tg_proactive_creative_chat_id
            if creative_chat_id:
                creative_report = self.compose_creative_report(
                    channel=channel,
                    posts=posts,
                    report_type=report_type,
                )
                await self._sender.send_text(creative_chat_id, creative_report)
            else:
                log.info("proactive_creative_chat_not_set", channel=channel.id)
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
