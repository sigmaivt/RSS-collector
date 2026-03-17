"""LLM-based content classifier via LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import asyncio

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

import db
from config import Settings
from models import ChannelConfig, ChannelJob, JobState
from prompt_manager import PromptManager

log = structlog.get_logger(__name__)


class Classifier:
    def __init__(self, settings: Settings, prompt_manager: PromptManager) -> None:
        self._settings = settings
        self._pm = prompt_manager
        self._sem = asyncio.Semaphore(settings.llm_semaphore)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_llm(self, system_prompt: str, text: str) -> str:
        url = f"{self._settings.lm_studio_url}/chat/completions"
        payload = {
            "model": self._settings.lm_studio_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self._settings.classifier_temperature,
            "max_tokens": self._settings.classifier_max_tokens,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip().upper()

    async def classify_job(self, job: ChannelJob, channel: ChannelConfig) -> ChannelJob:
        item = await db.get_item(job.item_id)
        if item is None:
            job.state = JobState.REJECTED
            job.last_error = "item not found"
            return job

        text = item.clean_text or item.title
        prompt = self._pm.get(channel.classifier_prompt_file)

        job.state = JobState.CLASSIFYING
        job.attempts += 1
        await db.update_job(job)

        async with self._sem:
            try:
                result = await self._call_llm(prompt, text)
            except Exception as exc:
                log.error("classify_error", item_id=job.item_id, error=str(exc))
                job.last_error = str(exc)
                job.state = JobState.NEW  # will retry
                await db.update_job(job)
                return job

        job.classifier_result = result
        if result.startswith("VALID"):
            job.state = JobState.VALIDATED
        else:
            job.state = JobState.REJECTED

        log.info(
            "classified",
            item_id=job.item_id,
            channel=channel.id,
            result=result,
        )
        await db.update_job(job)
        return job

    async def run_batch(self, channel: ChannelConfig, batch_size: int = 10) -> int:
        jobs = await db.acquire_jobs(
            channel.id, JobState.NEW, batch_size, self._settings.job_lease_minutes
        )
        if not jobs:
            return 0

        tasks = [self.classify_job(job, channel) for job in jobs]
        await asyncio.gather(*tasks)
        return len(jobs)
