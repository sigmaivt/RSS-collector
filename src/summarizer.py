"""LLM-based content summarizer via LM Studio (OpenAI-compatible API)."""

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


class Summarizer:
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
            "temperature": self._settings.summarizer_temperature,
            "max_tokens": self._settings.summarizer_max_tokens,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    async def summarize_job(self, job: ChannelJob, channel: ChannelConfig) -> ChannelJob:
        item = await db.get_item(job.item_id)
        if item is None:
            job.state = JobState.REJECTED
            job.last_error = "item not found"
            await db.update_job(job)
            return job

        text = item.clean_text or item.title
        prompt = self._pm.get(channel.summarizer_prompt_file)

        job.state = JobState.SUMMARIZING
        job.attempts += 1
        await db.update_job(job)

        async with self._sem:
            try:
                summary = await self._call_llm(prompt, text)
            except Exception as exc:
                log.error("summarize_error", item_id=job.item_id, error=str(exc))
                job.last_error = str(exc)
                job.state = JobState.VALIDATED  # back to validated, will retry
                await db.update_job(job)
                return job

        job.summary = summary
        job.state = JobState.READY_TO_SEND
        log.info("summarized", item_id=job.item_id, channel=channel.id)
        await db.update_job(job)
        return job

    async def run_batch(self, channel: ChannelConfig, batch_size: int = 5) -> int:
        jobs = await db.acquire_jobs(
            channel.id, JobState.VALIDATED, batch_size, self._settings.job_lease_minutes
        )
        if not jobs:
            return 0

        tasks = [self.summarize_job(job, channel) for job in jobs]
        await asyncio.gather(*tasks)
        return len(jobs)
