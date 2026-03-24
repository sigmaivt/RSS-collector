"""Telegram message sender with idempotent outbox pattern."""

from __future__ import annotations

import asyncio

import httpx
import structlog

import db
from config import Settings
from models import ChannelConfig, ChannelJob, JobState, TelegramOutboxEntry

log = structlog.get_logger(__name__)

_TG_API = "https://api.telegram.org/bot{token}/sendMessage"


def _format_message(job: ChannelJob, title: str, link: str, channel_name: str) -> str:
    lines = [
        f"<b>{channel_name}</b>",
        "",
        f"<b>{title}</b>",
    ]
    if job.summary:
        lines += ["", job.summary]
    if link:
        lines += ["", f'<a href="{link}">Читать далее</a>']
    return "\n".join(lines)


class TelegramSender:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = _TG_API.format(token=settings.tg_bot_token)

    async def _send_message(
        self,
        chat_id: str,
        text: str,
        client: httpx.AsyncClient,
    ) -> int:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        }
        resp = await client.post(self._url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["result"]["message_id"]

    async def enqueue_job(self, job: ChannelJob, channel: ChannelConfig) -> None:
        """Build message and enqueue to outbox; update job state."""
        item = await db.get_item(job.item_id)
        if item is None:
            return

        text = _format_message(
            job,
            title=item.title,
            link=item.link or "",
            channel_name=channel.name,
        )
        entry = TelegramOutboxEntry(
            item_id=job.item_id,
            channel_id=job.channel_id,
            chat_id=channel.telegram_chat_id,
            message_text=text,
        )
        await db.enqueue_outbox(entry)

    async def flush_outbox(self, max_retries: int | None = None) -> int:
        """Send pending outbox messages; return count sent."""
        if max_retries is None:
            max_retries = self._settings.max_send_retries

        entries = await db.get_pending_outbox(limit=20)
        if not entries:
            return 0

        sent = 0
        async with httpx.AsyncClient() as client:
            for entry in entries:
                if entry.attempts >= max_retries:
                    error = f"max retries reached ({entry.attempts})"
                    log.error(
                        "outbox_max_retries",
                        item_id=entry.item_id,
                        channel=entry.channel_id,
                    )
                    await db.mark_outbox_failed(entry.id, error)
                    await db.update_job_state_by_item_channel(
                        entry.item_id,
                        entry.channel_id,
                        JobState.SEND_FAILED,
                        "telegram send retries exhausted",
                    )
                    continue
                try:
                    msg_id = await self._send_message(
                        entry.chat_id, entry.message_text, client
                    )
                    await db.mark_outbox_sent(entry.id, msg_id)
                    await db.update_job_state_by_item_channel(
                        entry.item_id,
                        entry.channel_id,
                        JobState.SENT,
                    )
                    log.info(
                        "tg_sent",
                        item_id=entry.item_id,
                        channel=entry.channel_id,
                        msg_id=msg_id,
                    )
                    sent += 1
                    await asyncio.sleep(0.05)  # stay within Telegram rate limits
                except Exception as exc:
                    log.warning(
                        "tg_send_failed",
                        item_id=entry.item_id,
                        error=str(exc),
                    )
                    attempts = await db.increment_outbox_attempt(entry.id, str(exc))
                    if attempts >= max_retries:
                        await db.mark_outbox_failed(
                            entry.id, f"max retries reached: {exc}"
                        )
                        await db.update_job_state_by_item_channel(
                            entry.item_id,
                            entry.channel_id,
                            JobState.SEND_FAILED,
                            "telegram send retries exhausted",
                        )
                    else:
                        await db.update_job_state_by_item_channel(
                            entry.item_id,
                            entry.channel_id,
                            JobState.OUTBOX_PENDING,
                            str(exc),
                        )
        return sent

    async def send_text(self, chat_id: str, text: str) -> None:
        """Send a plain text message (used for alerts/digests)."""
        async with httpx.AsyncClient() as client:
            await self._send_message(chat_id, text, client)
