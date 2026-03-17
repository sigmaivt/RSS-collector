"""Alert manager — sends Telegram alerts with cooldown."""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog

log = structlog.get_logger(__name__)


class AlertManager:
    def __init__(self, sender, admin_chat_id: str, cooldown_minutes: int = 30) -> None:
        self._sender = sender
        self._admin_chat_id = admin_chat_id
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._last_sent: dict[str, datetime] = {}

    async def alert(self, key: str, message: str) -> None:
        now = datetime.utcnow()
        last = self._last_sent.get(key)
        if last and (now - last) < self._cooldown:
            log.debug("alert_suppressed", key=key)
            return
        self._last_sent[key] = now
        try:
            await self._sender.send_text(self._admin_chat_id, f"⚠️ {message}")
            log.info("alert_sent", key=key)
        except Exception as exc:
            log.error("alert_send_failed", key=key, error=str(exc))
