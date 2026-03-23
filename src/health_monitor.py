"""Health monitor — checks all system components every N minutes."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import structlog

from alert_manager import AlertManager
from config import Settings
from models import ChannelConfig

log = structlog.get_logger(__name__)


class HealthMonitor:
    def __init__(
        self,
        settings: Settings,
        channels: list[ChannelConfig],
        alert_manager: AlertManager,
        sender,
    ) -> None:
        self._s = settings
        self._channels = channels
        self._alerts = alert_manager
        self._sender = sender

    async def check_lm_studio(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._s.lm_studio_url}/models")
                return resp.status_code == 200
        except Exception:
            return False

    async def check_rsshub(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._s.rsshub_url}/healthz")
                return resp.status_code == 200
        except Exception:
            return False

    async def check_telegram(self) -> bool:
        url = f"https://api.telegram.org/bot{self._s.tg_bot_token}/getMe"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(url)
                    return resp.status_code == 200
            except Exception:
                if attempt == 2:
                    return False
        return False

    def check_disk(self, min_gb: float = 1.0) -> bool:
        usage = shutil.disk_usage(Path(self._s.db_path).parent)
        free_gb = usage.free / 1024 ** 3
        return free_gb >= min_gb

    async def run_checks(self) -> None:
        checks = {
            "lm_studio": await self.check_lm_studio(),
            "rsshub": await self.check_rsshub(),
            "telegram": await self.check_telegram(),
            "disk": self.check_disk(),
        }

        for name, ok in checks.items():
            if not ok:
                await self._alerts.alert(
                    f"health_{name}",
                    f"Health check failed: <b>{name}</b>",
                )
                log.warning("health_check_failed", component=name)
            else:
                log.debug("health_check_ok", component=name)

    async def send_daily_digest(self) -> None:
        import db
        lines = ["<b>Дневной дайджест</b>", ""]
        for ch in self._channels:
            stats = await db.get_daily_stats(ch.id)
            total = stats.get("total", 0)
            sent = stats.get("sent", 0)
            rejected = stats.get("rejected", 0)
            failed = stats.get("failed", 0)
            lines.append(
                f"<b>{ch.name}</b>: "
                f"всего {total}, отправлено {sent}, "
                f"отклонено {rejected}, ошибок {failed}"
            )
        text = "\n".join(lines)
        try:
            await self._sender.send_text(self._s.tg_admin_chat_id, text)
            log.info("digest_sent")
        except Exception as exc:
            log.error("digest_failed", error=str(exc))
