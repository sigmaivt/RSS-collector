"""Configuration loading from settings.env and channels.yaml."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from models import ChannelConfig, SourceConfig


BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LM Studio
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "lfm2-2.6b-exp"
    llm_semaphore: int = 2

    # RSSHub
    rsshub_url: str = "http://localhost:1200"
    github_access_token: Optional[str] = None

    # Telegram
    tg_bot_token: str
    tg_admin_chat_id: str
    tg_proactive_chat_id: str = ""
    tg_proactive_creative_chat_id: str = ""

    # Per-channel chat IDs
    tg_chat_ai_coding: Optional[str] = None
    tg_chat_ai_models: Optional[str] = None
    tg_chat_erp_mes: Optional[str] = None

    # Pipeline
    poll_interval_minutes: int = 30
    classifier_temperature: float = 0.0
    classifier_max_tokens: int = 5
    summarizer_temperature: float = 0.5
    summarizer_max_tokens: int = 300
    max_text_length: int = 15000
    job_lease_minutes: int = 10
    max_send_retries: int = 5

    # Monitoring
    health_check_interval_minutes: int = 5
    alert_cooldown_minutes: int = 30
    digest_hour: int = 21

    # Proactive reports
    proactive_enabled: bool = True
    proactive_digest_hour: int = 9
    proactive_min_items_24h: int = 10
    proactive_trigger_threshold: int = 10
    proactive_temperature: float = 0.3
    proactive_max_tokens: int = 700

    # Database
    db_path: str = "./data/pipeline.db"


def _expand_env(value: str, env: dict[str, str]) -> str:
    """Replace ${VAR} references with values from env dict."""
    return re.sub(
        r"\$\{([^}]+)\}",
        lambda m: env.get(m.group(1), m.group(0)),
        value,
    )


def load_channels(
    channels_file: Path | None = None,
    settings: Settings | None = None,
) -> list[ChannelConfig]:
    if channels_file is None:
        channels_file = BASE_DIR / "config" / "channels.yaml"
    if settings is None:
        settings = get_settings()

    env_map: dict[str, str] = {
        k: v for k, v in os.environ.items()
    }
    # Also expose pydantic settings as env vars for expansion
    env_map.update({
        "TG_CHAT_AI_CODING": settings.tg_chat_ai_coding or "",
        "TG_CHAT_AI_MODELS": settings.tg_chat_ai_models or "",
        "TG_CHAT_ERP_MES": settings.tg_chat_erp_mes or "",
    })

    with open(channels_file, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    channels: list[ChannelConfig] = []
    for ch in raw.get("channels", []):
        chat_id = _expand_env(str(ch["telegram_chat_id"]), env_map)
        sources = [
            SourceConfig(**src) for src in ch.get("sources", [])
        ]
        channels.append(
            ChannelConfig(
                id=ch["id"],
                name=ch["name"],
                description=ch["description"],
                classifier_prompt_file=ch["classifier_prompt_file"],
                summarizer_prompt_file=ch["summarizer_prompt_file"],
                telegram_chat_id=chat_id,
                schedule_minutes=ch.get("schedule_minutes", 30),
                enabled=ch.get("enabled", True),
                sources=sources,
            )
        )
    return channels


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
