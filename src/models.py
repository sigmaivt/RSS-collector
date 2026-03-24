"""Data models for the RSS → LLM → Telegram pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator


class JobState(str, Enum):
    NEW = "NEW"
    CLASSIFYING = "CLASSIFYING"
    REJECTED = "REJECTED"
    VALIDATED = "VALIDATED"
    SUMMARIZING = "SUMMARIZING"
    READY_TO_SEND = "READY_TO_SEND"
    OUTBOX_PENDING = "OUTBOX_PENDING"
    SENT = "SENT"
    SEND_FAILED = "SEND_FAILED"


class RssItem(BaseModel):
    item_id: str
    source_id: str
    guid: Optional[str] = None
    link: Optional[str] = None
    title: str
    raw_content: Optional[str] = None
    clean_text: Optional[str] = None
    published: Optional[datetime] = None
    fetched_at: Optional[datetime] = None


class ChannelJob(BaseModel):
    id: Optional[int] = None
    item_id: str
    channel_id: str
    state: JobState = JobState.NEW
    classifier_result: Optional[str] = None
    summary: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    leased_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TelegramOutboxEntry(BaseModel):
    id: Optional[int] = None
    item_id: str
    channel_id: str
    chat_id: str
    message_text: str
    sent: bool = False
    tg_message_id: Optional[int] = None
    attempts: int = 0
    last_error: Optional[str] = None
    created_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None


class SourceConfig(BaseModel):
    id: str
    type: str  # "rsshub" | "direct"
    url: str
    name: str
    language: Optional[str] = None
    note: Optional[str] = None


class ChannelConfig(BaseModel):
    id: str
    name: str
    description: str
    classifier_prompt_file: str
    summarizer_prompt_file: str
    telegram_chat_id: str
    schedule_minutes: int = 30
    enabled: bool = True
    sources: list[SourceConfig] = []

    @field_validator("telegram_chat_id", mode="before")
    @classmethod
    def strip_env_var(cls, v: str) -> str:
        return str(v)
