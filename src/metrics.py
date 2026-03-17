"""Pipeline metrics helpers."""

from __future__ import annotations

import db


async def inc(name: str, tags: dict | None = None) -> None:
    await db.record_metric(name, 1.0, tags)


async def gauge(name: str, value: float, tags: dict | None = None) -> None:
    await db.record_metric(name, value, tags)
