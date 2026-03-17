"""RSS feed poller — fetches and stores new items."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional

import feedparser
import httpx
import structlog

import db
from models import ChannelConfig, RssItem, SourceConfig
from normalizer import build_clean_text

log = structlog.get_logger(__name__)


async def fetch_feed(url: str, client: httpx.AsyncClient, timeout: int = 20) -> list[dict]:
    """Fetch an RSS/Atom feed and return raw entries."""
    try:
        resp = await client.get(url, timeout=timeout)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.text)
        return parsed.entries
    except Exception as exc:
        log.warning("feed_fetch_failed", url=url, error=str(exc))
        return []


def _entry_guid(entry: dict) -> str:
    return entry.get("id") or entry.get("link") or entry.get("title", "")


def _entry_link(entry: dict) -> str:
    return entry.get("link") or entry.get("id") or ""


def _entry_content(entry: dict) -> str:
    content = entry.get("content")
    if content and isinstance(content, list):
        return content[0].get("value", "")
    return entry.get("summary") or entry.get("description") or ""


def _entry_published(entry: dict) -> Optional[datetime]:
    ts = entry.get("published_parsed") or entry.get("updated_parsed")
    if ts:
        try:
            return datetime(*ts[:6])
        except Exception:
            pass
    return None


async def poll_source(
    source: SourceConfig,
    channels: list[ChannelConfig],
    client: httpx.AsyncClient,
    max_text_length: int = 15000,
) -> tuple[int, int]:
    """Poll one source; return (items_found, items_new)."""
    start = time.monotonic()
    entries = await fetch_feed(source.url, client)
    items_found = len(entries)
    items_new = 0
    error: Optional[str] = None

    try:
        for entry in entries:
            guid = _entry_guid(entry)
            if not guid:
                continue

            item_id = db.make_item_id(source.id, guid)
            raw_content = _entry_content(entry)
            title = entry.get("title", "").strip()
            clean_text = build_clean_text(title, raw_content, max_text_length)

            item = RssItem(
                item_id=item_id,
                source_id=source.id,
                guid=guid,
                link=_entry_link(entry),
                title=title or guid,
                raw_content=raw_content[:50000] if raw_content else None,
                clean_text=clean_text,
                published=_entry_published(entry),
            )

            is_new = await db.upsert_item(item)
            if is_new:
                items_new += 1
                for ch in channels:
                    if source.id in {s.id for s in ch.sources}:
                        await db.create_job(item_id, ch.id)

    except Exception as exc:
        error = str(exc)
        log.error("poll_source_error", source=source.id, error=error)

    duration_ms = int((time.monotonic() - start) * 1000)
    await db.log_poll(source.id, items_found, items_new, error, duration_ms)
    log.info(
        "poll_done",
        source=source.id,
        found=items_found,
        new=items_new,
        ms=duration_ms,
    )
    return items_found, items_new


async def poll_all_sources(
    channels: list[ChannelConfig],
    max_text_length: int = 15000,
    concurrency: int = 5,
) -> None:
    """Poll all enabled sources across all channels."""
    seen: set[str] = set()
    sources: list[tuple[SourceConfig, list[ChannelConfig]]] = []

    for ch in channels:
        if not ch.enabled:
            continue
        for src in ch.sources:
            if src.id not in seen:
                seen.add(src.id)
                sources.append((src, []))
            # attach this channel to the source entry
            for item in sources:
                if item[0].id == src.id:
                    if ch not in item[1]:
                        item[1].append(ch)

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def bounded_poll(src: SourceConfig, chs: list[ChannelConfig]) -> None:
            async with sem:
                await poll_source(src, chs, client, max_text_length)

        await asyncio.gather(*[bounded_poll(s, c) for s, c in sources])
