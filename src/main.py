"""Main entry point — orchestrates the RSS → LLM → Telegram pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))

import db
from alert_manager import AlertManager
from classifier import Classifier
from config import Settings, get_settings, load_channels
from feed_poller import poll_all_sources
from health_monitor import HealthMonitor
from models import ChannelConfig, JobState
from prompt_manager import PromptManager
from summarizer import Summarizer
from telegram_sender import TelegramSender

log = structlog.get_logger(__name__)


async def pipeline_tick(
    channels: list[ChannelConfig],
    classifier: Classifier,
    summarizer: Summarizer,
    sender: TelegramSender,
    settings: Settings,
) -> None:
    """One full pipeline cycle: poll → classify → summarize → send."""
    log.info("pipeline_tick_start")

    # 1. Poll all RSS sources
    await poll_all_sources(channels, max_text_length=settings.max_text_length)

    # 2. Classify + summarize for each channel
    for ch in channels:
        if not ch.enabled:
            continue
        await classifier.run_batch(ch)
        await summarizer.run_batch(ch)

        # Move READY_TO_SEND jobs into Telegram outbox
        jobs = await db.acquire_jobs(
            ch.id, JobState.READY_TO_SEND, 50, settings.job_lease_minutes
        )
        for job in jobs:
            await sender.enqueue_job(job, ch)
            job.state = JobState.SENT
            await db.update_job(job)

    # 3. Flush outbox
    sent = await sender.flush_outbox()
    log.info("pipeline_tick_done", sent=sent)


async def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    settings = get_settings()
    db.set_db_path(settings.db_path)
    await db.init_db()

    channels = load_channels(settings=settings)
    enabled = [ch for ch in channels if ch.enabled]
    log.info("channels_loaded", count=len(enabled))

    pm = PromptManager()
    sender = TelegramSender(settings)
    classifier = Classifier(settings, pm)
    summarizer = Summarizer(settings, pm)
    alert_mgr = AlertManager(sender, settings.tg_admin_chat_id, settings.alert_cooldown_minutes)
    health_mon = HealthMonitor(settings, enabled, alert_mgr, sender)

    scheduler = AsyncIOScheduler()

    # Main pipeline job
    scheduler.add_job(
        pipeline_tick,
        trigger=IntervalTrigger(minutes=settings.poll_interval_minutes),
        args=[enabled, classifier, summarizer, sender, settings],
        id="pipeline",
        max_instances=1,
        coalesce=True,
    )

    # Outbox flush — runs independently so slow summarization doesn't block sends
    scheduler.add_job(
        sender.flush_outbox,
        trigger=IntervalTrigger(minutes=2),
        id="flush_outbox",
        max_instances=1,
        coalesce=True,
    )

    # Health checks
    scheduler.add_job(
        health_mon.run_checks,
        trigger=IntervalTrigger(minutes=settings.health_check_interval_minutes),
        id="health",
        max_instances=1,
    )

    # Daily digest
    scheduler.add_job(
        health_mon.send_daily_digest,
        trigger=CronTrigger(hour=settings.digest_hour, minute=0),
        id="digest",
    )

    scheduler.start()
    log.info("scheduler_started", interval_min=settings.poll_interval_minutes)

    # Run immediately on startup
    await pipeline_tick(enabled, classifier, summarizer, sender, settings)

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        log.info("shutdown")
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
