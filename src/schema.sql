-- RSS → LLM → Telegram Pipeline
-- SQLite schema (WAL mode)

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ══════════════════════════════════════
-- Raw items from RSS feeds
-- ══════════════════════════════════════
CREATE TABLE IF NOT EXISTS items (
    item_id     TEXT PRIMARY KEY,          -- SHA256(source_id + guid/link)
    source_id   TEXT NOT NULL,             -- e.g. "techcrunch_rsshub"
    guid        TEXT,                      -- original RSS guid
    link        TEXT,                      -- original URL
    title       TEXT NOT NULL,
    raw_content TEXT,                      -- original HTML/text (for debug)
    clean_text  TEXT,                      -- normalized plain text
    published   TIMESTAMP,                 -- from RSS feed
    fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_id, guid)
);

CREATE INDEX IF NOT EXISTS idx_items_fetched ON items(fetched_at);
CREATE INDEX IF NOT EXISTS idx_items_source  ON items(source_id);

-- ══════════════════════════════════════
-- Per-channel processing jobs
-- One item can have jobs in multiple channels
-- ══════════════════════════════════════
CREATE TABLE IF NOT EXISTS channel_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         TEXT NOT NULL REFERENCES items(item_id),
    channel_id      TEXT NOT NULL,          -- e.g. "ai_coding", "ai_models", "erp_mes"
    state           TEXT NOT NULL DEFAULT 'NEW',
    -- States: NEW → CLASSIFYING → REJECTED|VALIDATED → SUMMARIZING → READY_TO_SEND → SENT|SEND_FAILED
    
    classifier_result TEXT,                 -- "VALID" or "NOT" or raw LLM response
    summary         TEXT,                   -- 2-3 sentences in Russian
    
    attempts        INTEGER DEFAULT 0,
    last_error      TEXT,
    leased_at       TIMESTAMP,             -- for job lock (lease pattern)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(item_id, channel_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_state    ON channel_jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_channel  ON channel_jobs(channel_id, state);
CREATE INDEX IF NOT EXISTS idx_jobs_leased   ON channel_jobs(leased_at) WHERE state IN ('CLASSIFYING', 'SUMMARIZING');

-- ══════════════════════════════════════
-- Dead letter queue
-- ══════════════════════════════════════
CREATE TABLE IF NOT EXISTS dead_letter_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         TEXT NOT NULL,
    channel_id      TEXT NOT NULL,
    original_state  TEXT NOT NULL,
    error           TEXT,
    attempts        INTEGER,
    moved_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ══════════════════════════════════════
-- Telegram outbox (idempotent sends)
-- ══════════════════════════════════════
CREATE TABLE IF NOT EXISTS telegram_outbox (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         TEXT NOT NULL,
    channel_id      TEXT NOT NULL,
    chat_id         TEXT NOT NULL,
    message_text    TEXT NOT NULL,
    sent            BOOLEAN DEFAULT FALSE,
    tg_message_id   INTEGER,               -- Telegram message ID after send
    attempts        INTEGER DEFAULT 0,
    last_error      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sent_at         TIMESTAMP,
    
    UNIQUE(item_id, channel_id)
);

-- ══════════════════════════════════════
-- Pipeline metrics
-- ══════════════════════════════════════
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name     TEXT NOT NULL,
    metric_value    REAL,
    tags            TEXT                    -- JSON: {"channel": "ai_coding", "source": "hackernews"}
);

CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON pipeline_metrics(metric_name, ts);

-- ══════════════════════════════════════
-- Feed poll log (for freshness monitoring)
-- ══════════════════════════════════════
CREATE TABLE IF NOT EXISTS feed_poll_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       TEXT NOT NULL,
    polled_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    items_found     INTEGER DEFAULT 0,
    items_new       INTEGER DEFAULT 0,
    error           TEXT,
    duration_ms     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_poll_source ON feed_poll_log(source_id, polled_at);
