"""HTML cleaning and text normalization."""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    """Strip HTML tags and return plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return normalize_text(text)


def normalize_text(text: str) -> str:
    """Collapse whitespace, normalize unicode."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def build_clean_text(title: str, raw_content: str | None, max_chars: int = 15000) -> str:
    """Combine title + body into clean plain text."""
    parts = [title]
    if raw_content:
        body = clean_html(raw_content) if "<" in raw_content else normalize_text(raw_content)
        if body:
            parts.append(body)
    combined = " ".join(parts)
    return truncate(combined, max_chars)
