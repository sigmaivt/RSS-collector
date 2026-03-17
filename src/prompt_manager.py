"""Prompt manager with hot-reload support."""

from __future__ import annotations

import time
from pathlib import Path


class PromptManager:
    """Loads prompts from disk and reloads them if the file changes."""

    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "prompts"
        self._dir = Path(prompts_dir)
        self._cache: dict[str, tuple[float, str]] = {}  # filename → (mtime, content)

    def get(self, filename: str) -> str:
        path = self._dir / filename
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {path}")

        cached = self._cache.get(filename)
        if cached is None or cached[0] != mtime:
            content = path.read_text(encoding="utf-8")
            self._cache[filename] = (mtime, content)
            return content
        return cached[1]
