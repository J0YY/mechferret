from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .models import Source
from .text import stable_id

TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm", ".json", ".csv"}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _positive_int(value: Any, default: int, *, upper: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        parsed = default
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def load_sources(paths: list[str] | None = None, urls: list[str] | None = None) -> list[Source]:
    sources: list[Source] = []
    for raw in _items(paths):
        path_value = _text(raw).strip()
        if not path_value:
            continue
        path = Path(path_value).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS:
                    sources.append(load_file_source(child))
        elif path.is_file():
            if path.suffix.lower() in TEXT_EXTENSIONS:
                sources.append(load_file_source(path))
        else:
            raise FileNotFoundError(f"Source path not found: {path_value}")
    for url in _items(urls):
        url_value = _text(url).strip()
        if url_value:
            sources.append(fetch_url_source(url_value))
    return dedupe_sources(sources)


def load_file_source(path: Path) -> Source:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = infer_title(text, path.name)
    normalized = strip_redundant_title(normalize_text(text), title)
    source_id = stable_id("src", f"{path.resolve()}:{text[:500]}")
    return Source(
        id=source_id,
        title=title,
        text=normalized,
        url=str(path.resolve()),
        kind="file",
        metadata={"path": str(path.resolve()), "bytes": path.stat().st_size},
    )


def fetch_url_source(url: str, timeout: int = 15) -> Source:
    url = _text(url).strip()
    if not url:
        raise ValueError("URL source is empty")
    timeout = _positive_int(timeout, 15, upper=120)
    request = Request(url, headers={"User-Agent": "MechFerret/0.1"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_000)
        headers = getattr(response, "headers", {})
        content_type = headers.get("content-type", "") if hasattr(headers, "get") else ""
    text = raw.decode("utf-8", errors="ignore")
    title = infer_title(text, url)
    normalized = strip_redundant_title(normalize_text(text), title)
    return Source(
        id=stable_id("src", f"{url}:{text[:500]}"),
        title=title,
        text=normalized,
        url=url,
        kind="url",
        metadata={"content_type": content_type, "bytes": len(raw)},
    )


def infer_title(text: str, fallback: str) -> str:
    text = _text(text)
    fallback = _text(fallback).strip() or "source"
    md_heading = re.search(r"^\s*#\s+(.+)$", text, re.MULTILINE)
    if md_heading:
        return md_heading.group(1).strip()
    html_title = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if html_title:
        return re.sub(r"\s+", " ", html_title.group(1)).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("title", "name", "headline"):
                if key in parsed and isinstance(parsed[key], str):
                    return parsed[key]
    except (TypeError, json.JSONDecodeError):
        pass
    return os.path.basename(fallback)


def normalize_text(text: str) -> str:
    text = _text(text)
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_redundant_title(text: str, title: str) -> str:
    text = _text(text)
    title = _text(title)
    if not title:
        return text
    normalized_title = re.escape(title.strip())
    return re.sub(rf"^{normalized_title}\s*", "", text, count=1).strip()


def dedupe_sources(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    unique: list[Source] = []
    for source in _items(sources):
        raw_text = getattr(source, "text", None)
        source_id = _text(getattr(source, "id", "")).strip()
        if raw_text is None and not source_id:
            continue
        fingerprint_value = raw_text if raw_text is not None else f"empty:{source_id}"
        fingerprint = stable_id("fp", fingerprint_value)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(source)
    return unique


def example_corpus_path() -> Path:
    packaged = Path(__file__).resolve().parent / "seed_corpus"
    if packaged.exists():
        return packaged
    return Path(__file__).resolve().parent.parent / "examples" / "seed_corpus"
