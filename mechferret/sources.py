from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

from .models import Source
from .text import stable_id

TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm", ".json", ".csv"}


def load_sources(paths: list[str] | None = None, urls: list[str] | None = None) -> list[Source]:
    sources: list[Source] = []
    for raw in paths or []:
        path = Path(raw).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS:
                    sources.append(load_file_source(child))
        elif path.is_file():
            if path.suffix.lower() in TEXT_EXTENSIONS:
                sources.append(load_file_source(path))
        else:
            raise FileNotFoundError(f"Source path not found: {raw}")
    for url in urls or []:
        sources.append(fetch_url_source(url))
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
    request = Request(url, headers={"User-Agent": "MechFerret/0.1"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_000)
        content_type = response.headers.get("content-type", "")
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
    except json.JSONDecodeError:
        pass
    return os.path.basename(fallback)


def normalize_text(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_redundant_title(text: str, title: str) -> str:
    if not title:
        return text
    normalized_title = re.escape(title.strip())
    return re.sub(rf"^{normalized_title}\s*", "", text, count=1).strip()


def dedupe_sources(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    unique: list[Source] = []
    for source in sources:
        fingerprint = stable_id("fp", source.text)
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
