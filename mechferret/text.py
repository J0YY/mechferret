from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
NEGATIONS = {"no", "not", "never", "without", "fails", "failed", "lack", "lacks", "insufficient"}
STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "during",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _id_text(value: Any) -> str:
    text = _text(value)
    if text:
        return text
    if value is None:
        return ""
    return str(value)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def stable_id(prefix: Any, value: Any, length: Any = 12) -> str:
    safe_prefix = _id_text(prefix).strip() or "id"
    safe_length = min(_positive_int(length, 12), 64)
    digest = hashlib.sha256(_id_text(value).encode("utf-8", errors="ignore")).hexdigest()
    return f"{safe_prefix}_{digest[:safe_length]}"


def tokenize(text: Any) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(_text(text)) if tok.lower() not in STOPWORDS]


def term_counter(text: Any) -> Counter[str]:
    return Counter(tokenize(text))


def sentence_split(text: Any) -> list[str]:
    candidates: list[str] = []
    for paragraph in re.split(r"\n\s*\n", _text(text)):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        candidates.extend(SENTENCE_RE.split(paragraph))
    return [s.strip() for s in candidates if 35 <= len(s.strip()) <= 500]


def domain(url: Any) -> str:
    url = _text(url).strip()
    if not url:
        return "local"
    parsed = urlparse(url)
    return parsed.netloc.lower() or parsed.path.split("/")[0].lower() or "local"


def cosine_overlap(a: Any, b: Any) -> float:
    ca = term_counter(a)
    cb = term_counter(b)
    if not ca or not cb:
        return 0.0
    shared = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in shared)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def has_negation(text: Any) -> bool:
    return bool(set(tokenize(text)) & NEGATIONS)


def compact_text(text: Any, limit: Any = 220) -> str:
    text = re.sub(r"\s+", " ", _text(text)).strip()
    limit = _positive_int(limit, 220)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 1].rstrip() + "..."
