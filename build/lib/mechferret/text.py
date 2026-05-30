from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
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


def stable_id(prefix: str, value: str, length: int = 12) -> str:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text) if tok.lower() not in STOPWORDS]


def term_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def sentence_split(text: str) -> list[str]:
    candidates: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        candidates.extend(SENTENCE_RE.split(paragraph))
    return [s.strip() for s in candidates if 35 <= len(s.strip()) <= 500]


def domain(url: str) -> str:
    if not url:
        return "local"
    parsed = urlparse(url)
    return parsed.netloc.lower() or parsed.path.split("/")[0].lower() or "local"


def cosine_overlap(a: str, b: str) -> float:
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


def has_negation(text: str) -> bool:
    return bool(set(tokenize(text)) & NEGATIONS)


def compact_text(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."

