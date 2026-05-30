from __future__ import annotations

import math
from collections import Counter, defaultdict

from .models import EvidenceChunk, Source
from .text import stable_id, tokenize


def chunk_source(source: Source, max_tokens: int = 145, overlap: int = 28) -> list[EvidenceChunk]:
    words = source.text.split()
    if not words:
        return []
    chunks: list[EvidenceChunk] = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + max_tokens]
        text = " ".join(window).strip()
        if len(text) > 35:
            chunk_id = stable_id("ev", f"{source.id}:{index}:{text[:200]}")
            chunks.append(
                EvidenceChunk(
                    id=chunk_id,
                    source_id=source.id,
                    title=source.title,
                    text=text,
                    url=source.url,
                )
            )
        if start + max_tokens >= len(words):
            break
        start += max_tokens - overlap
        index += 1
    return chunks


class BM25Index:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks
        self.doc_terms: list[Counter[str]] = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.doc_lengths = [sum(counter.values()) for counter in self.doc_terms]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        self.df: dict[str, int] = defaultdict(int)
        for counter in self.doc_terms:
            for term in counter:
                self.df[term] += 1

    @classmethod
    def from_sources(cls, sources: list[Source]) -> "BM25Index":
        chunks: list[EvidenceChunk] = []
        for source in sources:
            chunks.extend(chunk_source(source))
        return cls(chunks)

    def search(self, query: str, limit: int = 8) -> list[EvidenceChunk]:
        terms = tokenize(query)
        if not terms:
            return []
        scores: list[tuple[float, EvidenceChunk]] = []
        n_docs = len(self.chunks)
        for idx, counter in enumerate(self.doc_terms):
            score = 0.0
            dl = self.doc_lengths[idx] or 1
            for term in terms:
                tf = counter.get(term, 0)
                if tf == 0:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / self.avgdl))
            if score > 0:
                chunk = self.chunks[idx]
                scores.append(
                    (
                        score,
                        EvidenceChunk(
                            id=chunk.id,
                            source_id=chunk.source_id,
                            title=chunk.title,
                            text=chunk.text,
                            url=chunk.url,
                            score=round(score, 4),
                            highlights=[term for term in terms if term in counter][:8],
                        ),
                    )
                )
        scores.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scores[:limit]]
