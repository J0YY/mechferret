from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class Source:
    id: str
    title: str
    text: str
    url: str = ""
    kind: str = "document"
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceChunk:
    id: str
    source_id: str
    title: str
    text: str
    url: str = ""
    score: float = 0.0
    highlights: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanStep:
    id: str
    question: str
    intent: str
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchPlan:
    question: str
    steps: list[PlanStep]
    strategy: str


@dataclass(slots=True)
class Claim:
    id: str
    text: str
    citations: list[str]
    source_ids: list[str]
    confidence: float
    support_score: float
    stance: str = "finding"
    quality_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Contradiction:
    id: str
    claim_a: str
    claim_b: str
    reason: str
    severity: float


@dataclass(slots=True)
class ResearchRun:
    run_id: str
    question: str
    created_at: str
    plan: ResearchPlan
    sources: list[Source]
    evidence: list[EvidenceChunk]
    claims: list[Claim]
    contradictions: list[Contradiction]
    gaps: list[str]
    answer: str
    metrics: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

