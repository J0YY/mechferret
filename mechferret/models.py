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
class Hypothesis:
    """A falsifiable claim about a model's internal mechanism."""

    id: str
    statement: str
    rationale: str
    task: str
    predicted_effect: str
    target: dict[str, Any] = field(default_factory=dict)
    status: str = "open"  # open | confirmed | refuted | inconclusive
    confidence: float = 0.0
    experiment_ids: list[str] = field(default_factory=list)
    parent_id: str = ""
    source_ids: list[str] = field(default_factory=list)  # prior-art grounding


@dataclass(slots=True)
class ExperimentSpec:
    """A reproducible recipe for one interpretability experiment."""

    id: str
    name: str
    probe: str  # logit_lens | activation_patching | head_ablation | attention_pattern | direct_logit_attribution
    model: str
    task: str
    target: dict[str, Any]  # e.g. {"layer": 5, "head": 1}
    metric: str = "logit_diff"
    controls: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    backend: str = "auto"  # auto | synthetic | transformer_lens | modal
    hypothesis_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentResult:
    """The outcome of running an ExperimentSpec, with rigor metadata."""

    id: str
    spec_id: str
    probe: str
    status: str  # ran | error | skipped
    effect_size: float
    baseline: float
    per_seed: list[float] = field(default_factory=list)
    significant: bool = False
    reproduced: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    evidence_text: str = ""
    backend_used: str = ""
    gpu_seconds: float = 0.0
    error: str = ""
    target: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Discovery:
    """A confirmed mechanistic finding backed by reproducible experiments."""

    id: str
    statement: str
    confidence: float
    novelty: float
    effect_size: float
    reproducibility: float
    supporting_experiments: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    hypothesis_id: str = ""


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
    provenance: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    experiments: list[ExperimentResult] = field(default_factory=list)
    discoveries: list[Discovery] = field(default_factory=list)
    mode: str = "literature"  # literature | discovery

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
