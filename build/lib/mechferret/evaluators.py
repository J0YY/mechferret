from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluatorTemplate:
    name: str
    venue: str
    criteria: tuple[str, ...]
    must_have: tuple[str, ...]
    common_reject_reasons: tuple[str, ...]


TEMPLATES = {
    "neurips": EvaluatorTemplate(
        name="neurips",
        venue="NeurIPS main",
        criteria=("novelty", "technical correctness", "empirical rigor", "clarity", "broader impact", "limitations"),
        must_have=("baselines", "ablations", "error analysis", "compute budget", "reproducibility plan"),
        common_reject_reasons=("incremental contribution", "weak experiments", "unclear novelty", "missing baselines"),
    ),
    "biology": EvaluatorTemplate(
        name="biology",
        venue="Computational biology review",
        criteria=("mechanistic plausibility", "assay quality", "dataset fit", "reproducibility", "statistical power"),
        must_have=("source methods", "sample sizes", "negative controls", "replication evidence"),
        common_reject_reasons=("overstated mechanism", "weak assay match", "missing replication", "dataset leakage"),
    ),
    "law": EvaluatorTemplate(
        name="law",
        venue="Legal research memo",
        criteria=("authority", "jurisdiction", "recency", "procedural posture", "contrary precedent"),
        must_have=("primary authority", "citations", "jurisdiction match", "counterarguments"),
        common_reject_reasons=("non-binding authority", "stale law", "missing contrary cases", "fact mismatch"),
    ),
    "coding": EvaluatorTemplate(
        name="coding",
        venue="Software engineering review",
        criteria=("correctness", "test coverage", "maintainability", "security", "performance"),
        must_have=("tests", "regression analysis", "failure modes", "API compatibility"),
        common_reject_reasons=("untested edge cases", "breaking API", "security risk", "over-complexity"),
    ),
}


def template_for_venue(venue: str) -> EvaluatorTemplate:
    lowered = venue.lower()
    if "neurips" in lowered or "icml" in lowered or "iclr" in lowered:
        return TEMPLATES["neurips"]
    if "bio" in lowered or "biology" in lowered or "genomic" in lowered:
        return TEMPLATES["biology"]
    if "law" in lowered or "legal" in lowered:
        return TEMPLATES["law"]
    if "code" in lowered or "software" in lowered or "coding" in lowered:
        return TEMPLATES["coding"]
    return TEMPLATES["neurips"]

