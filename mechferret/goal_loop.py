from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .controller import MechFerret
from .evaluators import template_for_venue
from .models import ResearchRun, utc_now


class GoalLoop:
    def __init__(self, memory_path: str | Path = ".mechferret/memory.sqlite") -> None:
        self.memory_path = Path(memory_path)

    def run(
        self,
        question: str,
        venue: str,
        target: float,
        source_paths: list[str] | None = None,
        urls: list[str] | None = None,
        out_dir: str | Path = "runs/goal",
        max_iterations: int = 5,
        max_rounds: int = 2,
        provider: str = "auto",
        model: str | None = None,
        include_memory: bool = True,
    ) -> dict[str, Any]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        iterations: list[dict[str, Any]] = []
        best_probability = 0.0
        best_run: ResearchRun | None = None
        working_question = self._goal_question(question, venue, [])

        for index in range(max(1, max_iterations)):
            iteration_dir = out_path / f"iteration_{index + 1:02d}"
            run = MechFerret(self.memory_path).run(
                working_question,
                source_paths=source_paths,
                urls=urls,
                out_dir=iteration_dir,
                max_rounds=max_rounds,
                provider=provider,
                model=model,
                include_memory=include_memory,
            )
            probability = estimate_acceptance_probability(run, venue)
            actions = next_actions(run, venue, probability, target)
            record = {
                "iteration": index + 1,
                "run_id": run.run_id,
                "probability": probability,
                "readiness_score": run.metrics.get("readiness_score", 0),
                "report": run.artifacts.get("html"),
                "gaps": run.gaps,
                "actions": actions,
                "created_at": utc_now(),
            }
            iterations.append(record)
            if probability > best_probability:
                best_probability = probability
                best_run = run
            if probability >= target:
                break
            working_question = self._goal_question(question, venue, actions)

        status = "target_reached" if best_probability >= target else "budget_exhausted"
        summary = {
            "status": status,
            "question": question,
            "venue": venue,
            "target": target,
            "best_probability": best_probability,
            "best_run_id": best_run.run_id if best_run else "",
            "iterations": iterations,
            "created_at": utc_now(),
        }
        artifact = out_path / "goal.json"
        artifact.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["artifact"] = str(artifact)
        return summary

    def _goal_question(self, question: str, venue: str, actions: list[str]) -> str:
        template = template_for_venue(venue)
        suffix = ""
        if actions:
            suffix = "\n\nPrevious critic requested these next actions:\n" + "\n".join(f"- {action}" for action in actions)
        return (
            f"{question}\n\n"
            f"Target venue/bar: {venue}. Evaluate like a program chair or expert reviewer. "
            "Prioritize novelty, evidence quality, experimental rigor, clarity, and unresolved risks."
            f"\nEvaluation criteria: {', '.join(template.criteria)}."
            f"\nMust-have evidence: {', '.join(template.must_have)}."
            f"\nCommon rejection reasons to avoid: {', '.join(template.common_reject_reasons)}."
            f"{suffix}"
        )


def estimate_acceptance_probability(run: ResearchRun, venue: str) -> float:
    readiness = run.metrics.get("readiness_score", 0.0)
    source_diversity = min(run.metrics.get("source_diversity", 0.0) / 5.0, 1.0)
    citation_density = min(run.metrics.get("citation_density", 0.0) / 1.5, 1.0)
    confidence = run.metrics.get("mean_confidence", 0.0)
    gap_penalty = min(len(run.gaps) / 6.0, 1.0)
    contradiction_penalty = min(run.metrics.get("contradiction_pressure", 0.0), 1.0)
    venue_bar = venue_bar_adjustment(venue)
    score = (
        readiness * 0.42
        + source_diversity * 0.16
        + citation_density * 0.14
        + confidence * 0.18
        + min(len(run.claims) / 18.0, 1.0) * 0.10
        - gap_penalty * 0.18
        - contradiction_penalty * 0.14
        - venue_bar
    )
    return round(max(0.01, min(0.99, score)), 3)


def venue_bar_adjustment(venue: str) -> float:
    lowered = venue.lower()
    if "neurips" in lowered or "icml" in lowered or "iclr" in lowered:
        return 0.18
    if "main" in lowered or "top" in lowered:
        return 0.14
    if "workshop" in lowered:
        return 0.06
    return 0.10


def next_actions(run: ResearchRun, venue: str, probability: float, target: float) -> list[str]:
    actions: list[str] = []
    template = template_for_venue(venue)
    if run.gaps:
        actions.extend(f"Resolve gap: {gap}" for gap in run.gaps[:3])
    if run.metrics.get("source_diversity", 0) < 5:
        actions.append("Add independent sources, datasets, baselines, or reviewer-facing external validation.")
    if run.metrics.get("citation_density", 0) < 1.2:
        actions.append("Corroborate the strongest claims with additional evidence chunks.")
    actions.append(f"Address must-have evidence: {', '.join(template.must_have[:4])}.")
    actions.append(f"Preempt rejection reasons: {', '.join(template.common_reject_reasons[:3])}.")
    if probability < target:
        actions.append(f"Increase estimated acceptance probability from {probability:.2f} toward {target:.2f}.")
    return dedupe(actions)[:8]


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
