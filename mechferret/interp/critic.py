"""Experiment critic: enforces interpretability rigor and drives the next round.

This is the experimental analogue of the literature ``Critic``. Instead of
citation density and source diversity, it scores the things that make a
mechanistic claim trustworthy:

- **controls** -- was every effect compared against a matched control site?
- **significance** -- did the effect clear its floor and beat cross-seed noise?
- **reproducibility** -- was the sign stable across seeds?
- **triangulation** -- is each confirmed head backed by >= 2 independent probes?

It returns rigor gaps (which become the next experiments) and a metrics dict
the controller folds into the run's readiness score.
"""

from __future__ import annotations

import math
from typing import Any

from ..models import ExperimentResult, Hypothesis


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, bytes, dict)) or value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    return [_text(item).strip() for item in _items(value) if _text(item).strip()]


def _bool(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if type(value) in {int, float}:
        return bool(value)
    return False


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _status(value: Any) -> str:
    return _text(getattr(value, "status", "")).strip()


def _has_head_target(value: Any) -> bool:
    target = _mapping(getattr(value, "target", {}))
    return "layer" in target and "head" in target


class ExperimentCritic:
    def evaluate(
        self,
        hypotheses: list[Hypothesis],
        results: list[ExperimentResult],
    ) -> tuple[list[str], dict[str, float]]:
        result_rows = _items(results)
        hypothesis_rows = _items(hypotheses)
        ran = [r for r in result_rows if _status(r) == "ran"]
        errored = [r for r in result_rows if _status(r) == "error"]
        significant = [r for r in ran if _bool(getattr(r, "significant", False))]
        reproduced = [r for r in significant if _bool(getattr(r, "reproduced", False))]
        controlled = [r for r in ran if getattr(r, "baseline", None) is not None]

        confirmed = [h for h in hypothesis_rows if _status(h) == "confirmed"]
        targeted_confirmed = [h for h in confirmed if _has_head_target(h)]
        inconclusive = [h for h in hypothesis_rows if _status(h) == "inconclusive"]

        gaps: list[str] = []
        if not reproduced:
            gaps.append(
                "no head produced a significant, reproducible effect yet; widen the screen to more "
                "layers/heads or increase seed count"
            )
        for hyp in inconclusive:
            hyp_id = _text(getattr(hyp, "id", "")).strip() or "unknown"
            statement = _text(getattr(hyp, "statement", "")).strip()
            gaps.append(
                f"hypothesis {hyp_id} is inconclusive ({statement[:80]}...); add an independent "
                "probe to triangulate"
            )
        # Triangulation gap: confirmed targeted heads should have >= 2 distinct confirming probe types.
        for hyp in targeted_confirmed:
            confirming = {
                _text(getattr(r, "probe", "")).strip()
                for r in _results_for(hyp, ran)
                if _bool(getattr(r, "significant", False))
                and _bool(getattr(r, "reproduced", False))
            }
            confirming.discard("")
            if len(confirming) < 2:
                hyp_id = _text(getattr(hyp, "id", "")).strip() or "unknown"
                gaps.append(
                    f"confirmed head hypothesis {hyp_id} rests on < 2 independent probe types; "
                    "add attention-pattern or activation-patching evidence"
                )
        if errored:
            gaps.append(f"{len(errored)} experiment(s) errored; inspect specs/backend before trusting metrics")

        metrics = {
            "experiments_ran": float(len(ran)),
            "experiments_errored": float(len(errored)),
            "significant_effects": float(len(significant)),
            "reproduced_effects": float(len(reproduced)),
            "controlled_fraction": round(len(controlled) / max(len(ran), 1), 3),
            "reproducibility_rate": round(len(reproduced) / max(len(significant), 1), 3) if significant else 0.0,
            "confirmed_hypotheses": float(len(confirmed)),
            "confirmed_mechanisms": float(len(targeted_confirmed)),
            "open_hypotheses": float(len([h for h in hypothesis_rows if _status(h) == "open"])),
        }
        metrics["rigor_score"] = self._rigor_score(metrics, len(gaps))
        return gaps, metrics

    @staticmethod
    def _rigor_score(metrics: dict[str, float], gap_count: int) -> float:
        score = 0.2
        score += _number(metrics.get("controlled_fraction", 0.0)) * 0.2
        score += _number(metrics.get("reproducibility_rate", 0.0)) * 0.25
        score += min(_number(metrics.get("confirmed_mechanisms", 0.0)) / 2.0, 1.0) * 0.25
        score += min(_number(metrics.get("significant_effects", 0.0)) / 4.0, 1.0) * 0.1
        score -= min(gap_count / 5.0, 1.0) * 0.12
        score -= min(_number(metrics.get("experiments_errored", 0.0)) / 3.0, 1.0) * 0.1
        return round(max(0.0, min(0.99, score)), 3)


def _results_for(hyp: Hypothesis, results: list[ExperimentResult]) -> list[ExperimentResult]:
    ids = set(_strings(getattr(hyp, "experiment_ids", [])))
    return [
        r
        for r in _items(results)
        if _text(getattr(r, "spec_id", "")).strip() in ids or _text(getattr(r, "id", "")).strip() in ids
    ]
