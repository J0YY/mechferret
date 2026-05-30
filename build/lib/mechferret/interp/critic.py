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

from ..models import ExperimentResult, Hypothesis


class ExperimentCritic:
    def evaluate(
        self,
        hypotheses: list[Hypothesis],
        results: list[ExperimentResult],
    ) -> tuple[list[str], dict[str, float]]:
        ran = [r for r in results if r.status == "ran"]
        errored = [r for r in results if r.status == "error"]
        significant = [r for r in ran if r.significant]
        reproduced = [r for r in significant if r.reproduced]
        controlled = [r for r in ran if r.baseline is not None]

        confirmed = [h for h in hypotheses if h.status == "confirmed"]
        targeted_confirmed = [
            h for h in confirmed if "layer" in h.target and "head" in h.target
        ]
        inconclusive = [h for h in hypotheses if h.status == "inconclusive"]

        gaps: list[str] = []
        if not significant:
            gaps.append(
                "no head produced a significant, reproducible effect yet; widen the screen to more "
                "layers/heads or increase seed count"
            )
        for hyp in inconclusive:
            gaps.append(
                f"hypothesis {hyp.id} is inconclusive ({hyp.statement[:80]}...); add an independent "
                "probe to triangulate"
            )
        # Triangulation gap: confirmed targeted heads should have >= 2 distinct confirming probe types.
        for hyp in targeted_confirmed:
            confirming = {
                r.probe
                for r in ran
                if r.id in {res.id for res in _results_for(hyp, ran)} and r.significant and r.reproduced
            }
            if len(confirming) < 2:
                gaps.append(
                    f"confirmed head hypothesis {hyp.id} rests on < 2 independent probe types; "
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
            "open_hypotheses": float(len([h for h in hypotheses if h.status == "open"])),
        }
        metrics["rigor_score"] = self._rigor_score(metrics, len(gaps))
        return gaps, metrics

    @staticmethod
    def _rigor_score(metrics: dict[str, float], gap_count: int) -> float:
        score = 0.2
        score += metrics["controlled_fraction"] * 0.2
        score += metrics["reproducibility_rate"] * 0.25
        score += min(metrics["confirmed_mechanisms"] / 2.0, 1.0) * 0.25
        score += min(metrics["significant_effects"] / 4.0, 1.0) * 0.1
        score -= min(gap_count / 5.0, 1.0) * 0.12
        score -= min(metrics["experiments_errored"] / 3.0, 1.0) * 0.1
        return round(max(0.0, min(0.99, score)), 3)


def _results_for(hyp: Hypothesis, results: list[ExperimentResult]) -> list[ExperimentResult]:
    ids = set(hyp.experiment_ids)
    return [r for r in results if r.spec_id in ids or r.id in ids]
