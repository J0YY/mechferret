"""The interpretability experiment engine.

:class:`InterpEngine` executes an :class:`~mechferret.models.ExperimentSpec` by
running its probe across several seeds and aggregating the readings into an
:class:`~mechferret.models.ExperimentResult` with the three rigor signals the
critic cares about:

- **effect size** -- the mean signed effect across seeds.
- **significance** -- the effect clears a probe-specific floor *and* separates
  from its matched control by more than the cross-seed noise.
- **reproducibility** -- the sign is stable across every seed and the relative
  spread is small.
"""

from __future__ import annotations

import statistics

from ..models import ExperimentResult, ExperimentSpec
from ..text import stable_id
from .backends import resolve_backend
from .probes import get_probe
from .tasks import get_task

# Minimum |effect - control| that counts as a real effect, per probe unit.
MIN_EFFECT_BY_UNIT = {
    "logit_diff": 0.5,
    "prob_diff": 0.1,
    "recovered_fraction": 0.15,
    "logit_attribution": 0.5,
    "final_correct_prob": 0.25,
}
_DEFAULT_MIN_EFFECT = 0.3


class InterpEngine:
    def __init__(self, model: str = "gpt2", backend: str = "auto") -> None:
        self.model = model
        self.requested_backend = backend
        self._backends: dict[tuple[str, str], object] = {}

    def backend_for(self, model: str | None = None, backend: str | None = None):
        key = (model or self.model, backend or self.requested_backend)
        if key not in self._backends:
            self._backends[key] = resolve_backend(key[0], key[1])
        return self._backends[key]

    # Backwards-compatible private alias.
    _backend = backend_for

    def run_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        result_id = stable_id("exp", f"{spec.id}:{spec.probe}:{spec.target}")
        try:
            task = get_task(spec.task)
            probe = get_probe(spec.probe)
            backend = self._backend(spec.model or self.model, spec.backend or self.requested_backend)
        except (KeyError, RuntimeError) as exc:
            return ExperimentResult(
                id=result_id,
                spec_id=spec.id,
                probe=spec.probe,
                status="error",
                effect_size=0.0,
                baseline=0.0,
                error=str(exc),
                target=spec.target,
            )

        seeds = spec.seeds or [0]
        effects: list[float] = []
        controls: list[float] = []
        observations: list[str] = []
        unit = spec.metric
        extra: dict = {}
        for seed in seeds:
            reading = probe(backend, task, spec.target, seed)
            effects.append(reading.effect)
            controls.append(reading.control)
            unit = reading.unit
            extra = reading.extra or extra
            if seed == seeds[0]:
                observations.extend(reading.observations)

        effect_mean = round(statistics.fmean(effects), 4)
        control_mean = round(statistics.fmean(controls), 4)
        effect_std = round(statistics.pstdev(effects), 4) if len(effects) > 1 else 0.0
        separation = abs(effect_mean - control_mean)
        floor = MIN_EFFECT_BY_UNIT.get(unit, _DEFAULT_MIN_EFFECT)

        significant = separation >= floor and separation >= 2 * (effect_std + 1e-6)
        reproduced = self._reproduced(effects, effect_mean)

        backend_used = getattr(backend, "name", "synthetic")
        metrics = {
            "effect_mean": effect_mean,
            "control_mean": control_mean,
            "separation": round(separation, 4),
            "effect_std": effect_std,
            "n_seeds": float(len(seeds)),
            "floor": floor,
        }
        for key, value in extra.items():
            if key == "trajectory":
                continue
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        observations.append(
            f"Aggregated over {len(seeds)} seeds: effect={effect_mean:+.3f} ± {effect_std:.3f}, "
            f"control={control_mean:+.3f}, separation={separation:.3f} "
            f"(floor {floor}); significant={significant}, reproduced={reproduced}."
        )
        evidence_text = self._evidence_text(spec, task, effect_mean, control_mean, significant, reproduced, backend_used)

        return ExperimentResult(
            id=result_id,
            spec_id=spec.id,
            probe=spec.probe,
            status="ran",
            effect_size=effect_mean,
            baseline=control_mean,
            per_seed=[round(value, 4) for value in effects],
            significant=significant,
            reproduced=reproduced,
            metrics=metrics,
            observations=observations,
            evidence_text=evidence_text,
            backend_used=backend_used,
            gpu_seconds=0.0,
            target=spec.target,
        )

    def run_specs(self, specs: list[ExperimentSpec]) -> list[ExperimentResult]:
        return [self.run_spec(spec) for spec in specs]

    @staticmethod
    def _reproduced(effects: list[float], mean: float) -> bool:
        if len(effects) < 2:
            return False
        if abs(mean) < 1e-3:
            # A genuine near-zero effect reproduces only if every seed is also near zero.
            return all(abs(value) < 0.1 for value in effects)
        sign = 1 if mean > 0 else -1
        if any((value > 0) - (value < 0) != sign for value in effects):
            return False
        rel_spread = statistics.pstdev(effects) / abs(mean)
        return rel_spread < 0.5

    @staticmethod
    def _evidence_text(spec, task, effect, control, significant, reproduced, backend_used: str) -> str:
        verdict = "significant and reproducible" if (significant and reproduced) else (
            "significant but not reproduced" if significant else "not significant"
        )
        target = ", ".join(f"{k}={v}" for k, v in spec.target.items()) or "model-wide"
        return (
            f"Experiment {spec.name} ({spec.probe}) on {spec.model}/{task.name} at [{target}] "
            f"measured effect {effect:+.3f} vs control {control:+.3f} -> {verdict} "
            f"[backend={backend_used}]. Task reference: {task.reference}"
        )
