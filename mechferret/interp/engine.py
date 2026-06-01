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

import math
import secrets
import statistics
from typing import Any

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
DEFAULT_SEED_COUNT = 3


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _seeds(value: Any, *, default: list[int] | None = None) -> list[int]:
    if not isinstance(value, list):
        return list(default or _runtime_seed_plan())
    seeds: list[int] = []
    for seed in value:
        if type(seed) is bool:
            continue
        try:
            parsed = int(seed)
        except (TypeError, ValueError):
            continue
        if parsed not in seeds:
            seeds.append(parsed)
    return seeds or list(default or _runtime_seed_plan())


def _runtime_seed_plan(count: int = DEFAULT_SEED_COUNT) -> list[int]:
    rng = secrets.SystemRandom()
    seeds: list[int] = []
    while len(seeds) < max(1, int(count)):
        candidate = rng.randrange(1, 2**31 - 1)
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds


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


class InterpEngine:
    def __init__(self, model: str | None = None, backend: str = "auto") -> None:
        self.model = _text(model).strip()
        self.requested_backend = _text(backend).strip() or "auto"
        self._backends: dict[tuple[str, str], object] = {}
        self._default_seeds = _runtime_seed_plan()

    def backend_for(self, model: str | None = None, backend: str | None = None):
        model_name = _text(model).strip() or self.model
        if not model_name:
            raise ValueError("model is required; pass --model or use a skill that declares one.")
        key = (model_name, _text(backend).strip() or self.requested_backend)
        if key not in self._backends:
            self._backends[key] = resolve_backend(key[0], key[1])
        return self._backends[key]

    # Backwards-compatible private alias.
    _backend = backend_for

    def run_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        spec_id = _text(getattr(spec, "id", "")).strip() or "spec"
        probe_name = _text(getattr(spec, "probe", "")).strip()
        model = _text(getattr(spec, "model", "")).strip() or self.model
        target = _mapping(getattr(spec, "target", {}))
        result_id = stable_id("exp", f"{spec_id}:{probe_name}:{target}")
        try:
            task = get_task(_text(getattr(spec, "task", "")))
            probe = get_probe(probe_name)
            backend_raw = _text(getattr(spec, "backend", "")).strip()
            backend_choice = backend_raw if backend_raw and backend_raw != "auto" else self.requested_backend
            backend = self._backend(model, backend_choice)
        except Exception as exc:  # noqa: BLE001 - malformed specs become error result rows
            return self._error_result(result_id, spec_id, probe_name, target, str(exc))

        seeds = _seeds(getattr(spec, "seeds", []), default=self._default_seeds)
        effects: list[float] = []
        controls: list[float] = []
        observations: list[str] = []
        unit = _text(getattr(spec, "metric", "")).strip()
        extra: dict = {}
        for seed in seeds:
            try:
                reading = probe(backend, task, target, seed)
            except Exception as exc:  # noqa: BLE001 - one bad probe should become a result row
                return self._error_result(result_id, spec_id, probe_name, target, str(exc))
            effect = _number(getattr(reading, "effect", None), math.nan)
            control = _number(getattr(reading, "control", None), math.nan)
            if not math.isfinite(effect) or not math.isfinite(control):
                return self._error_result(result_id, spec_id, probe_name, target, "probe returned non-finite effect/control")
            effects.append(effect)
            controls.append(control)
            unit = _text(getattr(reading, "unit", "")).strip() or unit
            reading_extra = getattr(reading, "extra", {})
            if isinstance(reading_extra, dict):
                extra = reading_extra or extra
            if seed == seeds[0]:
                observations.extend(_text(item) for item in _items(getattr(reading, "observations", [])) if _text(item))

        effect_mean = round(statistics.fmean(effects), 4)
        control_mean = round(statistics.fmean(controls), 4)
        effect_std = round(statistics.pstdev(effects), 4) if len(effects) > 1 else 0.0
        separation = abs(effect_mean - control_mean)
        floor = MIN_EFFECT_BY_UNIT.get(unit, _DEFAULT_MIN_EFFECT)

        significant = separation >= floor and separation >= 2 * (effect_std + 1e-6)
        reproduced = self._reproduced(effects, effect_mean)

        backend_used = _text(getattr(backend, "name", "")).strip() or "unknown"
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
            numeric = _number(value, math.nan)
            if math.isfinite(numeric):
                metrics[str(key)] = float(numeric)

        observations.append(
            f"Aggregated over {len(seeds)} seeds: effect={effect_mean:+.3f} ± {effect_std:.3f}, "
            f"control={control_mean:+.3f}, separation={separation:.3f} "
            f"(floor {floor}); significant={significant}, reproduced={reproduced}."
        )
        evidence_text = self._evidence_text(spec, task, effect_mean, control_mean, significant, reproduced, backend_used)

        return ExperimentResult(
            id=result_id,
            spec_id=spec_id,
            probe=probe_name,
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
            target=target,
        )

    def run_specs(self, specs: list[ExperimentSpec]) -> list[ExperimentResult]:
        return [self.run_spec(spec) for spec in _items(specs)]

    @staticmethod
    def _error_result(result_id: str, spec_id: str, probe: str, target: dict[str, Any], error: str) -> ExperimentResult:
        return ExperimentResult(
            id=result_id,
            spec_id=spec_id,
            probe=probe,
            status="error",
            effect_size=0.0,
            baseline=0.0,
            error=error,
            target=target,
        )

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
        target = ", ".join(f"{k}={v}" for k, v in _mapping(getattr(spec, "target", {})).items()) or "model-wide"
        return (
            f"Experiment {_text(getattr(spec, 'name', '')) or 'experiment'} ({_text(getattr(spec, 'probe', ''))}) "
            f"on {_text(getattr(spec, 'model', '')) or 'model'}/{task.name} at [{target}] "
            f"measured effect {effect:+.3f} vs control {control:+.3f} -> {verdict} "
            f"[backend={backend_used}]. Task reference: {task.reference}"
        )
