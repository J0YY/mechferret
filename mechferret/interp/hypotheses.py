"""Hypothesis generation and updating for autonomous interpretability.

The agent never peeks at the synthetic ground truth. It discovers structure the
way a researcher would:

1. **Screen** -- hypothesise that *some* component drives the behaviour and
   ablate a budgeted set of candidate heads to find which ones matter.
2. **Promote** -- turn each head that survives screening into a targeted,
   falsifiable hypothesis.
3. **Triangulate** -- test each targeted hypothesis with *independent* probes
   (attention pattern, direct logit attribution, activation patching) so a
   confirmation never rests on a single measurement.
4. **Update** -- confirm / refute / mark inconclusive from the evidence.

An optional LLM can sharpen the natural-language statements, but the experiment
loop can still run without one.
"""

from __future__ import annotations

import math
from typing import Any

from ..models import ExperimentResult, ExperimentSpec, Hypothesis
from ..text import stable_id
from .synthetic import _shape
from .tasks import get_task

CONFIRMATORY_PROBES = ("attention_pattern", "direct_logit_attribution", "activation_patching")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _strings(value: Any) -> list[str]:
    return [_text(item).strip() for item in _items(value) if _text(item).strip()]


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _positive_int(value: Any, default: int) -> int:
    if type(value) is bool:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any) -> int | None:
    if type(value) is bool:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _bool(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if type(value) in {int, float}:
        return bool(value)
    return False


def _seeds(value: Any) -> list[int]:
    seeds: list[int] = []
    for seed in _items(value):
        parsed = _nonnegative_int(seed)
        if parsed is not None and parsed not in seeds:
            seeds.append(parsed)
    return seeds or [0, 1, 2]


def _task(name: Any):
    try:
        return get_task(_text(name).strip())
    except Exception:  # noqa: BLE001 - malformed boundary input falls back to the default task
        return get_task("ioi")


def _head_target(value: Any) -> tuple[int, int] | None:
    target = _mapping(value)
    layer = _nonnegative_int(target.get("layer"))
    head = _nonnegative_int(target.get("head"))
    if layer is None or head is None:
        return None
    return layer, head


def _hypothesis(value: Any) -> Hypothesis | None:
    target = _mapping(getattr(value, "target", {}))
    if isinstance(value, Hypothesis):
        value.target = target
        value.experiment_ids = _strings(getattr(value, "experiment_ids", []))
        value.source_ids = _strings(getattr(value, "source_ids", []))
        value.status = _text(getattr(value, "status", "")).strip() or "open"
        value.confidence = _number(getattr(value, "confidence", 0.0))
        return value
    hyp_id = _text(getattr(value, "id", "")).strip()
    statement = _text(getattr(value, "statement", "")).strip()
    if not hyp_id or not statement:
        return None
    return Hypothesis(
        id=hyp_id,
        statement=statement,
        rationale=_text(getattr(value, "rationale", "")).strip(),
        task=_text(getattr(value, "task", "")).strip(),
        predicted_effect=_text(getattr(value, "predicted_effect", "")).strip(),
        target=target,
        status=_text(getattr(value, "status", "")).strip() or "open",
        confidence=_number(getattr(value, "confidence", 0.0)),
        experiment_ids=_strings(getattr(value, "experiment_ids", [])),
        parent_id=_text(getattr(value, "parent_id", "")).strip(),
        source_ids=_strings(getattr(value, "source_ids", [])),
    )


class HypothesisGenerator:
    def __init__(self, model: str | None = None, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
        self.model = _text(model).strip()
        self.seeds = _seeds(seeds)

    # --- round 0: screening ---------------------------------------------------------

    def screen(
        self,
        question: str,
        task_name: str,
        max_heads: int = 96,
        source_ids: list[str] | None = None,
    ) -> tuple[list[Hypothesis], list[ExperimentSpec]]:
        task = _task(task_name)
        max_heads = _positive_int(max_heads, 96)
        source_ids = _strings(source_ids or [])
        n_layers, n_heads, _ = _shape(self.model)
        start_layer = n_layers // 3  # the upper two-thirds carry task-specific computation
        candidates = [
            (layer, head)
            for layer in range(start_layer, n_layers)
            for head in range(n_heads)
        ][:max_heads]

        screen_hyp = Hypothesis(
            id=stable_id("hyp", f"screen:{self.model}:{task.name}"),
            statement=(
                f"At least one attention head in the upper layers of {self.model} is causally "
                f"responsible for the {task.name} behaviour."
            ),
            rationale=(
                f"{task.description} Causal screening via head ablation localises the responsible "
                "components before committing compute to confirmation."
            ),
            task=task.name,
            predicted_effect="Ablating the responsible head(s) significantly reduces the task metric.",
            target={"scope": "upper_layers", "candidates": len(candidates)},
            source_ids=source_ids,
        )
        lens_hyp = Hypothesis(
            id=stable_id("hyp", f"lens:{self.model}:{task.name}"),
            statement=(
                f"The {task.name} prediction in {self.model} is formed in a specific mid-to-late layer "
                "rather than gradually across all layers."
            ),
            rationale="A logit-lens sweep reveals the layer at which the correct token becomes dominant.",
            task=task.name,
            predicted_effect="Correct-token probability rises sharply around one decision layer.",
            target={"probe": "logit_lens"},
            source_ids=source_ids,
        )

        specs: list[ExperimentSpec] = [
            ExperimentSpec(
                id=stable_id("spec", f"screen:{self.model}:{task.name}:{layer}:{head}"),
                name=f"screen ablation {layer}.{head}",
                probe="head_ablation",
                model=self.model,
                task=task.name,
                target={"layer": layer, "head": head},
                metric=task.metric,
                controls=["random control head"],
                seeds=self.seeds,
                hypothesis_id=screen_hyp.id,
            )
            for layer, head in candidates
        ]
        specs.append(
            ExperimentSpec(
                id=stable_id("spec", f"lens:{self.model}:{task.name}"),
                name="logit-lens sweep",
                probe="logit_lens",
                model=self.model,
                task=task.name,
                target={"probe": "logit_lens"},
                metric="final_correct_prob",
                controls=["layer-0 readout"],
                seeds=self.seeds,
                hypothesis_id=lens_hyp.id,
            )
        )
        screen_hyp.experiment_ids = [spec.id for spec in specs if spec.hypothesis_id == screen_hyp.id]
        lens_hyp.experiment_ids = [specs[-1].id]
        return [screen_hyp, lens_hyp], specs

    # --- round 1+: promote screen hits to targeted hypotheses -----------------------

    def promote(
        self,
        screen_results: list[ExperimentResult],
        task_name: str,
        top_k: int = 4,
        source_ids: list[str] | None = None,
    ) -> tuple[list[Hypothesis], list[ExperimentSpec]]:
        top_k = _positive_int(top_k, 4)
        hits = []
        for result in _items(screen_results):
            if (
                _text(getattr(result, "probe", "")).strip() != "head_ablation"
                or _text(getattr(result, "status", "")).strip() != "ran"
                or not _bool(getattr(result, "significant", False))
                or not _bool(getattr(result, "reproduced", False))
                or _head_target(getattr(result, "target", {})) is None
            ):
                continue
            hits.append(result)
        hits.sort(key=lambda r: abs(_number(getattr(r, "effect_size", 0.0))), reverse=True)
        task = _task(task_name)
        source_ids = _strings(source_ids or [])

        hypotheses: list[Hypothesis] = []
        specs: list[ExperimentSpec] = []
        for result in hits[:top_k]:
            layer, head = _head_target(getattr(result, "target", {})) or (0, 0)
            effect_size = _number(getattr(result, "effect_size", 0.0))
            polarity = "promotes" if effect_size > 0 else "suppresses"
            hyp = Hypothesis(
                id=stable_id("hyp", f"head:{self.model}:{task.name}:{layer}:{head}"),
                statement=(
                    f"In {self.model}, attention head {layer}.{head} causally {polarity} the correct "
                    f"answer for the {task.name} task (effect {effect_size:+.2f})."
                ),
                rationale=(
                    f"Screening ablation flagged head {layer}.{head} with a reproducible "
                    f"{effect_size:+.2f} effect; triangulate with independent probes."
                ),
                task=task.name,
                predicted_effect=(
                    "Direct logit attribution and activation patching agree in sign with the ablation, "
                    "and the attention pattern matches a known head type."
                ),
                target={"layer": layer, "head": head, "screen_effect": effect_size},
                source_ids=source_ids,
            )
            triangulation = self._triangulate(hyp, layer, head, task.name)
            hyp.experiment_ids = [spec.id for spec in triangulation]
            hypotheses.append(hyp)
            specs.extend(triangulation)
        return hypotheses, specs

    def _triangulate(self, hyp: Hypothesis, layer: int, head: int, task_name: str) -> list[ExperimentSpec]:
        specs: list[ExperimentSpec] = []
        task = _task(task_name)
        for probe in CONFIRMATORY_PROBES:
            target = {"layer": layer, "head": head}
            if probe == "attention_pattern":
                target["pattern"] = "induction"  # the probe reports the full profile regardless
            metric = {
                "attention_pattern": "induction_attention",
                "direct_logit_attribution": "logit_attribution",
                "activation_patching": "recovered_fraction",
            }[probe]
            specs.append(
                ExperimentSpec(
                    id=stable_id("spec", f"{probe}:{self.model}:{task_name}:{layer}:{head}"),
                    name=f"{probe} {layer}.{head}",
                    probe=probe,
                    model=self.model,
                    task=task.name,
                    target=target,
                    metric=metric,
                    controls=["random control head"],
                    seeds=self.seeds,
                    hypothesis_id=hyp.id,
                )
            )
        return specs


def update_hypotheses(
    hypotheses: list[Hypothesis],
    results_by_id: dict[str, ExperimentResult],
) -> None:
    """Mutate hypothesis status/confidence from the evidence gathered so far."""

    result_map = results_by_id if isinstance(results_by_id, dict) else {}
    for hyp_raw in _items(hypotheses):
        hyp = _hypothesis(hyp_raw)
        if hyp is None:
            continue
        results = [result_map[eid] for eid in hyp.experiment_ids if eid in result_map]
        ran = [r for r in results if _text(getattr(r, "status", "")).strip() == "ran"]
        if not ran:
            continue

        # Screening / sweep hypotheses: confirmed if any candidate showed a real effect.
        target = _mapping(getattr(hyp, "target", {}))
        if target.get("scope") == "upper_layers":
            winners = [
                r
                for r in ran
                if _bool(getattr(r, "significant", False)) and _bool(getattr(r, "reproduced", False))
            ]
            hyp.status = "confirmed" if winners else "refuted"
            hyp.confidence = round(min(0.95, 0.5 + 0.1 * len(winners)), 3) if winners else 0.1
            continue
        if target.get("probe") == "logit_lens":
            lens = ran[0]
            lens_effect = _number(getattr(lens, "effect_size", 0.0))
            hyp.status = "confirmed" if _bool(getattr(lens, "significant", False)) else "inconclusive"
            hyp.confidence = round(min(0.92, 0.4 + lens_effect * 0.5), 3)
            continue

        # Targeted single-head hypotheses: triangulation across independent probes.
        agree = [
            r
            for r in ran
            if _bool(getattr(r, "significant", False)) and _bool(getattr(r, "reproduced", False))
        ]
        n_agree = len(agree)
        if n_agree >= 2:
            hyp.status = "confirmed"
            hyp.confidence = round(min(0.97, 0.55 + 0.14 * n_agree), 3)
        elif n_agree == 1:
            hyp.status = "inconclusive"
            hyp.confidence = 0.5
        else:
            hyp.status = "refuted"
            hyp.confidence = 0.15


def classify_head_role(triangulation: list[ExperimentResult]) -> str:
    """Name a confirmed head from its attention profile and attribution sign."""

    rows = _items(triangulation)
    attn = next((r for r in rows if _text(getattr(r, "probe", "")).strip() == "attention_pattern"), None)
    dla = next((r for r in rows if _text(getattr(r, "probe", "")).strip() == "direct_logit_attribution"), None)
    if attn is not None:
        scores = attn_profile(attn)
        if scores:
            dominant = max(scores, key=scores.get)
            label = {
                "induction": "induction head",
                "previous_token": "previous-token head",
                "duplicate_token": "duplicate-token / name-mover head",
                "current_token": "current-token head",
            }.get(dominant)
            if label and scores[dominant] >= 0.4:
                return label
    if dla is not None and _number(getattr(dla, "effect_size", 0.0)) < 0:
        return "suppressor (negative) head"
    return "task-relevant attention head"


def attn_profile(result: ExperimentResult) -> dict[str, float]:
    # The attention probe stashes the full profile in its first observation extra; we
    # recover the dominant pattern from metrics where possible.
    keys = ("induction", "previous_token", "duplicate_token", "current_token")
    metrics = _mapping(getattr(result, "metrics", {}))
    return {k: _number(metrics[k]) for k in keys if k in metrics}
