"""Interpretability probes.

A probe is a pure function ``probe(backend, task, target, seed) -> ProbeReading``.
It is backend-agnostic: the same code runs against the synthetic or the real
TransformerLens backend. Each reading reports a primary ``effect`` (the signed
quantity the hypothesis is about) and a matched ``control`` (the same
measurement on a site that should *not* matter), plus human-readable
observations that become evidence text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class ProbeReading:
    effect: float
    control: float
    unit: str
    observations: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _target_head(target: dict[str, Any]) -> tuple[int, int]:
    return int(target.get("layer", 0)), int(target.get("head", 0))


def probe_head_ablation(backend, task, target, seed: int) -> ProbeReading:
    layer, head = _target_head(target)
    effect = backend.head_ablation_effect(task.name, layer, head, seed)
    ctrl_layer, ctrl_head = backend.control_head(task.name, seed)
    control = backend.head_ablation_effect(task.name, ctrl_layer, ctrl_head, seed)
    direction = "hurts" if effect > 0 else ("helps" if effect < 0 else "no effect on")
    obs = [
        f"Ablating head {layer}.{head} {direction} the {task.name} metric by {effect:+.3f} "
        f"(control head {ctrl_layer}.{ctrl_head}: {control:+.3f}).",
    ]
    return ProbeReading(effect=effect, control=control, unit=task.metric, observations=obs)


def probe_activation_patching(backend, task, target, seed: int) -> ProbeReading:
    layer, head = _target_head(target)
    effect = backend.patch_recovery(task.name, layer, head, seed)
    ctrl_layer, ctrl_head = backend.control_head(task.name, seed)
    control = backend.patch_recovery(task.name, ctrl_layer, ctrl_head, seed)
    obs = [
        f"Patching clean activations at {layer}.{head} into the corrupted run recovers "
        f"{effect * 100:.1f}% of the metric (control {ctrl_layer}.{ctrl_head}: {control * 100:.1f}%).",
    ]
    return ProbeReading(effect=effect, control=control, unit="recovered_fraction", observations=obs)


def probe_attention_pattern(backend, task, target, seed: int) -> ProbeReading:
    layer, head = _target_head(target)
    pattern = (target.get("pattern") or "induction").lower()
    scores = backend.attention_score(task.name, layer, head, seed)
    effect = scores.get(pattern, 0.0)
    others = [v for k, v in scores.items() if k != pattern]
    control = round(sum(others) / max(len(others), 1), 4)
    obs = [
        f"Head {layer}.{head} {pattern}-attention score = {effect:.3f} "
        f"(mean of other patterns {control:.3f}); full profile {scores}.",
    ]
    return ProbeReading(effect=effect, control=control, unit=f"{pattern}_attention", observations=obs, extra=scores)


def probe_direct_logit_attribution(backend, task, target, seed: int) -> ProbeReading:
    layer, head = _target_head(target)
    effect = backend.direct_logit_attribution(task.name, layer, head, seed)
    ctrl_layer, ctrl_head = backend.control_head(task.name, seed)
    control = backend.direct_logit_attribution(task.name, ctrl_layer, ctrl_head, seed)
    obs = [
        f"Direct logit attribution of {layer}.{head} = {effect:+.3f} "
        f"(control {ctrl_layer}.{ctrl_head}: {control:+.3f}).",
    ]
    return ProbeReading(effect=effect, control=control, unit="logit_attribution", observations=obs)


def probe_logit_lens(backend, task, target, seed: int) -> ProbeReading:
    rows = backend.logit_lens(task.name, seed)
    decision_layer = next((row["layer"] for row in rows if row["correct_prob"] >= 0.5), rows[-1]["layer"])
    final_prob = rows[-1]["correct_prob"]
    # "effect" = how decisively the right answer is encoded by the final layer.
    effect = round(final_prob, 4)
    control = round(rows[0]["correct_prob"], 4)  # earliest layer = chance-ish
    obs = [
        f"Correct-token probability crosses 0.5 at layer {decision_layer} of {len(rows)} "
        f"and reaches {final_prob:.3f} by the final layer (layer 0 = {control:.3f}).",
    ]
    return ProbeReading(
        effect=effect,
        control=control,
        unit="final_correct_prob",
        observations=obs,
        extra={"decision_layer": decision_layer, "trajectory": rows},
    )


PROBES: dict[str, Callable[..., ProbeReading]] = {
    "head_ablation": probe_head_ablation,
    "activation_patching": probe_activation_patching,
    "attention_pattern": probe_attention_pattern,
    "direct_logit_attribution": probe_direct_logit_attribution,
    "logit_lens": probe_logit_lens,
}


def get_probe(name: str) -> Callable[..., ProbeReading]:
    key = (name or "").strip().lower()
    if key not in PROBES:
        raise KeyError(f"Unknown probe: {name!r}. Known: {sorted(PROBES)}")
    return PROBES[key]
