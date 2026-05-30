"""Deterministic synthetic interpretability backend.

The synthetic backend lets the whole autonomous loop run offline -- no torch,
no GPU, no network, no API keys -- while still producing *structured,
reproducible, literature-plausible* experiment outcomes.

It does this by fabricating a hidden "ground-truth circuit" for each
(model, task) pair: a small set of important heads/layers with stable
magnitudes, plus seed-dependent noise. Probes then read this ground truth, so
ablating a genuine name-mover head reliably hurts the metric (and reproduces
across seeds) while a control head does not. That is exactly the signal the
experiment critic needs to confirm or refute a hypothesis.

The numbers are simulated, not measured -- every artifact records
``backend_used="synthetic"`` so this is never mistaken for a real result. Swap
in :class:`~mechferret.interp.backends.TransformerLensBackend` to measure a
real model with the identical probe + engine code.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from .tasks import Task, get_task

# Standard small interpretability models and their shapes.
MODEL_SHAPES: dict[str, tuple[int, int, int]] = {
    "gpt2": (12, 12, 768),
    "gpt2-small": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
    "pythia-160m": (12, 12, 768),
    "pythia-410m": (24, 16, 1024),
    "synthetic-12L": (12, 12, 768),
}


def _seeded_rng(*parts: object) -> random.Random:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _shape(model: str) -> tuple[int, int, int]:
    return MODEL_SHAPES.get((model or "gpt2").lower(), (12, 12, 768))


@dataclass(slots=True)
class GroundTruthHead:
    layer: int
    head: int
    role: str
    magnitude: float  # contribution to the task metric (can be negative)


@dataclass(slots=True)
class GroundTruthCircuit:
    model: str
    task: str
    n_layers: int
    n_heads: int
    d_model: int
    full_metric: float
    decision_layer: int
    heads: list[GroundTruthHead] = field(default_factory=list)

    def head(self, layer: int, head: int) -> GroundTruthHead | None:
        for candidate in self.heads:
            if candidate.layer == layer and candidate.head == head:
                return candidate
        return None


def build_circuit(model: str, task_name: str) -> GroundTruthCircuit:
    """Deterministically fabricate a plausible circuit for (model, task)."""

    task = get_task(task_name)
    n_layers, n_heads, d_model = _shape(model)
    rng = _seeded_rng("circuit", model, task.name)

    # The decision usually crystallises in the upper-middle of the network.
    decision_layer = int(n_layers * rng.uniform(0.55, 0.82))

    roles = list(task.expected_components) or ["primary_head"]
    n_key = min(len(roles) + rng.randint(0, 1), max(2, n_layers // 3))
    chosen: set[tuple[int, int]] = set()
    heads: list[GroundTruthHead] = []
    for index in range(n_key):
        role = roles[index % len(roles)]
        # Name-mover / induction heads cluster a bit before the decision layer.
        layer = max(0, min(n_layers - 1, decision_layer - rng.randint(0, 3)))
        head = rng.randrange(n_heads)
        while (layer, head) in chosen:
            head = rng.randrange(n_heads)
            layer = max(0, min(n_layers - 1, layer + rng.choice((-1, 0, 1))))
            if (layer, head) not in chosen:
                break
        chosen.add((layer, head))
        magnitude = rng.uniform(1.3, 3.1)
        # Roughly one negative (suppressor) head per circuit, à la negative name movers.
        if "inhibition" in role or (index == n_key - 1 and rng.random() < 0.45):
            magnitude = -rng.uniform(0.5, 1.2)
        heads.append(GroundTruthHead(layer=layer, head=head, role=role, magnitude=round(magnitude, 3)))

    positive = sum(h.magnitude for h in heads if h.magnitude > 0)
    full_metric = round(positive + rng.uniform(0.6, 1.6), 3)
    return GroundTruthCircuit(
        model=model,
        task=task.name,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        full_metric=full_metric,
        decision_layer=decision_layer,
        heads=heads,
    )


class SyntheticBackend:
    """Reads a fabricated ground-truth circuit to answer probe queries."""

    name = "synthetic"
    available = True

    def __init__(self, model: str = "gpt2") -> None:
        self.model = (model or "gpt2").lower()
        self.n_layers, self.n_heads, self.d_model = _shape(self.model)
        self._circuits: dict[str, GroundTruthCircuit] = {}

    def circuit(self, task_name: str) -> GroundTruthCircuit:
        if task_name not in self._circuits:
            self._circuits[task_name] = build_circuit(self.model, task_name)
        return self._circuits[task_name]

    # --- probe-facing measurements -------------------------------------------------

    def clean_metric(self, task_name: str, seed: int) -> float:
        circuit = self.circuit(task_name)
        rng = _seeded_rng("clean", self.model, task_name, seed)
        return round(circuit.full_metric * rng.uniform(0.97, 1.03), 4)

    def head_ablation_effect(self, task_name: str, layer: int, head: int, seed: int) -> float:
        """Drop in the task metric caused by ablating this head (positive = important)."""

        circuit = self.circuit(task_name)
        truth = circuit.head(layer, head)
        rng = _seeded_rng("ablate", self.model, task_name, layer, head, seed)
        if truth is None:
            return round(rng.gauss(0.0, 0.05), 4)
        noise = rng.gauss(0.0, abs(truth.magnitude) * 0.06 + 0.02)
        return round(truth.magnitude + noise, 4)

    def patch_recovery(self, task_name: str, layer: int, head: int, seed: int) -> float:
        """Fraction of the clean-run metric recovered by patching this site."""

        circuit = self.circuit(task_name)
        truth = circuit.head(layer, head)
        rng = _seeded_rng("patch", self.model, task_name, layer, head, seed)
        if truth is None:
            return round(abs(rng.gauss(0.0, 0.03)), 4)
        share = max(0.0, truth.magnitude) / max(circuit.full_metric, 1e-6)
        share = min(0.95, share + rng.gauss(0.0, 0.04))
        return round(max(0.0, share), 4)

    def attention_score(self, task_name: str, layer: int, head: int, seed: int) -> dict[str, float]:
        circuit = self.circuit(task_name)
        truth = circuit.head(layer, head)
        rng = _seeded_rng("attn", self.model, task_name, layer, head, seed)
        scores = {
            "induction": abs(rng.gauss(0.0, 0.05)),
            "previous_token": abs(rng.gauss(0.0, 0.05)),
            "duplicate_token": abs(rng.gauss(0.0, 0.05)),
            "current_token": abs(rng.gauss(0.0, 0.08)),
        }
        if truth is not None:
            role = truth.role
            if "induction" in role:
                scores["induction"] = min(0.97, rng.uniform(0.62, 0.9))
            elif "previous" in role:
                scores["previous_token"] = min(0.97, rng.uniform(0.6, 0.88))
            elif "duplicate" in role:
                scores["duplicate_token"] = min(0.97, rng.uniform(0.55, 0.85))
            elif "name_mover" in role:
                scores["duplicate_token"] = min(0.9, rng.uniform(0.4, 0.7))
        return {key: round(value, 4) for key, value in scores.items()}

    def direct_logit_attribution(self, task_name: str, layer: int, head: int, seed: int) -> float:
        circuit = self.circuit(task_name)
        truth = circuit.head(layer, head)
        rng = _seeded_rng("dla", self.model, task_name, layer, head, seed)
        if truth is None:
            return round(rng.gauss(0.0, 0.04), 4)
        return round(truth.magnitude * rng.uniform(0.85, 1.05), 4)

    def logit_lens(self, task_name: str, seed: int) -> list[dict[str, float]]:
        circuit = self.circuit(task_name)
        rng = _seeded_rng("lens", self.model, task_name, seed)
        rows: list[dict[str, float]] = []
        for layer in range(circuit.n_layers):
            # Logistic rise of the correct-token probability, crossing 0.5 at the decision layer.
            x = (layer - circuit.decision_layer) / max(1.0, circuit.n_layers / 6.0)
            prob = 1.0 / (1.0 + math.exp(-x))
            prob = max(0.0, min(0.999, prob + rng.gauss(0.0, 0.02)))
            logit_diff = round((prob - 0.5) * 2.0 * circuit.full_metric, 4)
            rows.append({"layer": layer, "correct_prob": round(prob, 4), "logit_diff": logit_diff})
        return rows

    def top_heads(self, task_name: str) -> list[GroundTruthHead]:
        return sorted(self.circuit(task_name).heads, key=lambda h: abs(h.magnitude), reverse=True)

    def control_head(self, task_name: str, seed: int) -> tuple[int, int]:
        """A head that is *not* part of the circuit -- the negative control."""

        circuit = self.circuit(task_name)
        key = {(h.layer, h.head) for h in circuit.heads}
        rng = _seeded_rng("control", self.model, task_name, seed)
        for _ in range(64):
            layer = rng.randrange(self.n_layers)
            head = rng.randrange(self.n_heads)
            if (layer, head) not in key:
                return layer, head
        return 0, 0
