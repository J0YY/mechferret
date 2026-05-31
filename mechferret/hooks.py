"""Budget, cost, and permission hooks for autonomous runs.

Port of Claude Code's ``hooks/toolPermission`` + cost tracking, repurposed as
the safety rail that makes a *minimally-human-in-the-loop* research agent safe
to leave running: hard ceilings on experiments, rounds, GPU-seconds, and wall
time, plus a permission gate that decides whether GPU/network tools may run.

The controller consults a :class:`BudgetGuard` before every round and records
usage after, so an autonomous loop always halts on a declared budget rather
than on a human's attention.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

# Tool permission classes, cheapest -> most expensive.
PERMISSION_CLASSES = ("local", "network", "gpu")


@dataclass(slots=True)
class Budget:
    max_experiments: int = 400
    max_rounds: int = 4
    max_gpu_seconds: float = 900.0
    max_wall_seconds: float = 1800.0
    allow_gpu: bool = True
    allow_network: bool = True


def _positive_int(value: Any, default: int) -> int:
    if type(value) is bool:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any) -> int:
    if type(value) is bool:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _positive_float(value: Any, default: float) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def _nonnegative_float(value: Any) -> float:
    if type(value) is bool:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, parsed) if math.isfinite(parsed) else 0.0


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


@dataclass(slots=True)
class BudgetGuard:
    budget: Budget = field(default_factory=Budget)
    experiments_run: int = 0
    gpu_seconds: float = 0.0
    rounds_run: int = 0
    notices: list[str] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        allow_gpu = getattr(self.budget, "allow_gpu", True)
        allow_network = getattr(self.budget, "allow_network", True)
        self.budget = Budget(
            max_experiments=_positive_int(getattr(self.budget, "max_experiments", 400), 400),
            max_rounds=_positive_int(getattr(self.budget, "max_rounds", 4), 4),
            max_gpu_seconds=_positive_float(getattr(self.budget, "max_gpu_seconds", 900.0), 900.0),
            max_wall_seconds=_positive_float(getattr(self.budget, "max_wall_seconds", 1800.0), 1800.0),
            allow_gpu=allow_gpu if type(allow_gpu) is bool else True,
            allow_network=allow_network if type(allow_network) is bool else True,
        )
        self.experiments_run = _nonnegative_int(self.experiments_run)
        self.gpu_seconds = _nonnegative_float(self.gpu_seconds)
        self.rounds_run = _nonnegative_int(self.rounds_run)
        self.notices = _items(self.notices)

    def remaining_experiments(self) -> int:
        return max(0, self.budget.max_experiments - self.experiments_run)

    def wall_seconds(self) -> float:
        return time.perf_counter() - self._start

    def admit(self, specs: list) -> list:
        """Truncate a planned batch to what the budget still allows."""

        specs = _items(specs)
        remaining = self.remaining_experiments()
        if len(specs) <= remaining:
            return specs
        self.notices.append(
            f"budget cap: ran {remaining} of {len(specs)} planned experiments this round "
            f"(max_experiments={self.budget.max_experiments})"
        )
        return specs[:remaining]

    def record(self, results: Iterable) -> None:
        results = _items(results)
        self.experiments_run += len(results)
        self.gpu_seconds += sum(_nonnegative_float(getattr(r, "gpu_seconds", 0.0)) for r in results)

    def start_round(self) -> None:
        self.rounds_run += 1

    def exhausted(self) -> tuple[bool, str]:
        if self.rounds_run >= self.budget.max_rounds:
            return True, f"max_rounds={self.budget.max_rounds} reached"
        if self.remaining_experiments() <= 0:
            return True, f"max_experiments={self.budget.max_experiments} reached"
        if self.gpu_seconds >= self.budget.max_gpu_seconds:
            return True, f"max_gpu_seconds={self.budget.max_gpu_seconds} reached"
        if self.wall_seconds() >= self.budget.max_wall_seconds:
            return True, f"max_wall_seconds={self.budget.max_wall_seconds} reached"
        return False, ""

    def permits(self, permission_class: str) -> bool:
        if permission_class == "gpu":
            return self.budget.allow_gpu
        if permission_class == "network":
            return self.budget.allow_network
        return True

    def usage(self) -> dict[str, float]:
        return {
            "experiments_run": float(self.experiments_run),
            "rounds_run": float(self.rounds_run),
            "gpu_seconds": round(self.gpu_seconds, 3),
            "wall_seconds": round(self.wall_seconds(), 3),
            "experiment_budget": float(self.budget.max_experiments),
        }
