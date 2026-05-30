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

import time
from dataclasses import dataclass, field
from typing import Iterable

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


@dataclass(slots=True)
class BudgetGuard:
    budget: Budget = field(default_factory=Budget)
    experiments_run: int = 0
    gpu_seconds: float = 0.0
    rounds_run: int = 0
    notices: list[str] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    def remaining_experiments(self) -> int:
        return max(0, self.budget.max_experiments - self.experiments_run)

    def wall_seconds(self) -> float:
        return time.perf_counter() - self._start

    def admit(self, specs: list) -> list:
        """Truncate a planned batch to what the budget still allows."""

        remaining = self.remaining_experiments()
        if len(specs) <= remaining:
            return specs
        self.notices.append(
            f"budget cap: ran {remaining} of {len(specs)} planned experiments this round "
            f"(max_experiments={self.budget.max_experiments})"
        )
        return specs[:remaining]

    def record(self, results: Iterable) -> None:
        results = list(results)
        self.experiments_run += len(results)
        self.gpu_seconds += sum(getattr(r, "gpu_seconds", 0.0) for r in results)

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
