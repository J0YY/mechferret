"""Parallel sub-agent coordinator (an "agent swarm").

Port of Claude Code's ``coordinator/`` + ``AgentTool`` fan-out, adapted to the
research loop. The coordinator runs many independent units of work -- screening
ablations, triangulation probes, parallel hypotheses -- concurrently while
preserving input order in the returned results, so the rest of the pipeline
stays deterministic.

For the offline synthetic backend the work is pure CPU and order-stable, so the
results are identical whether run serially or across threads. The parallelism
pays off for the network/GPU backends (TransformerLens, Modal), where each unit
spends most of its time waiting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class Coordinator:
    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max(1, int(max_workers))

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        materialised = list(items)
        if not materialised:
            return []
        if self.max_workers == 1 or len(materialised) == 1:
            return [fn(item) for item in materialised]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(materialised))) as pool:
            # executor.map preserves input ordering, keeping results deterministic.
            return list(pool.map(fn, materialised))


def default_workers(backend: str) -> int:
    """Network/GPU backends benefit from fan-out; synthetic stays serial + deterministic."""

    return 1 if (backend or "synthetic").lower() in {"synthetic", "auto"} else 8
