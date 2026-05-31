"""Parallel sub-agent coordinator (an "agent swarm").

Port of Claude Code's ``coordinator/`` + ``AgentTool`` fan-out, adapted to the
research loop. The coordinator runs many independent units of work -- screening
ablations, triangulation probes, parallel hypotheses -- concurrently while
preserving input order in the returned results, so the rest of the pipeline
gets stable result ordering.

For the offline synthetic backend the work is pure CPU and inexpensive, so the
local path stays serial. The parallelism pays off for the network/GPU backends
(TransformerLens, Modal), where each unit spends most of its time waiting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _positive_int(value: Any, default: int = 1) -> int:
    if type(value) is bool:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


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


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


class Coordinator:
    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = _positive_int(max_workers)

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        if not callable(fn):
            return []
        materialised = _items(items)
        if not materialised:
            return []
        if self.max_workers == 1 or len(materialised) == 1:
            return [fn(item) for item in materialised]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(materialised))) as pool:
            # executor.map preserves input ordering for downstream artifact diffs.
            return list(pool.map(fn, materialised))


def default_workers(backend: str) -> int:
    """Network/GPU backends benefit from fan-out; local fallback stays serial."""

    return 1 if (_text(backend).strip().lower() or "synthetic") in {"synthetic", "auto"} else 8
