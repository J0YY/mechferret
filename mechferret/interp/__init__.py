"""Mechanistic interpretability experiment engine.

This package turns interpretability research questions into auditable
experiments. It runs locally by default through ``SyntheticBackend`` (no torch,
no GPU, no network) and upgrades transparently to a real
``TransformerLensBackend`` (locally or on Modal GPUs) when the optional
dependencies are installed.

Public surface:

- :class:`InterpEngine` -- run :class:`ExperimentSpec` objects into
  :class:`ExperimentResult` objects with effect sizes, baselines/controls,
  significance, and cross-seed reproducibility.
- :func:`get_task` / :data:`TASKS` -- canonical interpretability tasks
  (IOI, induction, greater-than, factual recall).
- :data:`PROBES` -- the registered probe implementations.
"""

from __future__ import annotations

from .engine import InterpEngine
from .probes import PROBES
from .tasks import TASKS, get_task

__all__ = ["InterpEngine", "PROBES", "TASKS", "get_task"]
