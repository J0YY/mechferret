"""Runtime policy knobs for research depth and evidence gates."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_RETRIEVAL_RESULTS = 50
MAX_RETRIEVAL_RESULTS = 200
DEFAULT_NOVELTY_SOURCE_PASSES = 12
MAX_NOVELTY_SOURCE_PASSES = 64


def _positive_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw: Any = os.getenv(name)
    try:
        parsed = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        parsed = minimum
    if parsed > maximum:
        parsed = maximum
    return parsed


def retrieval_result_floor() -> int:
    return _positive_env_int(
        "MECHFERRET_RETRIEVAL_MIN_RESULTS",
        DEFAULT_RETRIEVAL_RESULTS,
        minimum=DEFAULT_RETRIEVAL_RESULTS,
        maximum=MAX_RETRIEVAL_RESULTS,
    )


def web_search_result_limit() -> int:
    floor = retrieval_result_floor()
    return _positive_env_int(
        "MECHFERRET_WEB_SEARCH_RESULTS",
        floor,
        minimum=floor,
        maximum=MAX_RETRIEVAL_RESULTS,
    )


def arxiv_search_result_limit() -> int:
    floor = retrieval_result_floor()
    return _positive_env_int(
        "MECHFERRET_ARXIV_SEARCH_RESULTS",
        floor,
        minimum=floor,
        maximum=MAX_RETRIEVAL_RESULTS,
    )


def novelty_arxiv_result_limit() -> int:
    floor = arxiv_search_result_limit()
    return _positive_env_int(
        "MECHFERRET_NOVELTY_ARXIV_RESULTS",
        floor,
        minimum=floor,
        maximum=MAX_RETRIEVAL_RESULTS,
    )


def novelty_web_result_limit() -> int:
    floor = web_search_result_limit()
    return _positive_env_int(
        "MECHFERRET_NOVELTY_WEB_RESULTS",
        floor,
        minimum=floor,
        maximum=MAX_RETRIEVAL_RESULTS,
    )


def novelty_min_arxiv_passes() -> int:
    return _positive_env_int(
        "MECHFERRET_NOVELTY_MIN_ARXIV_PASSES",
        DEFAULT_NOVELTY_SOURCE_PASSES,
        minimum=DEFAULT_NOVELTY_SOURCE_PASSES,
        maximum=MAX_NOVELTY_SOURCE_PASSES,
    )


def novelty_min_web_passes() -> int:
    return _positive_env_int(
        "MECHFERRET_NOVELTY_MIN_WEB_PASSES",
        DEFAULT_NOVELTY_SOURCE_PASSES,
        minimum=DEFAULT_NOVELTY_SOURCE_PASSES,
        maximum=MAX_NOVELTY_SOURCE_PASSES,
    )
