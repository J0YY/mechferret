"""Skill system: declarative interpretability playbooks.

Port of Claude Code's ``skills/`` + ``SkillTool``. A skill is a reusable,
shareable recipe that configures one autonomous discovery run -- which task,
which model, how wide to screen, how hard to triangulate, the compute budget,
and the bar that counts as "done". Skills live as JSON next to this module so
they can be versioned, diffed, and contributed without touching code.

The :class:`~mechferret.discovery.DiscoveryController` loads a skill to drive a
run; ``/skills`` lists them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_INTERP_MODEL
from .hooks import Budget

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    task: str
    model: str = DEFAULT_INTERP_MODEL
    question: str = ""
    max_screen_heads: int = 96
    promote_top_k: int = 5
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    budget: dict[str, Any] = field(default_factory=dict)
    stop: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)

    def to_budget(self) -> Budget:
        defaults = Budget()
        return Budget(
            max_experiments=_int_field(self.budget, "max_experiments", defaults.max_experiments, min_value=1),
            max_rounds=_int_field(self.budget, "max_rounds", defaults.max_rounds, min_value=1),
            max_gpu_seconds=_float_field(self.budget, "max_gpu_seconds", defaults.max_gpu_seconds, min_value=0.0),
            max_wall_seconds=_float_field(self.budget, "max_wall_seconds", defaults.max_wall_seconds, min_value=0.0),
            allow_gpu=_bool_field(self.budget, "allow_gpu", defaults.allow_gpu),
            allow_network=_bool_field(self.budget, "allow_network", defaults.allow_network),
        )

    @property
    def min_confirmed(self) -> int:
        return _int_field(self.stop, "min_confirmed_mechanisms", 1, min_value=0)

    @property
    def min_rigor(self) -> float:
        return _float_field(self.stop, "min_rigor_score", 0.6, min_value=0.0)


def _string(value: Any, default: str = "") -> str:
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _int(value: Any, default: int, *, min_value: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= min_value else default


def _float(value: Any, default: float, *, min_value: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= min_value else default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _int_list(value: Any, default: list[int]) -> list[int]:
    if not isinstance(value, list):
        return list(default)
    parsed = [_int(item, -1, min_value=0) for item in value]
    seeds = [item for item in parsed if item >= 0]
    return seeds or list(default)


def _int_field(payload: dict[str, Any], key: str, default: int, *, min_value: int = 0) -> int:
    return _int(_dict(payload).get(key, default), default, min_value=min_value)


def _float_field(payload: dict[str, Any], key: str, default: float, *, min_value: float = 0.0) -> float:
    return _float(_dict(payload).get(key, default), default, min_value=min_value)


def _bool_field(payload: dict[str, Any], key: str, default: bool) -> bool:
    return _bool(_dict(payload).get(key, default), default)


def _from_payload(payload: dict[str, Any]) -> Skill:
    if not isinstance(payload, dict):
        raise ValueError("skill JSON must be an object")
    name = _string(payload.get("name"))
    task = _string(payload.get("task"))
    if not name or not task:
        raise ValueError("skill requires non-empty name and task")
    return Skill(
        name=name,
        description=_string(payload.get("description")),
        task=task,
        model=_string(payload.get("model"), DEFAULT_INTERP_MODEL),
        question=_string(payload.get("question")),
        max_screen_heads=_int(payload.get("max_screen_heads", 96), 96, min_value=1),
        promote_top_k=_int(payload.get("promote_top_k", 5), 5, min_value=1),
        seeds=_int_list(payload.get("seeds", [0, 1, 2]), [0, 1, 2]),
        budget=_dict(payload.get("budget")),
        stop=_dict(payload.get("stop")),
        references=_str_list(payload.get("references", [])),
    )


def load_skill(name_or_path: str) -> Skill:
    name_or_path = _string(name_or_path)
    candidate = Path(name_or_path)
    if candidate.suffix == ".json" and candidate.exists():
        try:
            return _from_payload(json.loads(candidate.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            raise ValueError(f"invalid skill {candidate}: {exc}") from exc
    slug = name_or_path.lower().replace("_", "-")
    path = SKILLS_DIR / f"{slug}.json"
    if not path.exists():
        raise KeyError(f"Unknown skill: {name_or_path!r}. Known: {[s.name for s in list_skills()]}")
    try:
        return _from_payload(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ValueError(f"invalid skill {path}: {exc}") from exc


def list_skills() -> list[Skill]:
    if not SKILLS_DIR.exists():
        return []
    skills = []
    for path in sorted(SKILLS_DIR.glob("*.json")):
        try:
            skills.append(_from_payload(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return skills
