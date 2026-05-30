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

from .hooks import Budget

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    task: str
    model: str = "gpt2"
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
            max_experiments=int(self.budget.get("max_experiments", defaults.max_experiments)),
            max_rounds=int(self.budget.get("max_rounds", defaults.max_rounds)),
            max_gpu_seconds=float(self.budget.get("max_gpu_seconds", defaults.max_gpu_seconds)),
            max_wall_seconds=float(self.budget.get("max_wall_seconds", defaults.max_wall_seconds)),
            allow_gpu=bool(self.budget.get("allow_gpu", defaults.allow_gpu)),
            allow_network=bool(self.budget.get("allow_network", defaults.allow_network)),
        )

    @property
    def min_confirmed(self) -> int:
        return int(self.stop.get("min_confirmed_mechanisms", 1))

    @property
    def min_rigor(self) -> float:
        return float(self.stop.get("min_rigor_score", 0.6))


def _from_payload(payload: dict[str, Any]) -> Skill:
    return Skill(
        name=payload["name"],
        description=payload.get("description", ""),
        task=payload["task"],
        model=payload.get("model", "gpt2"),
        question=payload.get("question", ""),
        max_screen_heads=int(payload.get("max_screen_heads", 96)),
        promote_top_k=int(payload.get("promote_top_k", 5)),
        seeds=list(payload.get("seeds", [0, 1, 2])),
        budget=payload.get("budget", {}),
        stop=payload.get("stop", {}),
        references=list(payload.get("references", [])),
    )


def load_skill(name_or_path: str) -> Skill:
    candidate = Path(name_or_path)
    if candidate.suffix == ".json" and candidate.exists():
        return _from_payload(json.loads(candidate.read_text(encoding="utf-8")))
    slug = name_or_path.strip().lower().replace("_", "-")
    path = SKILLS_DIR / f"{slug}.json"
    if not path.exists():
        raise KeyError(f"Unknown skill: {name_or_path!r}. Known: {[s.name for s in list_skills()]}")
    return _from_payload(json.loads(path.read_text(encoding="utf-8")))


def list_skills() -> list[Skill]:
    if not SKILLS_DIR.exists():
        return []
    skills = []
    for path in sorted(SKILLS_DIR.glob("*.json")):
        try:
            skills.append(_from_payload(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, KeyError):
            continue
    return skills
