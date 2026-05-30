from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RegistryKind = Literal["tool", "task", "playbook", "evaluator"]


@dataclass(frozen=True, slots=True)
class RegistryItem:
    name: str
    kind: RegistryKind
    description: str
    module: str
    status: str = "available"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


TOOLS = [
    RegistryItem("load_sources", "tool", "Load local files, directories, and URLs into normalized sources.", "mechferret.sources"),
    RegistryItem("bm25_retrieval", "tool", "Retrieve source chunks with local BM25 scoring.", "mechferret.retrieval"),
    RegistryItem("provider_research", "tool", "Call configured OpenAI or Anthropic provider for research briefs.", "mechferret.llm"),
    RegistryItem("claim_extraction", "tool", "Extract atomic cited claims from evidence chunks.", "mechferret.agents"),
    RegistryItem("critic", "tool", "Score coverage, source diversity, citation density, and contradictions.", "mechferret.agents"),
    RegistryItem("report_writer", "tool", "Write HTML, Markdown, JSON, graph, eval, and trace artifacts.", "mechferret.report"),
]

TASKS = [
    RegistryItem("research_run", "task", "Single autonomous research run.", "mechferret.controller"),
    RegistryItem("goal_loop", "task", "Repeat research iterations until a target probability or budget stop.", "mechferret.goal_loop"),
    RegistryItem("doctor", "task", "Check local environment, config, optional packages, and demo readiness.", "mechferret.ops"),
    RegistryItem("memory_recall", "task", "Recall prior claims as labeled memory sources.", "mechferret.memory"),
]

PLAYBOOKS = [
    RegistryItem("neurips_main", "playbook", "Novelty, baselines, ablations, rigor, limitations, and reviewer objections.", "mechferret.evaluators"),
    RegistryItem("computational_biology", "playbook", "Mechanism, assay, dataset, reproducibility, and sample-quality checks.", "mechferret.evaluators"),
    RegistryItem("legal_research", "playbook", "Authority, jurisdiction, recency, procedural posture, and contrary precedent.", "mechferret.evaluators"),
    RegistryItem("coding_research", "playbook", "Correctness, tests, regressions, performance, API compatibility, and security.", "mechferret.evaluators"),
]

EVALUATORS = [
    RegistryItem("conference_acceptance", "evaluator", "Estimate venue acceptance probability from research evidence metrics.", "mechferret.goal_loop"),
    RegistryItem("evidence_quality", "evaluator", "Check claim count, source diversity, citation density, and contradictions.", "mechferret.report"),
    RegistryItem("gap_pressure", "evaluator", "Convert unresolved gaps into next research or experiment actions.", "mechferret.goal_loop"),
]


def all_items() -> list[RegistryItem]:
    return [*TOOLS, *TASKS, *PLAYBOOKS, *EVALUATORS]


def items_by_kind(kind: RegistryKind | str) -> list[RegistryItem]:
    return [item for item in all_items() if item.kind == kind]

