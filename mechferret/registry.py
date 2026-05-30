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
    permission: str = "local"  # local | network | gpu
    cost: str = "free"  # free | api | compute

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


TOOLS = [
    RegistryItem("load_sources", "tool", "Load local files, directories, and URLs into normalized sources.", "mechferret.sources"),
    RegistryItem("bm25_retrieval", "tool", "Retrieve source chunks with local BM25 scoring.", "mechferret.retrieval"),
    RegistryItem("provider_research", "tool", "Call configured OpenAI or Anthropic provider for research briefs.", "mechferret.llm", permission="network", cost="api"),
    RegistryItem("claim_extraction", "tool", "Extract atomic cited claims from evidence chunks.", "mechferret.agents"),
    RegistryItem("critic", "tool", "Score coverage, source diversity, citation density, and contradictions.", "mechferret.agents"),
    RegistryItem("report_writer", "tool", "Write HTML, Markdown, JSON, graph, eval, and trace artifacts.", "mechferret.report"),
    # Interpretability probes -- the experimental tools.
    RegistryItem("head_ablation", "tool", "Ablate an attention head and measure the change in the task metric.", "mechferret.interp.probes", permission="gpu", cost="compute"),
    RegistryItem("activation_patching", "tool", "Patch clean activations into a corrupted run (causal tracing) to localise a behaviour.", "mechferret.interp.probes", permission="gpu", cost="compute"),
    RegistryItem("attention_pattern", "tool", "Classify a head by its attention pattern (induction, previous-token, duplicate-token).", "mechferret.interp.probes", permission="gpu", cost="compute"),
    RegistryItem("direct_logit_attribution", "tool", "Attribute the correct-token logit to a specific component.", "mechferret.interp.probes", permission="gpu", cost="compute"),
    RegistryItem("logit_lens", "tool", "Project the residual stream to vocab at each layer to find the decision layer.", "mechferret.interp.probes", permission="gpu", cost="compute"),
    RegistryItem("modal_dispatch", "tool", "Run interpretability experiments on a Modal GPU function.", "mechferret.modal_app", permission="gpu", cost="compute"),
]

TASKS = [
    RegistryItem("research_run", "task", "Single autonomous literature research run.", "mechferret.controller"),
    RegistryItem("discovery_loop", "task", "Autonomous hypothesize/experiment/critique loop that returns reproducible mechanisms.", "mechferret.discovery"),
    RegistryItem("goal_loop", "task", "Repeat research iterations until a target probability or budget stop.", "mechferret.goal_loop"),
    RegistryItem("doctor", "task", "Check local environment, config, optional packages, and demo readiness.", "mechferret.ops"),
    RegistryItem("memory_recall", "task", "Recall prior claims and confirmed mechanisms as labeled memory sources.", "mechferret.memory"),
]

PLAYBOOKS = [
    RegistryItem("find_induction_heads", "playbook", "Screen + triangulate to locate induction heads on the repeated-sequence task.", "mechferret.skills", permission="gpu", cost="compute"),
    RegistryItem("ioi_circuit", "playbook", "Recover the Indirect Object Identification circuit (name movers, S-inhibition).", "mechferret.skills", permission="gpu", cost="compute"),
    RegistryItem("logit_lens_sweep", "playbook", "Find the layer at which a behaviour's prediction crystallises.", "mechferret.skills", permission="gpu", cost="compute"),
    RegistryItem("factual_recall_trace", "playbook", "Causal-trace where a factual association is stored (ROME-style).", "mechferret.skills", permission="gpu", cost="compute"),
    RegistryItem("neurips_main", "playbook", "Novelty, baselines, ablations, rigor, limitations, and reviewer objections.", "mechferret.evaluators"),
]

EVALUATORS = [
    RegistryItem("experiment_rigor", "evaluator", "Score controls, significance, reproducibility, and triangulation of experiments.", "mechferret.interp.critic", permission="local"),
    RegistryItem("conference_acceptance", "evaluator", "Estimate venue acceptance probability from research evidence metrics.", "mechferret.goal_loop"),
    RegistryItem("evidence_quality", "evaluator", "Check claim count, source diversity, citation density, and contradictions.", "mechferret.report"),
    RegistryItem("gap_pressure", "evaluator", "Convert unresolved gaps into next research or experiment actions.", "mechferret.goal_loop"),
]


def all_items() -> list[RegistryItem]:
    return [*TOOLS, *TASKS, *PLAYBOOKS, *EVALUATORS]


def items_by_kind(kind: RegistryKind | str) -> list[RegistryItem]:
    return [item for item in all_items() if item.kind == kind]

