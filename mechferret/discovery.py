"""Autonomous interpretability discovery loop.

This is the control loop that makes MechFerret an *autoresearch* system for
mechanistic interpretability rather than a literature summariser. With a single
question (or a skill), and minimal human involvement, it:

1. **Grounds** the question in prior art (BM25 over a seed/interp corpus, prior
   confirmed mechanisms from memory, optional provider web search).
2. **Hypothesises** falsifiable claims about the model's internals.
3. **Experiments** -- screens candidate heads, then triangulates survivors with
   independent probes, fanning the work out across the coordinator.
4. **Critiques** for rigor (controls, significance, reproducibility,
   triangulation) and spawns the next round's experiments from the gaps.
5. **Synthesises** confirmed mechanisms into discoveries, scores their novelty
   against prior art, and writes the same auditable dossier the literature mode
   produces -- now backed by reproducible experiment specs.

It halts on the skill's stop criteria, on convergence (no new leads), or on the
:class:`~mechferret.hooks.BudgetGuard` -- never on a human's attention.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from .agents import ClaimExtractor, Critic, Planner
from .config import load_config
from .coordinator import Coordinator, default_workers
from .hooks import Budget, BudgetGuard
from .interp.critic import ExperimentCritic
from .interp.engine import InterpEngine
from .interp.hypotheses import HypothesisGenerator, classify_head_role, update_hypotheses
from .interp.tasks import get_task, infer_task
from .llm import make_research_adapter
from .memory import ResearchMemory
from .models import (
    Claim,
    Discovery,
    EvidenceChunk,
    Hypothesis,
    PlanStep,
    ResearchPlan,
    ResearchRun,
    Source,
    utc_now,
)
from .report import write_artifacts
from .retrieval import BM25Index
from .skills import Skill, load_skill
from .sources import example_corpus_path, load_sources
from .text import compact_text, cosine_overlap, stable_id
from .tracing import TraceRecorder


class DiscoveryController:
    def __init__(self, memory_path: str | Path = ".mechferret/memory.sqlite") -> None:
        self.memory_path = Path(memory_path)

    def run(
        self,
        question: str = "",
        *,
        skill: str | Skill | None = None,
        task: str | None = None,
        model: str = "gpt2",
        backend: str = "auto",
        source_paths: list[str] | None = None,
        urls: list[str] | None = None,
        out_dir: str | Path = "runs/discovery",
        budget: Budget | None = None,
        provider: str = "auto",
        llm_model: str | None = None,
        include_memory: bool = True,
    ) -> ResearchRun:
        skill_obj = self._resolve_skill(skill)
        if skill_obj:
            question = question or skill_obj.question
            task = task or skill_obj.task
            model = skill_obj.model or model
            budget = budget or skill_obj.to_budget()
        task_name = (task or infer_task(question)).lower()
        get_task(task_name)  # validate
        question = question or f"Which components of {model} implement the {task_name} behaviour?"
        budget = budget or Budget()

        run_id = f"disc_{uuid.uuid4().hex[:10]}"
        out_path = Path(out_dir)
        tracer = TraceRecorder(run_id, out_path)
        memory = ResearchMemory(self.memory_path)
        guard = BudgetGuard(budget=budget)
        engine = InterpEngine(model=model, backend=backend)
        generator = HypothesisGenerator(
            model=model, seeds=tuple(skill_obj.seeds) if skill_obj else (0, 1, 2)
        )
        exp_critic = ExperimentCritic()

        try:
            with tracer.span("prior_art", task=task_name, model=model):
                prior_sources, prior_evidence, prior_claims = self._prior_art(
                    question, source_paths, urls, provider, llm_model, memory, include_memory, tracer
                )
            source_ids = [source.id for source in prior_sources]

            resolved_backend = engine.backend_for(model, backend)
            backend_name = getattr(resolved_backend, "name", "synthetic")
            coordinator = Coordinator(max_workers=default_workers(backend_name))

            hypotheses: list[Hypothesis] = []
            results_by_spec: dict = {}
            promoted_heads: set[tuple[int, int]] = set()
            max_screen = skill_obj.max_screen_heads if skill_obj else 96
            top_k = skill_obj.promote_top_k if skill_obj else 5
            min_confirmed = skill_obj.min_confirmed if skill_obj else 1
            min_rigor = skill_obj.min_rigor if skill_obj else 0.6

            gaps: list[str] = []
            metrics: dict[str, float] = {}
            round_index = 0
            while True:
                exhausted, reason = guard.exhausted()
                if exhausted:
                    gaps.append(f"stopped on budget: {reason}")
                    break
                guard.start_round()
                with tracer.span("round", round=round_index + 1, backend=backend_name):
                    if round_index == 0:
                        new_hyps, specs = generator.screen(question, task_name, max_screen, source_ids)
                    else:
                        new_hyps, specs = generator.promote(
                            list(results_by_spec.values()), task_name, top_k, source_ids
                        )
                        # Only pursue heads we have not already targeted.
                        fresh = []
                        for hyp in new_hyps:
                            key = (hyp.target.get("layer"), hyp.target.get("head"))
                            if key in promoted_heads:
                                continue
                            promoted_heads.add(key)
                            fresh.append(hyp)
                        new_hyps = fresh
                        specs = [s for s in specs if s.hypothesis_id in {h.id for h in new_hyps}]
                        if not new_hyps:
                            gaps.append("converged: no new significant leads to triangulate")
                            break

                    hypotheses.extend(new_hyps)
                    specs = guard.admit(specs)
                    with tracer.span("experiments", count=len(specs)):
                        results = coordinator.map(engine.run_spec, specs)
                    guard.record(results)
                    for result in results:
                        results_by_spec[result.spec_id] = result
                    update_hypotheses(hypotheses, results_by_spec)
                    gaps, metrics = exp_critic.evaluate(hypotheses, list(results_by_spec.values()))
                    tracer.event(
                        "round_summary",
                        round=round_index + 1,
                        confirmed=metrics.get("confirmed_mechanisms", 0),
                        rigor=metrics.get("rigor_score", 0),
                    )

                confirmed = metrics.get("confirmed_mechanisms", 0)
                if confirmed >= min_confirmed and metrics.get("rigor_score", 0) >= min_rigor and round_index >= 1:
                    break
                round_index += 1

            with tracer.span("synthesize"):
                discoveries = self._build_discoveries(hypotheses, results_by_spec, prior_claims, model)
                evidence, claims = self._ledger(
                    hypotheses, results_by_spec, discoveries, prior_evidence, prior_claims
                )
                plan = self._plan(question, task_name, model, round_index + 1)
                lit_gaps, _lit_contradictions, lit_metrics = Critic().evaluate(question, plan, claims, evidence)
                # Don't let the literature critic's readiness clobber the discovery readiness.
                lit_metrics.pop("readiness_score", None)
                metrics.update(lit_metrics)
                metrics.update(guard.usage())
                metrics["discoveries"] = float(len(discoveries))
                metrics["mean_discovery_confidence"] = round(
                    sum(d.confidence for d in discoveries) / max(len(discoveries), 1), 3
                )
                metrics["mean_novelty"] = round(
                    sum(d.novelty for d in discoveries) / max(len(discoveries), 1), 3
                )
                metrics["readiness_score"] = self._readiness(metrics)
                answer = self._synthesize(question, model, task_name, discoveries, gaps)

                run = ResearchRun(
                    run_id=run_id,
                    question=question,
                    created_at=utc_now(),
                    plan=plan,
                    sources=prior_sources + [self._experiment_source(model, task_name)],
                    evidence=evidence,
                    claims=claims,
                    contradictions=[],
                    gaps=gaps + lit_gaps[:2],
                    answer=answer,
                    metrics=metrics,
                    hypotheses=hypotheses,
                    experiments=list(results_by_spec.values()),
                    discoveries=discoveries,
                    mode="discovery",
                )
                artifacts = write_artifacts(run, out_path)
                run.artifacts.update(artifacts)
                memory.record_run(run)
                tracer.event("artifacts_written", **artifacts)
                return run
        finally:
            memory.close()

    # --- phases ---------------------------------------------------------------------

    def _resolve_skill(self, skill: str | Skill | None) -> Skill | None:
        if skill is None:
            return None
        if isinstance(skill, Skill):
            return skill
        return load_skill(skill)

    def _prior_art(self, question, source_paths, urls, provider, llm_model, memory, include_memory, tracer):
        sources = load_sources(source_paths, urls)
        if not sources:
            sources = load_sources([str(example_corpus_path())], [])
        if include_memory:
            sources.extend(memory.recall_sources(question))

        config = load_config()
        adapter = make_research_adapter(provider, llm_model, config)
        if adapter and getattr(adapter, "available", False):
            with tracer.span("provider_research", provider=provider):
                live = adapter.search_summary(question)
                if live:
                    sources.append(live)
        memory.upsert_sources(sources)

        index = BM25Index.from_sources(sources)
        evidence = index.search(question, limit=10)
        claims = ClaimExtractor().extract(question, evidence, limit=12)
        return sources, evidence, claims

    def _build_discoveries(self, hypotheses, results_by_spec, prior_claims, model) -> list[Discovery]:
        discoveries: list[Discovery] = []
        for hyp in hypotheses:
            if hyp.status != "confirmed" or "head" not in hyp.target:
                continue
            tri = [results_by_spec[e] for e in hyp.experiment_ids if e in results_by_spec]
            role = classify_head_role(tri)
            layer = hyp.target["layer"]
            head = hyp.target["head"]
            confirming = [r for r in tri if r.significant and r.reproduced]
            effect = max((abs(r.effect_size) for r in confirming), default=0.0)
            reproducibility = round(len(confirming) / max(len(tri), 1), 3)
            statement = (
                f"In {model}, attention head {layer}.{head} is a {role} for this task: "
                f"{len(confirming)} independent probes (out of {len(tri)}) confirm a reproducible effect."
            )
            novelty = self._novelty(statement, role, prior_claims)
            discoveries.append(
                Discovery(
                    id=stable_id("disc", f"{model}:{layer}:{head}:{role}"),
                    statement=statement,
                    confidence=hyp.confidence,
                    novelty=novelty,
                    effect_size=round(effect, 3),
                    reproducibility=reproducibility,
                    supporting_experiments=[r.id for r in confirming],
                    hypothesis_id=hyp.id,
                )
            )
        discoveries.sort(key=lambda d: (d.confidence, d.effect_size), reverse=True)
        return discoveries

    def _novelty(self, statement: str, role: str, prior_claims: list[Claim]) -> float:
        if not prior_claims:
            return 0.8
        best = max((cosine_overlap(statement, claim.text) for claim in prior_claims), default=0.0)
        # Mentioning a known head *type* in the literature lowers novelty modestly.
        return round(max(0.1, min(0.95, 0.9 - best)), 3)

    def _ledger(self, hypotheses, results_by_spec, discoveries, prior_evidence, prior_claims):
        evidence: list[EvidenceChunk] = list(prior_evidence)
        claims: list[Claim] = list(prior_claims)
        exp_source_id = stable_id("src", "mechferret:experiment-log")

        # Keep the ledger readable: include triangulation/lens results + significant screen hits.
        keep = {eid for hyp in hypotheses if "head" in hyp.target for eid in hyp.experiment_ids}
        chunk_by_spec: dict[str, str] = {}
        for spec_id, result in results_by_spec.items():
            include = spec_id in keep or result.probe == "logit_lens" or (
                result.probe == "head_ablation" and result.significant and result.reproduced
            )
            if not include:
                continue
            chunk = EvidenceChunk(
                id=stable_id("ev", f"exp:{result.id}"),
                source_id=exp_source_id,
                title=f"{result.probe} @ {result.target}",
                text=result.evidence_text + " " + " ".join(result.observations[-1:]),
                url=f"experiment://{result.backend_used}/{result.probe}",
                score=round(abs(result.effect_size), 3),
            )
            evidence.append(chunk)
            chunk_by_spec[spec_id] = chunk.id

        for discovery in discoveries:
            cites = [
                chunk_by_spec[spec_id]
                for spec_id, result in results_by_spec.items()
                if result.id in discovery.supporting_experiments and spec_id in chunk_by_spec
            ]
            claim = Claim(
                id=stable_id("claim", discovery.statement.lower()),
                text=discovery.statement,
                citations=cites or [exp_source_id],
                source_ids=[exp_source_id],
                confidence=discovery.confidence,
                support_score=discovery.reproducibility,
                stance="discovery",
                quality_flags=[] if discovery.reproducibility >= 0.5 else ["weak_triangulation"],
            )
            discovery.claim_ids = [claim.id]
            claims.append(claim)
        return evidence, claims

    def _experiment_source(self, model: str, task_name: str) -> Source:
        return Source(
            id=stable_id("src", "mechferret:experiment-log"),
            title=f"MechFerret experiment log ({model}/{task_name})",
            text=f"Reproducible interpretability experiments run by MechFerret on {model} for the {task_name} task.",
            url="experiment://mechferret",
            kind="experiment_log",
        )

    def _plan(self, question: str, task_name: str, model: str, rounds: int) -> ResearchPlan:
        steps = [
            PlanStep("step_1", f"Ground the {task_name} question in prior art and prior confirmed mechanisms.", "prior_art"),
            PlanStep("step_2", f"Screen upper-layer heads of {model} by causal ablation.", "screen"),
            PlanStep("step_3", "Promote significant heads to falsifiable, single-head hypotheses.", "promote"),
            PlanStep("step_4", "Triangulate each hypothesis with independent probes (attention, DLA, patching).", "triangulate"),
            PlanStep("step_5", "Critique for rigor and synthesise confirmed mechanisms into discoveries.", "synthesize"),
        ]
        return ResearchPlan(
            question=question,
            steps=steps,
            strategy=(
                f"Hypothesis-driven causal discovery over {rounds} round(s): screen -> promote -> "
                "triangulate -> critique, halting on a rigor + confirmed-mechanism bar or budget."
            ),
        )

    def _readiness(self, metrics: dict[str, float]) -> float:
        rigor = metrics.get("rigor_score", 0.0)
        confirmed = min(metrics.get("confirmed_mechanisms", 0.0) / 2.0, 1.0)
        confidence = metrics.get("mean_discovery_confidence", 0.0)
        return round(max(0.0, min(0.99, 0.55 * rigor + 0.25 * confirmed + 0.20 * confidence)), 3)

    def _synthesize(self, question, model, task_name, discoveries: list[Discovery], gaps) -> str:
        lines = [f"Answer to: {question}", ""]
        if not discoveries:
            lines.append("No mechanism cleared the rigor bar (significant + reproducible + triangulated).")
        else:
            lines.append(f"Confirmed mechanisms in {model} for the {task_name} task:")
            for d in discoveries:
                lines.append(
                    f"- {d.statement} (confidence={d.confidence:.2f}, effect={d.effect_size:.2f}, "
                    f"reproducibility={d.reproducibility:.2f}, novelty={d.novelty:.2f})"
                )
        if gaps:
            lines.extend(["", "Open rigor gaps / next experiments:"])
            for gap in gaps[:6]:
                lines.append(f"- {gap}")
        return "\n".join(lines)
