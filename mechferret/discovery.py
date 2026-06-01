"""Autonomous interpretability discovery loop.

This is the control loop that makes MechFerret an *autoresearch* system for
mechanistic interpretability rather than a literature summariser. With a single
question (or a skill), and minimal human involvement, it:

1. **Grounds** the question in prior art (BM25 over explicit source material,
   prior confirmed mechanisms from memory, optional provider web search, or an
   explicitly requested demo seed corpus).
2. **Hypothesises** falsifiable claims about the model's internals.
3. **Experiments** -- screens candidate heads, then triangulates survivors with
   independent probes, fanning the work out across the coordinator.
4. **Critiques** for rigor (controls, significance, reproducibility,
   triangulation) and spawns the next round's experiments from the gaps.
5. **Synthesises** confirmed mechanisms into discoveries, scores their novelty
   against prior art, and writes the same auditable dossier the literature mode
   produces -- now backed by auditable experiment specs.

It halts on the skill's stop criteria, on convergence (no new leads), or on the
:class:`~mechferret.hooks.BudgetGuard` -- never on a human's attention.
"""

from __future__ import annotations

import uuid
import math
from pathlib import Path
from typing import Any

from .agents import ClaimExtractor, Critic, Planner
from .config import load_config
from .coordinator import Coordinator, default_workers
from .hooks import Budget, BudgetGuard
from .interp.critic import ExperimentCritic
from .interp.engine import InterpEngine
from .interp.hypotheses import HypothesisGenerator, classify_head_role, update_hypotheses
from .interp.tasks import get_task, infer_task
from .llm import make_research_adapter, synthesize_answer_with_provider
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
from .provenance import refresh_run_manifest
from .report import write_artifacts
from .retrieval import BM25Index
from .skills import Skill, load_skill
from .sources import example_corpus_path, load_sources
from .text import compact_text, cosine_overlap, stable_id
from .tracing import TraceRecorder


TASK_KEYWORDS = {
    "ioi": ("ioi", "indirect object", "name mover", "name-mover", "duplicate token", "s-inhibition"),
    "induction": ("induction", "in-context", "copy", "repeated sequence", "previous token"),
    "greater_than": ("greater", "greater-than", "year", "comparison", "numeric"),
    "factual_recall": ("fact", "factual", "recall", "knowledge", "rome", "association"),
}

UNSUPPORTED_REQUEST_SIGNALS = {
    "sae": "SAE work needs an activation-cache/SAELens workflow, not the current head-circuit discovery skills",
    "saes": "SAE work needs an activation-cache/SAELens workflow, not the current head-circuit discovery skills",
    "sparse autoencoder": "SAE work needs an activation-cache/SAELens workflow, not the current head-circuit discovery skills",
    "openvla": "OpenVLA needs a VLA/vision-action backend, not the current text-task discovery skills",
    "vision-language-action": "VLA models need a vision-action backend, not the current text-task discovery skills",
    "vla": "VLA models need a vision-action backend, not the current text-task discovery skills",
    "robot": "robot/VLA work needs a vision-action backend, not the current text-task discovery skills",
}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


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


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    return [_text(item).strip() for item in _items(value) if _text(item).strip()]


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _flag(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if type(value) in {int, float}:
        return bool(value)
    return False


def _label(value: Any) -> str:
    if type(value) is bool:
        return ""
    if isinstance(value, (int, float)):
        return str(value) if math.isfinite(float(value)) else ""
    return _text(value).strip()


def _path_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes, Path)):
        text = _text(value) if not isinstance(value, Path) else str(value)
        return [text.strip()] if text.strip() else []
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _text(item) if not isinstance(item, Path) else str(item)
        text = text.strip()
        if text:
            items.append(text)
    return items


def _positive_int(value: Any, default: int, *, upper: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        parsed = default
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def _positive_float(value: Any, default: float, *, upper: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        parsed = default
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def _nonnegative_int(value: Any, default: int, *, upper: int | None = None) -> int:
    if type(value) is bool:
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
    if parsed < 0:
        parsed = default
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def _nonnegative_float(value: Any, default: float, *, upper: float | None = None) -> float:
    if type(value) is bool:
        parsed = default
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
    if not math.isfinite(parsed) or parsed < 0:
        parsed = default
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def _bool(value: Any, default: bool) -> bool:
    return value if type(value) is bool else default


def _budget(value: Any) -> Budget:
    if not isinstance(value, Budget):
        return Budget()
    return Budget(
        max_experiments=_positive_int(value.max_experiments, 400, upper=100_000),
        max_rounds=_positive_int(value.max_rounds, 4, upper=1_000),
        max_gpu_seconds=_positive_float(value.max_gpu_seconds, 900.0, upper=1_000_000.0),
        max_wall_seconds=_positive_float(value.max_wall_seconds, 1800.0, upper=1_000_000.0),
        allow_gpu=_bool(value.allow_gpu, True),
        allow_network=_bool(value.allow_network, True),
    )


def request_alignment_issue(
    question: str,
    skill: Skill | None,
    task_name: str,
    model: str,
    explicit_task: bool = False,
) -> str:
    """Return an actionable issue if discovery would run an unrelated task."""

    text = (question or "").lower()
    if not text.strip():
        return ""
    unsupported_hits: list[tuple[str, str]] = []
    for signal, reason in sorted(UNSUPPORTED_REQUEST_SIGNALS.items(), key=lambda item: len(item[0]), reverse=True):
        if signal not in text:
            continue
        if any(signal in prior for prior, _reason in unsupported_hits):
            continue
        unsupported_hits.append((signal, reason))
    unsupported = [reason for _signal, reason in unsupported_hits]
    if unsupported:
        signals = ", ".join(signal for signal, _reason in unsupported_hits[:4])
        return (
            f"Discovery request is not aligned with the available {model}/{task_name} workflow: "
            f"matched unsupported term(s): {signals}. {unsupported[0]}. "
            "Use `mechferret sae openvla plan`, use literature mode (`mechferret run ...`) for planning, add a matching skill/backend, "
            "or pass --allow-mismatch if you intentionally want the demo task."
        )
    matched_tasks = [
        name for name, keywords in TASK_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    if matched_tasks and task_name and task_name not in matched_tasks:
        return (
            f"Discovery task mismatch: the question looks like {matched_tasks[0]!r}, but the run would use "
            f"{task_name!r}. Pick a matching skill/task or pass --allow-mismatch."
        )
    if skill and matched_tasks and skill.task not in matched_tasks:
        return (
            f"Skill mismatch: skill {skill.name!r} is for {skill.task!r}, but the question looks like "
            f"{matched_tasks[0]!r}. Pick a matching skill or pass --allow-mismatch."
        )
    if not explicit_task and not skill and not matched_tasks:
        return (
            "Discovery could not infer a supported interpretability task from the question. "
            "Choose --task ioi|induction|greater_than|factual_recall, use --skill, or run literature mode first."
        )
    return ""


class DiscoveryController:
    def __init__(self, memory_path: str | Path = ".mechferret/memory.sqlite") -> None:
        self.memory_path = Path(memory_path)

    def run(
        self,
        question: str = "",
        *,
        skill: str | Skill | None = None,
        task: str | None = None,
        model: str | None = None,
        backend: str = "auto",
        source_paths: list[str] | None = None,
        urls: list[str] | None = None,
        out_dir: str | Path = "runs/discovery",
        budget: Budget | None = None,
        provider: str = "auto",
        llm_model: str | None = None,
        include_memory: bool = True,
        allow_mismatch: bool = False,
        allow_seed_corpus: bool = False,
    ) -> ResearchRun:
        question = _text(question).strip()
        task = _text(task).strip() or None
        model = _text(model).strip()
        backend = _text(backend).strip() or "auto"
        source_paths = _path_list(source_paths)
        urls = _path_list(urls)
        provider = _text(provider).strip() or "auto"
        llm_model = _text(llm_model).strip() or None
        include_memory = _bool(include_memory, True)
        allow_mismatch = _bool(allow_mismatch, False)
        allow_seed_corpus = _bool(allow_seed_corpus, False)
        skill_obj = self._resolve_skill(skill)
        explicit_task = task is not None
        if skill_obj:
            question = question or skill_obj.question
            task = task or skill_obj.task
            model = skill_obj.model or model
            budget = budget or skill_obj.to_budget()
        if not model:
            raise ValueError("discovery needs an explicit model; pass --model or use a skill that declares a model.")
        task_name = (task or infer_task(question)).lower()
        self._validate_alignment(question, skill_obj, task_name, model, explicit_task, allow_mismatch)
        if not task_name:
            raise ValueError(
                "Discovery needs an explicit task when the prompt does not name a supported task. "
                "Choose --task ioi|induction|greater_than|factual_recall, use --skill, or run literature mode first."
            )
        get_task(task_name)  # validate
        question = question or f"Which components of {model} implement the {task_name} behaviour?"
        budget = _budget(budget or Budget())

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
                prior_sources, prior_evidence, prior_claims, used_packaged_seed_corpus, provider_source_added = self._prior_art(
                    question,
                    source_paths,
                    urls,
                    provider,
                    llm_model,
                    memory,
                    include_memory,
                    allow_seed_corpus,
                    tracer,
                )
            source_ids = [_text(getattr(source, "id", "")).strip() for source in _items(prior_sources)]
            source_ids = [source_id for source_id in source_ids if source_id]

            resolved_backend = engine.backend_for(model, backend)
            backend_name = _text(getattr(resolved_backend, "name", "")).strip() or "synthetic"
            coordinator = Coordinator(max_workers=default_workers(backend_name))

            hypotheses: list[Hypothesis] = []
            results_by_spec: dict = {}
            promoted_heads: set[tuple[int, int]] = set()
            max_screen = _positive_int(getattr(skill_obj, "max_screen_heads", 96), 96, upper=100_000) if skill_obj else 96
            top_k = _positive_int(getattr(skill_obj, "promote_top_k", 5), 5, upper=10_000) if skill_obj else 5
            min_confirmed = _nonnegative_int(getattr(skill_obj, "min_confirmed", 1), 1, upper=100_000) if skill_obj else 1
            min_rigor = _nonnegative_float(getattr(skill_obj, "min_rigor", 0.6), 0.6, upper=1.0) if skill_obj else 0.6

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
                        for hyp in _items(new_hyps):
                            target = _mapping(getattr(hyp, "target", {}))
                            key = (_label(target.get("layer")), _label(target.get("head")))
                            if not all(key):
                                continue
                            if key in promoted_heads:
                                continue
                            promoted_heads.add(key)
                            fresh.append(hyp)
                        new_hyps = fresh
                        hyp_ids = {_text(getattr(h, "id", "")).strip() for h in new_hyps}
                        hyp_ids.discard("")
                        specs = [s for s in _items(specs) if _text(getattr(s, "hypothesis_id", "")).strip() in hyp_ids]
                        if not new_hyps:
                            gaps.append("converged: no new significant leads to triangulate")
                            break

                    hypotheses.extend(_items(new_hyps))
                    specs = guard.admit(specs)
                    with tracer.span("experiments", count=len(specs)):
                        results = coordinator.map(engine.run_spec, specs)
                    guard.record(results)
                    for result in _items(results):
                        spec_id = _text(getattr(result, "spec_id", "")).strip()
                        if spec_id:
                            results_by_spec[spec_id] = result
                    update_hypotheses(hypotheses, results_by_spec)
                    gaps, metrics = exp_critic.evaluate(hypotheses, list(results_by_spec.values()))
                    tracer.event(
                        "round_summary",
                        round=round_index + 1,
                        confirmed=_number(metrics.get("confirmed_mechanisms", 0)),
                        rigor=_number(metrics.get("rigor_score", 0)),
                    )

                confirmed = _number(metrics.get("confirmed_mechanisms", 0))
                rigor = _number(metrics.get("rigor_score", 0))
                if confirmed >= min_confirmed and rigor >= min_rigor and round_index >= 1:
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
                    sum(_number(getattr(d, "confidence", 0.0)) for d in discoveries) / max(len(discoveries), 1), 3
                )
                metrics["mean_novelty"] = round(
                    sum(_number(getattr(d, "novelty", 0.0)) for d in discoveries) / max(len(discoveries), 1), 3
                )
                metrics["readiness_score"] = self._readiness(metrics)
                answer = self._synthesize(question, model, task_name, discoveries, gaps)
                answer_author = "experiment_ledger_synthesizer"
                answer_provider = ""
                answer_model = ""
                answer_note = ""
                should_author_with_provider = provider in {"openai", "anthropic"} or (
                    provider == "auto" and provider_source_added
                )
                if should_author_with_provider:
                    provider_answer, answer_diag = synthesize_answer_with_provider(
                        provider,
                        llm_model,
                        question=question,
                        claims=claims,
                        evidence=evidence,
                        gaps=_strings(gaps) + _strings(lit_gaps)[:2],
                        discoveries=discoveries,
                        experiments=list(results_by_spec.values()),
                    )
                    if provider_answer:
                        answer = provider_answer
                        answer_author = "provider_model"
                        answer_provider = answer_diag.get("provider", "")
                        answer_model = answer_diag.get("model", "")
                    else:
                        answer_note = answer_diag.get("reason", "")
            try:
                from . import __version__ as _ver
            except ImportError:
                _ver = "0.1.0"

            run = ResearchRun(
                run_id=run_id,
                question=question,
                created_at=utc_now(),
                plan=plan,
                sources=prior_sources + [self._experiment_source(model, task_name)],
                evidence=evidence,
                claims=claims,
                contradictions=[],
                gaps=_strings(gaps) + _strings(lit_gaps)[:2],
                answer=answer,
                metrics=metrics,
                provenance={
                    "engine": "discovery",
                    "code_version": _ver,
                    "provider_requested": provider,
                    "provider_source_added": provider_source_added,
                    "llm_model": llm_model or "",
                    "answer_author": answer_author,
                    "answer_provider": answer_provider,
                    "answer_model": answer_model,
                    "answer_note": answer_note,
                    "model": model,
                    "backend_requested": backend,
                    "backend_used": backend_name,
                    "skill": skill_obj.name if skill_obj else "",
                    "task": task_name,
                    "budget": {
                        "max_rounds": budget.max_rounds,
                        "max_experiments": budget.max_experiments,
                        "max_gpu_seconds": budget.max_gpu_seconds,
                    },
                    "requested_source_paths": source_paths,
                    "requested_urls": urls,
                    "included_memory": include_memory,
                    "allow_seed_corpus": allow_seed_corpus,
                    "used_packaged_seed_corpus": used_packaged_seed_corpus,
                    "source_count": len(prior_sources) + 1,
                    "hypothesis_count": len(hypotheses),
                    "experiment_count": len(results_by_spec),
                    "discovery_count": len(discoveries),
                },
                hypotheses=hypotheses,
                experiments=list(results_by_spec.values()),
                discoveries=discoveries,
                mode="discovery",
            )
            artifacts = write_artifacts(run, out_path)
            memory.record_run(run)
            memory.record_experiments(model, task_name, hypotheses, list(results_by_spec.values()), code_version=_ver)
            tracer.event("artifacts_written", **artifacts)
            refresh_run_manifest(run.artifacts["json"])
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

    def _validate_alignment(
        self,
        question: str,
        skill: Skill | None,
        task_name: str,
        model: str,
        explicit_task: bool,
        allow_mismatch: bool,
    ) -> None:
        issue = request_alignment_issue(question, skill, task_name, model, explicit_task)
        if issue and not allow_mismatch:
            raise ValueError(issue)

    def _prior_art(
        self,
        question,
        source_paths,
        urls,
        provider,
        llm_model,
        memory,
        include_memory,
        allow_seed_corpus,
        tracer,
    ):
        sources = load_sources(source_paths, urls)
        requested_sources = bool(source_paths or urls)
        if not sources and requested_sources:
            raise ValueError(
                "No supported text sources were loaded from the requested paths or URLs. "
                "Use .md, .txt, .rst, .html, .json, or .csv files."
            )
        if include_memory:
            sources.extend(memory.recall_sources(question))

        used_packaged_seed_corpus = False
        config = load_config()
        adapter = make_research_adapter(provider, llm_model, config)
        provider_source_added = False
        if adapter and getattr(adapter, "available", False):
            with tracer.span("provider_research", provider=provider):
                live = adapter.search_summary(question)
                if live:
                    sources.append(live)
                    provider_source_added = True
        if not sources and allow_seed_corpus:
            sources = load_sources([str(example_corpus_path())], [])
            used_packaged_seed_corpus = True
        memory.upsert_sources(sources)

        index = BM25Index.from_sources(sources)
        evidence = index.search(question, limit=10)
        claims = ClaimExtractor().extract(question, evidence, limit=12)
        return sources, evidence, claims, used_packaged_seed_corpus, provider_source_added

    def _build_discoveries(self, hypotheses, results_by_spec, prior_claims, model) -> list[Discovery]:
        discoveries: list[Discovery] = []
        result_map = results_by_spec if isinstance(results_by_spec, dict) else {}
        model_name = _text(model).strip() or "model"
        for hyp in _items(hypotheses):
            target = _mapping(getattr(hyp, "target", {}))
            layer = _label(target.get("layer"))
            head = _label(target.get("head"))
            if _text(getattr(hyp, "status", "")).strip() != "confirmed" or not layer or not head:
                continue
            tri = [result_map[eid] for eid in _strings(getattr(hyp, "experiment_ids", [])) if eid in result_map]
            confirming = [
                r
                for r in tri
                if _flag(getattr(r, "significant", False)) and _flag(getattr(r, "reproduced", False))
            ]
            confirming_probes = {_text(getattr(r, "probe", "")).strip() for r in confirming}
            confirming_probes.discard("")
            if len(confirming_probes) < 2:
                continue
            role = classify_head_role(tri)
            effect = max((abs(_number(getattr(r, "effect_size", 0.0))) for r in confirming), default=0.0)
            reproducibility = round(len(confirming) / max(len(tri), 1), 3)
            statement = (
                f"In {model_name}, attention head {layer}.{head} is a {role} for this task: "
                f"{len(confirming)} independent probes (out of {len(tri)}) confirm a reproducible effect."
            )
            novelty = self._novelty(statement, role, prior_claims)
            supporting_experiments = _strings([getattr(r, "id", "") for r in confirming])
            hyp_id = _text(getattr(hyp, "id", "")).strip()
            discoveries.append(
                Discovery(
                    id=stable_id("disc", f"{model_name}:{layer}:{head}:{role}"),
                    statement=statement,
                    confidence=_number(getattr(hyp, "confidence", 0.0)),
                    novelty=novelty,
                    effect_size=round(effect, 3),
                    reproducibility=reproducibility,
                    supporting_experiments=supporting_experiments,
                    hypothesis_id=hyp_id,
                )
            )
        discoveries.sort(key=lambda d: (d.confidence, d.effect_size), reverse=True)
        return discoveries

    def _novelty(self, statement: str, role: str, prior_claims: list[Claim]) -> float:
        claims = [claim for claim in _items(prior_claims) if _text(getattr(claim, "text", "")).strip()]
        if not claims:
            return 0.8
        best = max((cosine_overlap(statement, _text(getattr(claim, "text", ""))) for claim in claims), default=0.0)
        # Mentioning a known head *type* in the literature lowers novelty modestly.
        return round(max(0.1, min(0.95, 0.9 - best)), 3)

    def _ledger(self, hypotheses, results_by_spec, discoveries, prior_evidence, prior_claims):
        evidence: list[EvidenceChunk] = [
            row
            for row in _items(prior_evidence)
            if _text(getattr(row, "id", "")).strip() and _text(getattr(row, "source_id", "")).strip()
        ]
        claims: list[Claim] = [
            row
            for row in _items(prior_claims)
            if _text(getattr(row, "id", "")).strip() and _text(getattr(row, "text", "")).strip()
        ]
        result_map = results_by_spec if isinstance(results_by_spec, dict) else {}
        exp_source_id = stable_id("src", "mechferret:experiment-log")

        # Keep the ledger readable: include triangulation/lens results + significant screen hits.
        keep = {
            eid
            for hyp in _items(hypotheses)
            if "head" in _mapping(getattr(hyp, "target", {}))
            for eid in _strings(getattr(hyp, "experiment_ids", []))
        }
        chunk_by_spec: dict[str, str] = {}
        for spec_id, result in result_map.items():
            spec_key = _text(spec_id).strip()
            result_id = _text(getattr(result, "id", "")).strip()
            probe = _text(getattr(result, "probe", "")).strip()
            if not spec_key or not result_id or not probe:
                continue
            include = spec_key in keep or probe == "logit_lens" or (
                probe == "head_ablation"
                and _flag(getattr(result, "significant", False))
                and _flag(getattr(result, "reproduced", False))
            )
            if not include:
                continue
            target = _mapping(getattr(result, "target", {}))
            observations = _strings(getattr(result, "observations", []))
            backend_used = _text(getattr(result, "backend_used", "")).strip() or "unknown"
            chunk = EvidenceChunk(
                id=stable_id("ev", f"exp:{result_id}"),
                source_id=exp_source_id,
                title=f"{probe} @ {target}",
                text=f"{_text(getattr(result, 'evidence_text', '')).strip()} {' '.join(observations[-1:])}".strip(),
                url=f"experiment://{backend_used}/{probe}",
                score=round(abs(_number(getattr(result, "effect_size", 0.0))), 3),
            )
            evidence.append(chunk)
            chunk_by_spec[spec_key] = chunk.id

        for discovery in _items(discoveries):
            statement = _text(getattr(discovery, "statement", "")).strip()
            if not statement:
                continue
            supporting = set(_strings(getattr(discovery, "supporting_experiments", [])))
            cites = [
                chunk_by_spec[spec_id]
                for spec_id, result in result_map.items()
                if _text(getattr(result, "id", "")).strip() in supporting
                and _text(spec_id).strip() in chunk_by_spec
            ]
            reproducibility = _number(getattr(discovery, "reproducibility", 0.0))
            claim = Claim(
                id=stable_id("claim", statement.lower()),
                text=statement,
                citations=cites or [exp_source_id],
                source_ids=[exp_source_id],
                confidence=_number(getattr(discovery, "confidence", 0.0)),
                support_score=reproducibility,
                stance="discovery",
                quality_flags=[] if reproducibility >= 0.5 else ["weak_triangulation"],
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
        metric_map = metrics if isinstance(metrics, dict) else {}
        rigor = _number(metric_map.get("rigor_score", 0.0))
        confirmed = min(_number(metric_map.get("confirmed_mechanisms", 0.0)) / 2.0, 1.0)
        confidence = _number(metric_map.get("mean_discovery_confidence", 0.0))
        return round(max(0.0, min(0.99, 0.55 * rigor + 0.25 * confirmed + 0.20 * confidence)), 3)

    def _synthesize(self, question, model, task_name, discoveries: list[Discovery], gaps) -> str:
        discovery_rows = [row for row in _items(discoveries) if _text(getattr(row, "statement", "")).strip()]
        lines = [f"Answer to: {_text(question).strip()}", ""]
        if not discovery_rows:
            lines.append("No mechanism cleared the rigor bar (significant + reproducible + triangulated).")
        else:
            model_name = _text(model).strip() or "model"
            task = _text(task_name).strip() or "task"
            lines.append(f"Confirmed mechanisms in {model_name} for the {task} task:")
            for d in discovery_rows:
                statement = _text(getattr(d, "statement", "")).strip()
                confidence = _number(getattr(d, "confidence", 0.0))
                effect = _number(getattr(d, "effect_size", 0.0))
                reproducibility = _number(getattr(d, "reproducibility", 0.0))
                novelty = _number(getattr(d, "novelty", 0.0))
                lines.append(
                    f"- {statement} (confidence={confidence:.2f}, effect={effect:.2f}, "
                    f"reproducibility={reproducibility:.2f}, novelty={novelty:.2f})"
                )
        clean_gaps = _strings(gaps)
        if clean_gaps:
            lines.extend(["", "Open rigor gaps / next experiments:"])
            for gap in clean_gaps[:6]:
                lines.append(f"- {gap}")
        return "\n".join(lines)
