from __future__ import annotations

import json
import math
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

from .models import (
    Claim,
    Contradiction,
    Discovery,
    EvidenceChunk,
    ExperimentResult,
    Hypothesis,
    PlanStep,
    ResearchPlan,
    ResearchRun,
    Source,
)
from .provenance import PAPER_ARTIFACT_REQUIRED_MARKERS
from .report import run_evals
from .text import tokenize


ACTION_BY_CHECK = {
    "extracts_at_least_five_claims": "Run another retrieval/synthesis pass and extract more atomic claims.",
    "uses_at_least_three_sources": "Add at least three independent sources with `--source` or `--url`.",
    "citations_per_claim": "Attach citations to unsupported claims or remove claims that cannot be grounded.",
    "plan_coverage": "Run the missing plan steps or narrow the research question.",
    "contradiction_pressure_bounded": "Resolve or explicitly discuss the strongest contradiction pairs.",
    "has_confirmed_mechanism": "Run `/discover --skill <skill> --model <model> --backend transformer_lens` or another task-specific workflow until one mechanism is confirmed.",
    "every_experiment_has_control": "Add matched controls for every experiment before treating effects as evidence.",
    "significant_effects_reproduce": "Re-run significant effects across seeds and require reproducibility before promotion.",
    "discoveries_are_triangulated": "Triangulate each discovery with at least two independent probes.",
    "experiments_log_seed_values": "Record per-seed values for every ran experiment.",
    "manifest_integrity": "Run `mechferret verify` and regenerate any stale or tampered artifacts.",
    "paper_artifact_exists": "Run `mechferret paper --provider local` to generate or refresh the run-bound `paper/main.tex` with evidence and experiment ledgers.",
    "paper_artifact_structure": "Run `mechferret paper --provider local` to generate or refresh the run-bound `paper/main.tex` with evidence and experiment ledgers.",
    "question_result_alignment": "Rerun or narrow the dossier so confirmed findings directly address the user question.",
}

ADVISORY_ACTIONS = {
    "local_synthesis_not_final": "Use `--provider openai` or `--provider anthropic` to produce model-authored final synthesis from the run ledger.",
    "synthetic_backend_not_final": "Rerun discovery with `--backend transformer_lens`, Modal, or cluster execution before treating the mechanism as a real-model result.",
    "packaged_seed_corpus_used": "Replace the packaged demo corpus with project-specific `--source` or `--url` evidence before sharing as original research.",
}

GENERIC_QUESTION_TERMS = {
    "find",
    "make",
    "build",
    "show",
    "help",
    "research",
    "project",
    "paper",
    "become",
    "question",
    "model",
    "models",
    "task",
    "tasks",
    "mechanism",
    "mechanisms",
}


def latest_run_json(root: str | Path = "runs") -> Path | None:
    base = Path(root)
    if not base.exists():
        return None
    candidates = [p for p in base.rglob("run.json") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def audit_run_artifact(path: str | Path | None = None, *, runs_root: str | Path = "runs") -> dict[str, Any]:
    target = Path(path) if path else latest_run_json(runs_root)
    if target is None:
        checks = [
            {
                "name": "run_artifact_exists",
                "passed": False,
                "observed": "missing",
                "threshold": "runs/**/run.json",
            }
        ]
        return {
            "path": "",
            "passed": False,
            "checks": checks,
            "failed_checks": _failed_check_names(checks),
            "next_actions": ["Run `mechferret demo` or `mechferret discover --skill <skill> --model <model> --backend synthetic` to create a smoke-test dossier."],
            "readiness_score": 0,
        }
    run = load_run_artifact(target)
    evals = run_evals(run)
    checks = list(evals["checks"])
    _add_seed_gate(run, checks)
    _add_paper_gate(target, run, checks)
    _add_alignment_gate(run, checks)
    _add_manifest_gate(target, run, checks)
    advisories = _audit_advisories(run)
    failed_checks = _failed_check_names(checks)
    next_actions = _actions_for_failed_checks(failed_checks)
    return {
        "path": str(target),
        "passed": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "next_actions": next_actions,
        "advisories": advisories,
        "advisory_actions": [item["action"] for item in advisories if item.get("action")],
        "readiness_score": evals.get("readiness_score", 0),
        "run_id": run.run_id,
        "question": run.question,
    }


def load_run_artifact(path: str | Path) -> ResearchRun:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _run_from_payload(payload if isinstance(payload, dict) else {})


def print_audit(result: dict[str, Any]) -> None:
    print(f"Audit: {'PASS' if result['passed'] else 'WARN'}")
    if result.get("path"):
        print(f"Run: {result.get('run_id', '')}  readiness={_float(result.get('readiness_score', 0)):.2f}")
        print(f"Artifact: {result['path']}")
    for check in result["checks"]:
        marker = "ok" if check["passed"] else "fix"
        print(f"{marker:4} {check['name']}: {check.get('observed')} / {check.get('threshold')}")
    if result["next_actions"]:
        print("\nNext actions:")
        for action in result["next_actions"][:6]:
            print(f"  - {action}")
    if result.get("advisories"):
        print("\nAdvisories:")
        for item in result["advisories"][:6]:
            print(f"  - {item['name']}: {item.get('observed', '')}")


def _failed_check_names(checks: list[dict[str, Any]]) -> list[str]:
    return [check["name"] for check in checks if not check["passed"]]


def _actions_for_failed_checks(failed_checks: list[str]) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for name in failed_checks:
        action = ACTION_BY_CHECK.get(name, f"Fix failed gate: {name}")
        if action in seen:
            continue
        seen.add(action)
        actions.append(action)
    return actions


def _audit_advisories(run: ResearchRun) -> list[dict[str, Any]]:
    advisories: list[dict[str, Any]] = []
    answer_author = run.provenance.get("answer_author", "")
    if answer_author in {"", "local_extractive_synthesizer", "experiment_ledger_synthesizer"}:
        advisories.append(
            {
                "name": "local_synthesis_not_final",
                "severity": "info",
                "observed": answer_author or "missing answer_author provenance",
                "action": ADVISORY_ACTIONS["local_synthesis_not_final"],
            }
        )
    backend_used = str(run.provenance.get("backend_used", "")).lower()
    experiment_backends = {str(experiment.backend_used).lower() for experiment in run.experiments}
    if run.mode == "discovery" and ("synthetic" in {backend_used, *experiment_backends}):
        advisories.append(
            {
                "name": "synthetic_backend_not_final",
                "severity": "warning",
                "observed": backend_used or ", ".join(sorted(experiment_backends)) or "synthetic",
                "action": ADVISORY_ACTIONS["synthetic_backend_not_final"],
            }
        )
    if run.provenance.get("used_packaged_seed_corpus") or _uses_packaged_seed_corpus(run):
        advisories.append(
            {
                "name": "packaged_seed_corpus_used",
                "severity": "info",
                "observed": "true",
                "action": ADVISORY_ACTIONS["packaged_seed_corpus_used"],
            }
        )
    return advisories


def _uses_packaged_seed_corpus(run: ResearchRun) -> bool:
    requested = [str(item) for item in _list(run.provenance.get("requested_source_paths", []))]
    source_paths = [
        str(source.metadata.get("path", source.url))
        for source in run.sources
        if isinstance(source.metadata, dict)
    ]
    return any("seed_corpus" in path for path in requested + source_paths)


def _add_seed_gate(run: ResearchRun, checks: list[dict[str, Any]]) -> None:
    if not (run.mode == "discovery" or run.experiments or run.discoveries):
        return
    ran = [e for e in run.experiments if e.status == "ran"]
    with_seeds = [e for e in ran if e.per_seed]
    checks.append(
        {
            "name": "experiments_log_seed_values",
            "passed": bool(ran) and len(with_seeds) == len(ran),
            "observed": f"{len(with_seeds)}/{len(ran)}",
            "threshold": "all",
        }
    )


def _add_paper_gate(path: Path, run: ResearchRun, checks: list[dict[str, Any]]) -> None:
    paper = path.parent / "paper" / "main.tex"
    should_have_paper = run.discoveries or _float(run.metrics.get("readiness_score", 0)) >= 0.7
    exists = paper.exists()
    checks.append(
        {
            "name": "paper_artifact_exists",
            "passed": (not should_have_paper) or exists,
            "observed": str(paper) if exists else "missing",
            "threshold": "required after confirmed discoveries or readiness>=0.70",
        }
    )
    if not (exists or should_have_paper):
        return
    structure = _paper_structure_status(paper) if exists else {"passed": False, "observed": "missing"}
    checks.append(
        {
            "name": "paper_artifact_structure",
            "passed": bool(structure["passed"]),
            "observed": structure["observed"],
            "threshold": "article TeX with document body, Results, Experiment Ledger, Evidence Ledger, and Limitations sections",
        }
    )


def _paper_structure_status(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"passed": False, "observed": f"unreadable: {_string(exc)[:120]}"}
    missing = [item for item in PAPER_ARTIFACT_REQUIRED_MARKERS if item not in text]
    return {
        "passed": not missing,
        "observed": "present" if not missing else "missing " + ", ".join(missing[:4]),
    }


def _add_alignment_gate(run: ResearchRun, checks: list[dict[str, Any]]) -> None:
    query_terms = {
        term for term in tokenize(run.question)
        if len(term) >= 4 and term not in GENERIC_QUESTION_TERMS
    }
    if not query_terms:
        return
    result_text = " ".join(
        [claim.text for claim in run.claims]
        + [discovery.statement for discovery in run.discoveries]
        + [str(experiment.target) for experiment in run.experiments]
    )
    result_terms = set(tokenize(result_text))
    matched = sorted(query_terms & result_terms)
    threshold = 1 if len(query_terms) <= 2 else 2
    checks.append(
        {
            "name": "question_result_alignment",
            "passed": len(matched) >= threshold,
            "observed": ", ".join(matched) or "no key question terms in findings",
            "threshold": f">={threshold} of {', '.join(sorted(query_terms))}",
        }
    )


def _add_manifest_gate(path: Path, run: ResearchRun, checks: list[dict[str, Any]]) -> None:
    manifest_declared = bool(run.artifacts.get("manifest") or (path.parent / "manifest.json").exists())
    if not manifest_declared:
        checks.append(
            {
                "name": "manifest_integrity",
                "passed": True,
                "observed": "not present",
                "threshold": "required for current MechFerret runs; rerun to create manifest.json",
            }
        )
        return
    from .provenance import verify_run_manifest

    verification = verify_run_manifest(path)
    checks.append(
        {
            "name": "manifest_integrity",
            "passed": verification["passed"],
            "observed": "passed" if verification["passed"] else ", ".join(verification.get("failed_checks", [])[:4]),
            "threshold": "verify_run_manifest passes",
        }
    )


LIST_FIELDS = {
    "steps",
    "sources",
    "evidence",
    "claims",
    "contradictions",
    "gaps",
    "citations",
    "source_ids",
    "quality_flags",
    "highlights",
    "notes",
    "experiment_ids",
    "controls",
    "seeds",
    "per_seed",
    "observations",
    "supporting_experiments",
    "claim_ids",
}
DICT_FIELDS = {"metadata", "target", "params", "metrics", "artifacts", "provenance"}
BOOL_FIELDS = {"significant", "reproduced"}
NUMBER_FIELDS = {
    "score",
    "confidence",
    "support_score",
    "severity",
    "effect_size",
    "baseline",
    "gpu_seconds",
    "novelty",
    "reproducibility",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _rows(value: Any) -> list[dict[str, Any]]:
    return [item for item in _list(value) if isinstance(item, dict)]


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _float(value: Any) -> float:
    if type(value) is bool:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _field_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.name == "created_at":
        return ""
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    if field.name in LIST_FIELDS:
        return []
    if field.name in DICT_FIELDS:
        return {}
    if field.name in BOOL_FIELDS:
        return False
    if field.name in NUMBER_FIELDS:
        return 0.0
    return ""


def _coerce_field(name: str, value: Any, default: Any) -> Any:
    if value is None:
        return default
    if name in LIST_FIELDS:
        return _list(value)
    if name in DICT_FIELDS:
        return _dict(value)
    if name in BOOL_FIELDS:
        return value if type(value) is bool else default
    if name in NUMBER_FIELDS:
        return _float(value)
    return _string(value)


def _pick(data: dict[str, Any], cls):
    names = {f.name for f in fields(cls)}
    values: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in names:
            continue
        default = _field_default(field)
        values[field.name] = _coerce_field(field.name, data.get(field.name, default), default)
    return cls(**values)


def _run_from_payload(payload: dict[str, Any]) -> ResearchRun:
    payload = _dict(payload)
    plan_payload = _dict(payload.get("plan"))
    steps = [_pick(step, PlanStep) for step in _rows(plan_payload.get("steps", []))]
    plan = ResearchPlan(
        question=_string(plan_payload.get("question", payload.get("question", ""))),
        steps=steps,
        strategy=_string(plan_payload.get("strategy", "")),
    )
    return ResearchRun(
        run_id=_string(payload.get("run_id", payload.get("id", ""))),
        question=_string(payload.get("question", "")),
        created_at=_string(payload.get("created_at", "")),
        plan=plan,
        sources=[_pick(item, Source) for item in _rows(payload.get("sources", []))],
        evidence=[_pick(item, EvidenceChunk) for item in _rows(payload.get("evidence", []))],
        claims=[_pick(item, Claim) for item in _rows(payload.get("claims", []))],
        contradictions=[_pick(item, Contradiction) for item in _rows(payload.get("contradictions", []))],
        gaps=[_string(item) for item in _list(payload.get("gaps", [])) if _string(item)],
        answer=_string(payload.get("answer", "")),
        metrics=_dict(payload.get("metrics", {})),
        artifacts=_dict(payload.get("artifacts", {})),
        provenance=_dict(payload.get("provenance", {})),
        hypotheses=[_pick(item, Hypothesis) for item in _rows(payload.get("hypotheses", []))],
        experiments=[_pick(item, ExperimentResult) for item in _rows(payload.get("experiments", []))],
        discoveries=[_pick(item, Discovery) for item in _rows(payload.get("discoveries", []))],
        mode=_string(payload.get("mode", "literature")) or "literature",
    )
