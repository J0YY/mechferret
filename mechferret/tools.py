"""Agent tool registry — the capabilities the conversational model can call.

Modeled on Claude Code's tool set, adapted to a Python interpretability-research
coding agent: shell, file read/write/edit, glob, grep, web search/fetch, plus
the interp-specific tools (arXiv, Neuronpedia, the discovery loop, skills).

Each tool is ``{name, description, parameters(JSON Schema)}`` with a handler in
:data:`HANDLERS` returning a string the model reads back. ``agent.py`` converts
these to the Anthropic/OpenAI tool formats.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

MAX_OUTPUT = 12000
PERSIST_THRESHOLD = 16000  # results larger than this are written to disk, not truncated
JSON_PREVIEW_FIELD_LIMIT = 2000
CHECK_LIST_STRUCTURED_LIMIT = 480
RESULTS_DIR = Path(".mechferret/tool_results")
DEFAULT_WEB_RESULTS = 12
DEFAULT_ARXIV_RESULTS = 20
NOVELTY_RELATED_LIMIT = 24
NOVELTY_FOCUSED_LIMIT = 10
NOVELTY_QUERY_RESULT_LIMIT = 20
NOVELTY_MAX_QUERY_PASSES = 12
NOVELTY_CLOSEST_PRIOR_LIMIT = 8
NOVELTY_WEB_RESULT_LIMIT = 12
NOVELTY_WEB_MAX_QUERY_PASSES = 5
NOVELTY_RISKS = {
    "high_prior_art_risk",
    "medium_prior_art_risk",
    "low_prior_art_risk",
    "unresolved_no_close_prior_found",
    "unknown_search_incomplete",
}
NOVELTY_WEB_SOURCE_TYPES = {
    "paper",
    "benchmark",
    "code_repository",
    "project_page",
    "documentation",
    "general_web",
}


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [output truncated]"


def _persist_if_large(name: str, result: str) -> str:
    """Large tool results are saved whole to disk; the model gets a preview + path."""

    if len(result) <= PERSIST_THRESHOLD:
        return result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(result.encode("utf-8", "ignore")).hexdigest()[:12]
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        path = RESULTS_DIR / f"{name}_{digest}.txt"
        path.write_text(result, encoding="utf-8")
        return f"{result[:MAX_OUTPUT]}\n… [full output saved to {path} — read_file it for the rest]"
    path = RESULTS_DIR / f"{name}_{digest}.json"
    path.write_text(result, encoding="utf-8")
    return _persisted_json_summary(name, payload, path, result)


def _saved_tool_result_rows(limit: int = 20) -> tuple[int, list[dict[str, Any]]]:
    if not RESULTS_DIR.exists():
        return 0, []
    files = [path for path in RESULTS_DIR.iterdir() if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    rows = [_tool_result_row(path) for path in files[: max(0, limit)]]
    return len(files), rows


def _tool_result_row(path: Path) -> dict[str, Any]:
    stat = path.stat()
    name = path.stem.rsplit("_", 1)[0] if "_" in path.stem else path.stem
    row = {
        "tool": name,
        "path": str(path),
        "bytes": stat.st_size,
        "modified_at": int(stat.st_mtime),
        "age_seconds": max(0, int(time.time() - stat.st_mtime)),
        "is_json": False,
    }
    row["is_json"] = _looks_like_json_file(path)
    return row


def _looks_like_json_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            prefix = handle.read(4096).lstrip()
    except OSError:
        return False
    return bool(prefix) and prefix[0] in "[{"


def _persisted_json_summary(name: str, payload: Any, path: Path, result: str) -> str:
    if isinstance(payload, dict):
        summary = _compact_json_object(payload)
        if isinstance(payload.get("checks"), list) and _preserve_structured_check_rows(name, payload["checks"]):
            summary["checks"] = [_compact_check_row(item) for item in payload["checks"]]
    else:
        summary = {
            "result_type": type(payload).__name__,
            "result_preview": _compact_json_value(payload),
        }
    actions = list(summary.get("next_actions", [])) if isinstance(summary.get("next_actions"), list) else []
    actions.append(f"Read the complete {name} result from {path}.")
    summary.update(
        {
            "tool_output_truncated": True,
            "full_output_path": str(path),
            "full_output_bytes": len(result.encode("utf-8")),
            "next_actions": actions,
        }
    )
    encoded = json.dumps(summary)
    if len(encoded) <= MAX_OUTPUT:
        return encoded
    minimal = {
        "tool_output_truncated": True,
        "tool": name,
        "full_output_path": str(path),
        "full_output_bytes": len(result.encode("utf-8")),
        "next_actions": [f"Read the complete {name} result from {path}."],
    }
    for key in (
        "ok",
        "passed",
        "path",
        "manifest",
        "run_json",
        "repairable",
        "repair_attempted",
        "repaired",
        "repair_blocked",
        "before_failed_checks",
        "failed_checks",
        "error",
        "risk",
        "verdict",
        "coverage",
        "required_delta",
        "closest_prior_art",
    ):
        if isinstance(payload, dict) and key in payload:
            minimal[key] = payload[key]
    if isinstance(summary.get("checks"), list):
        compact_checks = summary["checks"]
        minimal["checks"] = compact_checks
        if len(json.dumps(minimal)) > MAX_OUTPUT:
            minimal["checks"] = compact_checks[:80]
            minimal["checks_omitted"] = True
            minimal["check_count"] = len(compact_checks)
    return json.dumps(minimal)


def _compact_json_object(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        summary[key] = _compact_json_value(value)
    return summary


def _compact_json_value(value: Any) -> Any:
    try:
        encoded = json.dumps(value)
    except TypeError:
        return str(value)
    if len(encoded) <= JSON_PREVIEW_FIELD_LIMIT:
        return value
    if isinstance(value, list):
        if _looks_like_check_list(value) and len(value) <= CHECK_LIST_STRUCTURED_LIMIT:
            return [_compact_check_row(item) for item in value]
        return {
            "omitted": True,
            "type": "list",
            "count": len(value),
            "preview": [_compact_json_preview_item(item) for item in value[:3]],
        }
    if isinstance(value, dict):
        keys = list(value)[:20]
        summary = {
            "omitted": True,
            "type": "object",
            "key_count": len(value),
            "keys": keys,
        }
        for key in (
            "ok",
            "exists",
            "passed",
            "state",
            "path",
            "run_json",
            "failed_checks",
            "repairable",
            "risk",
            "verdict",
            "coverage",
            "required_delta",
            "closest_prior_art",
        ):
            if key in value and isinstance(value[key], (str, int, float, bool, list, type(None))):
                summary[key] = value[key]
            elif key in value and key == "coverage" and isinstance(value[key], dict):
                summary[key] = _compact_json_value(value[key])
        return summary
    if isinstance(value, str):
        return {
            "omitted": True,
            "type": "string",
            "bytes": len(value.encode("utf-8")),
            "preview": value[:500],
        }
    return {
        "omitted": True,
        "type": type(value).__name__,
    }


def _preserve_structured_check_rows(name: str, checks: list[Any]) -> bool:
    if name not in {"verify_bundle", "verify_run"}:
        return False
    verifier_prefixes = (
        "artifact_",
        "bundle_",
        "claim_",
        "discovery_",
        "evals_",
        "evidence_",
        "experiment_",
        "graph_",
        "hypothesis_",
        "manifest_",
        "paper_",
        "report_",
        "review_",
        "run_",
        "source_",
        "trace_",
    )
    return any(
        isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and row["name"].startswith(verifier_prefixes)
        for row in checks
    )


def _looks_like_check_list(value: list[Any]) -> bool:
    return all(isinstance(item, dict) and isinstance(item.get("name"), str) and "passed" in item for item in value)


def _compact_check_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item.get("name", ""),
        "passed": bool(item.get("passed")),
    }


def _compact_json_preview_item(value: Any) -> Any:
    try:
        encoded = json.dumps(value)
    except TypeError:
        return str(value)
    if len(encoded) <= 500:
        return value
    if isinstance(value, dict):
        return {"type": "object", "keys": list(value)[:8], "key_count": len(value)}
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if isinstance(value, str):
        return {"type": "string", "bytes": len(value.encode("utf-8")), "preview": value[:200]}
    return {"type": type(value).__name__}


# --- coding tools -------------------------------------------------------------------

def tool_bash(args: dict[str, Any]) -> str:
    cmd, invalid = _string_arg(args, "command")
    if invalid:
        return json.dumps(invalid)
    timeout, invalid = _int_arg(args, "timeout", 120, min_value=1)
    if invalid:
        return json.dumps(invalid)
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"
    out = proc.stdout or ""
    err = proc.stderr or ""
    body = out + (f"\n[stderr]\n{err}" if err.strip() else "")
    return _truncate(f"[exit {proc.returncode}]\n{body}".strip())


def tool_read_file(args: dict[str, Any]) -> str:
    raw_path, invalid = _string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return f"error: not a file: {path}"
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _truncate(_read_pdf(path))
    if suffix == ".ipynb":
        return _truncate(_read_notebook(path))
    offset, invalid = _int_arg(args, "offset", 0, min_value=0)
    if invalid:
        return json.dumps(invalid)
    limit, invalid = _int_arg(args, "limit", 2000, min_value=0)
    if invalid:
        return json.dumps(invalid)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    chunk = lines[offset: offset + limit]
    numbered = "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(chunk))
    return _truncate(numbered or "(empty)")


def tool_list_tool_results(args: dict[str, Any]) -> str:
    limit, invalid = _int_arg(args, "limit", 20, min_value=0)
    if invalid:
        return json.dumps(invalid)
    count, rows = _saved_tool_result_rows(limit)
    actions = []
    if count:
        actions.append("Open a path from `results[].path` to inspect the full saved output.")
        actions.append("Preview cleanup with `mechferret tool-results --clean --json`.")
        actions.append("Delete stale saved outputs with `mechferret tool-results --clean --confirm` after reviewing the preview.")
    return json.dumps(
        {
            "ok": True,
            "root": str(RESULTS_DIR),
            "count": count,
            "shown": len(rows),
            "results": rows,
            "next_actions": actions,
        }
    )


def tool_clean_tool_results(args: dict[str, Any]) -> str:
    keep_latest, invalid = _int_arg(args, "keep_latest", 20, min_value=0)
    if invalid:
        return json.dumps(invalid)
    max_age_days, invalid = _float_arg(args, "max_age_days", 7.0, min_value=0.0)
    if invalid:
        return json.dumps(invalid)
    confirm, invalid = _bool_arg(args, "confirm", False)
    if invalid:
        return json.dumps(invalid)
    dry_run, invalid = _bool_arg(args, "dry_run", not confirm)
    if invalid:
        return json.dumps(invalid)
    if not RESULTS_DIR.exists():
        return json.dumps({"ok": True, "root": str(RESULTS_DIR), "dry_run": dry_run, "deleted": [], "would_delete": [], "kept": []})

    files = [path for path in RESULTS_DIR.iterdir() if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    cutoff = time.time() - max(0.0, max_age_days) * 24 * 60 * 60
    stale = [
        path
        for index, path in enumerate(files)
        if index >= max(0, keep_latest) or path.stat().st_mtime < cutoff
    ]
    deleted: list[dict[str, Any]] = []
    would_delete = [_tool_result_row(path) for path in stale]
    if not dry_run:
        for path in stale:
            row = _tool_result_row(path)
            try:
                path.unlink()
            except OSError as exc:
                row["error"] = str(exc)
            deleted.append(row)
        would_delete = []
    kept = [_tool_result_row(path) for path in files if path not in stale][: max(0, keep_latest)]
    return json.dumps(
        {
            "ok": not any("error" in row for row in deleted),
            "root": str(RESULTS_DIR),
            "dry_run": dry_run,
            "deleted": deleted,
            "would_delete": would_delete,
            "kept": kept,
            "next_actions": [] if not dry_run else ["Re-run with confirm=true to delete the listed saved tool results."],
        }
    )


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "error: reading PDFs needs pypdf (pip install pypdf), or fetch the arXiv abstract instead"
    reader = PdfReader(str(path))
    return "\n\n".join(f"[page {i + 1}]\n{(p.extract_text() or '').strip()}" for i, p in enumerate(reader.pages))


def _read_notebook(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    out = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        out.append(f"[cell {i} · {cell.get('cell_type')}]\n{src}")
    return "\n\n".join(out) or "(empty notebook)"


def tool_write_file(args: dict[str, Any]) -> str:
    raw_path, invalid = _string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    content, invalid = _optional_string_arg(args, "content", "")
    if invalid:
        return json.dumps(invalid)
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")
    return f"wrote {len(content or '')} chars to {path}"


def tool_edit_file(args: dict[str, Any]) -> str:
    raw_path, invalid = _string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    old, invalid = _string_arg(args, "old_string")
    if invalid:
        return json.dumps(invalid)
    new, invalid = _optional_string_arg(args, "new_string", "")
    if invalid:
        return json.dumps(invalid)
    replace_all, invalid = _bool_arg(args, "replace_all", False)
    if invalid:
        return json.dumps(invalid)
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return f"error: not a file: {path}"
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if count > 1 and not replace_all:
        return f"error: old_string appears {count} times; pass replace_all=true or add context"
    text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    return f"edited {path} ({'all ' + str(count) if replace_all else '1'} occurrence(s))"


def tool_list_dir(args: dict[str, Any]) -> str:
    raw_path, invalid = _optional_string_arg(args, "path", ".")
    if invalid:
        return json.dumps(invalid)
    path = Path(raw_path or ".").expanduser()
    if not path.is_dir():
        return f"error: not a directory: {path}"
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    return _truncate("\n".join(("[dir] " if p.is_dir() else "      ") + p.name for p in entries) or "(empty)")


def tool_glob(args: dict[str, Any]) -> str:
    raw_path, invalid = _optional_string_arg(args, "path", ".")
    if invalid:
        return json.dumps(invalid)
    base = Path(raw_path or ".").expanduser()
    pattern, invalid = _string_arg(args, "pattern")
    if invalid:
        return json.dumps(invalid)
    matches = sorted(str(p) for p in base.glob(pattern))
    return _truncate("\n".join(matches) or "(no matches)")


def tool_grep(args: dict[str, Any]) -> str:
    pattern, invalid = _string_arg(args, "pattern")
    if invalid:
        return json.dumps(invalid)
    path, invalid = _optional_string_arg(args, "path", ".")
    if invalid:
        return json.dumps(invalid)
    glob, invalid = _optional_string_arg(args, "glob")
    if invalid:
        return json.dumps(invalid)
    cmd = ["rg", "-n", "--no-heading", pattern, path]
    if glob:
        cmd[1:1] = ["-g", glob]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return _truncate(proc.stdout or "(no matches)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Fall back to Python if ripgrep is unavailable.
        import re

        rx = re.compile(pattern)
        hits = []
        root = Path(path).expanduser()
        files = root.rglob(glob or "*") if root.is_dir() else [root]
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f}:{i}:{line}")
            except OSError:
                continue
        return _truncate("\n".join(hits) or "(no matches)")


# --- web + knowledge tools ----------------------------------------------------------

def tool_web_search(args: dict[str, Any]) -> str:
    from .knowledge import web_search

    query, invalid = _string_arg(args, "query")
    if invalid:
        return json.dumps(invalid)
    max_results, invalid = _int_arg(args, "max_results", DEFAULT_WEB_RESULTS, min_value=1)
    if invalid:
        return json.dumps(invalid)
    results = web_search(query, max_results=max_results)
    return json.dumps(results) if results else "(no results)"


def tool_web_fetch(args: dict[str, Any]) -> str:
    from .knowledge import web_fetch

    url, invalid = _string_arg(args, "url")
    if invalid:
        return json.dumps(invalid)
    max_chars, invalid = _int_arg(args, "max_chars", 6000, min_value=1)
    if invalid:
        return json.dumps(invalid)
    return _truncate(web_fetch(url, max_chars=max_chars))


def tool_arxiv_search(args: dict[str, Any]) -> str:
    from .knowledge import search_arxiv

    query, invalid = _string_arg(args, "query")
    if invalid:
        return json.dumps(invalid)
    max_results, invalid = _int_arg(args, "max_results", DEFAULT_ARXIV_RESULTS, min_value=1)
    if invalid:
        return json.dumps(invalid)
    sort_by, invalid = _enum_arg(args, "sort_by", "relevance", ARXIV_SORTS)
    if invalid:
        return json.dumps(invalid)
    total, papers = search_arxiv(
        query, max_results=max_results, sort_by=sort_by
    )
    return json.dumps({"total": total, "papers": papers})


def tool_neuronpedia_search(args: dict[str, Any]) -> str:
    from .knowledge import neuronpedia_search_explanations

    model_id, invalid = _string_arg(args, "model_id")
    if invalid:
        return json.dumps(invalid)
    query, invalid = _string_arg(args, "query")
    if invalid:
        return json.dumps(invalid)
    return json.dumps(neuronpedia_search_explanations(model_id, query))


# --- interp tools -------------------------------------------------------------------

def tool_run_research(args: dict[str, Any]) -> str:
    from .audit import audit_run_artifact
    from .controller import MechFerret

    provider, invalid = _enum_arg(args, "provider", "auto", PROVIDERS)
    if invalid:
        return json.dumps(invalid)
    question, invalid = _optional_string_arg(args, "question", "")
    if invalid:
        return json.dumps(invalid)
    source_paths, invalid = _optional_string_list_arg(args, "source_paths")
    if invalid:
        return json.dumps(invalid)
    urls, invalid = _optional_string_list_arg(args, "urls")
    if invalid:
        return json.dumps(invalid)
    out_dir, invalid = _optional_string_arg(args, "out_dir", "runs/agent")
    if invalid:
        return json.dumps(invalid)
    db_path, invalid = _optional_string_arg(args, "db_path", ".mechferret/memory.sqlite")
    if invalid:
        return json.dumps(invalid)
    model, invalid = _optional_string_arg(args, "model")
    if invalid:
        return json.dumps(invalid)
    max_rounds, invalid = _int_arg(args, "max_rounds", 2, min_value=1)
    if invalid:
        return json.dumps(invalid)
    no_memory, invalid = _bool_arg(args, "no_memory", False)
    if invalid:
        return json.dumps(invalid)
    include_memory, invalid = _bool_arg(args, "include_memory", not no_memory)
    if invalid:
        return json.dumps(invalid)
    seed_corpus, invalid = _bool_arg(args, "seed_corpus", False)
    if invalid:
        return json.dumps(invalid)
    allow_seed_corpus, invalid = _bool_arg(args, "allow_seed_corpus", seed_corpus)
    if invalid:
        return json.dumps(invalid)
    try:
        run = MechFerret(db_path or ".mechferret/memory.sqlite").run(
            question or "",
            source_paths=source_paths,
            urls=urls,
            out_dir=out_dir or "runs/agent",
            max_rounds=max_rounds,
            provider=provider,
            model=model,
            include_memory=include_memory,
            allow_seed_corpus=allow_seed_corpus,
        )
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "next_actions": [
                    "Add source_paths or urls for project-specific evidence.",
                    "Use provider=openai or provider=anthropic for live research if configured.",
                    "Set allow_seed_corpus=true only for an explicit packaged-corpus demo.",
                ],
            }
        )
    audit = audit_run_artifact(run.artifacts.get("json"))
    claims = run.claims[:8]
    evidence = _evidence_for_claims(run.evidence, claims, limit=16)
    sources = _sources_for_evidence(run.sources, evidence, limit=12)
    return json.dumps(
        {
            "ok": True,
            "answer": run.answer,
            "sources": [
                {
                    "id": source.id,
                    "title": source.title,
                    "kind": source.kind,
                    "url": source.url,
                }
                for source in sources
            ],
            "evidence": [
                {
                    "id": chunk.id,
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "url": chunk.url,
                    "score": chunk.score,
                    "highlights": chunk.highlights,
                }
                for chunk in evidence
            ],
            "claims": [
                {
                    "id": claim.id,
                    "text": claim.text,
                    "confidence": claim.confidence,
                    "support_score": claim.support_score,
                    "citations": claim.citations,
                    "quality_flags": claim.quality_flags,
                }
                for claim in claims
            ],
            "metrics": {
                key: run.metrics.get(key)
                for key in (
                    "readiness_score",
                    "claims",
                    "evidence_chunks",
                    "source_diversity",
                    "plan_coverage",
                    "citation_density",
                )
            },
            "provenance": {
                "answer_author": run.provenance.get("answer_author", ""),
                "answer_provider": run.provenance.get("answer_provider", ""),
                "answer_model": run.provenance.get("answer_model", ""),
                "provider_requested": run.provenance.get("provider_requested", ""),
                "provider_available": run.provenance.get("provider_available", False),
                "provider_source_added": run.provenance.get("provider_source_added", False),
                "used_packaged_seed_corpus": run.provenance.get("used_packaged_seed_corpus", False),
                "source_count": run.provenance.get("source_count", 0),
            },
            "audit": {
                "passed": audit.get("passed", False),
                "failed_checks": audit.get("failed_checks", []),
                "advisories": audit.get("advisories", []),
                "next_actions": audit.get("next_actions", []),
            },
            "artifacts": run.artifacts,
            "report_html": run.artifacts.get("html"),
        }
    )


def _evidence_for_claims(evidence, claims, *, limit: int):
    by_id = {chunk.id: chunk for chunk in evidence}
    selected = []
    seen: set[str] = set()
    for claim in claims:
        for citation in claim.citations:
            chunk = by_id.get(citation)
            if chunk is None or chunk.id in seen:
                continue
            selected.append(chunk)
            seen.add(chunk.id)
    for chunk in evidence:
        if len(selected) >= limit:
            break
        if chunk.id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk.id)
    return selected[:limit]


def _sources_for_evidence(sources, evidence, *, limit: int):
    by_id = {source.id: source for source in sources}
    selected = []
    seen: set[str] = set()
    for chunk in evidence:
        source = by_id.get(chunk.source_id)
        if source is None or source.id in seen:
            continue
        selected.append(source)
        seen.add(source.id)
    for source in sources:
        if len(selected) >= limit:
            break
        if source.id in seen:
            continue
        selected.append(source)
        seen.add(source.id)
    return selected[:limit]


def tool_run_discovery(args: dict[str, Any]) -> str:
    from .audit import audit_run_artifact
    from .discovery import DiscoveryController
    from .hooks import Budget

    provider, invalid = _enum_arg(args, "provider", "auto", PROVIDERS)
    if invalid:
        return json.dumps(invalid)
    backend, invalid = _enum_arg(args, "backend", "auto", DISCOVERY_BACKENDS)
    if invalid:
        return json.dumps(invalid)
    question, invalid = _optional_string_arg(args, "question", "")
    if invalid:
        return json.dumps(invalid)
    skill, invalid = _optional_string_arg(args, "skill")
    if invalid:
        return json.dumps(invalid)
    task, invalid = _optional_string_arg(args, "task")
    if invalid:
        return json.dumps(invalid)
    model, invalid = _optional_string_arg(args, "model")
    if invalid:
        return json.dumps(invalid)
    llm_model, invalid = _optional_string_arg(args, "llm_model")
    if invalid:
        return json.dumps(invalid)
    source_paths, invalid = _optional_string_list_arg(args, "source_paths")
    if invalid:
        return json.dumps(invalid)
    urls, invalid = _optional_string_list_arg(args, "urls")
    if invalid:
        return json.dumps(invalid)
    out_dir, invalid = _optional_string_arg(args, "out_dir", "runs/agent")
    if invalid:
        return json.dumps(invalid)
    db_path, invalid = _optional_string_arg(args, "db_path", ".mechferret/memory.sqlite")
    if invalid:
        return json.dumps(invalid)
    base_budget = Budget()
    max_rounds, invalid = _int_arg(args, "max_rounds", base_budget.max_rounds, min_value=1)
    if invalid:
        return json.dumps(invalid)
    max_experiments, invalid = _int_arg(args, "max_experiments", base_budget.max_experiments, min_value=1)
    if invalid:
        return json.dumps(invalid)
    max_gpu_seconds, invalid = _float_arg(args, "max_gpu_seconds", base_budget.max_gpu_seconds, min_value=0.0)
    if invalid:
        return json.dumps(invalid)
    allow_mismatch, invalid = _bool_arg(args, "allow_mismatch", False)
    if invalid:
        return json.dumps(invalid)
    no_memory, invalid = _bool_arg(args, "no_memory", False)
    if invalid:
        return json.dumps(invalid)
    include_memory, invalid = _bool_arg(args, "include_memory", not no_memory)
    if invalid:
        return json.dumps(invalid)
    seed_corpus, invalid = _bool_arg(args, "seed_corpus", False)
    if invalid:
        return json.dumps(invalid)
    allow_seed_corpus, invalid = _bool_arg(args, "allow_seed_corpus", seed_corpus)
    if invalid:
        return json.dumps(invalid)
    budget_requested = any(args.get(name) not in (None, "") for name in ("max_rounds", "max_experiments", "max_gpu_seconds"))
    budget = Budget(max_rounds=max_rounds, max_experiments=max_experiments, max_gpu_seconds=max_gpu_seconds) if budget_requested else None
    run = DiscoveryController(db_path or ".mechferret/memory.sqlite").run(
        question=question or "",
        skill=skill,
        task=task,
        model=model,
        backend=backend,
        source_paths=source_paths,
        urls=urls,
        out_dir=out_dir or "runs/agent",
        budget=budget,
        provider=provider,
        llm_model=llm_model,
        include_memory=include_memory,
        allow_mismatch=allow_mismatch,
        allow_seed_corpus=allow_seed_corpus,
    )
    discoveries = [
        {
            "id": d.id,
            "statement": d.statement,
            "confidence": d.confidence,
            "effect_size": d.effect_size,
            "reproducibility": d.reproducibility,
            "novelty": d.novelty,
            "supporting_experiments": d.supporting_experiments,
            "claim_ids": d.claim_ids,
            "hypothesis_id": d.hypothesis_id,
        }
        for d in run.discoveries
    ]
    # Auto-promote confirmed mechanisms to durable memory so findings compound.
    if discoveries:
        try:
            from .memory import ResearchMemory

            mem = ResearchMemory(db_path or ".mechferret/memory.sqlite")
            try:
                mem.record_mechanisms(model, discoveries)
            finally:
                mem.close()
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass
    audit = audit_run_artifact(run.artifacts.get("json"))
    experiments = _experiments_for_discoveries(run.experiments, run.discoveries, limit=24)
    return json.dumps({
        "discoveries": discoveries,
        "experiments": [
            {
                "id": experiment.id,
                "spec_id": experiment.spec_id,
                "probe": experiment.probe,
                "target": experiment.target,
                "effect_size": experiment.effect_size,
                "baseline": experiment.baseline,
                "per_seed": experiment.per_seed,
                "significant": experiment.significant,
                "reproduced": experiment.reproduced,
                "backend_used": experiment.backend_used,
                "status": experiment.status,
            }
            for experiment in experiments
        ],
        "metrics": {k: run.metrics.get(k) for k in ("rigor_score", "readiness_score", "confirmed_mechanisms", "experiments_run")},
        "provenance": {
            "answer_author": run.provenance.get("answer_author", ""),
            "answer_provider": run.provenance.get("answer_provider", ""),
            "answer_model": run.provenance.get("answer_model", ""),
            "backend_requested": run.provenance.get("backend_requested", ""),
            "backend_used": run.provenance.get("backend_used", ""),
            "provider_requested": run.provenance.get("provider_requested", ""),
            "provider_source_added": run.provenance.get("provider_source_added", False),
            "used_packaged_seed_corpus": run.provenance.get("used_packaged_seed_corpus", False),
            "budget": run.provenance.get("budget", {}),
        },
        "audit": {
            "passed": audit.get("passed", False),
            "failed_checks": audit.get("failed_checks", []),
            "advisories": audit.get("advisories", []),
            "next_actions": audit.get("next_actions", []),
        },
        "artifacts": run.artifacts,
        "report_html": run.artifacts.get("html"),
    })


def _experiments_for_discoveries(experiments, discoveries, *, limit: int):
    by_id = {experiment.id: experiment for experiment in experiments}
    selected = []
    seen: set[str] = set()
    for discovery in discoveries:
        for experiment_id in discovery.supporting_experiments:
            experiment = by_id.get(experiment_id)
            if experiment is None or experiment.id in seen:
                continue
            selected.append(experiment)
            seen.add(experiment.id)
    for experiment in sorted(experiments, key=lambda item: abs(item.effect_size), reverse=True):
        if len(selected) >= limit:
            break
        if experiment.id in seen:
            continue
        selected.append(experiment)
        seen.add(experiment.id)
    return selected[:limit]


def tool_verify_novelty(args: dict[str, Any]) -> str:
    from .knowledge import search_arxiv, web_search

    idea, invalid = _string_arg(args, "idea")
    if invalid:
        return json.dumps(invalid)
    queries, invalid = _optional_string_list_arg(args, "queries")
    if invalid:
        return json.dumps(invalid)
    plan = _novelty_search_plan(idea, queries)
    web_plan = _novelty_web_search_plan(idea, queries)
    seen: set[str] = set()
    related: list[dict[str, Any]] = []
    recent: list[dict[str, Any]] = []
    architecture: list[dict[str, Any]] = []
    web_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for item in plan:
        query = item["query"]
        sort_by = item["sort_by"]
        try:
            _, papers = search_arxiv(query, max_results=item["max_results"], sort_by=sort_by)
        except Exception as exc:  # noqa: BLE001
            papers = []
            errors.append({"source": "arxiv", "query": query, "error": str(exc)})
        for p in papers:
            row = _novelty_paper_row(p, focus=item["focus"])
            key = _novelty_paper_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            related.append(row)
            if sort_by in {"submittedDate", "lastUpdatedDate"} and len(recent) < NOVELTY_FOCUSED_LIMIT:
                recent.append(row)
            if "architecture" in item["focus"] and len(architecture) < NOVELTY_FOCUSED_LIMIT:
                architecture.append(row)
    for item in web_plan:
        query = item["query"]
        try:
            results = web_search(query, max_results=item["max_results"])
        except Exception as exc:  # noqa: BLE001
            results = []
            errors.append({"source": "web", "query": query, "error": str(exc)})
        for result in results:
            row = _novelty_web_row(result, focus=item["focus"])
            key = _novelty_paper_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            web_results.append(row)
    return json.dumps({
        "idea": idea,
        "search_plan": plan,
        "arxiv_search_plan": plan,
        "web_search_plan": web_plan,
        "related_papers": related[:NOVELTY_RELATED_LIMIT],
        "recent_papers": recent,
        "architecture_papers": architecture,
        "web_results": web_results[:NOVELTY_RELATED_LIMIT],
        "assessment": _novelty_assessment(idea, [*related, *web_results], errors),
        "errors": errors,
        "novelty_questions": _novelty_questions(idea),
        "guidance": "Do not claim high novelty unless the idea survives relevance, submitted-date, "
                    "updated-date, architecture, and discovery searches. Compare against the closest "
                    "recent papers, name the exact delta, cite likely prior art, and downgrade any "
                    "direction that only renames an existing method.",
    })


_NOVELTY_STOPWORDS = {
    "about",
    "after",
    "against",
    "already",
    "and",
    "are",
    "based",
    "between",
    "could",
    "from",
    "have",
    "into",
    "model",
    "models",
    "novel",
    "novelty",
    "paper",
    "papers",
    "research",
    "should",
    "that",
    "the",
    "their",
    "this",
    "through",
    "using",
    "with",
    "would",
}


def _novelty_search_plan(idea: str, queries: list[str] | None) -> list[dict[str, Any]]:
    seeds = _unique_strings([*(queries or []), idea])
    terms = _novelty_terms(idea)
    compact = " ".join(terms[:8]) or idea
    architecture_terms = " ".join(terms[:5]) or idea
    candidates = []
    for query in seeds[:3]:
        candidates.append(_novelty_plan_item(query, "relevance", "provided_relevance"))
        candidates.append(_novelty_plan_item(query, "submittedDate", "provided_recent_submitted"))
    candidates.extend(
        [
            _novelty_plan_item(compact, "relevance", "core_relevance"),
            _novelty_plan_item(compact, "submittedDate", "recent_submitted"),
            _novelty_plan_item(compact, "lastUpdatedDate", "recent_updated"),
            _novelty_plan_item(f"{architecture_terms} architecture transformer", "relevance", "architecture_relevance"),
            _novelty_plan_item(f"{architecture_terms} circuit sparse autoencoder", "relevance", "architecture_mechanism"),
            _novelty_plan_item(f"{architecture_terms} discovery benchmark state of the art", "submittedDate", "recent_discovery"),
        ]
    )
    plan: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        key = (item["query"].lower(), item["sort_by"])
        if key in seen:
            continue
        seen.add(key)
        plan.append(item)
    return plan[:NOVELTY_MAX_QUERY_PASSES]


def _novelty_web_search_plan(idea: str, queries: list[str] | None) -> list[dict[str, Any]]:
    seeds = _unique_strings([*(queries or []), idea])
    terms = _novelty_terms(idea)
    compact = " ".join(terms[:8]) or idea
    candidates = []
    for query in seeds[:2]:
        candidates.append(_novelty_web_plan_item(query, "provided_web_relevance"))
    candidates.extend(
        [
            _novelty_web_plan_item(f"{compact} recent paper implementation", "web_recent_implementation"),
            _novelty_web_plan_item(f"{compact} benchmark architecture discovery", "web_architecture_discovery"),
            _novelty_web_plan_item(f"{compact} github project", "web_code_prior"),
        ]
    )
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        key = item["query"].lower()
        if key in seen:
            continue
        seen.add(key)
        plan.append(item)
    return plan[:NOVELTY_WEB_MAX_QUERY_PASSES]


def _novelty_plan_item(query: str, sort_by: str, focus: str) -> dict[str, Any]:
    return {
        "query": query,
        "sort_by": sort_by,
        "max_results": NOVELTY_QUERY_RESULT_LIMIT,
        "focus": focus,
    }


def _novelty_web_plan_item(query: str, focus: str) -> dict[str, Any]:
    return {
        "query": query,
        "max_results": NOVELTY_WEB_RESULT_LIMIT,
        "focus": focus,
    }


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        key = " ".join(text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _novelty_terms(idea: str) -> list[str]:
    words = [word.lower() for word in idea.replace("-", " ").split()]
    terms: list[str] = []
    for word in words:
        cleaned = "".join(ch for ch in word if ch.isalnum())
        if len(cleaned) < 3 or cleaned in _NOVELTY_STOPWORDS:
            continue
        if cleaned not in terms:
            terms.append(cleaned)
    return terms


def _novelty_paper_row(paper: dict[str, Any], *, focus: str) -> dict[str, Any]:
    url = str(paper.get("url", "")).strip()
    return {
        "source": "arxiv",
        "source_type": "paper",
        "title": str(paper.get("title", "")).strip(),
        "url": url,
        "source_domain": _novelty_url_domain(url),
        "published": str(paper.get("published", "")).strip(),
        "abstract": str(paper.get("abstract", "")).strip()[:700],
        "authors": paper.get("authors", []) if isinstance(paper.get("authors"), list) else [],
        "focus": focus,
    }


def _novelty_web_row(result: dict[str, Any], *, focus: str) -> dict[str, Any]:
    url = str(result.get("url", "")).strip()
    abstract = result.get("snippet") or result.get("abstract") or result.get("description") or ""
    domain = str(result.get("source_domain", "")).strip() or _novelty_url_domain(url)
    source_type = _novelty_web_source_type(url=url, domain=domain, title=result.get("title", ""), abstract=abstract)
    return {
        "source": "web",
        "source_type": source_type,
        "title": str(result.get("title", "")).strip(),
        "url": url,
        "source_domain": domain,
        "published": "",
        "abstract": str(abstract).strip()[:700],
        "authors": [],
        "focus": focus,
    }


def _novelty_paper_key(row: dict[str, Any]) -> str:
    title = " ".join(str(row.get("title", "")).lower().split())
    url = " ".join(str(row.get("url", "")).lower().split())
    return title or url


def _novelty_assessment(idea: str, rows: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    terms = _novelty_terms(idea)
    scored = [_novelty_scored_prior(row, terms) for row in rows]
    scored.sort(key=lambda row: row["score"], reverse=True)
    closest = scored[:NOVELTY_CLOSEST_PRIOR_LIMIT]
    top_score = closest[0]["score"] if closest else 0.0
    arxiv_count = sum(1 for row in rows if row.get("source") == "arxiv")
    web_count = sum(1 for row in rows if row.get("source") == "web")
    web_source_types = _novelty_web_source_type_counts(rows)
    if not rows and errors:
        risk = "unknown_search_incomplete"
        verdict = "Novelty is not assessable because one or more retrieval passes failed."
    elif not rows:
        risk = "unresolved_no_close_prior_found"
        verdict = "No close prior art was found by this search plan; treat novelty as unresolved until expert review."
    elif top_score >= 0.55:
        risk = "high_prior_art_risk"
        verdict = "Closest retrieved papers appear to share core terms or mechanisms; novelty claim needs a very specific delta."
    elif top_score >= 0.25 or len(rows) >= NOVELTY_RELATED_LIMIT:
        risk = "medium_prior_art_risk"
        verdict = "Related work exists; novelty depends on the exact method, evaluation, and mechanism difference."
    else:
        risk = "low_prior_art_risk"
        verdict = "Retrieved papers look adjacent rather than directly overlapping; novelty still needs expert verification."
    return {
        "risk": risk,
        "verdict": verdict,
        "closest_prior_art": closest,
        "coverage": {
            "retrieved_evidence": len(rows),
            "retrieved_papers": arxiv_count,
            "web_results": web_count,
            "web_results_with_snippets": sum(1 for row in rows if row.get("source") == "web" and row.get("abstract")),
            "web_source_types": web_source_types,
            "failed_queries": len(errors),
            "failed_arxiv_queries": sum(1 for error in errors if error.get("source") == "arxiv"),
            "failed_web_queries": sum(1 for error in errors if error.get("source") == "web"),
            "idea_terms": terms,
            "recent_window": _novelty_recent_window_label(),
        },
        "required_delta": [
            "Name the nearest prior paper and the exact method component that differs.",
            "Show a benchmark, ablation, or causal test where the idea behaves differently.",
            "Downgrade novelty if the contribution is only a new wording of an existing architecture or probe.",
        ],
    }


def _novelty_scored_prior(row: dict[str, Any], terms: list[str]) -> dict[str, Any]:
    haystack = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
    matched = [term for term in terms if term in haystack]
    term_score = len(matched) / max(1, len(terms))
    focus = str(row.get("focus", ""))
    focus_score = 0.15 if any(key in focus for key in ("architecture", "mechanism", "relevance")) else 0.0
    recent_score = 0.1 if _novelty_is_recent(row.get("published", "")) else 0.0
    source_type = str(row.get("source_type", "general_web"))
    source_score = _novelty_source_type_score(source_type)
    score = min(1.0, round(term_score * 0.7 + focus_score + recent_score + source_score, 3))
    return {
        "source": row.get("source", ""),
        "source_type": source_type,
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "source_domain": row.get("source_domain", ""),
        "published": row.get("published", ""),
        "focus": focus,
        "score": score,
        "source_score": source_score,
        "matched_terms": matched,
        "evidence_excerpt": str(row.get("abstract", ""))[:240],
        "reason": _novelty_prior_reason(matched, focus, row.get("published", ""), row.get("source", ""), source_type),
    }


def _novelty_prior_reason(matched: list[str], focus: str, published: Any, source: Any, source_type: Any) -> str:
    bits = []
    if matched:
        bits.append("shares idea terms: " + ", ".join(matched[:6]))
    if source == "web":
        bits.append("retrieved from web search")
    if source_type in {"paper", "benchmark", "code_repository", "project_page", "documentation"}:
        bits.append(f"source type: {source_type}")
    if "architecture" in focus:
        bits.append("retrieved by architecture-focused search")
    if "mechanism" in focus:
        bits.append("retrieved by mechanism-focused search")
    if _novelty_is_recent(published):
        bits.append("within the recent-paper window")
    return "; ".join(bits) or "retrieved as adjacent prior art"


def _novelty_is_recent(published: Any) -> bool:
    year = _novelty_year(published)
    return year is not None and year >= datetime.now(UTC).year - 2


def _novelty_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if len(text) < 4:
        return None
    try:
        return int(text[:4])
    except ValueError:
        return None


def _novelty_url_domain(url: str) -> str:
    host = urlparse(str(url)).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _novelty_web_source_type(*, url: Any, domain: Any, title: Any, abstract: Any) -> str:
    url_text = str(url or "").lower()
    domain_text = str(domain or "").lower()
    haystack = f"{url_text} {domain_text} {title or ''} {abstract or ''}".lower()
    if any(key in domain_text for key in ("arxiv.org", "openreview.net", "aclanthology.org", "proceedings.mlr.press", "papers.nips.cc", "semanticscholar.org")):
        return "paper"
    if any(key in domain_text for key in ("paperswithcode.com", "benchmark", "evals")) or any(key in haystack for key in ("benchmark", "leaderboard", "evaluation suite")):
        return "benchmark"
    if any(key in domain_text for key in ("github.com", "gitlab.com", "bitbucket.org", "huggingface.co")) or any(key in haystack for key in ("repository", "implementation", "source code")):
        return "code_repository"
    if any(key in domain_text for key in ("readthedocs.io", "docs.", "documentation")) or any(key in haystack for key in ("api reference", "documentation")):
        return "documentation"
    if any(key in haystack for key in ("project page", "demo", "dataset page", "technical report")):
        return "project_page"
    return "general_web"


def _novelty_source_type_score(source_type: str) -> float:
    return {
        "paper": 0.06,
        "benchmark": 0.05,
        "code_repository": 0.04,
        "project_page": 0.035,
        "documentation": 0.02,
        "general_web": 0.0,
    }.get(source_type, 0.0)


def _novelty_web_source_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {source_type: 0 for source_type in sorted(NOVELTY_WEB_SOURCE_TYPES)}
    for row in rows:
        if row.get("source") != "web":
            continue
        source_type = str(row.get("source_type", "general_web"))
        if source_type not in counts:
            source_type = "general_web"
        counts[source_type] += 1
    return {key: value for key, value in counts.items() if value}


def _novelty_recent_window_label() -> str:
    year = datetime.now(UTC).year
    return f"{year - 2}-{year}"


def _novelty_questions(idea: str) -> list[str]:
    terms = ", ".join(_novelty_terms(idea)[:5]) or "the core mechanism"
    return [
        f"Which recent papers already combine {terms}?",
        "What is the nearest architecture or training-pipeline ancestor, and what exact component changes?",
        "Which benchmark, ablation, or negative result would distinguish this from adjacent work?",
        "Does the contribution depend on a new mechanism, a new measurement, or only a new application domain?",
    ]


def tool_present_options(args: dict[str, Any]) -> str:
    # Headless fallback; the REPL intercepts this for an interactive picker.
    options, invalid = _object_list_arg(args, "options")
    if invalid:
        return json.dumps(invalid)
    titles = []
    for index, option in enumerate(options):
        invalid = _validate_option_card(option, index)
        if invalid:
            return json.dumps(invalid)
        titles.append(str(option.get("title", "")).strip())
    return json.dumps({"options": titles, "option_details": [_option_detail(option) for option in options]})


def _validate_option_card(option: dict[str, Any], index: int) -> dict[str, Any] | None:
    for key in ("title", "summary", "detail", "novelty_verdict", "required_delta"):
        value = option.get(key)
        if not isinstance(value, str) or not value.strip():
            return _invalid_object_list_payload(
                "options",
                value,
                index=index,
                expected=f"objects with non-empty string {key}",
            )
    risk = option.get("novelty_risk")
    if not isinstance(risk, str) or risk.strip() not in NOVELTY_RISKS:
        return _invalid_object_list_payload(
            "options",
            risk,
            index=index,
            expected="objects with novelty_risk from verify_novelty assessment",
        )
    citations = _option_strings(option.get("citations", []))
    if not citations:
        return _invalid_object_list_payload(
            "options",
            option.get("citations"),
            index=index,
            expected="objects with non-empty citations list",
        )
    if not isinstance(option.get("closest_prior_art"), list):
        return _invalid_object_list_payload(
            "options",
            option.get("closest_prior_art"),
            index=index,
            expected="objects with closest_prior_art list from verify_novelty assessment",
        )
    return None


def _option_detail(option: dict[str, Any]) -> dict[str, Any]:
    detail = {
        "title": str(option.get("title", "")).strip(),
        "summary": str(option.get("summary", "")).strip(),
        "detail": str(option.get("detail", "")).strip(),
    }
    for key in ("novelty_risk", "novelty_verdict", "novelty", "required_delta"):
        value = option.get(key)
        if isinstance(value, str) and value.strip():
            detail[key] = value.strip()
    citations = _option_strings(option.get("citations", []))[:4]
    if citations:
        detail["citations"] = citations
    closest = _option_strings(option.get("closest_prior_art", []))[:3]
    if closest:
        detail["closest_prior_art"] = closest
    return detail


def _option_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def tool_list_skills(_args: dict[str, Any]) -> str:
    from .skills import list_skills

    return json.dumps([{"name": s.name, "task": s.task, "description": s.description} for s in list_skills()])


def tool_environment_status(_args: dict[str, Any]) -> str:
    from .cluster import load_cluster_config
    from .modal_app import modal_status
    from .skills import list_skills

    return json.dumps({
        "skills": [s.name for s in list_skills()],
        "modal": modal_status(),
        "cluster_configured": load_cluster_config().configured,
    })


def tool_project_status(args: dict[str, Any]) -> str:
    from .ops import project_status

    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    db_path, invalid = _optional_string_arg(args, "db_path", ".mechferret/memory.sqlite")
    if invalid:
        return json.dumps(invalid)
    notes_root, invalid = _optional_string_arg(args, "notes_root", ".")
    if invalid:
        return json.dumps(invalid)
    project_root, invalid = _optional_string_arg(args, "project_root", "projects/openvla_sae")
    if invalid:
        return json.dumps(invalid)
    return json.dumps(
        project_status(
            runs_root=runs_root,
            db_path=db_path,
            notes_root=notes_root,
            project_root=project_root,
            selection=selection,
        )
    )


def tool_list_runs(args: dict[str, Any]) -> str:
    from .ops import list_run_artifacts

    selection, invalid = _selection_arg(args, "best")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    limit, invalid = _int_arg(args, "limit", 10, min_value=0)
    if invalid:
        return json.dumps(invalid)
    no_audit, invalid = _bool_arg(args, "no_audit", False)
    if invalid:
        return json.dumps(invalid)
    return json.dumps(
        list_run_artifacts(
            runs_root=runs_root,
            limit=limit,
            include_audit=not no_audit,
            selection=selection,
        )
    )


SELECTION_POLICIES = {"latest", "best", "ready"}
PROVIDERS = {"anthropic", "auto", "local", "openai"}
REVIEW_PROVIDERS = {"anthropic", "auto", "openai"}
ARXIV_SORTS = {"lastUpdatedDate", "relevance", "submittedDate"}
DISCOVERY_BACKENDS = {"auto", "real", "synthetic", "tl", "transformer_lens"}
OPENVLA_ACTIONS = {
    "commands",
    "create-manifest",
    "dossier",
    "eval",
    "features",
    "init",
    "plan",
    "smoke",
    "status",
    "validate-manifest",
}


def _string_arg(args: dict[str, Any], name: str, default: str | None = None) -> tuple[str, dict[str, Any] | None]:
    raw = args.get(name, default)
    if isinstance(raw, str) and raw.strip():
        return raw, None
    return default or "", _invalid_string_payload(name, raw, expected="non-empty string")


def _optional_string_arg(
    args: dict[str, Any],
    name: str,
    default: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    raw = args.get(name, default)
    if raw in (None, ""):
        return default, None
    if isinstance(raw, str):
        return raw, None
    return default, _invalid_string_payload(name, raw, expected="string")


def _optional_string_list_arg(
    args: dict[str, Any],
    name: str,
) -> tuple[list[str] | None, dict[str, Any] | None]:
    raw = args.get(name)
    if raw in (None, ""):
        return None, None
    if not isinstance(raw, list):
        return None, _invalid_string_list_payload(name, raw, index=None)
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            return None, _invalid_string_list_payload(name, item, index=index)
        values.append(item)
    return values or None, None


def _object_list_arg(
    args: dict[str, Any],
    name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    raw = args.get(name)
    if not isinstance(raw, list):
        return [], _invalid_object_list_payload(name, raw, index=None)
    values: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            return [], _invalid_object_list_payload(name, item, index=index)
        values.append(item)
    return values, None


def _invalid_string_payload(name: str, raw: Any, *, expected: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": f"invalid string argument: {name}",
        "argument": name,
        "value": raw if isinstance(raw, (str, int, float, bool, type(None))) else type(raw).__name__,
        "expected": expected,
        "failed_checks": [f"{name}_argument"],
        "next_actions": [f"Pass `{name}` as {expected}."],
    }


def _invalid_string_list_payload(name: str, raw: Any, *, index: int | None) -> dict[str, Any]:
    payload = {
        "ok": False,
        "error": f"invalid string list argument: {name}",
        "argument": name,
        "value": raw if isinstance(raw, (str, int, float, bool, type(None))) else type(raw).__name__,
        "expected": "list of non-empty strings",
        "failed_checks": [f"{name}_argument"],
        "next_actions": [f"Pass `{name}` as a list of non-empty strings."],
    }
    if index is not None:
        payload["index"] = index
    return payload


def _invalid_object_list_payload(
    name: str,
    raw: Any,
    *,
    index: int | None,
    expected: str = "list of objects",
) -> dict[str, Any]:
    payload = {
        "ok": False,
        "error": f"invalid object list argument: {name}",
        "argument": name,
        "value": raw if isinstance(raw, (str, int, float, bool, type(None))) else type(raw).__name__,
        "expected": expected,
        "failed_checks": [f"{name}_argument"],
        "next_actions": [f"Pass `{name}` as {expected}."],
    }
    if index is not None:
        payload["index"] = index
    return payload


def _bool_arg(args: dict[str, Any], name: str, default: bool) -> tuple[bool, dict[str, Any] | None]:
    raw = args.get(name, default)
    if raw in (None, ""):
        raw = default
    if type(raw) is bool:
        return raw, None
    return default, {
        "ok": False,
        "error": f"invalid boolean argument: {name}",
        "argument": name,
        "value": raw if isinstance(raw, (str, int, float, bool, type(None))) else type(raw).__name__,
        "expected": "boolean",
        "failed_checks": [f"{name}_argument"],
        "next_actions": [f"Pass `{name}` as true or false."],
    }


def _enum_arg(
    args: dict[str, Any],
    name: str,
    default: str,
    allowed: set[str],
) -> tuple[str, dict[str, Any] | None]:
    raw = args.get(name, default)
    if raw in (None, ""):
        raw = default
    if isinstance(raw, str) and raw in allowed:
        return raw, None
    return default, {
        "ok": False,
        "error": f"invalid option argument: {name}",
        "argument": name,
        "value": raw if isinstance(raw, (str, int, float, bool, type(None))) else type(raw).__name__,
        "allowed_values": sorted(allowed),
        "failed_checks": [f"{name}_argument"],
        "next_actions": [f"Use one of: {', '.join(sorted(allowed))}."],
    }


def _int_arg(
    args: dict[str, Any],
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> tuple[int, dict[str, Any] | None]:
    raw = args.get(name, default)
    if raw in (None, ""):
        raw = default
    try:
        if type(raw) is int:
            value = raw
        elif isinstance(raw, str):
            value = int(raw)
        else:
            raise TypeError(type(raw).__name__)
    except (TypeError, ValueError):
        return default, _invalid_int_payload(name, raw, min_value=min_value, max_value=max_value)
    if min_value is not None and value < min_value:
        return default, _invalid_int_payload(name, raw, min_value=min_value, max_value=max_value)
    if max_value is not None and value > max_value:
        return default, _invalid_int_payload(name, raw, min_value=min_value, max_value=max_value)
    return value, None


def _invalid_int_payload(
    name: str,
    raw: Any,
    *,
    min_value: int | None,
    max_value: int | None,
) -> dict[str, Any]:
    if min_value is None and max_value is None:
        expected = "integer"
    elif max_value is None:
        expected = f"integer >= {min_value}"
    elif min_value is None:
        expected = f"integer <= {max_value}"
    else:
        expected = f"integer between {min_value} and {max_value}"
    return {
        "ok": False,
        "error": f"invalid integer argument: {name}",
        "argument": name,
        "value": raw if isinstance(raw, (str, int, float, bool, type(None))) else type(raw).__name__,
        "expected": expected,
        "failed_checks": [f"{name}_argument"],
        "next_actions": [f"Pass `{name}` as {expected}."],
    }


def _float_arg(
    args: dict[str, Any],
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[float, dict[str, Any] | None]:
    raw = args.get(name, default)
    if raw in (None, ""):
        raw = default
    try:
        if type(raw) in {int, float}:
            value = float(raw)
        elif isinstance(raw, str):
            value = float(raw)
        else:
            raise TypeError(type(raw).__name__)
    except (TypeError, ValueError):
        return default, _invalid_float_payload(name, raw, min_value=min_value, max_value=max_value)
    if not math.isfinite(value):
        return default, _invalid_float_payload(name, raw, min_value=min_value, max_value=max_value)
    if min_value is not None and value < min_value:
        return default, _invalid_float_payload(name, raw, min_value=min_value, max_value=max_value)
    if max_value is not None and value > max_value:
        return default, _invalid_float_payload(name, raw, min_value=min_value, max_value=max_value)
    return value, None


def _invalid_float_payload(
    name: str,
    raw: Any,
    *,
    min_value: float | None,
    max_value: float | None,
) -> dict[str, Any]:
    if min_value is None and max_value is None:
        expected = "number"
    elif max_value is None:
        expected = f"number >= {min_value:g}"
    elif min_value is None:
        expected = f"number <= {max_value:g}"
    else:
        expected = f"number between {min_value:g} and {max_value:g}"
    return {
        "ok": False,
        "error": f"invalid number argument: {name}",
        "argument": name,
        "value": raw if isinstance(raw, (str, int, float, bool, type(None))) else type(raw).__name__,
        "expected": expected,
        "failed_checks": [f"{name}_argument"],
        "next_actions": [f"Pass `{name}` as {expected}."],
    }


def _selection_arg(args: dict[str, Any], default: str) -> tuple[str, dict[str, Any] | None]:
    raw = args.get("selection") if "selection" in args else args.get("select", default)
    if raw in (None, ""):
        raw = default
    if isinstance(raw, str) and raw in SELECTION_POLICIES:
        return raw, None
    return default, {
        "ok": False,
        "error": "invalid selection policy",
        "selection": raw if isinstance(raw, str) else type(raw).__name__,
        "allowed_selection": sorted(SELECTION_POLICIES),
        "failed_checks": ["selection_policy"],
        "next_actions": ["Use one of: latest, best, ready."],
    }


def _run_selection_failure_payload(
    selected: dict[str, Any],
    *,
    selection: str,
    runs_root: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "path": "",
        "selection": selection,
        "runs_root": str(runs_root),
        "selected_path": selected.get("path", ""),
        "failed_checks": ["run_selection"],
        "next_actions": selected.get("next_actions", []),
    }
    if extra:
        payload.update(extra)
    return payload


def tool_audit_run(args: dict[str, Any]) -> str:
    from .audit import audit_run_artifact
    from .ops import select_run_artifact

    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    path, invalid = _optional_string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    if path is None and selection != "latest":
        selected = select_run_artifact(runs_root=runs_root, policy=selection)
        path = selected.get("path") or None
        if path is None:
            return json.dumps(
                _run_selection_failure_payload(
                    selected,
                    selection=selection,
                    runs_root=runs_root,
                    extra={"passed": False},
                )
            )
    return json.dumps(audit_run_artifact(path, runs_root=runs_root))


def tool_verify_run(args: dict[str, Any]) -> str:
    from .ops import select_run_artifact, verify_run_artifacts

    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    path, invalid = _optional_string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    repair, invalid = _bool_arg(args, "repair", False)
    if invalid:
        return json.dumps(invalid)
    if path is None and selection != "latest":
        selected = select_run_artifact(runs_root=runs_root, policy=selection)
        path = selected.get("path") or None
        if path is None:
            return json.dumps(
                _run_selection_failure_payload(
                    selected,
                    selection=selection,
                    runs_root=runs_root,
                    extra={"manifest": "", "passed": False},
                )
            )
    return json.dumps(verify_run_artifacts(path, runs_root=runs_root, repair=repair))


def tool_write_paper(args: dict[str, Any]) -> str:
    from .paper import write_paper_from_artifact
    from .ops import select_run_artifact

    provider, invalid = _enum_arg(args, "provider", "auto", PROVIDERS)
    if invalid:
        return json.dumps(invalid)
    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    path, invalid = _optional_string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    out_dir, invalid = _optional_string_arg(args, "out_dir")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    model, invalid = _optional_string_arg(args, "model")
    if invalid:
        return json.dumps(invalid)
    compile_pdf, invalid = _bool_arg(args, "compile", False)
    if invalid:
        return json.dumps(invalid)
    compile_timeout, invalid = _int_arg(args, "compile_timeout", 60, min_value=1)
    if invalid:
        return json.dumps(invalid)
    if path is None and selection != "latest":
        selected = select_run_artifact(runs_root=runs_root, policy=selection)
        path = selected.get("path") or None
        if path is None:
            return json.dumps(
                _run_selection_failure_payload(
                    selected,
                    selection=selection,
                    runs_root=runs_root,
                    extra={"ok": False, "error": "no run selected"},
                )
            )

    return json.dumps(
        write_paper_from_artifact(
            path,
            out_dir=out_dir,
            compile_pdf=compile_pdf,
            compile_timeout=compile_timeout,
            provider=provider,
            model=model,
        )
    )


def tool_review_paper(args: dict[str, Any]) -> str:
    from .paper import review_paper

    provider, invalid = _enum_arg(args, "provider", "auto", REVIEW_PROVIDERS)
    if invalid:
        return json.dumps(invalid)
    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    path, invalid = _optional_string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    out_dir, invalid = _optional_string_arg(args, "out_dir")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    model, invalid = _optional_string_arg(args, "model")
    if invalid:
        return json.dumps(invalid)
    return json.dumps(
        review_paper(
            path,
            out_dir=out_dir,
            provider=provider,
            model=model,
            runs_root=runs_root,
            selection=selection,
        )
    )


def tool_bundle_artifacts(args: dict[str, Any]) -> str:
    from .ops import bundle_run_artifacts

    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    path, invalid = _optional_string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    out, invalid = _optional_string_arg(args, "out")
    if invalid:
        return json.dumps(invalid)
    notes_root, invalid = _optional_string_arg(args, "notes_root", ".")
    if invalid:
        return json.dumps(invalid)
    project_root, invalid = _optional_string_arg(args, "project_root", "projects/openvla_sae")
    if invalid:
        return json.dumps(invalid)
    result = bundle_run_artifacts(
        path,
        runs_root=runs_root,
        selection=selection,
        out=out,
        notes_root=notes_root,
        project_root=project_root,
    )
    verification = result.get("bundle_verification")
    if isinstance(verification, dict):
        result["bundle_verification"] = {
            "path": verification.get("path", ""),
            "passed": verification.get("passed", False),
            "failed_checks": verification.get("failed_checks", []),
            "next_actions": verification.get("next_actions", []),
            "check_count": len(verification.get("checks", [])),
        }
    return json.dumps(result)


def tool_verify_bundle(args: dict[str, Any]) -> str:
    from .ops import verify_bundle_artifacts

    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    path, invalid = _optional_string_arg(args, "path")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    return json.dumps(
        verify_bundle_artifacts(
            path,
            runs_root=runs_root,
            selection=selection,
        )
    )


def tool_resolve_artifact(args: dict[str, Any]) -> str:
    from .ops import resolve_artifact

    selection, invalid = _selection_arg(args, "latest")
    if invalid:
        return json.dumps(invalid)
    target, invalid = _optional_string_arg(args, "target", "quickstart")
    if invalid:
        return json.dumps(invalid)
    runs_root, invalid = _optional_string_arg(args, "runs_root", "runs")
    if invalid:
        return json.dumps(invalid)
    project_root, invalid = _optional_string_arg(args, "project_root", "projects/openvla_sae")
    if invalid:
        return json.dumps(invalid)
    return json.dumps(
        resolve_artifact(
            target or "quickstart",
            runs_root=runs_root,
            project_root=project_root,
            selection=selection,
        )
    )


def tool_openvla_sae(args: dict[str, Any]) -> str:
    from .openvla_sae import command_lines, create_manifest, evaluate_artifacts, feature_report, init_project, smoke_test, status, validate_manifest, write_dossier, write_plan

    action, invalid = _enum_arg(args, "action", "status", OPENVLA_ACTIONS)
    if invalid:
        return json.dumps(invalid)
    project_root, invalid = _optional_string_arg(args, "project_root", "projects/openvla_sae")
    if invalid:
        return json.dumps(invalid)
    manifest, invalid = _optional_string_arg(args, "manifest")
    if invalid:
        return json.dumps(invalid)
    if action == "status":
        return json.dumps(status(project_root=project_root, manifest=manifest))
    if action == "init":
        force, invalid = _bool_arg(args, "force", False)
        if invalid:
            return json.dumps(invalid)
        return json.dumps(init_project(project_root, force=force))
    if action == "plan":
        out_dir, invalid = _optional_string_arg(args, "out_dir", "runs/openvla_sae/plan")
        if invalid:
            return json.dumps(invalid)
        return json.dumps(write_plan(out_dir=out_dir, project_root=project_root, manifest=manifest))
    if action == "commands":
        return command_lines(project_root)
    if action == "validate-manifest":
        manifest, invalid = _string_arg(args, "manifest")
        if invalid:
            return json.dumps(invalid)
        return json.dumps(validate_manifest(manifest))
    if action == "create-manifest":
        image_dir, invalid = _string_arg(args, "image_dir")
        if invalid:
            return json.dumps(invalid)
        instruction, invalid = _optional_string_arg(args, "instruction", "perform the task shown in the image")
        if invalid:
            return json.dumps(invalid)
        action_label, invalid = _optional_string_arg(args, "action_label", "")
        if invalid:
            return json.dumps(invalid)
        limit = None
        if args.get("limit") not in (None, ""):
            limit, invalid = _int_arg(args, "limit", 0, min_value=1)
            if invalid:
                return json.dumps(invalid)
        force, invalid = _bool_arg(args, "force", False)
        if invalid:
            return json.dumps(invalid)
        try:
            return json.dumps(
                create_manifest(
                    image_dir,
                    manifest or "data/openvla_sae_phase1.jsonl",
                    instruction=instruction or "perform the task shown in the image",
                    action=action_label or "",
                    limit=limit,
                    force=force,
                )
            )
        except FileNotFoundError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "failed_checks": ["image_dir_exists"],
                    "next_actions": ["Pass `image_dir` as a directory containing OpenVLA frame images."],
                }
            )
        except FileExistsError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "failed_checks": ["manifest_overwrite"],
                    "next_actions": ["Pass `force=true` to overwrite the existing manifest."],
                }
            )
    if action == "smoke":
        out_dir, invalid = _optional_string_arg(args, "out_dir", "runs/openvla_sae/smoke")
        if invalid:
            return json.dumps(invalid)
        d_model, invalid = _int_arg(args, "d_model", 32, min_value=1)
        if invalid:
            return json.dumps(invalid)
        tokens, invalid = _int_arg(args, "tokens", 256, min_value=1)
        if invalid:
            return json.dumps(invalid)
        steps, invalid = _int_arg(args, "steps", 20, min_value=1)
        if invalid:
            return json.dumps(invalid)
        k, invalid = _int_arg(args, "k", 4, min_value=1)
        if invalid:
            return json.dumps(invalid)
        return json.dumps(
            smoke_test(
                out_dir=out_dir,
                d_model=d_model,
                tokens=tokens,
                steps=steps,
                k=k,
            )
        )
    if action == "eval":
        cache_dir, invalid = _optional_string_arg(args, "cache_dir", "runs/openvla_sae/cache_l24")
        if invalid:
            return json.dumps(invalid)
        checkpoint, invalid = _optional_string_arg(args, "checkpoint", "runs/openvla_sae/sae_l24_topk.pt")
        if invalid:
            return json.dumps(invalid)
        out_dir, invalid = _optional_string_arg(args, "out_dir", "runs/openvla_sae/eval")
        if invalid:
            return json.dumps(invalid)
        return json.dumps(
            evaluate_artifacts(
                cache_dir=cache_dir,
                checkpoint=checkpoint,
                out_dir=out_dir,
            )
        )
    if action == "features":
        cache_dir, invalid = _optional_string_arg(args, "cache_dir", "runs/openvla_sae/cache_l24")
        if invalid:
            return json.dumps(invalid)
        checkpoint, invalid = _optional_string_arg(args, "checkpoint", "runs/openvla_sae/sae_l24_topk.pt")
        if invalid:
            return json.dumps(invalid)
        out_dir, invalid = _optional_string_arg(args, "out_dir", "runs/openvla_sae/features")
        if invalid:
            return json.dumps(invalid)
        top_features, invalid = _int_arg(args, "top_features", 20, min_value=1)
        if invalid:
            return json.dumps(invalid)
        max_files, invalid = _int_arg(args, "max_files", 64, min_value=1)
        if invalid:
            return json.dumps(invalid)
        return json.dumps(
            feature_report(
                cache_dir=cache_dir,
                checkpoint=checkpoint,
                out_dir=out_dir,
                top_k=top_features,
                max_files=max_files,
            )
        )
    if action == "dossier":
        out_dir, invalid = _optional_string_arg(args, "out_dir", "runs/openvla_sae/dossier")
        if invalid:
            return json.dumps(invalid)
        cache_dir, invalid = _optional_string_arg(args, "cache_dir", "runs/openvla_sae/cache_l24")
        if invalid:
            return json.dumps(invalid)
        checkpoint, invalid = _optional_string_arg(args, "checkpoint", "runs/openvla_sae/sae_l24_topk.pt")
        if invalid:
            return json.dumps(invalid)
        eval_dir, invalid = _optional_string_arg(args, "eval_dir", "runs/openvla_sae/eval")
        if invalid:
            return json.dumps(invalid)
        features_dir, invalid = _optional_string_arg(args, "features_dir", "runs/openvla_sae/features")
        if invalid:
            return json.dumps(invalid)
        return json.dumps(
            write_dossier(
                out_dir=out_dir,
                project_root=project_root,
                manifest=manifest,
                cache_dir=cache_dir,
                checkpoint=checkpoint,
                eval_dir=eval_dir,
                features_dir=features_dir,
            )
        )
    return json.dumps(
        {
            "ok": False,
            "error": f"unhandled OpenVLA action: {action}",
            "failed_checks": ["action_dispatch"],
        }
    )


# --- registry -----------------------------------------------------------------------

def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "bash", "description": "Run a shell command in the working directory and return its output. Use for running code, tests, git, installing deps, launching experiments.",
     "parameters": _obj({"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds (default 120)"}}, ["command"])},
    {"name": "read_file", "description": "Read a UTF-8 text file with line numbers.",
     "parameters": _obj({"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"])},
    {"name": "list_tool_results", "description": "List saved large tool outputs with paths, sizes, ages, and JSON detection.",
     "parameters": _obj({"limit": {"type": "integer"}}, [])},
    {"name": "clean_tool_results", "description": "Delete stale saved large tool outputs. Defaults to a dry run; set confirm=true to delete.",
     "parameters": _obj({"keep_latest": {"type": "integer"}, "max_age_days": {"type": "number"}, "dry_run": {"type": "boolean"}, "confirm": {"type": "boolean"}}, [])},
    {"name": "write_file", "description": "Create or overwrite a file with the given content.",
     "parameters": _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])},
    {"name": "edit_file", "description": "Replace an exact string in a file. Set replace_all to replace every occurrence.",
     "parameters": _obj({"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, ["path", "old_string", "new_string"])},
    {"name": "list_dir", "description": "List a directory's contents.",
     "parameters": _obj({"path": {"type": "string"}}, [])},
    {"name": "glob", "description": "Find files matching a glob pattern (e.g. '**/*.py').",
     "parameters": _obj({"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"])},
    {"name": "grep", "description": "Search file contents by regex (ripgrep if available). Optional glob filter.",
     "parameters": _obj({"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}, ["pattern"])},
    {"name": "web_search", "description": "Search the web (returns title + url results). Use for current information and finding sources.",
     "parameters": _obj({"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"])},
    {"name": "web_fetch", "description": "Fetch a URL and return its readable text content.",
     "parameters": _obj({"url": {"type": "string"}}, ["url"])},
    {"name": "arxiv_search", "description": "Search arXiv for papers. Defaults to 20 results. sort_by: relevance | submittedDate | lastUpdatedDate. Use for literature grounding.",
     "parameters": _obj({"query": {"type": "string", "description": "arXiv query, e.g. 'cat:cs.LG AND (abs:sparse autoencoder OR abs:linear probe)'"}, "max_results": {"type": "integer"}, "sort_by": {"type": "string", "enum": ["relevance", "submittedDate", "lastUpdatedDate"]}}, ["query"])},
    {"name": "neuronpedia_search", "description": "Semantic search over SAE-feature explanations for an explicit Neuronpedia model id.",
     "parameters": _obj({"model_id": {"type": "string"}, "query": {"type": "string"}}, ["model_id", "query"])},
    {"name": "verify_novelty", "description": "Deep novelty check for a research idea using multi-pass arXiv searches across relevance, recency, architecture, and recent-discovery angles. Call this for each proposed research direction before presenting it.",
     "parameters": _obj({"idea": {"type": "string", "description": "the research idea/direction to novelty-check"}, "queries": {"type": "array", "items": {"type": "string"}, "description": "optional arXiv queries to probe for prior work"}}, ["idea"])},
    {"name": "present_options", "description": "Present 2-5 research directions to the user as an interactive, expandable picker and return their choice. Use this instead of writing options as prose. Every option must include detail, citations, novelty_risk, novelty_verdict, closest_prior_art, and required_delta from verify_novelty assessment.",
     "parameters": _obj({"options": {"type": "array", "items": {"type": "object", "properties": {
         "title": {"type": "string"},
         "summary": {"type": "string", "description": "one line"},
         "detail": {"type": "string", "description": "a fuller paragraph: what the project is about"},
         "citations": {"type": "array", "items": {"type": "string"}, "description": "key paper titles/URLs"},
         "novelty": {"type": "string", "description": "legacy novelty summary from verify_novelty"},
         "novelty_risk": {"type": "string", "description": "assessment.risk from verify_novelty"},
         "novelty_verdict": {"type": "string", "description": "assessment.verdict from verify_novelty"},
         "closest_prior_art": {"type": "array", "items": {"type": "string"}, "description": "nearest prior paper titles/URLs from assessment.closest_prior_art"},
         "required_delta": {"type": "string", "description": "specific delta needed to justify novelty"},
     }, "required": ["title", "summary", "detail", "citations", "novelty_risk", "novelty_verdict", "closest_prior_art", "required_delta"]}}}, ["options"])},
    {"name": "run_research", "description": "Run the general prompt-to-dossier literature/research pipeline over explicit sources, URLs, memory, or a configured provider.",
     "parameters": _obj({
         "question": {"type": "string"},
         "source_paths": {"type": "array", "items": {"type": "string"}, "description": "Local source files/directories for project-specific evidence."},
         "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to fetch as evidence sources."},
         "out_dir": {"type": "string", "description": "Directory for run.json, report.html, report.md, graph.json, evals.json, trace.jsonl, and manifest.json."},
         "db_path": {"type": "string", "description": "SQLite memory path."},
         "max_rounds": {"type": "integer"},
         "provider": {"type": "string", "enum": ["auto", "local", "openai", "anthropic"]},
         "model": {"type": "string"},
         "include_memory": {"type": "boolean"},
         "no_memory": {"type": "boolean"},
         "allow_seed_corpus": {"type": "boolean", "description": "Opt in to the packaged demo corpus when no explicit sources, memory, or provider research are available."},
     }, ["question"])},
    {"name": "run_discovery", "description": "Run the autonomous interpretability discovery loop to find/confirm mechanisms (heads/circuits) for a behaviour.",
     "parameters": _obj({
         "question": {"type": "string"},
         "skill": {"type": "string", "description": "Skill name from `mechferret skills`, or a path to a skill JSON."},
         "task": {"type": "string", "enum": ["ioi", "induction", "greater_than", "factual_recall"]},
         "model": {"type": "string"},
         "backend": {"type": "string", "enum": ["auto", "synthetic", "transformer_lens", "tl", "real"]},
         "source_paths": {"type": "array", "items": {"type": "string"}, "description": "Optional local source files for literature grounding."},
         "urls": {"type": "array", "items": {"type": "string"}, "description": "Optional URLs for literature grounding."},
         "out_dir": {"type": "string", "description": "Directory for run.json, report.html, trace.jsonl, and related artifacts."},
         "db_path": {"type": "string", "description": "SQLite memory path."},
         "max_rounds": {"type": "integer"},
         "max_experiments": {"type": "integer"},
         "max_gpu_seconds": {"type": "number"},
         "provider": {"type": "string", "enum": ["auto", "local", "openai", "anthropic"]},
         "llm_model": {"type": "string"},
         "include_memory": {"type": "boolean"},
         "no_memory": {"type": "boolean"},
         "allow_mismatch": {"type": "boolean"},
         "seed_corpus": {"type": "boolean", "description": "Alias for allow_seed_corpus."},
         "allow_seed_corpus": {"type": "boolean", "description": "Opt in to the packaged demo corpus when no explicit sources, memory, or provider research are available."},
     }, [])},
    {"name": "list_skills", "description": "List available interpretability playbooks/skills.", "parameters": _obj({}, [])},
    {"name": "environment_status", "description": "Report skills, Modal status, and cluster configuration.", "parameters": _obj({}, [])},
    {"name": "project_status", "description": "Summarize project setup, selected run, audit/verify state, generated artifacts, memory counts, and concrete next actions.",
     "parameters": _obj({"runs_root": {"type": "string"}, "db_path": {"type": "string"}, "notes_root": {"type": "string"}, "project_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}}, [])},
    {"name": "list_runs", "description": "List recent run.json artifacts with audit pass/fail, readiness, question, key artifact availability, and a selected run.",
     "parameters": _obj({"runs_root": {"type": "string"}, "limit": {"type": "integer"}, "no_audit": {"type": "boolean"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}}, [])},
    {"name": "audit_run", "description": "Run offline paper-readiness gates on a run artifact and return failed checks plus next actions.",
     "parameters": _obj({"path": {"type": "string", "description": "Optional run.json path; defaults to selected runs/**/run.json"}, "runs_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}}, [])},
    {"name": "verify_run", "description": "Verify run manifest integrity, immutable artifact hashes, and declared artifact existence. Set repair=true to refresh stale manifest coverage when hashes are clean.",
     "parameters": _obj({"path": {"type": "string", "description": "Optional run.json path; defaults to selected runs/**/run.json"}, "runs_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}, "repair": {"type": "boolean"}}, [])},
    {"name": "write_paper", "description": "Generate main.tex from a saved run artifact. Defaults to the selected run's paper/ directory; local mode renders an evidence-bound manuscript scaffold from the run ledger.",
     "parameters": _obj({"path": {"type": "string", "description": "Optional run.json path; defaults to selected runs/**/run.json"}, "runs_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}, "out_dir": {"type": "string"}, "compile": {"type": "boolean"}, "compile_timeout": {"type": "integer", "description": "Seconds to wait for tectonic when compile=true."}, "provider": {"type": "string", "enum": ["auto", "local", "openai", "anthropic"]}, "model": {"type": "string"}}, [])},
    {"name": "review_paper", "description": "Review a generated run-bound main.tex with a configured provider and save review.md beside it unless out_dir is set.",
     "parameters": _obj({"path": {"type": "string", "description": "Optional paper path; defaults to selected run-bound runs/**/paper/main.tex"}, "runs_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}, "out_dir": {"type": "string"}, "provider": {"type": "string", "enum": ["auto", "openai", "anthropic"]}, "model": {"type": "string"}}, [])},
    {"name": "bundle_artifacts", "description": "Create a shareable zip bundle for a selected run dossier, including report, run ledger, audit/status JSON, paper/review files, and manifest.",
     "parameters": _obj({"path": {"type": "string", "description": "Optional run.json path; defaults to selected runs/**/run.json"}, "runs_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}, "out": {"type": "string"}, "notes_root": {"type": "string"}, "project_root": {"type": "string"}}, [])},
    {"name": "verify_bundle", "description": "Verify a selected shareable bundle zip using its portable manifest: archive entries, byte sizes, SHA-256 hashes, and run identity metadata.",
     "parameters": _obj({"path": {"type": "string", "description": "Optional bundle zip path; defaults to selected run-bound bundle artifact"}, "runs_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}}, [])},
    {"name": "resolve_artifact", "description": "Resolve generated artifact paths for all, quickstart, ci, report, markdown, graph, evals, trace, experiments, discoveries, paper, review, bundle, manifest, pdf, run, openvla, or an explicit path.",
     "parameters": _obj({"target": {"type": "string"}, "runs_root": {"type": "string"}, "project_root": {"type": "string"}, "selection": {"type": "string", "enum": ["latest", "best", "ready"]}, "select": {"type": "string", "enum": ["latest", "best", "ready"]}}, [])},
    {"name": "openvla_sae", "description": "Work with the OpenVLA sparse-autoencoder project: status, init, plan, commands, create-manifest, validate-manifest, smoke, eval, features, or dossier. Use this for OpenVLA/SAE prompts instead of run_discovery.",
     "parameters": _obj({"action": {"type": "string", "enum": ["status", "init", "plan", "commands", "validate-manifest", "create-manifest", "smoke", "eval", "features", "dossier"]}, "project_root": {"type": "string"}, "manifest": {"type": "string"}, "out_dir": {"type": "string"}, "image_dir": {"type": "string"}, "instruction": {"type": "string"}, "action_label": {"type": "string"}, "limit": {"type": "integer"}, "force": {"type": "boolean"}, "d_model": {"type": "integer"}, "tokens": {"type": "integer"}, "steps": {"type": "integer"}, "k": {"type": "integer"}, "cache_dir": {"type": "string"}, "checkpoint": {"type": "string"}, "top_features": {"type": "integer"}, "max_files": {"type": "integer"}, "eval_dir": {"type": "string"}, "features_dir": {"type": "string"}}, [])},
]

HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "list_tool_results": tool_list_tool_results,
    "clean_tool_results": tool_clean_tool_results,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "glob": tool_glob,
    "grep": tool_grep,
    "web_search": tool_web_search,
    "web_fetch": tool_web_fetch,
    "arxiv_search": tool_arxiv_search,
    "neuronpedia_search": tool_neuronpedia_search,
    "verify_novelty": tool_verify_novelty,
    "present_options": tool_present_options,
    "run_research": tool_run_research,
    "run_discovery": tool_run_discovery,
    "list_skills": tool_list_skills,
    "environment_status": tool_environment_status,
    "project_status": tool_project_status,
    "list_runs": tool_list_runs,
    "audit_run": tool_audit_run,
    "verify_run": tool_verify_run,
    "write_paper": tool_write_paper,
    "review_paper": tool_review_paper,
    "bundle_artifacts": tool_bundle_artifacts,
    "verify_bundle": tool_verify_bundle,
    "resolve_artifact": tool_resolve_artifact,
    "openvla_sae": tool_openvla_sae,
}

# Per-tool metadata for the permission system. read_only tools never prompt.
META: dict[str, dict[str, Any]] = {
    "bash": {"read_only": False, "permission": "exec"},
    "read_file": {"read_only": True, "permission": "local"},
    "list_tool_results": {"read_only": True, "permission": "local"},
    "clean_tool_results": {"read_only": False, "permission": "write"},
    "write_file": {"read_only": False, "permission": "write"},
    "edit_file": {"read_only": False, "permission": "write"},
    "list_dir": {"read_only": True, "permission": "local"},
    "glob": {"read_only": True, "permission": "local"},
    "grep": {"read_only": True, "permission": "local"},
    "web_search": {"read_only": True, "permission": "network"},
    "web_fetch": {"read_only": True, "permission": "network"},
    "arxiv_search": {"read_only": True, "permission": "network"},
    "neuronpedia_search": {"read_only": True, "permission": "network"},
    "verify_novelty": {"read_only": True, "permission": "network"},
    "present_options": {"read_only": False, "permission": "local"},
    "run_research": {"read_only": False, "permission": "network"},
    "run_discovery": {"read_only": False, "permission": "gpu"},
    "list_skills": {"read_only": True, "permission": "local"},
    "environment_status": {"read_only": True, "permission": "local"},
    "project_status": {"read_only": True, "permission": "local"},
    "list_runs": {"read_only": True, "permission": "local"},
    "audit_run": {"read_only": True, "permission": "local"},
    "verify_run": {"read_only": True, "permission": "local"},
    "write_paper": {"read_only": False, "permission": "write"},
    "review_paper": {"read_only": False, "permission": "write"},
    "bundle_artifacts": {"read_only": False, "permission": "write"},
    "verify_bundle": {"read_only": True, "permission": "local"},
    "resolve_artifact": {"read_only": True, "permission": "local"},
    "openvla_sae": {"read_only": False, "permission": "write"},
}


def tool_meta(name: str) -> dict[str, Any]:
    if name.startswith("mcp__"):
        return {"read_only": False, "permission": "network"}
    return META.get(name, {"read_only": False, "permission": "local"})


def dynamic_specs() -> list[dict[str, Any]]:
    """Tool specs contributed at runtime by MCP servers ([] if none configured)."""

    from . import mcp

    return mcp.tool_specs()


def all_specs() -> list[dict[str, Any]]:
    return TOOL_SPECS + dynamic_specs()


def run_tool(name: str, args: dict[str, Any]) -> str:
    handler = HANDLERS.get(name)
    try:
        if handler:
            return _persist_if_large(name, handler(args))
        if name.startswith("mcp__"):
            from . import mcp

            return _persist_if_large(name, _normalize_mcp_result(name, mcp.call(name, args)))
        return json.dumps(
            _tool_error_payload(
                name,
                f"unknown tool {name}",
                failed_check="tool_registered",
                next_action="Choose a tool from the registered tool list.",
            )
        )
    except Exception as exc:  # noqa: BLE001 - report failures back to the model
        return json.dumps(
            _tool_error_payload(
                name,
                f"{type(exc).__name__}: {exc}",
                failed_check="tool_exception",
                next_action="Inspect the tool arguments and retry with a valid payload.",
            )
        )


def _tool_error_payload(
    name: str,
    error: str,
    *,
    failed_check: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": name,
        "error": error,
        "failed_checks": [failed_check],
        "next_actions": [next_action],
    }


def _normalize_mcp_result(name: str, result: str) -> str:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return result
    if isinstance(payload, dict) and "error" in payload and "ok" not in payload:
        payload.update(
            {
                "ok": False,
                "tool": name,
                "failed_checks": payload.get("failed_checks", ["mcp_tool_call"]),
                "next_actions": payload.get("next_actions", ["Check MCP server configuration and retry."]),
            }
        )
        return json.dumps(payload)
    return result
