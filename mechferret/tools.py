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
DEFAULT_WEB_RESULTS = 24
DEFAULT_ARXIV_RESULTS = 50
NOVELTY_RELATED_LIMIT = 24
NOVELTY_FOCUSED_LIMIT = 10
NOVELTY_QUERY_RESULT_LIMIT = 50
NOVELTY_MAX_QUERY_PASSES = 36
NOVELTY_CLOSEST_PRIOR_LIMIT = 8
NOVELTY_WEB_RESULT_LIMIT = 24
NOVELTY_WEB_MAX_QUERY_PASSES = 24
NOVELTY_WEB_FETCH_LIMIT = 12
NOVELTY_WEB_FETCH_CHARS = 2400
NOVELTY_MIN_OPTION_ARXIV_PASSES = 10
NOVELTY_MIN_OPTION_WEB_PASSES = 8
NOVELTY_RISKS = {
    "high_prior_art_risk",
    "medium_prior_art_risk",
    "low_prior_art_risk",
    "unresolved_no_close_prior_found",
    "unknown_search_incomplete",
}
NOVELTY_CLAIM_READINESS_STATUSES = {
    "not_ready_search_incomplete",
    "not_ready_prior_art_overlap",
    "not_ready_needs_more_evidence",
    "delta_review_required",
    "provisional_low_overlap_after_deep_search",
}
NOVELTY_THREAT_RISKS = {
    "disqualifying_until_delta_is_demonstrated",
    "needs_delta_review",
    "searched_no_strong_overlap",
    "not_searched",
}
NOVELTY_DISQUALIFYING_RISKS = NOVELTY_THREAT_RISKS | {"missing_recent_prior_art"}
NOVELTY_REQUIRED_THREATS = {"exact_phrase_overlap", "claim_collision"}
NOVELTY_REQUIRED_OPTION_SEARCH_FOCUS = {
    "recency",
    "recent_discovery",
    "architecture",
    "frontier_architecture",
    "method",
    "mechanism",
    "evaluation",
    "implementation",
    "replication",
    "failure_modes",
    "protocol",
    "exact_phrase",
    "claim_collision",
    "peer_review",
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
        if name == "verify_novelty":
            summary.update(_essential_novelty_summary(payload))
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
        "evidence_strength",
        "source_diversity",
        "required_delta",
        "comparison_matrix",
        "recent_pressure",
        "closest_prior_art",
        "claim_readiness",
    ):
        if isinstance(payload, dict) and key in payload:
            minimal[key] = payload[key]
    if isinstance(payload, dict) and name == "verify_novelty":
        minimal.update(_essential_novelty_summary(payload))
    if isinstance(summary.get("checks"), list):
        compact_checks = summary["checks"]
        minimal["checks"] = compact_checks
        if len(json.dumps(minimal)) > MAX_OUTPUT:
            minimal["checks"] = compact_checks[:80]
            minimal["checks_omitted"] = True
            minimal["check_count"] = len(compact_checks)
    return json.dumps(minimal)


def _essential_novelty_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "idea" in payload:
        summary["idea"] = _compact_json_value(payload.get("idea"))
    for source_key, count_key in (
        ("search_plan", "search_plan_count"),
        ("arxiv_search_plan", "arxiv_search_plan_count"),
        ("web_search_plan", "web_search_plan_count"),
        ("search_audit", "search_audit_row_count"),
        ("related_papers", "related_papers_count"),
        ("recent_papers", "recent_papers_count"),
        ("focused_papers", "focused_papers_count"),
        ("method_papers", "method_papers_count"),
        ("web_results", "web_results_count"),
        ("errors", "error_count"),
    ):
        value = payload.get(source_key)
        if isinstance(value, list):
            summary[count_key] = len(value)
    for plan_key in ("search_plan", "arxiv_search_plan", "web_search_plan"):
        plan = payload.get(plan_key)
        if isinstance(plan, list):
            summary[f"{plan_key}_limits"] = _compact_novelty_plan_limits(plan)
    assessment = payload.get("assessment")
    if isinstance(assessment, dict):
        compact_assessment: dict[str, Any] = {}
        for key in (
            "risk",
            "verdict",
            "evidence_strength",
            "source_diversity",
            "required_delta",
            "comparison_matrix",
            "novelty_threat_model",
            "disqualifying_overlap_tests",
            "recent_pressure",
            "freshness_profile",
            "claim_readiness",
        ):
            if key in assessment:
                compact_assessment[key] = _compact_json_value(assessment[key])
        if isinstance(assessment.get("search_audit"), dict):
            compact_assessment["search_audit"] = _compact_novelty_search_audit_summary(assessment["search_audit"])
        if isinstance(assessment.get("coverage"), dict):
            compact_assessment["coverage"] = _compact_novelty_coverage(assessment["coverage"])
        closest = assessment.get("closest_prior_art")
        if isinstance(closest, list):
            compact_assessment["closest_prior_art_count"] = len(closest)
            compact_assessment["closest_prior_art"] = [_compact_novelty_prior(item) for item in closest[:5]]
        summary["assessment"] = compact_assessment
    return summary


def _compact_novelty_plan_limits(plan: list[Any]) -> dict[str, Any]:
    rows = [row for row in plan if isinstance(row, dict)]
    requested: list[int] = []
    for row in rows:
        try:
            parsed = int(row.get("max_results", 0) or 0)
        except (TypeError, ValueError):
            continue
        requested.append(parsed)
    sort_by = sorted({str(row.get("sort_by", "")).strip() for row in rows if row.get("sort_by")})
    focuses = sorted({str(row.get("focus", "")).strip() for row in rows if row.get("focus")})
    return {
        "count": len(rows),
        "requested_results_min": min(requested, default=0),
        "requested_results_max": max(requested, default=0),
        "sort_by": sort_by,
        "focuses": focuses,
    }


def _compact_novelty_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "search_strategy",
        "arxiv_query_count",
        "web_query_count",
        "arxiv_results_per_query",
        "web_results_per_query",
        "retrieved_evidence",
        "retrieved_papers",
        "recent_evidence",
        "structured_recent_evidence",
        "text_recent_evidence",
        "latest_evidence_year",
        "evidence_years",
        "web_results",
        "web_pages_fetched",
        "web_results_with_page_text",
        "unique_source_domains",
        "credible_source_count",
        "search_audit_rows",
        "empty_search_passes",
        "empty_arxiv_passes",
        "empty_web_passes",
        "duplicate_only_search_passes",
        "failed_queries",
        "failed_arxiv_queries",
        "failed_web_queries",
        "failed_web_fetches",
        "recent_window",
    ):
        if key in coverage:
            compact[key] = coverage[key]
    for key in (
        "focus_coverage",
        "threat_model_coverage",
        "web_source_types",
        "credible_source_types",
        "arxiv_focuses",
        "web_focuses",
        "empty_search_focuses",
        "failed_search_focuses",
    ):
        if key in coverage:
            compact[key] = _compact_json_value(coverage[key])
    return compact


def _compact_novelty_search_audit_summary(search_audit: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "pass_count",
        "failed_passes",
        "empty_search_passes",
        "empty_arxiv_passes",
        "empty_web_passes",
        "duplicate_only_search_passes",
        "focus_coverage",
        "missing_focus_coverage",
    ):
        if key in search_audit:
            compact[key] = _compact_json_value(search_audit[key])
    for key in ("empty_focuses", "failed_focuses"):
        value = search_audit.get(key)
        if isinstance(value, list):
            compact[key] = [_compact_json_preview_item(row) for row in value[:12]]
            compact[f"{key}_count"] = len(value)
    focus_summary = search_audit.get("focus_summary")
    if isinstance(focus_summary, list):
        compact["focus_summary_count"] = len(focus_summary)
        compact["focus_summary"] = [_compact_json_preview_item(row) for row in focus_summary[:12]]
    passes = search_audit.get("passes")
    if isinstance(passes, list):
        compact["passes_count"] = len(passes)
        compact["passes_preview"] = [_compact_json_preview_item(row) for row in passes[:5]]
    return compact


def _compact_novelty_prior(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_json_preview_item(value)
    row: dict[str, Any] = {}
    for key in (
        "title",
        "url",
        "source",
        "source_type",
        "source_domain",
        "published",
        "focus",
        "score",
        "source_credibility",
        "matched_terms",
        "reason",
    ):
        if key in value:
            row[key] = _compact_json_value(value[key])
    excerpt = value.get("evidence_excerpt")
    if isinstance(excerpt, str) and excerpt:
        row["evidence_excerpt"] = excerpt[:260]
    return row


def _compact_json_object(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"related_papers", "recent_papers", "focused_papers", "method_papers", "web_results", "errors"} and isinstance(value, list):
            summary[f"{key}_count" if key != "errors" else "error_count"] = len(value)
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
            "evidence_strength",
            "source_diversity",
            "required_delta",
            "comparison_matrix",
            "novelty_threat_model",
            "disqualifying_overlap_tests",
            "recent_pressure",
            "closest_prior_art",
            "claim_readiness",
        ):
            if key in value and isinstance(value[key], (str, int, float, bool, type(None))):
                summary[key] = value[key]
            elif key in value and key == "closest_prior_art" and isinstance(value[key], list):
                summary["closest_prior_art_count"] = len(value[key])
                summary[key] = [_compact_novelty_prior(item) for item in value[key][:5]]
            elif key in value and isinstance(value[key], list):
                summary[key] = _compact_json_value(value[key])
            elif key in value and key == "claim_readiness" and isinstance(value[key], dict):
                summary[key] = _compact_json_value(value[key])
            elif key in value and key == "coverage" and isinstance(value[key], dict):
                summary[key] = _compact_novelty_coverage(value[key])
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
    max_results = max(max_results, DEFAULT_WEB_RESULTS)
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
    max_results = max(max_results, DEFAULT_ARXIV_RESULTS)
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
    from .controller import RESEARCH_DEFAULT_ROUNDS, MechFerret

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
    max_rounds, invalid = _int_arg(args, "max_rounds", RESEARCH_DEFAULT_ROUNDS, min_value=1)
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
        error = str(exc)
        failed_checks = ["provider_research"] if "Live provider research failed" in error else ["source_material"]
        return json.dumps(
            {
                "ok": False,
                "error": error,
                "failed_checks": failed_checks,
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
                "provider_research": run.provenance.get("provider_research", {}),
                "used_packaged_seed_corpus": run.provenance.get("used_packaged_seed_corpus", False),
                "source_count": run.provenance.get("source_count", 0),
                "max_rounds": run.provenance.get("max_rounds", 0),
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
    if not isinstance(args.get("backend"), str) or not args.get("backend", "").strip():
        return json.dumps({
            "ok": False,
            "tool": "run_discovery",
            "error": "run_discovery needs an explicit backend; pass backend=auto, synthetic, transformer_lens, tl, or real.",
            "argument": "backend",
            "expected": "explicit backend: auto, synthetic, transformer_lens, tl, or real",
            "failed_checks": ["backend_required"],
            "next_actions": [
                "Pass backend=synthetic only for an intentional smoke/demo run.",
                "Pass backend=transformer_lens or backend=real for real model measurements.",
                "Pass backend=auto only when automatic real-backend detection is intentional.",
            ],
        })
    budget_requested = any(args.get(name) not in (None, "") for name in ("max_rounds", "max_experiments", "max_gpu_seconds"))
    budget = Budget(max_rounds=max_rounds, max_experiments=max_experiments, max_gpu_seconds=max_gpu_seconds) if budget_requested else None
    try:
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
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return json.dumps(_discovery_error_payload(str(exc), out_dir=out_dir or "runs/agent"))
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


def _discovery_error_payload(error: str, *, out_dir: str) -> dict[str, Any]:
    lowered = error.lower()
    failed_checks: list[str] = []
    next_actions: list[str] = []
    if "explicit model" in lowered or "model is required" in lowered:
        failed_checks.append("model_required")
        next_actions.append("Ask the user which model to investigate, then pass model explicitly.")
    if "could not infer" in lowered or "explicit task" in lowered or "unknown interpretability task" in lowered:
        failed_checks.append("task_required")
        next_actions.append("Ask the user to choose task ioi, induction, greater_than, or factual_recall, or use a matching skill.")
    if "not aligned" in lowered or "mismatch" in lowered or "unsupported term" in lowered:
        failed_checks.append("request_alignment")
        next_actions.append("Use run_research for planning, openvla_sae for OpenVLA/SAE work, or pass allow_mismatch only for an intentional demo.")
    if "no such file" in lowered or "not found" in lowered:
        failed_checks.append("source_missing")
        next_actions.append("Check source_paths, urls, skill name, and output paths before retrying.")
    if not failed_checks:
        failed_checks.append("discovery_request")
    if not next_actions:
        next_actions.append("Inspect the discovery request and retry with explicit model, task or skill, and evidence sources.")
    return {
        "ok": False,
        "tool": "run_discovery",
        "error": error,
        "failed_checks": failed_checks,
        "next_actions": next_actions,
        "out_dir": out_dir,
    }


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
    from .knowledge import search_arxiv, web_fetch, web_search

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
    focused: list[dict[str, Any]] = []
    web_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    search_audit: list[dict[str, Any]] = []
    for item in plan:
        query = item["query"]
        sort_by = item["sort_by"]
        audit_row: dict[str, Any] = {
            "source": "arxiv",
            "focus": item["focus"],
            "query": query,
            "sort_by": sort_by,
            "requested_results": item["max_results"],
            "retrieved": 0,
            "unique_added": 0,
            "failed": False,
        }
        try:
            _, papers = search_arxiv(query, max_results=item["max_results"], sort_by=sort_by)
        except Exception as exc:  # noqa: BLE001
            papers = []
            audit_row["failed"] = True
            audit_row["error"] = str(exc)
            errors.append({"source": "arxiv", "query": query, "error": str(exc)})
        unique_added = 0
        for p in papers:
            row = _novelty_paper_row(p, focus=item["focus"])
            key = _novelty_paper_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_added += 1
            related.append(row)
            if sort_by in {"submittedDate", "lastUpdatedDate"} and len(recent) < NOVELTY_FOCUSED_LIMIT:
                recent.append(row)
            if _novelty_focus_is_deep(item["focus"]) and len(focused) < NOVELTY_FOCUSED_LIMIT:
                focused.append(row)
        audit_row["retrieved"] = len(papers)
        audit_row["unique_added"] = unique_added
        search_audit.append(audit_row)
    for item in web_plan:
        query = item["query"]
        audit_row = {
            "source": "web",
            "focus": item["focus"],
            "query": query,
            "requested_results": item["max_results"],
            "retrieved": 0,
            "unique_added": 0,
            "failed": False,
        }
        try:
            results = web_search(query, max_results=item["max_results"])
        except Exception as exc:  # noqa: BLE001
            results = []
            audit_row["failed"] = True
            audit_row["error"] = str(exc)
            errors.append({"source": "web", "query": query, "error": str(exc)})
        unique_added = 0
        for result in results:
            row = _novelty_web_row(result, focus=item["focus"])
            key = _novelty_paper_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_added += 1
            web_results.append(row)
        audit_row["retrieved"] = len(results)
        audit_row["unique_added"] = unique_added
        search_audit.append(audit_row)
    _novelty_enrich_web_results(web_results, web_fetch, errors)
    return json.dumps({
        "idea": idea,
        "search_plan": plan,
        "arxiv_search_plan": plan,
        "web_search_plan": web_plan,
        "related_papers": related[:NOVELTY_RELATED_LIMIT],
        "recent_papers": recent,
        "focused_papers": focused,
        "method_papers": [row for row in focused if "method" in str(row.get("focus", ""))],
        "web_results": web_results[:NOVELTY_RELATED_LIMIT],
        "search_audit": search_audit,
        "assessment": _novelty_assessment(
            idea,
            [*related, *web_results],
            errors,
            arxiv_plan=plan,
            web_plan=web_plan,
            search_audit=search_audit,
        ),
        "errors": errors,
        "novelty_questions": _novelty_questions(idea),
        "guidance": (
            "Do not claim high novelty unless the idea survives relevance, submitted-date, "
            "updated-date, exact-phrase, claim-collision, recent-discovery, architecture-variant, "
            "frontier-architecture, model-family, method, mechanism, evaluation, implementation, "
            "replication, failure-mode, and protocol searches. Compare against the closest "
            "recent papers, name the exact delta, cite likely prior art, and downgrade any "
            "direction that only renames an existing method."
        ),
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
    phrases = _novelty_phrases(terms)
    candidates = []
    for query in seeds[:3]:
        candidates.append(_novelty_plan_item(query, "relevance", "provided_relevance"))
        candidates.append(_novelty_plan_item(query, "submittedDate", "provided_recent_submitted"))
        candidates.append(_novelty_plan_item(query, "lastUpdatedDate", "provided_recent_updated"))
    candidates.extend(
        [
            _novelty_plan_item(compact, "relevance", "core_relevance"),
            _novelty_plan_item(compact, "submittedDate", "recent_submitted"),
            _novelty_plan_item(compact, "lastUpdatedDate", "recent_updated"),
            _novelty_plan_item(f"{compact} method design", "relevance", "method_relevance"),
            _novelty_plan_item(f"{compact} mechanism ablation causal evidence", "relevance", "mechanism_evidence"),
            _novelty_plan_item(f"{compact} benchmark evaluation negative results", "submittedDate", "recent_evaluation"),
            _novelty_plan_item(f"{compact} recent discovery emerging method", "submittedDate", "recent_discovery"),
            _novelty_plan_item(f"{compact} architecture variant empirical finding", "lastUpdatedDate", "architecture_variant"),
            _novelty_plan_item(
                f"{compact} state of the art frontier architecture scaling law",
                "lastUpdatedDate",
                "frontier_architecture_recent",
            ),
            _novelty_plan_item(
                f"{compact} new architecture model family capability discovery",
                "submittedDate",
                "frontier_model_family_discovery",
            ),
            _novelty_plan_item(f"{compact} replication reproduction failure analysis", "lastUpdatedDate", "replication_failure_modes"),
            _novelty_plan_item(f"{compact} dataset task protocol", "relevance", "evaluation_protocol"),
            _novelty_plan_item(f"{compact} limitations failure modes", "submittedDate", "recent_limitations"),
            _novelty_plan_item(f"{compact} survey empirical study", "lastUpdatedDate", "recent_survey"),
        ]
    )
    for phrase in phrases[:3]:
        candidates.append(_novelty_plan_item(f"{phrase} ablation evaluation", "relevance", "phrase_evaluation"))
        candidates.append(_novelty_plan_item(f"{phrase} recent benchmark", "submittedDate", "phrase_recent_benchmark"))
        candidates.append(_novelty_plan_item(f"{phrase} frontier architecture", "lastUpdatedDate", "phrase_frontier_architecture"))
    for phrase in phrases[:4]:
        candidates.append(_novelty_plan_item(f'"{phrase}"', "relevance", "exact_phrase"))
        candidates.append(_novelty_plan_item(f'"{phrase}" same contribution', "relevance", "claim_collision"))
        candidates.append(_novelty_plan_item(f'"{phrase}" independent replication critique', "lastUpdatedDate", "peer_review_critique"))
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
    phrases = _novelty_phrases(terms)
    candidates = []
    for query in seeds[:2]:
        candidates.append(_novelty_web_plan_item(query, "provided_web_relevance"))
    candidates.extend(
        [
            _novelty_web_plan_item(f"{compact} recent paper method", "web_recent_method"),
            _novelty_web_plan_item(f"{compact} benchmark evaluation leaderboard", "web_benchmark_evaluation"),
            _novelty_web_plan_item(f"{compact} implementation repository code", "web_code_prior"),
            _novelty_web_plan_item(f"{compact} project page technical report", "web_project_or_report"),
            _novelty_web_plan_item(f"{compact} recent discovery technical report", "web_recent_discovery"),
            _novelty_web_plan_item(f"{compact} architecture variant implementation", "web_architecture_variant"),
            _novelty_web_plan_item(
                f"{compact} frontier model architecture release technical report",
                "web_frontier_architecture_release",
            ),
            _novelty_web_plan_item(
                f"{compact} recent model family benchmark capability",
                "web_frontier_model_family",
            ),
            _novelty_web_plan_item(f"{compact} replication reproduction results", "web_replication_results"),
            _novelty_web_plan_item(f"{compact} limitations failure modes", "web_failure_modes"),
            _novelty_web_plan_item(f"{compact} dataset benchmark protocol", "web_dataset_protocol"),
            _novelty_web_plan_item(f"{compact} recent survey empirical study", "web_recent_survey"),
        ]
    )
    for phrase in phrases[:2]:
        candidates.append(_novelty_web_plan_item(f"{phrase} code benchmark", "web_phrase_code_benchmark"))
        candidates.append(_novelty_web_plan_item(f"{phrase} technical report", "web_phrase_report"))
    for phrase in phrases[:3]:
        candidates.append(_novelty_web_plan_item(f'"{phrase}" exact method', "web_exact_phrase"))
        candidates.append(_novelty_web_plan_item(f'"{phrase}" "we propose" "novel"', "web_claim_collision"))
        candidates.append(_novelty_web_plan_item(f'"{phrase}" OpenReview review rebuttal', "web_peer_review"))
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        key = item["query"].lower()
        if key in seen:
            continue
        seen.add(key)
        plan.append(item)
    return plan[:NOVELTY_WEB_MAX_QUERY_PASSES]


def _novelty_focus_is_deep(focus: Any) -> bool:
    text = str(focus)
    return any(
        key in text
        for key in (
            "method",
            "mechanism",
            "evaluation",
            "implementation",
            "replication",
            "failure",
            "protocol",
            "survey",
            "benchmark",
            "discovery",
            "architecture",
            "frontier",
            "sota",
            "scaling",
            "model_family",
            "exact",
            "claim",
            "critique",
            "peer",
        )
    )


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


def _novelty_phrases(terms: list[str]) -> list[str]:
    phrases: list[str] = []
    for width in (2, 3, 4):
        for index in range(0, max(0, len(terms) - width + 1)):
            phrase = " ".join(terms[index: index + width])
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases


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
        "page_excerpt": "",
        "fetched": False,
        "authors": [],
        "focus": focus,
    }


def _novelty_paper_key(row: dict[str, Any]) -> str:
    title = " ".join(str(row.get("title", "")).lower().split())
    url = " ".join(str(row.get("url", "")).lower().split())
    return title or url


def _novelty_assessment(
    idea: str,
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    *,
    arxiv_plan: list[dict[str, Any]] | None = None,
    web_plan: list[dict[str, Any]] | None = None,
    search_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    terms = _novelty_terms(idea)
    scored = [_novelty_scored_prior(row, terms) for row in rows]
    scored.sort(key=lambda row: row["score"], reverse=True)
    closest = scored[:NOVELTY_CLOSEST_PRIOR_LIMIT]
    top_score = closest[0]["score"] if closest else 0.0
    arxiv_count = sum(1 for row in rows if row.get("source") == "arxiv")
    web_count = sum(1 for row in rows if row.get("source") == "web")
    web_source_types = _novelty_web_source_type_counts(rows)
    source_profile = _novelty_source_profile(rows)
    freshness_profile = _novelty_freshness_profile(rows)
    arxiv_plan = arxiv_plan or []
    web_plan = web_plan or []
    search_audit = search_audit or []
    audit_summary = _novelty_search_audit_summary(search_audit)
    coverage = {
        "search_strategy": "deep_recent_discovery_method_mechanism_architecture_frontier_model_family_evaluation_implementation_replication_exact_claim_collision",
        "arxiv_query_count": len(arxiv_plan),
        "web_query_count": len(web_plan),
        "arxiv_results_per_query": max((int(row.get("max_results", 0)) for row in arxiv_plan), default=0),
        "web_results_per_query": max((int(row.get("max_results", 0)) for row in web_plan), default=0),
        "arxiv_focuses": sorted({str(row.get("focus", "")) for row in arxiv_plan if row.get("focus")}),
        "web_focuses": sorted({str(row.get("focus", "")) for row in web_plan if row.get("focus")}),
        "retrieved_evidence": len(rows),
        "retrieved_papers": arxiv_count,
        "recent_evidence": freshness_profile["recent_evidence_count"],
        "structured_recent_evidence": freshness_profile["structured_recent_evidence_count"],
        "text_recent_evidence": freshness_profile["text_recent_evidence_count"],
        "latest_evidence_year": freshness_profile["latest_year"],
        "evidence_years": freshness_profile["years"],
        "web_results": web_count,
        "web_results_with_snippets": sum(1 for row in rows if row.get("source") == "web" and row.get("abstract")),
        "web_pages_fetched": sum(1 for row in rows if row.get("source") == "web" and row.get("fetched")),
        "web_results_with_page_text": sum(1 for row in rows if row.get("source") == "web" and row.get("page_excerpt")),
        "web_source_types": web_source_types,
        "unique_source_domains": source_profile["unique_domains"],
        "source_domain_counts": source_profile["domain_counts"],
        "credible_source_count": source_profile["credible_sources"],
        "credible_source_types": source_profile["credible_types"],
        "search_audit_rows": len(search_audit),
        "empty_search_passes": audit_summary["empty_search_passes"],
        "empty_arxiv_passes": audit_summary["empty_arxiv_passes"],
        "empty_web_passes": audit_summary["empty_web_passes"],
        "duplicate_only_search_passes": audit_summary["duplicate_only_search_passes"],
        "empty_search_focuses": audit_summary["empty_focuses"],
        "failed_search_focuses": audit_summary["failed_focuses"],
        "failed_queries": len(errors),
        "failed_arxiv_queries": sum(1 for error in errors if error.get("source") == "arxiv"),
        "failed_web_queries": sum(1 for error in errors if error.get("source") == "web"),
        "failed_web_fetches": sum(1 for error in errors if error.get("source") == "web_fetch"),
        "idea_terms": terms,
        "recent_window": _novelty_recent_window_label(),
    }
    coverage["frontier_architecture_focuses"] = _novelty_frontier_focuses(
        coverage["arxiv_focuses"],
        coverage["web_focuses"],
    )
    coverage["frontier_architecture_covered"] = bool(coverage["frontier_architecture_focuses"])
    coverage["focus_coverage"] = _novelty_focus_coverage(coverage["arxiv_focuses"], coverage["web_focuses"])
    coverage["threat_model_coverage"] = _novelty_threat_model_coverage(coverage["arxiv_focuses"], coverage["web_focuses"])
    comparison_matrix = _novelty_comparison_matrix(scored, coverage)
    threat_model = _novelty_threat_model(scored, coverage)
    recent_pressure = _novelty_recent_pressure(rows)
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
        "evidence_strength": _novelty_evidence_strength(top_score, source_profile),
        "source_diversity": source_profile["diversity"],
        "closest_prior_art": closest,
        "claim_readiness": _novelty_claim_readiness(risk, top_score, coverage),
        "coverage": coverage,
        "freshness_profile": freshness_profile,
        "search_audit": audit_summary,
        "comparison_matrix": comparison_matrix,
        "novelty_threat_model": threat_model,
        "disqualifying_overlap_tests": _novelty_disqualifying_overlap_tests(threat_model, coverage),
        "recent_pressure": recent_pressure,
        "required_delta": _novelty_required_delta(idea, closest, comparison_matrix, coverage),
    }


def _novelty_search_audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [_compact_novelty_search_audit_row(row) for row in rows]
    focus_summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in normalized:
        source = str(row.get("source", "")).strip() or "unknown"
        focus = str(row.get("focus", "")).strip() or "unknown"
        key = (source, focus)
        target = focus_summary.setdefault(
            key,
            {
                "source": source,
                "focus": focus,
                "passes": 0,
                "failed_passes": 0,
                "retrieved": 0,
                "unique_added": 0,
                "requested_results_max": 0,
            },
        )
        target["passes"] += 1
        target["failed_passes"] += 1 if row.get("failed") else 0
        target["retrieved"] += int(row.get("retrieved", 0) or 0)
        target["unique_added"] += int(row.get("unique_added", 0) or 0)
        target["requested_results_max"] = max(
            int(target.get("requested_results_max", 0) or 0),
            int(row.get("requested_results", 0) or 0),
        )
    focus_rows = sorted(focus_summary.values(), key=lambda row: (row["source"], row["focus"]))
    empty_focuses = [
        {"source": row["source"], "focus": row["focus"]}
        for row in focus_rows
        if int(row.get("retrieved", 0) or 0) == 0
    ]
    failed_focuses = [
        {"source": row["source"], "focus": row["focus"]}
        for row in focus_rows
        if int(row.get("failed_passes", 0) or 0) > 0
    ]
    focus_coverage = _option_search_focus_coverage(focus_rows)
    missing_focus_coverage = [
        focus for focus in sorted(NOVELTY_REQUIRED_OPTION_SEARCH_FOCUS)
        if not focus_coverage.get(focus)
    ]
    return {
        "pass_count": len(normalized),
        "failed_passes": sum(1 for row in normalized if row.get("failed")),
        "empty_search_passes": sum(1 for row in normalized if int(row.get("retrieved", 0) or 0) == 0),
        "empty_arxiv_passes": sum(
            1 for row in normalized if row.get("source") == "arxiv" and int(row.get("retrieved", 0) or 0) == 0
        ),
        "empty_web_passes": sum(
            1 for row in normalized if row.get("source") == "web" and int(row.get("retrieved", 0) or 0) == 0
        ),
        "duplicate_only_search_passes": sum(
            1
            for row in normalized
            if int(row.get("retrieved", 0) or 0) > 0 and int(row.get("unique_added", 0) or 0) == 0
        ),
        "empty_focuses": empty_focuses,
        "failed_focuses": failed_focuses,
        "focus_coverage": focus_coverage,
        "missing_focus_coverage": missing_focus_coverage,
        "focus_summary": focus_rows,
        "passes": normalized,
    }


def _compact_novelty_search_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("source", "focus", "query", "sort_by", "requested_results", "retrieved", "unique_added", "failed"):
        if key in row:
            compact[key] = _compact_json_value(row[key])
    if row.get("error"):
        compact["error"] = str(row.get("error", ""))[:240]
    return compact


def _novelty_scored_prior(row: dict[str, Any], terms: list[str]) -> dict[str, Any]:
    haystack = f"{row.get('title', '')} {row.get('abstract', '')} {row.get('page_excerpt', '')}".lower()
    matched = [term for term in terms if term in haystack]
    term_score = len(matched) / max(1, len(terms))
    focus = str(row.get("focus", ""))
    focus_score = 0.15 if any(key in focus for key in ("method", "mechanism", "relevance", "evaluation", "implementation")) else 0.0
    recent_score = 0.1 if _novelty_row_is_recent(row) else 0.0
    source_type = str(row.get("source_type", "general_web"))
    source_score = _novelty_source_type_score(source_type)
    source_credibility = _novelty_source_credibility(source_type, row.get("source_domain", ""))
    score = min(1.0, round(term_score * 0.68 + focus_score + recent_score + source_score + source_credibility["score"], 3))
    return {
        "source": row.get("source", ""),
        "source_type": source_type,
        "source_credibility": source_credibility["label"],
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "source_domain": row.get("source_domain", ""),
        "published": row.get("published", ""),
        "focus": focus,
        "score": score,
        "source_score": source_score,
        "source_credibility_score": source_credibility["score"],
        "matched_terms": matched,
        "evidence_excerpt": _novelty_evidence_excerpt(row),
        "reason": _novelty_prior_reason(matched, focus, row, row.get("source", ""), source_type),
    }


def _novelty_focus_coverage(arxiv_focuses: list[str], web_focuses: list[str]) -> dict[str, bool]:
    text = " ".join([*arxiv_focuses, *web_focuses]).lower()
    return {
        "relevance": "relevance" in text,
        "recency": "recent" in text or "submitted" in text or "updated" in text,
        "method": "method" in text,
        "mechanism": "mechanism" in text,
        "evaluation": "evaluation" in text or "benchmark" in text,
        "implementation": "implementation" in text or "code" in text,
        "replication": "replication" in text or "reproduction" in text,
        "failure_modes": "failure" in text or "limitations" in text,
        "protocol": "protocol" in text or "dataset" in text,
        "survey": "survey" in text,
        "recent_discovery": "discovery" in text,
        "architecture": "architecture" in text,
        "frontier_architecture": _novelty_focus_text_has_frontier(text),
        "exact_phrase": "exact" in text,
        "claim_collision": "claim" in text,
        "peer_review": "peer" in text or "critique" in text,
    }


_NOVELTY_COMPARISON_AXES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("exact_phrase", ("exact", "same contribution"), ()),
    ("claim_collision", ("claim", "same contribution", "we propose", "novel"), ()),
    ("recency", ("recent", "submitted", "updated", "discovery"), ()),
    ("method", ("method", "design", "approach"), ()),
    ("mechanism", ("mechanism", "ablation", "causal", "probe"), ()),
    ("architecture", ("architecture", "variant", "model"), ()),
    ("frontier_architecture", ("frontier", "sota", "state of the art", "model_family", "model family", "scaling"), ()),
    ("evaluation", ("evaluation", "benchmark", "leaderboard", "metric"), ("benchmark",)),
    ("implementation", ("implementation", "repository", "code"), ("code_repository",)),
    ("replication", ("replication", "reproduction", "reproduce"), ()),
    ("peer_review", ("peer", "critique", "review", "rebuttal"), ()),
    ("failure_modes", ("failure", "limitations", "negative"), ()),
    ("protocol", ("protocol", "dataset", "task"), ()),
)


def _novelty_comparison_matrix(scored: list[dict[str, Any]], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    focus_coverage = coverage.get("focus_coverage") if isinstance(coverage.get("focus_coverage"), dict) else {}
    matrix: list[dict[str, Any]] = []
    for axis, focus_terms, source_types in _NOVELTY_COMPARISON_AXES:
        matches = [row for row in scored if _novelty_prior_matches_axis(row, focus_terms, source_types)]
        top = max(matches, key=lambda row: row.get("score", 0.0), default={})
        covered = bool(focus_coverage.get(axis) or matches)
        if axis == "recency":
            covered = coverage.get("recent_evidence", 0) > 0
        representative = _novelty_representative_prior(top)
        matrix.append(
            {
                "axis": axis,
                "covered": bool(covered),
                "evidence_count": len(matches),
                "recent_evidence_count": sum(1 for row in matches if _novelty_row_is_recent(row)),
                "strongest_score": round(float(top.get("score", 0.0) or 0.0), 3) if top else 0.0,
                "representative_prior": representative,
                "next_action": _novelty_axis_next_action(axis, bool(covered), representative),
            }
        )
    return matrix


def _novelty_prior_matches_axis(row: dict[str, Any], focus_terms: tuple[str, ...], source_types: tuple[str, ...]) -> bool:
    focus = str(row.get("focus", "")).lower()
    source_type = str(row.get("source_type", "")).lower()
    reason = str(row.get("reason", "")).lower()
    title = str(row.get("title", "")).lower()
    haystack = f"{focus} {reason} {title}"
    return any(term in haystack for term in focus_terms) or bool(source_types and source_type in source_types)


def _novelty_representative_prior(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "source_type": row.get("source_type", ""),
        "published": row.get("published", ""),
        "matched_terms": row.get("matched_terms", []),
    }


def _novelty_axis_next_action(axis: str, covered: bool, representative: dict[str, Any]) -> str:
    if not covered:
        return f"Run a targeted {axis.replace('_', ' ')} search and add the closest counterexample before ranking this idea."
    title = str(representative.get("title", "")).strip()
    if title:
        return f"Compare the idea against '{title}' on the {axis.replace('_', ' ')} axis."
    return f"Summarize the strongest retrieved {axis.replace('_', ' ')} evidence and the remaining delta."


def _novelty_threat_model_coverage(arxiv_focuses: list[str], web_focuses: list[str]) -> dict[str, bool]:
    text = " ".join([*arxiv_focuses, *web_focuses]).lower()
    return {
        "exact_phrase_overlap": "exact" in text,
        "claim_collision": "claim" in text,
        "method_overlap": "method" in text,
        "mechanism_overlap": "mechanism" in text,
        "architecture_overlap": "architecture" in text,
        "evaluation_overlap": "evaluation" in text or "benchmark" in text,
        "implementation_overlap": "implementation" in text or "code" in text,
        "replication_or_critique": "replication" in text or "reproduction" in text or "critique" in text or "peer" in text,
    }


def _novelty_threat_model(scored: list[dict[str, Any]], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    threat_specs = (
        ("exact_phrase_overlap", ("exact", "same contribution"), "A prior uses the same phrase or names the same contribution."),
        ("claim_collision", ("claim", "same contribution", "we propose", "novel"), "A prior claims the same core contribution."),
        ("method_overlap", ("method", "design", "approach"), "A prior already implements the method."),
        ("mechanism_overlap", ("mechanism", "ablation", "causal", "probe"), "A prior already gives the mechanistic explanation or causal test."),
        ("architecture_overlap", ("architecture", "variant", "model"), "A prior already studies the same architectural variant."),
        ("evaluation_overlap", ("evaluation", "benchmark", "leaderboard", "metric"), "A prior already runs the key evaluation or benchmark."),
        ("implementation_overlap", ("implementation", "repository", "code"), "A prior already publishes implementation-level evidence."),
        ("replication_or_critique", ("replication", "reproduction", "critique", "rebuttal"), "A replication, critique, or negative result changes the claim."),
    )
    coverage_map = coverage.get("threat_model_coverage") if isinstance(coverage.get("threat_model_coverage"), dict) else {}
    rows: list[dict[str, Any]] = []
    for name, terms, failure_mode in threat_specs:
        matches = [row for row in scored if _novelty_prior_matches_axis(row, terms, ())]
        top = max(matches, key=lambda row: row.get("score", 0.0), default={})
        strongest = round(float(top.get("score", 0.0) or 0.0), 3) if top else 0.0
        if strongest >= 0.55:
            risk = "disqualifying_until_delta_is_demonstrated"
        elif strongest >= 0.25 or matches:
            risk = "needs_delta_review"
        elif coverage_map.get(name):
            risk = "searched_no_strong_overlap"
        else:
            risk = "not_searched"
        rows.append(
            {
                "threat": name,
                "searched": bool(coverage_map.get(name)),
                "risk": risk,
                "evidence_count": len(matches),
                "strongest_score": strongest,
                "representative_prior": _novelty_representative_prior(top),
                "failure_mode": failure_mode,
                "next_action": _novelty_threat_next_action(name, risk, top),
            }
        )
    return rows


def _novelty_threat_next_action(name: str, risk: str, top: dict[str, Any]) -> str:
    label = name.replace("_", " ")
    title = str(top.get("title", "")).strip() if top else ""
    if risk == "not_searched":
        return f"Run a targeted {label} search before considering the idea."
    if title:
        return f"Write the exact delta from '{title}' for the {label} threat."
    if risk == "searched_no_strong_overlap":
        return f"Record why retrieved evidence does not trigger the {label} threat."
    return f"Resolve the {label} threat with a concrete empirical or architectural delta."


def _novelty_disqualifying_overlap_tests(threat_model: list[dict[str, Any]], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for row in threat_model:
        threat = str(row.get("threat", ""))
        risk = str(row.get("risk", ""))
        representative = row.get("representative_prior") if isinstance(row.get("representative_prior"), dict) else {}
        tests.append(
            {
                "test": threat,
                "passed": risk == "searched_no_strong_overlap",
                "risk": risk,
                "representative_prior": representative,
                "required_evidence": _novelty_disqualifier_required_evidence(threat),
            }
        )
    if coverage.get("recent_evidence", 0) < 1:
        tests.append(
            {
                "test": "recent_prior_art_window",
                "passed": False,
                "risk": "missing_recent_prior_art",
                "representative_prior": {},
                "required_evidence": "Find at least one relevant prior in the recent-paper window or document why current search could not retrieve it.",
            }
        )
    return tests


def _novelty_disqualifier_required_evidence(threat: str) -> str:
    return {
        "exact_phrase_overlap": "Nearest exact-phrase prior and why its claim is materially different.",
        "claim_collision": "Closest paper/project claiming the same contribution and the specific delta.",
        "method_overlap": "Method-level comparison against the nearest implementation or paper.",
        "mechanism_overlap": "Mechanism-level comparison backed by causal or ablation evidence.",
        "architecture_overlap": "Architecture/model-family comparison showing what is new.",
        "evaluation_overlap": "Evaluation and benchmark delta, including negative or replication evidence.",
        "implementation_overlap": "Code/project comparison showing whether this is more than a reimplementation.",
        "replication_or_critique": "Replication, critique, or failure-mode evidence and how it changes the claim.",
    }.get(threat, "Closest counterexample and a concrete delta.")


def _novelty_recent_pressure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recent_rows = [row for row in rows if _novelty_row_is_recent(row)]
    years = sorted({year for row in rows for year in _novelty_evidence_years(row)}, reverse=True)
    recent_titles = []
    for row in recent_rows[:5]:
        title = str(row.get("title", "")).strip()
        if title:
            recent_titles.append(title)
    return {
        "recent_window": _novelty_recent_window_label(),
        "recent_evidence_count": len(recent_rows),
        "latest_year": years[0] if years else None,
        "recent_prior_titles": recent_titles,
        "status": "recent_prior_present" if recent_rows else "missing_recent_prior_art",
    }


def _novelty_required_delta(
    idea: str,
    closest: list[dict[str, Any]],
    comparison_matrix: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> list[str]:
    terms = _novelty_terms(idea)
    actions: list[str] = []
    if closest:
        top = closest[0]
        title = str(top.get("title", "")).strip() or str(top.get("url", "")).strip() or "the nearest retrieved prior"
        matched = ", ".join(str(term) for term in top.get("matched_terms", [])[:6]) or "the core idea terms"
        source_type = str(top.get("source_type", "")).strip() or "prior art"
        actions.append(
            f"Differentiate from '{title}' ({source_type}): matched {matched}; name the exact method, mechanism, architecture, or evaluation component that changes."
        )
    else:
        term_preview = ", ".join(terms[:6]) or "the idea's core mechanism"
        actions.append(
            f"No close prior was retrieved for {term_preview}; repeat targeted expert/literature searches before treating absence as novelty evidence."
        )
    missing_axes = [str(row.get("axis", "")) for row in comparison_matrix if not row.get("covered")]
    if missing_axes:
        actions.append(
            "Fill missing novelty comparison axes before selection: " + ", ".join(axis.replace("_", " ") for axis in missing_axes[:6]) + "."
        )
    if coverage.get("recent_evidence", 0) < 1:
        actions.append("Add recent submitted-date and updated-date evidence; novelty cannot be judged from stale-only prior art.")
    actions.append(
        "Require an empirical delta: benchmark result, ablation, causal intervention, replication, or failure-mode test where the idea differs from the closest prior."
    )
    return actions


def _novelty_claim_readiness(risk: str, top_score: float, coverage: dict[str, Any]) -> dict[str, Any]:
    focus_coverage = coverage.get("focus_coverage") if isinstance(coverage.get("focus_coverage"), dict) else {}
    threat_coverage = coverage.get("threat_model_coverage") if isinstance(coverage.get("threat_model_coverage"), dict) else {}
    checks = {
        "deep_query_plan": coverage.get("arxiv_query_count", 0) >= 10
        and coverage.get("web_query_count", 0) >= 8
        and coverage.get("arxiv_results_per_query", 0) >= 50
        and coverage.get("web_results_per_query", 0) >= 24,
        "focus_breadth": all(
            bool(focus_coverage.get(name))
            for name in (
                "recent_discovery",
                "architecture",
                "frontier_architecture",
                "method",
                "mechanism",
                "evaluation",
                "implementation",
                "replication",
                "failure_modes",
                "protocol",
            )
        ),
        "retrieved_prior_art": coverage.get("retrieved_evidence", 0) >= 8,
        "recent_prior_art": coverage.get("recent_evidence", 0) >= 1,
        "credible_source_diversity": coverage.get("unique_source_domains", 0) >= 3
        and coverage.get("credible_source_count", 0) >= 2,
        "web_page_enrichment": coverage.get("web_results_with_page_text", 0) >= 1,
        "threat_model_depth": all(
            bool(threat_coverage.get(name))
            for name in (
                "exact_phrase_overlap",
                "claim_collision",
                "method_overlap",
                "mechanism_overlap",
                "architecture_overlap",
                "evaluation_overlap",
                "implementation_overlap",
                "replication_or_critique",
            )
        ),
        "search_completed": coverage.get("failed_arxiv_queries", 0) == 0 and coverage.get("failed_web_queries", 0) == 0,
    }
    missing = [name for name, passed in checks.items() if not passed]
    if risk == "unknown_search_incomplete" or not checks["search_completed"]:
        status = "not_ready_search_incomplete"
    elif risk == "high_prior_art_risk":
        status = "not_ready_prior_art_overlap"
    elif missing:
        status = "not_ready_needs_more_evidence"
    elif top_score >= 0.25:
        status = "delta_review_required"
    else:
        status = "provisional_low_overlap_after_deep_search"
    return {
        "status": status,
        "can_claim_high_novelty": False,
        "checks": checks,
        "missing_checks": missing,
        "next_actions": _novelty_readiness_next_actions(status, missing),
    }


def _novelty_readiness_next_actions(status: str, missing: list[str]) -> list[str]:
    actions = []
    if status == "not_ready_search_incomplete":
        actions.append("Retry failed retrieval passes before judging novelty.")
    if status == "not_ready_prior_art_overlap":
        actions.append("Treat the direction as prior-art-overlapping until the exact method and evaluation delta is demonstrated.")
    if "focus_breadth" in missing:
        actions.append("Run follow-up searches covering recent discoveries, frontier architectures, model families, method, mechanism, evaluation, implementation, replication, failure modes, and protocol.")
    if "retrieved_prior_art" in missing or "credible_source_diversity" in missing:
        actions.append("Collect more independent papers, benchmarks, code, or project reports before selecting this idea.")
    if "recent_prior_art" in missing:
        actions.append("Add recent submitted-date and updated-date searches for the last two years.")
    if "web_page_enrichment" in missing:
        actions.append("Fetch the top credible web/code/project pages and compare their technical details.")
    if "threat_model_depth" in missing:
        actions.append("Run exact-phrase, claim-collision, method, mechanism, architecture, frontier model-family, evaluation, implementation, and critique searches before ranking novelty.")
    if not actions:
        actions.append("Use the closest prior art list to write a precise required delta; do not claim high novelty without expert review.")
    return actions


def _novelty_enrich_web_results(
    rows: list[dict[str, Any]],
    fetch: Callable[..., str],
    errors: list[dict[str, str]],
) -> None:
    candidates = sorted(
        (row for row in rows if row.get("url")),
        key=_novelty_fetch_priority,
        reverse=True,
    )[:NOVELTY_WEB_FETCH_LIMIT]
    for row in candidates:
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        try:
            text = fetch(url, max_chars=NOVELTY_WEB_FETCH_CHARS)
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": "web_fetch", "url": url, "error": str(exc)})
            continue
        excerpt = " ".join(str(text or "").split())[:900]
        if not excerpt:
            continue
        row["fetched"] = True
        row["fetched_chars"] = len(str(text or ""))
        row["page_excerpt"] = excerpt


def _novelty_fetch_priority(row: dict[str, Any]) -> float:
    source_type = str(row.get("source_type", "general_web"))
    credibility = _novelty_source_credibility(source_type, row.get("source_domain", ""))
    snippet_bonus = 0.01 if row.get("abstract") else 0.0
    return _novelty_source_type_score(source_type) + credibility["score"] + snippet_bonus


def _novelty_evidence_excerpt(row: dict[str, Any]) -> str:
    abstract = str(row.get("abstract", "")).strip()
    page = str(row.get("page_excerpt", "")).strip()
    if abstract and page:
        return f"{abstract[:180]} | page: {page[:180]}"
    return (abstract or page)[:240]


def _novelty_prior_reason(matched: list[str], focus: str, row: dict[str, Any], source: Any, source_type: Any) -> str:
    bits = []
    if matched:
        bits.append("shares idea terms: " + ", ".join(matched[:6]))
    if source == "web":
        bits.append("retrieved from web search")
    if source_type in {"paper", "benchmark", "code_repository", "project_page", "documentation"}:
        bits.append(f"source type: {source_type}")
    if "method" in focus:
        bits.append("retrieved by method-focused search")
    if "mechanism" in focus:
        bits.append("retrieved by mechanism-focused search")
    if "evaluation" in focus:
        bits.append("retrieved by evaluation-focused search")
    if _novelty_row_is_recent(row):
        bits.append("within the recent-paper window")
    return "; ".join(bits) or "retrieved as adjacent prior art"


def _novelty_row_is_recent(row: dict[str, Any]) -> bool:
    return any(year >= datetime.now(UTC).year - 2 for year in _novelty_evidence_years(row))


def _novelty_evidence_years(row: dict[str, Any]) -> list[int]:
    years: set[int] = set()
    published_year = _novelty_year(row.get("published", ""))
    if published_year is not None:
        years.add(published_year)
    current_year = datetime.now(UTC).year
    for key in ("title", "abstract", "page_excerpt"):
        text = str(row.get(key, "") or "")
        for index in range(max(0, len(text) - 3)):
            token = text[index: index + 4]
            if not token.isdigit():
                continue
            year = int(token)
            if 2020 <= year <= current_year:
                years.add(year)
    return sorted(years, reverse=True)


def _novelty_freshness_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_year = datetime.now(UTC).year
    recent_floor = current_year - 2
    all_years = sorted({year for row in rows for year in _novelty_evidence_years(row)}, reverse=True)
    structured_recent = 0
    text_recent = 0
    recent_titles: list[str] = []
    for row in rows:
        structured_year = _novelty_year(row.get("published", ""))
        years = _novelty_evidence_years(row)
        row_recent = any(year >= recent_floor for year in years)
        if structured_year is not None and structured_year >= recent_floor:
            structured_recent += 1
        elif row_recent:
            text_recent += 1
        if row_recent and len(recent_titles) < 8:
            title = str(row.get("title", "")).strip()
            if title:
                recent_titles.append(title)
    return {
        "recent_window": _novelty_recent_window_label(),
        "latest_year": all_years[0] if all_years else None,
        "years": all_years[:8],
        "recent_evidence_count": structured_recent + text_recent,
        "structured_recent_evidence_count": structured_recent,
        "text_recent_evidence_count": text_recent,
        "recent_titles": recent_titles,
        "status": "recent_prior_present" if structured_recent + text_recent else "missing_recent_prior_art",
    }


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


def _novelty_source_credibility(source_type: str, domain: Any) -> dict[str, Any]:
    domain_text = str(domain or "").lower()
    if source_type == "paper" or any(key in domain_text for key in ("arxiv.org", "openreview.net", "aclanthology.org", "proceedings.mlr.press", "papers.nips.cc")):
        return {"label": "scholarly", "score": 0.035}
    if source_type == "benchmark" or "paperswithcode.com" in domain_text:
        return {"label": "benchmark", "score": 0.03}
    if source_type == "code_repository" or any(key in domain_text for key in ("github.com", "gitlab.com", "huggingface.co")):
        return {"label": "implementation", "score": 0.02}
    if source_type in {"project_page", "documentation"}:
        return {"label": source_type, "score": 0.01}
    return {"label": "generic", "score": 0.0}


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


def _novelty_source_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain_counts: dict[str, int] = {}
    credible_types: set[str] = set()
    credible_sources = 0
    for row in rows:
        domain = str(row.get("source_domain", "")).strip().lower()
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        source_type = str(row.get("source_type", "general_web"))
        credibility = _novelty_source_credibility(source_type, domain)
        if credibility["score"] > 0:
            credible_sources += 1
            credible_types.add(source_type)
    unique_domains = len(domain_counts)
    unique_types = len({str(row.get("source_type", "general_web")) for row in rows})
    if unique_domains >= 3 and unique_types >= 2 and credible_sources >= 2:
        diversity = "broad_independent"
    elif unique_domains >= 2 and credible_sources >= 1:
        diversity = "moderate"
    elif rows:
        diversity = "narrow"
    else:
        diversity = "none"
    return {
        "unique_domains": unique_domains,
        "domain_counts": dict(sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
        "credible_sources": credible_sources,
        "credible_types": sorted(credible_types),
        "diversity": diversity,
    }


def _novelty_evidence_strength(top_score: float, source_profile: dict[str, Any]) -> str:
    diversity = source_profile.get("diversity")
    credible_sources = int(source_profile.get("credible_sources") or 0)
    if top_score >= 0.55 and diversity == "broad_independent":
        return "strong_multi_source_overlap"
    if top_score >= 0.55 and credible_sources:
        return "strong_but_narrow_overlap"
    if top_score >= 0.25 and diversity in {"broad_independent", "moderate"}:
        return "moderate_multi_source_overlap"
    if top_score >= 0.25:
        return "moderate_narrow_overlap"
    return "weak_or_adjacent_overlap"


def _novelty_recent_window_label() -> str:
    year = datetime.now(UTC).year
    return f"{year - 2}-{year}"


def _novelty_questions(idea: str) -> list[str]:
    terms = ", ".join(_novelty_terms(idea)[:5]) or "the core mechanism"
    return [
        f"Which recent papers already combine {terms}?",
        "What is the nearest method or training-pipeline ancestor, and what exact component changes?",
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
    if not 2 <= len(options) <= 5:
        return json.dumps(
            _invalid_object_list_payload(
                "options",
                len(options),
                index=None,
                expected="2-5 validated research direction objects",
            )
        )
    return json.dumps({"options": titles, "option_details": [_option_detail(option) for option in options]})


def _validate_option_card(option: dict[str, Any], index: int) -> dict[str, Any] | None:
    for key in ("title", "summary", "detail", "novelty_verdict"):
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
    readiness = option.get("claim_readiness")
    if not isinstance(readiness, dict):
        return _invalid_object_list_payload(
            "options",
            readiness,
            index=index,
            expected="objects with claim_readiness from verify_novelty assessment",
        )
    status = readiness.get("status")
    if not isinstance(status, str) or status.strip() not in NOVELTY_CLAIM_READINESS_STATUSES:
        return _invalid_object_list_payload(
            "options",
            status,
            index=index,
            expected="claim_readiness.status from verify_novelty assessment",
        )
    if readiness.get("can_claim_high_novelty") is not False:
        return _invalid_object_list_payload(
            "options",
            readiness.get("can_claim_high_novelty"),
            index=index,
            expected="claim_readiness.can_claim_high_novelty=false from verify_novelty assessment",
        )
    if not isinstance(readiness.get("missing_checks", []), list) or not isinstance(readiness.get("next_actions", []), list):
        return _invalid_object_list_payload(
            "options",
            readiness,
            index=index,
            expected="claim_readiness with missing_checks and next_actions lists",
        )
    comparison_matrix = option.get("comparison_matrix")
    if not _valid_option_comparison_matrix(comparison_matrix):
        return _invalid_object_list_payload(
            "options",
            comparison_matrix,
            index=index,
            expected="objects with comparison_matrix from verify_novelty assessment",
        )
    threat_model = option.get("novelty_threat_model")
    if not _valid_option_threat_model(threat_model):
        return _invalid_object_list_payload(
            "options",
            threat_model,
            index=index,
            expected="objects with novelty_threat_model from verify_novelty assessment",
        )
    disqualifying_tests = option.get("disqualifying_overlap_tests")
    if not _valid_option_disqualifying_tests(disqualifying_tests):
        return _invalid_object_list_payload(
            "options",
            disqualifying_tests,
            index=index,
            expected="objects with disqualifying_overlap_tests from verify_novelty assessment",
        )
    search_audit = option.get("search_audit")
    if not _valid_option_search_audit(search_audit):
        return _invalid_object_list_payload(
            "options",
            search_audit,
            index=index,
            expected="objects with search_audit from verify_novelty assessment",
        )
    recent_pressure = option.get("recent_pressure")
    if not _valid_option_recent_pressure(recent_pressure):
        return _invalid_object_list_payload(
            "options",
            recent_pressure,
            index=index,
            expected="objects with recent_pressure from verify_novelty assessment",
        )
    required_delta = option.get("required_delta")
    if not _option_required_delta(required_delta):
        return _invalid_object_list_payload(
            "options",
            required_delta,
            index=index,
            expected="objects with non-empty required_delta from verify_novelty assessment",
        )
    return None


def _option_detail(option: dict[str, Any]) -> dict[str, Any]:
    detail = {
        "title": str(option.get("title", "")).strip(),
        "summary": str(option.get("summary", "")).strip(),
        "detail": str(option.get("detail", "")).strip(),
    }
    for key in ("novelty_risk", "novelty_verdict", "novelty"):
        value = option.get(key)
        if isinstance(value, str) and value.strip():
            detail[key] = value.strip()
    required_delta = _option_required_delta(option.get("required_delta"))
    if required_delta:
        detail["required_delta"] = required_delta
    citations = _option_strings(option.get("citations", []))[:4]
    if citations:
        detail["citations"] = citations
    closest = _option_strings(option.get("closest_prior_art", []))[:3]
    if closest:
        detail["closest_prior_art"] = closest
    readiness = option.get("claim_readiness")
    if isinstance(readiness, dict):
        detail["claim_readiness"] = {
            "status": str(readiness.get("status", "")).strip(),
            "can_claim_high_novelty": bool(readiness.get("can_claim_high_novelty")),
            "missing_checks": _option_strings(readiness.get("missing_checks", []))[:8],
            "next_actions": _option_strings(readiness.get("next_actions", []))[:4],
        }
    comparison = _option_comparison_matrix(option.get("comparison_matrix"))
    if comparison:
        detail["comparison_matrix"] = comparison
    threat_model = _option_threat_model(option.get("novelty_threat_model"))
    if threat_model:
        detail["novelty_threat_model"] = threat_model
    disqualifying_tests = _option_disqualifying_tests(option.get("disqualifying_overlap_tests"))
    if disqualifying_tests:
        detail["disqualifying_overlap_tests"] = disqualifying_tests
    search_audit = _option_search_audit(option.get("search_audit"))
    if search_audit:
        detail["search_audit"] = search_audit
    recent_pressure = _option_recent_pressure(option.get("recent_pressure"))
    if recent_pressure:
        detail["recent_pressure"] = recent_pressure
    return detail


def _option_required_delta(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
        return " ".join(parts[:3]).strip()
    return ""


def _valid_option_comparison_matrix(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    valid_raw = [
        item
        for item in value
        if isinstance(item, dict)
        and isinstance(item.get("axis"), str)
        and item.get("axis", "").strip()
        and type(item.get("covered")) is bool
    ]
    if len(valid_raw) != len(value):
        return False
    rows = _option_comparison_matrix(value)
    if not rows:
        return False
    axes = {row["axis"] for row in rows}
    return bool({"method", "evaluation"} & axes) and all("covered" in row for row in rows)


def _option_comparison_matrix(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        axis = str(item.get("axis", "")).strip()
        if not axis:
            continue
        row: dict[str, Any] = {
            "axis": axis,
            "covered": item.get("covered") if type(item.get("covered")) is bool else False,
            "evidence_count": _safe_int(item.get("evidence_count")),
            "next_action": str(item.get("next_action", "")).strip(),
        }
        prior = item.get("representative_prior")
        if isinstance(prior, dict):
            title = str(prior.get("title", "")).strip()
            url = str(prior.get("url", "")).strip()
            if title or url:
                row["representative_prior"] = " ".join(part for part in (title, url) if part)
        rows.append(row)
    return rows[:9]


def _valid_option_threat_model(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    valid_raw = [
        item
        for item in value
        if isinstance(item, dict)
        and isinstance(item.get("threat"), str)
        and item.get("threat", "").strip()
        and type(item.get("searched")) is bool
        and isinstance(item.get("risk"), str)
        and item.get("risk", "").strip() in NOVELTY_THREAT_RISKS
    ]
    if len(valid_raw) != len(value):
        return False
    rows = _option_threat_model(value)
    threats = {row["threat"] for row in rows}
    return bool(rows) and NOVELTY_REQUIRED_THREATS <= threats


def _option_threat_model(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        threat = str(item.get("threat", "")).strip()
        risk = str(item.get("risk", "")).strip()
        if not threat or not risk:
            continue
        row: dict[str, Any] = {
            "threat": threat,
            "searched": item.get("searched") if type(item.get("searched")) is bool else False,
            "risk": risk,
            "evidence_count": _safe_int(item.get("evidence_count")),
            "strongest_score": _safe_float(item.get("strongest_score")),
            "failure_mode": str(item.get("failure_mode", "")).strip(),
            "next_action": str(item.get("next_action", "")).strip(),
        }
        prior = _option_prior(item.get("representative_prior"))
        if prior:
            row["representative_prior"] = prior
        rows.append(row)
    return rows[:8]


def _valid_option_disqualifying_tests(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    valid_raw = [
        item
        for item in value
        if isinstance(item, dict)
        and isinstance(item.get("test"), str)
        and item.get("test", "").strip()
        and type(item.get("passed")) is bool
        and isinstance(item.get("risk"), str)
        and item.get("risk", "").strip() in NOVELTY_DISQUALIFYING_RISKS
        and isinstance(item.get("required_evidence"), str)
        and item.get("required_evidence", "").strip()
    ]
    if len(valid_raw) != len(value):
        return False
    rows = _option_disqualifying_tests(value)
    tests = {row["test"] for row in rows}
    return bool(rows) and NOVELTY_REQUIRED_THREATS <= tests


def _option_disqualifying_tests(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        test = str(item.get("test", "")).strip()
        risk = str(item.get("risk", "")).strip()
        required = str(item.get("required_evidence", "")).strip()
        if not test or not risk or not required:
            continue
        row: dict[str, Any] = {
            "test": test,
            "passed": item.get("passed") if type(item.get("passed")) is bool else False,
            "risk": risk,
            "required_evidence": required,
        }
        prior = _option_prior(item.get("representative_prior"))
        if prior:
            row["representative_prior"] = prior
        rows.append(row)
    return rows[:8]


def _valid_option_search_audit(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required_keys = {
        "pass_count",
        "failed_passes",
        "empty_search_passes",
        "empty_arxiv_passes",
        "empty_web_passes",
        "duplicate_only_search_passes",
        "focus_summary",
    }
    if not required_keys <= set(value):
        return False
    audit = _option_search_audit(value)
    if not audit:
        return False
    focus_summary = audit.get("focus_summary")
    if not isinstance(focus_summary, list) or not focus_summary:
        return False
    arxiv_passes = _option_search_source_passes(focus_summary, "arxiv")
    web_passes = _option_search_source_passes(focus_summary, "web")
    arxiv_unique = _option_search_source_unique_added(focus_summary, "arxiv")
    web_unique = _option_search_source_unique_added(focus_summary, "web")
    if audit.get("failed_passes", 0) != 0 or audit.get("failed_focuses"):
        return False
    if audit.get("pass_count", 0) < NOVELTY_MIN_OPTION_ARXIV_PASSES + NOVELTY_MIN_OPTION_WEB_PASSES:
        return False
    if arxiv_passes < NOVELTY_MIN_OPTION_ARXIV_PASSES or web_passes < NOVELTY_MIN_OPTION_WEB_PASSES:
        return False
    if arxiv_unique <= 0 or web_unique <= 0:
        return False
    if _option_search_source_requested_max(focus_summary, "arxiv") < NOVELTY_QUERY_RESULT_LIMIT:
        return False
    if _option_search_source_requested_max(focus_summary, "web") < NOVELTY_WEB_RESULT_LIMIT:
        return False
    coverage = _option_search_focus_coverage(focus_summary)
    if not all(coverage.get(name) for name in NOVELTY_REQUIRED_OPTION_SEARCH_FOCUS):
        return False
    return True


def _option_search_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    audit: dict[str, Any] = {}
    for key in (
        "pass_count",
        "failed_passes",
        "empty_search_passes",
        "empty_arxiv_passes",
        "empty_web_passes",
        "duplicate_only_search_passes",
    ):
        audit[key] = _safe_int(value.get(key))
    focus_coverage = value.get("focus_coverage")
    if isinstance(focus_coverage, dict):
        audit["focus_coverage"] = {
            str(key): bool(flag)
            for key, flag in focus_coverage.items()
            if isinstance(key, str) and type(flag) is bool
        }
    missing_focus_coverage = _option_strings(value.get("missing_focus_coverage", []))
    if missing_focus_coverage:
        audit["missing_focus_coverage"] = missing_focus_coverage[:16]
    for key in ("empty_focuses", "failed_focuses"):
        rows = _option_search_focus_rows(value.get(key))
        if rows:
            audit[key] = rows[:8]
    focus_summary = _option_search_focus_summary(value.get("focus_summary"))
    if focus_summary:
        audit["focus_summary"] = focus_summary
    return audit if audit["pass_count"] > 0 else {}


def _option_search_focus_rows(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        focus = str(item.get("focus", "")).strip()
        if source and focus:
            rows.append({"source": source, "focus": focus})
    return rows


def _option_search_focus_summary(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        focus = str(item.get("focus", "")).strip()
        if not source or not focus:
            continue
        rows.append(
            {
                "source": source,
                "focus": focus,
                "passes": _safe_int(item.get("passes")),
                "failed_passes": _safe_int(item.get("failed_passes")),
                "retrieved": _safe_int(item.get("retrieved")),
                "unique_added": _safe_int(item.get("unique_added")),
                "requested_results_max": _safe_int(item.get("requested_results_max")),
            }
        )
    return rows


def _option_search_source_passes(rows: list[dict[str, Any]], source: str) -> int:
    return sum(_safe_int(row.get("passes")) for row in rows if row.get("source") == source)


def _option_search_source_requested_max(rows: list[dict[str, Any]], source: str) -> int:
    return max((_safe_int(row.get("requested_results_max")) for row in rows if row.get("source") == source), default=0)


def _option_search_source_unique_added(rows: list[dict[str, Any]], source: str) -> int:
    return sum(_safe_int(row.get("unique_added")) for row in rows if row.get("source") == source)


def _option_search_focus_coverage(rows: list[dict[str, Any]]) -> dict[str, bool]:
    text = " ".join(str(row.get("focus", "")) for row in rows).lower()
    return {
        "recency": "recent" in text or "submitted" in text or "updated" in text,
        "recent_discovery": "discovery" in text,
        "architecture": "architecture" in text,
        "frontier_architecture": _novelty_focus_text_has_frontier(text),
        "method": "method" in text,
        "mechanism": "mechanism" in text,
        "evaluation": "evaluation" in text or "benchmark" in text,
        "implementation": "implementation" in text or "code" in text,
        "replication": "replication" in text or "reproduction" in text,
        "failure_modes": "failure" in text or "limitations" in text,
        "protocol": "protocol" in text or "dataset" in text,
        "exact_phrase": "exact" in text,
        "claim_collision": "claim" in text,
        "peer_review": "peer" in text or "critique" in text or "review" in text,
    }


def _novelty_frontier_focuses(arxiv_focuses: list[str], web_focuses: list[str]) -> list[str]:
    return sorted(
        focus
        for focus in [*arxiv_focuses, *web_focuses]
        if _novelty_focus_text_has_frontier(focus)
    )


def _novelty_focus_text_has_frontier(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        term in lowered
        for term in (
            "frontier",
            "sota",
            "state of the art",
            "scaling",
            "model_family",
            "model family",
        )
    )


def _option_prior(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    row: dict[str, Any] = {}
    for key in ("title", "url", "source_type"):
        text = str(value.get(key, "")).strip()
        if text:
            row[key] = text
    if "score" in value:
        row["score"] = _safe_float(value.get("score"))
    return row


def _valid_option_recent_pressure(value: Any) -> bool:
    pressure = _option_recent_pressure(value)
    if pressure.get("status") != "recent_prior_present":
        return False
    if pressure.get("recent_window") != _novelty_recent_window_label():
        return False
    if _safe_int(pressure.get("recent_evidence_count")) <= 0:
        return False
    latest_year = _safe_int(pressure.get("latest_year"))
    return latest_year >= datetime.now(UTC).year - 2


def _option_recent_pressure(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    status = str(value.get("status", "")).strip()
    recent_window = str(value.get("recent_window", "")).strip()
    if not status or not recent_window:
        return {}
    return {
        "status": status,
        "recent_window": recent_window,
        "recent_evidence_count": _safe_int(value.get("recent_evidence_count")),
        "latest_year": _safe_int(value.get("latest_year")),
        "recent_prior_titles": _option_strings(value.get("recent_prior_titles", []))[:5],
    }


def _safe_int(value: Any) -> int:
    if type(value) is bool:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed


def _safe_float(value: Any) -> float:
    if type(value) is bool:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return round(parsed, 3)


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

    skills = [_skill_tool_payload(s) for s in list_skills()]
    return json.dumps(
        {
            "ok": True,
            "skills": skills,
            "count": len(skills),
            "model_policy": "No skill chooses an interpretability model unless its JSON explicitly declares one.",
            "next_actions": [
                "Ask the user which model to study before running a skill with model_required=true.",
                "Pass model and backend explicitly to run_discovery; do not infer a benchmark model from the skill name.",
            ],
        }
    )


def tool_environment_status(_args: dict[str, Any]) -> str:
    from .cluster import load_cluster_config
    from .modal_app import modal_status
    from .skills import list_skills

    skills = [_skill_tool_payload(s) for s in list_skills()]
    return json.dumps({
        "skills": skills,
        "skill_count": len(skills),
        "skills_requiring_model": [s["name"] for s in skills if s["model_required"]],
        "model_policy": "Discovery requires an explicitly selected model unless a skill JSON declares one.",
        "modal": modal_status(),
        "cluster_configured": load_cluster_config().configured,
        "next_actions": [
            "Choose a skill/task, model, and backend explicitly before starting discovery.",
            "Use run_research first when the model, behavior, or evidence scope is not yet specified.",
        ],
    })


def _skill_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _skill_tool_payload(skill: Any) -> dict[str, Any]:
    model = _skill_text(getattr(skill, "model", ""))
    seeds = getattr(skill, "seeds", [])
    seeds = seeds if isinstance(seeds, list) else []
    return {
        "name": _skill_text(getattr(skill, "name", "")),
        "task": _skill_text(getattr(skill, "task", "")),
        "description": _skill_text(getattr(skill, "description", "")),
        "model": model,
        "declares_model": bool(model),
        "model_required": not bool(model),
        "question": _skill_text(getattr(skill, "question", "")),
        "seed_policy": ", ".join(str(seed) for seed in seeds) if seeds else "run-specific generated seeds",
        "budget": getattr(skill, "budget", {}) if isinstance(getattr(skill, "budget", {}), dict) else {},
        "stop": getattr(skill, "stop", {}) if isinstance(getattr(skill, "stop", {}), dict) else {},
    }


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
    {"name": "web_search", "description": "Search the web (returns title + url results). Uses at least 24 results per query, even if a smaller max_results is requested. Use for current information and finding sources.",
     "parameters": _obj({"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"])},
    {"name": "web_fetch", "description": "Fetch a URL and return its readable text content.",
     "parameters": _obj({"url": {"type": "string"}}, ["url"])},
    {"name": "arxiv_search", "description": "Search arXiv for papers. Uses at least 50 results per query, even if a smaller max_results is requested. sort_by: relevance | submittedDate | lastUpdatedDate. Use for literature grounding.",
     "parameters": _obj({"query": {"type": "string", "description": "arXiv query, e.g. 'cat:cs.LG AND (abs:sparse autoencoder OR abs:linear probe)'"}, "max_results": {"type": "integer"}, "sort_by": {"type": "string", "enum": ["relevance", "submittedDate", "lastUpdatedDate"]}}, ["query"])},
    {"name": "neuronpedia_search", "description": "Semantic search over SAE-feature explanations for an explicit Neuronpedia model id.",
     "parameters": _obj({"model_id": {"type": "string"}, "query": {"type": "string"}}, ["model_id", "query"])},
    {"name": "verify_novelty", "description": "Deep novelty check for a research idea using multi-pass arXiv and web searches across relevance, recency, recent discoveries, architecture variants, recent frontier architectures, model-family discoveries, method, mechanism, evaluation, implementation, replication, failure-mode, and protocol angles. Call this for each proposed research direction before presenting it.",
     "parameters": _obj({"idea": {"type": "string", "description": "the research idea/direction to novelty-check"}, "queries": {"type": "array", "items": {"type": "string"}, "description": "optional arXiv queries to probe for prior work"}}, ["idea"])},
    {"name": "present_options", "description": "Present 2-5 research directions to the user as an interactive, expandable picker and return their choice. Use this instead of writing options as prose. Every option must include detail, citations, novelty_risk, novelty_verdict, closest_prior_art, claim_readiness, comparison_matrix, novelty_threat_model, disqualifying_overlap_tests, search_audit, recent_pressure, and required_delta from verify_novelty assessment. search_audit must show successful deep arXiv and web retrieval with unique evidence from both sources plus frontier_architecture coverage; recent_pressure must show recent_prior_present in the current recent window.",
     "parameters": _obj({"options": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "object", "properties": {
         "title": {"type": "string"},
         "summary": {"type": "string", "description": "one line"},
         "detail": {"type": "string", "description": "a fuller paragraph: what the project is about"},
         "citations": {"type": "array", "items": {"type": "string"}, "description": "key paper titles/URLs"},
         "novelty": {"type": "string", "description": "legacy novelty summary from verify_novelty"},
         "novelty_risk": {"type": "string", "description": "assessment.risk from verify_novelty"},
         "novelty_verdict": {"type": "string", "description": "assessment.verdict from verify_novelty"},
         "closest_prior_art": {"type": "array", "items": {"type": "string"}, "description": "nearest prior paper titles/URLs from assessment.closest_prior_art"},
         "claim_readiness": {"type": "object", "description": "assessment.claim_readiness from verify_novelty; include status, can_claim_high_novelty, missing_checks, and next_actions."},
         "comparison_matrix": {"type": "array", "items": {"type": "object", "properties": {
             "axis": {"type": "string"},
             "covered": {"type": "boolean"},
             "evidence_count": {"type": "integer"},
             "representative_prior": {"type": "object"},
             "next_action": {"type": "string"},
         }}, "description": "assessment.comparison_matrix from verify_novelty; include per-axis covered/evidence_count/next_action fields."},
         "novelty_threat_model": {"type": "array", "items": {"type": "object", "properties": {
             "threat": {"type": "string"},
             "searched": {"type": "boolean"},
             "risk": {"type": "string"},
             "evidence_count": {"type": "integer"},
             "strongest_score": {"type": "number"},
             "representative_prior": {"type": "object"},
             "failure_mode": {"type": "string"},
             "next_action": {"type": "string"},
         }}, "description": "assessment.novelty_threat_model from verify_novelty; include exact_phrase_overlap and claim_collision rows."},
         "disqualifying_overlap_tests": {"type": "array", "items": {"type": "object", "properties": {
             "test": {"type": "string"},
             "passed": {"type": "boolean"},
             "risk": {"type": "string"},
             "representative_prior": {"type": "object"},
             "required_evidence": {"type": "string"},
         }}, "description": "assessment.disqualifying_overlap_tests from verify_novelty; include exact_phrase_overlap and claim_collision rows."},
         "search_audit": {"type": "object", "properties": {
             "pass_count": {"type": "integer"},
             "failed_passes": {"type": "integer"},
             "empty_search_passes": {"type": "integer"},
             "empty_arxiv_passes": {"type": "integer"},
             "empty_web_passes": {"type": "integer"},
             "duplicate_only_search_passes": {"type": "integer"},
             "empty_focuses": {"type": "array", "items": {"type": "object"}},
             "failed_focuses": {"type": "array", "items": {"type": "object"}},
             "focus_coverage": {"type": "object"},
             "missing_focus_coverage": {"type": "array", "items": {"type": "string"}},
             "focus_summary": {"type": "array", "items": {"type": "object"}},
         }, "description": "assessment.search_audit from verify_novelty; include pass_count, failed_passes, empty_search_passes, duplicate_only_search_passes, focus_coverage, missing_focus_coverage, and focus_summary. The audit must prove at least 10 arXiv passes at 50 results, 8 web passes at 24 results, focused deep-search coverage, unique_added evidence from both arXiv and web, and zero failed retrieval passes."},
         "recent_pressure": {"type": "object", "properties": {
             "status": {"type": "string"},
             "recent_window": {"type": "string"},
             "recent_evidence_count": {"type": "integer"},
             "latest_year": {"type": "integer"},
             "recent_prior_titles": {"type": "array", "items": {"type": "string"}},
         }, "description": "assessment.recent_pressure from verify_novelty; include status=recent_prior_present, current recent_window, recent_evidence_count, latest_year, and recent_prior_titles."},
         "required_delta": {"type": "string", "description": "specific deltas from assessment.required_delta; join multiple entries into one concise string"},
     }, "required": ["title", "summary", "detail", "citations", "novelty_risk", "novelty_verdict", "closest_prior_art", "claim_readiness", "comparison_matrix", "novelty_threat_model", "disqualifying_overlap_tests", "search_audit", "recent_pressure", "required_delta"]}}}, ["options"])},
    {"name": "run_research", "description": "Run the general prompt-to-dossier literature/research pipeline over explicit sources, URLs, memory, or a configured provider. Defaults to a four-round retrieval, critique, and gap-expansion pass unless max_rounds is explicitly set.",
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
    {"name": "run_discovery", "description": "Run the autonomous interpretability discovery loop to find/confirm mechanisms (heads/circuits) for a behaviour. The backend must be explicit so synthetic smoke data is never selected by omission.",
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
     }, ["backend"])},
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
