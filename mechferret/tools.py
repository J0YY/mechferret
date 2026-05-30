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
import subprocess
from pathlib import Path
from typing import Any, Callable

MAX_OUTPUT = 12000
PERSIST_THRESHOLD = 16000  # results larger than this are written to disk, not truncated
RESULTS_DIR = Path(".mechferret/tool_results")


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} chars]"


def _persist_if_large(name: str, result: str) -> str:
    """Large tool results are saved whole to disk; the model gets a preview + path."""

    if len(result) <= PERSIST_THRESHOLD:
        return result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(result.encode("utf-8", "ignore")).hexdigest()[:12]
    path = RESULTS_DIR / f"{name}_{digest}.txt"
    path.write_text(result, encoding="utf-8")
    preview = result[:MAX_OUTPUT]
    return (
        f"{preview}\n… [full {len(result)} chars saved to {path}; "
        f"read_file that path for the rest]"
    )


# --- coding tools -------------------------------------------------------------------

def tool_bash(args: dict[str, Any]) -> str:
    cmd = args.get("command", "")
    timeout = int(args.get("timeout", 120))
    if not cmd:
        return "error: no command"
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"
    out = proc.stdout or ""
    err = proc.stderr or ""
    body = out + (f"\n[stderr]\n{err}" if err.strip() else "")
    return _truncate(f"[exit {proc.returncode}]\n{body}".strip())


def tool_read_file(args: dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser()
    if not path.is_file():
        return f"error: not a file: {path}"
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", 2000))
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    chunk = lines[offset: offset + limit]
    numbered = "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(chunk))
    return _truncate(numbered or "(empty)")


def tool_write_file(args: dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args.get("content", ""), encoding="utf-8")
    return f"wrote {len(args.get('content', ''))} chars to {path}"


def tool_edit_file(args: dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser()
    if not path.is_file():
        return f"error: not a file: {path}"
    text = path.read_text(encoding="utf-8")
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if count > 1 and not args.get("replace_all"):
        return f"error: old_string appears {count} times; pass replace_all=true or add context"
    text = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    return f"edited {path} ({'all ' + str(count) if args.get('replace_all') else '1'} occurrence(s))"


def tool_list_dir(args: dict[str, Any]) -> str:
    path = Path(args.get("path", ".")).expanduser()
    if not path.is_dir():
        return f"error: not a directory: {path}"
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    return _truncate("\n".join(("[dir] " if p.is_dir() else "      ") + p.name for p in entries) or "(empty)")


def tool_glob(args: dict[str, Any]) -> str:
    base = Path(args.get("path", ".")).expanduser()
    matches = sorted(str(p) for p in base.glob(args["pattern"]))
    return _truncate("\n".join(matches) or "(no matches)")


def tool_grep(args: dict[str, Any]) -> str:
    pattern = args["pattern"]
    path = args.get("path", ".")
    glob = args.get("glob")
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

    results = web_search(args["query"], max_results=int(args.get("max_results", 8)))
    return json.dumps(results) if results else "(no results)"


def tool_web_fetch(args: dict[str, Any]) -> str:
    from .knowledge import web_fetch

    return _truncate(web_fetch(args["url"], max_chars=int(args.get("max_chars", 6000))))


def tool_arxiv_search(args: dict[str, Any]) -> str:
    from .knowledge import search_arxiv

    total, papers = search_arxiv(
        args["query"], max_results=int(args.get("max_results", 8)), sort_by=args.get("sort_by", "relevance")
    )
    return json.dumps({"total": total, "papers": papers})


def tool_neuronpedia_search(args: dict[str, Any]) -> str:
    from .knowledge import neuronpedia_search_explanations

    return json.dumps(neuronpedia_search_explanations(args["model_id"], args["query"]))


# --- interp tools -------------------------------------------------------------------

def tool_run_discovery(args: dict[str, Any]) -> str:
    from .discovery import DiscoveryController

    run = DiscoveryController().run(
        question=args.get("question", ""),
        skill=args.get("skill"),
        task=args.get("task"),
        model=args.get("model", "gpt2"),
        out_dir=args.get("out_dir", "runs/agent"),
    )
    return json.dumps({
        "discoveries": [
            {"statement": d.statement, "confidence": d.confidence, "effect_size": d.effect_size,
             "reproducibility": d.reproducibility, "novelty": d.novelty}
            for d in run.discoveries
        ],
        "metrics": {k: run.metrics.get(k) for k in ("rigor_score", "readiness_score", "confirmed_mechanisms", "experiments_run")},
        "report_html": run.artifacts.get("html"),
    })


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


# --- registry -----------------------------------------------------------------------

def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "bash", "description": "Run a shell command in the working directory and return its output. Use for running code, tests, git, installing deps, launching experiments.",
     "parameters": _obj({"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds (default 120)"}}, ["command"])},
    {"name": "read_file", "description": "Read a UTF-8 text file with line numbers.",
     "parameters": _obj({"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"])},
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
     "parameters": _obj({"url": {"type": "string"}, "max_chars": {"type": "integer"}}, ["url"])},
    {"name": "arxiv_search", "description": "Search arXiv for papers. sort_by: relevance | submittedDate | lastUpdatedDate. Use for literature grounding.",
     "parameters": _obj({"query": {"type": "string", "description": "arXiv query, e.g. 'cat:cs.LG AND (abs:sparse autoencoder OR abs:linear probe)'"}, "max_results": {"type": "integer"}, "sort_by": {"type": "string", "enum": ["relevance", "submittedDate", "lastUpdatedDate"]}}, ["query"])},
    {"name": "neuronpedia_search", "description": "Semantic search over SAE-feature explanations for a model on Neuronpedia (e.g. model_id 'gpt2-small').",
     "parameters": _obj({"model_id": {"type": "string"}, "query": {"type": "string"}}, ["model_id", "query"])},
    {"name": "run_discovery", "description": "Run the autonomous interpretability discovery loop to find/confirm mechanisms (heads/circuits) for a behaviour.",
     "parameters": _obj({"question": {"type": "string"}, "skill": {"type": "string", "enum": ["ioi-circuit", "find-induction-heads", "logit-lens-sweep", "factual-recall-trace"]}, "task": {"type": "string", "enum": ["ioi", "induction", "greater_than", "factual_recall"]}, "model": {"type": "string"}}, [])},
    {"name": "list_skills", "description": "List available interpretability playbooks/skills.", "parameters": _obj({}, [])},
    {"name": "environment_status", "description": "Report skills, Modal status, and cluster configuration.", "parameters": _obj({}, [])},
]

HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "glob": tool_glob,
    "grep": tool_grep,
    "web_search": tool_web_search,
    "web_fetch": tool_web_fetch,
    "arxiv_search": tool_arxiv_search,
    "neuronpedia_search": tool_neuronpedia_search,
    "run_discovery": tool_run_discovery,
    "list_skills": tool_list_skills,
    "environment_status": tool_environment_status,
}

# Per-tool metadata for the permission system. read_only tools never prompt.
META: dict[str, dict[str, Any]] = {
    "bash": {"read_only": False, "permission": "exec"},
    "read_file": {"read_only": True, "permission": "local"},
    "write_file": {"read_only": False, "permission": "write"},
    "edit_file": {"read_only": False, "permission": "write"},
    "list_dir": {"read_only": True, "permission": "local"},
    "glob": {"read_only": True, "permission": "local"},
    "grep": {"read_only": True, "permission": "local"},
    "web_search": {"read_only": True, "permission": "network"},
    "web_fetch": {"read_only": True, "permission": "network"},
    "arxiv_search": {"read_only": True, "permission": "network"},
    "neuronpedia_search": {"read_only": True, "permission": "network"},
    "run_discovery": {"read_only": False, "permission": "gpu"},
    "list_skills": {"read_only": True, "permission": "local"},
    "environment_status": {"read_only": True, "permission": "local"},
}


def tool_meta(name: str) -> dict[str, Any]:
    return META.get(name, {"read_only": False, "permission": "local"})


def run_tool(name: str, args: dict[str, Any]) -> str:
    handler = HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        return _persist_if_large(name, handler(args))
    except Exception as exc:  # noqa: BLE001 - report failures back to the model
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
