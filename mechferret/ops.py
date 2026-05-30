from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import PROVIDERS, default_config_path, load_config
from .memory import ResearchMemory
from .registry import all_items
from .sources import example_corpus_path


def doctor() -> dict[str, Any]:
    config = load_config()
    checks = [
        check("python_version", sys.version_info >= (3, 11), ".".join(map(str, sys.version_info[:3]))),
        check("example_corpus", example_corpus_path().exists(), str(example_corpus_path())),
        check("registry_items", len(all_items()) >= 10, str(len(all_items()))),
        check("config_path", True, str(default_config_path())),
        check("openai_package", importlib.util.find_spec("openai") is not None, "optional", optional=True),
        check("anthropic_package", importlib.util.find_spec("anthropic") is not None, "optional", optional=True),
    ]
    for provider in sorted(PROVIDERS):
        env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        configured = bool(os.getenv(env_name) or config.providers.get(provider, None) and config.providers[provider].api_key)
        checks.append(check(f"{provider}_key", configured, "configured" if configured else "missing", optional=True))
    return {"passed": all(item["passed"] or item["optional"] for item in checks), "checks": checks}


def check(name: str, passed: bool, detail: str, optional: bool = False) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "optional": optional}


def print_doctor() -> None:
    result = doctor()
    print(f"Doctor: {'PASS' if result['passed'] else 'WARN'}")
    for item in result["checks"]:
        marker = "ok" if item["passed"] else ("optional" if item["optional"] else "warn")
        print(f"{marker:8} {item['name']}: {item['detail']}")


def memory_summary(db_path: str | Path) -> dict[str, int]:
    memory = ResearchMemory(db_path)
    try:
        runs = memory.conn.execute("select count(*) from runs").fetchone()[0]
        claims = memory.conn.execute("select count(*) from claims").fetchone()[0]
        sources = memory.conn.execute("select count(*) from sources").fetchone()[0]
        return {"runs": runs, "claims": claims, "sources": sources}
    finally:
        memory.close()


def memory_recent(db_path: str | Path, limit: int = 10) -> list[dict[str, Any]]:
    memory = ResearchMemory(db_path)
    try:
        rows = memory.conn.execute(
            "select id, question, metrics_json, artifacts_json, created_at from runs order by created_at desc limit ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "question": row["question"],
                "metrics": json.loads(row["metrics_json"]),
                "artifacts": json.loads(row["artifacts_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        memory.close()


def memory_clear(db_path: str | Path) -> None:
    path = Path(db_path)
    if path.exists():
        path.unlink()


def summarize_run_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "run_id": payload.get("run_id", ""),
        "question": payload.get("question", ""),
        "readiness_score": payload.get("metrics", {}).get("readiness_score", 0),
        "claims": len(payload.get("claims", [])),
        "evidence": len(payload.get("evidence", [])),
        "gaps": payload.get("gaps", []),
        "artifacts": payload.get("artifacts", {}),
    }
