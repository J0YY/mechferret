from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def estimate_run_cost(run_json: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(run_json).read_text(encoding="utf-8"))
    sources = payload.get("sources", [])
    answer = payload.get("answer", "")
    text_chars = len(answer) + sum(len(source.get("text", "")) for source in sources)
    estimated_tokens = max(1, text_chars // 4)
    provider_sources = [source for source in sources if source.get("kind") in {"openai_web_search", "anthropic_research"}]
    estimated_provider_calls = len(provider_sources)
    return {
        "run_id": payload.get("run_id", ""),
        "estimated_tokens_processed": estimated_tokens,
        "estimated_provider_calls": estimated_provider_calls,
        "local_steps": len(payload.get("plan", {}).get("steps", [])),
        "note": "Cost is an estimate from artifacts; provider billing requires API response usage metadata.",
    }

