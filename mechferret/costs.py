from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Approximate USD per 1M tokens (input, output). Used for a live readout, not billing.
PRICING = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5": (5.0, 15.0),
    "gpt-4o": (2.5, 10.0),
}
_DEFAULT_PRICE = (5.0, 15.0)


def _model_key(model: Any) -> str:
    if isinstance(model, str) and model.strip():
        return model
    return "unknown"


def _price(model: Any) -> tuple[float, float]:
    model_key = _model_key(model)
    for key, price in PRICING.items():
        if model_key.startswith(key):
            return price
    return _DEFAULT_PRICE


def _token_count(value: Any) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(parsed) or parsed <= 0:
        return 0
    return int(parsed)


@dataclass
class CostTracker:
    """Accumulates live token usage + USD across a chat session, per model."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, model: Any, usage: Any) -> None:
        # Normalise Anthropic (input_tokens/output_tokens) + OpenAI (prompt/completion_tokens).
        usage = usage if isinstance(usage, dict) else {}
        model_key = _model_key(model)
        inp = _token_count(usage.get("input_tokens") or usage.get("prompt_tokens"))
        out = _token_count(usage.get("output_tokens") or usage.get("completion_tokens"))
        cache = _token_count(usage.get("cache_read_input_tokens"))
        in_price, out_price = _price(model_key)
        cost = (inp / 1_000_000) * in_price + (out / 1_000_000) * out_price
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cache
        self.usd += cost
        slot = self.by_model.setdefault(model_key, {"input": 0, "output": 0, "usd": 0.0})
        slot["input"] += inp
        slot["output"] += out
        slot["usd"] += cost

    def format_total(self) -> str:
        parts = [
            f"${self.usd:.4f}",
            f"{self.input_tokens:,} in",
            f"{self.output_tokens:,} out",
        ]
        if self.cache_read_tokens:
            parts.append(f"{self.cache_read_tokens:,} cache-read")
        return " · ".join(parts)


def estimate_run_cost(run_json: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(run_json).read_text(encoding="utf-8"))
    payload = payload if isinstance(payload, dict) else {}
    sources = payload.get("sources", [])
    sources = sources if isinstance(sources, list) else []
    source_rows = [source for source in sources if isinstance(source, dict)]
    answer = payload.get("answer", "")
    text_chars = _text_len(answer) + sum(_text_len(source.get("text", "")) for source in source_rows)
    estimated_tokens = max(1, text_chars // 4)
    provider_sources = [
        source for source in source_rows if source.get("kind") in {"openai_web_search", "anthropic_research"}
    ]
    plan = payload.get("plan", {})
    plan = plan if isinstance(plan, dict) else {}
    steps = plan.get("steps", [])
    steps = steps if isinstance(steps, list) else []
    estimated_provider_calls = len(provider_sources)
    return {
        "run_id": payload.get("run_id", "") if isinstance(payload.get("run_id", ""), str) else "",
        "estimated_tokens_processed": estimated_tokens,
        "estimated_provider_calls": estimated_provider_calls,
        "local_steps": len(steps),
        "note": "Cost is an estimate from artifacts; provider billing requires API response usage metadata.",
    }


def _text_len(value: Any) -> int:
    return len(value) if isinstance(value, str) else 0
