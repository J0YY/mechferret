from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Approximate USD per 1M tokens (input, output). Used for a live readout, not billing.
PRICING = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.5": (10.0, 30.0),
    "gpt-5": (5.0, 15.0),
    "gpt-4o": (2.5, 10.0),
}
_DEFAULT_PRICE = (5.0, 15.0)


def _price(model: str) -> tuple[float, float]:
    for key, price in PRICING.items():
        if model.startswith(key):
            return price
    return _DEFAULT_PRICE


@dataclass
class CostTracker:
    """Accumulates live token usage + USD across a chat session, per model."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, model: str, usage: dict[str, Any]) -> None:
        # Normalise Anthropic (input_tokens/output_tokens) + OpenAI (prompt/completion_tokens).
        inp = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        cache = int(usage.get("cache_read_input_tokens") or 0)
        in_price, out_price = _price(model)
        cost = (inp / 1_000_000) * in_price + (out / 1_000_000) * out_price
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cache
        self.usd += cost
        slot = self.by_model.setdefault(model, {"input": 0, "output": 0, "usd": 0.0})
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

