from __future__ import annotations

import json
import math
from typing import Any

from .config import MechFerretConfig, configured_api_key, configured_model, load_config
from .models import Source
from .text import compact_text
from .text import stable_id


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


def _strings(value: Any) -> list[str]:
    return [_text(item).strip() for item in _items(value) if _text(item).strip()]


def _optional_strings(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        text = _text(value).strip()
        return [text] if text else []
    return _strings(value)


def _provider_text(value: Any) -> str:
    text = _text(value).strip()
    if text:
        return text
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:  # noqa: BLE001 - provider SDK objects can have unusual reprs
        return ""


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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if type(value) is bool:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            label = _text(key).strip() or str(key)
            if label:
                cleaned[label] = _jsonable(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return _text(value)


class OpenAIWebResearch:
    """Optional live-search adapter using the OpenAI Responses API."""

    def __init__(self, model: str | None = None, config: MechFerretConfig | None = None) -> None:
        self.config = config or load_config()
        self.model = configured_model("openai", self.config, model)
        self.last_diagnostic: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(configured_api_key("openai", self.config) and self.model)

    def search_summary(self, question: str, allowed_domains: list[str] | None = None) -> Source | None:
        question_text = _text(question).strip()
        if not question_text:
            self.last_diagnostic = {"ok": False, "provider": "openai", "reason": "empty question"}
            return None
        if not self.available:
            self.last_diagnostic = {"ok": False, "provider": "openai", "model": self.model, "reason": "provider not configured"}
            return None
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.last_diagnostic = {"ok": False, "provider": "openai", "model": self.model, "reason": "openai package missing"}
            return None

        tool: dict[str, object] = {"type": "web_search"}
        domains = _optional_strings(allowed_domains)
        if domains:
            tool["filters"] = {"allowed_domains": domains}
        client = OpenAI(api_key=configured_api_key("openai", self.config))
        try:
            response = client.responses.create(
                model=self.model,
                reasoning={"effort": "low"},
                tools=[tool],
                tool_choice="auto",
                include=["web_search_call.action.sources"],
                input=(
                    "Find current, citable information for this autoresearch question. "
                    "Return compact findings with URLs inline when available.\n\n"
                    f"Question: {question_text}"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - callers decide whether live search is required
            self.last_diagnostic = {"ok": False, "provider": "openai", "model": self.model, "reason": str(exc)[:180]}
            return None
        text = _provider_text(getattr(response, "output_text", "")) or _provider_text(response)
        if not text:
            self.last_diagnostic = {"ok": False, "provider": "openai", "model": self.model, "reason": "empty provider response"}
            return None
        self.last_diagnostic = {"ok": True, "provider": "openai", "model": self.model, "reason": ""}
        return Source(
            id=stable_id("src", f"openai:{question_text}:{text[:500]}"),
            title=f"OpenAI web search: {question_text[:80]}",
            text=text,
            url="openai://responses/web_search",
            kind="openai_web_search",
            metadata={"model": self.model},
        )


class AnthropicResearch:
    """Optional Anthropic adapter for provider parity and live synthesis."""

    def __init__(self, model: str | None = None, config: MechFerretConfig | None = None) -> None:
        self.config = config or load_config()
        self.model = configured_model("anthropic", self.config, model)
        self.last_diagnostic: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(configured_api_key("anthropic", self.config) and self.model)

    def search_summary(self, question: str, allowed_domains: list[str] | None = None) -> Source | None:
        question_text = _text(question).strip()
        if not question_text:
            self.last_diagnostic = {"ok": False, "provider": "anthropic", "reason": "empty question"}
            return None
        if not self.available:
            self.last_diagnostic = {"ok": False, "provider": "anthropic", "model": self.model, "reason": "provider not configured"}
            return None
        try:
            import anthropic  # type: ignore
        except ImportError:
            self.last_diagnostic = {"ok": False, "provider": "anthropic", "model": self.model, "reason": "anthropic package missing"}
            return None

        client = anthropic.Anthropic(api_key=configured_api_key("anthropic", self.config))
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=900,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Produce a compact research brief for this question. "
                            "Include uncertainty, likely evidence needs, and concrete search directions. "
                            "Do not invent citations.\n\n"
                            f"Question: {question_text}"
                        ),
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 - callers decide whether live research is required
            self.last_diagnostic = {"ok": False, "provider": "anthropic", "model": self.model, "reason": str(exc)[:180]}
            return None
        blocks = _items(getattr(message, "content", []))
        text = "\n".join(_provider_text(getattr(block, "text", "")) for block in blocks).strip()
        if not text:
            self.last_diagnostic = {"ok": False, "provider": "anthropic", "model": self.model, "reason": "empty provider response"}
            return None
        self.last_diagnostic = {"ok": True, "provider": "anthropic", "model": self.model, "reason": ""}
        return Source(
            id=stable_id("src", f"anthropic:{question_text}:{text[:500]}"),
            title=f"Anthropic research brief: {question_text[:80]}",
            text=text,
            url="anthropic://messages/research_brief",
            kind="anthropic_research",
            metadata={"model": self.model},
        )


def make_research_adapter(
    provider: str,
    model: str | None = None,
    config: MechFerretConfig | None = None,
) -> OpenAIWebResearch | AnthropicResearch | None:
    cfg = config or load_config()
    selected = _text(provider).strip() or "auto"
    if selected == "auto":
        selected = _text(cfg.default_provider).strip()
    if selected == "openai":
        return OpenAIWebResearch(model, cfg)
    if selected == "anthropic":
        return AnthropicResearch(model, cfg)
    return None


def synthesize_answer_with_provider(
    provider: str,
    model: str | None,
    *,
    question: str,
    claims: list[Any],
    evidence: list[Any],
    gaps: list[str],
    discoveries: list[Any] | None = None,
    experiments: list[Any] | None = None,
    config: MechFerretConfig | None = None,
) -> tuple[str, dict[str, str]]:
    """Return provider-authored synthesis text, or ("", diagnostics)."""

    cfg = config or load_config()
    provider_name = _text(provider).strip() or "auto"
    selected = cfg.default_provider if provider_name == "auto" else provider_name
    if selected not in {"openai", "anthropic"}:
        return "", {"provider": selected, "reason": "local provider has no model-authored synthesis"}
    key = configured_api_key(selected, cfg)
    if not key:
        return "", {"provider": selected, "reason": f"missing {selected} API key"}
    selected_model = configured_model(selected, cfg, _text(model).strip() or None)
    if not selected_model:
        return "", {"provider": selected, "reason": f"missing {selected} model"}
    prompt = _answer_prompt(question, claims, evidence, gaps, discoveries or [], experiments or [])
    try:
        if selected == "anthropic":
            text = _call_anthropic(selected_model, key, prompt)
        else:
            text = _call_openai(selected_model, key, prompt)
    except Exception as exc:  # noqa: BLE001 - callers record this as provenance
        return "", {"provider": selected, "model": selected_model, "reason": str(exc)[:180]}
    text = text.strip()
    return text, {"provider": selected, "model": selected_model, "reason": "" if text else "empty provider response"}


def _answer_prompt(
    question: str,
    claims: list[Any],
    evidence: list[Any],
    gaps: list[str],
    discoveries: list[Any],
    experiments: list[Any],
) -> str:
    payload = {
        "question": _text(question).strip(),
        "claims": [
            {
                "id": _text(getattr(claim, "id", "")).strip(),
                "text": _text(getattr(claim, "text", "")).strip(),
                "citations": _strings(getattr(claim, "citations", [])),
                "confidence": _number(getattr(claim, "confidence", 0)),
                "support_score": _number(getattr(claim, "support_score", 0)),
                "quality_flags": _strings(getattr(claim, "quality_flags", [])),
            }
            for claim in sorted(_items(claims), key=lambda item: _number(getattr(item, "confidence", 0)), reverse=True)[:18]
            if _text(getattr(claim, "id", "")).strip() or _text(getattr(claim, "text", "")).strip()
        ],
        "evidence": [
            {
                "id": _text(getattr(chunk, "id", "")).strip(),
                "source_id": _text(getattr(chunk, "source_id", "")).strip(),
                "title": _text(getattr(chunk, "title", "")).strip(),
                "url": _text(getattr(chunk, "url", "")).strip(),
                "score": _number(getattr(chunk, "score", 0)),
                "text": compact_text(_text(getattr(chunk, "text", "")), 700),
            }
            for chunk in sorted(_items(evidence), key=lambda item: _number(getattr(item, "score", 0)), reverse=True)[:24]
            if _text(getattr(chunk, "id", "")).strip() or _text(getattr(chunk, "text", "")).strip()
        ],
        "discoveries": [
            {
                "id": _text(getattr(discovery, "id", "")).strip(),
                "statement": _text(getattr(discovery, "statement", "")).strip(),
                "confidence": _number(getattr(discovery, "confidence", 0)),
                "effect_size": _number(getattr(discovery, "effect_size", 0)),
                "reproducibility": _number(getattr(discovery, "reproducibility", 0)),
                "novelty": _number(getattr(discovery, "novelty", 0)),
                "supporting_experiments": _strings(getattr(discovery, "supporting_experiments", [])),
            }
            for discovery in _items(discoveries)[:10]
            if _text(getattr(discovery, "id", "")).strip() or _text(getattr(discovery, "statement", "")).strip()
        ],
        "experiments": [
            {
                "id": _text(getattr(experiment, "id", "")).strip(),
                "probe": _text(getattr(experiment, "probe", "")).strip(),
                "target": _jsonable(getattr(experiment, "target", {})),
                "effect_size": _number(getattr(experiment, "effect_size", 0)),
                "baseline": _number(getattr(experiment, "baseline", 0)),
                "per_seed": [_number(item) for item in _items(getattr(experiment, "per_seed", []))],
                "significant": _flag(getattr(experiment, "significant", False)),
                "reproduced": _flag(getattr(experiment, "reproduced", False)),
            }
            for experiment in _items(experiments)[:24]
            if _text(getattr(experiment, "status", "")).strip() == "ran"
            and (_text(getattr(experiment, "id", "")).strip() or _text(getattr(experiment, "probe", "")).strip())
        ],
        "gaps": _strings(gaps)[:8],
    }
    return (
        "Write a concise, evidence-grounded research synthesis for the user. "
        "Use only the supplied ledger. Do not invent citations, experiments, effect sizes, or claims. "
        "Cite claim/evidence IDs inline where useful. If the evidence is insufficient, say exactly what is missing. "
        "Return plain Markdown, not JSON.\n\n"
        f"RUN LEDGER:\n{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)[:24000]}"
    )


def _call_anthropic(model: str, key: str, prompt: str) -> str:
    from .agent import _extract_anthropic_content, _extract_provider_text, _http_post

    data = _http_post(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 1400,
            "messages": [{"role": "user", "content": prompt}],
        },
        {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    content, error = _extract_anthropic_content(data)
    if error:
        raise RuntimeError(f"provider response envelope: {error}")
    return _extract_provider_text(content)


def _call_openai(model: str, key: str, prompt: str) -> str:
    from .agent import _extract_openai_message, _extract_provider_text, _http_post

    data = _http_post(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "You write rigorous, evidence-grounded research syntheses."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 1400,
        },
        {"authorization": f"Bearer {key}", "content-type": "application/json"},
    )
    message, error = _extract_openai_message(data)
    if error:
        raise RuntimeError(f"provider response envelope: {error}")
    return _extract_provider_text(message.get("content"))
