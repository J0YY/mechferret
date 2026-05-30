from __future__ import annotations

import os

from .config import MechFerretConfig, configured_api_key, configured_model, load_config
from .models import Source
from .text import stable_id


class OpenAIWebResearch:
    """Optional live-search adapter using the OpenAI Responses API."""

    def __init__(self, model: str | None = None, config: MechFerretConfig | None = None) -> None:
        self.config = config or load_config()
        self.model = configured_model("openai", self.config, model)

    @property
    def available(self) -> bool:
        return bool(configured_api_key("openai", self.config))

    def search_summary(self, question: str, allowed_domains: list[str] | None = None) -> Source | None:
        if not self.available:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            return None

        tool: dict[str, object] = {"type": "web_search"}
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}
        client = OpenAI(api_key=configured_api_key("openai", self.config))
        response = client.responses.create(
            model=self.model,
            reasoning={"effort": "low"},
            tools=[tool],
            tool_choice="auto",
            include=["web_search_call.action.sources"],
            input=(
                "Find current, citable information for this autoresearch question. "
                "Return compact findings with URLs inline when available.\n\n"
                f"Question: {question}"
            ),
        )
        text = getattr(response, "output_text", "") or str(response)
        if not text.strip():
            return None
        return Source(
            id=stable_id("src", f"openai:{question}:{text[:500]}"),
            title=f"OpenAI web search: {question[:80]}",
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

    @property
    def available(self) -> bool:
        return bool(configured_api_key("anthropic", self.config))

    def search_summary(self, question: str, allowed_domains: list[str] | None = None) -> Source | None:
        if not self.available:
            return None
        try:
            import anthropic  # type: ignore
        except ImportError:
            return None

        client = anthropic.Anthropic(api_key=configured_api_key("anthropic", self.config))
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
                        f"Question: {question}"
                    ),
                }
            ],
        )
        blocks = getattr(message, "content", [])
        text = "\n".join(getattr(block, "text", "") for block in blocks).strip()
        if not text:
            return None
        return Source(
            id=stable_id("src", f"anthropic:{question}:{text[:500]}"),
            title=f"Anthropic research brief: {question[:80]}",
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
    selected = provider
    if selected == "auto":
        selected = cfg.default_provider
    if selected == "openai":
        return OpenAIWebResearch(model, cfg)
    if selected == "anthropic":
        return AnthropicResearch(model, cfg)
    return None
