from __future__ import annotations

import os

from .models import Source
from .text import stable_id


class OpenAIWebResearch:
    """Optional live-search adapter using the OpenAI Responses API."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("MECHFERRET_OPENAI_MODEL", "gpt-5")

    @property
    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

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
        client = OpenAI()
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

