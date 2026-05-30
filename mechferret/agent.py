"""Conversational research agent.

The REPL pipes your prompts to a model (Claude or GPT). The model holds the
conversation and decides when to reach for MechFerret's *architecture/agent*
parts — the discovery loop, the skills, the environment — which are exposed to
it as tools. So "hello" just gets a reply; "find the IOI circuit in gpt2" makes
the model call ``run_discovery`` and narrate the result.

Provider calls go over stdlib ``urllib`` (no SDK needs installing into the pipx
venv). Anthropic and OpenAI tool-use are both supported.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import permissions
from .config import configured_api_key, configured_model, load_config
from .costs import CostTracker
from .tools import TOOL_SPECS, run_tool as _run_tool, tool_meta

BASE_SYSTEM_PROMPT = """You are MechFerret, an agentic coding assistant specialised for mechanistic-interpretability research (like Claude Code / Codex, but for interp).

You have real tools: run shell commands (bash), read/write/edit files, glob and
grep the codebase, search the web, fetch URLs, search arXiv, query Neuronpedia
for SAE features, run the interpretability discovery loop, and list skills. Use
them — don't guess. When the user wants to investigate a model's internals, plan
a paper, or run experiments, gather evidence with the search tools, then write
and run real code (TransformerLens / SAELens / nnsight) with bash, reading
results back. Prefer concrete action over speculation.

Be precise about experimental rigor: controls, multiple seeds, reproducibility,
and triangulation across independent probes before claiming a mechanism. Keep
replies concise; cite files and paper URLs you actually read."""

MAX_TOOL_STEPS = 12
MAX_TOKENS = int(os.getenv("MECHFERRET_MAX_TOKENS", "4096"))


def build_system_prompt() -> str:
    """Assemble the system prompt from base + enabled tools + project memory + git."""

    sections = [BASE_SYSTEM_PROMPT]
    tool_lines = ", ".join(t["name"] for t in TOOL_SPECS)
    sections.append(f"Available tools: {tool_lines}.")

    for fname in ("MECHFERRET.md", "CLAUDE.md"):
        path = Path.cwd() / fname
        if path.is_file():
            sections.append(f"Project notes ({fname}):\n" + path.read_text(encoding="utf-8", errors="ignore")[:4000])
            break

    try:
        status = subprocess.run(["git", "status", "-s"], capture_output=True, text=True, timeout=5)
        if status.returncode == 0 and status.stdout.strip():
            sections.append("Git status (short):\n" + status.stdout.strip()[:1500])
    except (OSError, subprocess.TimeoutExpired):
        pass

    mechanisms = _recall_mechanisms()
    if mechanisms:
        sections.append("Previously confirmed mechanisms (from memory):\n" + mechanisms)
    return "\n\n".join(sections)


def _recall_mechanisms(limit: int = 8) -> str:
    try:
        from .memory import ResearchMemory

        mem = ResearchMemory(".mechferret/memory.sqlite")
        try:
            rows = mem.conn.execute(
                "select text from claims where stance='discovery' order by created_at desc limit ?",
                (limit,),
            ).fetchall()
        finally:
            mem.close()
        return "\n".join(f"- {r['text']}" for r in rows)
    except Exception:  # noqa: BLE001 - memory is best-effort context
        return ""


# Tools live in tools.py (the full Claude-Code-style suite). TOOL_SPECS is the
# provider-neutral schema list; _run_tool dispatches by name.


# --- provider configuration ----------------------------------------------------------

def active_provider() -> tuple[str, str, str]:
    """Return (provider, model, api_key); provider is '' if nothing is configured."""

    config = load_config()
    provider = config.default_provider
    if provider in {"anthropic", "openai"}:
        key = configured_api_key(provider, config)
        if key:
            return provider, configured_model(provider, config), key
    for candidate in ("anthropic", "openai"):
        key = configured_api_key(candidate, config)
        if key:
            return candidate, configured_model(candidate, config), key
    return "", "", ""


def is_configured() -> bool:
    return bool(active_provider()[0])


# --- the agent -----------------------------------------------------------------------

class Agent:
    """Holds a conversation and runs the provider-native tool loop."""

    def __init__(self, on_tool: Callable[[str, dict], None] | None = None) -> None:
        self.provider, self.model, self._key = active_provider()
        self.on_tool = on_tool or (lambda name, args: None)
        self.confirm: Callable[[str, dict, str], bool] | None = None
        self.permission_mode = "auto"  # auto | plan
        self.messages: list[dict[str, Any]] = []  # provider-native message history
        self.cost = CostTracker()
        self.denials: list[str] = []

    @property
    def configured(self) -> bool:
        return bool(self.provider)

    def reload(self) -> None:
        self.provider, self.model, self._key = active_provider()
        self.messages = []

    def send(self, user_text: str) -> str:
        if not self.configured:
            raise RuntimeError("No provider configured.")
        if self.provider == "anthropic":
            return self._send_anthropic(user_text)
        return self._send_openai(user_text)

    def _dispatch(self, name: str, args: dict) -> str:
        """Permission-gate a tool call, then run it."""

        meta = tool_meta(name)
        decision = permissions.decide(
            name, args, read_only=meta["read_only"], permission_class=meta["permission"], mode=self.permission_mode
        )
        if decision.behavior in {"ask", "deny"}:
            approved = bool(self.confirm and self.confirm(name, args, decision.reason)) if decision.behavior == "ask" else False
            if not approved:
                self.denials.append(name)
                return json.dumps({"denied": True, "reason": decision.reason or "not approved", "tool": name})
        self.on_tool(name, args)
        return _run_tool(name, args)

    # --- Anthropic ------------------------------------------------------------------

    def _send_anthropic(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOL_SPECS
        ]
        final_text: list[str] = []
        for _ in range(MAX_TOOL_STEPS):
            payload = {
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "system": build_system_prompt(),
                "messages": self.messages,
                "tools": tools,
            }
            data = _http_post(
                "https://api.anthropic.com/v1/messages",
                payload,
                {
                    "x-api-key": self._key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            if isinstance(data.get("usage"), dict):
                self.cost.add(self.model, data["usage"])
            content = data.get("content", [])
            self.messages.append({"role": "assistant", "content": content})
            tool_results = []
            for block in content:
                if block.get("type") == "text":
                    final_text.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    result = self._dispatch(block["name"], block.get("input", {}))
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block["id"], "content": result}
                    )
            if data.get("stop_reason") == "tool_use" and tool_results:
                self.messages.append({"role": "user", "content": tool_results})
                continue
            break
        return "\n".join(t for t in final_text if t).strip()

    # --- OpenAI ---------------------------------------------------------------------

    def _send_openai(self, user_text: str) -> str:
        if not self.messages:
            self.messages.append({"role": "system", "content": build_system_prompt()})
        self.messages.append({"role": "user", "content": user_text})
        tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in TOOL_SPECS
        ]
        final_text: list[str] = []
        for _ in range(MAX_TOOL_STEPS):
            payload = {"model": self.model, "messages": self.messages, "tools": tools, "tool_choice": "auto", "max_tokens": MAX_TOKENS}
            data = _http_post(
                "https://api.openai.com/v1/chat/completions",
                payload,
                {"authorization": f"Bearer {self._key}", "content-type": "application/json"},
            )
            if isinstance(data.get("usage"), dict):
                self.cost.add(self.model, data["usage"])
            message = data["choices"][0]["message"]
            self.messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if message.get("content"):
                final_text.append(message["content"])
            if not tool_calls:
                break
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._dispatch(name, args)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        return "\n".join(t for t in final_text if t).strip()


def _http_post(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"provider HTTP {exc.code}: {detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"network error calling provider: {exc}") from exc
