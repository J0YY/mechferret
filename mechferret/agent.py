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
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import permissions, sessions
from .config import configured_api_key, configured_model, load_config
from .costs import CostTracker
from .tools import all_specs, run_tool as _run_tool, tool_meta

BASE_SYSTEM_PROMPT = """You are MechFerret, an agentic coding assistant specialised for mechanistic-interpretability research (like Claude Code / Codex, but for interp).

You have real tools: run shell commands (bash), read/write/edit files, glob and
grep the codebase, search the web, fetch URLs, search arXiv, query Neuronpedia
for SAE features, verify the novelty of an idea, present interactive options, run
the interpretability discovery loop, and list skills. Use them — don't guess.
When the user wants to investigate a model's internals, plan a paper, or run
experiments, gather evidence with the search tools, then write and run real code
(TransformerLens / SAELens / nnsight) with bash, reading results back.

Workflow when the user wants to plan research:
1. Gather evidence with retrieval tools (arxiv_search, web_search, web_fetch).
2. For each candidate direction, call verify_novelty to check whether prior
   papers already did it; fold the verdict into the proposal.
3. Call present_options with 2-5 concrete directions (each with a one-line
   summary, a fuller detail paragraph, key citations, and the novelty verdict)
   so the user can browse and pick. Do NOT write the options out as prose.

OUTPUT RULES (important):
- Plain text only. No markdown: no #, *, **, backticks, tables, or '-' bullets.
- Be concise. A few short paragraphs, not an essay. Lead with the answer.
- Cite paper URLs and file paths you actually read, inline.
- Be precise about rigor: controls, multiple seeds, reproducibility, and
  triangulation across independent probes before claiming a mechanism."""

MAX_TOOL_STEPS = 12
MAX_TOKENS = int(os.getenv("MECHFERRET_MAX_TOKENS", "4096"))
COMPACT_CHAR_THRESHOLD = int(os.getenv("MECHFERRET_COMPACT_CHARS", "240000"))  # ~60k tokens
COMPACT_KEEP_LAST = 4

COMPACT_SYSTEM = (
    "You are compacting a mechanistic-interpretability research session to free context. "
    "Write a dense summary that MUST retain: every confirmed mechanism (layer.head + role + "
    "effect size + seeds), open hypotheses and next experiments, key file paths and decisions, "
    "and any goal/acceptance bar. Drop chit-chat. Output only the summary."
)


def build_system_prompt() -> str:
    """Assemble the system prompt from base + enabled tools + project memory + git."""

    sections = [BASE_SYSTEM_PROMPT]
    tool_lines = ", ".join(t["name"] for t in all_specs())
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
            rows = mem.recent_mechanisms(limit)
        finally:
            mem.close()
        return "\n".join(f"- {r['statement']} (effect {r['effect_size']:.2f}, repro {r['reproducibility']:.2f})" for r in rows)
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
        self.on_text: Callable[[str], None] = lambda text: None  # incremental assistant text
        self.confirm: Callable[[str, dict, str], bool] | None = None
        self.on_options: Callable[[list], str] | None = None  # interactive option picker
        self.permission_mode = "auto"  # auto | plan
        self.messages: list[dict[str, Any]] = []  # provider-native message history
        self.cost = CostTracker()
        self.denials: list[str] = []
        self.session_id = sessions.new_session_id()
        self.abort = threading.Event()

    @property
    def configured(self) -> bool:
        return bool(self.provider)

    def reload(self) -> None:
        self.provider, self.model, self._key = active_provider()
        self.messages = []

    def send(self, user_text: str) -> str:
        if not self.configured:
            raise RuntimeError("No provider configured.")
        self.abort.clear()
        self._maybe_compact()
        try:
            reply = self._send_anthropic(user_text) if self.provider == "anthropic" else self._send_openai(user_text)
        except KeyboardInterrupt:
            self._persist()  # flush transcript on abort
            raise
        self._persist()
        return reply

    def _persist(self) -> None:
        try:
            sessions.save_session(
                self.session_id, self.provider, self.model, self.messages,
                {"usd": self.cost.usd, "input": self.cost.input_tokens, "output": self.cost.output_tokens},
            )
        except Exception:  # noqa: BLE001 - transcript persistence is best-effort
            pass

    # --- context compaction --------------------------------------------------------

    def _approx_chars(self) -> int:
        try:
            return len(json.dumps(self.messages))
        except (TypeError, ValueError):
            return 0

    def _maybe_compact(self) -> None:
        if self._approx_chars() > COMPACT_CHAR_THRESHOLD:
            self.compact()

    def compact(self) -> str:
        """Summarise older turns into one boundary message, keeping the last few."""

        if len(self.messages) <= COMPACT_KEEP_LAST:
            return "nothing to compact"
        head = self.messages[:-COMPACT_KEEP_LAST]
        tail = self.messages[-COMPACT_KEEP_LAST:]
        summary = self._summarize(_render_messages(head))
        boundary = {"role": "user", "content": f"[Summary of earlier conversation — retain these facts]\n{summary}"}
        if self.provider == "openai":
            system = self.messages[:1] if self.messages and self.messages[0].get("role") == "system" else []
            self.messages = system + [boundary] + tail
        else:
            self.messages = [boundary] + tail
        return summary

    def _summarize(self, convo: str) -> str:
        convo = convo[:60000]
        if self.provider == "anthropic":
            data = _http_post(
                "https://api.anthropic.com/v1/messages",
                {"model": self.model, "max_tokens": 1024, "system": COMPACT_SYSTEM,
                 "messages": [{"role": "user", "content": convo}]},
                {"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            )
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        data = _http_post(
            "https://api.openai.com/v1/chat/completions",
            {"model": self.model, "messages": [{"role": "system", "content": COMPACT_SYSTEM}, {"role": "user", "content": convo}], "max_completion_tokens": 1024},
            {"authorization": f"Bearer {self._key}", "content-type": "application/json"},
        )
        return (data["choices"][0]["message"].get("content") or "").strip()

    def load_session(self, session_id: str) -> None:
        data = sessions.load_session(session_id)
        self.session_id = data["id"]
        self.provider = data.get("provider", self.provider)
        self.model = data.get("model", self.model)
        self.messages = data.get("messages", [])
        self._key = configured_api_key(self.provider) or self._key
        c = data.get("cost", {})
        self.cost.usd = float(c.get("usd", 0.0))
        self.cost.input_tokens = int(c.get("input", 0))
        self.cost.output_tokens = int(c.get("output", 0))

    def _run_tool_calls(self, calls: list[tuple[str, str, dict]]) -> dict[str, str]:
        """Run a turn's tool calls: read-only ones in parallel, the rest serially.

        calls: list of (call_id, tool_name, args). Returns {call_id: result}.
        """

        results: dict[str, str] = {}
        read_only = [c for c in calls if tool_meta(c[1])["read_only"]]
        serial = [c for c in calls if not tool_meta(c[1])["read_only"]]
        if len(read_only) > 1:
            from .coordinator import Coordinator

            pairs = Coordinator(max_workers=min(8, len(read_only))).map(
                lambda c: (c[0], self._dispatch(c[1], c[2])), read_only
            )
            results.update(dict(pairs))
        else:
            for cid, name, args in read_only:
                results[cid] = self._dispatch(name, args)
        for cid, name, args in serial:
            if self.abort.is_set():
                results[cid] = json.dumps({"aborted": True})
                continue
            results[cid] = self._dispatch(name, args)
        return results

    def _dispatch(self, name: str, args: dict) -> str:
        """Permission-gate a tool call, then run it."""

        if name == "present_options":
            options = args.get("options", []) or []
            self.on_tool(name, args)
            if self.on_options:
                choice = self.on_options(options)
                return json.dumps({"user_selected": choice})
            return _run_tool(name, args)  # headless fallback

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
            for t in all_specs()
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
            for block in content:
                if block.get("type") == "text" and block.get("text", "").strip():
                    final_text.append(block["text"])
                    self.on_text(block["text"])
            calls = [
                (b["id"], b["name"], b.get("input", {}))
                for b in content
                if b.get("type") == "tool_use" and b.get("id") and b.get("name")
            ]
            if data.get("stop_reason") == "tool_use" and calls:
                results = self._run_tool_calls(calls)
                self.messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": cid, "content": results[cid]} for cid, _, _ in calls],
                })
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
            for t in all_specs()
        ]
        final_text: list[str] = []
        for _ in range(MAX_TOOL_STEPS):
            # Newer OpenAI models renamed max_tokens -> max_completion_tokens.
            payload = {"model": self.model, "messages": self.messages, "tools": tools, "tool_choice": "auto", "max_completion_tokens": MAX_TOKENS}
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
                self.on_text(message["content"])
            if not tool_calls:
                break
            calls = []
            for call in tool_calls:
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append((call["id"], call["function"]["name"], args))
            results = self._run_tool_calls(calls)
            for cid, _, _ in calls:
                self.messages.append({"role": "tool", "tool_call_id": cid, "content": results[cid]})
        return "\n".join(t for t in final_text if t).strip()


def _render_messages(messages: list) -> str:
    """Flatten provider-native messages to plain text for summarization."""

    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use {block.get('name')} {json.dumps(block.get('input', {}))[:300]}]")
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool_result {str(block.get('content', ''))[:600]}]")
            text = "\n".join(parts)
        else:
            text = str(content)
        if text.strip():
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _http_post(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        try:
            message = json.loads(detail).get("error", {}).get("message", "") or detail
        except (json.JSONDecodeError, AttributeError):
            message = detail
        raise RuntimeError(f"provider {exc.code}: {message.strip()[:200]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"network error calling provider: {exc}") from exc
