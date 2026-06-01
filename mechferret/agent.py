"""Conversational research agent.

The REPL pipes your prompts to a model. The model holds the
conversation and decides when to reach for MechFerret's *architecture/agent*
parts — the research pipeline, the discovery loop, the skills, the environment
— which are exposed to it as tools. So "hello" just gets a reply; "ground this
idea in sources" makes the model call ``run_research``, while an explicit
model-and-task experiment request makes it call ``run_discovery`` and narrate
the result.

Provider calls go over stdlib ``urllib`` (no SDK needs installing into the pipx
venv). Anthropic and OpenAI tool-use are both supported.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import permissions, sessions
from .config import configured_api_key, configured_model, default_config_path, load_config
from .costs import CostTracker
from .tracing import TraceRecorder
from .tools import all_specs, run_tool as _run_tool, tool_meta

BASE_SYSTEM_PROMPT = """You are MechFerret, an agentic coding assistant specialised for mechanistic-interpretability research (like Claude Code / Codex, but for interp).

You have real tools: run shell commands (bash), read/write/edit files, glob and
grep the codebase, search the web, fetch URLs, search arXiv, query Neuronpedia
for SAE features, verify the novelty of an idea, present interactive options, run
the general prompt-to-dossier research pipeline, run the interpretability
discovery loop, and list skills. Use them — don't guess.
Use run_research for general literature/source-grounded research questions and
planning. Use run_discovery only for supported mechanistic-interpretability
experiment tasks when the user or selected skill supplies the model under study.
Never choose a model, task, paper, or known circuit by default. If the user did
not name the model and the selected skill does not declare one, ask for the
model before proposing experiment details.
Treat audit advisories in tool results as user-facing caveats, not hidden
metadata.
When the user wants to investigate a model's internals, plan a paper, or run
experiments, gather evidence with the search tools, then write and run real code
(TransformerLens / SAELens / nnsight) with bash, reading results back.

Workflow when the user wants to plan research:
1. Gather evidence with retrieval tools (arxiv_search, web_search, web_fetch).
2. For each candidate direction, call verify_novelty to check whether prior
   papers already did it. Read its assessment.risk, assessment.verdict,
   assessment.closest_prior_art, and assessment.claim_readiness fields; do not
   summarize novelty from memory.
3. Call present_options with 2-5 concrete directions (each with a one-line
   summary, a fuller detail paragraph, key citations, novelty_risk,
   novelty_verdict, closest_prior_art, claim_readiness, and required_delta) so
   the user can browse and pick. Do NOT write the options out as prose.

OUTPUT RULES (important):
- Plain text only. No markdown: no #, *, **, backticks, tables, or '-' bullets.
- Be concise. A few short paragraphs, not an essay. Lead with the answer.
- Cite paper URLs and file paths you actually read, inline.
- Be precise about rigor: controls, multiple seeds, reproducibility, and
  triangulation across independent probes before claiming a mechanism.
- Do not ask the user to hit Return in assistant text; the REPL already handles
  continuation. End with a concrete next step only when the required model, task,
  evidence, and scope are known.
- If the next step depends on an unstated model, task, dataset, paper, compute
  target, or approval, ask one targeted clarifying question instead of inventing
  an experiment. Never fill that gap with GPT-2, IOI, known circuit heads, or
  any other benchmark example unless the user explicitly asked for it.
- When a long task is already running, the user can still type normal prompts,
  queue prompts, or use /btw for side questions; keep replies compatible with
  that interaction model."""

MAX_TOOL_STEPS = 12
MAX_TOKENS = int(os.getenv("MECHFERRET_MAX_TOKENS", "4096"))
COMPACT_CHAR_THRESHOLD = int(os.getenv("MECHFERRET_COMPACT_CHARS", "240000"))  # ~60k tokens
COMPACT_KEEP_LAST = 4

BENCHMARK_EXPLICIT_TERMS = (
    "gpt2",
    "gpt-2",
    "ioi",
    "indirect object",
    "name mover",
    "name-mover",
    "duplicate token",
    "duplicate-token",
)
BENCHMARK_LEAK_TERMS = (
    "gpt2",
    "gpt-2",
    "ioi",
    "name mover",
    "name-mover",
    "duplicate token",
    "duplicate-token",
    "s-inhibition",
    "known heads",
)
STALE_HEAD_RE = re.compile(r"\b(?:heads?\s+)?(?:[4567]\.(?:0|1|2|3|5|6|8|11))(?:\s*,\s*[4567]\.(?:0|1|2|3|5|6|8|11)){2,}\b")
STALE_CONTINUATION_RE = re.compile(r"\b(?:press|hit)\s+(?:enter|return)\b.*\b(?:proceed|continue|next)\b", re.IGNORECASE | re.DOTALL)

COMPACT_SYSTEM = (
    "You are compacting a mechanistic-interpretability research session to free context. "
    "Write a dense summary that MUST retain: every confirmed mechanism (layer.head + role + "
    "effect size + seeds), open hypotheses and next experiments, key file paths and decisions, "
    "and any goal/acceptance bar. Drop chit-chat. Output only the summary."
)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


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

    mechanisms = _recall_mechanisms() if os.getenv("MECHFERRET_INCLUDE_MEMORY_CONTEXT") == "1" else ""
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

    config_path = default_config_path()
    config = load_config(config_path)
    provider = config.default_provider
    if provider == "local" and config_path.exists():
        return "", "", ""
    if provider in {"anthropic", "openai"}:
        key = configured_api_key(provider, config)
        model = configured_model(provider, config)
        if key and model:
            return provider, model, key
    for candidate in ("anthropic", "openai"):
        key = configured_api_key(candidate, config)
        model = configured_model(candidate, config)
        if key and model:
            return candidate, model, key
    return "", "", ""


def is_configured() -> bool:
    return bool(active_provider()[0])


def _agent_tool_failure(
    name: str,
    error: str,
    *,
    failed_check: str,
    extra: dict[str, Any] | None = None,
    next_action: str,
) -> dict[str, Any]:
    payload = {
        "ok": False,
        "tool": name,
        "error": error,
        "failed_checks": [failed_check],
        "next_actions": [next_action],
    }
    if extra:
        payload.update(extra)
    return payload


def _known_tool_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    if name.startswith("mcp__"):
        return True
    return any(spec.get("name") == name for spec in all_specs())


def _tool_loop_exhausted_text() -> str:
    return (
        f"Tool loop stopped after reaching {MAX_TOOL_STEPS} steps before a final answer. "
        "Review the recent tool results and retry with a narrower request."
    )


def _provider_response_failure(provider: str, error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "provider": provider,
        "error": error,
        "failed_checks": ["provider_response_envelope"],
        "next_actions": ["Retry the request; if it repeats, inspect provider status and the local trace."],
    }


def _provider_response_failure_text(provider: str, error: str) -> str:
    label = "OpenAI" if provider == "openai" else "Anthropic" if provider == "anthropic" else provider
    return (
        f"{label} returned an invalid response envelope: {error}. "
        "Retry the request; if it repeats, inspect provider status and the local trace."
    )


def _reject_duplicate_tool_call_ids(
    calls: list[tuple[str, str, Any]],
) -> tuple[list[tuple[str, str, Any]], list[tuple[str, str]]]:
    seen: set[str] = set()
    unique: list[tuple[str, str, Any]] = []
    rejected: list[tuple[str, str]] = []
    for cid, name, args in calls:
        if cid in seen:
            rejected.append(
                (
                    cid,
                    json.dumps(
                        _agent_tool_failure(
                            name,
                            f"duplicate tool call id {cid}",
                            failed_check="tool_call_envelope",
                            next_action="Return each tool call with a unique id.",
                        )
                    ),
                )
            )
            continue
        seen.add(cid)
        unique.append((cid, name, args))
    return unique, rejected


def _extract_provider_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        nested_content = block.get("content")
        if isinstance(nested_content, str) and block.get("type") in {"text", "output_text"}:
            parts.append(nested_content)
    return "\n".join(part.strip() for part in parts if part.strip())


def _user_explicitly_selected_benchmark(user_text: str) -> bool:
    lowered = _text(user_text).lower()
    return any(term in lowered for term in BENCHMARK_EXPLICIT_TERMS)


def _looks_like_stale_benchmark_scaffold(text: str) -> bool:
    lowered = _text(text).lower()
    if not lowered:
        return False
    benchmark_hits = sum(1 for term in BENCHMARK_LEAK_TERMS if term in lowered)
    return benchmark_hits >= 2 or bool(STALE_HEAD_RE.search(lowered))


def _sanitize_assistant_text(user_text: str, text: str) -> str:
    if not text:
        return text
    stale_benchmark = _looks_like_stale_benchmark_scaffold(text) and not _user_explicitly_selected_benchmark(user_text)
    stale_continuation = bool(STALE_CONTINUATION_RE.search(text))
    if not (stale_benchmark or stale_continuation):
        return text
    return (
        "I need one missing research target before I can propose a concrete experiment: "
        "which model and behavior/task should I investigate? I will not substitute a "
        "benchmark model, task, or known circuit unless you explicitly ask for that demo."
    )


def _extract_anthropic_content(data: Any) -> tuple[Any | None, str | None]:
    if not isinstance(data, dict):
        return None, "response is not a JSON object"
    if "content" not in data:
        return None, "response.content is missing"
    content = data.get("content")
    if not isinstance(content, (list, str)):
        return None, "response.content must be a list or string"
    return content, None


def _extract_openai_message(data: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(data, dict):
        return None, "response is not a JSON object"
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "response.choices must be a non-empty list"
    choice = choices[0]
    if not isinstance(choice, dict):
        return None, "response.choices[0] must be an object"
    message = choice.get("message")
    if not isinstance(message, dict):
        return None, "response.choices[0].message must be an object"
    return message, None


def _extract_openai_tool_calls(message: dict[str, Any]) -> tuple[list[Any] | None, str | None]:
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        return [], None
    if not isinstance(tool_calls, list):
        return None, "message.tool_calls must be a list"
    return tool_calls, None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
        self.tracer = TraceRecorder(self.session_id, ".mechferret")  # -> .mechferret/trace.jsonl (+ Raindrop mirror)

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
        self.tracer.event("user_prompt", text=user_text[:200])
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
        except Exception as exc:  # noqa: BLE001 - transcript persistence is best-effort
            try:
                self.tracer.event(
                    "session_persist_failed",
                    error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    messages=len(self.messages),
                )
            except Exception:  # noqa: BLE001 - tracing is also best-effort
                pass

    # --- context compaction --------------------------------------------------------

    def _approx_chars(self) -> int:
        try:
            return len(json.dumps(self.messages))
        except (TypeError, ValueError):
            return 0

    def _maybe_compact(self) -> None:
        if self._approx_chars() > COMPACT_CHAR_THRESHOLD:
            try:
                self.compact()
            except Exception as exc:  # noqa: BLE001 - automatic compaction must not block a user turn
                self.tracer.event(
                    "compact_failed",
                    error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    messages=len(self.messages),
                    approx_chars=self._approx_chars(),
                )

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
            content, error = _extract_anthropic_content(data)
            if error:
                raise RuntimeError(f"provider response envelope: {error}")
            return _extract_provider_text(content)
        data = _http_post(
            "https://api.openai.com/v1/chat/completions",
            {"model": self.model, "messages": [{"role": "system", "content": COMPACT_SYSTEM}, {"role": "user", "content": convo}], "max_completion_tokens": 1024},
            {"authorization": f"Bearer {self._key}", "content-type": "application/json"},
        )
        message, error = _extract_openai_message(data)
        if error:
            raise RuntimeError(f"provider response envelope: {error}")
        return _extract_provider_text(message.get("content"))

    def load_session(self, session_id: str) -> None:
        data = sessions.load_session(session_id)
        loaded_id = data.get("id")
        self.session_id = loaded_id if sessions.is_valid_session_id(loaded_id) else session_id
        provider = data.get("provider")
        model = data.get("model")
        if provider in {"anthropic", "openai"}:
            self.provider = provider
        if isinstance(model, str) and model and provider in {"anthropic", "openai"}:
            self.model = model
        messages = data.get("messages", [])
        self.messages = [message for message in messages if isinstance(message, dict)] if isinstance(messages, list) else []
        self._key = configured_api_key(self.provider) or self._key
        c = data.get("cost", {})
        c = c if isinstance(c, dict) else {}
        self.cost.usd = _coerce_float(c.get("usd"), 0.0)
        self.cost.input_tokens = _coerce_int(c.get("input"), 0)
        self.cost.output_tokens = _coerce_int(c.get("output"), 0)

    def _run_tool_calls(self, calls: list[tuple[str, str, Any]]) -> dict[str, str]:
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
                results[cid] = json.dumps(
                    _agent_tool_failure(
                        name,
                        "tool call aborted",
                        failed_check="tool_aborted",
                        extra={"aborted": True},
                        next_action="Retry after clearing the abort signal.",
                    )
                )
                continue
            results[cid] = self._dispatch(name, args)
        return results

    def _dispatch(self, name: str, args: dict) -> str:
        """Permission-gate a tool call, then run it."""

        if not isinstance(args, dict):
            return json.dumps(
                _agent_tool_failure(
                    name,
                    "invalid tool arguments",
                    failed_check="tool_arguments",
                    next_action="Pass tool arguments as a JSON object.",
                )
            )

        if not _known_tool_name(name):
            return json.dumps(
                _agent_tool_failure(
                    name,
                    f"unknown tool {name}",
                    failed_check="tool_registered",
                    next_action="Choose a tool from the registered tool list.",
                )
            )

        if name == "present_options":
            options = args.get("options", []) or []
            self.on_tool(name, args)
            self.tracer.event("tool", tool=name, options=len(options))
            if self.on_options:
                choice = self.on_options(options)
                self.tracer.event("user_selected", choice=str(choice)[:120])
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
                self.tracer.event("tool_denied", tool=name, reason=decision.reason)
                return json.dumps(
                    _agent_tool_failure(
                        name,
                        decision.reason or "not approved",
                        failed_check="tool_permission",
                        extra={"denied": True, "reason": decision.reason or "not approved"},
                        next_action="Ask for approval or use a read-only tool.",
                    )
                )
        self.on_tool(name, args)
        self.tracer.event("tool", tool=name, permission=meta["permission"],
                          args={k: str(v)[:100] for k, v in args.items()})
        result = _run_tool(name, args)
        self.tracer.event("tool_done", tool=name, output_chars=len(result))
        return result

    def _record_provider_response_failure(self, provider: str, error: str, data: Any, *, append: bool = True) -> str:
        payload = _provider_response_failure(provider, error)
        self.tracer.event(
            "provider_response_invalid",
            provider=provider,
            error=error,
            failed_checks=payload["failed_checks"],
            response_type=type(data).__name__,
        )
        message = _provider_response_failure_text(provider, error)
        if append:
            self.messages.append({"role": "assistant", "content": message})
        self.on_text(message)
        return message

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
            if not isinstance(data, dict):
                return self._record_provider_response_failure("anthropic", "response is not a JSON object", data)
            if isinstance(data.get("usage"), dict):
                self.cost.add(self.model, data["usage"])
            content, error = _extract_anthropic_content(data)
            if error:
                return self._record_provider_response_failure("anthropic", error, data)
            calls = []
            malformed_results: list[tuple[str, str]] = []
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                parsed = _parse_anthropic_tool_use(block)
                if parsed is None:
                    continue
                if parsed["ok"]:
                    calls.append((parsed["id"], parsed["name"], parsed["args"]))
                else:
                    malformed_results.append((parsed["id"], json.dumps(parsed["payload"])))
            calls, duplicate_results = _reject_duplicate_tool_call_ids(calls)
            malformed_results.extend(duplicate_results)
            if data.get("stop_reason") == "tool_use" and not (calls or malformed_results):
                return self._record_provider_response_failure(
                    "anthropic",
                    "response.stop_reason is tool_use but no usable tool_use blocks were present",
                    data,
                    append=False,
                )
            text = _extract_provider_text(content)
            if text:
                sanitized = _sanitize_assistant_text(user_text, text)
                if sanitized != text:
                    content = [{"type": "text", "text": sanitized}]
                    text = sanitized
                    calls = []
                    malformed_results = []
            self.messages.append({"role": "assistant", "content": content})
            if text:
                final_text.append(text)
                self.on_text(text)
            if data.get("stop_reason") == "tool_use" and (calls or malformed_results):
                results = self._run_tool_calls(calls)
                result_items = [(cid, results[cid]) for cid, _, _ in calls] + malformed_results
                self.messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": cid, "content": content} for cid, content in result_items],
                })
                continue
            break
        else:
            message = _tool_loop_exhausted_text()
            final_text.append(message)
            self.on_text(message)
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
            if not isinstance(data, dict):
                return self._record_provider_response_failure("openai", "response is not a JSON object", data)
            if isinstance(data.get("usage"), dict):
                self.cost.add(self.model, data["usage"])
            message, error = _extract_openai_message(data)
            if error:
                return self._record_provider_response_failure("openai", error, data)
            tool_calls, error = _extract_openai_tool_calls(message)
            if error:
                return self._record_provider_response_failure("openai", error, data)
            text = _extract_provider_text(message.get("content"))
            if text:
                sanitized = _sanitize_assistant_text(user_text, text)
                if sanitized != text:
                    message = dict(message)
                    message["content"] = sanitized
                    message.pop("tool_calls", None)
                    text = sanitized
                    tool_calls = []
            self.messages.append(message)
            if text:
                final_text.append(text)
                self.on_text(text)
            if not tool_calls:
                break
            calls = []
            malformed_results: list[tuple[str, str]] = []
            for call in tool_calls:
                parsed = _parse_openai_tool_call(call)
                if parsed["ok"]:
                    calls.append((parsed["id"], parsed["name"], parsed["args"]))
                else:
                    malformed_results.append((parsed["id"], json.dumps(parsed["payload"])))
            calls, duplicate_results = _reject_duplicate_tool_call_ids(calls)
            malformed_results.extend(duplicate_results)
            results = self._run_tool_calls(calls)
            for cid, _, _ in calls:
                self.messages.append({"role": "tool", "tool_call_id": cid, "content": results[cid]})
            for cid, content in malformed_results:
                self.messages.append({"role": "tool", "tool_call_id": cid, "content": content})
        else:
            message = _tool_loop_exhausted_text()
            final_text.append(message)
            self.on_text(message)
        return "\n".join(t for t in final_text if t).strip()


def _parse_openai_tool_call(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        return _malformed_openai_tool_call("", "", "tool call is not an object")
    cid = call.get("id")
    if not isinstance(cid, str) or not cid:
        return _malformed_openai_tool_call("", "", "tool call id is missing")
    function = call.get("function")
    if not isinstance(function, dict):
        return _malformed_openai_tool_call(cid, "", "tool call function is missing")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return _malformed_openai_tool_call(cid, "", "tool call function name is missing")
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = []
    return {"ok": True, "id": cid, "name": name, "args": args}


def _parse_anthropic_tool_use(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    if block.get("type") != "tool_use":
        return None
    cid = block.get("id")
    if not isinstance(cid, str) or not cid:
        return _malformed_anthropic_tool_use("", "", "tool use id is missing")
    name = block.get("name")
    if not isinstance(name, str) or not name:
        return _malformed_anthropic_tool_use(cid, "", "tool use name is missing")
    return {"ok": True, "id": cid, "name": name, "args": block.get("input", {})}


def _malformed_anthropic_tool_use(cid: str, name: str, error: str) -> dict[str, Any]:
    tool_name = name or "anthropic_tool_use"
    return {
        "ok": False,
        "id": cid or "malformed_tool_use",
        "name": tool_name,
        "payload": _agent_tool_failure(
            tool_name,
            error,
            failed_check="tool_call_envelope",
            next_action="Return a tool_use block with id, name, and object input.",
        ),
    }


def _malformed_openai_tool_call(cid: str, name: str, error: str) -> dict[str, Any]:
    tool_name = name or "openai_tool_call"
    return {
        "ok": False,
        "id": cid or "malformed_tool_call",
        "name": tool_name,
        "payload": _agent_tool_failure(
            tool_name,
            error,
            failed_check="tool_call_envelope",
            next_action="Return a tool call with id, function.name, and JSON object arguments.",
        ),
    }


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
            text_content = _extract_provider_text(content)
            if text_content:
                parts.append(text_content)
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    parts.append(f"[tool_use {block.get('name')} {json.dumps(block.get('input', {}))[:300]}]")
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool_result {str(block.get('content', ''))[:600]}]")
            text = "\n".join(parts)
        else:
            text = str(content)
        if text.strip():
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _http_post(url: str, payload: dict, headers: dict) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                snippet = text.strip().replace("\n", " ")[:200] or "<empty>"
                raise RuntimeError(f"provider returned invalid JSON: {snippet}") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        try:
            message = json.loads(detail).get("error", {}).get("message", "") or detail
        except (json.JSONDecodeError, AttributeError):
            message = detail
        raise RuntimeError(f"provider {exc.code}: {message.strip()[:200]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"network error calling provider: {exc}") from exc
