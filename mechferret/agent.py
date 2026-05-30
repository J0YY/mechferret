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
import urllib.error
import urllib.request
from typing import Any, Callable

from .config import configured_api_key, configured_model, load_config

SYSTEM_PROMPT = """You are MechFerret, an autonomous mechanistic-interpretability research agent.

You converse normally and help the user reason about neural-network internals
(attention heads, MLPs, circuits, features). When the user wants to actually
investigate, find, localise, confirm, or explain a model's internal mechanism —
or asks to run an experiment, screen heads, or recover a circuit — call the
`run_discovery` tool rather than guessing. Use `list_skills` to see ready-made
playbooks and `environment_status` to check compute/providers.

After a tool runs, summarise the findings crisply: name the confirmed mechanisms
(layer.head + role), their effect size / reproducibility / novelty, and point to
the HTML dossier. Be precise about rigor: distinguish significant + reproducible
+ triangulated results from weaker ones. Keep replies concise and concrete."""

MAX_TOOL_STEPS = 6


# --- tools the model can call --------------------------------------------------------

def _tool_run_discovery(args: dict[str, Any]) -> str:
    from .discovery import DiscoveryController

    run = DiscoveryController().run(
        question=args.get("question", ""),
        skill=args.get("skill"),
        task=args.get("task"),
        model=args.get("model", "gpt2"),
        backend=args.get("backend", "auto"),
        out_dir=args.get("out_dir", "runs/agent"),
    )
    return json.dumps(
        {
            "discoveries": [
                {
                    "statement": d.statement,
                    "confidence": d.confidence,
                    "effect_size": d.effect_size,
                    "reproducibility": d.reproducibility,
                    "novelty": d.novelty,
                }
                for d in run.discoveries
            ],
            "metrics": {
                k: run.metrics.get(k)
                for k in ("rigor_score", "readiness_score", "confirmed_mechanisms", "experiments_run", "rounds_run")
            },
            "report_html": run.artifacts.get("html"),
            "discoveries_json": run.artifacts.get("discoveries"),
        }
    )


def _tool_list_skills(_args: dict[str, Any]) -> str:
    from .skills import list_skills

    return json.dumps([{"name": s.name, "task": s.task, "description": s.description} for s in list_skills()])


def _tool_environment_status(_args: dict[str, Any]) -> str:
    from .cluster import load_cluster_config
    from .interp.backends import transformer_lens_available
    from .modal_app import modal_status
    from .skills import list_skills

    return json.dumps(
        {
            "interp_backend": "transformer_lens" if transformer_lens_available() else "synthetic (offline)",
            "skills": [s.name for s in list_skills()],
            "modal": modal_status(),
            "cluster_configured": load_cluster_config().configured,
        }
    )


TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_discovery",
        "description": (
            "Run the autonomous interpretability discovery loop on a model to find and confirm the "
            "mechanisms (attention heads / circuits) responsible for a behaviour. Use whenever the user "
            "asks to investigate, find, localise, confirm, or explain a model's internals or a circuit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Free-form research question."},
                "skill": {
                    "type": "string",
                    "enum": ["ioi-circuit", "find-induction-heads", "logit-lens-sweep", "factual-recall-trace"],
                    "description": "Optional named playbook to run.",
                },
                "task": {
                    "type": "string",
                    "enum": ["ioi", "induction", "greater_than", "factual_recall"],
                    "description": "Interpretability task to investigate.",
                },
                "model": {"type": "string", "description": "Model to study (default gpt2)."},
                "backend": {"type": "string", "enum": ["auto", "synthetic", "transformer_lens"]},
            },
            "required": [],
        },
    },
    {
        "name": "list_skills",
        "description": "List the available interpretability playbooks/skills.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "environment_status",
        "description": "Report the environment: interp backend, skills, Modal status, cluster config.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

DISPATCH: dict[str, Callable[[dict[str, Any]], str]] = {
    "run_discovery": _tool_run_discovery,
    "list_skills": _tool_list_skills,
    "environment_status": _tool_environment_status,
}


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
        self.messages: list[dict[str, Any]] = []  # provider-native message history

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

    # --- Anthropic ------------------------------------------------------------------

    def _send_anthropic(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOLS
        ]
        final_text: list[str] = []
        for _ in range(MAX_TOOL_STEPS):
            payload = {
                "model": self.model,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
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
            content = data.get("content", [])
            self.messages.append({"role": "assistant", "content": content})
            tool_results = []
            for block in content:
                if block.get("type") == "text":
                    final_text.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    self.on_tool(block["name"], block.get("input", {}))
                    result = _run_tool(block["name"], block.get("input", {}))
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
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        self.messages.append({"role": "user", "content": user_text})
        tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in TOOLS
        ]
        final_text: list[str] = []
        for _ in range(MAX_TOOL_STEPS):
            payload = {"model": self.model, "messages": self.messages, "tools": tools, "tool_choice": "auto"}
            data = _http_post(
                "https://api.openai.com/v1/chat/completions",
                payload,
                {"authorization": f"Bearer {self._key}", "content-type": "application/json"},
            )
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
                self.on_tool(name, args)
                result = _run_tool(name, args)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        return "\n".join(t for t in final_text if t).strip()


def _run_tool(name: str, args: dict[str, Any]) -> str:
    handler = DISPATCH.get(name)
    if not handler:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        return handler(args)
    except Exception as exc:  # noqa: BLE001 - report tool failure back to the model
        return json.dumps({"error": str(exc)})


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
