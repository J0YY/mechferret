"""Interactive REPL — a Claude-Code-style prompt for MechFerret.

Run `mechferret` with no arguments (or `mechferret repl`) to drop into an
interactive session. Plain-English prompts are piped to a model (Claude or GPT)
that converses and calls MechFerret's discovery tools when you ask for
interpretability work. `/commands` drive the system directly. On your first
prompt, if no model is connected, it walks you through adding an API key.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

try:
    import readline  # noqa: F401  (enables arrow keys + history on input())
except ImportError:  # pragma: no cover
    readline = None

_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
VERSION = "0.1.0"
WIDTH = 78
PURPLE = "38;5;141"  # soft violet
PURPLE_B = "1;38;5;141"


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _vlen(text: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", text))


KNOWN_COMMANDS = {
    "run", "demo", "discover", "login", "api", "goal", "loop", "doctor",
    "registry", "memory", "cost", "resume", "inspect", "skills", "modal", "cluster",
}

HISTORY_FILE = Path.home() / ".mechferret" / "repl_history"

# Small, cute ferret — width-stable ASCII so the box stays aligned.
FERRET = [
    "  ___",
    " (o.o)__",
    "  >.<   )",
    '  (")_(")',
]


class Session:
    def __init__(self) -> None:
        self.out_root = Path("runs")
        self.last_report: str | None = None
        self.run_count = 0


# --- welcome screen ------------------------------------------------------------------

def _two_column_box(left: list[str], right: list[str]) -> str:
    left_w = 38
    right_w = WIDTH - left_w - 5  # borders + separator
    rows = max(len(left), len(right))
    left += [""] * (rows - len(left))
    right += [""] * (rows - len(right))
    title = _c(f" MechFerret v{VERSION} ", PURPLE_B)
    top = "╭─── " + title + "─" * (WIDTH - _vlen(title) - 6) + "╮"
    bottom = "╰" + "─" * (WIDTH - 2) + "╯"
    lines = [top]
    for l, r in zip(left, right):
        lpad = l + " " * max(0, left_w - _vlen(l))
        rpad = r + " " * max(0, right_w - _vlen(r))
        lines.append("│ " + lpad + " │ " + rpad + " │")
    lines.append(bottom)
    return "\n".join(lines)


def _welcome(session: Session) -> str:
    from .agent import active_provider

    provider, model, _key = active_provider()
    user = os.getenv("USER") or "there"
    cwd = str(Path.cwd()).replace(str(Path.home()), "~")
    status = (
        _c(model, PURPLE) if provider
        else _c("no model connected", "33") + _c(" · /login", "2")
    )

    left = [
        "",
        _c(f"Welcome back {user.capitalize()}!", "1"),
        "",
    ]
    left += [_c(line, PURPLE) for line in FERRET]
    left += [_c("mechferret", PURPLE_B), "", status, _c(cwd, "2")]

    right = [
        _c("About", "1"),
        _c("agentic interpretability", "2"),
        _c("research CLI", "2"),
        "",
        _c("Help", "1"),
        _c("<prompt>   chat + run work", "2"),
        _c("/login     connect a model", "2"),
        _c("/model     pick the model", "2"),
        _c("/help      all commands", "2"),
        _c("/exit      quit", "2"),
    ]
    return _two_column_box(left, right)


def _print_status_and_bar(agent) -> None:
    mode = getattr(agent, "permission_mode", "auto")
    bits = [_c(agent.model, PURPLE) if agent.configured else _c("no model · /login", "33")]
    if agent.configured and agent.cost.usd:
        bits.append(_c(agent.cost.format_total(), "2"))
    bits.append(_c(f"mode:{mode}" + (" ⏸" if mode == "plan" else ""), "33" if mode == "plan" else "2"))
    print(_c("─" * WIDTH, "2"))
    print("  " + _c(" · ", "2").join(bits))


# --- history -------------------------------------------------------------------------

def _setup_history() -> None:
    if readline is None:
        return
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)


def _save_history() -> None:
    if readline is None:
        return
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


# --- onboarding ----------------------------------------------------------------------

def onboard() -> bool:
    """Connect a model (provider + API key). Returns True if configured."""

    import getpass

    from .config import configure_provider
    from .picker import select

    MODELS = {
        "anthropic": ["claude-opus-4-8  (highest reasoning)", "claude-sonnet-4-6  (faster)"],
        "openai": ["gpt-5.5  (highest reasoning)", "gpt-5"],
    }
    print()
    try:
        choice = select(
            _c("  Connect a model  (↑/↓, Enter; Esc to cancel)", "1"),
            ["Anthropic (Claude)", "OpenAI (GPT)"],
        )
    except KeyboardInterrupt:
        print(_c("  Cancelled. Connect later with /login.", "2"))
        return False
    provider = "anthropic" if choice.startswith("Anthropic") else "openai"
    try:
        model_choice = select(_c(f"  Pick a {provider} model", "1"), MODELS[provider])
    except KeyboardInterrupt:
        print(_c("  Cancelled.", "2"))
        return False
    model = model_choice.split()[0]
    env_hint = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    try:
        key = getpass.getpass(_c(f"  Paste your {provider} API key ({env_hint}): ", "1")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not key:
        print(_c("  No key entered; cancelled.", "33"))
        return False
    path = configure_provider(provider, key, model=model, make_default=True)
    print(_c(f"  ✓ Connected {provider} ({model}). Stored in {path}.", "32"))
    print(_c("  (Keys are also read from env vars if you prefer not to store them.)", "2"))
    print()
    return True


# --- main loop -----------------------------------------------------------------------

def run_repl() -> None:
    from .agent import Agent

    session = Session()
    agent = Agent()
    _setup_history()
    print(_welcome(session))
    print()

    while True:
        _print_status_and_bar(agent)
        try:
            line = input(_c("❯ ", "1;36")).strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print(_c("  (Ctrl-D or /exit to quit)", "2"))
            continue
        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()
        head = tokens[0]
        bare = head.lstrip("/").lower()

        if bare in {"exit", "quit", "q"}:
            break
        if bare == "help":
            _print_help()
            continue
        if bare == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(_welcome(session))
            continue
        if bare == "open":
            _open_report(session)
            continue
        if bare in {"login", "connect"}:
            if onboard():
                agent.reload()
            continue
        if bare == "model" and head.startswith("/"):
            _set_model(agent, tokens[1:])
            continue
        if bare == "plan":
            agent.permission_mode = "auto" if agent.permission_mode == "plan" else "plan"
            on = agent.permission_mode == "plan"
            print(_c(f"  plan mode {'ON — write/exec/GPU tools will ask before running' if on else 'OFF'}", "33"))
            continue
        if bare == "cost" and len(tokens) == 1:
            print(_c(f"  session: {agent.cost.format_total()}  ·  model {agent.model}  ·  denied tools: {len(agent.denials)}", "2"))
            for model, slot in agent.cost.by_model.items():
                print(_c(f"    {model}: ${slot['usd']:.4f}  ({int(slot['input'])} in / {int(slot['output'])} out)", "2"))
            continue
        if bare == "compact":
            summary = agent.compact()
            if summary == "nothing to compact":
                print(_c("  " + summary, "2"))
            else:
                print(_c("  ✓ compacted older turns into a summary:", "32"))
                print(_indent(_c(summary[:800], "2")))
            continue
        if bare == "resume":
            _resume(agent, tokens[1:])
            continue
        if bare == "memory" and len(tokens) == 1:
            _show_memory()
            continue
        if bare == "mcp":
            _mcp(tokens[1:])
            continue
        if bare == "init":
            _init_project()
            continue
        if bare == "export":
            _export(agent, tokens[1:])
            continue
        if bare == "review":
            import subprocess

            diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True).stdout
            if not diff.strip():
                print(_c("  no uncommitted changes to review", "33"))
                continue
            _chat(agent, (
                "Review this git diff for an interpretability-research codebase. Flag correctness bugs, "
                "missing experimental controls or seed logging, data leakage into activations/probes, "
                "and reproducibility issues. Be specific with file:line.\n\n" + diff[:40000]
            ))
            continue

        if head.startswith("/") or bare in KNOWN_COMMANDS:
            _dispatch_command(tokens, bare)
            continue

        # Plain text => talk to the model.
        _chat(agent, line)

    _save_history()
    print(_c("bye 👋", "2"))


def _chat(agent, text: str) -> None:
    from .spinner import Spinner

    if not agent.configured:
        print(_c("  No model connected yet — let's fix that.", "2"))
        if not onboard():
            return
        agent.reload()
    spinner = Spinner()
    agent.on_tool = lambda name, args: spinner.log(_tool_line(name, args))

    def _confirm(name, args, reason):
        from .picker import select

        with spinner.pause():
            print(_c(f"  ⚠ {name} — {reason or 'approval needed'}", "33"))
            try:
                return select(_c("  Allow this tool call?", "1"), ["yes, run it", "no, skip it"]).startswith("yes")
            except KeyboardInterrupt:
                return False

    agent.confirm = _confirm
    streamed = {"any": False}

    def _emit(block: str):
        streamed["any"] = True
        spinner.log("\n" + _indent(block.strip()) + "\n")

    agent.on_text = _emit
    try:
        with spinner:
            reply = agent.send(text)
    except KeyboardInterrupt:
        print(_c("  (interrupted)", "2"))
        return
    except Exception as exc:  # noqa: BLE001
        print(_c(f"  error: {exc}", "31"))
        if "401" in str(exc) or "authentication" in str(exc).lower():
            print(_c("  Your API key may be invalid — reconnect with /login.", "33"))
        return
    if not streamed["any"] and reply:
        print()
        print(_indent(reply))
    print(_c(f"  ({agent.cost.format_total()})", "2"))
    print()


def _resume(agent, args: list[str]) -> None:
    from . import sessions

    if args:
        target = args[0]
    else:
        metas = sessions.list_sessions(10)
        if not metas:
            print(_c("  no saved sessions yet", "33"))
            return
        from .picker import select

        labels = [f"{m.id}  ({m.turns} turns · ${m.usd:.4f} · {m.model})" for m in metas]
        try:
            choice = select(_c("  Resume which session?", "1"), labels)
        except KeyboardInterrupt:
            return
        target = metas[labels.index(choice)].id
    try:
        agent.load_session(target)
        print(_c(f"  resumed {target} — {len([m for m in agent.messages if isinstance(m, dict) and m.get('role') == 'user'])} prior turns, {agent.cost.format_total()}", "32"))
    except KeyError as exc:
        print(_c(f"  {exc}", "33"))


def _init_project() -> None:
    import importlib.util

    path = Path.cwd() / "MECHFERRET.md"
    if path.exists():
        print(_c(f"  {path.name} already exists — leaving it as is", "33"))
        return
    have = lambda m: importlib.util.find_spec(m) is not None
    stack = [m for m in ("torch", "transformer_lens", "sae_lens", "nnsight") if have(m)]
    content = f"""# MechFerret project notes

This file is read into the agent's system prompt each turn. Keep it short and current.

## Stack
Installed: {", ".join(stack) or "none detected (install torch + transformer_lens for real experiments)"}

## Conventions
- Default model under study: gpt2
- Put run outputs under runs/
- Log seeds; every causal claim needs a negative control + reproduction across seeds.

## Current goal
(Describe the paper/result you're driving toward, and the acceptance bar.)
"""
    path.write_text(content, encoding="utf-8")
    print(_c(f"  ✓ wrote {path.name} (detected: {', '.join(stack) or 'no interp stack'})", "32"))


def _export(agent, args: list[str]) -> None:
    out = Path(args[0]) if args else Path(f"mechferret-session-{agent.session_id}.md")
    from .agent import _render_messages

    body = _render_messages(agent.messages)
    md = f"# MechFerret session {agent.session_id}\n\nModel: {agent.model}\nCost: {agent.cost.format_total()}\n\n---\n\n{body}\n"
    out.write_text(md, encoding="utf-8")
    print(_c(f"  ✓ exported transcript to {out}", "32"))


def _mcp(args: list[str]) -> None:
    from . import mcp

    action = args[0] if args else "status"
    if action in {"status", "list"}:
        st = mcp.status()
        servers = st["configured"]
        print(_c(f"  MCP servers: {', '.join(servers) if servers else '(none configured)'}", "2"))
        print(_c(f"  MCP tools available: {st['tool_count']}", "2"))
        if not servers:
            print(_c("  add one:  /mcp add <name> <command> [args…]", "2"))
        return
    if action == "add" and len(args) >= 3:
        path = mcp.add_server(args[1], args[2], args[3:])
        print(_c(f"  ✓ added MCP server '{args[1]}' to {path}; its tools will load on next prompt", "32"))
        return
    if action == "tools":
        for spec in mcp.tool_specs():
            print(_c(f"    {spec['name']}", "1;36") + _c(f"  {spec['description'][:70]}", "2"))
        return
    print(_c("  usage: /mcp [status | tools | add <name> <command> [args…]]", "33"))


def _show_memory() -> None:
    from .memory import ResearchMemory

    mem = ResearchMemory(".mechferret/memory.sqlite")
    try:
        rows = mem.recent_mechanisms(20)
    finally:
        mem.close()
    if not rows:
        print(_c("  no confirmed mechanisms in memory yet", "2"))
        return
    print(_c(f"  {len(rows)} confirmed mechanism(s) in memory:", "1"))
    for r in rows:
        print(_c(f"    • {r['statement']}", "2"))
        print(_c(f"      {r['model']} · effect {r['effect_size']:.2f} · repro {r['reproducibility']:.2f} · novelty {r['novelty']:.2f}", "2"))


def _tool_line(name: str, args: dict) -> str:
    detail = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items() if v) or ""
    return _c(f"  → {name}({detail})", "2")


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in (text or "").splitlines())


def _set_model(agent, args: list[str]) -> None:
    if not args:
        print(_c(f"  model = {agent.model or '(none)'}  provider = {agent.provider or '(none)'}", "2"))
        return
    agent.model = args[0]
    print(_c(f"  model → {args[0]}", "32"))


def _dispatch_command(tokens: list[str], bare: str) -> None:
    from .cli import main as cli_main

    try:
        cli_main([bare] + tokens[1:])
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print(_c("  (interrupted)", "2"))
    except Exception as exc:  # noqa: BLE001
        print(_c(f"  error: {exc}", "31"))


def _open_report(session: "Session") -> None:
    if not session.last_report:
        print(_c("  no report yet — runs land in ./runs (use /discover or ask the agent)", "33"))
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    os.system(f"{opener} {shlex.quote(session.last_report)}")


def _print_help() -> None:
    from .commands import REPL_COMMANDS

    print(_c("  commands:", "1"))
    for cmd in REPL_COMMANDS:
        print("    " + _c(f"{cmd.name:28}", "1;36") + _c(cmd.summary, "2"))
