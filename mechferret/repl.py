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
    print()
    print(_indent(reply))
    print(_c(f"  ({agent.cost.format_total()})", "2"))
    print()


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
    rows = [
        ("<your prompt>", "talk to the model; it runs experiments when you ask"),
        ("/login", "connect or change your model API key"),
        ("/model <name>", "set the conversation model (e.g. claude-sonnet-4-5)"),
        ("/plan", "toggle plan mode (approve write/exec/GPU tools)"),
        ("/cost", "show session token + USD usage"),
        ("/discover ...", "run discovery directly (--skill --task --model)"),
        ("/skills [name]", "list playbooks, or show one"),
        ("/modal <action>", "status | setup | run | deploy  (GPU on Modal)"),
        ("/cluster <action>", "status | setup | run  (your own SLURM cluster)"),
        ("/doctor /registry /memory", "environment, capabilities, recalled runs"),
        ("/open", "open the last run's HTML dossier"),
        ("/clear  /exit", "redraw welcome · quit"),
    ]
    print(_c("  commands:", "1"))
    for cmd, desc in rows:
        print("    " + _c(f"{cmd:28}", "1;36") + _c(desc, "2"))
