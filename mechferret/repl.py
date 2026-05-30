"""Interactive REPL — a Claude-Code-style prompt bar for MechFerret.

Run `mechferret` with no arguments (or `mechferret repl`) to drop into an
interactive session: type a research question in plain English and it runs the
autonomous discovery loop; type a `/command` to drive the system. Input editing
and history come from `readline` when available.
"""

from __future__ import annotations

import os
import shlex
import sys
import uuid
from pathlib import Path

try:
    import readline  # noqa: F401  (enables arrow keys + history on input())
except ImportError:  # pragma: no cover - readline missing on some platforms
    readline = None

# ANSI styling (auto-disabled when output is not a TTY or NO_COLOR is set).
_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


REPL_ONLY = {"help", "exit", "quit", "q", "clear", "open", "session"}
KNOWN_COMMANDS = {
    "run", "demo", "discover", "login", "api", "goal", "loop", "doctor",
    "registry", "memory", "cost", "resume", "inspect", "skills", "modal", "cluster",
}

HISTORY_FILE = Path.home() / ".mechferret" / "repl_history"


class Session:
    def __init__(self) -> None:
        self.model = "gpt2"
        self.backend = "auto"
        self.task: str | None = None
        self.provider = "auto"
        self.out_root = Path("runs")
        self.last_report: str | None = None
        self.run_count = 0


def _banner() -> str:
    title = _c("MechFerret", "1;36") + _c("  ·  autonomous interpretability research agent", "36")
    hint = _c("type a question, or ", "2") + _c("/help", "1;33") + _c(" for commands · ", "2") + _c("/exit", "1;33") + _c(" to quit", "2")
    rows = [title, hint]
    width = max(_visible_len(row) for row in rows) + 4  # 2-space padding each side
    top = "╭" + "─" * width + "╮"
    bottom = "╰" + "─" * width + "╯"
    body = "\n".join(
        "│  " + row + " " * (width - 2 - _visible_len(row)) + "│" for row in rows
    )
    return f"{top}\n{body}\n{bottom}"


def _visible_len(text: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", text))


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


def run_repl() -> None:
    from .discovery import DiscoveryController
    from .interp.tasks import infer_task

    session = Session()
    controller = DiscoveryController()
    _setup_history()
    print(_banner())
    print()

    while True:
        try:
            line = input(_c("❯ ", "1;36")).strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print(_c("  (press Ctrl-D or type /exit to quit)", "2"))
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
            continue
        if bare == "session":
            _print_session(session)
            continue
        if bare == "open":
            _open_report(session)
            continue
        if bare in {"model", "backend", "task", "provider"} and head.startswith("/"):
            _set_session(session, bare, tokens[1:])
            continue
        if bare == "skill":
            if len(tokens) < 2:
                print(_c("  usage: /skill <name>   (see /skills)", "33"))
            else:
                _run_skill(controller, session, tokens[1])
            continue

        if head.startswith("/") or bare in KNOWN_COMMANDS:
            _dispatch_command(tokens, bare)
            continue

        # Plain text => a research question.
        _run_question(controller, session, line, infer_task)

    _save_history()
    print(_c("bye 👋", "2"))


def _dispatch_command(tokens: list[str], bare: str) -> None:
    from .cli import main as cli_main

    argv = [bare] + tokens[1:]
    try:
        cli_main(argv)
    except SystemExit:
        # argparse calls sys.exit on bad input; keep the REPL alive.
        pass
    except KeyboardInterrupt:
        print(_c("  (interrupted)", "2"))
    except Exception as exc:  # noqa: BLE001 - surface, don't crash the session
        print(_c(f"  error: {exc}", "31"))


def _run_question(controller, session: "Session", question: str, infer_task) -> None:
    task = session.task or infer_task(question)
    session.run_count += 1
    out_dir = session.out_root / f"session-{session.run_count:02d}-{uuid.uuid4().hex[:6]}"
    print(_c(f"  ⠿ investigating [{task}] on {session.model} ({session.backend})…", "2"))
    try:
        run = controller.run(
            question=question,
            task=task,
            model=session.model,
            backend=session.backend,
            provider=session.provider,
            out_dir=out_dir,
        )
    except KeyboardInterrupt:
        print(_c("  (interrupted)", "2"))
        return
    except Exception as exc:  # noqa: BLE001
        print(_c(f"  error: {exc}", "31"))
        return
    _print_run(run)
    session.last_report = run.artifacts.get("html")


def _run_skill(controller, session: "Session", skill_name: str) -> None:
    session.run_count += 1
    out_dir = session.out_root / f"session-{session.run_count:02d}-{uuid.uuid4().hex[:6]}"
    print(_c(f"  ⠿ running skill [{skill_name}] on {session.model} ({session.backend})…", "2"))
    try:
        run = controller.run(
            skill=skill_name,
            model=session.model,
            backend=session.backend,
            provider=session.provider,
            out_dir=out_dir,
        )
    except KeyError:
        print(_c(f"  unknown skill: {skill_name}  (type /skills to list)", "33"))
        return
    except KeyboardInterrupt:
        print(_c("  (interrupted)", "2"))
        return
    except Exception as exc:  # noqa: BLE001
        print(_c(f"  error: {exc}", "31"))
        return
    _print_run(run)
    session.last_report = run.artifacts.get("html")


def _print_run(run) -> None:
    rigor = run.metrics.get("rigor_score", 0)
    ready = run.metrics.get("readiness_score", 0)
    print()
    print(_c(f"  {len(run.discoveries)} confirmed mechanism(s)", "1;32")
          + _c(f"  ·  rigor {rigor:.2f} · readiness {ready:.2f} · "
               f"{int(run.metrics.get('experiments_run', 0))} experiments", "2"))
    for d in run.discoveries:
        print("  " + _c("•", "32") + " " + d.statement)
        print(_c(f"      confidence {d.confidence:.2f} · effect {d.effect_size:.2f} · "
                 f"reproducibility {d.reproducibility:.2f} · novelty {d.novelty:.2f}", "2"))
    if not run.discoveries:
        print(_c("  No mechanism cleared the rigor bar. Try a different --task or model.", "33"))
    report = run.artifacts.get("html")
    if report:
        print(_c(f"  dossier: {report}", "36") + _c("   (type /open to view)", "2"))
    print()


def _set_session(session: "Session", field: str, args: list[str]) -> None:
    if not args:
        print(_c(f"  {field} = {getattr(session, field)}", "2"))
        return
    value = args[0]
    if field == "backend" and value not in {"auto", "synthetic", "transformer_lens"}:
        print(_c("  backend must be auto | synthetic | transformer_lens", "33"))
        return
    if field == "task" and value not in {"ioi", "induction", "greater_than", "factual_recall"}:
        print(_c("  task must be ioi | induction | greater_than | factual_recall", "33"))
        return
    setattr(session, field, value)
    print(_c(f"  {field} → {value}", "32"))


def _print_session(session: "Session") -> None:
    print(_c("  session settings:", "1"))
    for field in ("model", "backend", "task", "provider"):
        print(f"    {field:9} {getattr(session, field)}")
    print(f"    runs      {session.run_count}")


def _open_report(session: "Session") -> None:
    if not session.last_report:
        print(_c("  no report yet — ask a question first", "33"))
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    os.system(f"{opener} {shlex.quote(session.last_report)}")


def _print_help() -> None:
    rows = [
        ("<your question>", "run the discovery loop on a plain-English question"),
        ("/skill <name>", "run a saved playbook (e.g. /skill ioi-circuit) via /discover"),
        ("/discover ...", "discovery with explicit flags (--skill --task --model --backend)"),
        ("/skills [name]", "list playbooks, or show one"),
        ("/model <name>", "set the model under study (e.g. gpt2, pythia-160m)"),
        ("/backend <b>", "auto | synthetic | transformer_lens"),
        ("/task <t>", "pin the task: ioi | induction | greater_than | factual_recall"),
        ("/modal <action>", "status | setup | run | deploy  (GPU on Modal)"),
        ("/cluster <action>", "status | setup | run  (your own SLURM cluster)"),
        ("/doctor /registry /memory", "environment, capabilities, recalled runs"),
        ("/open", "open the last run's HTML dossier"),
        ("/session", "show current session settings"),
        ("/clear  /exit", "clear the screen · quit"),
    ]
    print(_c("  commands:", "1"))
    for cmd, desc in rows:
        print("    " + _c(f"{cmd:28}", "1;36") + _c(desc, "2"))
    print(_c("\n  note: `/skill ioi-circuit` is shorthand for `/discover --skill ioi-circuit`.", "2"))
