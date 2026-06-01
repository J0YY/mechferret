"""Interactive REPL — a Claude-Code-style prompt for MechFerret.

Run `mechferret` with no arguments (or `mechferret repl`) to drop into an
interactive session. Plain-English prompts are piped to a model (Claude or GPT)
that converses and calls MechFerret's discovery tools when you ask for
interpretability work. `/commands` drive the system directly. On your first
prompt, if no model is connected, it walks you through adding an API key.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import sys
import threading
import time
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .commands import COMMAND_WORDS

try:
    import readline  # noqa: F401  (enables arrow keys + history on input())
except ImportError:  # pragma: no cover
    readline = None

_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
VERSION = "0.1.0"
WIDTH = 78
PURPLE = "38;5;141"  # soft violet
PURPLE_B = "1;38;5;141"
TERMINAL_JOB_STATUSES = {"done", "error", "canceled"}


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _vlen(text: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", text))


KNOWN_COMMANDS = COMMAND_WORDS

HISTORY_FILE = Path.home() / ".mechferret" / "repl_history"
QUEUE_FILE = Path.home() / ".mechferret" / "repl_queue.json"

# Small, cute ferret — equal-width lines so it centers as one aligned block.
FERRET = [
    " /\\_/\\ ",
    "( o.o )",
    " > ^ < ",
]


class Session:
    def __init__(self) -> None:
        self.out_root = Path("runs")
        self.last_report: str | None = None
        self.run_count = 0
        self.goal = ""
        self.step = ""


@dataclass(slots=True)
class PromptJob:
    id: int
    text: str
    kind: str = "prompt"
    status: str = "queued"
    reply: str | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)


class ChatJobRunner:
    def __init__(
        self,
        agent: Any,
        session: Session,
        *,
        chat_fn: Callable[..., str | None] | None = None,
        side_agent_factory: Callable[[Any], Any] | None = None,
        queue_path: Path | None = None,
    ) -> None:
        self.agent = agent
        self.session = session
        self._chat_fn = chat_fn or _chat
        self._side_agent_factory = side_agent_factory or _clone_agent_for_side_chat
        self._queue_path = queue_path or QUEUE_FILE
        self._queue: queue.Queue[PromptJob | None] = queue.Queue()
        self._jobs: list[PromptJob] = []
        self._active: PromptJob | None = None
        self._side_threads: list[threading.Thread] = []
        self._next_id = 1
        self._lock = threading.Lock()
        self._stopped = False
        self._paused = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, text: str, *, kind: str = "prompt") -> PromptJob:
        with self._lock:
            job = PromptJob(id=self._next_id, text=text, kind=kind)
            self._next_id += 1
            self._jobs.append(job)
        self._queue.put(job)
        self.save_pending()
        return job

    def submit_side(self, text: str) -> PromptJob:
        with self._lock:
            job = PromptJob(id=self._next_id, text=text, kind="btw", status="running")
            self._next_id += 1
            self._jobs.append(job)
        thread = threading.Thread(target=self._run_side, args=(job,), daemon=True)
        with self._lock:
            self._side_threads.append(thread)
        self.save_pending(include_active=True)
        thread.start()
        return job

    def saved(self) -> list[PromptJob]:
        return _load_saved_queue(self._queue_path)

    def restore_saved(self) -> list[PromptJob]:
        saved_jobs = self.saved()
        if not saved_jobs:
            return []
        self.clear_saved()
        restored: list[PromptJob] = []
        for saved in saved_jobs:
            if saved.kind == "btw":
                restored.append(self.submit_side(saved.text))
            else:
                restored.append(self.submit(saved.text, kind=saved.kind))
        return restored

    def clear_saved(self) -> int:
        saved = self.saved()
        try:
            self._queue_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return 0
        return len(saved)

    def pause(self) -> bool:
        with self._lock:
            was_paused = self._paused
            self._paused = True
        self.save_pending(include_active=True)
        return not was_paused

    def resume(self) -> bool:
        with self._lock:
            was_paused = self._paused
            self._paused = False
        return was_paused

    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def active(self) -> PromptJob | None:
        with self._lock:
            return self._active

    def side_active(self) -> list[PromptJob]:
        with self._lock:
            return [job for job in self._jobs if job.kind == "btw" and job.status == "running"]

    def queued(self) -> list[PromptJob]:
        with self._lock:
            return [job for job in self._jobs if job.status == "queued"]

    def cancel(self, target: str) -> list[PromptJob]:
        target = target.strip().lower()
        canceled: list[PromptJob] = []
        with self._lock:
            for job in self._jobs:
                if job.status != "queued":
                    continue
                if target == "all" or str(job.id) == target:
                    job.status = "canceled"
                    canceled.append(job)
        if canceled:
            self.save_pending()
        return canceled

    def edit(self, target: str, text: str) -> tuple[PromptJob | None, str]:
        target = target.strip().lstrip("#")
        edited: PromptJob | None = None
        with self._lock:
            for job in self._jobs:
                if str(job.id) != target:
                    continue
                if job.status != "queued":
                    return job, job.status
                job.text = text
                edited = job
                break
        if edited is None:
            return None, "missing"
        self.save_pending()
        return edited, "updated"

    def move(self, target: str, where: str, anchor: str = "") -> tuple[PromptJob | None, str]:
        target = target.strip().lstrip("#")
        where = where.strip().lower()
        anchor = anchor.strip().lstrip("#")
        with self._lock:
            job = next((item for item in self._jobs if str(item.id) == target), None)
            if job is None:
                return None, "missing"
            if job.status != "queued":
                return job, job.status
            queued = [item for item in self._jobs if item.status == "queued"]
            if where not in {"first", "last", "before", "after"}:
                return job, "usage"
            if where in {"before", "after"}:
                anchor_job = next((item for item in queued if str(item.id) == anchor), None)
                if anchor_job is None:
                    return job, "anchor"
                if anchor_job is job:
                    return job, "same"
                queued.remove(job)
                index = queued.index(anchor_job)
                if where == "after":
                    index += 1
                queued.insert(index, job)
            else:
                queued.remove(job)
                if where == "first":
                    queued.insert(0, job)
                else:
                    queued.append(job)

            queued_iter = iter(queued)
            self._jobs = [next(queued_iter) if item.status == "queued" else item for item in self._jobs]
        self.save_pending()
        return job, "moved"

    def recent(self, limit: int = 8) -> list[PromptJob]:
        with self._lock:
            return list(self._jobs[-limit:])

    def find_job(self, target: str) -> tuple[PromptJob | None, bool]:
        target = target.strip().lstrip("#")
        with self._lock:
            for job in self._jobs:
                if str(job.id) == target:
                    return job, False
        for job in self.saved():
            if str(job.id) == target:
                return job, True
        return None, False

    def retry(self, target: str) -> tuple[PromptJob | None, PromptJob | None, bool]:
        original, saved = self.find_job(target)
        if original is None:
            return None, None, False
        if not saved and original.status in {"queued", "running"}:
            return original, None, saved
        if original.kind == "btw":
            retried = self.submit_side(original.text)
        else:
            retried = self.submit(original.text, kind=original.kind)
        return original, retried, saved

    def is_busy(self) -> bool:
        with self._lock:
            return self._active is not None or any(job.status in {"queued", "running"} for job in self._jobs)

    def main_busy(self) -> bool:
        with self._lock:
            return self._active is not None or any(job.status == "queued" for job in self._jobs)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_busy():
                self.save_pending()
                return True
            time.sleep(0.01)
        return False

    def wait_job(self, target: str, timeout: float = 3600.0) -> tuple[PromptJob | None, bool, bool]:
        deadline = time.monotonic() + timeout
        while True:
            job, saved = self.find_job(target)
            if job is None or saved or job.status in TERMINAL_JOB_STATUSES:
                if job is not None and not saved:
                    self.save_pending()
                return job, saved, job is not None and job.status in TERMINAL_JOB_STATUSES
            if time.monotonic() >= deadline:
                return job, saved, False
            time.sleep(0.01)

    def stop(self, *, wait: bool = False) -> None:
        if self._stopped:
            if wait:
                self._thread.join(timeout=2)
            return
        self._stopped = True
        self._queue.put(None)
        if wait:
            self._thread.join(timeout=2)
            for thread in list(self._side_threads):
                thread.join(timeout=2)
        self.save_pending(include_active=True)

    def save_pending(self, *, include_active: bool = False) -> int:
        with self._lock:
            pending = [job for job in self._jobs if job.status == "queued"]
            if include_active and self._active is not None and self._active.status == "running":
                pending = [self._active, *pending]
            if include_active:
                pending.extend(job for job in self._jobs if job.kind == "btw" and job.status == "running")
        return _save_queue_jobs(self._queue_path, pending)

    def _set_active(self, job: PromptJob | None) -> None:
        with self._lock:
            self._active = job

    def _wait_if_paused(self, job: PromptJob) -> None:
        while True:
            with self._lock:
                if self._stopped or not self._paused or job.status == "canceled":
                    return
            time.sleep(0.05)

    def _is_next_queued(self, job: PromptJob) -> bool:
        with self._lock:
            return next((item is job for item in self._jobs if item.status == "queued"), False)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._wait_if_paused(job)
                if self._stopped:
                    continue
                if job.status == "canceled":
                    print(_c(f"  skipped canceled #{job.id}", "2"))
                    continue
                if not self._is_next_queued(job):
                    self._queue.put(job)
                    continue
                job.status = "running"
                self._set_active(job)
                self.save_pending()
                print(_c(f"  ▶ queued #{job.id} {job.kind}: {_short_job_text(job.text)}", "2"))
                reply = self._chat_fn(self.agent, self.session, job.text, background=True)
                job.reply = reply
                job.status = "done"
                self.save_pending()
                _print_finished(job)
            except Exception as exc:  # noqa: BLE001 - background work should not kill the prompt
                if job is not None:
                    job.status = "error"
                    job.error = str(exc)
                    self.save_pending()
                    print(_c(f"  error in queued #{job.id}: {exc}", "31"))
                    _print_job_result_hint(job)
            finally:
                self._set_active(None)
                self._queue.task_done()

    def _run_side(self, job: PromptJob) -> None:
        try:
            print(_c(f"  ▶ side #{job.id}: {_short_job_text(job.text)}", "2"))
            side_agent = self._side_agent_factory(self.agent)
            side_session = Session()
            reply = self._chat_fn(side_agent, side_session, job.text, background=True)
            job.reply = reply
            job.status = "done"
            self.save_pending()
            _print_finished(job, label="side")
        except Exception as exc:  # noqa: BLE001 - side work should not kill the prompt
            job.status = "error"
            job.error = str(exc)
            self.save_pending()
            print(_c(f"  error in side #{job.id}: {exc}", "31"))
            _print_job_result_hint(job)


def _job_to_dict(job: PromptJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "text": job.text,
        "kind": job.kind,
        "status": job.status,
        "created_at": job.created_at,
    }


def _load_saved_queue(path: Path = QUEUE_FILE) -> list[PromptJob]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    rows = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    jobs: list[PromptJob] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        kind = row.get("kind") if isinstance(row.get("kind"), str) else "prompt"
        created_at = row.get("created_at") if isinstance(row.get("created_at"), (int, float)) else time.time()
        raw_id = row.get("id")
        job_id = int(raw_id) if isinstance(raw_id, int) and raw_id > 0 else len(jobs) + 1
        jobs.append(PromptJob(id=job_id, text=text, kind=kind, created_at=float(created_at)))
    return jobs


def _save_queue_jobs(path: Path, jobs: list[PromptJob]) -> int:
    if not jobs:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"jobs": [_job_to_dict(job) for job in jobs]}, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return len(jobs)


def _clone_agent_for_side_chat(agent: Any) -> Any:
    from .agent import Agent

    side = Agent()
    for name in ("provider", "model", "_key", "permission_mode"):
        if hasattr(agent, name):
            setattr(side, name, getattr(agent, name))
    side.messages = deepcopy(getattr(agent, "messages", []))
    return side


# --- welcome screen ------------------------------------------------------------------

def _two_column_box(left: list[str], right: list[str]) -> str:
    """A snug, content-sized box; every cell centered in its column."""

    lw = max((_vlen(x) for x in left), default=0)
    rw = max((_vlen(x) for x in right), default=0)
    rows = max(len(left), len(right))

    def vcenter(col: list[str]) -> list[str]:
        gap = rows - len(col)
        top = gap // 2
        return [""] * top + list(col) + [""] * (gap - top)

    def cell(s: str, w: int) -> str:
        pad = w - _vlen(s)
        a = pad // 2
        return " " * a + s + " " * (pad - a)

    L, R = vcenter(left), vcenter(right)
    body = [f"│ {cell(l, lw)} │ {cell(r, rw)} │" for l, r in zip(L, R)]
    inner = _vlen(body[0]) - 2  # chars between the two border columns
    title = _c(f" MechFerret v{VERSION} ", PURPLE_B)
    top = "╭─" + title + "─" * max(0, inner - _vlen(title) - 1) + "╮"
    bottom = "╰" + "─" * inner + "╯"
    return "\n".join([top, *body, bottom])


def _welcome(session: Session) -> str:
    from .agent import active_provider

    provider, model, _key = active_provider()
    user = os.getenv("USER") or "there"
    cwd = str(Path.cwd()).replace(str(Path.home()), "~")
    status = _c(model, PURPLE) if provider else _c("offline", "2")

    left = [
        _c(f"Welcome back {user.capitalize()}!", "1"),
        *[_c(line, PURPLE) for line in FERRET],
        _c("mechferret", PURPLE_B),
        status,
        _c(cwd, "2"),
    ]
    right = [
        _c("About", "1"),
        _c("agentic interp research agent", "2"),
        "",
        _c("Help", "1"),
        _c("/login   connect a model", "2"),
        _c("/model   pick the model", "2"),
        _c("/help    all commands", "2"),
        _c("ctrl-c   quit", "2"),
    ]
    return _two_column_box(left, right)


def _short_job_text(text: str, limit: int = 60) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _print_status_and_bar(agent, session, runner: ChatJobRunner | None = None) -> None:
    mode = getattr(agent, "permission_mode", "auto")
    bits = [_c(agent.model, PURPLE) if agent.configured else _c("offline", "2")]
    if agent.configured and agent.cost.usd:
        bits.append(_c(agent.cost.format_total(), "2"))
    bits.append(_c(f"mode:{mode}" + (" ⏸" if mode == "plan" else ""), "33" if mode == "plan" else "2"))
    if runner is not None:
        active = runner.active()
        side_active = len(runner.side_active())
        queued = len(runner.queued())
        saved = len(runner.saved())
        if runner.paused():
            bits.append(_c("paused", "33"))
        if active is not None:
            bits.append(_c(f"running:#{active.id}", PURPLE))
        if side_active:
            bits.append(_c(f"btw:{side_active}", PURPLE))
        if queued:
            bits.append(_c(f"queued:{queued}", "33"))
        if saved:
            bits.append(_c(f"saved:{saved}", "33"))
    print(_c("─" * WIDTH, "2"))
    print("  " + _c(" · ", "2").join(bits))
    # sticky tl;dr line: research goal + current step
    tldr = []
    if session.goal:
        tldr.append(_c(f"🎯 {session.goal}", PURPLE))
    tldr.append(_c(f"◴ {session.step or 'ready — type a prompt, or /goal to set an objective'}", "2"))
    print("  " + _c(" · ", "2").join(tldr))


def _ferret_walk() -> None:
    """A tiny ferret scampers across the screen before the welcome box appears."""

    if not _COLOR:
        return
    import time

    n = len(FERRET)
    span = max(8, WIDTH - 14)
    sys.stdout.write("\n" * n)  # reserve n lines
    for step in range(span // 2 + 1):
        x = step * 2
        sys.stdout.write(f"\x1b[{n}A")  # cursor up to top of the reserved block
        for i, line in enumerate(FERRET):
            bob = 1 if (step + i) % 2 == 0 else 0  # little gait wiggle
            sys.stdout.write("\r\x1b[2K" + _c(" " * (x + bob) + line, PURPLE) + "\n")
        sys.stdout.flush()
        time.sleep(0.03)
    sys.stdout.write(f"\x1b[{n}A")  # clear the walk and reset cursor for the box
    for _ in range(n):
        sys.stdout.write("\r\x1b[2K\n")
    sys.stdout.write(f"\x1b[{n}A")
    sys.stdout.flush()


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
    runner = ChatJobRunner(agent, session)
    _setup_history()
    _ferret_walk()
    print(_welcome(session))
    print()

    while True:
        _print_status_and_bar(agent, session, runner)
        try:
            prompt = "queued ❯ " if runner.is_busy() else "❯ "
            line = input(_c(prompt, "1;36")).strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-C / Ctrl-D at the prompt quits.
            print()
            break
        if not line:
            # Empty enter mid-conversation = "keep going" (accept the proposed next step).
            if runner.is_busy():
                _print_queue(runner)
            elif agent.configured and agent.messages:
                job = runner.submit("Proceed with the next step you proposed. Keep building.")
                _print_queued(job, runner)
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
        if bare == "queue":
            if len(tokens) > 1 and tokens[1].lower() == "restore":
                restored = runner.restore_saved()
                if restored:
                    ids = ", ".join(f"#{job.id}" for job in restored)
                    print(_c(f"  restored {ids}", "32"))
                else:
                    print(_c("  no saved queued prompts to restore", "2"))
            elif len(tokens) > 1 and tokens[1].lower() == "clear":
                _queue_clear(runner, tokens[2:])
            elif len(tokens) > 1 and tokens[1].lower() == "cancel":
                _queue_cancel(runner, tokens[2:])
            elif len(tokens) > 1 and tokens[1].lower() == "wait":
                _queue_wait(runner, tokens[2:])
            elif len(tokens) > 1 and tokens[1].lower() == "join":
                _queue_join(runner, tokens[2:])
            elif len(tokens) > 1 and tokens[1].lower() == "show":
                _queue_show(runner, tokens[2:])
            elif len(tokens) > 1 and tokens[1].lower() == "retry":
                _queue_retry(runner, tokens[2:])
            elif len(tokens) > 1 and tokens[1].lower() == "edit":
                _queue_edit(runner, tokens[2:], _line_after_words(line, 3))
            elif len(tokens) > 1 and tokens[1].lower() == "move":
                _queue_move(runner, tokens[2:])
            elif len(tokens) > 1 and tokens[1].lower() == "pause":
                if runner.pause():
                    print(_c("  queue paused; active work can finish, but queued prompts will wait", "32"))
                else:
                    print(_c("  queue already paused", "2"))
            elif len(tokens) > 1 and tokens[1].lower() == "resume":
                if runner.resume():
                    print(_c("  queue resumed", "32"))
                else:
                    print(_c("  queue was not paused", "2"))
            else:
                _print_queue(runner)
            continue
        if bare == "cancel":
            target = tokens[1] if len(tokens) > 1 else ""
            if not target:
                print(_c("  usage: /cancel <job id|all>", "33"))
                continue
            canceled = runner.cancel(target)
            if canceled:
                _print_canceled(canceled)
            else:
                print(_c(f"  no queued job matched {target!r}", "33"))
            continue
        if bare == "btw":
            text = _line_after_command(line)
            if not text:
                print(_c("  usage: /btw <side prompt>", "33"))
                continue
            if not agent.configured:
                print(_c("  No model connected yet — let's fix that.", "2"))
                if not onboard():
                    continue
                agent.reload()
            job = runner.submit_side(_btw_prompt(text))
            print(_c(f"  started side #{job.id} btw", "2"))
            continue
        if bare == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(_welcome(session))
            continue
        if bare == "open" and _uses_repl_shortcut(bare, tokens):
            _open_report(session)
            continue
        if bare in {"login", "connect"}:
            if not _guard_agent_idle(runner, f"/{bare}"):
                continue
            if onboard():
                agent.reload()
            continue
        if bare == "model" and head.startswith("/"):
            if not _guard_agent_idle(runner, "/model"):
                continue
            _set_model(agent, tokens[1:])
            continue
        if bare == "plan":
            if not _guard_agent_idle(runner, "/plan"):
                continue
            agent.permission_mode = "auto" if agent.permission_mode == "plan" else "plan"
            on = agent.permission_mode == "plan"
            print(_c(f"  plan mode {'ON — write/exec/GPU tools will ask before running' if on else 'OFF'}", "33"))
            continue
        if bare == "goal":
            if len(tokens) > 1:
                if not _guard_agent_idle(runner, "/goal"):
                    continue
                _goal_loop(agent, session, " ".join(tokens[1:]))
            elif session.goal:
                print(_c(f"  🎯 current goal: {session.goal}", PURPLE))
            else:
                print(_c("  usage: /goal <your research objective>", "33"))
            continue
        if bare == "demo" and _uses_repl_shortcut(bare, tokens):
            from .ops import print_quickstart_run, run_quickstart

            try:
                result = run_quickstart("demo")
                print_quickstart_run(result)
            except KeyboardInterrupt:
                print(_c("\n  (demo stopped)", "2"))
            except Exception as exc:  # noqa: BLE001
                print(_c(f"  demo failed: {exc}", "31"))
            continue
        if bare == "why":
            _why()
            continue
        if bare == "arch":
            _arch()
            continue
        if bare == "status" and _uses_repl_shortcut(bare, tokens):
            _status(tokens[1:])
            continue
        if bare == "paper" and _uses_repl_shortcut(bare, tokens):
            _paper(agent, session, tokens[1:])
            continue
        if bare == "audit" and _uses_repl_shortcut(bare, tokens):
            _audit(tokens[1:])
            continue
        if bare == "quickstart" and _uses_repl_shortcut(bare, tokens):
            from .ops import print_quickstart, quickstart

            mode = tokens[1] if len(tokens) > 1 else "all"
            try:
                print_quickstart(quickstart(mode))
            except ValueError as exc:
                print(_c(f"  {exc}", "31"))
            continue
        if bare == "review-paper" and _uses_repl_shortcut(bare, tokens):
            _review_paper(agent, tokens[1:])
            continue
        if bare == "cost" and len(tokens) == 1:
            print(_c(f"  session: {agent.cost.format_total()}  ·  model {agent.model}  ·  denied tools: {len(agent.denials)}", "2"))
            for model, slot in agent.cost.by_model.items():
                print(_c(f"    {model}: ${slot['usd']:.4f}  ({int(slot['input'])} in / {int(slot['output'])} out)", "2"))
            continue
        if bare == "compact":
            if not _guard_agent_idle(runner, "/compact"):
                continue
            summary = agent.compact()
            if summary == "nothing to compact":
                print(_c("  " + summary, "2"))
            else:
                print(_c("  ✓ compacted older turns into a summary:", "32"))
                print(_indent(_c(summary[:800], "2")))
            continue
        if bare == "resume":
            if not _guard_agent_idle(runner, "/resume"):
                continue
            _resume(agent, tokens[1:])
            continue
        if bare == "memory" and len(tokens) == 1:
            _show_memory()
            continue
        if bare == "trace":
            _show_trace()
            continue
        if bare == "mcp":
            _mcp(tokens[1:])
            continue
        if bare == "init" and _uses_repl_shortcut(bare, tokens):
            _init_project()
            continue
        if bare == "export":
            if not _guard_agent_idle(runner, "/export"):
                continue
            _export(agent, tokens[1:])
            continue
        if bare == "review":
            if not _guard_agent_idle(runner, "/review"):
                continue
            import subprocess

            diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True).stdout
            if not diff.strip():
                print(_c("  no uncommitted changes to review", "33"))
                continue
            _chat(agent, session, (
                "Review this git diff for an interpretability-research codebase. Flag correctness bugs, "
                "missing experimental controls or seed logging, data leakage into activations/probes, "
                "and reproducibility issues. Be specific with file:line.\n\n" + diff[:40000]
            ))
            continue

        if head.startswith("/") or bare in KNOWN_COMMANDS:
            _dispatch_command(tokens, bare)
            continue

        # Plain text => talk to the model.
        if not agent.configured:
            print(_c("  No model connected yet — let's fix that.", "2"))
            if not onboard():
                continue
            agent.reload()
        job = runner.submit(line)
        _print_queued(job, runner)

    runner.stop(wait=False)
    _save_history()
    print(_c("bye 👋", "2"))


def _line_after_command(line: str) -> str:
    parts = line.strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _line_after_words(line: str, count: int) -> str:
    text = line.strip()
    for _ in range(count):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return ""
        text = parts[1].strip()
    return text


def _btw_prompt(text: str) -> str:
    return (
        "Side request entered with /btw while another prompt was running. "
        "Answer it as a compact aside unless it changes the active work:\n\n"
        + text
    )


def _print_queue(runner: ChatJobRunner) -> None:
    active = runner.active()
    side_active = runner.side_active()
    queued = runner.queued()
    recent = runner.recent()
    saved = runner.saved()
    if runner.paused():
        print(_c("  queue paused; queued prompts will wait for /queue resume", "33"))
    if active is None and not side_active and not queued and not saved:
        print(_c("  queue empty", "2"))
    else:
        if active is not None:
            print(_c(f"  running #{active.id} {active.kind}: {_short_job_text(active.text)}", PURPLE))
        for job in side_active:
            print(_c(f"  side    #{job.id} {job.kind}: {_short_job_text(job.text)}", PURPLE))
        for job in queued:
            print(_c(f"  queued  #{job.id} {job.kind}: {_short_job_text(job.text)}", "2"))
        for job in saved:
            print(_c(f"  saved   #{job.id} {job.kind}: {_short_job_text(job.text)}", "33"))
        if saved:
            print(_c("  run `/queue restore` to enqueue saved prompts, or `/queue clear` to drop them", "2"))
    done = [job for job in recent if job.status in {"done", "error", "canceled"}]
    for job in done[-3:]:
        status = job.status
        detail = f" ({job.error})" if job.error else ""
        print(_c(f"  {status:8} #{job.id} {job.kind}{detail}", "31" if job.error else "2"))


def _print_queued(job: PromptJob, runner: ChatJobRunner) -> None:
    queued = runner.queued()
    if job in queued:
        position = queued.index(job) + 1
        print(_c(f"  queued #{job.id} (position {position}/{len(queued)})", "2"))
    else:
        print(_c(f"  queued #{job.id}", "2"))
    print(_c(f"  use /queue edit #{job.id} <prompt>, /queue move #{job.id} first, or /queue cancel #{job.id}", "2"))


def _print_job_result_hint(job: PromptJob) -> None:
    print(_c(f"  use /queue show #{job.id} to view the prompt, reply, or error", "2"))


def _print_finished(job: PromptJob, *, label: str = "queued") -> None:
    prefix = f"finished {label}" if label != "queued" else "finished"
    print(_c(f"  ✓ {prefix} #{job.id}", "32"))
    _print_job_result_hint(job)


def _print_canceled(canceled: list[PromptJob]) -> None:
    ids = ", ".join(f"#{job.id}" for job in canceled)
    print(_c(f"  canceled {ids}", "32"))


def _queue_cancel(runner: ChatJobRunner, args: list[str]) -> None:
    target = args[0] if args else ""
    if not target:
        print(_c("  usage: /queue cancel <job id|all>", "33"))
        return
    canceled = runner.cancel(target)
    if canceled:
        _print_canceled(canceled)
    else:
        print(_c(f"  no queued job matched {target!r}", "33"))


def _queue_clear(runner: ChatJobRunner, args: list[str]) -> None:
    scope = args[0].lower() if args else "saved"
    if scope not in {"queued", "saved", "all"}:
        print(_c("  usage: /queue clear [queued|saved|all]", "33"))
        return
    canceled: list[PromptJob] = []
    cleared = 0
    if scope in {"queued", "all"}:
        canceled = runner.cancel("all")
    if scope in {"saved", "all"}:
        cleared = runner.clear_saved()
    if canceled:
        _print_canceled(canceled)
    elif scope in {"queued", "all"}:
        print(_c("  no queued prompts to cancel", "2"))
    if scope in {"saved", "all"}:
        print(_c(f"  cleared {cleared} saved queued prompt(s)", "32" if cleared else "2"))


def _queue_wait(runner: ChatJobRunner, args: list[str]) -> None:
    timeout = 3600.0
    if args:
        try:
            timeout = float(args[0])
        except ValueError:
            print(_c("  usage: /queue wait [seconds]", "33"))
            return
        if timeout <= 0:
            print(_c("  wait timeout must be positive", "33"))
            return
    if runner.paused():
        print(_c("  queue paused; use /queue resume before waiting for queued work", "33"))
        return
    if not runner.is_busy():
        print(_c("  queue already idle", "2"))
        return
    print(_c("  waiting for active queued work…", "2"))
    if runner.wait_idle(timeout=timeout):
        print(_c("  queue idle", "32"))
    else:
        print(_c("  queue still active after timeout", "33"))
        _print_queue(runner)


def _queue_join(runner: ChatJobRunner, args: list[str]) -> None:
    target = args[0] if args else ""
    timeout = 3600.0
    if not target:
        print(_c("  usage: /queue join <job id> [seconds]", "33"))
        return
    if len(args) > 1:
        try:
            timeout = float(args[1])
        except ValueError:
            print(_c("  usage: /queue join <job id> [seconds]", "33"))
            return
        if timeout <= 0:
            print(_c("  join timeout must be positive", "33"))
            return
    job, saved = runner.find_job(target)
    if job is None:
        print(_c(f"  no queue job matched {target!r}", "33"))
        return
    if saved:
        print(_c(f"  job #{job.id} is saved; run /queue restore before joining it.", "33"))
        return
    if runner.paused() and job.status == "queued":
        print(_c("  queue paused; use /queue resume before joining queued work", "33"))
        return
    if job.status not in TERMINAL_JOB_STATUSES:
        print(_c(f"  waiting for job #{job.id}…", "2"))
    job, _saved, finished = runner.wait_job(target, timeout=timeout)
    if job is None:
        print(_c(f"  no queue job matched {target!r}", "33"))
        return
    if finished:
        print(_c(f"  job #{job.id} {job.status}", "32" if job.status == "done" else "33"))
        _print_job_result_hint(job)
        return
    print(_c(f"  job #{job.id} still {job.status} after timeout", "33"))
    _print_queue(runner)


def _queue_show(runner: ChatJobRunner, args: list[str]) -> None:
    target = args[0] if args else ""
    if not target:
        print(_c("  usage: /queue show <job id>", "33"))
        return
    job, saved = runner.find_job(target)
    if job is None:
        print(_c(f"  no queue job matched {target!r}", "33"))
        return
    saved_label = " saved" if saved else ""
    print(_c(f"  job #{job.id}{saved_label} · {job.kind} · {job.status}", PURPLE_B))
    if job.error:
        print(_c("  error:", "31"))
        print(_indent(job.error))
    print(_c("  prompt:", "1"))
    print(_render_reply(job.text))
    if job.reply:
        print(_c("  reply:", "1"))
        print(_render_reply(job.reply))


def _queue_retry(runner: ChatJobRunner, args: list[str]) -> None:
    target = args[0] if args else ""
    if not target:
        print(_c("  usage: /queue retry <job id>", "33"))
        return
    original, retried, saved = runner.retry(target)
    if original is None:
        print(_c(f"  no queue job matched {target!r}", "33"))
        return
    if retried is None:
        print(_c(f"  job #{original.id} is {original.status}; wait or cancel it before retrying.", "33"))
        return
    saved_label = " saved" if saved else ""
    print(_c(f"  retried #{original.id}{saved_label} as #{retried.id}", "32"))


def _queue_edit(runner: ChatJobRunner, args: list[str], text: str) -> None:
    target = args[0] if args else ""
    if not target or not text:
        print(_c("  usage: /queue edit <job id> <new prompt>", "33"))
        return
    job, status = runner.edit(target, text)
    if job is None:
        print(_c(f"  no queued job matched {target!r}", "33"))
        return
    if status != "updated":
        print(_c(f"  job #{job.id} is {status}; only queued prompts can be edited.", "33"))
        return
    print(_c(f"  edited #{job.id}", "32"))


def _queue_move(runner: ChatJobRunner, args: list[str]) -> None:
    target = args[0] if args else ""
    where = args[1].lower() if len(args) > 1 else ""
    anchor = args[2] if len(args) > 2 else ""
    if not target or where not in {"first", "last", "before", "after"} or (where in {"before", "after"} and not anchor):
        print(_c("  usage: /queue move <job id> first|last|before <job id>|after <job id>", "33"))
        return
    job, status = runner.move(target, where, anchor)
    if job is None:
        print(_c(f"  no queued job matched {target!r}", "33"))
        return
    if status == "usage":
        print(_c("  usage: /queue move <job id> first|last|before <job id>|after <job id>", "33"))
        return
    if status == "anchor":
        print(_c(f"  no queued anchor matched {anchor!r}", "33"))
        return
    if status == "same":
        print(_c(f"  job #{job.id} is already in that spot", "2"))
        return
    if status != "moved":
        print(_c(f"  job #{job.id} is {status}; only queued prompts can be moved.", "33"))
        return
    print(_c(f"  moved #{job.id} {where}{(' #' + anchor.lstrip('#')) if anchor else ''}", "32"))


def _guard_agent_idle(runner: ChatJobRunner, action: str) -> bool:
    if not runner.main_busy():
        return True
    print(_c(f"  {action} waits for the active prompt so conversation state stays consistent.", "33"))
    print(_c("  use /btw for side prompts, /queue to inspect, or /cancel <id|all> for queued work.", "2"))
    _print_queue(runner)
    return False


class _BackgroundPrinter:
    enabled = False

    def pause(self):
        return nullcontext()

    def log(self, text: str) -> None:
        print(text)

    def __enter__(self) -> "_BackgroundPrinter":
        return self

    def __exit__(self, *exc) -> None:
        return None


def _chat(agent, session, text: str, *, background: bool = False) -> str | None:
    from .spinner import Spinner

    if not agent.configured:
        if background:
            print(_c("  queued prompt needs a model; run /login, then submit it again.", "33"))
            return None
        print(_c("  No model connected yet — let's fix that.", "2"))
        if not onboard():
            return None
        agent.reload()
    spinner = _BackgroundPrinter() if background else Spinner()
    session.step = "thinking…"

    def _on_tool(name, args):
        session.step = f"running {name}"
        spinner.log(_tool_line(name, args))

    agent.on_tool = _on_tool

    def _confirm(name, args, reason):
        from .picker import select

        if background:
            spinner.log(_c(f"  skipped {name}: approval needed; rerun in foreground or turn plan mode off", "33"))
            return False
        with spinner.pause():
            print(_c(f"  ⚠ {name} — {reason or 'approval needed'}", "33"))
            try:
                return select(_c("  Allow this tool call?", "1"), ["yes, run it", "no, skip it"]).startswith("yes")
            except KeyboardInterrupt:
                return False

    agent.confirm = _confirm

    def _on_options(options):
        from .picker import select_rich

        if background:
            spinner.log(_c("  skipped interactive option picker in queued mode", "33"))
            return "none"
        with spinner.pause():
            choice = select_rich(_c("  Pick a direction  (↑/↓ move · → expand · enter select · esc skip)", "1"), options)
        spinner.log(_c(f"  ✓ selected: {choice}", "32") if choice != "none" else _c("  (skipped)", "2"))
        return choice

    agent.on_options = _on_options
    streamed = {"any": False}

    def _emit(block: str):
        streamed["any"] = True
        spinner.log("\n" + _render_reply(block) + "\n")

    agent.on_text = _emit
    try:
        with spinner:
            reply = agent.send(text)
    except KeyboardInterrupt:
        session.step = "interrupted"
        print(_c("  (interrupted — Ctrl-C again at the prompt to quit)", "2"))
        return None
    except Exception as exc:  # noqa: BLE001
        session.step = "error"
        print(_c(f"  error: {exc}", "31"))
        if "401" in str(exc) or "authentication" in str(exc).lower():
            print(_c("  Your API key may be invalid — reconnect with /login.", "33"))
        return None
    if not streamed["any"] and reply:
        print()
        print(_render_reply(reply))
    print(_c(f"  ({agent.cost.format_total()})", "2"))
    if not background:
        print(_c("  ↵ press enter to continue · or type to redirect", "38;5;141"))
        print()
    first_line = next((ln for ln in (reply or "").strip().splitlines() if ln.strip()), "")
    session.step = (first_line[:60] + "…") if len(first_line) > 60 else (first_line or "ready")
    return reply


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
    except (KeyError, ValueError, OSError) as exc:  # missing / corrupted / unreadable session
        print(_c(f"  could not resume {target}: {exc}", "33"))


def _init_project() -> None:
    from .ops import init_project_notes

    result = init_project_notes(Path.cwd())
    if result.get("created"):
        stack = ", ".join(result.get("detected_stack") or ["no interp stack"])
        print(_c(f"  ✓ wrote {Path(result['path']).name} (detected: {stack})", "32"))
    else:
        print(_c(f"  {Path(result['path']).name} already exists — leaving it as is", "33"))
    for action in result.get("next_actions", [])[:3]:
        print(_c(f"    - {action}", "2"))


def _export(agent, args: list[str]) -> None:
    out = Path(args[0]) if args else Path(f"mechferret-session-{agent.session_id}.md")
    from .agent import _render_messages

    body = _render_messages(agent.messages)
    md = f"# MechFerret session {agent.session_id}\n\nModel: {agent.model}\nCost: {agent.cost.format_total()}\n\n---\n\n{body}\n"
    out.write_text(md, encoding="utf-8")
    print(_c(f"  ✓ exported transcript to {out}", "32"))


def _goal_loop(agent, session, goal: str, max_iters: int = 6) -> None:
    session.goal = goal
    print(_c(f"  🎯 goal: {goal}", PURPLE))
    print(_c(f"  working autonomously (up to {max_iters} steps) — Ctrl-C to stop", "2"))
    prompt = (
        f"Goal: {goal}\nTake the next concrete step toward this goal using your tools "
        "(retrieval, experiments, code). When the goal is fully achieved, reply with "
        "'GOAL ACHIEVED' on the first line, then a one-paragraph summary."
    )
    for i in range(max_iters):
        print(_c(f"  ── step {i + 1}/{max_iters} ──", PURPLE))
        try:
            reply = _chat(agent, session, prompt)
        except KeyboardInterrupt:
            print(_c("  (goal loop stopped)", "2"))
            return
        if reply is None:
            return
        if reply.strip().upper().startswith("GOAL ACHIEVED"):
            print(_c("  ✓ goal achieved", "32"))
            session.step = "goal achieved"
            return
        prompt = (
            f"Continue toward the goal: {goal}. Build on prior steps; avoid repeating work. "
            "Reply with 'GOAL ACHIEVED' on the first line when fully done."
        )
    print(_c("  reached the step cap; goal still open — run /goal again to keep going", "33"))


def _why() -> None:
    text = """Why interpretability — the short version.

Modern AI models are black boxes: we can see their inputs and outputs, but not
the reasoning in between. Mechanistic interpretability reverse-engineers that
middle — finding the specific circuits (attention heads, MLP features) a model
uses. Example: in GPT-2, a small set of "name-mover" heads literally copy the
right name when completing "When John and Mary went to the store, Mary gave a
drink to ___". We can point to the parts and show, by ablating them, that they
cause the behaviour.

Why it matters for safety: you cannot trust, audit, or correct what you cannot
inspect. Interpretability is how we catch deception, hidden goals, and unsafe
shortcuts before deployment — turning "the model usually behaves" into "we know
why it behaves."

Why this domain over law or bio: those benefit from autoresearch too, but they
study the external world. Interpretability studies the very systems doing the
research — so progress here compounds across every other domain and is the most
direct lever on AI safety.

The accessibility argument: understanding the black box is in everyone's
interest, so the tools should be as open and popular as possible. If researchers
are going to optimise for paper count, interpretability is the category where
more shots on goal most benefits the world. MechFerret exists to make that loop
cheap, reproducible, and fast."""
    print(_render_reply(text))


def _strength(x: float) -> str:
    return "strong" if x >= 1.0 else ("medium" if x >= 0.4 else "weak")


def _arch() -> None:
    from .memory import ResearchMemory

    mem = ResearchMemory(".mechferret/memory.sqlite")
    try:
        grouped = mem.experiments_by_hypothesis()
    finally:
        mem.close()
    if not grouped:
        print(_c("  no experiments recorded yet — run /discover (or ask the agent to investigate)", "33"))
        return
    # Separate the broad screen/lens hypotheses (noisy: ~96 ablations) from the
    # targeted single-head hypotheses we actually want a flowchart of.
    def _is_screen(h: str) -> bool:
        return h.startswith("At least one") or "is formed in a specific" in h or h == "screen"

    screen = [e for h, v in grouped.items() if _is_screen(h) for e in v]
    targeted = {h: v for h, v in grouped.items() if not _is_screen(h)}

    print(_c("  ARCHITECTURE OF EVIDENCE — what each experiment proves", PURPLE_B))
    if screen:
        hits = sum(1 for e in screen if e["verdict"] == "good")
        print(_c(f"  screen: {len(screen)} candidate heads ablated → {hits} significant + reproducible", "2"))
    if not targeted:
        print(_c("  no targeted mechanism hypotheses yet (screen found no leads to triangulate)", "33"))
        return
    for hyp, exps in targeted.items():
        print()
        print(_c(f"  ▸ {hyp}", "1"))
        for j, e in enumerate(exps):
            branch = "└─" if j == len(exps) - 1 else "├─"
            mark = _c("✓ good", "32") if e["verdict"] == "good" else _c("✗ weak", "31")
            drift = _c(f"  ⚠ drift×{e['drift_count']}", "33") if e["drift_count"] else ""
            print(f"    {branch} {mark}  {e['probe']:24} effect {e['effect_size']:+.2f} vs ctrl {e['control']:+.2f}  [{_strength(abs(e['effect_size']))}]{drift}")
        good_n = sum(1 for e in exps if e["verdict"] == "good")
        verdict, col = (("CONFIRMED", "32") if good_n >= 2 else ("SUPPORTED", "33") if good_n == 1 else ("UNSUPPORTED", "31"))
        print("    " + _c(f"⇒ {verdict}  ({good_n}/{len(exps)} independent probes agree)", col))
    print()
    print(_c("  memory model: each experiment is keyed by a hash of (model, task, probe, target).", "2"))
    print(_c("  Re-runs upsert in place; a flip in significance or effect-sign counts as drift —", "2"))
    print(_c("  so a mechanism that stops reproducing (new model/code) surfaces instead of rotting.", "2"))


def _paper(agent, session, args: list[str] | None = None) -> None:
    from .paper import print_paper_result, write_paper_from_artifact

    try:
        parsed = _parse_run_command_args(args or [])
        run_json = _resolve_run_json(parsed["run_json"], parsed["runs_root"], parsed["selection"])
        print_paper_result(
            write_paper_from_artifact(
                run_json,
                out_dir=parsed["out"],
                compile_pdf=parsed["compile"],
                compile_timeout=parsed["compile_timeout"],
                provider=parsed["provider"],
                model=parsed["model"],
            )
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(_c(f"  no run artifact to write from: {exc}", "33"))
        print(_c("  run /quickstart --run, /demo, or /discover first, then use /paper", "33"))


def _audit(args: list[str]) -> None:
    from .audit import audit_run_artifact, print_audit

    try:
        parsed = _parse_run_command_args(args)
        run_json = _resolve_run_json(parsed["run_json"], parsed["runs_root"], parsed["selection"])
    except (FileNotFoundError, ValueError) as exc:
        print(_c(f"  audit failed: {exc}", "31"))
        return
    print_audit(audit_run_artifact(run_json))


def _status(args: list[str]) -> None:
    from .ops import print_project_status, project_status

    try:
        parsed = _parse_run_command_args(args)
    except ValueError as exc:
        print(_c(f"  status failed: {exc}", "31"))
        return
    runs_root = parsed["runs_root"]
    if parsed["run_json"]:
        runs_root = parsed["run_json"]
    print_project_status(project_status(runs_root=runs_root, selection=parsed["selection"]))


def _parse_run_command_args(args: list[str]) -> dict[str, str | int | bool | None]:
    parsed: dict[str, str | int | bool | None] = {
        "run_json": None,
        "runs_root": "runs",
        "selection": "latest",
        "out": None,
        "provider": "auto",
        "model": None,
        "compile": False,
        "compile_timeout": 60,
    }
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--compile":
            parsed["compile"] = True
            index += 1
            continue
        if token in {"--runs-root", "--select", "--out", "--provider", "--model", "--compile-timeout"}:
            if index + 1 >= len(args):
                raise ValueError(f"{token} requires a value")
            value = args[index + 1]
            if token == "--runs-root":
                parsed["runs_root"] = value
            elif token == "--select":
                if value not in {"latest", "best", "ready"}:
                    raise ValueError("--select must be latest, best, or ready")
                parsed["selection"] = value
            elif token == "--out":
                parsed["out"] = value
            elif token == "--provider":
                if value not in {"auto", "local", "openai", "anthropic"}:
                    raise ValueError("--provider must be auto, local, openai, or anthropic")
                parsed["provider"] = value
            elif token == "--model":
                parsed["model"] = value
            elif token == "--compile-timeout":
                try:
                    timeout = int(value)
                except ValueError as exc:
                    raise ValueError("--compile-timeout must be an integer") from exc
                if timeout <= 0:
                    raise ValueError("--compile-timeout must be positive")
                parsed["compile_timeout"] = timeout
            index += 2
            continue
        if token.startswith("--"):
            raise ValueError(f"unknown option: {token}")
        if parsed["run_json"] is not None:
            raise ValueError(f"unexpected extra argument: {token}")
        parsed["run_json"] = token
        index += 1
    return parsed


def _resolve_run_json(run_json: str | bool | None, runs_root: str | bool | None, selection: str | bool | None) -> Path:
    from .audit import latest_run_json
    from .ops import select_run_artifact

    if isinstance(run_json, str) and run_json:
        return Path(run_json)
    root = str(runs_root or "runs")
    policy = str(selection or "latest")
    if policy == "latest":
        latest = latest_run_json(root)
        if latest is None:
            raise FileNotFoundError(f"no run artifact found under {root}/**/run.json")
        return latest
    selected = select_run_artifact(runs_root=root, policy=policy)
    if selected.get("path"):
        return Path(selected["path"])
    actions = "; ".join(selected.get("next_actions", []))
    suffix = f": {actions}" if actions else ""
    raise FileNotFoundError(f"no {policy} run artifact found under {root}{suffix}")


def _review_paper(agent, args: list[str]) -> None:
    from .paper import review_paper
    from .spinner import Spinner

    path = None
    runs_root = "runs"
    selection = "latest"
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--runs-root" and index + 1 < len(args):
            runs_root = args[index + 1]
            index += 2
            continue
        if token == "--select" and index + 1 < len(args):
            selection = args[index + 1]
            index += 2
            continue
        if path is None:
            path = token
        index += 1
    print(_c("  reviewing paper…", "2"))
    try:
        with Spinner():
            result = review_paper(path, runs_root=runs_root, selection=selection)
    except Exception as exc:  # noqa: BLE001
        print(_c(f"  review failed: {exc}", "31"))
        return
    if not result["ok"]:
        print(_c("  review not available", "33"))
        for action in result.get("next_actions", []):
            print(_c(f"    - {action}", "33"))
        return
    print()
    print(_render_reply(result["review"]))
    if result.get("review_path"):
        print(_c(f"  wrote {result['review_path']}", "32"))


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


def _show_trace(limit: int = 30) -> None:
    path = _latest_trace_path()
    if not path.exists():
        print(_c("  no trace yet — run /demo, /discover, or chat first", "33"))
        return
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    raindrop = os.getenv("RAINDROP_LOCAL_DEBUGGER") or os.getenv("MECHFERRET_RAINDROP")
    print(_c(f"  trace · {len(lines)} recent spans" + ("  · mirroring to Raindrop ✓" if raindrop else "  · set RAINDROP_LOCAL_DEBUGGER=1 to stream to Raindrop Workshop"), "38;5;141"))
    for ln in lines:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        name = rec.get("name", "")
        attrs = rec.get("attributes", {})
        ms = rec.get("elapsed_ms", 0)
        detail = attrs.get("tool") or attrs.get("probe") or attrs.get("text") or attrs.get("statement") or ""
        dur = f" {ms:.0f}ms" if ms else ""
        print("    " + _c(f"{name:18}", "1;36") + _c(f"{str(detail)[:60]}{dur}", "2"))


def _latest_trace_path() -> Path:
    local = Path(".mechferret/trace.jsonl")
    candidates = [local] if local.exists() else []
    runs = Path("runs")
    if runs.exists():
        candidates.extend(path for path in runs.rglob("trace.jsonl") if path.is_file())
    if not candidates:
        return local
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


# Technical-sounding labels shown in the chain of thought (model still calls the real names).
_TOOL_DISPLAY = {
    "bash": "shell.exec",
    "read_file": "fs.read",
    "write_file": "fs.write",
    "edit_file": "fs.patch",
    "list_dir": "fs.list",
    "glob": "fs.glob",
    "grep": "code.search",
    "web_search": "retrieval.web",
    "web_fetch": "retrieval.fetch",
    "arxiv_search": "retrieval.arxiv",
    "neuronpedia_search": "neuronpedia.query",
    "verify_novelty": "novelty.verify",
    "present_options": "options.present",
    "run_discovery": "interp.discover",
    "list_skills": "skills.list",
    "environment_status": "env.status",
    "audit_run": "eval.audit",
    "write_paper": "paper.write",
    "review_paper": "paper.review",
    "openvla_sae": "sae.openvla",
}


def _display_name(name: str) -> str:
    if name.startswith("mcp__"):
        return "mcp." + name[len("mcp__"):].replace("__", ".")
    return _TOOL_DISPLAY.get(name, name)


def _tool_line(name: str, args: dict) -> str:
    detail = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items() if v) or ""
    return _c(f"  → {_display_name(name)}({detail})", "2")


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in (text or "").splitlines())


def _strip_markdown(text: str) -> str:
    """Flatten any markdown the model emits to clean plain text."""

    import re

    out = []
    for line in (text or "").splitlines():
        s = line
        if re.match(r"^\s*([-=*_]\s*){3,}$", s):  # horizontal rules / table separators
            continue
        s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s)  # headings
        s = s.replace("```", "")
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)  # bold
        s = re.sub(r"__(.+?)__", r"\1", s)
        s = re.sub(r"`([^`]+)`", r"\1", s)  # inline code
        s = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"\1", s)  # italics
        if re.match(r"^\s*\|.*\|\s*$", s):  # table row -> spaced cells (skip separator rows)
            cells = [c.strip() for c in s.strip().strip("|").split("|")]
            if all(re.match(r"^:?-{2,}:?$", c) for c in cells):
                continue
            s = "   ".join(cells)
        s = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", s)  # bullets -> •
        out.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _render_reply(text: str) -> str:
    return _indent(_strip_markdown(text))


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


def _is_short_quickstart(args: list[str]) -> bool:
    return not args or (len(args) == 1 and args[0] in {"all", "demo", "openvla", "ci"})


def _uses_repl_shortcut(bare: str, tokens: list[str]) -> bool:
    if bare in {"open", "demo", "init"}:
        return len(tokens) == 1
    if bare == "status":
        return not any(token in {"--json", "--db", "--notes-root", "--project-root"} for token in tokens[1:])
    if bare == "audit":
        return not any(token in {"--json", "--strict"} for token in tokens[1:])
    if bare == "paper":
        return "--help" not in tokens[1:] and "-h" not in tokens[1:]
    if bare == "review-paper":
        cli_only = {"--json", "--provider", "--model", "--out", "--help", "-h"}
        return not any(token in cli_only for token in tokens[1:])
    if bare == "quickstart":
        return _is_short_quickstart(tokens[1:])
    return False


def _open_report(session: "Session") -> None:
    target = session.last_report
    if not target:
        latest = _latest_report()
        target = str(latest) if latest else None
    if not target:
        print(_c("  no report yet — runs land in ./runs (use /discover or ask the agent)", "33"))
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    os.system(f"{opener} {shlex.quote(target)}")


def _latest_report() -> Path | None:
    root = Path("runs")
    if not root.exists():
        return None
    reports = [p for p in root.rglob("report.html") if p.is_file()]
    return max(reports, key=lambda p: p.stat().st_mtime) if reports else None


def _print_help() -> None:
    from .commands import SECTIONS

    width = max(26, *(len(cmd.name) + 2 for _title, cmds in SECTIONS for cmd in cmds))
    for title, cmds in SECTIONS:
        print(_c(f"  {title}", PURPLE_B))
        for cmd in cmds:
            print("    " + _c(f"{cmd.name:{width}}", "1;36") + _c(cmd.summary, "2"))
