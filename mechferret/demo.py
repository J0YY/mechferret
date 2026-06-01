"""Presenter trace player for authored walkthroughs.

Reads ``.mechferret/demo.json`` from the working directory and plays it with the
same visual language as a live session (purple banner, ❯ prompts, technical tool
lines, a thinking spinner, plain-text replies). Experiment/mechanism beats are
recorded into the project's memory as they play, so afterwards ``/arch``,
``/memory`` and ``/paper`` render real data that matches the story.

This is not wired into the product-facing ``/demo`` command. It is kept as a
presenter utility for projects that explicitly ship a ``.mechferret/demo.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEMO_PATH = Path(".mechferret/demo.json")
DB_PATH = ".mechferret/memory.sqlite"


def has_demo() -> bool:
    return DEMO_PATH.exists()


def load_demo(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEMO_PATH
    if not p.exists():
        raise FileNotFoundError(f"no demo at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def play(demo: dict, render: bool = True, record: bool = True, speed: float = 1.0, reset: bool = False) -> None:
    """Play (and/or seed) a demo. render=False + record=True just seeds memory.

    reset=True clears the experiment ledger + mechanisms first, so repeated
    presenter walkthroughs do not accumulate drift counts.
    """

    from .repl import _c, _render_reply, _tool_line

    mem = None
    if record:
        from .memory import ResearchMemory

        mem = ResearchMemory(DB_PATH)
        if reset:
            mem.clear_experiments_and_mechanisms()
    try:
        tracer = None
        if render:
            from .tracing import TraceRecorder

            tracer = TraceRecorder(f"demo_{demo.get('title', 'demo')[:20]}", ".mechferret")
        beats = demo.get("beats", [])
        if render and demo.get("goal"):
            print(_c(f"  🎯 goal: {demo['goal']}", "38;5;141"))
            print()
        for beat in beats:
            kind = beat.get("type")
            if record and kind == "experiment":
                _seed_experiment(mem, demo, beat)
            if record and kind == "mechanism":
                _seed_mechanism(mem, demo, beat)
            if tracer is not None:
                _trace_beat(tracer, beat)
            if render:
                _render_beat(beat, speed, _c, _render_reply, _tool_line)
        if render:
            print(_c("\n  demo complete — try /arch, /audit, /paper, /review-paper, or /quickstart", "38;5;141"))
    finally:
        if mem is not None:
            mem.close()


def _model_task(demo: dict, beat: dict) -> tuple[str, str]:
    model = _demo_field(demo, beat, "model")
    task = _demo_field(demo, beat, "task")
    if not model or not task:
        missing = "model" if not model else "task"
        raise ValueError(f"demo beat requires explicit {missing}; set it on the beat or demo root")
    return model, task


def _demo_field(demo: dict, beat: dict, name: str) -> str:
    value = beat.get(name, demo.get(name, ""))
    return value.strip() if isinstance(value, str) else ""


def _seed_experiment(mem, demo, beat) -> None:
    model, task = _model_task(demo, beat)
    mem.record_experiment(
        model, task,
        beat.get("probe", "experiment"),
        beat.get("target", {"label": beat.get("probe", "")}),
        beat.get("hypothesis", "screen"),
        float(beat.get("effect", 0.0)),
        float(beat.get("control", 0.0)),
        bool(beat.get("significant", beat.get("verdict") == "good")),
        bool(beat.get("reproduced", beat.get("verdict") == "good")),
        code_version=demo.get("code_version", "demo"),
    )


def _seed_mechanism(mem, demo, beat) -> None:
    model, _ = _model_task(demo, beat)
    mem.record_mechanisms(model, [{
        "statement": beat.get("statement", ""),
        "effect_size": float(beat.get("effect", 0.0)),
        "reproducibility": float(beat.get("reproducibility", 1.0)),
        "novelty": float(beat.get("novelty", 0.5)),
    }])


def _trace_beat(tracer, beat) -> None:
    """Emit a Raindrop-visible span/event for a beat."""

    kind = beat.get("type")
    if kind == "tool":
        tracer.event("tool", tool=beat.get("name", "tool"), args=beat.get("args", {}))
    elif kind == "experiment":
        tracer.event("experiment", probe=beat.get("probe", ""), effect=beat.get("effect", 0),
                     verdict=beat.get("verdict", ""))
    elif kind == "pivot":
        tracer.event("pivot", text=beat.get("text", "")[:160])
    elif kind == "modal":
        tracer.event("modal", text=beat.get("text", "")[:160])
    elif kind == "mechanism":
        tracer.event("mechanism_confirmed", statement=beat.get("statement", "")[:160])
    elif kind == "assistant":
        tracer.event("assistant", text=beat.get("text", "")[:160])
    elif kind == "user":
        tracer.event("user_prompt", text=beat.get("text", "")[:160])
    elif kind == "phase":
        tracer.event("phase", text=beat.get("text", "")[:120])
    elif kind in {"insight", "deadend", "metric"}:
        tracer.event(kind, text=beat.get("text", "")[:160])


def _sleep(seconds: float, speed: float) -> None:
    if sys.stdout.isatty() and seconds > 0:
        time.sleep(seconds / max(speed, 0.1))


# colour palette (xterm-256)
PURPLE = "38;5;141"
CYAN = "38;5;44"
BLUE = "38;5;75"
GREEN = "32"
RED = "38;5;203"
YELLOW = "38;5;221"
GOLD = "1;38;5;220"
PINK = "38;5;213"
GREY = "2"


def _tool_color(name: str) -> str:
    if name.startswith(("retrieval", "web")):
        return CYAN
    if name.startswith(("interp", "scoring", "eval", "attribut")):
        return PURPLE
    if name.startswith(("neuronpedia", "sae")):
        return PINK
    if name.startswith("novelty"):
        return GOLD
    return BLUE


def _render_beat(beat, speed, _c, _render_reply, _tool_line) -> None:
    kind = beat.get("type")
    if kind == "key":  # presenter-paced break (Enter to advance on a TTY)
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                input(_c("  [enter ↵] ", GOLD))
            except (EOFError, KeyboardInterrupt):
                pass
        return
    text = beat.get("text", "")
    if kind == "phase":
        bar = "━" * 68
        print()
        print(_c(f"  ┏{bar}", PURPLE))
        print(_c(f"  ┃  {text}", "1;" + PURPLE))
        print(_c(f"  ┗{bar}", PURPLE))
        _sleep(0.4, speed)
    elif kind == "note":
        print(_c(f"  {text}", GREY))
    elif kind == "user":
        print(_c("─" * 78, GREY))
        print(_c("❯ ", "1;36") + _c(text, "1"))
        _sleep(0.5, speed)
    elif kind == "think":
        from .spinner import Spinner

        with Spinner():
            _sleep(float(beat.get("seconds", 1.2)), speed)
    elif kind == "tool":
        name = beat.get("name", "tool")
        args = ", ".join(f"{k}={str(v)[:48]}" for k, v in beat.get("args", {}).items())
        print("  " + _c("→ ", GREY) + _c(name, _tool_color(name)) + _c(f"({args})", GREY))
        _sleep(float(beat.get("seconds", 0.35)), speed)
        if beat.get("result"):
            print(_c(f"     ↳ {beat['result']}", GREY))
    elif kind == "code":
        print("  " + _c("$ ", "2;36") + _c(text, "38;5;108"))
        _sleep(0.3, speed)
    elif kind == "dataset":
        print("  " + _c("▤ ", BLUE) + _c(text, GREY))
        _sleep(0.25, speed)
    elif kind == "experiment":
        label = beat.get("probe", "experiment")
        good = beat.get("verdict") == "good"
        mark = _c("✓ good", GREEN) if good else _c("✗ weak", RED)
        eff = _c(f"{float(beat.get('effect', 0)):+.3f}", GREEN if good else RED)
        print("  " + _c("⚗ ", PURPLE) + _c(label, "1" if good else GREY) + f"  {mark}  effect {eff}")
        _sleep(0.3, speed)
    elif kind == "metric":
        print("  " + _c("📊 " + text, GOLD))
        _sleep(0.3, speed)
    elif kind == "insight":
        print("  " + _c("✸ " + text, "1;" + CYAN))
        _sleep(0.5, speed)
    elif kind == "pivot":
        print("  " + _c("↪ pivot: ", "1;" + YELLOW) + _c(text, YELLOW))
        _sleep(0.5, speed)
    elif kind == "deadend":
        print("  " + _c("✗ dead end: ", "1;" + RED) + _c(text, RED))
        _sleep(0.5, speed)
    elif kind == "modal":
        print("  " + _c("⛁ modal: ", "1;" + BLUE) + _c(text, BLUE))
        _sleep(float(beat.get("seconds", 0.6)), speed)
    elif kind == "assistant":
        print()
        print(_render_reply(text))
        print()
        _sleep(0.6, speed)
    elif kind == "mechanism":
        print("  " + _c("★ confirmed: ", "1;" + GREEN) + _c(beat.get("statement", ""), GREEN))
        _sleep(0.3, speed)
    elif kind == "pause":
        _sleep(float(beat.get("seconds", 1.0)), speed)


def seed_only(path: str | Path | None = None) -> dict:
    """Populate memory from a demo without rendering (so /arch works immediately)."""

    demo = load_demo(path)
    play(demo, render=False, record=True)
    return {"beats": len(demo.get("beats", []))}
