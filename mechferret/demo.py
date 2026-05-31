"""Scripted demo player — replay a believable autoresearch trace.

Reads ``.mechferret/demo.json`` from the working directory and plays it with the
same visual language as a live session (purple banner, ❯ prompts, technical tool
lines, a thinking spinner, plain-text replies). Experiment/mechanism beats are
recorded into the project's memory as they play, so afterwards ``/arch``,
``/memory`` and ``/paper`` render real data that matches the story.

This exists for demos where running the real (hours-long) pipeline live isn't
practical — it shows the trace the agent *would* have produced. Beats are data,
authored per project in ``.mechferret/demo.json``; nothing here is paper-specific.
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

    reset=True clears the experiment ledger + mechanisms first, so replaying the
    same demo is deterministic (drift counts don't accumulate across replays).
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
            print(_c("\n  demo complete — try /arch, /memory, /paper, /review-paper", "38;5;141"))
    finally:
        if mem is not None:
            mem.close()


def _model_task(demo: dict, beat: dict) -> tuple[str, str]:
    return beat.get("model", demo.get("model", "model")), beat.get("task", demo.get("task", "task"))


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


def _sleep(seconds: float, speed: float) -> None:
    if sys.stdout.isatty() and seconds > 0:
        time.sleep(seconds / max(speed, 0.1))


def _render_beat(beat, speed, _c, _render_reply, _tool_line) -> None:
    kind = beat.get("type")
    P = "38;5;141"
    if kind == "key":  # presenter-paced break (Enter to advance on a TTY)
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                input(_c("  [enter] ", "2"))
            except (EOFError, KeyboardInterrupt):
                pass
        return
    text = beat.get("text", "")
    if kind == "note":
        print(_c(f"  {text}", "2"))
    elif kind == "user":
        print(_c("─" * 78, "2"))
        print(_c("❯ ", "1;36") + text)
        _sleep(0.5, speed)
    elif kind == "think":
        from .spinner import Spinner

        with Spinner():
            _sleep(float(beat.get("seconds", 1.2)), speed)
    elif kind == "tool":
        print(_tool_line(beat.get("name", "tool"), beat.get("args", {})))
        _sleep(float(beat.get("seconds", 0.4)), speed)
        if beat.get("result"):
            print(_c(f"     {beat['result']}", "2"))
    elif kind == "experiment":
        label = beat.get("probe", "experiment")
        mark = _c("✓", "32") if beat.get("verdict") == "good" else _c("✗", "31")
        print(_c(f"  ⚗  {label}", "2") + f"  {mark} effect {float(beat.get('effect', 0)):+.3f}")
        _sleep(0.3, speed)
    elif kind == "pivot":
        print(_c(f"  ↪ pivot: {text}", "33"))
        _sleep(0.4, speed)
    elif kind == "modal":
        print(_c(f"  ⛁ modal: {text}", "38;5;75"))
        _sleep(float(beat.get("seconds", 0.6)), speed)
    elif kind == "assistant":
        print()
        print(_render_reply(text))
        print()
        _sleep(0.6, speed)
    elif kind == "mechanism":
        print(_c(f"  ★ confirmed: {beat.get('statement', '')}", "32"))
        _sleep(0.3, speed)
    elif kind == "pause":
        _sleep(float(beat.get("seconds", 1.0)), speed)


def seed_only(path: str | Path | None = None) -> dict:
    """Populate memory from a demo without rendering (so /arch works immediately)."""

    demo = load_demo(path)
    play(demo, render=False, record=True)
    return {"beats": len(demo.get("beats", []))}
