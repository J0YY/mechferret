"""Animated thinking spinner — ferret verbs + a moving ombre colour wave.

Runs on a background thread while the (blocking) model call is in flight, so the
prompt feels alive like Claude Code's spinner. Falls back to a plain line when
stdout isn't a TTY or NO_COLOR is set.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time

# Ferret-flavoured present participles, à la Claude Code's spinner verbs.
VERBS = [
    "Sniffing", "Burrowing", "Digging", "Farting", "Scurrying", "Tunnelling",
    "Foraging", "Rummaging", "Noodling", "Wiggling", "Scampering", "Snooping",
    "Nuzzling", "Pondering", "Weaseling", "Ferreting", "Chittering", "Zoomies-ing",
    "Squirreling", "Pouncing", "Snuffling", "Wrangling", "Untangling", "Marauding",
]

# Blue → purple → pink ombre, as xterm-256 colour codes.
PALETTE = [63, 99, 105, 141, 177, 213, 212, 176, 170, 135]
BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _ombre(text: str, phase: int) -> str:
    out = []
    for i, ch in enumerate(text):
        code = PALETTE[(i + phase) % len(PALETTE)]
        out.append(f"\x1b[38;5;{code}m{ch}")
    return "".join(out) + "\x1b[0m"


class Spinner:
    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty() and not os.getenv("NO_COLOR")
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._verb = random.choice(VERBS)

    def pause(self):
        """Context manager: stop animating + clear the line so prompts are clean."""

        spinner = self

        class _Pause:
            def __enter__(self_inner):
                spinner._paused.set()
                with spinner._lock:
                    sys.stdout.write("\r\x1b[2K\x1b[?25h")
                    sys.stdout.flush()

            def __exit__(self_inner, *exc):
                if spinner.enabled:
                    sys.stdout.write("\x1b[?25l")
                    sys.stdout.flush()
                spinner._paused.clear()

        return _Pause()

    def _run(self) -> None:
        phase = frames = 0
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue
                if frames % 18 == 0:
                    self._verb = random.choice(VERBS)
                line = f"{BRAILLE[frames % len(BRAILLE)]} {self._verb}…"
                with self._lock:
                    sys.stdout.write("\r\x1b[2K  " + _ombre(line, phase))
                    sys.stdout.flush()
                phase += 1
                frames += 1
                time.sleep(0.08)
        finally:
            with self._lock:
                sys.stdout.write("\r\x1b[2K\x1b[?25h")
                sys.stdout.flush()

    def log(self, text: str) -> None:
        """Print a line above the live spinner without disrupting it."""

        if not self.enabled:
            print(text)
            return
        with self._lock:
            sys.stdout.write("\r\x1b[2K" + text + "\n")
            sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            sys.stdout.write("  thinking…\n")
            sys.stdout.flush()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
