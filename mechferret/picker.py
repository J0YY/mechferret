"""Interactive arrow-key option picker (stdlib only).

Single mode returns the chosen option; multi mode returns a list. Uses termios
raw mode to read ANSI arrow escape sequences when attached to a TTY, and falls
back to a numbered prompt when stdin is piped/redirected or termios is missing
(e.g. Windows). Prompt/menu text goes to stderr so a chosen value can be
captured cleanly from stdout.
"""

from __future__ import annotations

import sys
from typing import Sequence


def select(prompt: str, options: Sequence, multi: bool = False):
    options = list(options)
    if not options:
        raise ValueError("select() requires at least one option")

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            interactive = False
    if not interactive:
        return _select_fallback(prompt, options, multi)
    return _select_tty(prompt, options, multi)


def _select_fallback(prompt, options, multi):
    out = sys.stderr
    if prompt:
        out.write(prompt + "\n")
    for i, opt in enumerate(options, 1):
        out.write(f"  {i}. {opt}\n")
    if multi:
        out.write("Enter numbers (space/comma separated), blank for none: ")
        out.flush()
        line = sys.stdin.readline()
        if line == "":
            raise KeyboardInterrupt
        picks = []
        for tok in line.replace(",", " ").split():
            try:
                idx = int(tok) - 1
            except ValueError:
                continue
            if 0 <= idx < len(options) and idx not in picks:
                picks.append(idx)
        return [options[i] for i in picks]
    out.write("Enter number [1]: ")
    out.flush()
    line = sys.stdin.readline()
    if line == "":
        raise KeyboardInterrupt
    line = line.strip()
    if not line:
        return options[0]
    try:
        idx = int(line) - 1
    except ValueError:
        idx = 0
    return options[max(0, min(idx, len(options) - 1))]


def _select_tty(prompt, options, multi):
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    write = sys.stdout.write
    flush = sys.stdout.flush
    n = len(options)
    cursor = 0
    checked: set[int] = set()

    def render(first):
        if not first:
            write(f"\x1b[{n}A")
        for i, opt in enumerate(options):
            pointer = "❯ " if i == cursor else "  "
            box = ("[x] " if i in checked else "[ ] ") if multi else ""
            line = pointer + box + str(opt)
            if i == cursor:
                line = "\x1b[7m" + line + "\x1b[0m"
            write("\r\x1b[2K" + line + "\n")
        flush()

    def read_key():
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "esc"
        return ch

    write("\x1b[?25l")
    flush()
    if prompt:
        write(prompt + "\n")
        flush()
    try:
        tty.setcbreak(fd)
        render(first=True)
        while True:
            key = read_key()
            if key in ("up", "k"):
                cursor = (cursor - 1) % n
                render(first=False)
            elif key in ("down", "j"):
                cursor = (cursor + 1) % n
                render(first=False)
            elif key == " " and multi:
                checked.symmetric_difference_update({cursor})
                render(first=False)
            elif key in ("\r", "\n"):
                if multi:
                    chosen = sorted(checked) if checked else [cursor]
                    return [options[i] for i in chosen]
                return options[cursor]
            elif key == "\x03" or key in ("esc", "q"):
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        write("\x1b[?25h")
        flush()
