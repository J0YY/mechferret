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


def select_rich(prompt: str, options: list[dict]):
    """Expandable option picker for option dictionaries.

    Returns the chosen title, or "none" if cancelled. Up/Down move, Right/Tab/d
    expands detail for the highlighted option, Enter selects, Esc/q skips.
    """

    options = list(options)
    if not options:
        return "none"
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            interactive = False
    if not interactive:
        return _select_rich_fallback(prompt, options)
    return _select_rich_tty(prompt, options)


def _wrap(text: str, width: int = 72) -> list[str]:
    import textwrap

    lines: list[str] = []
    for para in (text or "").splitlines() or [""]:
        lines.extend(textwrap.wrap(para, width) or [""])
    return lines


def _threat_bits(value) -> list[str]:
    if not isinstance(value, list):
        return []
    bits: list[str] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        threat = str(row.get("threat", "")).replace("_", " ").strip()
        risk = str(row.get("risk", "")).replace("_", " ").strip()
        searched = "searched" if row.get("searched") is True else "unsearched"
        if threat and risk:
            bits.append(f"{threat}:{risk} ({searched})")
    return bits


def _disqualifier_bits(value) -> list[str]:
    if not isinstance(value, list):
        return []
    bits: list[str] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        test = str(row.get("test", "")).replace("_", " ").strip()
        risk = str(row.get("risk", "")).replace("_", " ").strip()
        status = "pass" if row.get("passed") is True else "needs delta"
        if test and risk:
            bits.append(f"{test}:{status}/{risk}")
    return bits


def _search_audit_bits(value) -> list[str]:
    if not isinstance(value, dict):
        return []
    bits: list[str] = []
    for key, label in (
        ("pass_count", "passes"),
        ("failed_passes", "failed"),
        ("empty_search_passes", "empty"),
        ("duplicate_only_search_passes", "duplicates"),
    ):
        count = value.get(key)
        if isinstance(count, int) and count:
            bits.append(f"{label}:{count}")
    failed_focuses = value.get("failed_focuses")
    if isinstance(failed_focuses, list) and failed_focuses:
        bits.append("failed focuses:" + str(len(failed_focuses)))
    empty_focuses = value.get("empty_focuses")
    if isinstance(empty_focuses, list) and empty_focuses:
        bits.append("empty focuses:" + str(len(empty_focuses)))
    return bits


def _select_rich_fallback(prompt, options):
    out = sys.stderr
    if prompt:
        out.write(prompt + "\n")
    for i, o in enumerate(options, 1):
        out.write(f"  {i}. {o.get('title', '')} — {o.get('summary', '')}\n")
        if o.get("novelty_risk"):
            out.write(f"      novelty risk: {o['novelty_risk']}\n")
        if o.get("novelty_verdict"):
            out.write(f"      novelty verdict: {o['novelty_verdict']}\n")
        readiness = o.get("claim_readiness") or {}
        if isinstance(readiness, dict) and readiness.get("status"):
            out.write(f"      claim readiness: {readiness['status']}\n")
        pressure = o.get("recent_pressure") or {}
        if isinstance(pressure, dict) and pressure.get("status"):
            out.write(f"      recent pressure: {pressure['status']} ({pressure.get('recent_window', '')})\n")
        threats = _threat_bits(o.get("novelty_threat_model"))
        if threats:
            out.write("      novelty threats: " + ", ".join(threats[:3]) + "\n")
        disqualifiers = _disqualifier_bits(o.get("disqualifying_overlap_tests"))
        if disqualifiers:
            out.write("      disqualifiers: " + ", ".join(disqualifiers[:3]) + "\n")
        audit = _search_audit_bits(o.get("search_audit"))
        if audit:
            out.write("      search audit: " + ", ".join(audit[:5]) + "\n")
        if o.get("novelty"):
            out.write(f"      novelty: {o['novelty']}\n")
        closest = o.get("closest_prior_art") or []
        if closest:
            out.write("      closest prior: " + "; ".join(str(c) for c in closest[:2]) + "\n")
    out.write("Enter number to select (blank to skip): ")
    out.flush()
    line = sys.stdin.readline()
    if not line.strip():
        return "none"
    try:
        idx = int(line.strip()) - 1
    except ValueError:
        return "none"
    return options[idx]["title"] if 0 <= idx < len(options) else "none"


def _select_rich_tty(prompt, options):
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    write, flush = sys.stdout.write, sys.stdout.flush
    cursor = 0
    expanded = False
    prev_lines = 0

    def build() -> list[str]:
        rows: list[str] = []
        for i, o in enumerate(options):
            mark = "❯ " if i == cursor else "  "
            head = f"{mark}{o.get('title', '')}  —  {o.get('summary', '')}"
            rows.append("\x1b[7m" + head + "\x1b[0m" if i == cursor else head)
            if i == cursor and expanded:
                for dl in _wrap(o.get("detail") or o.get("summary", "")):
                    rows.append("      " + dl)
                if o.get("novelty"):
                    rows.append("      novelty: " + str(o["novelty"]))
                if o.get("novelty_risk"):
                    rows.append("      novelty risk: " + str(o["novelty_risk"]))
                if o.get("novelty_verdict"):
                    rows.append("      novelty verdict: " + str(o["novelty_verdict"]))
                readiness = o.get("claim_readiness") or {}
                if isinstance(readiness, dict) and readiness.get("status"):
                    rows.append("      claim readiness: " + str(readiness["status"]))
                    missing = readiness.get("missing_checks") or []
                    if missing:
                        rows.append("      missing checks: " + ", ".join(str(c) for c in missing[:5]))
                pressure = o.get("recent_pressure") or {}
                if isinstance(pressure, dict) and pressure.get("status"):
                    rows.append(
                        "      recent pressure: "
                        + str(pressure["status"])
                        + (" (" + str(pressure.get("recent_window", "")) + ")" if pressure.get("recent_window") else "")
                    )
                threats = _threat_bits(o.get("novelty_threat_model"))
                if threats:
                    rows.append("      novelty threats: " + ", ".join(threats[:5]))
                disqualifiers = _disqualifier_bits(o.get("disqualifying_overlap_tests"))
                if disqualifiers:
                    rows.append("      disqualifiers: " + ", ".join(disqualifiers[:5]))
                audit = _search_audit_bits(o.get("search_audit"))
                if audit:
                    rows.append("      search audit: " + ", ".join(audit[:5]))
                matrix = o.get("comparison_matrix") or []
                if isinstance(matrix, list) and matrix:
                    axis_bits = []
                    for row in matrix[:6]:
                        if not isinstance(row, dict):
                            continue
                        axis = str(row.get("axis", "")).replace("_", " ")
                        mark = "ok" if row.get("covered") else "gap"
                        if axis:
                            axis_bits.append(f"{axis}:{mark}")
                    if axis_bits:
                        rows.append("      comparison axes: " + ", ".join(axis_bits))
                closest = o.get("closest_prior_art") or []
                if closest:
                    rows.append("      closest prior: " + "; ".join(str(c) for c in closest[:3]))
                if o.get("required_delta"):
                    rows.append("      required delta: " + str(o["required_delta"]))
                cites = o.get("citations") or []
                if cites:
                    rows.append("      cite: " + "; ".join(str(c) for c in cites[:4]))
        rows.append("\x1b[2m  ↑/↓ move · → expand · enter select · esc skip\x1b[0m")
        return rows

    def render(first: bool):
        nonlocal prev_lines
        if not first:
            write(f"\x1b[{prev_lines}A")
        rows = build()
        for r in rows:
            write("\r\x1b[2K" + r + "\n")
        # clear any leftover lines from a taller previous render
        for _ in range(max(0, prev_lines - len(rows))):
            write("\r\x1b[2K\n")
        if prev_lines > len(rows):
            write(f"\x1b[{prev_lines - len(rows)}A")
        prev_lines = len(rows)
        flush()

    def read_key():
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc")
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
                cursor = (cursor - 1) % len(options)
                expanded = False
                render(False)
            elif key in ("down", "j"):
                cursor = (cursor + 1) % len(options)
                expanded = False
                render(False)
            elif key in ("right", "d", " ", "\t", "left"):
                expanded = not expanded
                render(False)
            elif key in ("\r", "\n"):
                return options[cursor].get("title", "none")
            elif key in ("esc", "q", "\x03"):
                return "none"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        write("\x1b[?25h")
        flush()


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
