"""Tool permission decisions (ported from Claude Code's tool-scoped permissions).

Every tool call is gated before dispatch. Read-only tools always pass. In
``auto`` mode non-read-only tools run (the user is driving), except commands
that match a dangerous pattern, which always prompt. In ``plan`` mode every
non-read-only tool prompts for approval. Headless callers (no confirm callback)
auto-deny anything that would prompt.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

# Bash commands that always require confirmation, even in auto mode.
DANGEROUS_BASH = [
    "*rm -rf*", "*rm -fr*", "*sudo *", "*mkfs*", "*dd if=*", "* > /dev/sd*",
    "*:(){*", "*git push*", "*git reset --hard*", "*chmod -R*", "*curl*| sh*",
    "*curl*| bash*", "*wget*| sh*", "*> /dev/null 2>&1 &*",
]

MODES = ("auto", "plan")


@dataclass(slots=True)
class PermissionDecision:
    behavior: str  # allow | ask | deny
    reason: str = ""


def decide(
    tool_name: str,
    args: dict,
    *,
    read_only: bool,
    permission_class: str,
    mode: str,
) -> PermissionDecision:
    if read_only:
        return PermissionDecision("allow")
    if tool_name == "bash":
        cmd = str(args.get("command", ""))
        if any(fnmatch.fnmatch(cmd, pat) for pat in DANGEROUS_BASH):
            return PermissionDecision("ask", "looks destructive / irreversible")
    if mode == "plan":
        return PermissionDecision("ask", f"{permission_class} action in plan mode")
    return PermissionDecision("allow")
