"""In-process REPL command registry.

The REPL handles these commands directly (no argparse round-trip); anything not
listed here and not a known pipeline subcommand falls back to the CLI parser.
Drives ``/help`` so the command list stays in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    summary: str


# REPL-native commands (handled in repl.run_repl loop).
REPL_COMMANDS = [
    Command("<your prompt>", "talk to the model; it runs tools/experiments when you ask"),
    Command("/login", "connect or change your model API key"),
    Command("/model <name>", "set the conversation model"),
    Command("/goal <text>", "set the research objective shown in the sticky line"),
    Command("/plan", "toggle plan mode (approve write/exec/GPU tools)"),
    Command("/cost", "show session token + USD usage"),
    Command("/compact", "summarise older turns to free context"),
    Command("/resume [id]", "resume a saved session (picker if no id)"),
    Command("/memory", "list confirmed mechanisms in memory"),
    Command("/export [path]", "export this session to Markdown"),
    Command("/init", "scaffold a MECHFERRET.md for this project"),
    Command("/review", "review the current git diff for interp-research issues"),
    Command("/mcp <action>", "list | add | tools  (Model Context Protocol servers)"),
    Command("/discover ...", "run discovery directly (--skill --task --model)"),
    Command("/skills [name]", "list playbooks, or show one"),
    Command("/modal <action>", "status | setup | run | deploy  (GPU on Modal)"),
    Command("/cluster <action>", "status | setup | run  (your own SLURM cluster)"),
    Command("/doctor /registry", "environment and capabilities"),
    Command("/open", "open the last run's HTML dossier"),
    Command("/clear", "redraw the welcome screen"),
    Command("ctrl-c / /exit", "quit"),
]

# Bare command words the REPL handles in-process (not shelled to the CLI parser).
REPL_HANDLED = {
    "exit", "quit", "q", "help", "clear", "open", "login", "connect", "model",
    "goal", "plan", "cost", "compact", "resume", "memory", "export", "init", "review", "mcp",
}
