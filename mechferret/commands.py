"""In-process REPL command registry, grouped into sections for a readable /help.

The REPL handles these directly (no argparse round-trip); anything not listed
here and not a known pipeline subcommand falls back to the CLI parser.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    summary: str


# Ordered, grouped command sections (drives /help).
SECTIONS: list[tuple[str, list[Command]]] = [
    ("Chat", [
        Command("<your prompt>", "talk to the model; it runs tools/experiments"),
        Command("/goal <text>", "set an objective and loop autonomously until reached"),
        Command("/plan", "toggle plan mode (approve write/exec/GPU tools)"),
        Command("/compact", "summarise older turns to free context"),
    ]),
    ("Research", [
        Command("/discover ...", "run discovery directly (--skill --task --model)"),
        Command("/skills [name]", "list interpretability playbooks, or show one"),
        Command("/arch", "show the evidence flowchart: what each experiment proves"),
        Command("/paper", "generate main.tex + main.pdf for submission"),
        Command("/review-paper", "spin up a reviewer instance to score the paper 1-10"),
    ]),
    ("Compute", [
        Command("/modal <action>", "status | setup | run | deploy  (GPU on Modal)"),
        Command("/cluster <action>", "status | setup | run  (your own SLURM cluster)"),
        Command("/mcp <action>", "list | add | tools  (MCP servers)"),
    ]),
    ("Session", [
        Command("/login", "connect or change your model API key"),
        Command("/model <name>", "set the conversation model"),
        Command("/cost", "show session token + USD usage"),
        Command("/memory", "list confirmed mechanisms in memory"),
        Command("/resume [id]", "resume a saved session"),
        Command("/export [path]", "export this session to Markdown"),
        Command("/init", "scaffold a MECHFERRET.md for this project"),
        Command("/doctor /registry", "environment and capabilities"),
        Command("/open", "open the last run's HTML dossier"),
    ]),
    ("Narration", [
        Command("/why", "why interpretability, why now, why this domain"),
        Command("/clear", "redraw the welcome screen"),
        Command("ctrl-c / /exit", "quit"),
    ]),
]

# Bare command words the REPL handles in-process (not shelled to the CLI parser).
REPL_HANDLED = {
    "exit", "quit", "q", "help", "clear", "open", "login", "connect", "model",
    "goal", "plan", "cost", "compact", "resume", "memory", "export", "init",
    "review", "mcp", "why", "arch", "paper", "review-paper",
}
