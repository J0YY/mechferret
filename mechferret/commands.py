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
        Command("/btw <text>", "run a compact side prompt while another reply is running"),
        Command("/queue", "show the active and queued prompts"),
        Command("/queue restore", "restore saved queued/running prompts from the last session"),
        Command("/queue wait [seconds]", "block until active queued and side work finishes"),
        Command("/cancel <id|all>", "remove queued prompts that have not started yet"),
        Command("/goal <text>", "set an objective and loop autonomously until reached"),
        Command("/plan", "toggle plan mode (approve write/exec/GPU tools)"),
        Command("/compact", "summarise older turns to free context"),
    ]),
    ("Research", [
        Command("/status", "show setup, selected run, audit/verify state, artifacts, and next actions"),
        Command("/next", "print the next recommended project actions"),
        Command("/runs", "list recent run artifacts with audit and artifact status"),
        Command("/quickstart", "show the recommended demo/OpenVLA/CI command path"),
        Command("/selftest", "run offline readiness checks and optionally verify demo artifacts"),
        Command("/support", "write a shareable self-test report for issues or PRs"),
        Command("/demo", "run the local prompt-to-dossier demo"),
        Command("run_research tool", "agent tool for source-grounded prompt-to-dossier research"),
        Command("/discover ...", "run discovery directly (--skill --task --model)"),
        Command("/sae openvla ...", "OpenVLA SAE init, status, plan, commands, manifest checks"),
        Command("/skills [name]", "list interpretability playbooks, or show one"),
        Command("/arch", "show the evidence flowchart: what each experiment proves"),
        Command("/paper [run.json]", "generate main.tex from a saved dossier"),
        Command("/audit [run.json]", "offline readiness gates + next actions"),
        Command("/verify [run.json]", "manifest hashes + artifact integrity"),
        Command("/review-paper", "review the selected run-bound paper with a configured model"),
        Command("/bundle [run.json]", "package the run dossier into a shareable zip"),
        Command("/verify-bundle [zip]", "portable bundle manifest + archive hash checks"),
    ]),
    ("Compute", [
        Command("/modal <action>", "status | setup | run | deploy  (GPU on Modal)"),
        Command("/cluster <action>", "status | setup | run  (your own SLURM cluster)"),
        Command("/mcp <action>", "list | add | tools  (MCP servers)"),
    ]),
    ("Session", [
        Command("/version", "show installed package and runtime information"),
        Command("/commands", "list the installed CLI command surface"),
        Command("/commands --workflow first_run", "show a runnable workflow recipe"),
        Command("/completion <shell>", "print shell completion for bash, zsh, or fish"),
        Command("/login", "connect or change your model API key"),
        Command("/api", "show or change provider configuration"),
        Command("/model <name>", "set the conversation model"),
        Command("/cost", "show session token + USD usage"),
        Command("/memory", "list confirmed mechanisms in memory"),
        Command("/tool-results", "list or clean saved large tool outputs"),
        Command("/trace", "show the run trace (mirrors to Raindrop Workshop)"),
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

# Bare command words the REPL may handle in-process before falling back to the
# CLI parser. Keep this list aligned with the explicit branches in repl.py.
REPL_HANDLED = {
    "exit", "quit", "q", "help", "clear", "open", "login", "connect", "model",
    "btw", "queue", "cancel",
    "goal", "plan", "cost", "compact", "resume", "memory", "tool-results",
    "export", "init", "review", "mcp", "why", "arch", "paper", "audit",
    "verify", "review-paper", "bundle", "verify-bundle", "demo", "trace",
    "sae", "quickstart",
    "status",
    "runs",
}


# Bare command words that should be routed to argparse when they are not handled
# by a richer in-process shortcut. This prevents commands like
# `quickstart --run` or `open bundle` from being treated as chat prompts.
CLI_FALLBACK = {
    "version", "about", "commands", "completion", "api",
    "run", "demo", "discover", "doctor", "registry", "memory", "tool-results",
    "cost", "resume", "inspect", "audit", "skills", "modal", "cluster", "sae",
    "status", "next", "runs", "open", "init", "quickstart", "selftest", "support", "diagnostics", "paper", "review-paper",
    "verify", "bundle", "verify-bundle", "goal", "loop", "list-runs",
}


COMMAND_WORDS = REPL_HANDLED | CLI_FALLBACK
