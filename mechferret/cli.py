from __future__ import annotations

import argparse
import difflib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .audit import audit_run_artifact, latest_run_json, print_audit
from .config import PROVIDERS, configure_provider, default_config_path, load_config, prompt_api_key, save_config
from .controller import MechFerret
from .costs import estimate_run_cost
from .discovery import DiscoveryController
from .goal_loop import GoalLoop
from .hooks import Budget
from .ops import bundle_run_artifacts, init_project_notes, list_run_artifacts, memory_clear, memory_recent, memory_summary, open_artifact, print_artifact_result, print_bundle_result, print_doctor, print_project_init, print_project_status, print_quickstart, print_quickstart_run, print_run_list, print_selftest, print_verify_bundle_result, print_verify_result, project_status, quickstart, resolve_artifact, run_quickstart, select_run_artifact, selftest, summarize_run_artifact, verify_bundle_artifacts, verify_run_artifacts
from .openvla_sae import command_lines as openvla_sae_commands
from .openvla_sae import create_manifest as create_openvla_sae_manifest
from .openvla_sae import evaluate_artifacts as evaluate_openvla_sae_artifacts
from .openvla_sae import feature_report as openvla_sae_feature_report
from .openvla_sae import init_project as init_openvla_sae_project
from .openvla_sae import print_dossier_result as print_openvla_sae_dossier
from .openvla_sae import print_eval_result as print_openvla_sae_eval
from .openvla_sae import print_feature_result as print_openvla_sae_features
from .openvla_sae import print_init_result as print_openvla_sae_init
from .openvla_sae import print_manifest_result as print_openvla_sae_manifest
from .openvla_sae import print_plan_result as print_openvla_sae_plan
from .openvla_sae import print_smoke_result as print_openvla_sae_smoke
from .openvla_sae import print_status as print_openvla_sae_status
from .openvla_sae import smoke_test as openvla_sae_smoke
from .openvla_sae import status as openvla_sae_status
from .openvla_sae import validate_manifest as validate_openvla_sae_manifest
from .openvla_sae import write_dossier as write_openvla_sae_dossier
from .openvla_sae import write_plan as write_openvla_sae_plan
from .paper import TECTONIC_TIMEOUT_SECONDS, print_paper_result, print_review_result, review_paper, write_paper_from_artifact
from .registry import all_items, items_by_kind
from .skills import list_skills, load_skill
from .sources import example_corpus_path
from .tools import tool_clean_tool_results, tool_list_tool_results

DEMO_QUESTION = (
    "What should a team build to win an autoresearch systems hackathon, "
    "and what reliability risks must the implementation address?"
)

COMMAND_GROUPS = [
    ("Start", {"quickstart", "selftest", "support", "init", "status", "doctor", "commands", "completion", "version"}),
    ("Research", {"run", "demo", "goal", "discover", "sae"}),
    ("Artifacts", {"runs", "open", "resume", "inspect", "audit", "verify", "paper", "review-paper", "bundle", "verify-bundle", "cost"}),
    ("Config", {"login", "api", "memory", "tool-results", "registry", "skills", "repl"}),
    ("Compute", {"modal", "cluster"}),
]
COMMAND_GROUP_CHOICES = [title.lower() for title, _names in COMMAND_GROUPS]

COMMAND_WORKFLOWS = [
    {
        "name": "first_run",
        "title": "First Run",
        "description": "Create project notes, build a local dossier, inspect status, and write a support report.",
        "commands": [
            "mechferret init",
            "mechferret quickstart --run",
            "mechferret status",
            "mechferret support",
        ],
    },
    {
        "name": "publish_dossier",
        "title": "Publish Dossier",
        "description": "Generate, audit, verify, and package the selected research dossier.",
        "commands": [
            "mechferret paper --select best --provider local",
            "mechferret audit --select best --strict",
            "mechferret verify --select best --strict",
            "mechferret bundle --select best",
            "mechferret verify-bundle --select best --strict",
        ],
    },
    {
        "name": "support_report",
        "title": "Support Report",
        "description": "Collect redacted diagnostics and artifact state for issues or pull requests.",
        "commands": [
            "mechferret support --json",
            "mechferret doctor --strict",
            "mechferret status --json",
            "mechferret open all --json",
        ],
    },
    {
        "name": "openvla_sae",
        "title": "OpenVLA SAE",
        "description": "Create and inspect the OpenVLA sparse-autoencoder project scaffold.",
        "commands": [
            "mechferret quickstart --mode openvla --run",
            "mechferret sae openvla init",
            "mechferret sae openvla status",
            "mechferret sae openvla commands",
        ],
    },
]
COMMAND_WORKFLOW_CHOICES = [str(workflow["name"]) for workflow in COMMAND_WORKFLOWS]
COMMAND_WORKFLOW_OPTION_CHOICES = ["all", *COMMAND_WORKFLOW_CHOICES]

COMMAND_EXAMPLES = {
    "version": ["mechferret version --json"],
    "commands": [
        'mechferret commands --search "max gpu"',
        "mechferret commands --workflow first_run",
        "mechferret commands run",
        "mechferret commands --examples --markdown --out docs/CLI_EXAMPLES.md",
        "mechferret commands --group research --markdown",
    ],
    "completion": ["mechferret completion zsh --json"],
    "run": [
        'mechferret run "What should I investigate?" --seed-corpus --out runs/custom',
        'mechferret run "What should I investigate?" --seed-corpus --json',
    ],
    "demo": ["mechferret demo --out runs/demo", "mechferret demo --json"],
    "login": ['mechferret login openai --api-key "$OPENAI_API_KEY" --json'],
    "api": ["mechferret api --show --json"],
    "goal": ['mechferret goal "Make this investigation publishable" --seed-corpus --max-iterations 1 --max-rounds 1 --json'],
    "discover": ["mechferret discover --skill ioi-circuit --backend synthetic --json"],
    "skills": ["mechferret skills --json", "mechferret skills ioi-circuit"],
    "modal": ["mechferret modal status --json"],
    "cluster": ["mechferret cluster run --skill ioi-circuit --dry-run --json"],
    "init": ["mechferret init", "mechferret init --json"],
    "status": ["mechferret status --select best", "mechferret status --json"],
    "runs": ["mechferret runs --select best", "mechferret runs --json"],
    "repl": ["mechferret repl"],
    "open": ["mechferret open report --select best", "mechferret open all"],
    "quickstart": [
        "mechferret quickstart",
        "mechferret quickstart --run",
        "mechferret quickstart --mode ci --run",
    ],
    "selftest": [
        "mechferret selftest --json",
        "mechferret selftest --report runs/selftest/selftest.json",
        "mechferret selftest --run --out runs/selftest",
    ],
    "support": ["mechferret support", "mechferret support --json"],
    "doctor": ["mechferret doctor --strict", "mechferret doctor --json"],
    "registry": ["mechferret registry --kind tool --json"],
    "memory": ["mechferret memory --recent 5", "mechferret memory --json"],
    "tool-results": ["mechferret tool-results", "mechferret tool-results --clean --json"],
    "cost": ["mechferret cost --select best --json"],
    "resume": ["mechferret resume --select best"],
    "inspect": ["mechferret inspect --select best --json"],
    "audit": ["mechferret audit --select best --strict", "mechferret audit --json"],
    "verify": ["mechferret verify --select best --strict", "mechferret verify --repair --json"],
    "paper": ["mechferret paper --select best --json", "mechferret paper --select best --compile"],
    "review-paper": ["mechferret review-paper --select best --json"],
    "bundle": ["mechferret bundle --select best", "mechferret bundle --select best --json"],
    "verify-bundle": ["mechferret verify-bundle --select best --strict"],
    "sae": ["mechferret sae openvla status", "mechferret sae openvla plan --json"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mechferret",
        description="Autonomous mechanistic-interpretability research. Run with no arguments for the interactive prompt.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=False, metavar="COMMAND")

    version = sub.add_parser("version", aliases=["/version", "about", "/about"], help="Print package and runtime version information.")
    version.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    commands = sub.add_parser("commands", aliases=["/commands", "help", "/help"], help="List the installed CLI command surface.")
    commands.add_argument("name", nargs="?", help="Show details for one command or alias.")
    commands.add_argument("--group", choices=COMMAND_GROUP_CHOICES, help="Limit output to a workflow group.")
    commands.add_argument("--search", help="Filter commands by name, alias, help text, option, positional, or choice.")
    commands.add_argument("--workflow", nargs="?", const="all", metavar="WORKFLOW", help="Show one workflow recipe, or all recipes when omitted.")
    commands.add_argument("--examples", action="store_true", help="Print runnable examples instead of command summaries.")
    commands.add_argument("--markdown", action="store_true", help="Render the selected command reference as Markdown.")
    commands.add_argument("--out", help="Write rendered command output to a file instead of printing it.")
    commands.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    completion = sub.add_parser("completion", aliases=["/completion"], help="Print shell completion script for mechferret.")
    completion.add_argument("shell", choices=["bash", "zsh", "fish"], help="Shell to generate completions for.")
    completion.add_argument("--command", dest="executable", default="mechferret", help="Executable name to complete.")
    completion.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    run = sub.add_parser("run", help="Run research on a question.")
    run.add_argument("question")
    run.add_argument("--source", action="append", default=[], help="File or directory of seed documents.")
    run.add_argument("--url", action="append", default=[], help="URL to fetch as a source.")
    run.add_argument("--out", default="runs/latest", help="Output directory.")
    run.add_argument("--db", default=".mechferret/memory.sqlite", help="SQLite memory path.")
    run.add_argument("--max-rounds", type=int, default=2)
    run.add_argument("--openai", action="store_true", help="Use OpenAI Responses API web search when available.")
    run.add_argument("--provider", choices=["auto", "local", "openai", "anthropic"], default="auto")
    run.add_argument("--model", help="Override the configured provider model.")
    run.add_argument("--no-memory", action="store_true", help="Do not recall prior-run memory.")
    run.add_argument("--seed-corpus", action="store_true", help="Use the packaged demo corpus when no sources, memory, or provider research are available.")
    run.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    demo = sub.add_parser("demo", help="Run the built-in hackathon demo corpus.")
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--db", default=".mechferret/memory.sqlite")
    demo.add_argument("--max-rounds", type=int, default=2)
    demo.add_argument("--openai", action="store_true")
    demo.add_argument("--provider", choices=["auto", "local", "openai", "anthropic"], default="local")
    demo.add_argument("--model", help="Override the configured provider model.")
    demo.add_argument("--with-memory", action="store_true", help="Recall prior-run memory during the demo.")
    demo.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    login = sub.add_parser("login", aliases=["/login"], help="Store an OpenAI or Anthropic API key.")
    login.add_argument("provider", choices=sorted(PROVIDERS))
    login.add_argument("--api-key", help="API key. If omitted, MechFerret prompts securely.")
    login.add_argument("--model", help="Default model for this provider.")
    login.add_argument("--no-default", action="store_true", help="Store key without making this the default provider.")
    login.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    api = sub.add_parser("api", aliases=["/api"], help="Show or change provider configuration.")
    api.add_argument("--provider", choices=sorted(PROVIDERS) + ["local"], help="Set default provider.")
    api.add_argument("--api-key", help="Store or replace the key for --provider.")
    api.add_argument("--model", help="Store or replace the default model for --provider.")
    api.add_argument("--show", action="store_true", help="Show configured provider status.")
    api.add_argument("--clear", choices=sorted(PROVIDERS), help="Remove a stored provider key.")
    api.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    goal = sub.add_parser(
        "goal",
        aliases=["/goal", "loop", "/loop"],
        help="Loop research/experiments until a target acceptance probability is reached.",
    )
    goal.add_argument("question")
    goal.add_argument("--venue", default="NeurIPS main", help="Target venue or acceptance bar.")
    goal.add_argument("--target", type=float, default=0.9, help="Target estimated acceptance probability.")
    goal.add_argument("--source", action="append", default=[], help="File or directory of seed documents.")
    goal.add_argument("--url", action="append", default=[], help="URL to fetch as a source.")
    goal.add_argument("--out", default="runs/goal", help="Output directory.")
    goal.add_argument("--db", default=".mechferret/memory.sqlite", help="SQLite memory path.")
    goal.add_argument("--max-iterations", type=int, default=5)
    goal.add_argument("--max-rounds", type=int, default=2)
    goal.add_argument("--provider", choices=["auto", "local", "openai", "anthropic"], default="auto")
    goal.add_argument("--model", help="Override the configured provider model.")
    goal.add_argument("--no-memory", action="store_true")
    goal.add_argument("--seed-corpus", action="store_true", help="Use the packaged demo corpus when no sources, memory, or provider research are available.")
    goal.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    discover = sub.add_parser(
        "discover",
        aliases=["/discover"],
        help="Autonomous interpretability discovery loop: hypothesize, experiment, critique, synthesize.",
    )
    discover.add_argument("question", nargs="?", default="", help="Research question (optional if --skill is given).")
    discover.add_argument("--skill", help="Named skill/playbook (see `mechferret /skills`) or a path to a skill JSON.")
    discover.add_argument("--task", choices=["ioi", "induction", "greater_than", "factual_recall"], help="Interpretability task.")
    discover.add_argument("--model", default="gpt2", help="Model to investigate (e.g. gpt2, pythia-160m).")
    discover.add_argument("--backend", choices=["auto", "synthetic", "transformer_lens"], default="auto")
    discover.add_argument("--source", action="append", default=[], help="Prior-art documents to ground hypotheses.")
    discover.add_argument("--url", action="append", default=[])
    discover.add_argument("--out", default="runs/discovery")
    discover.add_argument("--db", default=".mechferret/memory.sqlite")
    discover.add_argument("--max-rounds", type=int, help="Override the budget's max experiment rounds.")
    discover.add_argument("--max-experiments", type=int, help="Override the budget's max experiments.")
    discover.add_argument("--max-gpu-seconds", type=float, help="Override the budget's GPU-second ceiling.")
    discover.add_argument("--provider", choices=["auto", "local", "openai", "anthropic"], default="auto")
    discover.add_argument("--llm-model", help="Override the configured provider model for prior-art search.")
    discover.add_argument("--no-memory", action="store_true")
    discover.add_argument("--allow-mismatch", action="store_true", help="Run even if the prompt appears mismatched to the chosen skill/task.")
    discover.add_argument("--seed-corpus", action="store_true", help="Use the packaged demo corpus as prior art when no explicit sources, memory, or provider research are available.")
    discover.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    skills_cmd = sub.add_parser("skills", aliases=["/skills"], help="List or show interpretability skills/playbooks.")
    skills_cmd.add_argument("name", nargs="?", help="Show details for one skill.")
    skills_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    modal_cmd = sub.add_parser("modal", aliases=["/modal"], help="Connect to Modal for GPU compute and run experiments remotely.")
    modal_cmd.add_argument("action", nargs="?", default="status", choices=["status", "setup", "run", "deploy"])
    modal_cmd.add_argument("question", nargs="?", default="")
    modal_cmd.add_argument("--skill", help="Skill to run remotely (e.g. ioi-circuit).")
    modal_cmd.add_argument("--task", choices=["ioi", "induction", "greater_than", "factual_recall"])
    modal_cmd.add_argument("--model", default="gpt2")
    modal_cmd.add_argument("--out", default="runs/modal")
    modal_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    cluster_cmd = sub.add_parser("cluster", aliases=["/cluster"], help="Run experiments on a generic SLURM cluster over SSH (srun).")
    cluster_cmd.add_argument("action", nargs="?", default="status", choices=["status", "setup", "run"])
    cluster_cmd.add_argument("question", nargs="?", default="")
    cluster_cmd.add_argument("--skill", help="Skill to run remotely (e.g. ioi-circuit).")
    cluster_cmd.add_argument("--task", choices=["ioi", "induction", "greater_than", "factual_recall"])
    cluster_cmd.add_argument("--model", default="gpt2")
    cluster_cmd.add_argument("--out", default="runs/cluster")
    cluster_cmd.add_argument("--dry-run", action="store_true", help="Print the ssh+srun command without executing.")
    cluster_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    init = sub.add_parser("init", aliases=["/init"], help="Create MECHFERRET.md project notes for the interactive agent.")
    init.add_argument("--project-root", default=".", help="Directory where MECHFERRET.md should be written.")
    init.add_argument("--force", action="store_true", help="Overwrite an existing MECHFERRET.md.")
    init.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    status = sub.add_parser("status", aliases=["/status"], help="Summarize project setup, selected run, audit state, artifacts, and next actions.")
    status.add_argument("--runs-root", default="runs", help="Root to search for run artifacts.")
    status.add_argument("--db", default=".mechferret/memory.sqlite", help="SQLite memory path.")
    status.add_argument("--notes-root", default=".", help="Directory containing MECHFERRET.md.")
    status.add_argument("--project-root", default="projects/openvla_sae", help="OpenVLA project root.")
    status.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy for run-bound status.")
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    runs_cmd = sub.add_parser("runs", aliases=["/runs", "list-runs"], help="List recent run artifacts with audit and artifact status.")
    runs_cmd.add_argument("--runs-root", default="runs", help="Root to search for run artifacts.")
    runs_cmd.add_argument("--limit", type=int, default=10, help="Maximum runs to show.")
    runs_cmd.add_argument("--no-audit", action="store_true", help="Skip audit checks for faster listing.")
    runs_cmd.add_argument("--select", choices=["latest", "best", "ready"], default="best", help="Print the selected run policy in JSON output.")
    runs_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    sub.add_parser("repl", aliases=["chat", "/repl"], help="Launch the interactive prompt (default when run with no arguments).")

    open_cmd = sub.add_parser("open", aliases=["/open"], help="Resolve a generated artifact path, optionally opening it in a browser.")
    open_cmd.add_argument("target", nargs="?", default="quickstart", help="all | quickstart | ci | report | markdown | graph | evals | trace | experiments | discoveries | paper | review | bundle | manifest | pdf | run | openvla | explicit path")
    open_cmd.add_argument("--runs-root", default="runs", help="Root to search for run artifacts.")
    open_cmd.add_argument("--project-root", default="projects/openvla_sae", help="OpenVLA project root for target=openvla.")
    open_cmd.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy for run-bound artifacts.")
    open_cmd.add_argument("--browser", action="store_true", help="Open the resolved artifact with the system browser.")
    open_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    quick = sub.add_parser("quickstart", aliases=["/quickstart"], help="Print the recommended first commands for demo, OpenVLA, and CI gates.")
    quick.add_argument("--mode", choices=["all", "demo", "openvla", "ci"], default="all")
    quick.add_argument("--run", action="store_true", help="Execute a quickstart path. Defaults to the local demo; use --mode openvla or --mode ci for those paths.")
    quick.add_argument("--out", default="runs/demo", help="Output directory for --run --mode demo.")
    quick.add_argument("--db", default=".mechferret/memory.sqlite", help="Memory database for --run --mode demo.")
    quick.add_argument("--project-root", default="projects/openvla_sae", help="Project root for --run --mode openvla.")
    quick.add_argument("--force", action="store_true", help="Overwrite existing files for --run --mode openvla.")
    quick.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    selftest_cmd = sub.add_parser("selftest", aliases=["/selftest"], help="Run offline readiness checks and optionally execute the local demo artifact path.")
    selftest_cmd.add_argument("--run", action="store_true", help="Execute the local demo quickstart and verify its artifacts.")
    selftest_cmd.add_argument("--out", default="runs/selftest", help="Output directory for --run artifacts.")
    selftest_cmd.add_argument("--db", default=".mechferret/selftest.sqlite", help="Memory database for --run artifacts.")
    selftest_cmd.add_argument("--runs-root", default="runs", help="Root to inspect for existing run artifacts.")
    selftest_cmd.add_argument("--notes-root", default=".", help="Directory containing MECHFERRET.md.")
    selftest_cmd.add_argument("--project-root", default="projects/openvla_sae", help="OpenVLA project root to inspect.")
    selftest_cmd.add_argument("--select", choices=["latest", "best", "ready"], default="best", help="Run-selection policy for status inspection.")
    selftest_cmd.add_argument("--report", help="Write the self-test JSON payload to this path for sharing or issue reports.")
    selftest_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    support_cmd = sub.add_parser("support", aliases=["/support", "diagnostics", "/diagnostics"], help="Write a shareable self-test report for issues or PRs.")
    support_cmd.add_argument("--run", action="store_true", help="Execute the local demo quickstart before writing the support report.")
    support_cmd.add_argument("--out", default="runs/selftest", help="Output directory for --run artifacts.")
    support_cmd.add_argument("--db", default=".mechferret/selftest.sqlite", help="Memory database for --run artifacts.")
    support_cmd.add_argument("--runs-root", default="runs", help="Root to inspect for existing run artifacts.")
    support_cmd.add_argument("--notes-root", default=".", help="Directory containing MECHFERRET.md.")
    support_cmd.add_argument("--project-root", default="projects/openvla_sae", help="OpenVLA project root to inspect.")
    support_cmd.add_argument("--select", choices=["latest", "best", "ready"], default="best", help="Run-selection policy for status inspection.")
    support_cmd.add_argument("--report", default="runs/selftest/selftest.json", help="Path for the support JSON report.")
    support_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    doctor = sub.add_parser("doctor", aliases=["/doctor"], help="Check config, packages, corpus, and registry health.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.add_argument("--strict", action="store_true", help="Exit nonzero if core release-critical checks fail.")
    doctor.add_argument("--all-integrations", action="store_true", help="Treat optional packages, API keys, cluster, and manifests as required.")
    doctor.set_defaults(_doctor=True)

    registry = sub.add_parser("registry", aliases=["/registry"], help="List available tools, tasks, playbooks, and evaluators.")
    registry.add_argument("--kind", choices=["tool", "task", "playbook", "evaluator"])
    registry.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    memory = sub.add_parser("memory", aliases=["/memory"], help="Inspect or clear research memory.")
    memory.add_argument("--db", default=".mechferret/memory.sqlite")
    memory.add_argument("--recent", type=int, default=0, help="Show recent remembered runs.")
    memory.add_argument("--clear", action="store_true", help="Delete the memory database.")
    memory.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    tool_results = sub.add_parser("tool-results", aliases=["/tool-results"], help="List or clean saved large agent-tool outputs.")
    tool_results.add_argument("--limit", type=int, default=20, help="Maximum saved outputs to show.")
    tool_results.add_argument("--clean", action="store_true", help="Clean stale saved outputs instead of listing them.")
    tool_results.add_argument("--keep-latest", type=int, default=20, help="Keep at least this many newest saved outputs when cleaning.")
    tool_results.add_argument("--max-age-days", type=float, default=7.0, help="Delete saved outputs older than this many days when cleaning.")
    tool_results.add_argument("--confirm", action="store_true", help="Actually delete files for --clean. Without this, cleanup is a dry run.")
    tool_results.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    cost = sub.add_parser("cost", aliases=["/cost"], help="Estimate cost/usage from a run artifact.")
    cost.add_argument("run_json", nargs="?", help="Run artifact (defaults to selected runs/**/run.json).")
    cost.add_argument("--runs-root", default="runs", help="Root to search when run_json is omitted.")
    cost.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when run_json is omitted.")
    cost.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    resume = sub.add_parser("resume", aliases=["/resume"], help="Summarize a prior run artifact.")
    resume.add_argument("run_json", nargs="?", help="Run artifact (defaults to selected runs/**/run.json).")
    resume.add_argument("--runs-root", default="runs", help="Root to search when run_json is omitted.")
    resume.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when run_json is omitted.")
    resume.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    inspect = sub.add_parser("inspect", help="Print a compact summary of a run JSON artifact.")
    inspect.add_argument("run_json", nargs="?", help="Run artifact (defaults to selected runs/**/run.json).")
    inspect.add_argument("--runs-root", default="runs", help="Root to search when run_json is omitted.")
    inspect.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when run_json is omitted.")
    inspect.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    audit = sub.add_parser("audit", aliases=["/audit"], help="Run offline paper-readiness gates on a dossier.")
    audit.add_argument("run_json", nargs="?", help="Run artifact to audit (defaults to selected runs/**/run.json).")
    audit.add_argument("--runs-root", default="runs", help="Root to search when run_json is omitted.")
    audit.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when run_json is omitted.")
    audit.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    audit.add_argument("--strict", action="store_true", help="Exit nonzero if any audit gate fails.")

    verify = sub.add_parser("verify", aliases=["/verify"], help="Verify run manifest integrity and artifact existence.")
    verify.add_argument("run_json", nargs="?", help="Run artifact to verify (defaults to selected runs/**/run.json).")
    verify.add_argument("--runs-root", default="runs", help="Root to search when run_json is omitted.")
    verify.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when run_json is omitted.")
    verify.add_argument("--repair", action="store_true", help="Refresh manifest.json when only manifest coverage metadata is stale.")
    verify.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    verify.add_argument("--strict", action="store_true", help="Exit nonzero if verification fails.")

    paper = sub.add_parser("paper", aliases=["/paper"], help="Generate main.tex from a run artifact.")
    paper.add_argument("run_json", nargs="?", help="Run artifact to write from (defaults to selected runs/**/run.json).")
    paper.add_argument("--runs-root", default="runs", help="Root to search when run_json is omitted.")
    paper.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when run_json is omitted.")
    paper.add_argument("--out", help="Output directory for main.tex (defaults to the run artifact's paper/ directory).")
    paper.add_argument("--compile", action="store_true", help="Compile main.pdf with tectonic if installed.")
    paper.add_argument("--compile-timeout", type=int, default=TECTONIC_TIMEOUT_SECONDS, help="Seconds to wait for tectonic before reporting a timeout.")
    paper.add_argument("--provider", choices=["auto", "local", "openai", "anthropic"], default="auto")
    paper.add_argument("--model", help="Override the configured provider model.")
    paper.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    review = sub.add_parser("review-paper", aliases=["/review-paper"], help="Review a generated paper with a configured model.")
    review.add_argument("paper_tex", nargs="?", help="Paper to review (defaults to selected run-bound paper).")
    review.add_argument("--runs-root", default="runs", help="Root to search when paper_tex is omitted.")
    review.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when paper_tex is omitted.")
    review.add_argument("--out", help="Directory for review.md (defaults beside the paper).")
    review.add_argument("--provider", choices=["auto", "openai", "anthropic"], default="auto")
    review.add_argument("--model", help="Override the configured provider model.")
    review.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    bundle = sub.add_parser("bundle", aliases=["/bundle"], help="Create a shareable zip bundle for a run dossier.")
    bundle.add_argument("run_json", nargs="?", help="Run artifact to bundle (defaults to selected runs/**/run.json).")
    bundle.add_argument("--runs-root", default="runs", help="Root to search when run_json is omitted.")
    bundle.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when run_json is omitted.")
    bundle.add_argument("--out", help="Output .zip path or directory (defaults beside the run).")
    bundle.add_argument("--notes-root", default=".", help="Directory containing MECHFERRET.md.")
    bundle.add_argument("--project-root", default="projects/openvla_sae", help="OpenVLA project root.")
    bundle.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    verify_bundle = sub.add_parser("verify-bundle", aliases=["/verify-bundle"], help="Verify a shareable bundle zip using its portable manifest.")
    verify_bundle.add_argument("bundle_zip", nargs="?", help="Bundle zip to verify (defaults to selected run-bound bundle artifact).")
    verify_bundle.add_argument("--runs-root", default="runs", help="Root to search when bundle_zip is omitted.")
    verify_bundle.add_argument("--select", choices=["latest", "best", "ready"], default="latest", help="Run-selection policy when bundle_zip is omitted.")
    verify_bundle.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    verify_bundle.add_argument("--strict", action="store_true", help="Exit nonzero if bundle verification fails.")

    sae = sub.add_parser("sae", aliases=["/sae"], help="SAE project workflows, including OpenVLA.")
    sae.add_argument("project", choices=["openvla"])
    sae.add_argument("action", nargs="?", default="status", choices=["status", "init", "plan", "commands", "validate-manifest", "create-manifest", "smoke", "eval", "features", "dossier"])
    sae.add_argument("--project-root", default="projects/openvla_sae", help="OpenVLA SAE workflow directory.")
    sae.add_argument("--manifest", help="JSONL manifest of image_path/instruction rows.")
    sae.add_argument("--image-dir", help="Directory of images for create-manifest.")
    sae.add_argument("--instruction", default="perform the task shown in the image", help="Starter instruction for create-manifest rows.")
    sae.add_argument("--action-label", default="", help="Optional action label value for create-manifest rows.")
    sae.add_argument("--limit", type=int, help="Maximum images to include.")
    sae.add_argument("--force", action="store_true", help="Overwrite an existing manifest for create-manifest.")
    sae.add_argument("--out", default="runs/openvla_sae/plan", help="Output directory for plan artifacts.")
    sae.add_argument("--cache-dir", default="runs/openvla_sae/cache_l24", help="Activation cache directory for eval.")
    sae.add_argument("--checkpoint", default="runs/openvla_sae/sae_l24_topk.pt", help="SAE checkpoint for eval.")
    sae.add_argument("--eval-dir", default="runs/openvla_sae/eval", help="Eval artifact directory for dossier.")
    sae.add_argument("--features-dir", default="runs/openvla_sae/features", help="Feature artifact directory for dossier.")
    sae.add_argument("--top-features", type=int, default=20, help="Number of top SAE features for features.")
    sae.add_argument("--max-files", type=int, default=64, help="Maximum cached activation files for features.")
    sae.add_argument("--d-model", type=int, default=32, help="Synthetic hidden size for smoke.")
    sae.add_argument("--tokens", type=int, default=256, help="Synthetic token count for smoke.")
    sae.add_argument("--steps", type=int, default=20, help="Training steps for smoke.")
    sae.add_argument("--k", type=int, default=4, help="Top-K value for smoke.")
    sae.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    _exit_on_unknown_command(parser, argv_list)
    args = parser.parse_args(argv_list)
    if args.command is None or args.command in {"repl", "chat", "/repl"}:
        from .repl import run_repl

        run_repl()
        return
    if args.command in {"version", "/version", "about", "/about"}:
        payload = _version_payload()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"MechFerret {payload['version']}")
            print(f"Python: {payload['python']}")
            print(f"Executable: {payload['executable']}")
            print(f"Config: {payload['config_path']}")
        return
    if args.command in {"commands", "/commands", "help", "/help"}:
        if args.json and args.markdown:
            payload = {
                "ok": False,
                "name": "mechferret",
                "version": __version__,
                "error": "json and markdown are mutually exclusive",
                "next_actions": [
                    "Run `mechferret commands --json` for structured data.",
                    "Run `mechferret commands --markdown` for Markdown output.",
                ],
            }
        elif args.workflow and (args.name or args.search or args.group or args.examples):
            payload = {
                "ok": False,
                "name": "mechferret",
                "version": __version__,
                "error": "workflow cannot be combined with name, search, group, or examples",
                "next_actions": [
                    "Run `mechferret commands --workflow first_run` for one workflow.",
                    "Run `mechferret commands` to list commands and workflows.",
                ],
            }
        elif args.name and (args.search or args.group):
            payload = {
                "ok": False,
                "name": "mechferret",
                "version": __version__,
                "error": "name cannot be combined with search or group",
                "next_actions": [
                    "Run `mechferret commands <name>` for one command.",
                    "Run `mechferret commands --search <query>` to filter commands.",
                    "Run `mechferret commands --group research` to list a workflow group.",
                ],
            }
        else:
            payload = _command_workflow_payload(args.workflow) if args.workflow else _command_index_payload(parser, query=args.name, search=args.search, group=args.group)
        if payload["ok"] and args.examples:
            payload = _command_examples_payload(payload, query=args.name, search=args.search, group=args.group)
        if args.out and payload["ok"]:
            rendered, format_name = _render_command_output(
                payload,
                query=args.name,
                search=args.search,
                group=args.group,
                examples=args.examples,
                markdown=args.markdown,
                as_json=args.json,
            )
            result = _write_command_output(args.out, rendered, format_name=format_name, count=int(payload.get("count", 0)))
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"Wrote command {format_name}: {result['path']}")
            return
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
            if not payload["ok"]:
                raise SystemExit(1)
        elif args.markdown:
            if not payload["ok"]:
                _print_command_error(payload)
                raise SystemExit(1)
            print(_command_markdown(payload))
        else:
            if not payload["ok"]:
                _print_command_error(payload)
                raise SystemExit(1)
            if args.examples:
                _print_command_examples(payload["commands"], query=args.name, search=args.search, group=args.group)
            elif args.name:
                _print_command_detail(payload["commands"][0])
            elif args.workflow:
                if payload.get("workflow_list"):
                    print(_command_workflows_text(payload["workflows"]))
                else:
                    print(_command_workflow_detail_text(payload["workflows"][0]))
            elif args.search:
                print(
                    _command_list_text(
                        payload["commands"],
                        count=int(payload["count"]),
                        search=args.search,
                        group=args.group,
                        workflows=payload.get("workflows", []),
                        workflow_count=int(payload.get("workflow_count", 0)),
                    )
                )
            else:
                print(
                    _command_list_text(
                        payload["commands"],
                        count=int(payload["count"]),
                        group=args.group,
                        workflows=payload.get("workflows", []),
                    )
                )
        return
    if args.command in {"completion", "/completion"}:
        payload = _completion_payload(parser, args.shell, executable=args.executable)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(payload["script"])
        return
    if args.command == "run":
        engine = MechFerret(args.db)
        try:
            run = engine.run(
                args.question,
                source_paths=args.source,
                urls=args.url,
                out_dir=args.out,
                max_rounds=args.max_rounds,
                use_openai=args.openai,
                provider=args.provider,
                model=args.model,
                include_memory=not args.no_memory,
                allow_seed_corpus=args.seed_corpus,
            )
        except (ValueError, FileNotFoundError) as exc:
            if args.json:
                print(json.dumps(_error_payload("run", exc, out_dir=args.out), indent=2, sort_keys=True))
                raise SystemExit(2) from None
            print(f"Run not started: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        if args.json:
            print(json.dumps(_run_payload(run, command="run"), indent=2, sort_keys=True))
            return
        print_summary(run)
    elif args.command == "demo":
        engine = MechFerret(args.db)
        run = engine.run(
            DEMO_QUESTION,
            source_paths=[str(example_corpus_path())],
            out_dir=args.out,
            max_rounds=args.max_rounds,
            use_openai=args.openai,
            provider=args.provider,
            model=args.model,
            include_memory=args.with_memory,
            allow_seed_corpus=True,
        )
        if args.json:
            print(json.dumps(_run_payload(run, command="demo"), indent=2, sort_keys=True))
            return
        print_summary(run)
    elif args.command in {"login", "/login"}:
        if args.json and not args.api_key:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "action": "login",
                        "provider": args.provider,
                        "error": "--api-key is required with --json",
                        "next_actions": [f"Pass --api-key for {args.provider}, or run without --json for an interactive prompt."],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            raise SystemExit(2)
        key = args.api_key or prompt_api_key(args.provider)
        if not key:
            if args.json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "action": "login",
                            "provider": args.provider,
                            "error": "No API key provided.",
                            "next_actions": [f"Pass --api-key for {args.provider}."],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                raise SystemExit(2)
            raise SystemExit("No API key provided.")
        path = configure_provider(
            args.provider,
            key,
            model=args.model,
            make_default=not args.no_default,
        )
        if args.json:
            print(json.dumps(_api_payload(load_config(), action="login", path=path, provider=args.provider), indent=2, sort_keys=True))
            return
        print(f"Stored {args.provider} credentials in {path}")
        print(f"Default provider: {load_config().default_provider}")
    elif args.command in {"api", "/api"}:
        handle_api_command(args)
    elif args.command in {"goal", "/goal", "loop", "/loop"}:
        loop = GoalLoop(args.db)
        try:
            result = loop.run(
                args.question,
                venue=args.venue,
                target=args.target,
                source_paths=args.source,
                urls=args.url,
                out_dir=args.out,
                max_iterations=args.max_iterations,
                max_rounds=args.max_rounds,
                provider=args.provider,
                model=args.model,
                include_memory=not args.no_memory,
                allow_seed_corpus=args.seed_corpus,
            )
        except (ValueError, FileNotFoundError) as exc:
            if args.json:
                print(json.dumps(_error_payload("goal", exc, out_dir=args.out), indent=2, sort_keys=True))
                raise SystemExit(2) from None
            print(f"Goal loop not started: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        if args.json:
            payload = dict(result)
            payload["ok"] = True
            payload["command"] = "goal"
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        print(f"Goal status: {result['status']}")
        print(f"Best probability: {result['best_probability']:.2f}")
        print(f"Iterations: {len(result['iterations'])}")
        print(f"Report: {result['artifact']}")
    elif args.command in {"discover", "/discover"}:
        handle_discover(args)
    elif args.command in {"skills", "/skills"}:
        handle_skills(args)
    elif args.command in {"modal", "/modal"}:
        handle_modal(args)
    elif args.command in {"cluster", "/cluster"}:
        handle_cluster(args)
    elif args.command in {"init", "/init"}:
        result = init_project_notes(args.project_root, force=args.force)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_project_init(result)
        if not result["ok"]:
            raise SystemExit(1)
    elif args.command in {"status", "/status"}:
        result = project_status(
            runs_root=args.runs_root,
            db_path=args.db,
            notes_root=args.notes_root,
            project_root=args.project_root,
            selection=args.select,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_project_status(result)
    elif args.command in {"runs", "/runs", "list-runs"}:
        result = list_run_artifacts(runs_root=args.runs_root, limit=args.limit, include_audit=not args.no_audit, selection=args.select)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_run_list(result)
    elif args.command in {"open", "/open"}:
        result = resolve_artifact(args.target, runs_root=args.runs_root, project_root=args.project_root, selection=args.select)
        if args.browser:
            result["opened"] = open_artifact(result)
        if args.json:
            print(json.dumps(_artifact_payload(result), indent=2, sort_keys=True))
        else:
            print_artifact_result(result)
        if not result["exists"]:
            raise SystemExit(1)
    elif args.command in {"quickstart", "/quickstart"}:
        if args.run:
            result = run_quickstart(
                args.mode,
                out_dir=args.out,
                db_path=args.db,
                project_root=args.project_root,
                force=args.force,
            )
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print_quickstart_run(result)
            if not result["ok"]:
                raise SystemExit(1)
            return
        result = quickstart(args.mode)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_quickstart(result)
    elif args.command in {"selftest", "/selftest", "support", "/support", "diagnostics", "/diagnostics"}:
        result = selftest(
            run=args.run,
            out_dir=args.out,
            db_path=args.db,
            runs_root=args.runs_root,
            notes_root=args.notes_root,
            project_root=args.project_root,
            selection=args.select,
            report_path=args.report if args.command in {"selftest", "/selftest"} else (args.report or "runs/selftest/selftest.json"),
            command="selftest" if args.command in {"selftest", "/selftest"} else "support",
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_selftest(result)
        if not result["ok"]:
            raise SystemExit(1)
    elif args.command in {"doctor", "/doctor"}:
        from .ops import doctor

        result = doctor()
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_doctor(strict=args.strict, all_integrations=args.all_integrations)
        if args.all_integrations and not result["all_integrations_passed"]:
            raise SystemExit(1)
        if args.strict and not result["strict_passed"]:
            raise SystemExit(1)
    elif args.command in {"registry", "/registry"}:
        items = items_by_kind(args.kind) if args.kind else all_items()
        if args.json:
            payload = {
                "ok": True,
                "kind": args.kind or "all",
                "count": len(items),
                "items": [item.to_dict() for item in items],
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for item in items:
                print(f"{item.kind:9} {item.name:24} {item.status:10} {item.description}")
    elif args.command in {"memory", "/memory"}:
        if args.clear:
            memory_clear(args.db)
            if args.json:
                print(json.dumps({"ok": True, "db": str(args.db), "cleared": True}, indent=2, sort_keys=True))
            else:
                print(f"Cleared memory at {args.db}")
            return
        summary = memory_summary(args.db)
        recent = memory_recent(args.db, args.recent) if args.recent else []
        if args.json:
            print(
                json.dumps(
                    {"ok": True, "db": str(args.db), "summary": summary, "recent": recent},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Memory: runs={summary['runs']} claims={summary['claims']} sources={summary['sources']}")
            for row in recent:
                score = row["metrics"].get("readiness_score", 0)
                print(f"{row['created_at']} {row['id']} readiness={score:.2f} {row['question'][:90]}")
    elif args.command in {"tool-results", "/tool-results"}:
        if args.clean:
            result = json.loads(
                tool_clean_tool_results(
                    {
                        "keep_latest": args.keep_latest,
                        "max_age_days": args.max_age_days,
                        "confirm": args.confirm,
                        "dry_run": not args.confirm,
                    }
                )
            )
        else:
            result = json.loads(tool_list_tool_results({"limit": args.limit}))
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_tool_results(result)
        if not result.get("ok", True):
            raise SystemExit(1)
    elif args.command in {"cost", "/cost"}:
        if args.json:
            resolved, error = _resolve_run_json_for_json(args.run_json, args.runs_root, args.select)
            if error is not None:
                print(json.dumps(error, indent=2, sort_keys=True))
                raise SystemExit(1)
            cost = estimate_run_cost(resolved)
            cost["ok"] = True
            cost["path"] = str(resolved)
            cost["runs_root"] = str(args.runs_root)
            cost["selection"] = args.select
            print(json.dumps(cost, indent=2, sort_keys=True))
        else:
            run_json = _resolve_run_json_arg(args.run_json, args.runs_root, args.select)
            cost = estimate_run_cost(run_json)
            print(f"Run: {cost['run_id']}")
            print(f"Estimated tokens processed: {cost['estimated_tokens_processed']}")
            print(f"Estimated provider calls: {cost['estimated_provider_calls']}")
            print(f"Local plan steps: {cost['local_steps']}")
            print(cost["note"])
    elif args.command in {"resume", "/resume"}:
        if args.json:
            resolved, error = _resolve_run_json_for_json(args.run_json, args.runs_root, args.select)
            if error is not None:
                print(json.dumps(error, indent=2, sort_keys=True))
                raise SystemExit(1)
            summary = summarize_run_artifact(resolved)
            summary["ok"] = True
            summary["path"] = str(resolved)
            summary["runs_root"] = str(args.runs_root)
            summary["selection"] = args.select
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            summary = summarize_run_artifact(_resolve_run_json_arg(args.run_json, args.runs_root, args.select))
            print(f"Run: {summary['run_id']}")
            print(f"Question: {summary['question']}")
            print(f"Readiness: {summary['readiness_score']:.2f}")
            print(f"Claims: {summary['claims']}")
            print(f"Evidence chunks: {summary['evidence']}")
            print(f"Gaps: {len(summary['gaps'])}")
            if summary["artifacts"].get("html"):
                print(f"Report: {summary['artifacts']['html']}")
    elif args.command == "inspect":
        if args.json:
            resolved, error = _resolve_run_json_for_json(args.run_json, args.runs_root, args.select)
            if error is not None:
                print(json.dumps(error, indent=2, sort_keys=True))
                raise SystemExit(1)
            summary = summarize_run_artifact(resolved)
            summary["ok"] = True
            summary["path"] = str(resolved)
            summary["runs_root"] = str(args.runs_root)
            summary["selection"] = args.select
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            summary = summarize_run_artifact(_resolve_run_json_arg(args.run_json, args.runs_root, args.select))
            print(f"Question: {summary['question']}")
            print(f"Readiness: {summary['readiness_score']:.2f}")
            print(f"Claims: {summary['claims']}")
            print(f"Evidence chunks: {summary['evidence']}")
            print(f"Gaps: {len(summary['gaps'])}")
    elif args.command in {"audit", "/audit"}:
        if args.json and (args.run_json or args.select != "latest"):
            resolved_run, error = _resolve_run_json_for_json(args.run_json, args.runs_root, args.select)
            if error is not None:
                print(json.dumps(error, indent=2, sort_keys=True))
                raise SystemExit(1)
        else:
            resolved_run = _resolve_run_json_arg(args.run_json, args.runs_root, args.select) if args.run_json or args.select != "latest" else None
        result = audit_run_artifact(resolved_run, runs_root=args.runs_root)
        if args.json:
            print(json.dumps(_passed_payload(result), indent=2, sort_keys=True))
        else:
            print_audit(result)
        if args.strict and not result["passed"]:
            raise SystemExit(1)
    elif args.command in {"verify", "/verify"}:
        if args.json and (args.run_json or args.select != "latest"):
            resolved_run, error = _resolve_run_json_for_json(args.run_json, args.runs_root, args.select)
            if error is not None:
                print(json.dumps(error, indent=2, sort_keys=True))
                raise SystemExit(1)
        else:
            resolved_run = _resolve_run_json_arg(args.run_json, args.runs_root, args.select) if args.run_json or args.select != "latest" else None
        result = verify_run_artifacts(resolved_run, runs_root=args.runs_root, repair=args.repair)
        if args.json:
            print(json.dumps(_passed_payload(result), indent=2, sort_keys=True))
        else:
            print_verify_result(result)
        if args.strict and not result["passed"]:
            raise SystemExit(1)
    elif args.command in {"paper", "/paper"}:
        if args.json:
            resolved, error = _resolve_run_json_for_json(args.run_json, args.runs_root, args.select)
            if error is not None:
                print(json.dumps(error, indent=2, sort_keys=True))
                raise SystemExit(1)
        else:
            resolved = _resolve_run_json_arg(args.run_json, args.runs_root, args.select)
        result = write_paper_from_artifact(
            resolved,
            out_dir=args.out,
            compile_pdf=args.compile,
            compile_timeout=args.compile_timeout,
            provider=args.provider,
            model=args.model,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_paper_result(result)
    elif args.command in {"review-paper", "/review-paper"}:
        result = review_paper(
            args.paper_tex,
            out_dir=args.out,
            provider=args.provider,
            model=args.model,
            runs_root=args.runs_root,
            selection=args.select,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_review_result(result)
        if not result["ok"]:
            raise SystemExit(1)
    elif args.command in {"bundle", "/bundle"}:
        if args.json and (args.run_json or args.select != "latest"):
            resolved_run, error = _resolve_run_json_for_json(args.run_json, args.runs_root, args.select)
            if error is not None:
                print(json.dumps(error, indent=2, sort_keys=True))
                raise SystemExit(1)
        else:
            resolved_run = _resolve_run_json_arg(args.run_json, args.runs_root, args.select) if args.run_json or args.select != "latest" else None
        result = bundle_run_artifacts(
            resolved_run,
            runs_root=args.runs_root,
            selection=args.select,
            out=args.out,
            notes_root=args.notes_root,
            project_root=args.project_root,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_bundle_result(result)
        if not result["ok"]:
            raise SystemExit(1)
    elif args.command in {"verify-bundle", "/verify-bundle"}:
        result = verify_bundle_artifacts(args.bundle_zip, runs_root=args.runs_root, selection=args.select)
        if args.json:
            print(json.dumps(_passed_payload(result), indent=2, sort_keys=True))
        else:
            print_verify_bundle_result(result)
        if args.strict and not result["passed"]:
            raise SystemExit(1)
    elif args.command in {"sae", "/sae"}:
        handle_sae(args)


def handle_discover(args) -> None:
    skill = args.skill
    note = ""
    if not skill and not args.question and not args.task:
        skill = "ioi-circuit"  # the headline demo
        note = "No question/skill/task given; running the `ioi-circuit` skill."
        if not args.json:
            print(f"{note}\n")
    budget = _budget_override(args)
    try:
        run = DiscoveryController(args.db).run(
            question=args.question,
            skill=skill,
            task=args.task,
            model=args.model,
            backend=args.backend,
            source_paths=args.source,
            urls=args.url,
            out_dir=args.out,
            budget=budget,
            provider=args.provider,
            llm_model=args.llm_model,
            include_memory=not args.no_memory,
            allow_mismatch=args.allow_mismatch,
            allow_seed_corpus=args.seed_corpus,
        )
    except ValueError as exc:
        if args.json:
            print(json.dumps(_error_payload("discover", exc, out_dir=args.out), indent=2, sort_keys=True))
            raise SystemExit(2) from None
        print(f"Discovery not started: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.json:
        payload = _run_payload(run, command="discover")
        if note:
            payload["note"] = note
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print_discovery_summary(run)


def handle_sae(args) -> None:
    if args.project != "openvla":
        raise SystemExit(f"Unsupported SAE project: {args.project}")
    if args.action == "status":
        result = openvla_sae_status(project_root=args.project_root, manifest=args.manifest)
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_status(result)
    elif args.action == "init":
        result = init_openvla_sae_project(args.project_root, force=args.force)
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_init(result)
    elif args.action == "plan":
        result = write_openvla_sae_plan(out_dir=args.out, project_root=args.project_root, manifest=args.manifest)
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_plan(result)
    elif args.action == "commands":
        commands = openvla_sae_commands(args.project_root)
        if args.json:
            print(
                json.dumps(
                    _sae_payload(args.action, {"ok": True, "project_root": str(args.project_root), "commands": commands}),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        print(commands)
    elif args.action == "validate-manifest":
        if not args.manifest:
            _raise_sae_usage(args, "--manifest is required for validate-manifest")
        result = validate_openvla_sae_manifest(args.manifest)
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.action == "create-manifest":
        if not args.image_dir:
            _raise_sae_usage(args, "--image-dir is required for create-manifest")
        try:
            result = create_openvla_sae_manifest(
                args.image_dir,
                args.manifest or "data/openvla_sae_phase1.jsonl",
                instruction=args.instruction,
                action=args.action_label,
                limit=args.limit,
                force=args.force,
            )
        except (FileExistsError, FileNotFoundError) as exc:
            if args.json:
                print(json.dumps(_sae_error(args.action, exc), indent=2, sort_keys=True))
                raise SystemExit(2) from None
            raise
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_manifest(result)
    elif args.action == "smoke":
        result = openvla_sae_smoke(
            out_dir=args.out,
            d_model=args.d_model,
            tokens=args.tokens,
            steps=args.steps,
            k=args.k,
        )
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_smoke(result)
    elif args.action == "eval":
        result = evaluate_openvla_sae_artifacts(
            cache_dir=args.cache_dir,
            checkpoint=args.checkpoint,
            out_dir=args.out,
        )
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_eval(result)
    elif args.action == "features":
        result = openvla_sae_feature_report(
            cache_dir=args.cache_dir,
            checkpoint=args.checkpoint,
            out_dir=args.out,
            top_k=args.top_features,
            max_files=args.max_files,
        )
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_features(result)
    elif args.action == "dossier":
        result = write_openvla_sae_dossier(
            out_dir=args.out,
            project_root=args.project_root,
            manifest=args.manifest,
            cache_dir=args.cache_dir,
            checkpoint=args.checkpoint,
            eval_dir=args.eval_dir,
            features_dir=args.features_dir,
        )
        if args.json:
            print(json.dumps(_sae_payload(args.action, result), indent=2, sort_keys=True))
            return
        print_openvla_sae_dossier(result)


def _raise_sae_usage(args, message: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(_sae_error(getattr(args, "action", "unknown"), ValueError(message)), indent=2, sort_keys=True))
        raise SystemExit(2) from None
    raise SystemExit(message)


def _sae_payload(action: str, result: dict[str, Any]) -> dict[str, Any]:
    payload = _json_ready(result)
    if not isinstance(payload, dict):
        payload = {"result": payload}
    payload.setdefault("ok", True)
    payload["action"] = action
    payload["project"] = "openvla"
    return payload


def _sae_error(action: str, exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "project": "openvla",
        "action": action,
        "error": str(exc),
        "next_actions": [
            "Run `mechferret sae openvla status --json` to inspect the workflow.",
            "Run `mechferret sae openvla commands --json` for cache/train command templates.",
        ],
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str) or value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else 0.0
    try:
        return str(value)
    except Exception:
        return ""


def _version_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "name": "mechferret",
        "version": __version__,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "executable": sys.executable,
        "config_path": str(default_config_path()),
    }


def _command_index_payload(
    parser: argparse.ArgumentParser,
    query: str | None = None,
    search: str | None = None,
    group: str | None = None,
) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    subparsers = _subparser_actions(parser)
    seen: set[int] = set()
    for action in subparsers:
        for name, subparser in action.choices.items():
            parser_id = id(subparser)
            if parser_id in seen:
                continue
            seen.add(parser_id)
            aliases = [
                alias
                for alias, candidate in action.choices.items()
                if id(candidate) == parser_id and alias != name
            ]
            commands.append(
                {
                    "name": name,
                    "aliases": aliases,
                    "help": _command_help(action, name),
                    "usage": subparser.format_usage().strip(),
                    "examples": list(COMMAND_EXAMPLES.get(name, [])),
                    "options": [_argument_payload(item) for item in subparser._actions if item.option_strings],
                    "positionals": [
                        _argument_payload(item)
                        for item in subparser._actions
                        if not item.option_strings and item.dest != argparse.SUPPRESS
                    ],
                }
            )
    commands = _filter_commands_by_group(commands, group)
    if query:
        selected = _find_command(commands, query)
        if selected is None:
            suggestions = _suggest_commands(commands, query)
            return {
                "ok": False,
                "name": "mechferret",
                "version": __version__,
                "query": query,
                "error": "unknown command",
                "count": 0,
                "commands": [],
                "available": [command["name"] for command in commands],
                "suggestions": suggestions,
                "next_actions": ["Run `mechferret commands` to list available commands."],
            }
        return {
            "ok": True,
            "name": "mechferret",
            "version": __version__,
            "query": query,
            **({"group": group} if group else {}),
            "count": 1,
            "commands": [selected],
        }
    if search:
        filtered = _search_commands(commands, search)
        workflows = _search_command_workflows(search) if not group else []
        return {
            "ok": True,
            "name": "mechferret",
            "version": __version__,
            "search": search,
            **({"group": group} if group else {}),
            "count": len(filtered),
            "workflow_count": len(workflows),
            "workflows": workflows,
            "commands": filtered,
        }
    return {
        "ok": True,
        "name": "mechferret",
        "version": __version__,
        **({"group": group} if group else {}),
        "count": len(commands),
        "workflows": _command_workflows_payload() if not group else [],
        "commands": commands,
    }


def _exit_on_unknown_command(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if not argv:
        return
    command = argv[0]
    if not command or command.startswith("-"):
        return
    choices = _command_choice_names(parser)
    if command in choices:
        return
    payload = _unknown_command_payload(parser, command)
    if "--json" in argv:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        if payload["suggestions"]:
            print(f"Did you mean: {', '.join(payload['suggestions'])}?", file=sys.stderr)
        print("Run `mechferret commands` to list available commands.", file=sys.stderr)
    raise SystemExit(2)


def _unknown_command_payload(parser: argparse.ArgumentParser, command: str) -> dict[str, Any]:
    index = _command_index_payload(parser)
    commands = index["commands"]
    return {
        "ok": False,
        "name": "mechferret",
        "version": __version__,
        "query": command,
        "error": "unknown command",
        "available": [item["name"] for item in commands],
        "suggestions": _suggest_commands(commands, command),
        "next_actions": ["Run `mechferret commands` to list available commands."],
    }


def _command_examples_payload(
    payload: dict[str, Any],
    *,
    query: str | None = None,
    search: str | None = None,
    group: str | None = None,
) -> dict[str, Any]:
    rows = [
        {
            "name": command["name"],
            "aliases": command.get("aliases", []),
            "help": command.get("help", ""),
            "examples": command.get("examples", []),
        }
        for command in payload.get("commands", [])
        if command.get("examples")
    ]
    result = {
        "ok": True,
        "name": payload.get("name", "mechferret"),
        "version": payload.get("version", __version__),
        "examples_only": True,
        "count": len(rows),
        "commands": rows,
    }
    if query:
        result["query"] = query
    if search:
        result["search"] = search
    if group:
        result["group"] = group
    return result


def _command_workflow_payload(name: str) -> dict[str, Any]:
    if _workflow_key(name) == "all":
        workflows = _command_workflows_payload()
        return {
            "ok": True,
            "name": "mechferret",
            "version": __version__,
            "workflow": "all",
            "workflow_only": True,
            "workflow_list": True,
            "count": len(workflows),
            "commands": [],
            "workflows": workflows,
        }
    workflow = _find_command_workflow(name)
    if workflow is None:
        suggestions = _suggest_command_workflows(str(name))
        next_actions = []
        if suggestions:
            next_actions.append(f"Run `mechferret commands --workflow {suggestions[0]}`.")
        next_actions.append("Run `mechferret commands` to list commands and workflows.")
        return {
            "ok": False,
            "name": "mechferret",
            "version": __version__,
            "workflow": name,
            "error": "unknown workflow",
            "available": COMMAND_WORKFLOW_OPTION_CHOICES,
            "suggestions": suggestions,
            "next_actions": next_actions,
        }
    return {
        "ok": True,
        "name": "mechferret",
        "version": __version__,
        "workflow": workflow["name"],
        **({"workflow_query": name} if str(name).strip() != workflow["name"] else {}),
        "workflow_only": True,
        "count": 1,
        "commands": [],
        "workflows": [workflow],
    }


def _command_choice_names(parser: argparse.ArgumentParser) -> set[str]:
    names: set[str] = set()
    for action in _subparser_actions(parser):
        names.update(str(name) for name in action.choices)
    return names


def _find_command(commands: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    needle = str(query).strip()
    for command in commands:
        names = [command["name"], *command.get("aliases", [])]
        if needle in names:
            return command
    return None


def _filter_commands_by_group(commands: list[dict[str, Any]], group: str | None) -> list[dict[str, Any]]:
    if not group:
        return commands
    names = _command_group_names(group)
    return [command for command in commands if str(command["name"]) in names]


def _command_group_names(group: str) -> set[str]:
    normalized = str(group).strip().lower()
    for title, names in COMMAND_GROUPS:
        if title.lower() == normalized:
            return set(names)
    return set()


def _search_commands(commands: list[dict[str, Any]], search: str) -> list[dict[str, Any]]:
    terms = _search_terms(search)
    if not terms:
        return commands
    scored = [
        (score, index, command)
        for index, command in enumerate(commands)
        if (score := _search_match_score(_command_search_fields(command), terms)) > 0
    ]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [command for _score, _index, command in scored]


def _search_command_workflows(search: str) -> list[dict[str, Any]]:
    terms = _search_terms(search)
    if not terms:
        return _command_workflows_payload()
    scored = [
        (score, index, workflow)
        for index, workflow in enumerate(_command_workflows_payload())
        if (score := _search_match_score(_workflow_search_fields(workflow), terms)) > 0
    ]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [workflow for _score, _index, workflow in scored]


def _search_terms(search: str) -> list[str]:
    normalized = _normalize_search_text(search)
    return [term for term in normalized.split() if term]


def _search_match_score(fields: dict[str, str], terms: list[str]) -> int:
    if not terms:
        return 1
    raw = fields["raw"]
    normalized = fields["normalized"]
    phrase = " ".join(terms)
    if phrase in raw or phrase in normalized:
        return 50 + len(terms)
    if all(term in raw or term in normalized for term in terms):
        return 10 + len(terms)
    return 0


def _command_search_fields(command: dict[str, Any]) -> dict[str, str]:
    text = _command_search_text(command)
    return {"raw": text, "normalized": _normalize_search_text(text)}


def _workflow_search_fields(workflow: dict[str, Any]) -> dict[str, str]:
    text = _workflow_search_text(workflow)
    return {"raw": text, "normalized": _normalize_search_text(text)}


def _normalize_search_text(value: Any) -> str:
    return " ".join(str(value).lower().replace("-", " ").replace("_", " ").split())


def _command_search_text(command: dict[str, Any]) -> str:
    words: list[str] = [
        str(command.get("name", "")),
        *[str(alias) for alias in command.get("aliases", [])],
        str(command.get("help", "")),
        str(command.get("usage", "")),
        *[str(example) for example in command.get("examples", [])],
    ]
    for argument in [*command.get("positionals", []), *command.get("options", [])]:
        words.append(str(argument.get("dest", "")))
        words.append(str(argument.get("help", "")))
        words.extend(str(flag) for flag in argument.get("flags", []))
        words.extend(str(choice) for choice in argument.get("choices", []))
    return " ".join(words).lower()


def _workflow_search_text(workflow: dict[str, Any]) -> str:
    words = [
        str(workflow.get("name", "")),
        str(workflow.get("title", "")),
        str(workflow.get("description", "")),
        *[str(command) for command in workflow.get("commands", [])],
    ]
    return " ".join(words).lower()


def _suggest_commands(commands: list[dict[str, Any]], query: str) -> list[str]:
    by_word: dict[str, str] = {}
    for command in commands:
        primary = str(command["name"])
        by_word[primary] = primary
        for alias in command.get("aliases", []):
            by_word[str(alias)] = primary
    matches = difflib.get_close_matches(str(query), list(by_word), n=8, cutoff=0.55)
    suggestions: list[str] = []
    for match in matches:
        primary = by_word[match]
        if primary not in suggestions:
            suggestions.append(primary)
    return suggestions[:3]


def _find_command_workflow(name: str) -> dict[str, Any] | None:
    needle = _workflow_key(name)
    for workflow in _command_workflows_payload():
        if needle in {_workflow_key(workflow["name"]), _workflow_key(workflow["title"])}:
            return workflow
    return None


def _suggest_command_workflows(query: str) -> list[str]:
    by_key: dict[str, str] = {}
    for workflow in _command_workflows_payload():
        name = str(workflow["name"])
        by_key[_workflow_key(name)] = name
        by_key[_workflow_key(workflow["title"])] = name
    matches = difflib.get_close_matches(_workflow_key(query), list(by_key), n=6, cutoff=0.45)
    suggestions: list[str] = []
    for match in matches:
        name = by_key[match]
        if name not in suggestions:
            suggestions.append(name)
    return suggestions[:3]


def _workflow_key(value: Any) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join(part for part in "".join(chars).split("_") if part)


def _print_command_error(payload: dict[str, Any]) -> None:
    if payload.get("error") == "unknown command" and payload.get("query"):
        print(f"Unknown command: {payload['query']}", file=sys.stderr)
        if payload.get("suggestions"):
            print(f"Did you mean: {', '.join(payload['suggestions'])}?", file=sys.stderr)
    elif payload.get("error") == "unknown workflow" and payload.get("workflow"):
        print(f"Unknown workflow: {payload['workflow']}", file=sys.stderr)
        if payload.get("suggestions"):
            print(f"Did you mean: {', '.join(payload['suggestions'])}?", file=sys.stderr)
    else:
        print(f"Error: {payload.get('error', 'command request failed')}", file=sys.stderr)
    for action in payload.get("next_actions", ["Run `mechferret commands` to list available commands."]):
        print(action, file=sys.stderr)


def _command_scope(*, group: str | None = None, search: str | None = None) -> str:
    parts: list[str] = []
    if group:
        parts.append(f"{group} commands")
    else:
        parts.append("commands")
    if search:
        parts.append(f"matching {search!r}")
    return " ".join(parts)


def _render_command_output(
    payload: dict[str, Any],
    *,
    query: str | None = None,
    search: str | None = None,
    group: str | None = None,
    examples: bool = False,
    markdown: bool = False,
    as_json: bool = False,
) -> tuple[str, str]:
    if as_json:
        return json.dumps(payload, indent=2, sort_keys=True) + "\n", "json"
    if markdown:
        return _command_markdown(payload), "markdown"
    if examples:
        return _command_examples_text(payload["commands"], query=query, search=search, group=group) + "\n", "text"
    if payload.get("workflow_only"):
        if payload.get("workflow_list"):
            return _command_workflows_text(payload["workflows"]) + "\n", "text"
        return _command_workflow_detail_text(payload["workflows"][0]) + "\n", "text"
    if query:
        return _command_detail_text(payload["commands"][0]) + "\n", "text"
    return (
        _command_list_text(
            payload["commands"],
            count=int(payload.get("count", 0)),
            search=search,
            group=group,
            workflows=payload.get("workflows", []),
            workflow_count=int(payload.get("workflow_count", 0)),
        )
        + "\n",
        "text",
    )


def _write_command_output(out_path: str | Path, rendered: str, *, format_name: str, count: int) -> dict[str, Any]:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return {
        "ok": True,
        "path": str(path),
        "format": format_name,
        "bytes": len(rendered.encode("utf-8")),
        "count": count,
    }


def _print_command_detail(command: dict[str, Any]) -> None:
    print(_command_detail_text(command))


def _command_detail_text(command: dict[str, Any]) -> str:
    alias_text = f" ({', '.join(command['aliases'])})" if command["aliases"] else ""
    lines = [f"{command['name']}{alias_text}"]
    if command.get("help"):
        lines.append(f"  {command['help']}")
    lines.append(f"  {command['usage']}")
    positionals = command.get("positionals", [])
    if positionals:
        lines.append("  Positionals:")
        for positional in positionals:
            detail = _argument_detail(positional)
            lines.append(f"    {positional['dest']}{detail}")
    options = command.get("options", [])
    if options:
        lines.append("  Options:")
        for option in options:
            flags = ", ".join(option.get("flags", []))
            detail = _argument_detail(option)
            lines.append(f"    {flags}{detail}")
    examples = command.get("examples", [])
    if examples:
        lines.append("  Examples:")
        for example in examples:
            lines.append(f"    {example}")
    return "\n".join(lines)


def _print_command_groups(commands: list[dict[str, Any]]) -> None:
    print(_command_groups_text(commands))


def _command_groups_text(commands: list[dict[str, Any]]) -> str:
    by_name = {str(command["name"]): command for command in commands}
    printed: set[str] = set()
    lines: list[str] = []
    for title, names in COMMAND_GROUPS:
        rows = [by_name[name] for name in by_name if name in names]
        if not rows:
            continue
        if lines:
            lines.append("")
        lines.append(f"{title}:")
        for command in rows:
            lines.append(_command_summary_text(command))
            printed.add(str(command["name"]))
    remaining = [command for command in commands if str(command["name"]) not in printed]
    if remaining:
        if lines:
            lines.append("")
        lines.append("Other:")
        for command in remaining:
            lines.append(_command_summary_text(command))
    return "\n".join(lines)


def _print_command_summary(command: dict[str, Any]) -> None:
    print(_command_summary_text(command))


def _command_summary_text(command: dict[str, Any]) -> str:
    alias_text = f" ({', '.join(command['aliases'])})" if command["aliases"] else ""
    return f"  {command['name']}{alias_text}: {command['help']}"


def _command_list_text(
    commands: list[dict[str, Any]],
    *,
    count: int,
    search: str | None = None,
    group: str | None = None,
    workflows: list[dict[str, Any]] | None = None,
    workflow_count: int | None = None,
) -> str:
    scope = _command_scope(group=group, search=search)
    if search and workflow_count is not None:
        lines = [f"MechFerret {scope} ({_count_label(count, 'command')}, {_count_label(workflow_count, 'workflow')}):"]
    else:
        lines = [f"MechFerret {scope} ({count}):"]
    if search:
        if commands:
            lines.extend(_command_summary_text(command) for command in commands)
    else:
        grouped = _command_groups_text(commands)
        if grouped:
            lines.extend(["", grouped])
    if workflows:
        lines.extend(["", "Workflows:"])
        for workflow in workflows:
            commands_text = " -> ".join(workflow.get("commands", [])[:4])
            if len(workflow.get("commands", [])) > 4:
                commands_text += " -> ..."
            lines.append(f"  {workflow['name']}: {workflow['description']}")
            lines.append(f"    {commands_text}")
    return "\n".join(lines)


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun if count == 1 else noun + 's'}"


def _print_command_examples(
    commands: list[dict[str, Any]],
    *,
    query: str | None = None,
    search: str | None = None,
    group: str | None = None,
) -> None:
    print(_command_examples_text(commands, query=query, search=search, group=group))


def _command_examples_text(
    commands: list[dict[str, Any]],
    *,
    query: str | None = None,
    search: str | None = None,
    group: str | None = None,
) -> str:
    if query:
        lines = [f"MechFerret examples for {query}:"]
    elif search:
        scope = f"{group} examples" if group else "examples"
        lines = [f"MechFerret {scope} matching {search!r}:"]
    elif group:
        lines = [f"MechFerret {group} examples:"]
    else:
        lines = ["MechFerret examples:"]
    for command in commands:
        lines.extend(["", f"{command['name']}:"])
        for example in command.get("examples", []):
            lines.append(f"  {example}")
    return "\n".join(lines)


def _command_workflow_detail_text(workflow: dict[str, Any]) -> str:
    lines = [f"{workflow['name']}: {workflow['title']}"]
    if workflow.get("description"):
        lines.append(f"  {workflow['description']}")
    lines.append("  Commands:")
    for command in workflow.get("commands", []):
        lines.append(f"    {command}")
    return "\n".join(lines)


def _command_workflows_text(workflows: list[dict[str, Any]]) -> str:
    lines = [f"MechFerret workflows ({len(workflows)}):"]
    for workflow in workflows:
        lines.extend(["", _command_workflow_detail_text(workflow)])
    return "\n".join(lines)


def _command_markdown(payload: dict[str, Any]) -> str:
    commands = payload.get("commands", [])
    title = "MechFerret Examples" if payload.get("examples_only") else "MechFerret Commands"
    if payload.get("workflow_only"):
        title = "MechFerret Workflows" if payload.get("workflow_list") else "MechFerret Workflow"
    lines = [f"# {title}", ""]
    scope = _markdown_scope(payload)
    if scope:
        lines.extend([scope, ""])
    if payload.get("workflow_only"):
        lines.extend(
            _command_workflows_markdown(
                payload.get("workflows", []),
                level=2,
                include_heading=bool(payload.get("workflow_list")),
            )
        )
        return "\n".join(lines).rstrip() + "\n"
    workflows = payload.get("workflows", [])
    if workflows and not payload.get("examples_only") and payload.get("search") and not payload.get("group"):
        lines.extend(_command_workflows_markdown(workflows))
    if not commands:
        if not workflows:
            lines.append("_No commands matched._")
        return "\n".join(lines).rstrip() + "\n"
    if workflows and not payload.get("examples_only") and not payload.get("query") and not payload.get("search") and not payload.get("group"):
        lines.extend(_command_workflows_markdown(workflows))
    if payload.get("examples_only"):
        for command in commands:
            lines.extend(_command_examples_markdown(command))
    elif payload.get("query") and len(commands) == 1:
        lines.extend(_command_detail_markdown(commands[0]))
    else:
        for group_title, grouped in _group_commands(commands):
            lines.append(f"## {group_title}")
            lines.append("")
            for command in grouped:
                lines.extend(_command_detail_markdown(command, level=3))
    return "\n".join(lines).rstrip() + "\n"


def _markdown_scope(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    if payload.get("group"):
        parts.append(f"group: `{payload['group']}`")
    if payload.get("search"):
        parts.append(f"search: `{payload['search']}`")
    if payload.get("query"):
        parts.append(f"command: `{payload['query']}`")
    if payload.get("workflow"):
        parts.append(f"workflow: `{payload['workflow']}`")
    return "_Filtered by " + ", ".join(parts) + "._" if parts else ""


def _group_commands(commands: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    by_name = {str(command["name"]): command for command in commands}
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    printed: set[str] = set()
    for title, names in COMMAND_GROUPS:
        rows = [by_name[name] for name in by_name if name in names]
        if rows:
            grouped.append((title, rows))
            printed.update(str(row["name"]) for row in rows)
    remaining = [command for command in commands if str(command["name"]) not in printed]
    if remaining:
        grouped.append(("Other", remaining))
    return grouped


def _command_detail_markdown(command: dict[str, Any], *, level: int = 2) -> list[str]:
    aliases = _alias_markdown(command)
    marker = "#" * max(1, level)
    lines = [f"{marker} `{command['name']}`", ""]
    if aliases:
        lines.extend([f"Aliases: {aliases}", ""])
    if command.get("help"):
        lines.extend([str(command["help"]), ""])
    lines.extend(["Usage:", "", "```text", str(command.get("usage", "")), "```", ""])
    if command.get("positionals"):
        lines.extend(["Positionals:", ""])
        for positional in command["positionals"]:
            lines.append(f"- `{positional['dest']}`{_markdown_argument_detail(positional)}")
        lines.append("")
    if command.get("options"):
        lines.extend(["Options:", ""])
        for option in command["options"]:
            flags = ", ".join(f"`{flag}`" for flag in option.get("flags", []))
            lines.append(f"- {flags}{_markdown_argument_detail(option)}")
        lines.append("")
    if command.get("examples"):
        lines.extend(["Examples:", ""])
        for example in command["examples"]:
            lines.append(f"- `{example}`")
        lines.append("")
    return lines


def _command_examples_markdown(command: dict[str, Any]) -> list[str]:
    lines = [f"## `{command['name']}`", ""]
    for example in command.get("examples", []):
        lines.append(f"- `{example}`")
    lines.append("")
    return lines


def _command_workflows_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": str(workflow["name"]),
            "title": str(workflow["title"]),
            "description": str(workflow["description"]),
            "commands": list(workflow["commands"]),
        }
        for workflow in COMMAND_WORKFLOWS
    ]


def _command_workflows_markdown(
    workflows: list[dict[str, Any]],
    *,
    level: int = 2,
    include_heading: bool = True,
) -> list[str]:
    marker = "#" * max(1, level)
    lines = [f"{marker} Workflows", ""] if include_heading else []
    for workflow in workflows:
        title_marker = f"{marker}#" if include_heading else marker
        lines.extend([f"{title_marker} {workflow['title']}", "", str(workflow.get("description", "")), ""])
        for command in workflow.get("commands", []):
            lines.append(f"- `{command}`")
        lines.append("")
    return lines


def _alias_markdown(command: dict[str, Any]) -> str:
    return ", ".join(f"`{alias}`" for alias in command.get("aliases", []))


def _markdown_argument_detail(argument: dict[str, Any]) -> str:
    parts: list[str] = []
    if argument.get("choices"):
        parts.append("choices: " + ", ".join(f"`{choice}`" for choice in argument["choices"]))
    if argument.get("help"):
        parts.append(str(argument["help"]))
    return ": " + "; ".join(parts) if parts else ""


def _argument_detail(argument: dict[str, Any]) -> str:
    parts: list[str] = []
    if argument.get("choices"):
        parts.append("choices: " + ", ".join(argument["choices"]))
    if argument.get("help"):
        parts.append(str(argument["help"]))
    return ": " + "; ".join(parts) if parts else ""


def _completion_payload(parser: argparse.ArgumentParser, shell: str, *, executable: str = "mechferret") -> dict[str, Any]:
    index = _command_index_payload(parser)
    executable = str(executable or "mechferret")
    if shell == "bash":
        script = _bash_completion_script(index["commands"], executable)
        hint = f"mechferret completion bash > ~/.local/share/bash-completion/completions/{executable}"
    elif shell == "zsh":
        script = _zsh_completion_script(index["commands"], executable)
        hint = f"mechferret completion zsh > ~/.zfunc/_{executable} && fpath=(~/.zfunc $fpath)"
    elif shell == "fish":
        script = _fish_completion_script(index["commands"], executable)
        hint = f"mechferret completion fish > ~/.config/fish/completions/{executable}.fish"
    else:
        raise ValueError(f"Unsupported shell: {shell}")
    return {
        "ok": True,
        "shell": shell,
        "command": executable,
        "script": script,
        "install_hint": hint,
    }


def _bash_completion_script(commands: list[dict[str, Any]], executable: str) -> str:
    names = _completion_words(commands)
    fn_name = _shell_identifier(executable)
    lines = [
        f"# bash completion for {executable}",
        f"_{fn_name}_completion() {{",
        "    local cur first commands opts",
        "    COMPREPLY=()",
        '    cur="${COMP_WORDS[COMP_CWORD]}"',
        '    first="${COMP_WORDS[1]}"',
        f"    commands={shlex.quote(' '.join(names))}",
        "    if [[ ${COMP_CWORD} -eq 1 ]]; then",
        '        COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )',
        "        return 0",
        "    fi",
        '    case "${first}" in',
    ]
    for command in commands:
        patterns = _command_patterns(command)
        match_words = _command_completion_words(command)
        lines.extend(
            [
                f"        {'|'.join(patterns)})",
                f"            opts={shlex.quote(' '.join(match_words))}",
                '            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )',
                "            ;;",
            ]
        )
    lines.extend(
        [
            "    esac",
            "}",
            f"complete -F _{fn_name}_completion {shlex.quote(executable)}",
            "",
        ]
    )
    return "\n".join(lines)


def _zsh_completion_script(commands: list[dict[str, Any]], executable: str) -> str:
    fn_name = _shell_identifier(executable)
    lines = [
        f"#compdef {executable}",
        f"_{fn_name}_completion() {{",
        "    local -a commands matches",
        "    commands=(",
    ]
    for command in commands:
        for name in _command_patterns(command):
            text = f"{name}:{command.get('help', '')}"
            lines.append(f"        {_single_quote(text.replace(':', '\\:'))}")
    lines.extend(
        [
            "    )",
            "    if (( CURRENT == 2 )); then",
            "        _describe 'command' commands",
            "        return",
            "    fi",
            "    case \"${words[2]}\" in",
        ]
    )
    for command in commands:
        match_words = _command_completion_words(command)
        if not match_words:
            continue
        lines.extend(
            [
                f"        {'|'.join(_command_patterns(command))})",
                "            matches=(",
                *[f"                {_single_quote(word)}" for word in match_words],
                "            )",
                "            compadd -- $matches",
                "            ;;",
            ]
        )
    lines.extend(
        [
            "    esac",
            "}",
            f"compdef _{fn_name}_completion {shlex.quote(executable)}",
            "",
        ]
    )
    return "\n".join(lines)


def _fish_completion_script(commands: list[dict[str, Any]], executable: str) -> str:
    exe = shlex.quote(executable)
    lines = [f"# fish completion for {executable}"]
    for command in commands:
        help_text = str(command.get("help", ""))
        for name in _command_patterns(command):
            lines.append(
                f"complete -c {exe} -f -n '__fish_use_subcommand' -a {shlex.quote(name)} -d {shlex.quote(help_text)}"
            )
        condition = _fish_subcommand_condition(command)
        seen_long: set[str] = set()
        seen_short: set[str] = set()
        for option in command.get("options", []):
            desc = str(option.get("help", ""))
            for flag in option.get("flags", []):
                if flag.startswith("--"):
                    long_name = flag[2:]
                    if long_name not in seen_long:
                        lines.append(
                            f"complete -c {exe} -n {shlex.quote(condition)} -l {shlex.quote(long_name)} -d {shlex.quote(desc)}"
                        )
                        seen_long.add(long_name)
                elif flag.startswith("-") and len(flag) == 2:
                    short_name = flag[1:]
                    if short_name not in seen_short:
                        lines.append(
                            f"complete -c {exe} -n {shlex.quote(condition)} -s {shlex.quote(short_name)} -d {shlex.quote(desc)}"
                        )
                        seen_short.add(short_name)
        for word in _choice_words(command):
            lines.append(f"complete -c {exe} -f -n {shlex.quote(condition)} -a {shlex.quote(word)}")
    lines.append("")
    return "\n".join(lines)


def _completion_words(commands: list[dict[str, Any]]) -> list[str]:
    words: list[str] = []
    for command in commands:
        words.append(str(command["name"]))
        words.extend(str(alias) for alias in command.get("aliases", []))
    return words


def _option_words(command: dict[str, Any]) -> list[str]:
    words: list[str] = []
    for option in command.get("options", []):
        words.extend(str(flag) for flag in option.get("flags", []))
    return words


def _choice_words(command: dict[str, Any]) -> list[str]:
    words: list[str] = []
    for argument in [*command.get("positionals", []), *command.get("options", [])]:
        words.extend(str(choice) for choice in argument.get("choices", []))
    return _unique_words(words)


def _command_completion_words(command: dict[str, Any]) -> list[str]:
    return _unique_words([*_option_words(command), *_choice_words(command)])


def _command_patterns(command: dict[str, Any]) -> list[str]:
    return [str(command["name"]), *[str(alias) for alias in command.get("aliases", [])]]


def _fish_subcommand_condition(command: dict[str, Any]) -> str:
    return "__fish_seen_subcommand_from " + " ".join(_command_patterns(command))


def _unique_words(words: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        result.append(word)
    return result


def _shell_identifier(value: str) -> str:
    chars = [char if char.isalnum() else "_" for char in str(value)]
    ident = "".join(chars).strip("_") or "mechferret"
    return ident.lower()


def _single_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def _subparser_actions(parser: argparse.ArgumentParser) -> list[argparse._SubParsersAction]:
    return [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]


def _command_help(action: argparse._SubParsersAction, name: str) -> str:
    for choice in action._choices_actions:
        if choice.dest == name:
            return str(choice.help or "")
    return ""


def _argument_payload(action: argparse.Action) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dest": action.dest,
        "help": str(action.help or ""),
        "required": bool(getattr(action, "required", False)),
    }
    if action.option_strings:
        payload["flags"] = list(action.option_strings)
    if action.dest == "workflow":
        payload["choices"] = list(COMMAND_WORKFLOW_OPTION_CHOICES)
    elif getattr(action, "choices", None) is not None:
        payload["choices"] = [str(choice) for choice in action.choices]
    if action.nargs is not None:
        payload["nargs"] = str(action.nargs)
    default = getattr(action, "default", None)
    if default not in (None, argparse.SUPPRESS):
        payload["default"] = _json_ready(default)
    return payload


def _budget_override(args) -> Budget | None:
    if not any(
        getattr(args, name, None) is not None
        for name in ("max_rounds", "max_experiments", "max_gpu_seconds")
    ):
        return None
    base = Budget()
    return Budget(
        max_experiments=args.max_experiments if args.max_experiments is not None else base.max_experiments,
        max_rounds=args.max_rounds if args.max_rounds is not None else base.max_rounds,
        max_gpu_seconds=args.max_gpu_seconds if args.max_gpu_seconds is not None else base.max_gpu_seconds,
    )


def handle_skills(args) -> None:
    if args.name:
        skill = load_skill(args.name)
        if args.json:
            print(json.dumps({"ok": True, "skill": _skill_payload(skill)}, indent=2, sort_keys=True))
            return
        print(f"Skill: {skill.name}")
        print(f"Description: {skill.description}")
        print(f"Task: {skill.task}  Model: {skill.model}")
        print(f"Question: {skill.question}")
        print(f"Screen heads: {skill.max_screen_heads}  Promote top-k: {skill.promote_top_k}  Seeds: {skill.seeds}")
        print(f"Budget: {skill.budget}")
        print(f"Stop when: confirmed>={skill.min_confirmed}, rigor>={skill.min_rigor}")
        for reference in skill.references:
            print(f"  ref: {reference}")
        return
    skills = list_skills()
    if args.json:
        print(
            json.dumps(
                {"ok": True, "count": len(skills), "skills": [_skill_payload(skill) for skill in skills]},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not skills:
        print("No skills found.")
        return
    print(f"{len(skills)} interpretability skills:")
    for skill in skills:
        print(f"  {skill.name:24} [{skill.task}] {skill.description}")


def _skill_payload(skill) -> dict[str, Any]:
    return {
        "name": skill.name,
        "description": skill.description,
        "task": skill.task,
        "model": skill.model,
        "question": skill.question,
        "max_screen_heads": skill.max_screen_heads,
        "promote_top_k": skill.promote_top_k,
        "seeds": skill.seeds,
        "budget": skill.budget,
        "stop": skill.stop,
        "references": skill.references,
        "min_confirmed": skill.min_confirmed,
        "min_rigor": skill.min_rigor,
    }


def handle_modal(args) -> None:
    from .modal_app import dispatch_discovery, modal_status

    status = modal_status()
    if args.action == "status":
        if args.json:
            print(json.dumps(_modal_payload("status", status), indent=2, sort_keys=True))
            return
        print(f"Modal installed:       {status['installed']}")
        print(f"Modal authenticated:   {status['authenticated']}")
        print(f"GPU type:              {status['gpu']}")
        print(f"Local torch:           {status['torch_local']}")
        print(f"Local transformer_lens:{status['transformer_lens_local']}")
        if not status["installed"]:
            print("\nInstall with: pip install -e '.[modal]'")
        elif not status["authenticated"]:
            print("\nAuthenticate with: modal token new")
        else:
            print("\nReady. Run: mechferret /modal run --skill ioi-circuit")
        return
    if args.action == "setup":
        if args.json:
            print(json.dumps(_modal_payload("setup", status), indent=2, sort_keys=True))
            return
        print("Modal setup steps:")
        print("  1. pip install -e '.[modal,interp]'")
        print("  2. modal token new            # browser auth")
        print("  3. (optional) modal secret create openai-api-key OPENAI_API_KEY=sk-...")
        print("  4. mechferret /modal run --skill ioi-circuit")
        print(f"\nCurrent status: installed={status['installed']} authenticated={status['authenticated']}")
        return
    if args.action == "deploy":
        if args.json:
            print(json.dumps(_modal_payload("deploy", status), indent=2, sort_keys=True))
            return
        print("Deploy the GPU app with:\n  modal deploy mechferret/modal_app.py")
        print(f"App name: {status['app']} (gpu={status['gpu']})")
        return
    # action == "run"
    skill = args.skill or (None if (args.question or args.task) else "ioi-circuit")
    print(f"Dispatching discovery to Modal (skill={skill}, task={args.task}, model={args.model})...")
    result = dispatch_discovery(
        question=args.question, skill=skill, task=args.task, model=args.model, out_dir=args.out
    )
    if args.json:
        print(json.dumps(_dispatch_payload("modal", result, skill=skill, task=args.task, model=args.model), indent=2, sort_keys=True))
        return
    print(f"Executed on: {result['backend']} backend")
    if result.get("note"):
        print(result["note"])
    payload = result["run"]
    metrics = payload.get("metrics", {})
    print(f"Discoveries: {len(payload.get('discoveries', []))}")
    print(f"Readiness: {metrics.get('readiness_score', 0)}")
    if "modal_gpu_seconds" in metrics:
        print(f"Modal GPU seconds: {metrics['modal_gpu_seconds']}")
    print(f"Artifacts under: {result['out_dir']}")


def handle_cluster(args) -> None:
    from .cluster import cluster_status, dispatch_discovery_cluster, load_cluster_config

    if args.action == "status":
        status = cluster_status()
        if args.json:
            print(json.dumps(_cluster_payload("status", status), indent=2, sort_keys=True))
            return
        print(f"Configured:   {status['configured']}")
        print(f"Host:         {status['host'] or '(unset: REMOTE_HOST)'}")
        print(f"SSH reachable:{status['ssh_ok']}")
        print(f"Partition:    {status['partition'] or '(unset: SLURM_PARTITION)'}")
        print(f"GRES/GPU:     {status['gres']}  CPUs: {status['cpus']}  Mem: {status['mem']}  Time: {status['time']}")
        print(f"Project dir:  {status['remote_project_dir'] or '(unset: REMOTE_PROJECT_DIR)'}")
        print(f"Env setup:    {status['remote_setup'] or '(unset: REMOTE_RUN_SETUP)'}")
        if not status["configured"]:
            print("\nRun `mechferret /cluster setup` for connection steps.")
        return
    if args.action == "setup":
        if args.json:
            status = cluster_status(load_cluster_config())
            print(json.dumps(_cluster_payload("setup", status), indent=2, sort_keys=True))
            return
        _print_cluster_setup()
        return
    # action == "run"
    skill = args.skill or (None if (args.question or args.task) else "ioi-circuit")
    if args.dry_run:
        result = dispatch_discovery_cluster(
            question=args.question, skill=skill, task=args.task, model=args.model, out_dir=args.out, dry_run=True
        )
        if args.json:
            print(json.dumps(_dispatch_payload("cluster", result, skill=skill, task=args.task, model=args.model), indent=2, sort_keys=True))
            return
        print("Dry run -- command that would execute:\n")
        print(result["command"])
        return
    print(f"Dispatching discovery to cluster (skill={skill}, task={args.task}, model={args.model})...")
    result = dispatch_discovery_cluster(
        question=args.question, skill=skill, task=args.task, model=args.model, out_dir=args.out
    )
    if args.json:
        print(json.dumps(_dispatch_payload("cluster", result, skill=skill, task=args.task, model=args.model), indent=2, sort_keys=True))
        return
    print(f"Executed on: {result['backend']} backend")
    if result.get("note"):
        print(result["note"])
    payload = result.get("run", {})
    if payload:
        print(f"Discoveries: {len(payload.get('discoveries', []))}")
        print(f"Readiness: {payload.get('metrics', {}).get('readiness_score', 0)}")
    print(f"Artifacts under: {result['out_dir']}")


def _modal_payload(action: str, status: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "action": action,
        "status": status,
    }
    if action == "status":
        if not status.get("installed"):
            payload["next_action"] = "install"
        elif not status.get("authenticated"):
            payload["next_action"] = "authenticate"
        else:
            payload["next_action"] = "run"
    elif action == "setup":
        payload["steps"] = [
            "pip install -e '.[modal,interp]'",
            "modal token new",
            "modal secret create openai-api-key OPENAI_API_KEY=<redacted>",
            "mechferret /modal run --skill ioi-circuit",
        ]
    elif action == "deploy":
        payload["command"] = "modal deploy mechferret/modal_app.py"
        payload["app"] = status.get("app")
        payload["gpu"] = status.get("gpu")
    return payload


def _cluster_payload(action: str, status: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "action": action,
        "status": status,
        "configured": bool(status.get("configured")),
    }
    if action == "status":
        payload["next_action"] = "dry_run" if status.get("configured") else "setup"
    elif action == "setup":
        payload["env"] = [
            "REMOTE_HOST",
            "REMOTE_PROJECT_DIR",
            "SLURM_PARTITION",
            "SLURM_GRES",
            "SLURM_CPUS",
            "SLURM_MEM",
            "SLURM_TIME",
            "REMOTE_RUN_SETUP",
        ]
        payload["steps"] = [
            "ssh -o BatchMode=yes <your-ssh-alias> 'echo ok'",
            "cd <remote-project-dir> && pip install -e '.[interp]'",
            "export REMOTE_HOST=<your-ssh-alias>",
            "export REMOTE_PROJECT_DIR=<remote-project-dir>",
            "mechferret /cluster run --skill ioi-circuit --dry-run",
        ]
    return payload


def _dispatch_payload(kind: str, result: dict[str, Any], *, skill: str | None, task: str | None, model: str) -> dict[str, Any]:
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    metrics = run.get("metrics", {}) if isinstance(run, dict) else {}
    payload: dict[str, Any] = {
        "ok": True,
        "kind": kind,
        "action": "run",
        "backend": result.get("backend"),
        "dry_run": bool(result.get("dry_run")),
        "skill": skill,
        "task": task,
        "model": model,
        "out_dir": result.get("out_dir"),
        "result": result,
    }
    if run:
        payload["summary"] = {
            "discoveries": len(run.get("discoveries", [])),
            "readiness": metrics.get("readiness_score", 0),
        }
    if result.get("note"):
        payload["note"] = result.get("note")
    return payload


def _print_cluster_setup() -> None:
    print("Connect a SLURM cluster (works with any cluster that has srun + non-interactive SSH):\n")
    print("  1. Make non-interactive SSH to your login node work:")
    print("       ssh -o BatchMode=yes <your-ssh-alias> 'echo ok'")
    print("  2. Install MechFerret on the cluster (in your project dir, in your env):")
    print("       ssh <your-ssh-alias>")
    print("       cd <remote-project-dir> && pip install -e '.[interp]'")
    print("  3. Point MechFerret at the cluster via env vars (or .mechferret/cluster.json):")
    print("       export REMOTE_HOST=<your-ssh-alias>")
    print("       export REMOTE_PROJECT_DIR=<remote-project-dir>")
    print("       export SLURM_PARTITION=<partition>     # e.g. gpu")
    print("       export SLURM_GRES=gpu:1                # e.g. gpu:a100:1")
    print("       export SLURM_CPUS=8  SLURM_MEM=32G  SLURM_TIME=02:00:00")
    print("       export REMOTE_RUN_SETUP='source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>'")
    print("       # optional: export REMOTE_GIT_PULL=1   # git pull --ff-only before each run")
    print("  4. Verify, dry-run, then run:")
    print("       mechferret /cluster status")
    print("       mechferret /cluster run --skill ioi-circuit --dry-run")
    print("       mechferret /cluster run --skill ioi-circuit")
    print("\n  MechFerret will: ssh -> srun (with your flags) -> `mechferret discover --backend transformer_lens`")
    print("  on a compute node, then scp the dossier back to your --out directory.")


def print_discovery_summary(run) -> None:
    print(f"Run: {run.run_id} (mode={run.mode})")
    print(f"Readiness score: {run.metrics.get('readiness_score', 0):.2f}  rigor: {run.metrics.get('rigor_score', 0):.2f}")
    print(f"Experiments ran: {int(run.metrics.get('experiments_run', 0))} over {int(run.metrics.get('rounds_run', 0))} round(s)")
    print(f"Confirmed mechanisms: {len(run.discoveries)}")
    for discovery in run.discoveries:
        print(f"  - {discovery.statement}")
        print(f"      confidence={discovery.confidence:.2f} effect={discovery.effect_size:.2f} "
              f"reproducibility={discovery.reproducibility:.2f} novelty={discovery.novelty:.2f}")
    print(f"Report: {run.artifacts.get('html')}")
    print(f"Discoveries JSON: {run.artifacts.get('discoveries')}")
    print(f"Experiments JSON: {run.artifacts.get('experiments')}")
    print(f"Trace: {run.artifacts.get('trace')}")


def handle_api_command(args) -> None:
    config = load_config()
    if args.clear:
        config.providers.pop(args.clear, None)
        if config.default_provider == args.clear:
            config.default_provider = "local"
        path = save_config(config)
        if args.json:
            print(json.dumps(_api_payload(config, action="clear", path=path, provider=args.clear), indent=2, sort_keys=True))
            return
        print(f"Cleared {args.clear} credentials in {path}")
        return
    if args.provider:
        if args.provider == "local":
            config.default_provider = "local"
            path = save_config(config)
            if args.json:
                print(json.dumps(_api_payload(config, action="set-default", path=path, provider="local"), indent=2, sort_keys=True))
                return
            print(f"Default provider: local ({path})")
            return
        settings = config.provider(args.provider)
        if args.api_key:
            settings.api_key = args.api_key
        if args.model:
            settings.model = args.model
        if args.api_key or args.model:
            config.default_provider = args.provider
            path = save_config(config)
            if args.json:
                print(json.dumps(_api_payload(config, action="update", path=path, provider=args.provider), indent=2, sort_keys=True))
                return
            print(f"Updated {args.provider} in {path}")
            return
        config.default_provider = args.provider
        path = save_config(config)
        if args.json:
            print(json.dumps(_api_payload(config, action="set-default", path=path, provider=args.provider), indent=2, sort_keys=True))
            return
        print(f"Default provider: {args.provider} ({path})")
        return
    if args.show or not any([args.provider, args.api_key, args.model, args.clear]):
        if args.json:
            print(json.dumps(_api_payload(config, action="show", path=default_config_path()), indent=2, sort_keys=True))
            return
        print(f"Default provider: {config.default_provider}")
        for provider in sorted(PROVIDERS):
            settings = config.providers.get(provider)
            key_state = "configured" if settings and settings.api_key else "missing"
            model = settings.model if settings and settings.model else "default"
            print(f"{provider}: key={key_state}, model={model}")
        return
    if args.json:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "--api-key and --model require --provider",
                    "next_actions": ["Pass --provider openai or --provider anthropic with --api-key/--model."],
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
    print("--api-key and --model require --provider", file=sys.stderr)
    raise SystemExit(2)


def _api_payload(config, *, action: str, path: str | Path, provider: str = "") -> dict[str, Any]:
    providers = {}
    for name in sorted(PROVIDERS):
        settings = config.providers.get(name)
        providers[name] = {
            "key": "configured" if settings and settings.api_key else "missing",
            "model": settings.model if settings and settings.model else "default",
        }
    return {
        "ok": True,
        "action": action,
        "provider": provider,
        "default_provider": config.default_provider,
        "config_path": str(path),
        "providers": providers,
    }


def _run_payload(run, *, command: str) -> dict[str, Any]:
    metrics = run.metrics if isinstance(run.metrics, dict) else {}
    artifacts = run.artifacts if isinstance(run.artifacts, dict) else {}
    return {
        "ok": True,
        "command": command,
        "run_id": run.run_id,
        "mode": run.mode,
        "question": run.question,
        "created_at": run.created_at,
        "path": artifacts.get("json", ""),
        "artifacts": artifacts,
        "metrics": metrics,
        "provenance": run.provenance,
        "summary": {
            "sources": len(run.sources),
            "claims": len(run.claims),
            "evidence": len(run.evidence),
            "gaps": len(run.gaps),
            "contradictions": len(run.contradictions),
            "hypotheses": len(run.hypotheses),
            "experiments": len(run.experiments),
            "discoveries": len(run.discoveries),
            "readiness": metrics.get("readiness_score", 0),
            "rigor": metrics.get("rigor_score", 0),
        },
    }


def _error_payload(command: str, exc: Exception, *, out_dir: str | Path = "") -> dict[str, Any]:
    return {
        "ok": False,
        "command": command,
        "error": str(exc),
        "out_dir": str(out_dir),
        "next_actions": [
            "Add --source or --url for literature grounding.",
            "Pass --seed-corpus when you intentionally want the packaged demo corpus.",
            "Use --provider openai or --provider anthropic when configured provider research is desired.",
        ],
    }


def _passed_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    if "passed" in payload and "ok" not in payload:
        payload["ok"] = bool(payload["passed"])
    return payload


def _artifact_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    if "exists" in payload and "ok" not in payload:
        payload["ok"] = bool(payload["exists"])
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        payload["artifacts"] = {
            name: _artifact_payload(item) if isinstance(item, dict) else item
            for name, item in artifacts.items()
        }
    return payload


def print_summary(run) -> None:
    print(f"Run: {run.run_id}")
    print(f"Readiness score: {run.metrics.get('readiness_score', 0):.2f}")
    print(f"Claims: {len(run.claims)}")
    print(f"Evidence chunks: {len(run.evidence)}")
    print(f"Report: {run.artifacts.get('html')}")
    print(f"Graph: {run.artifacts.get('graph')}")
    print(f"Evals: {run.artifacts.get('evals')}")
    print(f"Trace: {run.artifacts.get('trace')}")


def print_tool_results(result: dict) -> None:
    if "results" in result:
        print(f"Tool results: {result['count']} saved ({result['root']})")
        for row in result.get("results", []):
            kind = "json" if row.get("is_json") else "text"
            print(f"{row['tool']:20} {row['bytes']:8} bytes {kind:4} {row['path']}")
    else:
        action = "Would delete" if result.get("dry_run") else "Deleted"
        rows = result.get("would_delete", []) if result.get("dry_run") else result.get("deleted", [])
        print(f"Tool results cleanup: {'DRY RUN' if result.get('dry_run') else 'DONE'} ({result['root']})")
        print(f"{action}: {len(rows)}")
        for row in rows[:20]:
            suffix = f" error={row['error']}" if row.get("error") else ""
            print(f"  - {row['path']} ({row['bytes']} bytes){suffix}")
        kept = result.get("kept", [])
        print(f"Kept: {len(kept)}")
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"]:
            print(f"  - {action}")


def _resolve_run_json_arg(run_json: str | None, runs_root: str | Path = "runs", selection: str = "latest") -> Path:
    if run_json:
        path = Path(run_json)
        if path.is_file():
            return path
        print(f"Run artifact not found: {path}", file=sys.stderr)
        print("Run `mechferret quickstart --run` or pass an existing run.json path.", file=sys.stderr)
        raise SystemExit(1)
    if selection != "latest":
        selected = select_run_artifact(runs_root=runs_root, policy=selection)
        if selected.get("path"):
            return Path(selected["path"])
        for action in selected.get("next_actions", []):
            print(action, file=sys.stderr)
        raise SystemExit(1)
    latest = latest_run_json(runs_root)
    if latest is None:
        print(f"No run artifact found under {runs_root}/**/run.json", file=sys.stderr)
        raise SystemExit(1)
    return latest


def _resolve_run_json_for_json(
    run_json: str | None,
    runs_root: str | Path = "runs",
    selection: str = "latest",
) -> tuple[Path | None, dict[str, Any] | None]:
    if run_json:
        path = Path(run_json)
        if path.is_file():
            return path, None
        return None, {
            "ok": False,
            "error": "run artifact not found",
            "path": str(path),
            "runs_root": str(runs_root),
            "selection": selection,
            "next_actions": ["Run `mechferret quickstart --run` or pass an existing run.json path."],
        }
    if selection != "latest":
        selected = select_run_artifact(runs_root=runs_root, policy=selection)
        if selected.get("path"):
            return Path(selected["path"]), None
        return None, {
            "ok": False,
            "error": "no run selected",
            "runs_root": str(runs_root),
            "selection": selection,
            "failed_check": selected.get("failed_check", ""),
            "next_actions": selected.get("next_actions", []) or ["Run `mechferret quickstart --run` to create a dossier."],
        }
    latest = latest_run_json(runs_root)
    if latest is not None:
        return latest, None
    return None, {
        "ok": False,
        "error": "no run artifact found",
        "runs_root": str(runs_root),
        "selection": selection,
        "next_actions": ["Run `mechferret quickstart --run` or `mechferret run ... --out runs/<name>` first."],
    }
