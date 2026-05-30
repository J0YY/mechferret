from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROVIDERS, configure_provider, load_config, prompt_api_key, save_config
from .controller import MechFerret
from .costs import estimate_run_cost
from .discovery import DiscoveryController
from .goal_loop import GoalLoop
from .hooks import Budget
from .ops import memory_clear, memory_recent, memory_summary, print_doctor, summarize_run_artifact
from .registry import all_items, items_by_kind
from .skills import list_skills, load_skill
from .sources import example_corpus_path

DEMO_QUESTION = (
    "What should a team build to win an autoresearch systems hackathon, "
    "and what reliability risks must the implementation address?"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechferret", description="Run autonomous research loops with citations.")
    sub = parser.add_subparsers(dest="command", required=True)

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

    demo = sub.add_parser("demo", help="Run the built-in hackathon demo corpus.")
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--db", default=".mechferret/memory.sqlite")
    demo.add_argument("--max-rounds", type=int, default=2)
    demo.add_argument("--openai", action="store_true")
    demo.add_argument("--provider", choices=["auto", "local", "openai", "anthropic"], default="local")
    demo.add_argument("--model", help="Override the configured provider model.")
    demo.add_argument("--with-memory", action="store_true", help="Recall prior-run memory during the demo.")

    login = sub.add_parser("login", aliases=["/login"], help="Store an OpenAI or Anthropic API key.")
    login.add_argument("provider", choices=sorted(PROVIDERS))
    login.add_argument("--api-key", help="API key. If omitted, MechFerret prompts securely.")
    login.add_argument("--model", help="Default model for this provider.")
    login.add_argument("--no-default", action="store_true", help="Store key without making this the default provider.")

    api = sub.add_parser("api", aliases=["/api"], help="Show or change provider configuration.")
    api.add_argument("--provider", choices=sorted(PROVIDERS) + ["local"], help="Set default provider.")
    api.add_argument("--api-key", help="Store or replace the key for --provider.")
    api.add_argument("--model", help="Store or replace the default model for --provider.")
    api.add_argument("--show", action="store_true", help="Show configured provider status.")
    api.add_argument("--clear", choices=sorted(PROVIDERS), help="Remove a stored provider key.")

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

    skills_cmd = sub.add_parser("skills", aliases=["/skills"], help="List or show interpretability skills/playbooks.")
    skills_cmd.add_argument("name", nargs="?", help="Show details for one skill.")

    modal_cmd = sub.add_parser("modal", aliases=["/modal"], help="Connect to Modal for GPU compute and run experiments remotely.")
    modal_cmd.add_argument("action", nargs="?", default="status", choices=["status", "setup", "run", "deploy"])
    modal_cmd.add_argument("question", nargs="?", default="")
    modal_cmd.add_argument("--skill", help="Skill to run remotely (e.g. ioi-circuit).")
    modal_cmd.add_argument("--task", choices=["ioi", "induction", "greater_than", "factual_recall"])
    modal_cmd.add_argument("--model", default="gpt2")
    modal_cmd.add_argument("--out", default="runs/modal")

    doctor = sub.add_parser("doctor", aliases=["/doctor"], help="Check config, packages, corpus, and registry health.")
    doctor.set_defaults(_doctor=True)

    registry = sub.add_parser("registry", aliases=["/registry"], help="List available tools, tasks, playbooks, and evaluators.")
    registry.add_argument("--kind", choices=["tool", "task", "playbook", "evaluator"])

    memory = sub.add_parser("memory", aliases=["/memory"], help="Inspect or clear research memory.")
    memory.add_argument("--db", default=".mechferret/memory.sqlite")
    memory.add_argument("--recent", type=int, default=0, help="Show recent remembered runs.")
    memory.add_argument("--clear", action="store_true", help="Delete the memory database.")

    cost = sub.add_parser("cost", aliases=["/cost"], help="Estimate cost/usage from a run artifact.")
    cost.add_argument("run_json")

    resume = sub.add_parser("resume", aliases=["/resume"], help="Summarize a prior run artifact.")
    resume.add_argument("run_json")

    inspect = sub.add_parser("inspect", help="Print a compact summary of a run JSON artifact.")
    inspect.add_argument("run_json")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        engine = MechFerret(args.db)
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
        )
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
        )
        print_summary(run)
    elif args.command in {"login", "/login"}:
        key = args.api_key or prompt_api_key(args.provider)
        if not key:
            raise SystemExit("No API key provided.")
        path = configure_provider(
            args.provider,
            key,
            model=args.model,
            make_default=not args.no_default,
        )
        print(f"Stored {args.provider} credentials in {path}")
        print(f"Default provider: {load_config().default_provider}")
    elif args.command in {"api", "/api"}:
        handle_api_command(args)
    elif args.command in {"goal", "/goal", "loop", "/loop"}:
        loop = GoalLoop(args.db)
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
        )
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
    elif args.command in {"doctor", "/doctor"}:
        print_doctor()
    elif args.command in {"registry", "/registry"}:
        items = items_by_kind(args.kind) if args.kind else all_items()
        for item in items:
            print(f"{item.kind:9} {item.name:24} {item.status:10} {item.description}")
    elif args.command in {"memory", "/memory"}:
        if args.clear:
            memory_clear(args.db)
            print(f"Cleared memory at {args.db}")
            return
        summary = memory_summary(args.db)
        print(f"Memory: runs={summary['runs']} claims={summary['claims']} sources={summary['sources']}")
        if args.recent:
            for row in memory_recent(args.db, args.recent):
                score = row["metrics"].get("readiness_score", 0)
                print(f"{row['created_at']} {row['id']} readiness={score:.2f} {row['question'][:90]}")
    elif args.command in {"cost", "/cost"}:
        cost = estimate_run_cost(args.run_json)
        print(f"Run: {cost['run_id']}")
        print(f"Estimated tokens processed: {cost['estimated_tokens_processed']}")
        print(f"Estimated provider calls: {cost['estimated_provider_calls']}")
        print(f"Local plan steps: {cost['local_steps']}")
        print(cost["note"])
    elif args.command in {"resume", "/resume"}:
        summary = summarize_run_artifact(args.run_json)
        print(f"Run: {summary['run_id']}")
        print(f"Question: {summary['question']}")
        print(f"Readiness: {summary['readiness_score']:.2f}")
        print(f"Claims: {summary['claims']}")
        print(f"Evidence chunks: {summary['evidence']}")
        print(f"Gaps: {len(summary['gaps'])}")
        if summary["artifacts"].get("html"):
            print(f"Report: {summary['artifacts']['html']}")
    elif args.command == "inspect":
        payload = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
        print(f"Question: {payload['question']}")
        print(f"Readiness: {payload['metrics'].get('readiness_score', 0):.2f}")
        print(f"Claims: {len(payload['claims'])}")
        print(f"Evidence chunks: {len(payload['evidence'])}")
        print(f"Gaps: {len(payload['gaps'])}")


def handle_discover(args) -> None:
    skill = args.skill
    if not skill and not args.question and not args.task:
        skill = "ioi-circuit"  # the headline demo
        print("No question/skill/task given; running the `ioi-circuit` skill.\n")
    budget = _budget_override(args)
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
    )
    print_discovery_summary(run)


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
    if not skills:
        print("No skills found.")
        return
    print(f"{len(skills)} interpretability skills:")
    for skill in skills:
        print(f"  {skill.name:24} [{skill.task}] {skill.description}")


def handle_modal(args) -> None:
    from .modal_app import dispatch_discovery, modal_status

    status = modal_status()
    if args.action == "status":
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
        print("Modal setup steps:")
        print("  1. pip install -e '.[modal,interp]'")
        print("  2. modal token new            # browser auth")
        print("  3. (optional) modal secret create openai-api-key OPENAI_API_KEY=sk-...")
        print("  4. mechferret /modal run --skill ioi-circuit")
        print(f"\nCurrent status: installed={status['installed']} authenticated={status['authenticated']}")
        return
    if args.action == "deploy":
        print("Deploy the GPU app with:\n  modal deploy mechferret/modal_app.py")
        print(f"App name: {status['app']} (gpu={status['gpu']})")
        return
    # action == "run"
    skill = args.skill or (None if (args.question or args.task) else "ioi-circuit")
    print(f"Dispatching discovery to Modal (skill={skill}, task={args.task}, model={args.model})...")
    result = dispatch_discovery(
        question=args.question, skill=skill, task=args.task, model=args.model, out_dir=args.out
    )
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
        print(f"Cleared {args.clear} credentials in {path}")
        return
    if args.provider:
        if args.provider == "local":
            config.default_provider = "local"
            path = save_config(config)
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
            print(f"Updated {args.provider} in {path}")
            return
        config.default_provider = args.provider
        path = save_config(config)
        print(f"Default provider: {args.provider} ({path})")
        return
    if args.show or not any([args.provider, args.api_key, args.model, args.clear]):
        print(f"Default provider: {config.default_provider}")
        for provider in sorted(PROVIDERS):
            settings = config.providers.get(provider)
            key_state = "configured" if settings and settings.api_key else "missing"
            model = settings.model if settings and settings.model else "default"
            print(f"{provider}: key={key_state}, model={model}")
        return
    print("--api-key and --model require --provider", file=sys.stderr)
    raise SystemExit(2)


def print_summary(run) -> None:
    print(f"Run: {run.run_id}")
    print(f"Readiness score: {run.metrics.get('readiness_score', 0):.2f}")
    print(f"Claims: {len(run.claims)}")
    print(f"Evidence chunks: {len(run.evidence)}")
    print(f"Report: {run.artifacts.get('html')}")
    print(f"Graph: {run.artifacts.get('graph')}")
    print(f"Evals: {run.artifacts.get('evals')}")
    print(f"Trace: {run.artifacts.get('trace')}")
