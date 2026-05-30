from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROVIDERS, configure_provider, load_config, prompt_api_key, save_config
from .controller import MechFerret
from .costs import estimate_run_cost
from .goal_loop import GoalLoop
from .ops import memory_clear, memory_recent, memory_summary, print_doctor, summarize_run_artifact
from .registry import all_items, items_by_kind
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
