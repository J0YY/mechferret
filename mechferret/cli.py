from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import MechFerret
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
    run.add_argument("--no-memory", action="store_true", help="Do not recall prior-run memory.")

    demo = sub.add_parser("demo", help="Run the built-in hackathon demo corpus.")
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--db", default=".mechferret/memory.sqlite")
    demo.add_argument("--max-rounds", type=int, default=2)
    demo.add_argument("--openai", action="store_true")
    demo.add_argument("--with-memory", action="store_true", help="Recall prior-run memory during the demo.")

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
            include_memory=args.with_memory,
        )
        print_summary(run)
    elif args.command == "inspect":
        payload = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
        print(f"Question: {payload['question']}")
        print(f"Readiness: {payload['metrics'].get('readiness_score', 0):.2f}")
        print(f"Claims: {len(payload['claims'])}")
        print(f"Evidence chunks: {len(payload['evidence'])}")
        print(f"Gaps: {len(payload['gaps'])}")


def print_summary(run) -> None:
    print(f"Run: {run.run_id}")
    print(f"Readiness score: {run.metrics.get('readiness_score', 0):.2f}")
    print(f"Claims: {len(run.claims)}")
    print(f"Evidence chunks: {len(run.evidence)}")
    print(f"Report: {run.artifacts.get('html')}")
    print(f"Graph: {run.artifacts.get('graph')}")
    print(f"Evals: {run.artifacts.get('evals')}")
    print(f"Trace: {run.artifacts.get('trace')}")
