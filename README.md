# MechFerret

MechFerret is a hackathon-ready autonomous research system. It runs an explicit
plan/search/extract/critic/synthesize loop, keeps a durable evidence memory, and
generates an inspectable dossier with citations, confidence scores, gaps,
contradictions, a claim graph, eval assertions, and trace artifacts.

The project is designed for the Autoresearch Systems Hackathon track:

- Agent architectures and control loops: budget-aware multi-round planning,
  critic-driven gap expansion, deterministic replay artifacts, and trace spans.
- Retrieval and synthesis: local BM25 retrieval, claim extraction, citation
  ledger, source diversity scoring, contradiction pressure, graph JSON, eval
  assertions, and HTML/Markdown reports.
- Applied autonomous research: works offline from seed corpora, can optionally
  use OpenAI Responses API web search, and includes Modal and Raindrop Workshop
  integration points.

## Quickstart

```bash
python3 -m mechferret demo
open runs/demo/report.html
```

Run on your own corpus:

```bash
python3 -m mechferret run "What evidence supports this research direction?" \
  --source ./papers \
  --source ./notes.md \
  --out runs/my-run \
  --max-rounds 3
```

Optional live search through OpenAI:

```bash
export OPENAI_API_KEY=...
python3 -m mechferret run "Find current evidence and risks for agentic research systems" \
  --provider openai \
  --out runs/live
```

Store API keys and choose a provider:

```bash
python3 -m mechferret /login openai
python3 -m mechferret /login anthropic
python3 -m mechferret /api --show
python3 -m mechferret /api --provider anthropic
```

Run until a target research bar:

```bash
python3 -m mechferret goal "Can this idea reach NeurIPS main?" \
  --venue "NeurIPS main" \
  --target 0.9 \
  --source ./proposal \
  --max-iterations 5 \
  --provider anthropic
```

## Raindrop Workshop

MechFerret writes `trace.jsonl` for every run. If Raindrop Workshop is running,
set this to mirror spans to the local debugger:

```bash
export RAINDROP_LOCAL_DEBUGGER=1
raindrop workshop
python3 -m mechferret demo
```

The local JSONL trace remains the source of truth even if Workshop is not
installed.

## Modal

`mechferret/modal_app.py` contains a deployable Modal entrypoint. Install the
optional dependency and run:

```bash
pip install -e '.[modal]'
modal run mechferret/modal_app.py
```

## What Judges Should Notice

1. Every answer has an auditable evidence ledger and `graph.json` claim graph.
2. The critic expands the plan when coverage or source diversity is weak.
3. Runs are replayable from local artifacts, including `trace.jsonl` and
   `evals.json`.
4. The system is useful without API keys but upgrades cleanly to OpenAI or
   Anthropic provider calls.
5. The Raindrop trace gives a visible decision timeline for each agent phase.

## Development

```bash
python3 -m unittest discover -s tests
python3 -m mechferret demo --out runs/demo
```

See `docs/ARCHITECTURE.md` and `docs/DEMO_SCRIPT.md` for the judging narrative.
