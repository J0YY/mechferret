# Demo Script

Run:

```bash
python3 -m mechferret demo --out runs/demo
open runs/demo/report.html
```

Narrative:

1. "This is not a prompt wrapper. It is a replayable research control loop."
2. Show the readiness score and metrics at the top of `report.html`.
3. Show the claims section: every claim has confidence and citations.
4. Show the evidence ledger: source chunks are preserved for audit.
5. Open `runs/demo/graph.json`: claims are connected to evidence and sources.
6. Open `runs/demo/evals.json`: the run self-checks against research quality
   assertions.
7. Open `runs/demo/trace.jsonl`: planner, retrieval, extraction, critic, and
   synthesis phases are traceable. With Raindrop Workshop running, these spans
   mirror to the local debugger.

Optional sponsor demo:

```bash
export OPENAI_API_KEY=...
python3 -m mechferret demo --provider openai --out runs/openai-demo
```

Provider login:

```bash
python3 -m mechferret /login openai
python3 -m mechferret /login anthropic
python3 -m mechferret /api --provider anthropic
```

Goal-loop demo:

```bash
python3 -m mechferret /loop "Can this autoresearch project reach NeurIPS main?" \
  --venue "NeurIPS main" \
  --target 0.9 \
  --source examples/seed_corpus \
  --max-iterations 3 \
  --provider local
```

Ops quick checks:

```bash
python3 -m mechferret /doctor
python3 -m mechferret /registry --kind task
python3 -m mechferret /memory --recent 3
```

```bash
export RAINDROP_LOCAL_DEBUGGER=1
raindrop workshop
python3 -m mechferret demo --out runs/raindrop-demo
```

```bash
pip install -e '.[modal]'
modal run mechferret/modal_app.py
```
