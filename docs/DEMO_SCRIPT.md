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
python3 -m mechferret demo --openai --out runs/openai-demo
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

