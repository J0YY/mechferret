# MechFerret

**MechFerret is an autonomous mechanistic-interpretability research system.**
Give it a question about a model's internals — or a reusable *skill* — and it
forms falsifiable hypotheses, runs causal experiments, checks its own
experimental rigor, and returns **reproducible mechanistic discoveries** with an
auditable dossier: hypotheses, an experiment ledger, controls, effect sizes,
confidence and novelty scores, a claim graph, eval assertions, and trace spans.

It runs **fully offline and deterministically** by default (a synthetic backend
that fabricates a hidden ground-truth circuit), and scales to **real models on
Modal GPUs** (`transformer_lens`) with the identical probe + engine code.

Built for the Autoresearch Systems Hackathon (Modal · OpenAI · Raindrop ·
Antler) — *Applied Autonomous Research* in a high-impact domain:

- **Agent architectures & control loops** — a hypothesize → screen → triangulate
  → critique → spawn loop; a parallel experiment coordinator; a budget guard
  that makes the agent safe to leave running unattended.
- **Retrieval & knowledge synthesis** — prior-art grounding (BM25 + memory +
  optional provider web search), a citation ledger, and novelty scoring of
  discoveries against the literature.
- **Applied autonomous research** — it actually runs interpretability
  experiments and makes confirmed, reproducible discoveries.

## Quickstart (offline, deterministic — no GPU, no keys)

```bash
python3 -m mechferret discover --skill ioi-circuit --out runs/demo
open runs/demo/report.html
```

In ~1 second this screens ~96 GPT-2 attention heads by causal ablation, promotes
the significant + reproducible ones to single-head hypotheses, triangulates each
with three independent probes, and confirms name-mover / duplicate-token heads
of the Indirect Object Identification circuit.

```bash
python3 -m mechferret discover --skill find-induction-heads --out runs/induction
python3 -m mechferret discover "Where is a fact stored in GPT-2?" --task factual_recall
```

## Skills (reusable playbooks)

```bash
python3 -m mechferret /skills                  # list playbooks
python3 -m mechferret /skills ioi-circuit      # show one
```

Built-in: `ioi-circuit`, `find-induction-heads`, `logit-lens-sweep`,
`factual-recall-trace`. Each is a versionable JSON in `mechferret/skills/`
defining the task, model, screen width, triangulation depth, compute budget, and
the bar that counts as "done".

## Modal (GPU compute)

The whole discovery loop runs remotely on a GPU with a real model:

```bash
pip install -e '.[modal,interp]'
python3 -m mechferret /modal status            # detect install + auth + GPU
python3 -m mechferret /modal setup             # one-time setup steps
python3 -m mechferret /modal run --skill ioi-circuit   # real GPT-2 on a Modal A10G
```

`/modal run` dispatches to Modal when installed + authenticated, and falls back
to the local synthetic backend otherwise — always returning a run and reporting
which path executed. Or run the entrypoint directly:

```bash
modal run mechferret/modal_app.py --skill ioi-circuit
```

## How rigor works

Every experiment compares its target against a matched **negative control** and
runs across multiple **seeds**. The engine marks an effect:

- **significant** if it clears a probe-specific floor *and* separates from its
  control by more than the cross-seed noise, and
- **reproduced** if the sign is stable across seeds with small relative spread.

A head is only named a "name mover" / "induction head" after **≥2 independent
probes** (ablation, attention pattern, direct logit attribution, activation
patching) agree — triangulation, not a single measurement.

## Real models

```bash
pip install -e '.[interp]'                     # torch + transformer_lens
python3 -m mechferret discover --skill ioi-circuit --backend transformer_lens
```

## Prior-art search and providers

```bash
python3 -m mechferret /login openai            # or anthropic
python3 -m mechferret discover --skill ioi-circuit --provider openai
python3 -m mechferret /api --show
```

## Raindrop Workshop

```bash
export RAINDROP_LOCAL_DEBUGGER=1
raindrop workshop
python3 -m mechferret discover --skill ioi-circuit
```

Every run writes `trace.jsonl`; with Workshop running, spans mirror to the local
debugger. The local JSONL trace is the source of truth either way.

## Operational commands

```bash
python3 -m mechferret /doctor                  # env, packages, skills, interp backend, Modal
python3 -m mechferret /registry                # tools/tasks/playbooks/evaluators (with permission + cost)
python3 -m mechferret /memory --recent 5       # recall prior runs + confirmed mechanisms
python3 -m mechferret /cost runs/demo/run.json
python3 -m mechferret /resume runs/demo/run.json
```

The original literature-research loop (`run`, `demo`, `/goal` / `/loop`) is still
available and is what grounds the discovery loop in prior art.

## What judges should notice

1. It **runs real experiments** and returns **reproducible discoveries**, with
   negative controls and cross-seed reproducibility — not a summary of papers.
2. The same probe + engine code runs **offline (deterministic)** or on a
   **Modal GPU** against a real model.
3. Autonomy is **budget-bounded**, not attention-bounded: it stops on a rigor +
   confirmed-mechanism bar or a compute budget.
4. Every claim is **auditable**: `experiments.json`, `discoveries.json`,
   `graph.json`, `evals.json`, and `trace.jsonl`.
5. Architecture patterns are ported from Claude Code (coordinator/swarm, skills,
   hooks/permissions, typed tool registry) — see `docs/ARCHITECTURE.md`.

## Development

```bash
python3 -m unittest discover -s tests
python3 -m mechferret discover --skill ioi-circuit --out runs/demo
```

See `docs/ARCHITECTURE.md` and `docs/DEMO_SCRIPT.md` for the full design and the
judging narrative.
