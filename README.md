# MechFerret

**MechFerret is an autonomous mechanistic-interpretability research system.**
Give it a question about a model's internals — or a reusable *skill* — and it
forms falsifiable hypotheses, runs causal experiments, checks its own
experimental rigor, and returns **reproducible mechanistic discoveries** with an
auditable dossier: hypotheses, an experiment ledger, controls, effect sizes,
confidence and novelty scores, a claim graph, eval assertions, and trace spans.

It runs **fully offline and deterministically** by default (a synthetic backend
that fabricates a hidden ground-truth circuit), and scales to **real models on
GPUs** (`transformer_lens`, via Modal or your own SLURM cluster) with the
identical probe + engine code.

What it gives you:

- **Agent architecture & control loops** — a hypothesize → screen → triangulate
  → critique → spawn loop; a parallel experiment coordinator; a budget guard
  that makes the agent safe to leave running unattended.
- **Retrieval & knowledge synthesis** — prior-art grounding (BM25 + memory +
  optional provider web search), a citation ledger, and novelty scoring of
  discoveries against the literature.
- **Autonomous research** — it actually runs interpretability experiments and
  makes confirmed, reproducible discoveries, not just summaries.

## Install — one command, then no `python`

The fastest path is **pipx** (installs an isolated global `mechferret` command):

```bash
pipx install .          # from the repo root; gives you global `mechferret` and `mf`
pipx ensurepath         # makes sure ~/.local/bin is on your PATH (only needed once)
mechferret              # opens the interactive prompt, like `claude`
```

`./install.sh` does the same (pipx if present, otherwise `pip install --user`).

If a shell that was already open says `command not found: mechferret`, that
session has a stale command cache — run `hash -r` or open a new terminal tab.
Still not found? Add `~/.local/bin` to your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

**Editing the code?** Install editable so the command tracks your changes:

```bash
pipx install --force --editable .
```

**Prefer zero install?** `./bin/mechferret discover --skill ioi-circuit` runs
straight from the repo (no install needed). `mf` is a short alias for
`mechferret`, and `python3 -m mechferret ...` works everywhere too.

> Requires Python 3.11+. The core has **no dependencies** and runs offline; GPU,
> provider search, Modal, and cluster features are optional extras (see below).

## Interactive prompt (conversational agent)

Run `mechferret` with no arguments to open a Claude-Code-style prompt. Your
messages are piped to a model (Claude or GPT); it converses and **calls
MechFerret's discovery tools** when you ask for interpretability work. On your
first message, if no model is connected, it walks you through adding an API key.

```text
❯ hi, what can you do?                         # just talks to the model
❯ find the IOI circuit in gpt2                 # model calls run_discovery, narrates results
❯ now check induction heads on the synthetic backend
❯ /login                                       # connect / change your model API key
❯ /model claude-sonnet-4-6                     # set the conversation model
❯ /modal run --skill ioi-circuit              # GPU on Modal
❯ /open                                        # open the last run's HTML dossier
❯ /help                                        # all commands · /exit to quit
```

Plain text → the model. `/commands` → driven directly. The model holds the
conversation; the architecture/agent parts (discovery loop, skills, experiments)
are tools it invokes. API keys are stored locally (or read from
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).

## Quickstart (offline, deterministic — no GPU, no keys)

Prefer one-shot commands? Everything is available non-interactively too:

```bash
mechferret discover --skill ioi-circuit --out runs/demo   # or: python3 -m mechferret ...
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

## Cluster (your own SLURM, instead of / alongside Modal)

If you have a SLURM cluster, MechFerret runs the loop there over SSH (`srun`) —
no Modal required. It is fully generic: configure it with env vars (or
`.mechferret/cluster.json`), nothing host-specific is baked in.

```bash
mechferret /cluster setup            # intuitive connection steps for any SLURM cluster
export REMOTE_HOST=<your-ssh-alias>
export REMOTE_PROJECT_DIR=<remote-project-dir>
export SLURM_PARTITION=gpu SLURM_GRES=gpu:a100:1 SLURM_TIME=02:00:00
export REMOTE_RUN_SETUP='source ~/miniconda3/etc/profile.d/conda.sh && conda activate mf'
mechferret /cluster status           # checks config + non-interactive SSH reachability
mechferret /cluster run --skill ioi-circuit --dry-run   # see the exact ssh+srun command
mechferret /cluster run --skill ioi-circuit             # run it, then scp the dossier back
```

MechFerret will `ssh → srun (with your flags) → mechferret discover --backend
transformer_lens` on a compute node and copy `run.json` back to `--out`. It
falls back to the local synthetic backend (with a note) if the cluster isn't
configured or reachable.

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

## Design highlights

1. It **runs real experiments** and returns **reproducible discoveries**, with
   negative controls and cross-seed reproducibility — not a summary of papers.
2. The same probe + engine code runs **offline (deterministic)** or on a
   **real GPU** (Modal or your own cluster) against a real model.
3. Autonomy is **budget-bounded**, not attention-bounded: it stops on a rigor +
   confirmed-mechanism bar or a compute budget.
4. Every claim is **auditable**: `experiments.json`, `discoveries.json`,
   `graph.json`, `evals.json`, and `trace.jsonl`.
5. Clean control-loop architecture — coordinator/swarm, skills, hooks/permissions,
   and a typed tool registry — see `docs/ARCHITECTURE.md`.

## Development

```bash
python3 -m unittest discover -s tests
python3 -m mechferret discover --skill ioi-circuit --out runs/demo
```

See `docs/ARCHITECTURE.md` and `docs/DEMO_SCRIPT.md` for the full design and the
judging narrative.
