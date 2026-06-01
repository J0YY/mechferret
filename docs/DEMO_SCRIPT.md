# Demo Script (≈3 minutes)

## The one-liner

> "MechFerret is an autonomous interpretability researcher. Give it a question,
> and it forms hypotheses, runs causal experiments on a model, checks its own
> rigor, and hands back reproducible discoveries — offline, or on Modal GPUs."

## 0. Install once (then no `python`)

```bash
./install.sh        # global `mechferret` (pipx); or use ./bin/mechferret with zero install
mechferret          # bare command = interactive prompt, like `claude`
```

## 1. Run the headline discovery

```bash
mechferret discover --skill <skill> --model <model> --backend synthetic --out runs/demo
mechferret discover --skill <skill> --model <model> --backend synthetic --out runs/demo --json
open runs/demo/report.html
```

Talk track while it runs (~1s):

1. "It grounded the question in prior art, then **screened ~96 attention heads**
   by causal ablation."
2. "Heads with a **significant, reproducible** effect were promoted to
   single-head hypotheses and **triangulated** with three independent probes —
   attention pattern, direct logit attribution, activation patching."
3. "Its critic confirms a head only when ≥2 independent probes agree against a
   **negative control**."

## 2. Show the dossier (`runs/demo/report.html`)

- **Confirmed Mechanisms** — candidate components that cleared independent
  probes, each with effect size, reproducibility, and **novelty vs. the
  literature**.
- **Experiment Ledger** — every probe, its effect vs. control, significance,
  reproducibility, and the backend used.
- **Metrics** — `rigor_score`, `reproducibility_rate`, `confirmed_mechanisms`,
  `readiness_score`, and budget usage.

## 3. Show it self-checks and is auditable

```bash
cat runs/demo/evals.json        # has_confirmed_mechanism, every_experiment_has_control,
                                # significant_effects_reproduce, discoveries_are_triangulated
python3 -m mechferret verify runs/demo/run.json --strict
cat runs/demo/discoveries.json  # discoveries + full hypothesis lifecycle
cat runs/demo/graph.json        # sources -> evidence -> claims -> hypotheses -> discoveries
cat runs/demo/trace.jsonl       # per-phase spans (mirror to Raindrop Workshop)
```

## 4. Minimal-human-in-the-loop autonomy

```bash
python3 -m mechferret /skills                       # reusable playbooks
python3 -m mechferret discover --skill find-induction-heads --model <model> --backend synthetic --out runs/induction
```

"Different skill, different task — same loop finds an **induction head** and its
upstream **previous-token head**. The budget (`hooks.py`) is the only stop
condition; no human approves each step. The model under study is explicit, not
silently chosen by MechFerret."

## 5. Scale to real models on Modal

```bash
python3 -m mechferret /modal status                 # detects install + auth + GPU
python3 -m mechferret /modal setup                  # one-time setup steps
python3 -m mechferret /modal status --json          # script-friendly status
python3 -m mechferret /modal run --skill <skill> --model <model> # whole loop on a GPU
```

"Same code, same probes — the local backend swaps for `transformer_lens` on a
Modal A10G when you want real model measurements."

## 5b. Or run it on your own SLURM cluster

```bash
mechferret /cluster setup                            # generic connection steps
mechferret /cluster run --skill <skill> --model <model> --dry-run # shows the exact ssh+srun command
mechferret /cluster run --skill <skill> --model <model> --dry-run --json
mechferret /cluster run --skill <skill> --model <model>          # ssh -> srun -> discover -> scp dossier back
```

"Have a cluster instead of Modal? Set a few env vars and the same loop runs via
`srun` on your GPUs — nothing host-specific is hard-coded."

## Sponsor upgrade paths

```bash
python3 -m mechferret /login openai                 # prior-art web search
python3 -m mechferret /login anthropic
export RAINDROP_LOCAL_DEBUGGER=1 && raindrop workshop  # live trace of every phase
```

## Judging-criteria map

- **Technical depth** — causal screen → triangulation → rigor critic; budgeted
  autonomy; GPU offload.
- **Originality** — an agent that *runs experiments and makes reproducible
  discoveries*, not a summariser.
- **Demo clarity** — fast local run, fully inspectable dossier.
- **Applied autonomous research** — mechanistic interpretability, end to end.
