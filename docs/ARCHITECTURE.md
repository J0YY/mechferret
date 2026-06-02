# Architecture

MechFerret is an **autonomous mechanistic-interpretability research system**. It
takes an interpretability question (or a reusable *skill*) and, with minimal
human involvement, hypothesises, experiments, critiques for rigor, and
synthesises **reproducible mechanistic discoveries** — every claim backed by an
auditable experiment spec.

## Two control loops

### 1. Literature loop (`controller.py` — `MechFerret`)
Plan → retrieve (BM25 + memory + optional provider web search) → extract cited
claims → critique (coverage, diversity, citation density, contradictions) →
expand gaps → synthesise. Produces the evidence ledger and grounds the
discovery loop in prior art.

### 2. Discovery loop (`discovery.py` — `DiscoveryController`)
The autoresearch core. One round of human input → autonomous causal discovery:

```text
question / skill
  -> prior art        (literature loop, memory of confirmed mechanisms)
  -> hypothesize      (HypothesisGenerator: falsifiable claims about internals)
  -> screen           (ablate budgeted candidate heads across the coordinator)
  -> promote          (significant + reproducible heads become single-head hypotheses)
  -> triangulate      (independent probes: attention pattern, DLA, activation patching)
  -> critique         (ExperimentCritic: controls, significance, reproducibility, triangulation)
  -> spawn / repeat   (gaps become the next round's experiments)
  -> synthesize       (confirmed mechanisms -> discoveries, scored for novelty vs prior art)
  -> dossier + experiments.json + discoveries.json + graph + evals + trace
```

It halts on the skill's stop criteria (confirmed-mechanism count + rigor score),
on convergence (no new leads), or on the budget — **never on a human's
attention**.

## Interpretability engine (`interp/`)

| Module | Role |
|---|---|
| `tasks.py` | Supported named text tasks: induction, greater-than, factual recall, and IOI when explicitly selected (clean/corrupt prompts + answer pairs). |
| `probes.py` | Backend-agnostic probes: `head_ablation`, `activation_patching`, `attention_pattern`, `direct_logit_attribution`, `logit_lens`. Each reports a signed **effect** and a matched **control**. |
| `synthetic.py` | Explicit offline smoke backend. Provides structured probe results for tests and demos — **no torch, no GPU, no keys**. |
| `backends.py` | Backend resolution + the real `TransformerLensBackend` (identical probe surface, measures a real `HookedTransformer`). Auto uses synthetic only when real deps are absent or `MECHFERRET_FORCE_SYNTHETIC=1`; real-model load failures do not silently downgrade. |
| `engine.py` | Runs a spec across seeds → `ExperimentResult` with effect size, significance (clears a floor *and* beats cross-seed noise), and reproducibility (sign stable across seeds). |
| `hypotheses.py` | Screen → promote → triangulate; confirm/refute/inconclusive; classify head roles from probe evidence. |
| `critic.py` | Interpretability rigor: controls, significance, reproducibility rate, triangulation; emits gaps that drive the next round. |

The local and real backends share **the same probe + engine + critic code**, so
a run that works locally scales to a real model unchanged.

## Patterns ported from Claude Code

The leaked Claude Code architecture informed the control surfaces (no source was
copied — these are Python re-implementations adapted to research):

| Claude Code | MechFerret |
|---|---|
| `coordinator/` + `AgentTool` swarms | `coordinator.py` — parallel, order-preserving experiment fan-out. |
| `hooks/toolPermission` + cost tracking | `hooks.py` — `BudgetGuard` (max experiments/rounds/GPU-seconds/wall) + permission classes (local/network/gpu). |
| `skills/` + `SkillTool` | `skills.py` + `skills/*.json` — declarative, shareable interpretability playbooks. |
| `tools.ts` / `Tool.ts` registry | `registry.py` — typed tools/tasks/playbooks/evaluators with permission + cost class. |
| `extractMemories` / session memory | `memory.py` — confirmed mechanisms persist as recallable sources. |
| Structured output | dataclass `ExperimentResult` / `Discovery`, serialised to JSON artifacts. |

## Compute (`modal_app.py`)

Modal supplies the GPU. The **entire discovery loop** runs remotely on a GPU
container (`torch` + `transformer_lens`) via `run_discovery_remote`; the dossier
returns as JSON. `dispatch_discovery` runs on Modal when installed +
authenticated and fails closed with structured setup/dispatch errors otherwise.
Synthetic local fallback is available only through explicit opt-in such as
`--local-fallback`, so a requested remote run cannot quietly turn into smoke
data.

## Artifacts (every run)

`report.html`, `report.md`, `run.json`, `graph.json` (sources → evidence →
claims → hypotheses → discoveries), `evals.json` (research + interpretability
rigor self-checks), `manifest.json` (source digests, provenance, artifact
hashes), `trace.jsonl` (per-phase spans, mirrored to Raindrop
Workshop when `RAINDROP_LOCAL_DEBUGGER=1`), plus `experiments.json` and
`discoveries.json` for discovery runs.

## Why this is judge-friendly

- **Technical depth**: a real causal-discovery loop (screen → triangulate →
  rigor critic), backend-agnostic probes, budget-bounded autonomy, and GPU
  offload to Modal.
- **Originality**: an autoresearch agent that *runs experiments and makes
  reproducible mechanistic discoveries*, not a literature summariser or prompt
  wrapper.
- **Demo clarity**: works locally; the dossier exposes hypotheses, experiments,
  controls, and discoveries with effect sizes.
- **Applied autonomous research** in a high-impact domain (interpretability),
  with sponsor upgrade paths: Modal (compute), OpenAI/Anthropic (prior art),
  Raindrop Workshop (trace).
