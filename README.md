# MechFerret

**MechFerret is a Track 3 applied autonomous research agent for mechanistic
interpretability.** Give it an interpretability question or a reusable
playbook, and it plans the investigation, grounds the question in prior art,
forms falsifiable hypotheses about model internals, runs causal experiments,
checks its own rigor, and returns an auditable research dossier.

This project is built for the Raindrop autoresearch hackathon's **Applied
Autonomous Research** track: an end-to-end research agent tailored to a
high-impact domain. The domain is mechanistic interpretability because it
studies the AI systems that will increasingly do research in every other
field. Better interpretability tools compound across law, biology, coding, and
AI safety because they make the models themselves inspectable.

The short version:

- **Autonomous research, not just synthesis.** MechFerret does not stop at
  summarizing papers. It runs experiments against model components and promotes
  only reproducible, triangulated mechanisms into discoveries.
- **Modal is the scale path.** The same discovery loop that runs locally can run
  inside a Modal GPU function with `torch` and `transformer_lens`, so the demo
  can move from deterministic offline mode to real GPT-2 measurements.
- **Raindrop Workshop is the visibility layer.** Every phase emits structured
  spans to `trace.jsonl` and can mirror them live to Raindrop Workshop, turning
  the agent's reasoning, experiments, critic loop, and artifact generation into
  an inspectable timeline.
- **The output is judge-readable.** Each run writes hypotheses, experiment
  ledgers, controls, effect sizes, confidence and novelty scores, a claim graph,
  self-evaluations, and a replay trace.

## Why Interpretability

Modern AI systems are mostly black boxes: we can inspect inputs and outputs, but
not the computation in between. Mechanistic interpretability tries to reverse
engineer that hidden computation into human-understandable parts such as
attention heads, MLP features, and circuits.

For example, in GPT-2's Indirect Object Identification task, a model completes a
sentence like "When John and Mary went to the store, Mary gave a drink to" with
"John". Prior interpretability work found that specific attention heads act as
name movers, duplicate-token detectors, and suppression heads. A rigorous
interpretability claim is causal: if a head really matters, ablating or patching
it should change the behavior in the predicted direction, while matched negative
controls should not.

That is why this is a strong Track 3 domain:

- **It is high impact.** Interpretability is a direct lever for AI safety,
  auditing, debugging, and trust. You cannot reliably correct what you cannot
  inspect.
- **It benefits from autonomy.** There are many candidate circuits, models,
  prompts, probes, controls, and ablations. A bounded agent can screen and
  triangulate far more hypotheses than a human can manually inspect during a
  short session.
- **It compounds into other domains.** Law and biology autoresearch are useful,
  but they study the external world. Interpretability studies the systems doing
  the research, so progress here improves every downstream research agent.
- **It needs reproducibility.** Interpretability is easy to overclaim from one
  pretty activation map. MechFerret makes the bar explicit: negative controls,
  cross-seed reproducibility, independent probes, and an auditable experiment
  ledger.

The `/why` command in the interactive prompt gives the concise version of this
argument during the demo.

## What MechFerret Does

MechFerret runs a complete autonomous discovery loop:

```text
question or skill
  -> prior-art grounding
  -> falsifiable hypothesis generation
  -> broad causal screen over candidate model components
  -> promotion of significant and reproducible candidates
  -> triangulation with independent probes
  -> rigor critique and gap detection
  -> follow-up experiments when the budget allows
  -> synthesized discoveries and auditable dossier
```

For the headline demo, the `ioi-circuit` skill screens GPT-2 attention heads for
the IOI behavior, promotes significant heads, and triangulates them with
multiple probes. A head is not named a mechanism from one measurement. It must
clear a causal effect threshold, separate from a matched control, reproduce
across seeds, and agree with independent evidence such as attention pattern,
direct logit attribution, or activation patching.

MechFerret can run in two modes:

- **Synthetic offline backend:** deterministic, no API key, no GPU, no torch.
  This backend fabricates hidden ground-truth circuits so the entire agentic
  loop can be demonstrated in seconds and every artifact is replayable.
- **Real model backend:** `transformer_lens` on a local GPU, Modal GPU, or a
  configured SLURM cluster. The probe and engine interfaces are the same, so
  successful offline runs upgrade to real model measurements without rewriting
  the agent.

## Hackathon Fit

The Raindrop hackathon rubric emphasizes technical depth, originality, demo
clarity, and standout execution in one track. MechFerret targets Track 3:
**Applied Autonomous Research**.

| Criterion | How MechFerret addresses it |
| --- | --- |
| Technical depth | A real control loop over hypotheses, causal screens, triangulation probes, an experiment critic, budgets, permissions, memory, and reproducible artifacts. |
| Originality | The agent runs mechanistic interpretability experiments and produces confirmed discoveries, instead of wrapping search and summarization. |
| Demo clarity | The offline path runs in seconds and writes HTML, Markdown, JSON ledgers, self-evals, and trace logs that show exactly what happened. |
| Track 3: Applied Autonomous Research | The domain is mechanistic interpretability, a high-impact AI safety and debugging field where autonomous hypothesis search is naturally useful. |
| Raindrop Workshop use case | Raindrop becomes a live microscope for an autonomous research run: prior-art retrieval, experiment batches, critic decisions, gaps, synthesis, errors, and artifacts all become trace spans. |

## Architecture

The codebase has two cooperating research loops.

### Literature and Knowledge Loop

`mechferret/controller.py` handles research synthesis:

- plans a research strategy
- retrieves evidence from local seed corpora, memory, URLs, and optional
  provider search
- extracts cited claims
- critiques coverage, source diversity, contradictions, and gaps
- synthesizes a dossier with citations and confidence scores

This loop grounds the discovery system in prior work. It prevents the agent from
treating every mechanism as novel simply because it rediscovered it.

### Interpretability Discovery Loop

`mechferret/discovery.py` is the Track 3 core:

- loads a question or JSON skill
- infers the interpretability task
- retrieves prior art and remembered discoveries
- uses `HypothesisGenerator` to screen candidate heads or components
- fans experiments out through `Coordinator`
- runs probes through the backend-agnostic `InterpEngine`
- updates hypothesis state from experiment results
- uses `ExperimentCritic` to enforce rigor
- stops on skill criteria, convergence, or budget
- writes the run artifacts and records confirmed mechanisms to memory

The loop is intentionally budget-bounded instead of attention-bounded. A human
sets the goal and constraints; the system decides which candidates to test,
which candidates deserve triangulation, and when the evidence is strong enough.

### Interpretability Engine

The `mechferret/interp/` package keeps model science separate from orchestration:

| Module | Role |
| --- | --- |
| `tasks.py` | Canonical tasks such as IOI, induction, greater-than, and factual recall. |
| `probes.py` | Backend-agnostic probes: head ablation, activation patching, attention pattern checks, direct logit attribution, and logit lens. |
| `synthetic.py` | Deterministic offline backend with hidden ground-truth circuits for reproducible demos. |
| `backends.py` | Backend resolution plus the real `TransformerLensBackend`. |
| `engine.py` | Runs experiment specs across seeds and reports effect size, control effect, significance, and reproducibility. |
| `hypotheses.py` | Turns screens into hypotheses and updates their status from experimental evidence. |
| `critic.py` | Scores controls, significance, reproducibility, and triangulation; emits gaps for follow-up. |

### Artifacts

Every discovery run writes:

- `report.html` and `report.md` - human-readable dossier
- `run.json` - complete structured run
- `experiments.json` - experiment ledger with targets, probes, effects,
  controls, seeds, backend, significance, and reproducibility
- `discoveries.json` - confirmed mechanisms with confidence, novelty, effect
  size, and supporting experiments
- `graph.json` - source -> evidence -> claim -> hypothesis -> discovery graph
- `evals.json` - self-checks such as controls present, significant effects
  reproduce, and discoveries are triangulated
- `trace.jsonl` - per-phase spans, optionally mirrored to Raindrop Workshop

## Modal Integration

Modal is how MechFerret leaves laptop-demo mode and runs real interpretability
experiments on GPUs.

The integration lives in `mechferret/modal_app.py` and has three layers:

- `modal_status()` checks whether Modal is installed, authenticated, and which
  GPU type will be requested.
- `run_interp_remote()` runs a batch of experiment specs on a Modal GPU.
- `run_discovery_remote()` runs the entire discovery loop remotely against a
  real `transformer_lens` model and returns the dossier payload as JSON.

The default Modal GPU is `A10G`, configurable with `MECHFERRET_MODAL_GPU`.

```bash
pip install -e '.[modal,interp]'
mechferret /modal status
mechferret /modal setup
mechferret /modal run --skill ioi-circuit
```

`/modal run` is designed for a live hackathon demo. If Modal is installed and
authenticated, it dispatches the discovery run to Modal. If not, it falls back
to the local synthetic backend and still produces artifacts, while reporting
which backend executed.

You can also run the Modal entrypoint directly:

```bash
modal run mechferret/modal_app.py --skill ioi-circuit
```

Why Modal matters here:

- interpretability probes can be compute-heavy on real models
- the same skill can move from offline synthetic validation to GPU measurement
- the agent can keep the control loop local while pushing model execution to
  remote infrastructure
- GPU seconds are recorded into run metrics when the Modal path executes

## Raindrop Workshop Integration

Raindrop is the observability story for the autonomous research loop. The goal
is not just to show a final answer; it is to show how the agent behaved while
getting there.

`mechferret/tracing.py` writes a JSONL trace for every run. Each span contains:

- `trace_id`
- `run_id`
- `span_id`
- phase (`start`, `end`, `event`, or `error`)
- span name
- timestamp
- elapsed milliseconds
- structured attributes

Key spans include:

- `prior_art` - source loading, memory recall, and optional provider research
- `provider_research` - OpenAI or Anthropic research calls when configured
- `round` - each autonomous experiment round
- `experiments` - the batch of admitted probe specs
- `round_summary` - confirmed mechanism count and rigor score
- `synthesize` - final discovery and dossier construction
- `artifacts_written` - paths to generated outputs
- error spans - failed phases with exception type and message

To mirror traces into Raindrop Workshop:

```bash
export RAINDROP_LOCAL_DEBUGGER=1
raindrop workshop
mechferret discover --skill ioi-circuit --out runs/demo
```

By default, MechFerret posts spans to:

```text
http://127.0.0.1:5899/v1/traces
```

Override it with:

```bash
export RAINDROP_ENDPOINT=http://127.0.0.1:5899/v1/traces
```

The local `trace.jsonl` file remains the source of truth even if Workshop is not
running. That makes the Raindrop integration demo-friendly: observability is
live when available, but the research run never depends on a tracing service.

The intended Raindrop Workshop use case is a live replay of an autonomous
interpretability investigation. Judges can inspect not only "what answer did it
produce?" but also "what did the agent retrieve, test, reject, promote, and
trust?"

## Install

The fastest path is `pipx`, which installs an isolated global `mechferret`
command:

```bash
pipx install .
pipx ensurepath
mechferret
```

`./install.sh` does the same, using `pipx` when available and falling back to
`pip install --user`.

If an existing shell says `command not found: mechferret`, refresh the command
cache or open a new terminal:

```bash
hash -r
```

For development, install editable:

```bash
pipx install --force --editable .
```

For optional integrations:

```bash
pip install -e '.[modal,interp]'      # Modal + real model backend
pip install -e '.[openai]'            # OpenAI provider research
pip install -e '.[anthropic]'         # Anthropic provider research
pip install -e '.[all]'               # everything
```

Requires Python 3.11+. The core package has no required third-party runtime
dependencies and works offline.

## Quickstart

Run the deterministic headline discovery:

```bash
mechferret discover --skill ioi-circuit --out runs/demo
open runs/demo/report.html
```

This screens candidate GPT-2 heads for IOI, promotes significant and
reproducible heads, triangulates them with independent probes, runs the critic,
and writes the dossier.

Run a different built-in playbook:

```bash
mechferret discover --skill find-induction-heads --out runs/induction
```

Ask a free-form interpretability question:

```bash
mechferret discover "Where is a fact stored in GPT-2?" --task factual_recall --out runs/factual
```

Launch the interactive prompt:

```bash
mechferret
```

Useful prompt commands:

```text
/why                         why interpretability, why this domain
/skills                      list reusable interpretability playbooks
/modal status                check Modal readiness
/modal run --skill ioi-circuit
/doctor                      check environment, registry, Modal, and backend status
/open                        open the last HTML report
/memory --recent 5           inspect remembered runs and mechanisms
```

## Skills

Skills are reusable JSON playbooks in `mechferret/skills/`. They define the
task, model, screen width, triangulation depth, seeds, compute budget, and stop
criteria.

```bash
mechferret /skills
mechferret /skills ioi-circuit
```

Built-in skills:

- `ioi-circuit`
- `find-induction-heads`
- `logit-lens-sweep`
- `factual-recall-trace`

This is the mechanism that makes the agent more than a one-off demo. A new
interpretability workflow can be expressed as a skill without rewriting the
core control loop.

## Rigor Model

MechFerret's critic is intentionally conservative. An experiment result carries
both the target effect and a matched control. A finding is treated as strong
only when:

- the target effect clears the probe-specific floor
- the target separates from its control by more than cross-seed noise
- the sign is stable across seeds
- multiple independent probes agree before the system names a mechanism

This matters for the demo because the output is not a vibe-based explanation.
The report shows exactly which experiments supported each mechanism and which
gaps remain.

## Provider Research

The offline path uses local seed corpora and memory. If a provider is
configured, MechFerret can add live prior-art research to the grounding phase:

```bash
mechferret /login openai
mechferret /login anthropic
mechferret discover --skill ioi-circuit --provider openai
```

Provider search is optional. The core demo does not require API keys.

## Cluster Option

Modal is the primary sponsor-aligned GPU path, but the same discovery loop can
also run on a generic SLURM cluster over SSH:

```bash
mechferret /cluster setup
mechferret /cluster status
mechferret /cluster run --skill ioi-circuit --dry-run
mechferret /cluster run --skill ioi-circuit
```

The cluster path exists so the research loop is not locked to one compute
provider. For the hackathon story, Modal is the cleanest remote GPU demo.

## Demo Script

1. Explain the domain:

   ```bash
   mechferret
   /why
   ```

2. Run the offline deterministic discovery:

   ```bash
   mechferret discover --skill ioi-circuit --out runs/demo
   open runs/demo/report.html
   ```

3. Show the audit files:

   ```bash
   cat runs/demo/evals.json
   cat runs/demo/discoveries.json
   cat runs/demo/graph.json
   cat runs/demo/trace.jsonl
   ```

4. Turn on Raindrop Workshop and rerun:

   ```bash
   export RAINDROP_LOCAL_DEBUGGER=1
   raindrop workshop
   mechferret discover --skill ioi-circuit --out runs/raindrop-demo
   ```

5. Scale to Modal:

   ```bash
   mechferret /modal status
   mechferret /modal run --skill ioi-circuit --out runs/modal-demo
   ```

The talk track is simple: MechFerret is an autonomous interpretability
researcher. It turns a question into experiments, uses Modal when real GPU
measurements are wanted, and uses Raindrop Workshop to make the agent's internal
research process inspectable.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the main discovery path:

```bash
python3 -m mechferret discover --skill ioi-circuit --out runs/demo
```

See also:

- `docs/ARCHITECTURE.md`
- `docs/DEMO_SCRIPT.md`
- `mechferret/repl.py` for `/why`
- `mechferret/modal_app.py` for Modal dispatch
- `mechferret/tracing.py` for Raindrop trace emission
