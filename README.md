# MechFerret

**MechFerret is an autoresearch agent for interpretability.**

In plain terms: it is a research assistant that tries to figure out what is
going on inside AI models.

You give it a question like:

```text
Which parts of GPT-2 are responsible for this behavior?
```

MechFerret turns that into a research run. It looks at prior work, makes a few
testable guesses, runs experiments on model components, checks whether the
results are real, and writes up what it found.

The important bit is that it does not just write a summary. It actually runs the
interpretability experiments.

## Why Interpretability?

AI models are getting wildly capable, but most of the time we still interact
with them like black boxes. We give them text, images, code, or data. They give
us an answer. The scary part is the middle: we usually do not know what
calculation happened inside the model to produce that answer.

Interpretability is the field that tries to open up that middle.

Instead of asking only "did the model get the answer right?", interpretability
asks:

- What internal parts caused the answer?
- Did the model use a sensible strategy or a shortcut?
- Can we find the circuit that implements a behavior?
- If we remove or change that part, does the behavior go away?
- Can we explain the model in a way a human researcher can audit?

That is why this is such a good domain for autoresearch. There are tons of
candidate model parts to test, tons of possible prompts, tons of possible
controls, and lots of small experiments that need to be run carefully. A human
can do this, but it is slow. An agent can help by doing the repetitive research
loop: pick a hypothesis, run the check, compare against controls, keep the good
leads, drop the weak ones, and write down exactly what happened.

We picked interpretability for Track 3 because it is not just another research
domain. It is the domain that helps us understand the systems that will do more
and more research everywhere else. If we make interpretability faster, cheaper,
and easier to audit, that helps AI safety, model debugging, and every future
agent built on top of these models.

That is the bet behind MechFerret: make the "open the black box" loop feel more
like running a tool and less like hand-driving a pile of fragile notebooks.

## What It Does

MechFerret runs an end-to-end interpretability research loop:

```text
question
  -> read prior context
  -> make testable hypotheses
  -> run model experiments
  -> compare against controls
  -> keep results that reproduce
  -> check the evidence from multiple angles
  -> write a report with the full trail
```

For the demo, MechFerret focuses on classic mechanistic interpretability tasks
like Indirect Object Identification in GPT-2.

Example sentence:

```text
When John and Mary went to the store, Mary gave a drink to ___
```

GPT-2 often completes that with `John`. Interpretability asks: which attention
heads made that happen? Are there heads that copy the right name? Are there
heads that notice the repeated name? If we remove those heads, does the answer
change?

MechFerret takes that kind of question and runs the workflow:

- screens candidate attention heads
- promotes the promising ones into specific hypotheses
- tests them with multiple probes
- checks each result against negative controls
- verifies that the effect is stable across seeds
- writes a report with the evidence for each claimed mechanism

The result is a dossier, not just a paragraph. You get the answer, the
experiments, the controls, the confidence scores, the open gaps, and the trace
of what the agent did.

## The Hackathon Story

This is built for the Raindrop autoresearch hackathon, specifically **Track 3:
Applied Autonomous Research**.

Our applied domain is interpretability.

The core idea is simple:

> If AI agents are going to help with serious research, we should also build
> agents that research the AI systems themselves.

MechFerret is meant to show that loop clearly. It is not a chatbot that says
"here are some papers." It is an agent that takes a model behavior, investigates
the model internals, runs checks, and gives you artifacts you can inspect.

What we want judges to see:

- **Technical depth:** it has a real research loop with hypotheses,
  experiments, controls, budgets, memory, and reports.
- **Originality:** it applies autoresearch to interpretability itself, not just
  web search or summarization.
- **Demo clarity:** the offline demo runs quickly and produces readable files.
- **Raindrop fit:** every step can be traced, replayed, and inspected.
- **Modal fit:** the same loop can move from a deterministic local demo to GPU
  experiments on real models.

## Why Not Just A Claude Code Skill?

A Claude Code skill is great when you want to teach an assistant a workflow. You
can write instructions like "when doing interpretability, look for prior work,
make hypotheses, run checks, and summarize results."

But MechFerret is doing something different.

A skill is mostly guidance. MechFerret is a runnable research system.

That matters because interpretability research is not just a writing task. The
hard part is keeping the whole experiment loop honest:

- Which model behavior are we testing?
- Which model components are candidates?
- What exactly was ablated or patched?
- What was the negative control?
- Did the result reproduce?
- Did another probe agree?
- Where is the experiment log?
- Can we run the same thing on a GPU?
- Can we inspect the agent's decisions after the fact?

A Claude Code skill can remind an agent to care about those things. MechFerret
actually implements them.

Here is the difference:

| A Claude Code skill | MechFerret |
| --- | --- |
| Tells an assistant how to approach interpretability work. | Runs an interpretability research loop end to end. |
| Depends on the assistant to remember and follow the process. | Has code paths for planning, screening, experiments, controls, criticism, and reports. |
| Usually produces prose or code edits. | Produces experiment artifacts: `experiments.json`, `discoveries.json`, `evals.json`, `graph.json`, and `trace.jsonl`. |
| Does not own compute. | Can run locally, on Modal GPUs, or on a SLURM cluster. |
| Does not automatically create an audit trail. | Writes a trace that can mirror into Raindrop Workshop. |
| Is hard to replay exactly. | Stores structured runs and repeatable skill configs. |

The exciting part is that MechFerret can still feel like an assistant, but the
important research mechanics live in software instead of vibes. The agent is not
just saying "I think this head matters." It is running the check, comparing the
control, writing the result, and showing the trail.

That is why this is stronger than a skill file. A skill can describe the
researcher we want. MechFerret is the research machine.

## How Modal Fits In

The local demo is deterministic on purpose. It works without a GPU, without API
keys, and without installing heavy model packages. That makes it great for a
live demo.

But interpretability gets much more interesting when you run against real
models. That is where Modal comes in.

MechFerret can send the heavy model work to Modal:

- Modal starts a GPU container.
- The container installs `torch` and `transformer_lens`.
- MechFerret runs the same discovery loop against a real model.
- The result comes back as the same kind of report.

The Modal integration lives in `mechferret/modal_app.py`.

Useful commands:

```bash
pip install -e '.[modal,interp]'
mechferret /modal status
mechferret /modal setup
mechferret /modal run --skill ioi-circuit
```

The default GPU is `A10G`. You can change it with:

```bash
export MECHFERRET_MODAL_GPU=A100
```

The nice part is that `/modal run` is demo-safe. If Modal is installed and
authenticated, MechFerret runs remotely. If not, it falls back to the local
synthetic backend and still produces a report, while telling you which path it
used.

Direct Modal entrypoint:

```bash
modal run mechferret/modal_app.py --skill ioi-circuit
```

## How Raindrop Fits In

Raindrop is how we make the agent's work visible.

A final answer is not enough for an autoresearch system. You want to see what it
tried, what it skipped, what it trusted, what failed, and why it stopped.

MechFerret writes a trace for every run:

```text
runs/demo/trace.jsonl
```

That trace includes spans for:

- loading prior context
- recalling memory
- optional provider research
- each experiment round
- experiment batches
- critic summaries
- final synthesis
- artifact writing
- errors, if something fails

To mirror the trace into Raindrop Workshop:

```bash
export RAINDROP_LOCAL_DEBUGGER=1
raindrop workshop
mechferret discover --skill ioi-circuit --out runs/raindrop-demo
```

By default, traces are posted to:

```text
http://127.0.0.1:5899/v1/traces
```

You can override that with:

```bash
export RAINDROP_ENDPOINT=http://127.0.0.1:5899/v1/traces
```

The local `trace.jsonl` file is always written, even if Raindrop Workshop is not
running. So the observability path is useful live, but it never becomes a hard
dependency.

## Quick Demo

Install:

```bash
pipx install .
pipx ensurepath
```

Run the headline interpretability discovery:

```bash
mechferret discover --skill ioi-circuit --out runs/demo
open runs/demo/report.html
```

Look at the audit trail:

```bash
cat runs/demo/discoveries.json
cat runs/demo/experiments.json
cat runs/demo/evals.json
cat runs/demo/trace.jsonl
```

Run with Raindrop Workshop:

```bash
export RAINDROP_LOCAL_DEBUGGER=1
raindrop workshop
mechferret discover --skill ioi-circuit --out runs/raindrop-demo
```

Run on Modal:

```bash
mechferret /modal status
mechferret /modal run --skill ioi-circuit --out runs/modal-demo
```

## What Gets Written

Every run writes a small research packet:

- `report.html` - the readable report
- `report.md` - the Markdown version
- `run.json` - the full structured run
- `experiments.json` - every experiment, target, probe, effect, control, and
  backend
- `discoveries.json` - mechanisms that cleared the evidence bar
- `graph.json` - source -> evidence -> claim -> hypothesis -> discovery
- `evals.json` - self-checks for rigor
- `trace.jsonl` - the replayable trace for Raindrop or local inspection

This is a big part of the project. The output should make it easy to answer:

- What did the agent claim?
- What evidence supports it?
- What controls did it run?
- What is still uncertain?
- Could someone replay or audit this?

## Running It Locally

The core package has no required runtime dependencies and works offline on
Python 3.11+.

Install with `pipx`:

```bash
pipx install .
pipx ensurepath
```

For development:

```bash
pipx install --force --editable .
```

Or run without installing:

```bash
python3 -m mechferret discover --skill ioi-circuit --out runs/demo
```

Optional extras:

```bash
pip install -e '.[modal,interp]'      # Modal + real model backend
pip install -e '.[openai]'            # OpenAI provider research
pip install -e '.[anthropic]'         # Anthropic provider research
pip install -e '.[all]'               # everything
```

## Interactive Mode

Run:

```bash
mechferret
```

Useful commands inside the prompt:

```text
/why                         why we picked interpretability
/skills                      list built-in research playbooks
/modal status                check Modal setup
/modal run --skill ioi-circuit
/doctor                      check the local environment
/open                        open the last report
/memory --recent 5           show recent remembered runs
```

The `/why` command is especially useful for demos. It gives the casual version
of why interpretability is the right applied domain for this project.

## Skills

Skills are reusable research playbooks. They live in `mechferret/skills/` and
define things like:

- the task
- the model
- how many candidate components to screen
- which probes to run
- how many seeds to use
- what counts as "good enough"
- the compute budget

List them:

```bash
mechferret /skills
```

Show one:

```bash
mechferret /skills ioi-circuit
```

Built-in skills:

- `ioi-circuit`
- `find-induction-heads`
- `logit-lens-sweep`
- `factual-recall-trace`

## How The Research Loop Works

For readers who want the slightly more technical version, the system has two
loops.

First, there is the context loop. It loads local source documents, remembered
findings, optional URLs, and optional provider research. This gives the agent a
grounding layer so it knows what prior work already says.

Second, there is the experiment loop. This is the interpretability part:

1. Pick a task, like IOI or induction.
2. Make candidate hypotheses about model components.
3. Run a broad screen to find promising candidates.
4. Promote the best candidates into specific hypotheses.
5. Test each one from multiple angles.
6. Compare against controls.
7. Keep only results that reproduce.
8. Write discoveries and gaps.

The code for this lives mostly in:

- `mechferret/discovery.py`
- `mechferret/interp/`
- `mechferret/modal_app.py`
- `mechferret/tracing.py`

## Why The Evidence Bar Matters

Interpretability can be tempting to overclaim. A model has lots of moving parts,
and a single pretty plot can look more convincing than it really is.

MechFerret tries to avoid that by asking for several kinds of evidence:

- Did the target part have an effect?
- Did the matched control stay quiet?
- Did the result reproduce across seeds?
- Did more than one kind of probe agree?
- Did the critic find remaining gaps?

That makes the final report much more useful. It does not just say "we found a
thing." It says what was tested, what passed, what failed, and what should be
tested next.

## Other Compute Paths

Modal is the main remote GPU path for the hackathon demo.

There is also a generic SLURM path for people with their own cluster:

```bash
mechferret /cluster setup
mechferret /cluster status
mechferret /cluster run --skill ioi-circuit --dry-run
mechferret /cluster run --skill ioi-circuit
```

The point is that the research loop is portable. You can run it locally,
dispatch it to Modal, or point it at another GPU environment.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the main demo path:

```bash
python3 -m mechferret discover --skill ioi-circuit --out runs/demo
```

More docs:

- `docs/ARCHITECTURE.md`
- `docs/DEMO_SCRIPT.md`
- `mechferret/repl.py` for `/why`
- `mechferret/modal_app.py` for Modal dispatch
- `mechferret/tracing.py` for Raindrop traces
