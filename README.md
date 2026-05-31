# MechFerret

**MechFerret is a prompt-to-paper autoresearch pipeline for interpretability.**

The goal is not "ask an AI to explain a model once." The goal is much bigger:
help more people, especially beginners, turn an interpretability idea into a
real research project and eventually a paper.

You start with a prompt like:

```text
I want to do an interpretability project on induction heads.
```

MechFerret helps turn that into:

- a research direction
- prior work and citations
- concrete hypotheses
- experiments to run
- evidence tracking
- follow-up gaps
- a paper outline or LaTeX draft
- a reviewer-style critique
- a trace of what happened along the way

In other words: **prompt -> research plan -> experiments -> evidence -> paper**.

That is the product. MechFerret is meant to make interpretability research feel
less like "you need to already be an expert with a giant notebook stack" and
more like "you can start with an idea and get pulled into a real research
workflow."

## Why Interpretability?

Interpretability is one of the best possible domains for an autoresearch agent.

AI is becoming more important every year, but we still do not understand these
models nearly well enough. Interpretability is the field trying to change that.
It studies what models learn, how internal features work, what circuits appear,
why behaviors emerge, and how we can make AI systems easier to understand and
debug.

That matters for humanity.

Better interpretability means:

- safer AI systems
- more understandable failures
- better debugging tools
- better tools for studying model internals
- stronger scientific foundations for AI
- more people able to contribute to AI safety and model understanding

This is why we picked interpretability as the applied domain. If autoresearch
can speed up interpretability, it does not just help one field publish more
papers. It helps the field that tries to understand the systems everyone else
will build on top of.

There is also a huge access problem. Interpretability is exciting, but it is
hard to enter. A beginner has to figure out the literature, pick a problem, set
up experiments, understand what counts as evidence, avoid overclaiming, and
write the work up in a way that looks like a real ML paper.

MechFerret is built for that gap.

It is not replacing researchers. It is giving new researchers a guided path
from "I have an idea" to "I have a structured project with evidence, gaps, and a
paper draft."

## What MechFerret Actually Is

MechFerret is an interpretability research operating system.

It has a conversational interface, but the useful part is the pipeline behind
it:

```text
prompt
  -> understand the research goal
  -> search and summarize prior work
  -> suggest concrete interp directions
  -> turn a direction into hypotheses
  -> run or script experiments
  -> track evidence and mechanisms
  -> show the evidence architecture
  -> draft a paper
  -> review the paper
  -> keep memory so the project can continue
```

The system is designed around the shape of an actual research project, not just
a one-off answer.

Some examples of what it can do:

- replay a recorded research trace with `/demo`
- explain why interpretability is the domain with `/why`
- show how evidence supports claims with `/arch`
- generate `paper/main.tex` with `/paper`
- ask an independent reviewer agent to score the draft with `/review-paper`
- run discovery-style interpretability experiments with `/discover`
- remember confirmed findings and experiment results across sessions
- stream trace events to Raindrop Workshop
- dispatch heavier experiment work to Modal

The important shift is this: MechFerret is not just "an assistant that knows
about interpretability." It is a workflow for producing interpretability
research artifacts.

## Who This Is For

MechFerret is especially for people who want to do interpretability research but
do not yet know how to turn an idea into a paper.

That includes:

- beginners trying to enter mechanistic interpretability
- hackathon teams trying to build a serious research demo quickly
- researchers who want a faster way to scaffold projects
- people with an interp idea who need help turning it into experiments
- anyone who wants a traceable research pipeline instead of a one-off chat
  transcript

The dream version is simple: a motivated beginner can show up with curiosity,
use MechFerret to pick a tractable direction, run the loop, and end up with
something close to a publishable artifact.

Not a guaranteed accepted paper. But a real path toward one.

## The Hackathon Story

This is built for the Raindrop autoresearch hackathon, specifically **Track 3:
Applied Autonomous Research**.

Our applied domain is interpretability because interpretability is good for
humanity, hard to enter, and perfect for research automation.

The project we want to show is:

> An autoresearch agent that helps interpretability research itself.

That is different from a normal literature assistant. A literature assistant
can summarize papers. MechFerret tries to move the whole research process
forward: idea, plan, prior work, experiments, evidence, draft, review, iterate.

What judges should see:

- **Applied domain:** interpretability, because understanding AI systems is
  important and more people should be able to contribute.
- **Autonomous research loop:** the system does more than answer questions; it
  keeps state, runs tools, records evidence, and writes outputs.
- **Beginner leverage:** it helps someone go from prompt to paper-shaped
  research instead of staring at a blank repo.
- **Raindrop fit:** the agent's steps are visible as a trace, not hidden in a
  chat transcript.
- **Modal fit:** when research needs compute, experiments can move to a Modal
  GPU while the same project loop stays intact.

## Why Not Just A Claude Code Skill?

A Claude Code skill is useful for giving an assistant instructions. You can
write a skill that says:

```text
When doing interpretability research, read prior work, propose hypotheses,
write experiments, and draft a paper.
```

That is helpful, but it is not enough.

MechFerret is better because it turns the workflow into an actual system. The
research state lives outside the assistant's vibes. It has commands, memory,
traces, artifacts, experiment ledgers, and paper generation.

The difference is:

| Claude Code skill | MechFerret |
| --- | --- |
| Tells an assistant what to do. | Gives the researcher an end-to-end pipeline. |
| Lives mostly as instructions in context. | Stores project state, memory, traces, and artifacts. |
| Can suggest a paper structure. | Writes `paper/main.tex` from recorded findings. |
| Can remind you to evaluate evidence. | Tracks experiments, mechanisms, claims, gaps, and drift. |
| Can say "run this on a GPU." | Has Modal dispatch for heavier runs. |
| Produces a chat unless you manually organize outputs. | Produces reports, JSON ledgers, traces, and paper files. |
| Helps an expert move faster. | Also helps a beginner know what the next research step is. |

The key point: a skill is a recipe. MechFerret is the kitchen.

For a beginner, that matters a lot. They do not just need a list of best
practices. They need a guided path, persistent project memory, commands that
produce real files, and a way to see whether their evidence is paper-worthy.

That is why MechFerret is more than a Claude Code skill. It is infrastructure
for making interpretability research easier to start, continue, audit, and
write up.

## How The Pipeline Feels

In the interactive prompt:

```bash
mechferret
```

You can do things like:

```text
/why            explain why interpretability is the right domain
/demo           replay this project's research trace
/arch           show how the evidence supports the claims
/paper          generate paper/main.tex
/review-paper   have a separate reviewer agent critique the paper
/trace          show recent trace events
/memory         show remembered findings
```

The intended loop is:

1. Start with a research prompt.
2. Let the agent help shape it into a tractable interp project.
3. Use tools and experiments to collect evidence.
4. Use `/arch` to see whether the evidence actually supports the claim.
5. Use `/paper` to turn the project into a draft.
6. Use `/review-paper` to get a harsh review.
7. Iterate.

That is the prompt-to-paper story.

## Modal

Some parts of interpretability research are light: reading papers, planning,
writing, reviewing, organizing evidence.

Some parts need real compute.

Modal is the compute path. When the project needs heavier model experiments,
MechFerret can dispatch work to a Modal GPU container instead of pretending
everything should run on a laptop.

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

For the hackathon, Modal shows that the pipeline is not only a local scripted
demo. It has a path to real compute when an interpretability project needs it.

## Raindrop Workshop

Raindrop is the visibility layer.

For autoresearch, the trace is part of the product. We do not just want a final
paper draft. We want to see what the agent did to get there: what it searched,
what tools it ran, what evidence it recorded, where it changed direction, and
what artifacts it wrote.

MechFerret writes trace events to:

```text
.mechferret/trace.jsonl
```

Discovery runs also write traces under their run directory, such as:

```text
runs/demo/trace.jsonl
```

To stream spans into Raindrop Workshop:

```bash
export RAINDROP_LOCAL_DEBUGGER=1
raindrop workshop
mechferret
```

By default, traces post to:

```text
http://127.0.0.1:5899/v1/traces
```

The local trace file is always written, even if Workshop is not running. So the
demo can be live and reviewable, but the pipeline does not depend on the
debugger being up.

## Quickstart

Install:

```bash
pipx install .
pipx ensurepath
```

Open the prompt:

```bash
mechferret
```

Useful demo flow:

```text
/why
/demo
/arch
/paper
/review-paper
```

Run the deterministic discovery path:

```bash
mechferret discover --skill ioi-circuit --out runs/demo
open runs/demo/report.html
```

Run on Modal:

```bash
mechferret /modal status
mechferret /modal run --skill ioi-circuit --out runs/modal-demo
```

## What Gets Produced

Depending on the path you run, MechFerret can produce:

- `.mechferret/memory.sqlite` - project memory
- `.mechferret/trace.jsonl` - replayable trace
- `paper/main.tex` - generated paper draft
- `paper/main.pdf` - compiled paper, if `tectonic` is installed
- `runs/*/report.html` - readable research report
- `runs/*/report.md` - Markdown report
- `runs/*/run.json` - full structured run
- `runs/*/experiments.json` - experiment records
- `runs/*/discoveries.json` - confirmed findings
- `runs/*/graph.json` - evidence graph
- `runs/*/evals.json` - self-checks and rigor checks

The point of these files is to make the research legible. A beginner should be
able to see what happened, what is supported, what is weak, and what to do next.

## Skills

Skills are reusable research playbooks in `mechferret/skills/`.

They define things like:

- the task
- the model
- the experiment budget
- the evidence bar
- the stopping condition

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

Skills are useful, but they are only one piece. The bigger value is that
MechFerret wraps skills in memory, tracing, reports, paper generation, review,
and compute dispatch.

## Local Setup

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
python3 -m mechferret
```

Optional extras:

```bash
pip install -e '.[modal,interp]'      # Modal + real model backend
pip install -e '.[openai]'            # OpenAI provider research
pip install -e '.[anthropic]'         # Anthropic provider research
pip install -e '.[all]'               # everything
```

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the prompt:

```bash
python3 -m mechferret
```

More docs:

- `docs/ARCHITECTURE.md`
- `docs/DEMO_SCRIPT.md`
- `mechferret/repl.py` for `/why`, `/paper`, and `/review-paper`
- `mechferret/modal_app.py` for Modal dispatch
- `mechferret/tracing.py` for Raindrop traces
