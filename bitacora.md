# Bitacora

Date: 2026-05-30

## Goal

Build MechFerret into a hackathon-grade autonomous research system inspired by
Claude Code's architecture while keeping the implementation original and scoped
to autoresearch.

We are not copying Claude Code source verbatim. We are using its structure as a
checklist and implementing analogous MechFerret-native modules.

## Current Build State

Implemented:

- Autonomous research loop: planner, retriever, extractor, critic, synthesizer.
- Durable SQLite memory and prior-run recall.
- Local BM25 retrieval over text corpora.
- Offline seed corpus packaged inside `mechferret/seed_corpus`.
- Demo/report artifacts: `report.html`, `report.md`, `run.json`,
  `graph.json`, `evals.json`, `trace.jsonl`.
- Modal entrypoint for remote research jobs.
- Raindrop-compatible trace emission.
- OpenAI research adapter.
- Anthropic research adapter.
- Provider config file with secure-ish local permissions.
- `/login` and `/api` command aliases.
- Provider selection via `--provider local|auto|openai|anthropic`.
- Goal loop for "run until acceptance probability target" workflows.
- `goal.json` summary artifact with iteration history and next actions.
- Review-loop fixes from independent `codex review`.

In progress:

- Bitacora-driven implementation checklist against Claude Code structure.

Verified:

- `python3 -m unittest discover -s tests` passed before this provider/goal
  expansion.
- `python3 -m mechferret demo --out runs/demo` passed before this expansion.

## Claude Code Structure Checklist

Reference path:

`/Users/joyyang/Library/Mobile Documents/com~apple~CloudDocs/Downloads/claude-code-main`

Mapped architectural areas:

- Core query engine: `QueryEngine.ts`
  - MechFerret analogue: `mechferret/controller.py`, `mechferret/agents.py`.
  - Status: implemented for deterministic autoresearch, not general chat.
- Tool registry: `Tool.ts`, `tools/`
  - MechFerret analogue planned: research tool registry for search, file load,
    provider call, experiment execution, report writing.
  - Status: partial via modules, no explicit registry yet.
- Task registry: `Task.ts`, `tasks/`
  - MechFerret analogue planned: goal-loop tasks and experiment tasks.
  - Status: in progress via `goal_loop.py`.
- Commands: `commands/`
  - MechFerret analogue: `mechferret/cli.py`.
  - Status: `run`, `demo`, `inspect`, `/login`, `/api`; planned `goal`,
    `doctor`, `memory`, `status`, `cost`.
- Memory directory: `memdir/`
  - MechFerret analogue: `mechferret/memory.py`.
  - Status: implemented SQLite memory; planned typed memory and selective
    manifest views.
- Skills/plugins: `skills/`, `plugins/`
  - MechFerret analogue planned: reusable research playbooks and domain
    evaluator plugins.
  - Status: not implemented.
- Bridge/remote/server: `bridge/`, `remote/`, `server/`
  - MechFerret analogue: Modal remote entrypoint and trace mirroring.
  - Status: partial; no IDE bridge.
- Permissions/policy: `hooks/toolPermission/`, `services/policyLimits/`
  - MechFerret analogue planned: run budgets, max iterations, tool allowlist.
  - Status: partial via `goal --max-iterations` and no destructive tools.
- Cost/token tracking: `cost-tracker.ts`, `query/tokenBudget.ts`
  - MechFerret analogue planned: per-run provider calls, estimates, budget stop.
  - Status: not implemented.
- Diagnostics: `commands/doctor`, diagnostics components.
  - MechFerret analogue planned: `doctor` command for API/config/package checks.
  - Status: not implemented.
- Session/history/resume: `history.ts`, `assistant/sessionHistory.ts`,
  `commands/resume`.
  - MechFerret analogue: run artifacts and SQLite memory.
  - Status: partial; no resume command yet.
- MCP/LSP/tools integration.
  - MechFerret analogue planned: optional connectors for paper search, code
    search, experiment runners.
  - Status: not implemented.

## Near-Term Work Queue

1. Add `doctor` command.
2. Add explicit tool/task registries.
3. Add cost/budget tracking.
4. Add memory management commands.
5. Add domain evaluator templates for NeurIPS, biology, law, and coding.
6. Add skills/playbooks and plugin loading.
7. Add remote/session resume.
