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
- `/goal` and `/loop` aliases for the primary autonomy loop.
- Explicit registries for tools, tasks, playbooks, and evaluators.
- `doctor`, `registry`, `memory`, `cost`, and `resume` commands.
- Domain evaluator templates for NeurIPS, biology, law, and coding.
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
  - Status: implemented in `mechferret/registry.py`.
- Task registry: `Task.ts`, `tasks/`
  - MechFerret analogue planned: goal-loop tasks and experiment tasks.
  - Status: implemented in `mechferret/registry.py` and `mechferret/goal_loop.py`.
- Commands: `commands/`
  - MechFerret analogue: `mechferret/cli.py`.
  - Status: `run`, `demo`, `inspect`, `/login`, `/api`, `/goal`, `/loop`,
    `/doctor`, `/registry`, `/memory`, `/cost`, `/resume`.
- Memory directory: `memdir/`
  - MechFerret analogue: `mechferret/memory.py`.
  - Status: implemented SQLite memory; planned typed memory and selective
    manifest views.
- Skills/plugins: `skills/`, `plugins/`
  - MechFerret analogue planned: reusable research playbooks and domain
    evaluator plugins.
  - Status: partial via evaluator templates and playbook registry.
- Bridge/remote/server: `bridge/`, `remote/`, `server/`
  - MechFerret analogue: Modal remote entrypoint and trace mirroring.
  - Status: partial; no IDE bridge.
- Permissions/policy: `hooks/toolPermission/`, `services/policyLimits/`
  - MechFerret analogue planned: run budgets, max iterations, tool allowlist.
  - Status: partial via `goal --max-iterations` and no destructive tools.
- Cost/token tracking: `cost-tracker.ts`, `query/tokenBudget.ts`
  - MechFerret analogue planned: per-run provider calls, estimates, budget stop.
  - Status: partial via artifact-based cost estimate and loop iteration budget.
- Diagnostics: `commands/doctor`, diagnostics components.
  - MechFerret analogue planned: `doctor` command for API/config/package checks.
  - Status: implemented in `mechferret/ops.py`.
- Session/history/resume: `history.ts`, `assistant/sessionHistory.ts`,
  `commands/resume`.
  - MechFerret analogue: run artifacts and SQLite memory.
  - Status: partial with `/resume` artifact summary and `/memory --recent`.
- MCP/LSP/tools integration.
  - MechFerret analogue planned: optional connectors for paper search, code
    search, experiment runners.
  - Status: not implemented.

## Near-Term Work Queue

1. Add stronger `/loop` budget controls: max provider calls, max dollars,
   wall-clock time, and stop-on-regression.
2. Add experiment runners that can execute scripts/notebooks and feed metrics
   back into acceptance probability.
3. Add real plugin loading for evaluator templates and search connectors.
4. Add richer memory commands: export, prune, typed memories, and stale-memory
   checks.
5. Add remote/session resume beyond artifact summary.
6. Add MCP-style connector layer for paper search, code search, and datasets.
