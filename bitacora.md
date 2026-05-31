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
- Provenance verification for run ledgers, source manifests, sidecar artifacts,
  bundle manifests, and exported report/paper/review/PDF/trace artifacts.
- Strict provenance byte metadata validation for local and archived manifests,
  including precise declaration failures before value comparisons.
- Bundle verification refuses duplicate archive entry names before semantic
  reads, preventing ambiguous package contents from being treated as valid.
- Bundle manifest rows now require non-empty string archive paths and skip
  content reads for unsafe or malformed paths.
- Bundle semantic checks resolve labels only to safe archive paths, so unsafe
  metadata entries cannot be parsed through deeper audit/status/readme checks.
- Artifact resolution reports the active selection policy and selected run,
  making agent and CLI decisions inspectable when using `best` or `ready`.
- Run listing now reports selection failure actions when runs exist but none
  satisfy the requested `ready`/artifact policy.
- Run-dependent agent tools return consistent selection failure payloads with
  policy, runs root, failed check, and recovery actions.
- Agent tools validate selection policy inputs and return JSON recovery
  guidance instead of raw exceptions for malformed values.
- Agent list tools validate numeric limits and return structured recovery
  guidance for malformed or negative values.
- File-reading and tool-result cleanup validate offsets, limits, retention
  counts, and ages before executing.
- Command, web/arXiv, and research tools validate timeout, result-count,
  character-count, and round-count arguments before executing.
- OpenVLA SAE agent actions validate smoke-test and feature-report numeric
  arguments before invoking project workflows.
- Agent tools validate required string arguments for file, shell, search,
  glob/grep, and Neuronpedia actions before execution.
- File and search agent tools validate mutation content, edit replacement, and
  optional directory/search path filters before filesystem or subprocess calls.
- Research and discovery agent tools validate source path/URL lists, run
  metadata strings, provider choices, and discovery backend choices before
  controller dispatch.
- Novelty and option-presentation tools validate required idea text, query
  lists, and option rows before search or UI fallback handling.
- Agent tools validate boolean flags for cleanup, editing, research/discovery,
  run listing, verification, paper compilation, and OpenVLA manifest actions.
- Agent tools validate provider, arXiv sort, and OpenVLA action choices before
  executing downstream workflows.
- OpenVLA SAE agent manifest actions validate required and optional string
  fields, parse row limits, and return structured recovery payloads for missing
  image directories or existing manifests.
- OpenVLA SAE agent artifact actions validate optional path fields before
  dispatching plan, smoke, eval, feature, and dossier workflows.
- Run and bundle agent tools validate path-like string fields before dispatching
  project status, run selection, paper, bundle, verification, and artifact
  resolution workflows.
- Paper and review agent tools validate optional model override strings before
  provider dispatch.
- Tool dispatch returns structured recovery payloads for unknown tools and
  handler exceptions.
- Agent dispatch returns structured recovery payloads for denied, aborted, and
  malformed tool calls.
- Agent dispatch rejects unknown provider tool names before permission checks
  or callback notifications.
- OpenAI tool-loop malformed argument JSON is surfaced as a structured tool
  result instead of silently dispatching with empty defaults.
- OpenAI tool-loop malformed tool-call envelopes are surfaced as structured
  tool results without invoking the requested tool.
- Anthropic tool-loop malformed content/tool-use blocks are surfaced as
  structured tool results without invoking malformed calls.
- Provider tool loops reject repeated tool-call IDs before dispatch, preventing
  ambiguous model envelopes from overwriting or double-reporting tool results.
- Provider tool loops now return a clear step-cap message instead of empty
  output when a model keeps requesting tools without a final answer.
- Provider text extraction now tolerates mixed string/list content blocks for
  live replies, compaction summaries, and transcript rendering.
- Automatic context compaction failures are traced but no longer block the next
  user turn from reaching the provider.
- Session persistence rejects path-like IDs and resumed agents sanitize corrupt
  message/cost payloads before restoring live state.
- Session listing tolerates corrupt-but-parseable transcript metadata so one
  bad file cannot break resume discovery.
- Direct session resume ignores embedded invalid IDs/providers instead of
  carrying corrupt transcript identity into live agent state.
- Session saves normalize non-JSON transcript values into readable JSON instead
  of losing persistence when provider/tool payloads contain richer objects.
- Session saves and listings sanitize non-finite numeric values and write strict
  JSON so resumed transcripts remain portable across JSON parsers.
- Direct session loading rejects non-object transcript JSON before agent resume
  mutates live state.
- Transcript persistence failures now emit trace events while preserving
  best-effort chat progress.
- Trace emission normalizes rich attributes, writes strict JSON, and treats
  local trace write failures as best-effort observability.
- Provider usage accounting ignores malformed, non-finite, or negative token
  counts instead of blocking otherwise valid replies.
- Provider response envelopes are validated before indexing provider-specific
  fields, so malformed OpenAI/Anthropic replies produce clear recoverable
  messages and trace events instead of crashes or blank turns.
- OpenAI tool-call lists are shape-checked before dispatch, preventing malformed
  provider envelopes from creating unmatched synthetic tool responses.
- Provider HTTP helpers report malformed response bodies with bounded snippets
  and paper/synthesis adapters share the chat agent's provider envelope checks.
- Cost tracking now tolerates malformed model/usage payloads and run-cost
  estimates ignore malformed artifact rows instead of failing the CLI.
- Trace recorder startup is best-effort when local trace directories are
  unavailable, and session listing normalizes malformed limits for resume views.
- Session loading now wraps corrupt transcript JSON in clean resume errors, and
  session listing skips entries whose embedded and filename IDs are both invalid.
- Memory storage and recall sanitize malformed mechanism, experiment, metric,
  artifact, and numeric rows so prior context cannot break agent startup or CLI views.
- Experiment-memory ingestion skips malformed hypothesis/result rows while
  preserving drift accounting for valid repeated experiment specs.
- Memory run/source persistence skips malformed source and claim rows while
  preserving usable run metadata and strict JSON-safe artifact fields.
- Provider config loading now falls back cleanly on corrupt files, sanitizes
  malformed provider settings, and saves only supported provider entries.
- MCP connector config now filters malformed server rows and tool specs, and
  add-server rewrites corrupt config files into a valid shape.
- Cluster config loading and SSH/srun command builders now coerce malformed
  local config and environment values back to safe defaults.
- Skill/playbook loading now skips malformed listed JSON files, sanitizes
  optional budget/reference fields, and reports explicit bad skill loads clearly.
- Retrieval now skips malformed source rows and normalizes chunk/search limits,
  protecting literature search and memory recall from bad local rows.
- Agent tool summaries that preserve structured verifier check rows while
  persisting oversized full JSON outputs.
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
  - Status: implemented for repeatable autoresearch, not general chat.
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

## Current Hardening Notes

- Text helpers now tolerate malformed text values, byte strings, and invalid
  truncation/ID lengths so retrieval, reports, and source dedupe keep running
  when upstream payloads are partially corrupted.
- Audit artifact loading now filters malformed nested rows and report/eval
  rendering normalizes malformed numeric fields instead of crashing status,
  bundle, or dossier views.
- Knowledge/search helpers now sanitize malformed public-service responses,
  bad XML/JSON payloads, empty inputs, and oversized/bad limits before agent
  web, arXiv, or Neuronpedia tools consume them.
- Source ingestion now skips empty malformed path/URL entries, normalizes bad
  URL fetch parameters, tolerates malformed title/text inputs, and keeps source
  dedupe faithful to full text while avoiding malformed-row collisions.
- Literature and discovery run controllers now normalize direct API boundary
  values for questions, path/URL lists, provider/model fields, booleans, round
  counts, and discovery budgets before assembling runs or provenance.
- Budget guards now sanitize malformed direct budget fields, counters, result
  rows, and permission booleans so autonomous loops keep bounded behavior even
  outside controller-created budgets.
- Coordinator fan-out now normalizes malformed worker counts, backend labels,
  non-callable functions, and invalid item collections before dispatching
  serial or threaded experiment work.
- Interpretability experiment execution now sanitizes malformed specs, seeds,
  targets, backend fields, probe readings, and batch collections into explicit
  experiment error rows instead of aborting discovery loops.
- Planner, claim extraction, critique, synthesis, citation labeling, and claim
  merging now tolerate malformed evidence/claim rows, bad numeric scores,
  malformed gaps, and invalid limits before scoring dossiers.
- Hypothesis screening, promotion, status updates, and head-role labeling now
  normalize malformed task/model/seed inputs, source IDs, result flags,
  effect sizes, targets, metrics, and hypothesis rows before planning or
  updating experiment evidence.
- Experiment-critic scoring now filters malformed hypothesis/result collections
  and parses status, flags, IDs, targets, and numeric metrics before computing
  rigor gaps or run stop metrics.
- Discovery artifact construction now skips malformed hypothesis/result rows,
  requires at least two distinct confirming probe types for emitted
  discoveries, and normalizes evidence chunks and discovery claims for the
  experiment ledger.
- Discovery readiness scoring and final answer synthesis now normalize
  malformed metric maps, discovery rows, numeric fields, and gap lists before
  producing user-facing run summaries.
- Discovery loop dispatch now normalizes prior-source IDs, backend labels,
  skill thresholds, promoted hypothesis targets, worker result rows, and stop
  metrics before updating run state or deciding to halt.
- Provider-authored answer prompts now sanitize malformed claims, evidence,
  discoveries, experiments, gaps, numeric fields, and JSON payload values before
  sending the run ledger to OpenAI or Anthropic.
- Live OpenAI/Anthropic research adapters now normalize question/domain inputs,
  decode provider text blocks, and treat import/provider failures as absent
  optional context instead of aborting runs.
- Literature controller runs now normalize memory/provider sources, extracted
  claims, contradictions, evidence chunks, gaps, and metric maps before
  artifact writing or manifest refresh.
- Report artifact writers now normalize malformed public-output rows, graph
  fields, eval inputs, sidecar JSON values, source manifests, and artifact path
  maps before rendering or manifest digest emission.
- Paper generation now sanitizes malformed run rows, prompt JSON values,
  artifact maps, optional path/provider inputs, review records, and TeX compile
  arguments before writing run-bound paper or review artifacts.
- Ops artifact selection, status, bundle, verification, memory, and summary
  helpers now normalize malformed path, policy, limit, artifact, and run-ledger
  values before scanning runs or reporting recovery actions.
- CLI run-artifact helpers now inspect malformed run JSON through the sanitized
  summary path, preserve selected-run bundle metadata, and accept a direct
  `run.json` wherever status/listing helpers search for run artifacts.
- OpenVLA SAE workflows now normalize malformed manifest rows, path-like
  inputs, booleans, numeric limits, metrics JSON, tensor-inspection failures,
  and report payloads before writing plans, evals, feature reports, or dossiers.
- Local paper generation now renders evidence-bound TeX from run ledgers,
  including plan, results, experiment, evidence, artifact, gap, contradiction,
  and metric sections, while the doctor gate rejects placeholder regressions.
- Paper generation now supports `--json`, returns an explicit `ok` flag, and
  documents local-mode drafts as evidence-bound run-ledger manuscripts for
  scripted prompt-to-paper workflows.
- Audit readiness now rejects run-bound `paper/main.tex` files that lack the
  expected article body and evidence/experiment/limitations sections, and
  bundle audit verification mirrors that stricter paper gate.
- Provenance and bundle verification now use the same evidence-bound paper
  structure markers as audit, so boundary-only TeX cannot pass after manifest
  refresh or archive repackaging.
- `mechferret paper --json` now returns structured `ok: false` recovery
  payloads for missing run roots or explicit missing `run.json` paths instead
  of emitting plain stderr before exiting.
- `mechferret review-paper --json` now preserves missing-paper error,
  selection, runs-root, selected-run, resolver reason, and requested-path
  context so automation can recover without parsing human text.
- `mechferret cost --json` now returns structured cost estimates and
  structured missing-run recovery payloads, matching the paper/review
  automation contract.
- `mechferret resume --json` and `mechferret inspect --json` now return
  structured run summaries or structured missing-run recovery payloads, keeping
  the run-artifact helper commands scriptable end to end.
- `mechferret registry --json` and `mechferret memory --json` now expose
  machine-readable capability inventory, memory summaries, recent rows, and
  clear confirmations while preserving the existing human output.
- `mechferret skills --json` now exposes machine-readable playbook lists and
  detailed skill budgets, stop thresholds, seeds, references, and task metadata.
- `mechferret api --json` now exposes redacted provider configuration status,
  update/default/clear actions, and structured setup errors without printing
  stored API keys.
