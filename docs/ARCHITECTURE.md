# Architecture

MechFerret is organized around a replayable autoresearch loop:

1. **Planner** creates research facets from the user question.
2. **Retriever** builds a local BM25 index over files, URLs, optional memory,
   and optional OpenAI web-search summaries.
3. **Extractor** turns retrieved chunks into atomic claims with citations.
4. **Critic** scores coverage, source diversity, citation density,
   contradiction pressure, and unresolved gaps.
5. **Controller** expands the plan from critic gaps and repeats.
6. **Synthesizer** writes the final answer and evidence ledger.
7. **Reporter** emits `report.html`, `report.md`, `run.json`, `graph.json`,
   `evals.json`, and `trace.jsonl`.

The implementation borrows high-level product lessons from mature coding-agent
systems: task registries, explicit memory boundaries, selective recall,
side-channel traces, and critic loops. It does not copy source code from the
referenced Claude Code tree.

## Data Flow

```text
question
  -> plan steps
  -> source loader + memory recall + optional OpenAI web search
  -> BM25 chunks
  -> claims
  -> critic metrics and gaps
  -> optional plan expansion
  -> dossier + graph + evals + trace
```

## Why This Is Judge-Friendly

- It exposes internal agent state instead of only showing a final answer.
- It works offline, so the demo is not hostage to credentials.
- It has upgrade paths for sponsor technologies: OpenAI for live web search,
  Modal for distributed runs, and Raindrop Workshop for local tracing.
- It emits machine-readable artifacts that another agent can inspect, replay,
  score, or visualize.

