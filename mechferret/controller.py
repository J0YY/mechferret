from __future__ import annotations

import uuid
from pathlib import Path

from .agents import ClaimExtractor, Critic, Planner, Synthesizer
from .config import load_config
from .llm import make_research_adapter
from .memory import ResearchMemory
from .models import EvidenceChunk, ResearchRun, Source, utc_now
from .report import write_artifacts
from .retrieval import BM25Index
from .sources import example_corpus_path, load_sources
from .tracing import TraceRecorder


class MechFerret:
    def __init__(self, memory_path: str | Path = ".mechferret/memory.sqlite") -> None:
        self.memory_path = Path(memory_path)
        self.planner = Planner()
        self.extractor = ClaimExtractor()
        self.critic = Critic()
        self.synthesizer = Synthesizer()

    def run(
        self,
        question: str,
        source_paths: list[str] | None = None,
        urls: list[str] | None = None,
        out_dir: str | Path = "runs/latest",
        max_rounds: int = 2,
        use_openai: bool = False,
        provider: str = "auto",
        model: str | None = None,
        include_memory: bool = True,
    ) -> ResearchRun:
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        out_path = Path(out_dir)
        tracer = TraceRecorder(run_id, out_path)
        memory = ResearchMemory(self.memory_path)
        try:
            with tracer.span("load_sources", source_paths=source_paths or [], urls=urls or []):
                sources = load_sources(source_paths, urls)
                requested_sources = bool(source_paths or urls)
                if not sources and requested_sources:
                    raise ValueError(
                        "No supported text sources were loaded from the requested paths or URLs. "
                        "Use .md, .txt, .rst, .html, .json, or .csv files, or omit --source to run the demo corpus."
                    )
                if not sources:
                    sources = load_sources([str(example_corpus_path())], [])
                if include_memory:
                    sources.extend(memory.recall_sources(question))
                memory.upsert_sources(sources)

            with tracer.span("plan", question=question):
                plan = self.planner.plan(question)

            evidence: list[EvidenceChunk] = []
            claims_by_id = {}
            config = load_config()
            selected_provider = "openai" if use_openai else provider
            research_adapter = make_research_adapter(selected_provider, model, config)
            for round_index in range(max(1, max_rounds)):
                with tracer.span("round", round=round_index + 1, steps=len(plan.steps), sources=len(sources)):
                    index = BM25Index.from_sources(sources)
                    round_chunks: list[EvidenceChunk] = []
                    for step in plan.steps:
                        chunks = index.search(step.question, limit=7)
                        round_chunks.extend(chunks)
                        step.status = "searched"
                        step.notes.append(f"retrieved {len(chunks)} chunks in round {round_index + 1}")
                    evidence = dedupe_evidence(evidence + round_chunks)

                    if research_adapter and round_index == 0:
                        with tracer.span(
                            "provider_research",
                            provider=selected_provider,
                            available=research_adapter.available,
                        ):
                            live_source = research_adapter.search_summary(question)
                            if live_source:
                                sources.append(live_source)
                                memory.upsert_sources([live_source])
                                tracer.event("provider_source_added", source_id=live_source.id, provider=selected_provider)
                                index = BM25Index.from_sources(sources)
                                evidence = dedupe_evidence(evidence + index.search(question, limit=8))

                    with tracer.span("extract_claims", evidence=len(evidence)):
                        extracted = self.extractor.extract(question, evidence, limit=24)
                        for claim in extracted:
                            claims_by_id[claim.id] = claim

                    with tracer.span("critique", claims=len(claims_by_id)):
                        claims = list(claims_by_id.values())
                        gaps, contradictions, metrics = self.critic.evaluate(question, plan, claims, evidence)
                        if round_index < max_rounds - 1 and gaps:
                            self.planner.expand_for_gaps(plan, gaps)

            claims = sorted(claims_by_id.values(), key=lambda c: (c.confidence, c.support_score), reverse=True)
            gaps, contradictions, metrics = self.critic.evaluate(question, plan, claims, evidence)
            with tracer.span("synthesize", claims=len(claims), gaps=len(gaps)):
                answer = self.synthesizer.synthesize(question, claims, evidence, gaps)
                run = ResearchRun(
                    run_id=run_id,
                    question=question,
                    created_at=utc_now(),
                    plan=plan,
                    sources=sources,
                    evidence=evidence,
                    claims=claims,
                    contradictions=contradictions,
                    gaps=gaps,
                    answer=answer,
                    metrics=metrics,
                )
                artifacts = write_artifacts(run, out_path)
                run.artifacts.update(artifacts)
                write_artifacts(run, out_path)
                memory.record_run(run)
                tracer.event("artifacts_written", **artifacts)
                return run
        finally:
            memory.close()


def dedupe_evidence(chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    by_id: dict[str, EvidenceChunk] = {}
    for chunk in chunks:
        current = by_id.get(chunk.id)
        if current is None or chunk.score > current.score:
            by_id[chunk.id] = chunk
    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)
