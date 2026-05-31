from __future__ import annotations

import uuid
import math
from pathlib import Path
from typing import Any

from .agents import ClaimExtractor, Critic, Planner, Synthesizer
from .config import load_config
from .llm import make_research_adapter, synthesize_answer_with_provider
from .memory import ResearchMemory
from .models import Claim, Contradiction, EvidenceChunk, ResearchRun, Source, utc_now
from .provenance import refresh_run_manifest
from .report import write_artifacts
from .retrieval import BM25Index
from .sources import example_corpus_path, load_sources
from .tracing import TraceRecorder


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, bytes, dict)) or value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _strings(value: Any) -> list[str]:
    return [_text(item).strip() for item in _items(value) if _text(item).strip()]


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _source(value: Any) -> Source | None:
    source_id = _text(getattr(value, "id", "")).strip()
    text = _text(getattr(value, "text", "")).strip()
    if not source_id or not text:
        return None
    if isinstance(value, Source):
        value.id = source_id
        value.title = _text(getattr(value, "title", "")).strip() or source_id
        value.text = text
        value.url = _text(getattr(value, "url", "")).strip()
        value.kind = _text(getattr(value, "kind", "")).strip() or "document"
        value.metadata = getattr(value, "metadata", {}) if isinstance(getattr(value, "metadata", {}), dict) else {}
        return value
    return Source(
        id=source_id,
        title=_text(getattr(value, "title", "")).strip() or source_id,
        text=text,
        url=_text(getattr(value, "url", "")).strip(),
        kind=_text(getattr(value, "kind", "")).strip() or "document",
        metadata=getattr(value, "metadata", {}) if isinstance(getattr(value, "metadata", {}), dict) else {},
    )


def _sources(value: Any) -> list[Source]:
    rows: list[Source] = []
    for item in _items(value):
        row = _source(item)
        if row is not None:
            rows.append(row)
    return rows


def _claims(value: Any) -> list[Any]:
    rows: list[Claim] = []
    for claim in _items(value):
        claim_id = _text(getattr(claim, "id", "")).strip()
        text = _text(getattr(claim, "text", "")).strip()
        if not claim_id or not text:
            continue
        if isinstance(claim, Claim):
            claim.id = claim_id
            claim.text = text
            claim.citations = _strings(getattr(claim, "citations", []))
            claim.source_ids = _strings(getattr(claim, "source_ids", []))
            claim.confidence = _number(getattr(claim, "confidence", 0.0))
            claim.support_score = _number(getattr(claim, "support_score", 0.0))
            claim.stance = _text(getattr(claim, "stance", "")).strip() or "finding"
            claim.quality_flags = _strings(getattr(claim, "quality_flags", []))
            rows.append(claim)
        else:
            rows.append(
                Claim(
                    claim_id,
                    text,
                    _strings(getattr(claim, "citations", [])),
                    _strings(getattr(claim, "source_ids", [])),
                    _number(getattr(claim, "confidence", 0.0)),
                    _number(getattr(claim, "support_score", 0.0)),
                    stance=_text(getattr(claim, "stance", "")).strip() or "finding",
                    quality_flags=_strings(getattr(claim, "quality_flags", [])),
                )
            )
    return rows


def _contradictions(value: Any) -> list[Any]:
    rows: list[Contradiction] = []
    for row in _items(value):
        row_id = _text(getattr(row, "id", "")).strip()
        claim_a = _text(getattr(row, "claim_a", "")).strip()
        claim_b = _text(getattr(row, "claim_b", "")).strip()
        reason = _text(getattr(row, "reason", "")).strip()
        if not row_id or not claim_a or not claim_b:
            continue
        if isinstance(row, Contradiction):
            row.id = row_id
            row.claim_a = claim_a
            row.claim_b = claim_b
            row.reason = reason
            row.severity = _number(getattr(row, "severity", 0.0))
            rows.append(row)
        else:
            rows.append(Contradiction(row_id, claim_a, claim_b, reason, _number(getattr(row, "severity", 0.0))))
    return rows


def _metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, item in value.items():
        label = _text(key).strip()
        if label:
            metrics[label] = _number(item)
    return metrics


def _path_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes, Path)):
        text = _text(value) if not isinstance(value, Path) else str(value)
        return [text.strip()] if text.strip() else []
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _text(item) if not isinstance(item, Path) else str(item)
        text = text.strip()
        if text:
            items.append(text)
    return items


def _positive_int(value: Any, default: int, *, upper: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        parsed = default
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def _bool(value: Any, default: bool) -> bool:
    return value if type(value) is bool else default


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
        allow_seed_corpus: bool = False,
    ) -> ResearchRun:
        question = _text(question).strip()
        source_paths = _path_list(source_paths)
        urls = _path_list(urls)
        max_rounds = _positive_int(max_rounds, 2, upper=50)
        use_openai = _bool(use_openai, False)
        include_memory = _bool(include_memory, True)
        allow_seed_corpus = _bool(allow_seed_corpus, False)
        provider = _text(provider).strip() or "auto"
        model = _text(model).strip() or None
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        out_path = Path(out_dir)
        try:
            (out_path / "trace.jsonl").unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        tracer = TraceRecorder(run_id, out_path)
        memory = ResearchMemory(self.memory_path)
        try:
            with tracer.span("load_sources", source_paths=source_paths, urls=urls):
                sources = _sources(load_sources(source_paths, urls))
                requested_sources = bool(source_paths or urls)
                used_packaged_seed_corpus = False
                if not sources and requested_sources:
                    raise ValueError(
                        "No supported text sources were loaded from the requested paths or URLs. "
                        "Use .md, .txt, .rst, .html, .json, or .csv files."
                    )
                if include_memory:
                    sources.extend(_sources(memory.recall_sources(question)))
                if not sources and allow_seed_corpus:
                    sources = _sources(load_sources([str(example_corpus_path())], []))
                    used_packaged_seed_corpus = True
                memory.upsert_sources(sources)

            with tracer.span("plan", question=question):
                plan = self.planner.plan(question)

            evidence: list[EvidenceChunk] = []
            claims_by_id = {}
            config = load_config()
            selected_provider = "openai" if use_openai else provider
            research_adapter = make_research_adapter(selected_provider, model, config)
            provider_source_added = False
            if not sources and not (research_adapter and research_adapter.available):
                raise ValueError(
                    "No source material is available for this question. Add --source/--url, use --openai or a configured provider "
                    "for live research, reuse memory from a prior run, or pass --seed-corpus for the packaged demo corpus."
                )
            for round_index in range(max_rounds):
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
                            live_source = _source(live_source)
                            if live_source:
                                sources.append(live_source)
                                memory.upsert_sources([live_source])
                                provider_source_added = True
                                tracer.event("provider_source_added", source_id=live_source.id, provider=selected_provider)
                                index = BM25Index.from_sources(sources)
                                evidence = dedupe_evidence(evidence + index.search(question, limit=8))

                    with tracer.span("extract_claims", evidence=len(evidence)):
                        extracted = self.extractor.extract(question, evidence, limit=24)
                        for claim in _claims(extracted):
                            claims_by_id[_text(getattr(claim, "id", "")).strip()] = claim

                    with tracer.span("critique", claims=len(claims_by_id)):
                        claims = _claims(claims_by_id.values())
                        gaps, contradictions, metrics = self.critic.evaluate(question, plan, claims, evidence)
                        if round_index < max_rounds - 1 and gaps:
                            self.planner.expand_for_gaps(plan, gaps)

            claims = sorted(
                _claims(claims_by_id.values()),
                key=lambda c: (_number(getattr(c, "confidence", 0.0)), _number(getattr(c, "support_score", 0.0))),
                reverse=True,
            )
            gaps, contradictions, metrics = self.critic.evaluate(question, plan, claims, evidence)
            gaps = _strings(gaps)
            contradictions = _contradictions(contradictions)
            metrics = _metrics(metrics)
            with tracer.span("synthesize", claims=len(claims), gaps=len(gaps)):
                answer = self.synthesizer.synthesize(question, claims, evidence, gaps)
                answer_author = "local_extractive_synthesizer"
                answer_provider = ""
                answer_model = ""
                answer_note = ""
                if selected_provider in {"openai", "anthropic"} and research_adapter and research_adapter.available:
                    provider_answer, answer_diag = synthesize_answer_with_provider(
                        selected_provider,
                        model,
                        question=question,
                        claims=claims,
                        evidence=evidence,
                        gaps=gaps,
                        config=config,
                    )
                    if provider_answer:
                        answer = provider_answer
                        answer_author = "provider_model"
                        answer_provider = answer_diag.get("provider", "")
                        answer_model = answer_diag.get("model", "")
                    else:
                        answer_note = answer_diag.get("reason", "")
            try:
                from . import __version__ as version
            except ImportError:
                version = "0.1.0"
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
                provenance={
                    "engine": "literature",
                    "code_version": version,
                    "provider_requested": selected_provider,
                    "provider_available": bool(research_adapter and research_adapter.available),
                    "provider_source_added": provider_source_added,
                    "answer_author": answer_author,
                    "answer_provider": answer_provider,
                    "answer_model": answer_model,
                    "answer_note": answer_note,
                    "model": model or "",
                    "max_rounds": max_rounds,
                    "requested_source_paths": source_paths,
                    "requested_urls": urls,
                    "allow_seed_corpus": allow_seed_corpus,
                    "used_packaged_seed_corpus": used_packaged_seed_corpus,
                    "included_memory": include_memory,
                    "source_count": len(sources),
                    "evidence_count": len(evidence),
                    "claim_count": len(claims),
                },
            )
            artifacts = write_artifacts(run, out_path)
            memory.record_run(run)
            tracer.event("artifacts_written", **artifacts)
            refresh_run_manifest(run.artifacts["json"])
            return run
        finally:
            memory.close()


def dedupe_evidence(chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    by_id: dict[str, EvidenceChunk] = {}
    for chunk in _items(chunks):
        chunk_id = _text(getattr(chunk, "id", "")).strip()
        source_id = _text(getattr(chunk, "source_id", "")).strip()
        if not chunk_id or not source_id:
            continue
        if isinstance(chunk, EvidenceChunk):
            chunk.id = chunk_id
            chunk.source_id = source_id
            chunk.title = _text(getattr(chunk, "title", "")).strip()
            chunk.text = _text(getattr(chunk, "text", "")).strip()
            chunk.url = _text(getattr(chunk, "url", "")).strip()
            chunk.score = _number(getattr(chunk, "score", 0.0))
            chunk.highlights = _strings(getattr(chunk, "highlights", []))
            row = chunk
        else:
            row = EvidenceChunk(
                id=chunk_id,
                source_id=source_id,
                title=_text(getattr(chunk, "title", "")).strip(),
                text=_text(getattr(chunk, "text", "")).strip(),
                url=_text(getattr(chunk, "url", "")).strip(),
                score=_number(getattr(chunk, "score", 0.0)),
                highlights=_strings(getattr(chunk, "highlights", [])),
            )
        current = by_id.get(chunk_id)
        if current is None or row.score > current.score:
            by_id[chunk_id] = row
    return sorted(by_id.values(), key=lambda c: _number(getattr(c, "score", 0.0)), reverse=True)
