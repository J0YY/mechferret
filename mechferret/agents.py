from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

from .models import Claim, Contradiction, EvidenceChunk, PlanStep, ResearchPlan
from .text import compact_text, cosine_overlap, domain, has_negation, sentence_split, stable_id, tokenize


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _strings(value: Any) -> list[str]:
    return [_text(item).strip() for item in _items(value) if _text(item).strip()]


def _limit(value: Any, default: int) -> int:
    if type(value) is bool:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _claim_id(value: Any) -> str:
    return _text(value).strip() or stable_id("claim", "missing")


def _claim(value: Any) -> Claim | None:
    text = _text(getattr(value, "text", "")).strip()
    if not text:
        return None
    if isinstance(value, Claim):
        value.citations = _strings(getattr(value, "citations", []))
        value.source_ids = _strings(getattr(value, "source_ids", []))
        value.quality_flags = _strings(getattr(value, "quality_flags", []))
        value.confidence = _number(getattr(value, "confidence", 0.0))
        value.support_score = _number(getattr(value, "support_score", 0.0))
        return value
    return Claim(
        id=_claim_id(getattr(value, "id", "")),
        text=text,
        citations=_strings(getattr(value, "citations", [])),
        source_ids=_strings(getattr(value, "source_ids", [])),
        confidence=_number(getattr(value, "confidence", 0.0)),
        support_score=_number(getattr(value, "support_score", 0.0)),
        quality_flags=_strings(getattr(value, "quality_flags", [])),
    )


class Planner:
    def plan(self, question: str) -> ResearchPlan:
        question = _text(question).strip()
        key_terms = [term for term in tokenize(question)[:10]]
        subject = " ".join(key_terms[:6]) or question
        steps = [
            PlanStep("step_1", f"What are the strongest mechanisms or approaches for {subject}?", "map_mechanisms"),
            PlanStep("step_2", f"What evidence supports or weakens {subject}?", "validate_evidence"),
            PlanStep("step_3", f"What implementation, scaling, or reliability risks matter for {subject}?", "risk_scan"),
            PlanStep("step_4", f"What gaps remain and what searches would reduce uncertainty for {subject}?", "gap_finding"),
        ]
        return ResearchPlan(
            question=question,
            steps=steps,
            strategy="Iterative fan-out: retrieve evidence for each facet, extract claims, critique coverage, then expand gaps.",
        )

    def expand_for_gaps(self, plan: ResearchPlan, gaps: list[str]) -> None:
        if not isinstance(plan, ResearchPlan):
            return
        existing = {step.question for step in plan.steps}
        next_index = len(plan.steps) + 1
        for gap in _strings(gaps)[:3]:
            question = f"Resolve evidence gap: {gap}"
            if question in existing:
                continue
            plan.steps.append(PlanStep(f"step_{next_index}", question, "critic_gap"))
            next_index += 1


class ClaimExtractor:
    ACTION_TERMS = {
        "agent",
        "agents",
        "autoresearch",
        "control",
        "critic",
        "retrieval",
        "retriever",
        "search",
        "evidence",
        "memory",
        "citation",
        "synthesis",
        "evaluation",
        "benchmark",
        "risk",
        "scale",
        "parallel",
        "planner",
        "trace",
        "traces",
        "tracing",
        "synthesizer",
        "workflow",
    }

    def extract(self, question: str, chunks: list[EvidenceChunk], limit: int = 18) -> list[Claim]:
        limit = _limit(limit, 18)
        query_terms = set(tokenize(question))
        candidates: dict[str, Claim] = {}
        for chunk in _items(chunks):
            chunk_id = _text(getattr(chunk, "id", "")).strip()
            source_id = _text(getattr(chunk, "source_id", "")).strip()
            if not chunk_id or not source_id:
                continue
            chunk_score = _number(getattr(chunk, "score", 0.0))
            chunk_url = _text(getattr(chunk, "url", ""))
            for sentence in sentence_split(getattr(chunk, "text", "")):
                terms = set(tokenize(sentence))
                query_overlap = len(terms & query_terms)
                action_overlap = len(terms & self.ACTION_TERMS)
                new_information = len(terms - query_terms)
                if query_overlap + action_overlap < 2 and not (
                    query_overlap >= 1 and chunk_score > 0 and new_information >= 3
                ):
                    continue
                source_diversity = 1.0
                support = min(1.0, (len(terms & query_terms) / max(len(query_terms), 1)) + 0.25)
                confidence = round(min(0.92, 0.42 + support * 0.38 + min(chunk_score, 8.0) / 60), 3)
                claim_id = stable_id("claim", sentence.lower())
                if claim_id in candidates:
                    current = candidates[claim_id]
                    if chunk_id not in current.citations:
                        current.citations.append(chunk_id)
                    if source_id not in current.source_ids:
                        current.source_ids.append(source_id)
                    current.support_score = min(1.0, current.support_score + 0.12)
                    current.confidence = min(0.95, current.confidence + 0.04)
                    continue
                flags = []
                if len(sentence) < 70:
                    flags.append("thin_sentence")
                if chunk_url.startswith("memory://"):
                    flags.append("prior_memory")
                candidates[claim_id] = Claim(
                    id=claim_id,
                    text=compact_text(sentence, 360),
                    citations=[chunk_id],
                    source_ids=[source_id],
                    confidence=confidence,
                    support_score=round(min(1.0, support * source_diversity), 3),
                    quality_flags=flags,
                )
        claims = merge_similar_claims(list(candidates.values()))
        claims = sorted(claims, key=lambda c: (c.confidence, len(c.citations)), reverse=True)
        return claims[:limit]


class Critic:
    def evaluate(
        self,
        question: str,
        plan: ResearchPlan,
        claims: list[Claim],
        evidence: list[EvidenceChunk],
    ) -> tuple[list[str], list[Contradiction], dict[str, float]]:
        gaps: list[str] = []
        plan = plan if isinstance(plan, ResearchPlan) else ResearchPlan(_text(question), [], "")
        claims = [claim for claim in (_claim(item) for item in _items(claims)) if claim is not None]
        evidence = [chunk for chunk in _items(evidence) if _text(getattr(chunk, "id", "")).strip()]
        domains = {domain(getattr(chunk, "url", "")) for chunk in evidence}
        source_ids = {_text(getattr(chunk, "source_id", "")).strip() for chunk in evidence if _text(getattr(chunk, "source_id", "")).strip()}
        source_diversity = len(source_ids)
        cited_chunks = {citation for claim in claims for citation in _strings(getattr(claim, "citations", []))}
        total_citations = sum(len(_strings(getattr(claim, "citations", []))) for claim in claims)
        plan_coverage = self._plan_coverage(plan, claims)
        if len(claims) < 5:
            gaps.append("too few distinct extracted claims; broaden retrieval or add live search")
        if source_diversity < 3 and len(evidence) >= 3:
            gaps.append("source diversity is low; seek independent sources or datasets")
        if plan_coverage < 0.65:
            gaps.append("not every plan facet has enough supporting evidence")
        unique_citation_ratio = len(cited_chunks) / max(total_citations, 1)
        if total_citations / max(len(claims), 1) < 0.85 or unique_citation_ratio < 0.5:
            gaps.append("citation concentration is high; find corroborating chunks")

        contradictions = self._contradictions(claims)
        if contradictions:
            gaps.append("contradictory claims require adjudication before high-confidence synthesis")

        metrics = {
            "claims": float(len(claims)),
            "evidence_chunks": float(len(evidence)),
            "source_diversity": float(source_diversity),
            "domain_diversity": float(len(domains)),
            "plan_coverage": round(plan_coverage, 3),
            "citation_density": round(total_citations / max(len(claims), 1), 3),
            "unique_citation_ratio": round(unique_citation_ratio, 3),
            "contradiction_pressure": round(sum(c.severity for c in contradictions), 3),
            "mean_confidence": round(sum(_number(getattr(c, "confidence", 0.0)) for c in claims) / max(len(claims), 1), 3),
        }
        metrics["readiness_score"] = self._readiness(metrics, len(gaps))
        return gaps, contradictions, metrics

    def _plan_coverage(self, plan: ResearchPlan, claims: list[Claim]) -> float:
        steps = [step for step in _items(getattr(plan, "steps", [])) if _text(getattr(step, "question", "")).strip()]
        if not steps:
            return 0.0
        covered = 0
        for step in steps:
            step_question = _text(getattr(step, "question", ""))
            step_terms = set(tokenize(step_question))
            facet_terms = self._step_facet_terms(step)
            for claim in claims:
                claim_text = _text(getattr(claim, "text", ""))
                claim_terms = set(tokenize(claim_text))
                shared_terms = step_terms & claim_terms
                facet_overlap = bool(facet_terms & claim_terms)
                if not facet_terms and cosine_overlap(step_question, claim_text) > 0.18 and len(shared_terms) >= 2:
                    covered += 1
                    break
                if facet_overlap and len(shared_terms) >= 1:
                    covered += 1
                    break
        return covered / len(steps)

    def _step_facet_terms(self, step: PlanStep) -> set[str]:
        intent = _text(getattr(step, "intent", ""))
        question = _text(getattr(step, "question", ""))
        by_intent = {
            "map_mechanisms": {"mechanism", "mechanisms", "approach", "approaches", "architecture", "architectures"},
            "validate_evidence": {"evidence", "supports", "support", "weakens", "validate", "validation"},
            "risk_scan": {"risk", "risks", "reliability", "scaling", "implementation", "failure", "fail"},
            "gap_finding": {"gap", "gaps", "uncertainty", "uncertain", "searches", "missing"},
            "critic_gap": set(tokenize(question)) - set(tokenize(self.plan_subject_noise(question))),
        }
        return by_intent.get(intent, set())

    def plan_subject_noise(self, question: str) -> str:
        return question.replace("Resolve evidence gap:", "")

    def _contradictions(self, claims: list[Claim]) -> list[Contradiction]:
        contradictions: list[Contradiction] = []
        clean_claims = [claim for claim in (_claim(item) for item in _items(claims)) if claim is not None]
        for left, right in itertools.combinations(clean_claims, 2):
            left_text = _text(getattr(left, "text", ""))
            right_text = _text(getattr(right, "text", ""))
            if has_negation(left_text) == has_negation(right_text):
                continue
            overlap = cosine_overlap(left_text, right_text)
            if overlap < 0.34:
                continue
            contradictions.append(
                Contradiction(
                    id=stable_id("contra", f"{_claim_id(getattr(left, 'id', ''))}:{_claim_id(getattr(right, 'id', ''))}"),
                    claim_a=_claim_id(getattr(left, "id", "")),
                    claim_b=_claim_id(getattr(right, "id", "")),
                    reason="semantic overlap with opposing negation polarity",
                    severity=round(min(1.0, overlap), 3),
                )
            )
        return contradictions[:8]

    def _readiness(self, metrics: dict[str, float], gap_count: int) -> float:
        score = 0.25
        score += min(metrics["claims"] / 12, 1.0) * 0.18
        score += min(metrics["source_diversity"] / 4, 1.0) * 0.18
        score += min(metrics["citation_density"] / 1.2, 1.0) * 0.16
        score += metrics["plan_coverage"] * 0.18
        score -= min(metrics["contradiction_pressure"], 1.0) * 0.14
        score -= min(gap_count / 5, 1.0) * 0.11
        return round(max(0.0, min(0.99, score)), 3)


class Synthesizer:
    def synthesize(self, question: str, claims: list[Claim], evidence: list[EvidenceChunk], gaps: list[str]) -> str:
        citation_labels = self.citation_labels(evidence)
        clean_claims = [claim for claim in (_claim(item) for item in _items(claims)) if claim is not None]
        top_claims = sorted(
            clean_claims,
            key=lambda c: (_number(getattr(c, "confidence", 0.0)), _number(getattr(c, "support_score", 0.0))),
            reverse=True,
        )[:7]
        lines = [f"Answer to: {question}", ""]
        if not top_claims:
            lines.append("No well-supported claims were extracted from the available corpus.")
        else:
            lines.append("Best-supported synthesis:")
            for claim in top_claims:
                citations = _strings(getattr(claim, "citations", []))
                labels = ", ".join(citation_labels.get(cid, cid) for cid in citations[:3])
                lines.append(f"- {_text(getattr(claim, 'text', ''))} [{labels}] confidence={_number(getattr(claim, 'confidence', 0.0)):.2f}")
        clean_gaps = _strings(gaps)
        if clean_gaps:
            lines.extend(["", "Unresolved gaps:"])
            for gap in clean_gaps[:6]:
                lines.append(f"- {gap}")
        return "\n".join(lines)

    def citation_labels(self, evidence: list[EvidenceChunk]) -> dict[str, str]:
        grouped: dict[str, list[EvidenceChunk]] = defaultdict(list)
        clean_evidence = [
            chunk for chunk in _items(evidence)
            if _text(getattr(chunk, "id", "")).strip() and _text(getattr(chunk, "source_id", "")).strip()
        ]
        for chunk in clean_evidence:
            grouped[_text(getattr(chunk, "source_id", "")).strip()].append(chunk)
        labels: dict[str, str] = {}
        source_index: dict[str, int] = {}
        chunk_index: dict[str, int] = defaultdict(int)
        for chunk in clean_evidence:
            chunk_id = _text(getattr(chunk, "id", "")).strip()
            source_id = _text(getattr(chunk, "source_id", "")).strip()
            source_index.setdefault(source_id, len(source_index) + 1)
            chunk_index[source_id] += 1
            source_label = f"S{source_index[source_id]}"
            if len(grouped[source_id]) == 1:
                labels[chunk_id] = source_label
            else:
                labels[chunk_id] = f"{source_label}.{chunk_index[source_id]}"
        return labels


def merge_similar_claims(claims: list[Claim]) -> list[Claim]:
    merged: list[Claim] = []
    for raw_claim in _items(claims):
        claim = _claim(raw_claim)
        if claim is None:
            continue
        target = None
        for existing in merged:
            if has_negation(existing.text) != has_negation(getattr(claim, "text", "")):
                continue
            if cosine_overlap(existing.text, getattr(claim, "text", "")) >= 0.76:
                target = existing
                break
        if target is None:
            merged.append(claim)
            continue
        for citation in _strings(getattr(claim, "citations", [])):
            if citation not in target.citations:
                target.citations.append(citation)
        for source_id in _strings(getattr(claim, "source_ids", [])):
            if source_id not in target.source_ids:
                target.source_ids.append(source_id)
        for flag in _strings(getattr(claim, "quality_flags", [])):
            if flag not in target.quality_flags:
                target.quality_flags.append(flag)
        target.support_score = round(min(1.0, target.support_score + 0.15), 3)
        target.confidence = round(min(0.96, target.confidence + 0.05), 3)
    return merged
