from __future__ import annotations

import itertools
from collections import defaultdict

from .models import Claim, Contradiction, EvidenceChunk, PlanStep, ResearchPlan
from .text import compact_text, cosine_overlap, domain, has_negation, sentence_split, stable_id, tokenize


class Planner:
    def plan(self, question: str) -> ResearchPlan:
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
        existing = {step.question for step in plan.steps}
        next_index = len(plan.steps) + 1
        for gap in gaps[:3]:
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
        query_terms = set(tokenize(question))
        candidates: dict[str, Claim] = {}
        for chunk in chunks:
            for sentence in sentence_split(chunk.text):
                terms = set(tokenize(sentence))
                if len(terms & (query_terms | self.ACTION_TERMS)) < 2:
                    continue
                source_diversity = 1.0
                support = min(1.0, (len(terms & query_terms) / max(len(query_terms), 1)) + 0.25)
                confidence = round(min(0.92, 0.42 + support * 0.38 + min(chunk.score, 8.0) / 60), 3)
                claim_id = stable_id("claim", sentence.lower())
                if claim_id in candidates:
                    current = candidates[claim_id]
                    if chunk.id not in current.citations:
                        current.citations.append(chunk.id)
                    if chunk.source_id not in current.source_ids:
                        current.source_ids.append(chunk.source_id)
                    current.support_score = min(1.0, current.support_score + 0.12)
                    current.confidence = min(0.95, current.confidence + 0.04)
                    continue
                flags = []
                if len(sentence) < 70:
                    flags.append("thin_sentence")
                if chunk.url.startswith("memory://"):
                    flags.append("prior_memory")
                candidates[claim_id] = Claim(
                    id=claim_id,
                    text=compact_text(sentence, 360),
                    citations=[chunk.id],
                    source_ids=[chunk.source_id],
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
        domains = {domain(chunk.url) for chunk in evidence}
        source_ids = {chunk.source_id for chunk in evidence}
        source_diversity = len(source_ids)
        cited_chunks = {citation for claim in claims for citation in claim.citations}
        total_citations = sum(len(claim.citations) for claim in claims)
        plan_coverage = self._plan_coverage(plan, claims)
        if len(claims) < 5:
            gaps.append("too few distinct extracted claims; broaden retrieval or add live search")
        if source_diversity < 3 and len(evidence) >= 3:
            gaps.append("source diversity is low; seek independent sources or datasets")
        if plan_coverage < 0.65:
            gaps.append("not every plan facet has enough supporting evidence")
        if total_citations / max(len(claims), 1) < 0.85:
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
            "unique_citation_ratio": round(len(cited_chunks) / max(total_citations, 1), 3),
            "contradiction_pressure": round(sum(c.severity for c in contradictions), 3),
            "mean_confidence": round(sum(c.confidence for c in claims) / max(len(claims), 1), 3),
        }
        metrics["readiness_score"] = self._readiness(metrics, len(gaps))
        return gaps, contradictions, metrics

    def _plan_coverage(self, plan: ResearchPlan, claims: list[Claim]) -> float:
        if not plan.steps:
            return 0.0
        covered = 0
        for step in plan.steps:
            step_terms = set(tokenize(step.question))
            if any(cosine_overlap(step.question, claim.text) > 0.12 or step_terms & set(tokenize(claim.text)) for claim in claims):
                covered += 1
        return covered / len(plan.steps)

    def _contradictions(self, claims: list[Claim]) -> list[Contradiction]:
        contradictions: list[Contradiction] = []
        for left, right in itertools.combinations(claims, 2):
            if has_negation(left.text) == has_negation(right.text):
                continue
            overlap = cosine_overlap(left.text, right.text)
            if overlap < 0.34:
                continue
            contradictions.append(
                Contradiction(
                    id=stable_id("contra", f"{left.id}:{right.id}"),
                    claim_a=left.id,
                    claim_b=right.id,
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
        top_claims = sorted(claims, key=lambda c: (c.confidence, c.support_score), reverse=True)[:7]
        lines = [f"Answer to: {question}", ""]
        if not top_claims:
            lines.append("No well-supported claims were extracted from the available corpus.")
        else:
            lines.append("Best-supported synthesis:")
            for claim in top_claims:
                labels = ", ".join(citation_labels.get(cid, cid) for cid in claim.citations[:3])
                lines.append(f"- {claim.text} [{labels}] confidence={claim.confidence:.2f}")
        if gaps:
            lines.extend(["", "Unresolved gaps:"])
            for gap in gaps[:6]:
                lines.append(f"- {gap}")
        return "\n".join(lines)

    def citation_labels(self, evidence: list[EvidenceChunk]) -> dict[str, str]:
        grouped: dict[str, list[EvidenceChunk]] = defaultdict(list)
        for chunk in evidence:
            grouped[chunk.source_id].append(chunk)
        labels: dict[str, str] = {}
        source_index: dict[str, int] = {}
        for chunk in evidence:
            source_index.setdefault(chunk.source_id, len(source_index) + 1)
            labels[chunk.id] = f"S{source_index[chunk.source_id]}"
        return labels


def merge_similar_claims(claims: list[Claim]) -> list[Claim]:
    merged: list[Claim] = []
    for claim in claims:
        target = None
        for existing in merged:
            if has_negation(existing.text) != has_negation(claim.text):
                continue
            if cosine_overlap(existing.text, claim.text) >= 0.76:
                target = existing
                break
        if target is None:
            merged.append(claim)
            continue
        for citation in claim.citations:
            if citation not in target.citations:
                target.citations.append(citation)
        for source_id in claim.source_ids:
            if source_id not in target.source_ids:
                target.source_ids.append(source_id)
        for flag in claim.quality_flags:
            if flag not in target.quality_flags:
                target.quality_flags.append(flag)
        target.support_score = round(min(1.0, target.support_score + 0.15), 3)
        target.confidence = round(min(0.96, target.confidence + 0.05), 3)
    return merged
