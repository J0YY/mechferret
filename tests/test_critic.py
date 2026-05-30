import unittest

from mechferret.agents import Critic, Planner
from mechferret.models import Claim, EvidenceChunk


class CriticTest(unittest.TestCase):
    def test_detects_negation_contradiction(self):
        claims = [
            Claim("a", "The agent uses memory to validate evidence before synthesis.", ["e1"], ["s1"], 0.8, 0.8),
            Claim("b", "The agent does not use memory to validate evidence before synthesis.", ["e2"], ["s2"], 0.8, 0.8),
        ]
        contradictions = Critic()._contradictions(claims)
        self.assertEqual(len(contradictions), 1)

    def test_generic_subject_claim_does_not_cover_all_plan_facets(self):
        plan = Planner().plan("What should BRCA1 research prioritize?")
        claims = [
            Claim("a", "BRCA1 research is important for cancer biology.", ["e1"], ["s1"], 0.8, 0.8),
        ]
        gaps, _, metrics = Critic().evaluate("What should BRCA1 research prioritize?", plan, claims, [])
        self.assertLess(metrics["plan_coverage"], 1.0)
        self.assertIn("not every plan facet has enough supporting evidence", gaps)

    def test_repeated_single_chunk_citations_trigger_concentration_gap(self):
        plan = Planner().plan("What should an autoresearch agent do?")
        claims = [
            Claim(f"c{i}", f"Evidence claim {i} about autoresearch evidence validation risk gaps.", ["e1"], ["s1"], 0.8, 0.8)
            for i in range(6)
        ]
        evidence = [EvidenceChunk("e1", "s1", "Only Source", "Many claims come from one chunk.", score=5.0)]
        gaps, _, metrics = Critic().evaluate("What should an autoresearch agent do?", plan, claims, evidence)
        self.assertLess(metrics["unique_citation_ratio"], 0.5)
        self.assertIn("citation concentration is high; find corroborating chunks", gaps)


if __name__ == "__main__":
    unittest.main()
