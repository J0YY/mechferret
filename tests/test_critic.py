import unittest
from types import SimpleNamespace

from mechferret.agents import Critic, Planner, Synthesizer
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

    def test_critic_and_synthesizer_tolerate_malformed_rows(self):
        claims = [
            object(),
            SimpleNamespace(id="c1", text="Agents use retrieval evidence before synthesis.", citations="e1", confidence="bad"),
            Claim("c2", "Agents do not use retrieval evidence before synthesis.", ["e2"], ["s2"], "0.8", 0.7),  # type: ignore[arg-type]
        ]
        evidence = [
            object(),
            SimpleNamespace(id="e1", source_id="s1", url=None),
            EvidenceChunk("e2", "s2", "Source", "Evidence", score="bad"),  # type: ignore[arg-type]
        ]

        gaps, contradictions, metrics = Critic().evaluate("Do agents use retrieval evidence?", object(), claims, evidence)
        self.assertGreaterEqual(metrics["claims"], 2)
        self.assertGreaterEqual(metrics["source_diversity"], 2)
        self.assertIsInstance(contradictions, list)

        text = Synthesizer().synthesize(
            b"Do agents use retrieval evidence?",
            claims,
            evidence,
            ["needs more sources", None],
        )
        self.assertIn("Best-supported synthesis", text)
        self.assertIn("needs more sources", text)

    def test_citation_labels_skip_malformed_evidence(self):
        labels = Synthesizer().citation_labels([object(), SimpleNamespace(id="e1", source_id="s1")])
        self.assertEqual(labels, {"e1": "S1"})


if __name__ == "__main__":
    unittest.main()
