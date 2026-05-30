import unittest
import json
import re

from mechferret.agents import Synthesizer
from mechferret.models import EvidenceChunk, ResearchPlan, ResearchRun
from mechferret.report import html_report


class ReportingTest(unittest.TestCase):
    def test_citation_labels_are_unique_for_multiple_chunks_per_source(self):
        evidence = [
            EvidenceChunk("c1", "s1", "Long Source", "first"),
            EvidenceChunk("c2", "s1", "Long Source", "second"),
            EvidenceChunk("c3", "s2", "Short Source", "third"),
        ]
        labels = Synthesizer().citation_labels(evidence)
        self.assertEqual(labels["c1"], "S1.1")
        self.assertEqual(labels["c2"], "S1.2")
        self.assertEqual(labels["c3"], "S2")
        self.assertEqual(len(set(labels.values())), 3)

    def test_html_embeds_parseable_json_payload(self):
        run = ResearchRun(
            run_id="run_test",
            question="Can <script> break JSON?",
            created_at="2026-05-30T00:00:00+00:00",
            plan=ResearchPlan("Can <script> break JSON?", [], "test"),
            sources=[],
            evidence=[],
            claims=[],
            contradictions=[],
            gaps=[],
            answer="No </script> breakage.",
            metrics={},
        )
        rendered = html_report(run)
        payload = re.search(r'<script id="run-json" type="application/json">(.*?)</script>', rendered, re.S)
        self.assertIsNotNone(payload)
        parsed = json.loads(payload.group(1))
        self.assertEqual(parsed["run_id"], "run_test")


if __name__ == "__main__":
    unittest.main()
