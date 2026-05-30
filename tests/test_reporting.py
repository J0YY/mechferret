import unittest

from mechferret.agents import Synthesizer
from mechferret.models import EvidenceChunk


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


if __name__ == "__main__":
    unittest.main()

