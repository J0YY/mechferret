import unittest

from mechferret.agents import ClaimExtractor
from mechferret.models import EvidenceChunk


class ExtractorTest(unittest.TestCase):
    def test_extracts_single_entity_domain_evidence(self):
        chunk = EvidenceChunk(
            "e1",
            "s1",
            "BRCA1 note",
            "BRCA1 repairs DNA double strand breaks through homologous recombination and helps maintain genome stability.",
            score=4.0,
        )
        claims = ClaimExtractor().extract("What does BRCA1 do?", [chunk])
        self.assertGreaterEqual(len(claims), 1)
        self.assertIn("homologous recombination", claims[0].text)


if __name__ == "__main__":
    unittest.main()

