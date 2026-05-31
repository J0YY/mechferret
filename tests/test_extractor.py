import unittest
from types import SimpleNamespace

from mechferret.agents import ClaimExtractor, merge_similar_claims
from mechferret.models import Claim, EvidenceChunk


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

    def test_extract_tolerates_malformed_chunks_and_limits(self):
        chunks = [
            object(),
            EvidenceChunk("e1", "s1", "Bad", ["not text"], score=4.0),  # type: ignore[arg-type]
            SimpleNamespace(
                id="e2",
                source_id="s2",
                text=(
                    "Autoresearch agents need retrieval evidence, citation checks, memory traces, "
                    "and critic loops to improve reliability."
                ),
                score="5",
                url=b"memory://prior",
            ),
        ]
        claims = ClaimExtractor().extract(b"How should autoresearch improve reliability?", chunks, limit="bad")
        self.assertEqual(len(claims), 1)
        self.assertIn("prior_memory", claims[0].quality_flags)

    def test_merge_similar_claims_tolerates_malformed_rows(self):
        claims = [
            object(),
            SimpleNamespace(
                id="first",
                text="Agents use retrieval evidence before synthesis.",
                citations=["e0"],
                source_ids=["s0"],
                quality_flags=["first"],
            ),
            Claim("a", "Agents use retrieval evidence before synthesis.", ["e1"], ["s1"], 0.7, 0.7),
            SimpleNamespace(
                id="b",
                text="Agents use retrieval evidence before synthesis.",
                citations=["e2"],
                source_ids=["s2"],
                quality_flags=["merged"],
            ),
        ]
        merged = merge_similar_claims(claims)
        self.assertEqual(len(merged), 1)
        self.assertIsInstance(merged[0], Claim)
        self.assertIn("e2", merged[0].citations)


if __name__ == "__main__":
    unittest.main()
