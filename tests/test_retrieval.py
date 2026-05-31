import unittest
from types import SimpleNamespace

from mechferret.models import Source
from mechferret.retrieval import BM25Index, chunk_source


class RetrievalTest(unittest.TestCase):
    def test_bm25_prefers_relevant_document(self):
        sources = [
            Source("a", "Biology", "Computational biology agents extract mechanistic claims from papers."),
            Source("b", "Cooking", "A recipe uses flour, water, salt, and heat."),
        ]
        index = BM25Index.from_sources(sources)
        results = index.search("biology mechanistic claims", limit=1)
        self.assertEqual(results[0].source_id, "a")

    def test_retrieval_tolerates_malformed_sources_and_limits(self):
        sources = [
            Source("good", "Good", "Mechanistic retrieval needs robust source chunking and search."),
            {"id": "dict", "title": "Dict", "text": "Dictionary sources can also mention retrieval robustness."},
            {"id": "bad-text", "title": "Bad", "text": []},
            SimpleNamespace(id="", title="No id", text="Enough words but missing source identity"),
            "not a source",
        ]

        index = BM25Index.from_sources(sources)

        self.assertEqual(index.search("retrieval", limit=0), [])
        self.assertEqual(index.search("retrieval", limit=-1), [])
        self.assertGreaterEqual(len(index.search("retrieval", limit="bad")), 1)
        self.assertEqual({chunk.source_id for chunk in index.search("retrieval", limit="bad")}, {"good", "dict"})

        chunks = chunk_source(
            Source("tiny", "Tiny", "retrieval robustness requires careful source chunking for evidence search ranking"),
            max_tokens=6,
            overlap=99,
        )
        self.assertTrue(chunks)


if __name__ == "__main__":
    unittest.main()
