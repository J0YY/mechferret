import unittest

from mechferret.models import Source
from mechferret.retrieval import BM25Index


class RetrievalTest(unittest.TestCase):
    def test_bm25_prefers_relevant_document(self):
        sources = [
            Source("a", "Biology", "Computational biology agents extract mechanistic claims from papers."),
            Source("b", "Cooking", "A recipe uses flour, water, salt, and heat."),
        ]
        index = BM25Index.from_sources(sources)
        results = index.search("biology mechanistic claims", limit=1)
        self.assertEqual(results[0].source_id, "a")


if __name__ == "__main__":
    unittest.main()

