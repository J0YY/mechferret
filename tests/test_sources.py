import unittest

from mechferret.models import Source
from mechferret.sources import dedupe_sources, example_corpus_path


class SourcesTest(unittest.TestCase):
    def test_packaged_example_corpus_exists(self):
        path = example_corpus_path()
        self.assertTrue(path.exists())
        self.assertGreaterEqual(len(list(path.glob("*.md"))), 4)
        self.assertIn("mechferret", str(path))

    def test_dedupe_uses_full_source_text(self):
        prefix = "shared boilerplate " * 140
        sources = [
            Source("a", "A", prefix + "unique conclusion alpha"),
            Source("b", "B", prefix + "unique conclusion beta"),
        ]
        self.assertEqual(len(dedupe_sources(sources)), 2)


if __name__ == "__main__":
    unittest.main()
