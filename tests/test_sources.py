import unittest

from mechferret.sources import example_corpus_path


class SourcesTest(unittest.TestCase):
    def test_packaged_example_corpus_exists(self):
        path = example_corpus_path()
        self.assertTrue(path.exists())
        self.assertGreaterEqual(len(list(path.glob("*.md"))), 4)
        self.assertIn("mechferret", str(path))


if __name__ == "__main__":
    unittest.main()

