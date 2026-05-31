import unittest
from unittest.mock import patch

from mechferret.models import Source
from mechferret.sources import (
    dedupe_sources,
    example_corpus_path,
    fetch_url_source,
    infer_title,
    load_sources,
    normalize_text,
    strip_redundant_title,
)


class _Response:
    headers = {"content-type": "text/html"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *_args):
        return b"<html><title>Fetched</title><body><h1>Fetched</h1>Body text.</body></html>"


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

    def test_dedupe_tolerates_malformed_source_text(self):
        sources = [
            Source("a", "A", None),  # type: ignore[arg-type]
            Source("b", "B", {"not": "text"}),  # type: ignore[arg-type]
            Source("c", "C", b"byte text"),  # type: ignore[arg-type]
        ]
        self.assertEqual([source.id for source in dedupe_sources(sources)], ["a", "b", "c"])

    def test_dedupe_skips_fully_malformed_rows(self):
        sources = [
            Source("a", "A", "same text"),
            Source("b", "B", "same text"),
            object(),
        ]
        self.assertEqual([source.id for source in dedupe_sources(sources)], ["a"])

    def test_text_normalizers_tolerate_malformed_inputs(self):
        self.assertEqual(infer_title(None, None), "source")
        self.assertEqual(infer_title('{"headline": "JSON Title"}', "fallback"), "JSON Title")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(strip_redundant_title(None, "Title"), "")

    def test_load_sources_ignores_malformed_empty_entries(self):
        self.assertEqual(load_sources(paths=[], urls=[]), [])
        self.assertEqual(load_sources(paths=None, urls=[None, ""]), [])

    def test_fetch_url_source_sanitizes_url_and_timeout(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            fetch_url_source("")
        with patch("mechferret.sources.urlopen", return_value=_Response()) as opened:
            source = fetch_url_source(" https://example.com/page ", timeout="bad")
        self.assertEqual(source.title, "Fetched")
        self.assertEqual(source.url, "https://example.com/page")
        self.assertEqual(source.metadata["content_type"], "text/html")
        self.assertEqual(opened.call_args.kwargs["timeout"], 15)


if __name__ == "__main__":
    unittest.main()
