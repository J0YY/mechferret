import json
import urllib.parse
import unittest
from unittest.mock import patch

from mechferret.knowledge import (
    neuronpedia_feature,
    neuronpedia_search_explanations,
    search_arxiv,
    web_fetch,
    web_search,
)


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *_args):
        return self.body


class KnowledgeTest(unittest.TestCase):
    def _query_param(self, url: str, name: str) -> str:
        values = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get(name, [])
        return values[0] if values else ""

    def test_web_search_sanitizes_limits_and_result_rows(self):
        html = b"""
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A &amp; B</a>
        <a class="result__snippet">Snippet with <b>SAE</b> details &amp; code.</a>
        <a class="result__a" href="">missing</a>
        """

        with patch("urllib.request.urlopen", return_value=_Response(html)) as opened:
            results = web_search(b" sparse autoencoders ", max_results="bad", timeout="bad")

        self.assertEqual(
            results,
            [
                {
                    "title": "A & B",
                    "url": "https://example.com/a",
                    "source_domain": "example.com",
                    "snippet": "Snippet with SAE details & code.",
                }
            ],
        )
        self.assertEqual(opened.call_args.kwargs["timeout"], 20)

    def test_web_fetch_handles_empty_url_and_bad_limits(self):
        self.assertEqual(web_fetch("", max_chars=[]), "")
        body = b"<html><script>x()</script><body>Hello&nbsp;world</body></html>"
        with patch("urllib.request.urlopen", return_value=_Response(body)):
            self.assertEqual(web_fetch("https://example.com", max_chars=5, timeout=[]), "Hello")

    def test_arxiv_search_tolerates_bad_xml_and_counts(self):
        with patch("urllib.request.urlopen", return_value=_Response(b"<not xml")):
            self.assertEqual(search_arxiv("sae"), (0, []))

        feed = b"""<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
          <opensearch:totalResults>many</opensearch:totalResults>
          <entry>
            <id>https://arxiv.org/abs/1234.5678</id>
            <title> Test Paper </title>
            <summary> Abstract text. </summary>
            <author><name>Ada</name></author>
            <link rel="alternate" href="https://arxiv.org/abs/1234.5678"/>
            <link title="pdf" href="https://arxiv.org/pdf/1234.5678"/>
          </entry>
        </feed>
        """
        with patch("urllib.request.urlopen", return_value=_Response(feed)) as opened:
            total, rows = search_arxiv("sae", max_results="bad", sort_by="invalid", timeout="bad")
        self.assertEqual(total, 0)
        self.assertEqual(rows[0]["title"], "Test Paper")
        self.assertEqual(rows[0]["authors"], ["Ada"])
        self.assertEqual(self._query_param(opened.call_args.args[0].full_url, "max_results"), "50")

        with patch("urllib.request.urlopen", return_value=_Response(feed)) as opened:
            search_arxiv("sae", max_results=2 + 3)
        self.assertEqual(self._query_param(opened.call_args.args[0].full_url, "max_results"), "50")

    def test_neuronpedia_helpers_tolerate_bad_json_and_inputs(self):
        self.assertEqual(neuronpedia_search_explanations("", "query"), {})
        self.assertEqual(neuronpedia_feature("", "source", 1), {})

        with patch("urllib.request.urlopen", return_value=_Response(b"[]")):
            self.assertEqual(neuronpedia_search_explanations("gpt2-small", "feature"), {})

        with patch("urllib.request.urlopen", return_value=_Response(json.dumps({"ok": True}).encode())) as opened:
            result = neuronpedia_feature("gpt2/small", "6 res", "bad")
        self.assertEqual(result, {"ok": True})
        self.assertIn("/feature/gpt2%2Fsmall/6%20res/0", opened.call_args.args[0].full_url)


if __name__ == "__main__":
    unittest.main()
