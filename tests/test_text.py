import unittest

from mechferret.text import (
    compact_text,
    cosine_overlap,
    domain,
    has_negation,
    sentence_split,
    stable_id,
    tokenize,
)


class TextTest(unittest.TestCase):
    def test_text_helpers_tolerate_non_text_inputs(self):
        self.assertEqual(tokenize(None), [])
        self.assertEqual(tokenize({"not": "text"}), [])
        self.assertEqual(sentence_split(["not text"]), [])
        self.assertEqual(domain(None), "local")
        self.assertEqual(compact_text({"not": "text"}), "")
        self.assertEqual(cosine_overlap(None, "real text"), 0.0)
        self.assertFalse(has_negation({"not": "semantic text"}))

    def test_bytes_are_treated_as_text(self):
        self.assertEqual(
            tokenize(b"Mechanistic circuits need evidence"),
            ["mechanistic", "circuits", "need", "evidence"],
        )
        self.assertEqual(domain(b"https://Example.com/paper"), "example.com")

    def test_compact_text_normalizes_bad_limits(self):
        self.assertEqual(compact_text("alpha beta", limit="bad"), "alpha beta")
        self.assertEqual(compact_text("alpha beta", limit=0), "alpha beta")
        self.assertEqual(compact_text("alpha beta", limit=3), "alp")

    def test_stable_id_normalizes_bad_inputs(self):
        self.assertTrue(stable_id("", {"a": 1}, length="bad").startswith("id_"))
        self.assertEqual(len(stable_id("x", "value", length=100).split("_", 1)[1]), 64)


if __name__ == "__main__":
    unittest.main()
