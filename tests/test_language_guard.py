import unittest
from pathlib import Path


class LanguageGuardTest(unittest.TestCase):
    def test_no_product_text_mentions_the_fixed_output_mode(self):
        token = "determ" + "inistic"
        roots = [Path("mechferret"), Path("tests"), Path("docs")]
        files = [Path("README.md")]
        for root in roots:
            files.extend(path for path in root.rglob("*") if path.is_file())
        text_suffixes = {
            "",
            ".cfg",
            ".css",
            ".html",
            ".ini",
            ".json",
            ".jsonl",
            ".md",
            ".py",
            ".sh",
            ".svg",
            ".tex",
            ".toml",
            ".txt",
            ".yaml",
            ".yml",
        }
        hits = []
        for path in files:
            if path.suffix.lower() not in text_suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if token in text.lower():
                hits.append(path.as_posix())
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
