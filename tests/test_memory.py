import tempfile
import unittest
from pathlib import Path

from mechferret.controller import MechFerret


class MemoryTest(unittest.TestCase):
    def test_memory_recall_adds_prior_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "# Tracing\nRaindrop-style tracing records planner retriever critic and synthesizer spans for replay.",
                encoding="utf-8",
            )
            engine = MechFerret(root / "memory.sqlite")
            first = engine.run("Why trace autoresearch agents?", [str(source)], out_dir=root / "first")
            second = engine.run("What prior trace evidence exists?", [], out_dir=root / "second")
            self.assertGreaterEqual(len(first.claims), 1)
            self.assertTrue(any(source.kind == "memory" for source in second.sources))


if __name__ == "__main__":
    unittest.main()

