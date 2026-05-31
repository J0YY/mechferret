import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mechferret import demo


SAMPLE = {
    "title": "t", "model": "gpt2", "task": "ioi", "goal": "g",
    "beats": [
        {"type": "note", "text": "hi"},
        {"type": "experiment", "hypothesis": "H1", "probe": "p1", "effect": 0.5, "control": 0.0, "verdict": "good"},
        {"type": "experiment", "hypothesis": "H1", "probe": "p2", "effect": 0.1, "control": 0.0, "verdict": "weak"},
        {"type": "mechanism", "statement": "m1", "effect": 0.5, "reproducibility": 1.0, "novelty": 0.8},
    ],
}


class DemoTest(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)
        Path(".mechferret").mkdir()
        Path(".mechferret/demo.json").write_text(json.dumps(SAMPLE), encoding="utf-8")

    def tearDown(self):
        os.chdir(self._cwd)

    def test_has_and_load(self):
        self.assertTrue(demo.has_demo())
        self.assertEqual(demo.load_demo()["title"], "t")

    def test_seed_populates_memory(self):
        demo.seed_only()
        from mechferret.memory import ResearchMemory

        mem = ResearchMemory(demo.DB_PATH)
        try:
            grouped = mem.experiments_by_hypothesis()
            self.assertIn("H1", grouped)
            self.assertEqual(len(grouped["H1"]), 2)
            self.assertEqual(len(mem.recent_mechanisms()), 1)
        finally:
            mem.close()

    def test_seed_is_idempotent(self):
        demo.seed_only()
        demo.seed_only()  # re-seeding must not duplicate rows
        from mechferret.memory import ResearchMemory

        mem = ResearchMemory(demo.DB_PATH)
        try:
            grouped = mem.experiments_by_hypothesis()
            self.assertEqual(sum(len(v) for v in grouped.values()), 2)
            # same data twice => no drift
            self.assertEqual(sum(e["drift_count"] for v in grouped.values() for e in v), 0)
        finally:
            mem.close()

    def test_render_play_emits_trace(self):
        # render path (with a tool beat) must not crash and should write a trace
        d = dict(SAMPLE)
        d["beats"] = [
            {"type": "user", "text": "go"},
            {"type": "tool", "name": "retrieval.arxiv", "args": {"query": "x"}, "result": "ok"},
            {"type": "pivot", "text": "p"},
            {"type": "assistant", "text": "done"},
        ] + SAMPLE["beats"]
        out = StringIO()
        with redirect_stdout(out):
            demo.play(d, render=True, record=True, reset=True, speed=100.0)
        self.assertIn("demo complete", out.getvalue())
        trace = Path(".mechferret/trace.jsonl")
        self.assertTrue(trace.exists())
        self.assertGreater(len(trace.read_text().splitlines()), 0)

    def test_missing_demo_raises(self):
        Path(".mechferret/demo.json").unlink()
        self.assertFalse(demo.has_demo())
        with self.assertRaises(FileNotFoundError):
            demo.load_demo()


if __name__ == "__main__":
    unittest.main()
