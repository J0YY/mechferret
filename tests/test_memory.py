import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mechferret.controller import MechFerret


class MemoryTest(unittest.TestCase):
    def test_memory_recall_adds_prior_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "# Tracing\nRaindrop-style tracing records planner retriever critic and synthesizer spans for inspection.",
                encoding="utf-8",
            )
            engine = MechFerret(root / "memory.sqlite")
            first = engine.run("Why trace autoresearch agents?", [str(source)], out_dir=root / "first")
            second = engine.run("What prior trace evidence exists?", [], out_dir=root / "second")
            self.assertGreaterEqual(len(first.claims), 1)
            self.assertTrue(any(source.kind == "memory" for source in second.sources))

    def test_memory_sanitizes_malformed_mechanisms_and_experiments(self):
        from mechferret.memory import ResearchMemory

        with tempfile.TemporaryDirectory() as tmp:
            mem = ResearchMemory(Path(tmp) / "memory.sqlite")
            try:
                count = mem.record_mechanisms(
                    None,
                    [
                        {"statement": "  clean mechanism  ", "effect_size": "bad", "reproducibility": math.inf},
                        {"statement": ""},
                        ["not a row"],
                    ],
                )
                self.assertEqual(count, 1)
                mechanisms = mem.recent_mechanisms(limit="bad")
                self.assertEqual(mechanisms[0]["statement"], "clean mechanism")
                self.assertEqual(mechanisms[0]["effect_size"], 0.0)
                self.assertEqual(mechanisms[0]["reproducibility"], 0.0)

                drift = mem.record_experiment(
                    None,
                    "task",
                    "probe",
                    {"path": Path("artifact")},
                    "hypothesis",
                    math.nan,
                    "bad",
                    True,
                    False,
                )
                self.assertEqual(drift, 0)
                grouped = mem.experiments_by_hypothesis(limit="bad")
                row = grouped["hypothesis"][0]
                self.assertEqual(row["effect_size"], 0.0)
                self.assertEqual(row["control"], 0.0)
                self.assertIn("artifact", row["target_json"])
            finally:
                mem.close()

    def test_record_experiments_skips_malformed_rows_and_tracks_drift(self):
        from mechferret.memory import ResearchMemory

        with tempfile.TemporaryDirectory() as tmp:
            mem = ResearchMemory(Path(tmp) / "memory.sqlite")
            try:
                first = mem.record_experiments(
                    "model",
                    "task",
                    [SimpleNamespace(statement="hypothesis", experiment_ids=["spec-a"]), {"statement": [], "experiment_ids": ["bad"]}],
                    [
                        {"status": "ran", "spec_id": "spec-a", "probe": "head_ablation", "target": {"layer": 1}, "effect_size": 1.0, "baseline": 0.0, "significant": True, "reproduced": True},
                        {"status": "ran", "spec_id": "missing-probe"},
                        ["not a result"],
                    ],
                )
                second = mem.record_experiments(
                    "model",
                    "task",
                    "not a hypothesis list",
                    [
                        SimpleNamespace(
                            status="ran",
                            spec_id="spec-a",
                            probe="head_ablation",
                            target={"layer": 1},
                            effect_size=-1.0,
                            baseline=0.0,
                            significant=False,
                            reproduced=False,
                        )
                    ],
                )

                self.assertEqual(first, {"recorded": 1, "drifted": 0})
                self.assertEqual(second, {"recorded": 1, "drifted": 1})
                grouped = mem.experiments_by_hypothesis()
                self.assertIn("screen", grouped)
                self.assertEqual(grouped["screen"][0]["drift_count"], 1)
                self.assertEqual(grouped["screen"][0]["observed_count"], 2)
            finally:
                mem.close()

    def test_record_run_and_sources_skip_malformed_rows(self):
        from mechferret.memory import ResearchMemory

        with tempfile.TemporaryDirectory() as tmp:
            mem = ResearchMemory(Path(tmp) / "memory.sqlite")
            try:
                mem.upsert_sources(
                    [
                        SimpleNamespace(id="s1", title="Source", text="Useful text", url=None, kind="local", metadata={"path": Path("source.md")}),
                        {"id": "s2", "title": "No text"},
                        "not a source",
                    ]
                )
                mem.record_run(
                    SimpleNamespace(
                        run_id="run1",
                        question="Question?",
                        answer="Answer.",
                        metrics={"score": math.nan},
                        artifacts={"path": Path("run.json")},
                        created_at="2026-01-01T00:00:00+00:00",
                        claims=[
                            SimpleNamespace(id="c1", text="A valid claim.", citations=[Path("citation")], source_ids=["s1"], confidence=math.inf, support_score="bad", stance="finding"),
                            {"id": "c2", "text": ""},
                            ["not a claim"],
                        ],
                    )
                )
                mem.record_run(SimpleNamespace(question="missing id"))

                source_count = mem.conn.execute("select count(*) from sources").fetchone()[0]
                run_count = mem.conn.execute("select count(*) from runs").fetchone()[0]
                claim = mem.conn.execute("select * from claims").fetchone()
                run = mem.conn.execute("select metrics_json, artifacts_json from runs where id='run1'").fetchone()

                self.assertEqual(source_count, 1)
                self.assertEqual(run_count, 1)
                self.assertEqual(claim["id"], "c1")
                self.assertEqual(claim["confidence"], 0.0)
                self.assertEqual(claim["support_score"], 0.0)
                self.assertIsNone(json.loads(run["metrics_json"])["score"])
                self.assertIn("run.json", json.loads(run["artifacts_json"])["path"])
            finally:
                mem.close()

    def test_memory_recall_and_recent_tolerate_corrupt_rows(self):
        from mechferret.memory import ResearchMemory
        from mechferret.ops import memory_recent

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "memory.sqlite"
            mem = ResearchMemory(db)
            try:
                mem.conn.execute(
                    "insert into runs (id, question, answer, metrics_json, artifacts_json, created_at) values (?,?,?,?,?,?)",
                    ("r1", "Why trace?", "", "{", "[", "2026-01-01T00:00:00+00:00"),
                )
                mem.conn.execute(
                    "insert into claims (id, run_id, text, citations_json, source_ids_json, confidence, support_score, stance, quality_flags_json, created_at) values (?,?,?,?,?,?,?,?,?,?)",
                    ("c1", "r1", "Tracing helps inspect agent steps.", "[]", "[]", "bad", math.inf, "finding", "[]", "2026-01-01T00:00:00+00:00"),
                )
                mem.conn.commit()

                self.assertEqual(mem.recall_sources("trace", limit=0), [])
                recalled = mem.recall_sources("Tracing", limit="bad")
                self.assertEqual(len(recalled), 1)
                self.assertIn("Confidence: 0.00", recalled[0].text)
            finally:
                mem.close()

            recent = memory_recent(db, limit="bad")
            self.assertEqual(recent[0]["metrics"], {})
            self.assertEqual(recent[0]["artifacts"], {})


if __name__ == "__main__":
    unittest.main()
