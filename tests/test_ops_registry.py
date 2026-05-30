import tempfile
import unittest
from pathlib import Path

from mechferret.costs import estimate_run_cost
from mechferret.ops import doctor, memory_recent, memory_summary, summarize_run_artifact
from mechferret.registry import all_items, items_by_kind
from mechferret.controller import MechFerret


class OpsRegistryTest(unittest.TestCase):
    def test_registry_has_core_items(self):
        names = {item.name for item in all_items()}
        self.assertIn("goal_loop", names)
        self.assertIn("provider_research", names)
        self.assertGreaterEqual(len(items_by_kind("tool")), 5)

    def test_doctor_returns_checks(self):
        result = doctor()
        self.assertIn("checks", result)
        self.assertTrue(any(check["name"] == "example_corpus" for check in result["checks"]))

    def test_memory_resume_and_cost_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "# Evidence\nAutoresearch systems need claims, citations, memory, retrieval, and critic loops.",
                encoding="utf-8",
            )
            run = MechFerret(root / "memory.sqlite").run(
                "What does autoresearch need?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )
            summary = memory_summary(root / "memory.sqlite")
            self.assertEqual(summary["runs"], 1)
            self.assertGreaterEqual(len(memory_recent(root / "memory.sqlite", 1)), 1)
            resume = summarize_run_artifact(root / "run" / "run.json")
            self.assertEqual(resume["run_id"], run.run_id)
            cost = estimate_run_cost(root / "run" / "run.json")
            self.assertGreater(cost["estimated_tokens_processed"], 0)


if __name__ == "__main__":
    unittest.main()

