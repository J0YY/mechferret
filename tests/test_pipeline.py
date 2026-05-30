import json
import tempfile
import unittest
from pathlib import Path

from mechferret.controller import MechFerret


class PipelineTest(unittest.TestCase):
    def test_run_creates_artifacts_and_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "# Agent Research\n"
                "Autoresearch agents need planning, retrieval, evidence citations, and critic loops. "
                "A reliable implementation tracks source diversity and contradiction pressure. "
                "Replayable traces make agent failures easier to debug.",
                encoding="utf-8",
            )
            engine = MechFerret(root / "memory.sqlite")
            run = engine.run(
                "How should an autoresearch agent be made reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                max_rounds=2,
                include_memory=False,
            )
            self.assertGreaterEqual(len(run.claims), 1)
            self.assertTrue((root / "run" / "report.html").exists())
            payload = json.loads((root / "run" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], run.run_id)


if __name__ == "__main__":
    unittest.main()

