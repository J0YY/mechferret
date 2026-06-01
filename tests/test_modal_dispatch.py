import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mechferret.cli import main
from mechferret.modal_app import dispatch_discovery, modal_status


class ModalDispatchTest(unittest.TestCase):
    def test_modal_status_keys(self):
        status = modal_status()
        for key in ("installed", "authenticated", "gpu", "app", "torch_local"):
            self.assertIn(key, status)

    def test_modal_cli_json_status_setup_and_deploy(self):
        for args, action in (
            (["modal", "status", "--json"], "status"),
            (["modal", "setup", "--json"], "setup"),
            (["modal", "deploy", "--json"], "deploy"),
        ):
            out = StringIO()
            with redirect_stdout(out):
                main(args)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["action"], action)
            self.assertIn("status", payload)
            self.assertIn("installed", payload["status"])

    def test_dispatch_falls_back_to_local(self):
        # Modal is not installed in CI, so dispatch must run locally and still produce a run.
        with tempfile.TemporaryDirectory() as tmp:
            result = dispatch_discovery(skill="ioi-circuit", model="gpt2", out_dir=Path(tmp) / "run")
            self.assertIn(result["backend"], {"local", "modal"})
            self.assertGreaterEqual(len(result["run"]["discoveries"]), 1)
            self.assertTrue((Path(tmp) / "run").exists())


if __name__ == "__main__":
    unittest.main()
