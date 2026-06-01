import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from mechferret import modal_app
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

    def test_dispatch_fails_closed_without_modal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(modal_app, "modal", None):
                result = dispatch_discovery(skill="ioi-circuit", model="gpt2", out_dir=Path(tmp) / "run")
            self.assertFalse(result["ok"])
            self.assertEqual(result["backend"], "modal")
            self.assertIn("modal_installed", result["failed_checks"])
            self.assertFalse((Path(tmp) / "run" / "run.json").exists())

    def test_dispatch_local_fallback_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(modal_app, "modal", None):
                result = dispatch_discovery(
                    skill="ioi-circuit",
                    model="gpt2",
                    out_dir=Path(tmp) / "run",
                    allow_local_fallback=True,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(result["backend"], "local")
            self.assertGreaterEqual(len(result["run"]["discoveries"]), 1)
            self.assertTrue((Path(tmp) / "run").exists())

    def test_modal_run_json_is_machine_readable_on_setup_failure(self):
        out = StringIO()
        with patch.object(modal_app, "modal", None), redirect_stdout(out):
            main(["modal", "run", "--skill", "ioi-circuit", "--model", "gpt2", "--json"])
        payload = json.loads(out.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["backend"], "modal")
        self.assertIn("modal_installed", payload["failed_checks"])


if __name__ == "__main__":
    unittest.main()
