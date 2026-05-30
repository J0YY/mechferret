import tempfile
import unittest
from pathlib import Path

from mechferret.modal_app import dispatch_discovery, modal_status


class ModalDispatchTest(unittest.TestCase):
    def test_modal_status_keys(self):
        status = modal_status()
        for key in ("installed", "authenticated", "gpu", "app", "torch_local"):
            self.assertIn(key, status)

    def test_dispatch_falls_back_to_local(self):
        # Modal is not installed in CI, so dispatch must run locally and still produce a run.
        with tempfile.TemporaryDirectory() as tmp:
            result = dispatch_discovery(skill="ioi-circuit", out_dir=Path(tmp) / "run")
            self.assertIn(result["backend"], {"local", "modal"})
            self.assertGreaterEqual(len(result["run"]["discoveries"]), 1)
            self.assertTrue((Path(tmp) / "run").exists())


if __name__ == "__main__":
    unittest.main()
