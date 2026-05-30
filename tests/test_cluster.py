import tempfile
import unittest
from pathlib import Path

from mechferret.cluster import (
    ClusterConfig,
    build_remote_command,
    build_srun_invocation,
    dispatch_discovery_cluster,
)


class ClusterTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ClusterConfig(
            host="mycluster",
            partition="gpu",
            gres="gpu:a100:1",
            cpus=8,
            mem="32G",
            time="02:00:00",
            remote_project_dir="/home/me/mechferret",
            remote_setup="conda activate mf",
            git_pull=True,
        )

    def test_remote_command_includes_setup_pull_and_discover(self):
        cmd = build_remote_command(self.cfg, "ioi-circuit", "", None, "gpt2", "/home/me/mechferret/runs/x")
        self.assertIn("conda activate mf", cmd)
        self.assertIn("cd /home/me/mechferret", cmd)
        self.assertIn("git pull --ff-only", cmd)
        self.assertIn("mechferret discover --backend transformer_lens", cmd)
        self.assertIn("--skill ioi-circuit", cmd)

    def test_srun_invocation_has_resource_flags(self):
        inv = build_srun_invocation(self.cfg, "echo hi")
        self.assertEqual(inv[0], "ssh")
        self.assertEqual(inv[1], "mycluster")
        joined = inv[2]
        self.assertIn("srun", joined)
        self.assertIn("--partition gpu", joined)
        self.assertIn("--gres gpu:a100:1", joined)
        self.assertIn("--time 02:00:00", joined)

    def test_dry_run_returns_command_without_executing(self):
        result = dispatch_discovery_cluster(skill="ioi-circuit", dry_run=True, out_dir=tempfile.mkdtemp())
        self.assertTrue(result["dry_run"])
        self.assertIn("ssh", result["command"])

    def test_unconfigured_falls_back_to_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No REMOTE_HOST/REMOTE_PROJECT_DIR in this process env -> unconfigured.
            import os

            for var in ("REMOTE_HOST", "REMOTE_PROJECT_DIR"):
                os.environ.pop(var, None)
            result = dispatch_discovery_cluster(skill="ioi-circuit", out_dir=Path(tmp) / "run")
            self.assertEqual(result["backend"], "local")
            self.assertGreaterEqual(len(result["run"]["discoveries"]), 1)


if __name__ == "__main__":
    unittest.main()
