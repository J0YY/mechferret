import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from mechferret import cluster as cluster_mod
from mechferret.cli import main
from mechferret.cluster import (
    ClusterConfig,
    build_remote_command,
    build_srun_invocation,
    dispatch_discovery_cluster,
    load_cluster_config,
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

    def test_cluster_config_tolerates_malformed_file_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cluster.json"
            path.write_text(
                json.dumps(
                    {
                        "host": " cluster ",
                        "partition": ["bad"],
                        "gres": "",
                        "cpus": "many",
                        "mem": 123,
                        "time": None,
                        "remote_project_dir": " /proj ",
                        "remote_setup": {"bad": "shape"},
                        "python": "",
                        "git_pull": "yes",
                    }
                ),
                encoding="utf-8",
            )
            old_paths = cluster_mod.CONFIG_PATHS
            old_env = {key: os.environ.pop(key, None) for key in ("SLURM_CPUS", "REMOTE_HOST")}
            cluster_mod.CONFIG_PATHS = (path,)
            try:
                cfg = load_cluster_config()
                self.assertEqual(cfg.host, "cluster")
                self.assertEqual(cfg.partition, "")
                self.assertEqual(cfg.gres, "gpu:1")
                self.assertEqual(cfg.cpus, 8)
                self.assertEqual(cfg.mem, "32G")
                self.assertEqual(cfg.time, "02:00:00")
                self.assertEqual(cfg.remote_project_dir, "/proj")
                self.assertEqual(cfg.remote_setup, "")
                self.assertEqual(cfg.python, "python3")
                self.assertTrue(cfg.git_pull)

                os.environ["SLURM_CPUS"] = "0"
                os.environ["REMOTE_HOST"] = " env-host "
                cfg = load_cluster_config()
                self.assertEqual(cfg.host, "env-host")
                self.assertEqual(cfg.cpus, 8)
            finally:
                cluster_mod.CONFIG_PATHS = old_paths
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_cluster_command_builders_sanitize_bad_config_values(self):
        cfg = ClusterConfig(
            host=[],
            partition=[],
            gres={},
            cpus="many",
            mem=[],
            time=None,
            remote_project_dir=[],
            remote_setup={},
            python="",
            git_pull="no",
        )

        cmd = build_remote_command(cfg, ["skill"], 123, {"task": "bad"}, None, Path("runs/x"))
        inv = build_srun_invocation(cfg, cmd)

        self.assertIn("cd ''", cmd)
        self.assertIn("python3 -m mechferret", cmd)
        self.assertNotIn("--model", cmd)
        self.assertEqual(inv[1], "")
        self.assertIn("--cpus-per-task 8", inv[2])
        self.assertIn("--mem 32G", inv[2])

    def test_dry_run_returns_command_without_executing(self):
        result = dispatch_discovery_cluster(skill="ioi-circuit", model="gpt2", dry_run=True, out_dir=tempfile.mkdtemp())
        self.assertTrue(result["dry_run"])
        self.assertIn("ssh", result["command"])

    def test_cluster_cli_json_status_setup_and_dry_run(self):
        env = {
            "REMOTE_HOST": "",
            "REMOTE_PROJECT_DIR": "",
            "SLURM_PARTITION": "",
            "SLURM_GRES": "",
            "SLURM_CPUS": "",
            "SLURM_MEM": "",
            "SLURM_TIME": "",
            "REMOTE_RUN_SETUP": "",
            "REMOTE_GIT_PULL": "",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(cluster_mod, "CONFIG_PATHS", ()), patch.dict(os.environ, env, clear=False):
            cases = (
                (["cluster", "status", "--json"], "status"),
                (["cluster", "setup", "--json"], "setup"),
                (["cluster", "run", "--skill", "ioi-circuit", "--dry-run", "--out", str(Path(tmp) / "run"), "--json"], "run"),
            )
            for args, action in cases:
                out = StringIO()
                with redirect_stdout(out):
                    main(args)
                payload = json.loads(out.getvalue())
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["action"], action)
                if action == "run":
                    self.assertTrue(payload["dry_run"])
                    self.assertIn("command", payload["result"])
                else:
                    self.assertIn("status", payload)

    def test_cluster_run_json_is_machine_readable_on_setup_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cluster_mod, "CONFIG_PATHS", ()):
            env = {"REMOTE_HOST": "", "REMOTE_PROJECT_DIR": ""}
            out = StringIO()
            with patch.dict(os.environ, env, clear=False), redirect_stdout(out):
                main([
                    "cluster",
                    "run",
                    "--skill",
                    "ioi-circuit",
                    "--model",
                    "gpt2",
                    "--out",
                    str(Path(tmp) / "run"),
                    "--json",
                ])
            payload = json.loads(out.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["backend"], "cluster")
        self.assertIn("cluster_configured", payload["failed_checks"])

    def test_unconfigured_fails_closed_by_default(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cluster_mod, "CONFIG_PATHS", ()):
            env = {"REMOTE_HOST": "", "REMOTE_PROJECT_DIR": ""}
            with patch.dict(os.environ, env, clear=False):
                result = dispatch_discovery_cluster(skill="ioi-circuit", model="gpt2", out_dir=Path(tmp) / "run")
            self.assertFalse(result["ok"])
            self.assertEqual(result["backend"], "cluster")
            self.assertIn("cluster_configured", result["failed_checks"])
            self.assertFalse((Path(tmp) / "run" / "run.json").exists())

    def test_unconfigured_local_fallback_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cluster_mod, "CONFIG_PATHS", ()):
            env = {"REMOTE_HOST": "", "REMOTE_PROJECT_DIR": ""}
            with patch.dict(os.environ, env, clear=False):
                result = dispatch_discovery_cluster(
                    skill="ioi-circuit",
                    model="gpt2",
                    out_dir=Path(tmp) / "run",
                    allow_local_fallback=True,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(result["backend"], "local")
            self.assertEqual(result["run"]["provenance"]["backend_requested"], "synthetic")
            self.assertEqual(result["run"]["provenance"]["backend_used"], "synthetic")
            self.assertGreaterEqual(len(result["run"]["discoveries"]), 1)


if __name__ == "__main__":
    unittest.main()
