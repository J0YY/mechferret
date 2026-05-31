import json
import tempfile
import unittest
import argparse
import importlib.util
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mechferret.cli import main
from mechferret.openvla_sae import command_lines, create_manifest, evaluate_artifacts, feature_report, init_project, smoke_test, status, validate_manifest, write_dossier, write_plan


def load_train_module():
    path = Path("projects/openvla_sae/src/train_sae_from_cache.py")
    spec = importlib.util.spec_from_file_location("train_sae_from_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_cache_module():
    path = Path("projects/openvla_sae/src/cache_openvla_activations.py")
    spec = importlib.util.spec_from_file_location("cache_openvla_activations_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class OpenVLASAETest(unittest.TestCase):
    def test_init_project_scaffolds_packaged_openvla_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "openvla_sae"
            result = init_project(root)
            self.assertTrue(result["ok"])
            self.assertTrue((root / "README.md").exists())
            self.assertTrue((root / "src" / "train_sae_from_cache.py").exists())
            self.assertTrue((root / "scripts" / "install_openvla_min.sh").stat().st_mode & 0o111)
            self.assertTrue((root / "scripts" / "phase1_commands.sh").stat().st_mode & 0o111)
            self.assertGreaterEqual(len(result["files_written"]), 8)
            st = status(project_root=root)
            self.assertTrue(st["ready_local"])
            self.assertTrue(st["template_available"])

    def test_init_project_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "openvla_sae"
            init_project(root)
            result = init_project(root)
            self.assertFalse(result["ok"])
            self.assertTrue(result["existing_files"])
            forced = init_project(root, force=True)
            self.assertTrue(forced["ok"])

    def test_validate_manifest_reports_valid_rows_and_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "img.png"
            image.write_bytes(b"x")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"image_path": str(image), "instruction": "pick up the cup"}),
                        json.dumps({"image_path": str(root / "missing.png"), "instruction": "open drawer"}),
                        json.dumps({"image_path": str(image)}),
                    ]
                ),
                encoding="utf-8",
            )
            result = validate_manifest(manifest)
            self.assertEqual(result["rows"], 3)
            self.assertEqual(result["valid_rows"], 2)
            self.assertEqual(len(result["missing_images"]), 1)
            self.assertTrue(result["errors"])

    def test_manifest_helpers_tolerate_malformed_rows_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "img.png"
            image.write_bytes(b"x")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        "[]",
                        json.dumps({"image_path": str(image), "instruction": b"pick".decode()}),
                        json.dumps({"image_path": 123, "instruction": "bad path"}),
                        "{not-json",
                    ]
                ),
                encoding="utf-8",
            )

            result = validate_manifest(manifest, max_rows="many")  # type: ignore[arg-type]
            self.assertEqual(result["rows"], 4)
            self.assertEqual(result["valid_rows"], 1)
            self.assertTrue(any("expected object" in row for row in result["errors"]))
            self.assertTrue(any("missing image_path" in row for row in result["errors"]))

            images = root / "images"
            images.mkdir()
            (images / "a.png").write_bytes(b"x")
            (images / "b.jpg").write_bytes(b"x")
            created = create_manifest(
                images,
                root / "created.jsonl",
                instruction=[],
                action={},
                limit="bad",
                force="yes",
            )  # type: ignore[arg-type]
            rows = [json.loads(line) for line in (root / "created.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(created["rows_written"], 2)
            self.assertEqual(rows[0]["instruction"], "perform the task shown in the image")
            self.assertNotIn("action", rows[0])

    def test_status_sees_project_files(self):
        result = status()
        self.assertTrue(result["ready_local"])
        self.assertIn("src/cache_openvla_activations.py", result["files"])

    def test_write_plan_creates_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = write_plan(out_dir=tmp)
            self.assertTrue(Path(result["markdown"]).exists())
            self.assertTrue(Path(result["json"]).exists())
            self.assertIn("OpenVLA SAE Workflow", Path(result["markdown"]).read_text(encoding="utf-8"))

    def test_commands_reference_phase1_script(self):
        self.assertIn("phase1_commands.sh", command_lines())

    def test_create_manifest_scans_images_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "a.png").write_bytes(b"x")
            (images / "b.jpg").write_bytes(b"x")
            (images / "ignore.txt").write_text("no", encoding="utf-8")
            manifest = root / "data" / "manifest.jsonl"
            result = create_manifest(images, manifest, instruction="pick up the block")
            self.assertEqual(result["rows_written"], 2)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["instruction"], "pick up the block")
            with self.assertRaises(FileExistsError):
                create_manifest(images, manifest)

    def test_smoke_reports_missing_torch_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = smoke_test(out_dir=tmp)
            if result["ok"]:
                self.assertTrue(Path(result["checkpoint"]).exists())
                self.assertIn("final_loss", result)
            else:
                self.assertEqual(result["reason"], "torch is not installed")
                self.assertIn("pip install", result["install"])
            self.assertTrue(Path(result["metrics"]).exists())
            self.assertTrue(Path(result["report"]).exists())

    def test_train_script_loads_config_and_applies_quick_overrides_without_torch(self):
        mod = load_train_module()
        cfg = mod.load_config("projects/openvla_sae/configs/phase1.yaml")
        self.assertEqual(cfg["sae"]["k"], 64)
        args = argparse.Namespace(
            steps=3,
            batch_size=8,
            k=2,
            lr=None,
            expansion_factor=None,
            seed=123,
        )
        merged = mod.apply_overrides(cfg, args)
        self.assertEqual(merged["sae"]["steps"], 3)
        self.assertEqual(merged["sae"]["batch_size"], 8)
        self.assertEqual(merged["sae"]["k"], 2)
        self.assertEqual(merged["sae"]["seed"], 123)

    def test_cache_script_dry_run_helpers_work_without_heavy_imports(self):
        mod = load_cache_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "img.png"
            image.write_bytes(b"x")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"image_path": str(image), "instruction": "pick"}) + "\n", encoding="utf-8")
            rows, errors = mod.load_manifest(manifest, 10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(errors, [])
            args = mod.build_parser().parse_args([
                "--manifest", str(manifest),
                "--out-dir", str(root / "cache"),
                "--site", "language_model.model.layers.24",
                "--dry-run",
            ])
            report = mod.dry_run_report(args)
            self.assertEqual(report["valid_rows"], 1)
            self.assertIn("torch", report["dependencies"])

    def test_evaluate_artifacts_writes_report_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "000000.pt").write_bytes(b"not a real torch file")
            checkpoint = root / "sae.pt"
            checkpoint.write_bytes(b"not a real checkpoint")
            (root / "metrics.json").write_text(json.dumps({"final_loss": 0.5}), encoding="utf-8")
            result = evaluate_artifacts(cache, checkpoint, root / "eval")
            self.assertEqual(result["cache_files"], 1)
            self.assertTrue(result["checkpoint_exists"])
            self.assertTrue(Path(result["json"]).exists())
            self.assertTrue(Path(result["report"]).exists())
            if not result["dependencies"]["torch"]:
                self.assertIn("Install torch", " ".join(result["next_actions"]))

    def test_eval_feature_and_dossier_tolerate_malformed_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "000000.pt").write_bytes(b"not a real torch file")
            checkpoint = root / "sae.pt"
            checkpoint.write_bytes(b"not a real checkpoint")
            (root / "metrics.json").write_text(json.dumps({"final_loss": float("nan")}), encoding="utf-8")

            eval_result = evaluate_artifacts(cache, checkpoint, root / "eval_out")
            self.assertTrue(Path(eval_result["json"]).exists())
            self.assertIn("final_loss", eval_result["metrics"])
            feature_result = feature_report(cache, checkpoint, root / "features_out", top_k="bad", max_files="bad")  # type: ignore[arg-type]
            self.assertEqual(feature_result["top_k"], 20)
            self.assertEqual(feature_result["cache_files_used"], 1)
            self.assertTrue(Path(feature_result["json"]).exists())

            eval_dir = root / "eval"
            feature_dir = root / "features"
            eval_dir.mkdir()
            feature_dir.mkdir()
            (eval_dir / "openvla_sae_eval.json").write_text("[]", encoding="utf-8")
            (feature_dir / "openvla_sae_features.json").write_text(
                json.dumps({"features": [{"feature": object()}]}, default=str),
                encoding="utf-8",
            )
            dossier = write_dossier(
                out_dir=root / "dossier",
                project_root=object(),
                cache_dir=object(),
                checkpoint=object(),
                eval_dir=eval_dir,
                features_dir=feature_dir,
            )  # type: ignore[arg-type]
            self.assertTrue(Path(dossier["json"]).exists())
            self.assertTrue(Path(dossier["markdown"]).exists())

    def test_feature_report_writes_actionable_report_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "000000.pt").write_bytes(b"not a real torch file")
            checkpoint = root / "sae.pt"
            checkpoint.write_bytes(b"not a real checkpoint")
            result = feature_report(cache, checkpoint, root / "features")
            self.assertEqual(result["cache_files_used"], 1)
            self.assertTrue(Path(result["json"]).exists())
            self.assertTrue(Path(result["report"]).exists())
            if not result["dependencies"]["torch"]:
                self.assertIn("Install torch", " ".join(result["next_actions"]))

    def test_write_dossier_collects_eval_feature_and_outline_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            feature_dir = root / "features"
            eval_dir.mkdir()
            feature_dir.mkdir()
            (eval_dir / "openvla_sae_eval.json").write_text(
                json.dumps({"artifacts": {"report": str(eval_dir / "openvla_sae_eval.md")}, "metrics": {"final_loss": 0.4}}),
                encoding="utf-8",
            )
            (feature_dir / "openvla_sae_features.json").write_text(
                json.dumps({"artifacts": {"report": str(feature_dir / "openvla_sae_features.md")}, "features": []}),
                encoding="utf-8",
            )
            result = write_dossier(
                out_dir=root / "dossier",
                cache_dir=root / "cache",
                checkpoint=root / "sae.pt",
                eval_dir=eval_dir,
                features_dir=feature_dir,
            )
            self.assertTrue(Path(result["json"]).exists())
            self.assertTrue(Path(result["markdown"]).exists())
            text = Path(result["markdown"]).read_text(encoding="utf-8")
            self.assertIn("OpenVLA SAE Research Dossier", text)
            self.assertIn("Paper Outline", text)
            self.assertIn(str(eval_dir / "openvla_sae_eval.md"), text)
            self.assertIn(str(feature_dir / "openvla_sae_features.md"), text)

    def test_cli_openvla_sae_json_workflow_outputs_parseable_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "openvla_sae"
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "scene.png").write_bytes(b"image")
            manifest = root / "manifest.jsonl"

            cases = (
                (
                    ["sae", "openvla", "status", "--project-root", str(project), "--json"],
                    "status",
                    lambda payload: self.assertFalse(payload["ready_local"]),
                ),
                (
                    ["sae", "openvla", "init", "--project-root", str(project), "--json"],
                    "init",
                    lambda payload: self.assertTrue(payload["files_written"]),
                ),
                (
                    ["sae", "openvla", "plan", "--project-root", str(project), "--out", str(root / "plan"), "--json"],
                    "plan",
                    lambda payload: self.assertTrue(Path(payload["markdown"]).exists()),
                ),
                (
                    ["sae", "openvla", "commands", "--project-root", str(project), "--json"],
                    "commands",
                    lambda payload: self.assertIn("phase1_commands.sh", payload["commands"]),
                ),
                (
                    [
                        "sae",
                        "openvla",
                        "create-manifest",
                        "--image-dir",
                        str(image_dir),
                        "--manifest",
                        str(manifest),
                        "--json",
                    ],
                    "create-manifest",
                    lambda payload: self.assertEqual(payload["rows_written"], 1),
                ),
                (
                    ["sae", "openvla", "validate-manifest", "--manifest", str(manifest), "--json"],
                    "validate-manifest",
                    lambda payload: self.assertEqual(payload["valid_rows"], 1),
                ),
                (
                    [
                        "sae",
                        "openvla",
                        "smoke",
                        "--out",
                        str(root / "smoke"),
                        "--d-model",
                        "4",
                        "--tokens",
                        "8",
                        "--steps",
                        "1",
                        "--k",
                        "2",
                        "--json",
                    ],
                    "smoke",
                    lambda payload: self.assertTrue(Path(payload["out_dir"]).exists()),
                ),
            )

            for args, action, check in cases:
                out = StringIO()
                with redirect_stdout(out):
                    main(args)
                payload = json.loads(out.getvalue())
                self.assertIn("ok", payload)
                if action != "smoke":
                    self.assertTrue(payload["ok"])
                self.assertEqual(payload["project"], "openvla")
                self.assertEqual(payload["action"], action)
                check(payload)

            cache = root / "cache"
            cache.mkdir()
            (cache / "000000.pt").write_bytes(b"not a real torch file")
            checkpoint = root / "sae.pt"
            checkpoint.write_bytes(b"not a real checkpoint")
            (root / "metrics.json").write_text(json.dumps({"final_loss": 0.5}), encoding="utf-8")

            eval_out = StringIO()
            with redirect_stdout(eval_out):
                main(
                    [
                        "sae",
                        "openvla",
                        "eval",
                        "--cache-dir",
                        str(cache),
                        "--checkpoint",
                        str(checkpoint),
                        "--out",
                        str(root / "eval"),
                        "--json",
                    ]
                )
            eval_payload = json.loads(eval_out.getvalue())
            self.assertIn("ok", eval_payload)
            self.assertEqual(eval_payload["action"], "eval")
            self.assertTrue(Path(eval_payload["json"]).exists())

            features_out = StringIO()
            with redirect_stdout(features_out):
                main(
                    [
                        "sae",
                        "openvla",
                        "features",
                        "--cache-dir",
                        str(cache),
                        "--checkpoint",
                        str(checkpoint),
                        "--out",
                        str(root / "features"),
                        "--json",
                    ]
                )
            features_payload = json.loads(features_out.getvalue())
            self.assertIn("ok", features_payload)
            self.assertEqual(features_payload["action"], "features")
            self.assertTrue(Path(features_payload["json"]).exists())

            dossier_out = StringIO()
            with redirect_stdout(dossier_out):
                main(
                    [
                        "sae",
                        "openvla",
                        "dossier",
                        "--project-root",
                        str(project),
                        "--manifest",
                        str(manifest),
                        "--cache-dir",
                        str(cache),
                        "--checkpoint",
                        str(checkpoint),
                        "--eval-dir",
                        str(root / "eval"),
                        "--features-dir",
                        str(root / "features"),
                        "--out",
                        str(root / "dossier"),
                        "--json",
                    ]
                )
            dossier_payload = json.loads(dossier_out.getvalue())
            self.assertIn("ok", dossier_payload)
            self.assertEqual(dossier_payload["action"], "dossier")
            self.assertTrue(Path(dossier_payload["markdown"]).exists())

    def test_cli_openvla_sae_json_reports_usage_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for args, action in (
                (["sae", "openvla", "validate-manifest", "--json"], "validate-manifest"),
                (["sae", "openvla", "create-manifest", "--manifest", str(root / "manifest.jsonl"), "--json"], "create-manifest"),
            ):
                out = StringIO()
                with self.assertRaises(SystemExit) as ctx:
                    with redirect_stdout(out):
                        main(args)
                self.assertEqual(ctx.exception.code, 2)
                payload = json.loads(out.getvalue())
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["project"], "openvla")
                self.assertEqual(payload["action"], action)
                self.assertIn("next_actions", payload)


if __name__ == "__main__":
    unittest.main()
