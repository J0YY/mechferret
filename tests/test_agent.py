import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mechferret import agent


class AgentToolTest(unittest.TestCase):
    def test_tool_schemas_are_well_formed(self):
        from mechferret import tools

        names = {t["name"] for t in tools.TOOL_SPECS}
        self.assertEqual(names, set(tools.HANDLERS))
        # the full Claude-Code-style suite is present
        for expected in (
            "bash", "read_file", "write_file", "edit_file", "grep", "glob",
            "web_search", "web_fetch", "audit_run", "write_paper", "review_paper",
            "bundle_artifacts", "verify_bundle", "resolve_artifact", "project_status", "list_runs", "verify_run", "openvla_sae", "run_research",
            "list_tool_results", "clean_tool_results",
        ):
            self.assertIn(expected, names)
        for tool in tools.TOOL_SPECS:
            self.assertIn("description", tool)
            self.assertEqual(tool["parameters"]["type"], "object")

        discovery = next(tool for tool in tools.TOOL_SPECS if tool["name"] == "run_discovery")
        discovery_props = discovery["parameters"]["properties"]
        for expected in (
            "backend",
            "out_dir",
            "db_path",
            "source_paths",
            "urls",
            "provider",
            "llm_model",
            "max_rounds",
            "max_experiments",
            "max_gpu_seconds",
            "include_memory",
            "no_memory",
            "allow_mismatch",
            "allow_seed_corpus",
        ):
            self.assertIn(expected, discovery_props)
        self.assertIn("local", discovery_props["provider"]["enum"])
        resolve_props = next(tool for tool in tools.TOOL_SPECS if tool["name"] == "resolve_artifact")["parameters"]["properties"]
        self.assertIn("selection", resolve_props)
        self.assertEqual(resolve_props["selection"]["enum"], ["latest", "best", "ready"])
        research_props = next(tool for tool in tools.TOOL_SPECS if tool["name"] == "run_research")["parameters"]["properties"]
        self.assertIn("allow_seed_corpus", research_props)
        self.assertIn("source_paths", research_props)
        self.assertIn("selection", next(tool for tool in tools.TOOL_SPECS if tool["name"] == "list_runs")["parameters"]["properties"])
        self.assertIn("selection", next(tool for tool in tools.TOOL_SPECS if tool["name"] == "audit_run")["parameters"]["properties"])
        self.assertIn("selection", next(tool for tool in tools.TOOL_SPECS if tool["name"] == "verify_run")["parameters"]["properties"])
        self.assertIn("selection", next(tool for tool in tools.TOOL_SPECS if tool["name"] == "review_paper")["parameters"]["properties"])
        self.assertIn("selection", next(tool for tool in tools.TOOL_SPECS if tool["name"] == "verify_bundle")["parameters"]["properties"])
        write_props = next(tool for tool in tools.TOOL_SPECS if tool["name"] == "write_paper")["parameters"]["properties"]
        self.assertEqual(write_props["compile_timeout"]["type"], "integer")

    def test_resolve_artifact_tool_returns_json(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "runs" / "demo" / "QUICKSTART.md"
            target.parent.mkdir(parents=True)
            target.write_text("# quickstart\n", encoding="utf-8")
            payload = json.loads(tools.run_tool("resolve_artifact", {"target": "quickstart", "runs_root": str(root / "runs")}))
            self.assertTrue(payload["exists"])
            self.assertEqual(Path(payload["path"]), target)
            index = json.loads(tools.run_tool("resolve_artifact", {"target": "all", "runs_root": str(root / "runs")}))
            self.assertTrue(index["exists"])
            self.assertTrue(index["artifacts"]["quickstart"]["exists"])
            self.assertIn("graph", index["artifacts"])
            self.assertIn("trace", index["artifacts"])

    def test_project_status_tool_returns_json(self):
        from mechferret import tools
        from mechferret.ops import init_project_notes, run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            payload = json.loads(
                tools.run_tool(
                    "project_status",
                    {
                        "runs_root": str(root / "runs"),
                        "db_path": str(root / "memory.sqlite"),
                        "notes_root": str(root),
                        "project_root": str(root / "openvla"),
                    },
                )
            )
            self.assertEqual(payload["state"], "ready")
            self.assertTrue(payload["latest_run"]["exists"])
            self.assertTrue(payload["verification"]["passed"])
            self.assertIn("paper", payload["available_artifacts"])
            if payload.get("tool_output_truncated"):
                self.assertTrue(payload["verification"]["omitted"])

    def test_bundle_artifacts_tool_returns_zip_path(self):
        from mechferret import tools
        from mechferret.ops import init_project_notes, run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            payload = json.loads(
                tools.run_tool(
                    "bundle_artifacts",
                    {
                        "runs_root": str(root / "runs"),
                        "out": str(root / "bundle"),
                        "notes_root": str(root),
                        "project_root": str(root / "openvla"),
                    },
                )
            )
            self.assertTrue(payload["ok"])
            self.assertTrue(Path(payload["path"]).exists())
            self.assertIn("manifest", payload)

    def test_list_runs_tool_returns_recent_runs(self):
        from mechferret import tools
        from mechferret.ops import run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            payload = json.loads(
                tools.run_tool(
                    "list_runs",
                    {"runs_root": str(root / "runs"), "limit": 5},
                )
            )
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["shown"], 1)
            self.assertEqual(payload["selection"], "best")
            self.assertTrue(payload["selected"]["audit"]["passed"])
            self.assertTrue(payload["runs"][0]["audit"]["passed"])
            self.assertTrue(payload["runs"][0]["artifacts"]["report"])

    def test_verify_run_tool_returns_manifest_status(self):
        from mechferret import tools
        from mechferret.ops import run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            payload = json.loads(
                tools.run_tool(
                    "verify_run",
                    {"path": str(root / "runs" / "demo" / "run.json")},
                )
            )
            self.assertTrue(payload["passed"])
            self.assertTrue(payload["manifest"].endswith("manifest.json"))

    def test_verify_run_tool_can_repair_stale_manifest_coverage(self):
        from mechferret import tools
        from mechferret.ops import run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            run_json = root / "runs" / "demo" / "run.json"
            note = root / "runs" / "demo" / "notes.txt"
            note.write_text("extra artifact\n", encoding="utf-8")
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["artifacts"]["notes"] = str(note)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            result = json.loads(tools.run_tool("verify_run", {"path": str(run_json), "repair": True}))
            self.assertTrue(result["passed"])
            self.assertTrue(result["repair_attempted"])
            self.assertIn("manifest_tracks_declared_artifact:notes", result["before_failed_checks"])

    def test_verify_bundle_tool_checks_portable_bundle_manifest(self):
        from mechferret import tools
        from mechferret.ops import bundle_run_artifacts, run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            bundle = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports")["path"]

            result = json.loads(tools.run_tool("verify_bundle", {"path": bundle}))
            self.assertTrue(result["passed"])
            self.assertTrue(any(check["name"].startswith("bundle_file_sha256:") for check in result["checks"]))

    def test_verify_bundle_tool_supports_run_selection(self):
        from mechferret import tools
        from mechferret.ops import bundle_run_artifacts, run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "good", db_path=root / "memory.sqlite")
            run_quickstart("demo", out_dir=root / "runs" / "bad", db_path=root / "memory.sqlite")
            good = root / "runs" / "good" / "run.json"
            bad = root / "runs" / "bad" / "run.json"
            (root / "runs" / "bad" / "paper" / "main.tex").unlink()
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload.get("artifacts", {}).pop("paper", None)
            bad.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(bad, (1_700_001_000, 1_700_001_000))
            bundle_run_artifacts(runs_root=root / "runs", selection="best", out=root / "exports")
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(bad, (1_700_001_000, 1_700_001_000))

            latest = json.loads(tools.run_tool("verify_bundle", {"runs_root": str(root / "runs"), "selection": "latest"}))
            best = json.loads(tools.run_tool("verify_bundle", {"runs_root": str(root / "runs"), "selection": "best"}))
            self.assertFalse(latest["passed"])
            self.assertTrue(best["passed"])

    def test_audit_and_verify_tools_support_run_selection(self):
        from mechferret import tools
        from mechferret.ops import run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "good", db_path=root / "memory.sqlite")
            run_quickstart("demo", out_dir=root / "runs" / "bad", db_path=root / "memory.sqlite")
            good = root / "runs" / "good" / "run.json"
            bad_report = root / "runs" / "bad" / "report.md"
            bad_report.write_text("tampered\n", encoding="utf-8")
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(root / "runs" / "bad" / "run.json", (1_700_001_000, 1_700_001_000))

            latest_audit = json.loads(tools.run_tool("audit_run", {"runs_root": str(root / "runs"), "selection": "latest"}))
            best_audit = json.loads(tools.run_tool("audit_run", {"runs_root": str(root / "runs"), "selection": "best"}))
            best_verify = json.loads(tools.run_tool("verify_run", {"runs_root": str(root / "runs"), "selection": "best"}))
            best_paper = json.loads(tools.run_tool("resolve_artifact", {"target": "paper", "runs_root": str(root / "runs"), "selection": "best"}))
            self.assertFalse(latest_audit["passed"])
            self.assertTrue(best_audit["passed"])
            self.assertEqual(Path(best_audit["path"]), good)
            self.assertTrue(best_verify["passed"])
            self.assertEqual(Path(best_verify["path"]), good)
            self.assertTrue(best_paper["exists"])
            self.assertEqual(Path(best_paper["selected_run"]), good)
            self.assertEqual(best_paper["selection"], "best")

    def test_run_dependent_tools_explain_selection_failure(self):
        from mechferret import tools
        from mechferret.ops import run_quickstart

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "broken", db_path=root / "memory.sqlite")
            (root / "runs" / "broken" / "report.md").write_text("tampered\n", encoding="utf-8")

            args = {"runs_root": str(root / "runs"), "selection": "ready"}
            audit = json.loads(tools.run_tool("audit_run", args))
            verify = json.loads(tools.run_tool("verify_run", args))
            paper = json.loads(tools.run_tool("write_paper", args))

            for payload in (audit, verify, paper):
                self.assertEqual(payload["selection"], "ready")
                self.assertEqual(Path(payload["runs_root"]), root / "runs")
                self.assertEqual(payload["selected_path"], "")
                self.assertIn("run_selection", payload["failed_checks"])
                self.assertIn("No audit-passing run found", " ".join(payload["next_actions"]))
            self.assertFalse(audit["passed"])
            self.assertFalse(verify["passed"])
            self.assertFalse(paper["ok"])

    def test_selection_tools_return_json_for_invalid_selection_values(self):
        from mechferret import tools

        cases = [
            ("project_status", {"selection": "newest"}),
            ("list_runs", {"select": "release"}),
            ("audit_run", {"selection": ["best"]}),
            ("verify_run", {"selection": "broken"}),
            ("write_paper", {"selection": "accepted"}),
            ("review_paper", {"selection": "published"}),
            ("bundle_artifacts", {"selection": "ship"}),
            ("verify_bundle", {"selection": "ship"}),
            ("resolve_artifact", {"target": "paper", "selection": "ship"}),
        ]
        for tool_name, args in cases:
            with self.subTest(tool=tool_name):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], "invalid selection policy")
                self.assertIn("selection_policy", payload["failed_checks"])
                self.assertEqual(payload["allowed_selection"], ["best", "latest", "ready"])
                self.assertIn("Use one of", " ".join(payload["next_actions"]))

    def test_list_tools_return_json_for_invalid_limit_values(self):
        from mechferret import tools

        cases = [
            ("list_runs", {"limit": "many"}),
            ("list_tool_results", {"limit": -1}),
        ]
        for tool_name, args in cases:
            with self.subTest(tool=tool_name):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], "invalid integer argument: limit")
                self.assertIn("limit_argument", payload["failed_checks"])
                self.assertIn("integer >=", payload["expected"])

    def test_file_and_cleanup_tools_return_json_for_invalid_numeric_values(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "notes.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")

            read_offset = json.loads(tools.run_tool("read_file", {"path": str(target), "offset": "late"}))
            read_limit = json.loads(tools.run_tool("read_file", {"path": str(target), "limit": -1}))
            clean_keep = json.loads(tools.run_tool("clean_tool_results", {"keep_latest": "many"}))
            clean_age = json.loads(tools.run_tool("clean_tool_results", {"max_age_days": "soon"}))
            discovery_gpu = json.loads(tools.run_tool("run_discovery", {"max_gpu_seconds": -1}))

            self.assertEqual(read_offset["error"], "invalid integer argument: offset")
            self.assertIn("offset_argument", read_offset["failed_checks"])
            self.assertEqual(read_limit["error"], "invalid integer argument: limit")
            self.assertIn("limit_argument", read_limit["failed_checks"])
            self.assertEqual(clean_keep["error"], "invalid integer argument: keep_latest")
            self.assertIn("keep_latest_argument", clean_keep["failed_checks"])
            self.assertEqual(clean_age["error"], "invalid number argument: max_age_days")
            self.assertIn("max_age_days_argument", clean_age["failed_checks"])
            self.assertEqual(discovery_gpu["error"], "invalid number argument: max_gpu_seconds")
            self.assertIn("max_gpu_seconds_argument", discovery_gpu["failed_checks"])

    def test_command_search_and_research_tools_validate_numeric_values(self):
        from mechferret import tools

        cases = [
            ("bash", {"command": "echo ok", "timeout": 0}, "timeout"),
            ("web_search", {"query": "mechanistic interpretability", "max_results": "many"}, "max_results"),
            ("web_fetch", {"url": "https://example.com", "max_chars": -1}, "max_chars"),
            ("arxiv_search", {"query": "sparse autoencoder", "max_results": []}, "max_results"),
            ("run_research", {"question": "What should we test?", "max_rounds": 0}, "max_rounds"),
            ("run_discovery", {"max_rounds": 0}, "max_rounds"),
            ("run_discovery", {"max_experiments": "many"}, "max_experiments"),
            ("openvla_sae", {"action": "create-manifest", "image_dir": "frames", "limit": 0}, "limit"),
        ]
        for tool_name, args, argument in cases:
            with self.subTest(tool=tool_name):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid integer argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertIn("integer >=", payload["expected"])

    def test_openvla_tool_validates_numeric_values(self):
        from mechferret import tools

        cases = [
            ({"action": "smoke", "d_model": 0}, "d_model"),
            ({"action": "smoke", "tokens": "many"}, "tokens"),
            ({"action": "smoke", "steps": -1}, "steps"),
            ({"action": "smoke", "k": []}, "k"),
            ({"action": "features", "top_features": 0}, "top_features"),
            ({"action": "features", "max_files": "all"}, "max_files"),
        ]
        for args, argument in cases:
            with self.subTest(argument=argument):
                payload = json.loads(tools.run_tool("openvla_sae", args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid integer argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertIn("integer >=", payload["expected"])

    def test_tools_validate_required_string_values(self):
        from mechferret import tools

        cases = [
            ("bash", {}, "command"),
            ("read_file", {}, "path"),
            ("write_file", {"path": ""}, "path"),
            ("edit_file", {"path": []}, "path"),
            ("edit_file", {"path": "notes.txt"}, "old_string"),
            ("glob", {}, "pattern"),
            ("grep", {"pattern": ""}, "pattern"),
            ("web_search", {"query": ""}, "query"),
            ("web_fetch", {"url": None}, "url"),
            ("arxiv_search", {}, "query"),
            ("neuronpedia_search", {"model_id": "gpt2-small"}, "query"),
            ("openvla_sae", {"action": "validate-manifest"}, "manifest"),
            ("openvla_sae", {"action": "create-manifest"}, "image_dir"),
        ]
        for tool_name, args, argument in cases:
            with self.subTest(tool=tool_name):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid string argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertEqual(payload["expected"], "non-empty string")

    def test_openvla_tool_validates_optional_string_values(self):
        from mechferret import tools

        cases = [
            ({"action": "status", "project_root": []}, "project_root"),
            ({"action": "status", "manifest": []}, "manifest"),
            ({"action": "plan", "out_dir": []}, "out_dir"),
            ({"action": "create-manifest", "image_dir": "frames", "instruction": []}, "instruction"),
            ({"action": "create-manifest", "image_dir": "frames", "action_label": {}}, "action_label"),
            ({"action": "smoke", "out_dir": []}, "out_dir"),
            ({"action": "eval", "cache_dir": []}, "cache_dir"),
            ({"action": "eval", "checkpoint": {}}, "checkpoint"),
            ({"action": "features", "out_dir": []}, "out_dir"),
            ({"action": "dossier", "eval_dir": []}, "eval_dir"),
            ({"action": "dossier", "features_dir": {}}, "features_dir"),
        ]
        for args, argument in cases:
            with self.subTest(argument=argument):
                payload = json.loads(tools.run_tool("openvla_sae", args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid string argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertEqual(payload["expected"], "string")

    def test_run_artifact_tools_validate_path_like_string_values(self):
        from mechferret import tools

        cases = [
            ("project_status", {"runs_root": []}, "runs_root"),
            ("project_status", {"db_path": {}}, "db_path"),
            ("project_status", {"notes_root": []}, "notes_root"),
            ("project_status", {"project_root": {}}, "project_root"),
            ("list_runs", {"runs_root": []}, "runs_root"),
            ("audit_run", {"path": []}, "path"),
            ("audit_run", {"runs_root": {}}, "runs_root"),
            ("verify_run", {"path": []}, "path"),
            ("verify_run", {"runs_root": {}}, "runs_root"),
            ("write_paper", {"path": []}, "path"),
            ("write_paper", {"out_dir": {}}, "out_dir"),
            ("write_paper", {"runs_root": []}, "runs_root"),
            ("write_paper", {"model": {}}, "model"),
            ("review_paper", {"path": []}, "path"),
            ("review_paper", {"out_dir": {}}, "out_dir"),
            ("review_paper", {"runs_root": []}, "runs_root"),
            ("review_paper", {"model": []}, "model"),
            ("bundle_artifacts", {"path": []}, "path"),
            ("bundle_artifacts", {"runs_root": {}}, "runs_root"),
            ("bundle_artifacts", {"out": []}, "out"),
            ("bundle_artifacts", {"notes_root": {}}, "notes_root"),
            ("bundle_artifacts", {"project_root": []}, "project_root"),
            ("verify_bundle", {"path": []}, "path"),
            ("verify_bundle", {"runs_root": {}}, "runs_root"),
            ("resolve_artifact", {"target": []}, "target"),
            ("resolve_artifact", {"runs_root": {}}, "runs_root"),
            ("resolve_artifact", {"project_root": []}, "project_root"),
        ]
        for tool_name, args, argument in cases:
            with self.subTest(tool=tool_name, argument=argument):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid string argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertEqual(payload["expected"], "string")

    def test_file_search_tools_validate_optional_string_values(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "notes.txt"
            target.write_text("alpha\n", encoding="utf-8")

            cases = [
                ("write_file", {"path": str(target), "content": []}, "content"),
                ("edit_file", {"path": str(target), "old_string": "alpha", "new_string": []}, "new_string"),
                ("list_dir", {"path": []}, "path"),
                ("glob", {"pattern": "*.py", "path": {}}, "path"),
                ("grep", {"pattern": "alpha", "path": []}, "path"),
                ("grep", {"pattern": "alpha", "glob": []}, "glob"),
            ]
            for tool_name, args, argument in cases:
                with self.subTest(tool=tool_name, argument=argument):
                    payload = json.loads(tools.run_tool(tool_name, args))
                    self.assertFalse(payload["ok"])
                    self.assertEqual(payload["error"], f"invalid string argument: {argument}")
                    self.assertIn(f"{argument}_argument", payload["failed_checks"])
                    self.assertEqual(payload["expected"], "string")

    def test_research_tools_validate_source_and_metadata_values(self):
        from mechferret import tools

        string_cases = [
            ("run_research", {"question": []}, "question"),
            ("run_research", {"out_dir": []}, "out_dir"),
            ("run_research", {"db_path": {}}, "db_path"),
            ("run_research", {"model": []}, "model"),
            ("run_discovery", {"question": []}, "question"),
            ("run_discovery", {"skill": {}}, "skill"),
            ("run_discovery", {"task": []}, "task"),
            ("run_discovery", {"model": {}}, "model"),
            ("run_discovery", {"llm_model": []}, "llm_model"),
            ("run_discovery", {"out_dir": {}}, "out_dir"),
            ("run_discovery", {"db_path": []}, "db_path"),
        ]
        for tool_name, args, argument in string_cases:
            with self.subTest(tool=tool_name, argument=argument):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid string argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertEqual(payload["expected"], "string")

        list_cases = [
            ("run_research", {"source_paths": "notes.md"}, "source_paths"),
            ("run_research", {"source_paths": ["notes.md", ""]}, "source_paths"),
            ("run_research", {"urls": ["https://example.com", []]}, "urls"),
            ("run_discovery", {"source_paths": {}}, "source_paths"),
            ("run_discovery", {"urls": [""]}, "urls"),
        ]
        for tool_name, args, argument in list_cases:
            with self.subTest(tool=tool_name, argument=argument):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid string list argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertEqual(payload["expected"], "list of non-empty strings")

    def test_novelty_and_option_tools_validate_structured_inputs(self):
        from mechferret import tools

        missing_idea = json.loads(tools.run_tool("verify_novelty", {}))
        self.assertFalse(missing_idea["ok"])
        self.assertEqual(missing_idea["error"], "invalid string argument: idea")
        self.assertIn("idea_argument", missing_idea["failed_checks"])

        bad_queries = json.loads(tools.run_tool("verify_novelty", {"idea": "x", "queries": [""]}))
        self.assertFalse(bad_queries["ok"])
        self.assertEqual(bad_queries["error"], "invalid string list argument: queries")
        self.assertIn("queries_argument", bad_queries["failed_checks"])

        bad_options = json.loads(tools.run_tool("present_options", {"options": "pick one"}))
        self.assertFalse(bad_options["ok"])
        self.assertEqual(bad_options["error"], "invalid object list argument: options")
        self.assertEqual(bad_options["expected"], "list of objects")

        bad_title = json.loads(tools.run_tool("present_options", {"options": [{"summary": "missing title"}]}))
        self.assertFalse(bad_title["ok"])
        self.assertEqual(bad_title["error"], "invalid object list argument: options")
        self.assertEqual(bad_title["expected"], "objects with non-empty string title")
        self.assertEqual(bad_title["index"], 0)

        bad_novelty = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "missing novelty evidence",
                            "detail": "A direction without verify_novelty fields should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "required_delta": "Show a measurable delta.",
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_novelty["ok"])
        self.assertEqual(bad_novelty["expected"], "objects with novelty_risk from verify_novelty assessment")

        ok = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Run audit",
                            "summary": "...",
                            "detail": "Audit a candidate direction against retrieved papers and required experimental deltas.",
                            "citations": ["Closest Paper https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists; specify the delta.",
                            "closest_prior_art": ["Closest Paper https://arxiv.org/abs/2501.0001"],
                            "required_delta": "Show a causal ablation that differs from prior work.",
                        }
                    ]
                },
            )
        )
        self.assertEqual(ok["options"], ["Run audit"])
        self.assertEqual(ok["option_details"][0]["novelty_risk"], "medium_prior_art_risk")
        self.assertIn("Closest Paper", ok["option_details"][0]["citations"][0])
        self.assertIn("Closest Paper", ok["option_details"][0]["closest_prior_art"][0])
        self.assertIn("causal ablation", ok["option_details"][0]["required_delta"])

    def test_run_discovery_requires_explicit_model_without_modelled_skill(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            payload = json.loads(
                tools.run_tool(
                    "run_discovery",
                    {
                        "skill": "ioi-circuit",
                        "backend": "synthetic",
                        "out_dir": str(Path(tmp) / "run"),
                        "db_path": str(Path(tmp) / "memory.sqlite"),
                        "include_memory": False,
                    },
                )
            )
        self.assertFalse(payload["ok"])
        self.assertIn("explicit model", payload["error"])

    def test_system_prompt_does_not_inject_memory_by_default(self):
        with patch.dict(os.environ, {"MECHFERRET_INCLUDE_MEMORY_CONTEXT": ""}):
            prompt = agent.build_system_prompt()

        self.assertNotIn("Previously confirmed mechanisms", prompt)
        self.assertNotIn("find the IOI circuit in gpt2", prompt)
        self.assertIn("novelty_risk", prompt)
        self.assertIn("closest_prior_art", prompt)

    def test_verify_novelty_runs_deep_recent_architecture_search(self):
        from mechferret import tools

        calls = []
        web_calls = []

        def fake_search(query, max_results=20, sort_by="relevance"):
            calls.append({"query": query, "max_results": max_results, "sort_by": sort_by})
            return 99, [
                {
                    "title": "Sparse autoencoder architecture paper",
                    "url": "https://arxiv.org/abs/2501.0001",
                    "published": "2025-01-01T00:00:00Z",
                    "abstract": "Sparse autoencoder architecture for vision language action policies and mechanism discovery.",
                    "authors": ["A. Researcher"],
                }
            ]

        def fake_web_search(query, max_results=12):
            web_calls.append({"query": query, "max_results": max_results})
            return [
                {
                    "title": "Sparse autoencoder architecture implementation for VLA mechanism discovery",
                    "url": "https://example.org/vla-sae",
                    "snippet": "Recent benchmark implementation for sparse autoencoder architecture in vision language action policy mechanisms.",
                },
                {
                    "title": "VLA sparse autoencoder leaderboard",
                    "url": "https://paperswithcode.com/task/vla-sae",
                    "snippet": "Benchmark leaderboard for sparse autoencoder architecture mechanism discovery.",
                },
                {
                    "title": "VLA SAE implementation",
                    "url": "https://github.com/example/vla-sae",
                    "snippet": "Repository with source code for sparse autoencoder probes on action policies.",
                },
            ]

        with (
            patch("mechferret.knowledge.search_arxiv", side_effect=fake_search),
            patch("mechferret.knowledge.web_search", side_effect=fake_web_search),
        ):
            payload = json.loads(
                tools.run_tool(
                    "verify_novelty",
                    {
                        "idea": "sparse autoencoder architecture for vision language action policies",
                        "queries": ["VLA sparse autoencoder mechanisms"],
                    },
                )
            )

        self.assertGreaterEqual(len(calls), 6)
        self.assertTrue(all(call["max_results"] == 20 for call in calls))
        self.assertIn("submittedDate", {call["sort_by"] for call in calls})
        self.assertIn("lastUpdatedDate", {call["sort_by"] for call in calls})
        self.assertTrue(any("architecture" in call["query"].lower() for call in calls))
        self.assertTrue(any("discovery" in call["query"].lower() for call in calls))
        self.assertEqual(len(payload["search_plan"]), len(calls))
        self.assertEqual(payload["arxiv_search_plan"], payload["search_plan"])
        self.assertGreaterEqual(len(web_calls), 3)
        self.assertTrue(all(call["max_results"] == 12 for call in web_calls))
        self.assertIn("web_search_plan", payload)
        self.assertEqual(payload["web_results"][0]["source"], "web")
        self.assertEqual(payload["web_results"][0]["source_domain"], "example.org")
        web_source_types = {row["source_type"] for row in payload["web_results"]}
        self.assertIn("benchmark", web_source_types)
        self.assertIn("code_repository", web_source_types)
        self.assertIn("recent_papers", payload)
        self.assertIn("architecture_papers", payload)
        self.assertEqual(payload["assessment"]["risk"], "high_prior_art_risk")
        self.assertTrue(payload["assessment"]["closest_prior_art"])
        self.assertIn("sparse", payload["assessment"]["closest_prior_art"][0]["matched_terms"])
        self.assertIn("source_type", payload["assessment"]["closest_prior_art"][0])
        self.assertIn("source_domain", payload["assessment"]["closest_prior_art"][0])
        self.assertIn("evidence_excerpt", payload["assessment"]["closest_prior_art"][0])
        self.assertIn("recent_window", payload["assessment"]["coverage"])
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_results"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_results_with_snippets"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_source_types"]["benchmark"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_source_types"]["code_repository"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["retrieved_evidence"], 2)
        self.assertIn("Do not claim high novelty", payload["guidance"])

    def test_verify_novelty_reports_unknown_when_search_fails(self):
        from mechferret import tools

        with (
            patch("mechferret.knowledge.search_arxiv", side_effect=RuntimeError("network unavailable")),
            patch("mechferret.knowledge.web_search", side_effect=RuntimeError("network unavailable")),
        ):
            payload = json.loads(tools.run_tool("verify_novelty", {"idea": "adaptive probe routing for activation patches"}))

        self.assertEqual(payload["assessment"]["risk"], "unknown_search_incomplete")
        self.assertGreater(payload["assessment"]["coverage"]["failed_queries"], 0)
        self.assertGreater(payload["assessment"]["coverage"]["failed_arxiv_queries"], 0)
        self.assertGreater(payload["assessment"]["coverage"]["failed_web_queries"], 0)
        self.assertEqual(payload["related_papers"], [])
        self.assertEqual(payload["web_results"], [])

    def test_tools_validate_boolean_values(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "notes.txt"
            target.write_text("alpha\nalpha\n", encoding="utf-8")
            frames = Path(tmp) / "frames"
            frames.mkdir()

            cases = [
                ("clean_tool_results", {"confirm": "yes"}, "confirm"),
                ("clean_tool_results", {"dry_run": []}, "dry_run"),
                ("edit_file", {"path": str(target), "old_string": "alpha", "replace_all": "yes"}, "replace_all"),
                ("run_research", {"include_memory": "yes"}, "include_memory"),
                ("run_research", {"no_memory": []}, "no_memory"),
                ("run_research", {"allow_seed_corpus": "yes"}, "allow_seed_corpus"),
                ("run_research", {"seed_corpus": []}, "seed_corpus"),
                ("run_discovery", {"allow_mismatch": "yes"}, "allow_mismatch"),
                ("run_discovery", {"include_memory": "yes"}, "include_memory"),
                ("run_discovery", {"no_memory": []}, "no_memory"),
                ("run_discovery", {"allow_seed_corpus": []}, "allow_seed_corpus"),
                ("run_discovery", {"seed_corpus": "yes"}, "seed_corpus"),
                ("list_runs", {"no_audit": "yes"}, "no_audit"),
                ("verify_run", {"repair": []}, "repair"),
                ("write_paper", {"compile": "yes"}, "compile"),
                ("openvla_sae", {"action": "init", "force": "yes", "project_root": str(Path(tmp) / "openvla")}, "force"),
                (
                    "openvla_sae",
                    {"action": "create-manifest", "image_dir": str(frames), "manifest": str(Path(tmp) / "m.jsonl"), "force": []},
                    "force",
                ),
            ]
            for tool_name, args, argument in cases:
                with self.subTest(tool=tool_name, argument=argument):
                    payload = json.loads(tools.run_tool(tool_name, args))
                    self.assertFalse(payload["ok"])
                    self.assertEqual(payload["error"], f"invalid boolean argument: {argument}")
                    self.assertIn(f"{argument}_argument", payload["failed_checks"])
                    self.assertEqual(payload["expected"], "boolean")

    def test_openvla_create_manifest_returns_structured_file_errors(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing"
            payload = json.loads(
                tools.run_tool(
                    "openvla_sae",
                    {"action": "create-manifest", "image_dir": str(missing), "manifest": str(root / "manifest.jsonl")},
                )
            )
            self.assertFalse(payload["ok"])
            self.assertIn("image_dir_exists", payload["failed_checks"])

            frames = root / "frames"
            frames.mkdir()
            manifest = root / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            exists = json.loads(
                tools.run_tool(
                    "openvla_sae",
                    {"action": "create-manifest", "image_dir": str(frames), "manifest": str(manifest)},
                )
            )
            self.assertFalse(exists["ok"])
            self.assertIn("manifest_overwrite", exists["failed_checks"])

    def test_tools_validate_option_values(self):
        from mechferret import tools

        provider_values = ["anthropic", "auto", "local", "openai"]
        cases = [
            ("run_research", {"question": "What should we test?", "provider": "bogus"}, "provider", provider_values),
            ("run_discovery", {"provider": "bogus"}, "provider", provider_values),
            ("run_discovery", {"backend": "quantum"}, "backend", ["auto", "real", "synthetic", "tl", "transformer_lens"]),
            ("write_paper", {"provider": "bogus"}, "provider", provider_values),
            ("review_paper", {"provider": "local"}, "provider", ["anthropic", "auto", "openai"]),
            (
                "arxiv_search",
                {"query": "sparse autoencoder", "sort_by": "popular"},
                "sort_by",
                ["lastUpdatedDate", "relevance", "submittedDate"],
            ),
            (
                "openvla_sae",
                {"action": "launch"},
                "action",
                [
                    "commands",
                    "create-manifest",
                    "dossier",
                    "eval",
                    "features",
                    "init",
                    "plan",
                    "smoke",
                    "status",
                    "validate-manifest",
                ],
            ),
            (
                "openvla_sae",
                {"action": []},
                "action",
                [
                    "commands",
                    "create-manifest",
                    "dossier",
                    "eval",
                    "features",
                    "init",
                    "plan",
                    "smoke",
                    "status",
                    "validate-manifest",
                ],
            ),
        ]
        for tool_name, args, argument, allowed in cases:
            with self.subTest(tool=tool_name, argument=argument):
                payload = json.loads(tools.run_tool(tool_name, args))
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"], f"invalid option argument: {argument}")
                self.assertIn(f"{argument}_argument", payload["failed_checks"])
                self.assertEqual(payload["allowed_values"], allowed)

    def test_review_paper_tool_uses_provider_review(self):
        from mechferret import paper as paper_mod
        from mechferret import tools

        original_provider = paper_mod._paper_provider
        original_call = paper_mod._call_openai
        try:
            paper_mod._paper_provider = lambda provider, model: ("openai", "test-model", "key")
            paper_mod._call_openai = lambda model, key, prompt: "Soundness: 8\nOverall: 8\nRecommendation: Accept"
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tex = root / "paper" / "main.tex"
                tex.parent.mkdir()
                tex.write_text("\\documentclass{article}\\begin{document}Evidence.\\end{document}", encoding="utf-8")
                payload = json.loads(
                    tools.run_tool(
                        "review_paper",
                        {"path": str(tex), "out_dir": str(root / "review"), "provider": "openai"},
                    )
                )
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["model"], "test-model")
                self.assertTrue((root / "review" / "review.md").exists())
                self.assertIn("Recommendation", payload["review"])
        finally:
            paper_mod._paper_provider = original_provider
            paper_mod._call_openai = original_call

    def test_write_paper_tool_passes_compile_timeout(self):
        from mechferret import paper as paper_mod
        from mechferret import tools
        from mechferret.models import ResearchPlan, ResearchRun

        original_compile = paper_mod.compile_tex
        try:
            def fake_compile(tex, *, timeout=60):
                pdf = Path(tex).with_suffix(".pdf")
                pdf.write_bytes(b"%PDF fake")
                return {"pdf": str(pdf), "compiled": True, "stderr": "", "timeout_seen": timeout}

            paper_mod.compile_tex = fake_compile
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = ResearchRun(
                    run_id="run_tool",
                    question="Does a head copy names?",
                    created_at="2026-05-30T00:00:00+00:00",
                    plan=ResearchPlan("Does a head copy names?", [], "ablate"),
                    sources=[],
                    evidence=[],
                    claims=[],
                    contradictions=[],
                    gaps=[],
                    answer="a",
                    metrics={"readiness_score": 0.8},
                )
                run_json = root / "runs" / "demo" / "run.json"
                run_json.parent.mkdir(parents=True)
                run_json.write_text(json.dumps(run.to_dict()), encoding="utf-8")
                payload = json.loads(
                    tools.run_tool(
                        "write_paper",
                        {
                            "path": str(run_json),
                            "provider": "local",
                            "compile": True,
                            "compile_timeout": 7,
                        },
                    )
                )
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["compiled"])
            self.assertEqual(payload["timeout_seen"], 7)
        finally:
            paper_mod.compile_tex = original_compile

    def test_provider_adapters_validate_response_envelopes(self):
        from mechferret import llm as llm_mod
        from mechferret import paper as paper_mod

        original = agent._http_post
        try:
            agent._http_post = lambda url, payload, headers: {"choices": []}
            with self.assertRaisesRegex(RuntimeError, "provider response envelope"):
                paper_mod._call_openai("gpt-test", "key", "prompt")

            agent._http_post = lambda url, payload, headers: {"content": {"bad": "shape"}}
            with self.assertRaisesRegex(RuntimeError, "provider response envelope"):
                llm_mod._call_anthropic("claude-test", "key", "prompt")
        finally:
            agent._http_post = original

    def test_run_discovery_tool_returns_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "r"
            out = agent._run_tool(
                "run_discovery",
                {
                    "skill": "ioi-circuit",
                    "model": "gpt2",
                    "backend": "synthetic",
                    "out_dir": str(run_dir),
                    "db_path": str(Path(tmp) / "memory.sqlite"),
                    "max_rounds": 3,
                    "max_experiments": 400,
                    "max_gpu_seconds": 120,
                    "include_memory": False,
                },
            )
            payload = json.loads(out)
            self.assertGreaterEqual(len(payload["discoveries"]), 1)
            self.assertGreaterEqual(len(payload["experiments"]), 1)
            experiment_ids = {experiment["id"] for experiment in payload["experiments"]}
            for discovery in payload["discoveries"]:
                self.assertIn("id", discovery)
                for experiment_id in discovery["supporting_experiments"]:
                    self.assertIn(experiment_id, experiment_ids)
            self.assertIn("rigor_score", payload["metrics"])
            self.assertEqual(payload["provenance"]["backend_used"], "synthetic")
            self.assertEqual(payload["provenance"]["budget"]["max_rounds"], 3)
            self.assertEqual(payload["provenance"]["budget"]["max_experiments"], 400)
            self.assertEqual(payload["provenance"]["budget"]["max_gpu_seconds"], 120.0)
            self.assertEqual(payload["provenance"]["answer_author"], "experiment_ledger_synthesizer")
            self.assertIn("audit", payload)
            advisory_names = {item["name"] for item in payload["audit"]["advisories"]}
            self.assertIn("synthetic_backend_not_final", advisory_names)
            self.assertIn("local_synthesis_not_final", advisory_names)
            self.assertEqual(Path(payload["artifacts"]["json"]), run_dir / "run.json")
            self.assertTrue((run_dir / "run.json").exists())

    def test_run_research_tool_requires_sources_or_explicit_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked = json.loads(
                agent._run_tool(
                    "run_research",
                    {
                        "question": "What should I investigate?",
                        "provider": "local",
                        "include_memory": False,
                        "out_dir": str(root / "blocked"),
                        "db_path": str(root / "memory.sqlite"),
                    },
                )
            )
            self.assertFalse(blocked["ok"])
            self.assertIn("No source material", blocked["error"])

            missing = json.loads(
                agent._run_tool(
                    "run_research",
                    {
                        "question": "What is in this source?",
                        "source_paths": [str(root / "missing.md")],
                        "provider": "local",
                        "include_memory": False,
                        "out_dir": str(root / "missing"),
                        "db_path": str(root / "memory.sqlite"),
                    },
                )
            )
            self.assertFalse(missing["ok"])
            self.assertIn("Source path not found", missing["error"])

            source = root / "source.md"
            source.write_text(
                "Reliable autoresearch needs planning, retrieval, evidence citations, critic loops, and inspectable traces. "
                "Source diversity and contradiction pressure should be tracked before writing a paper.",
                encoding="utf-8",
            )
            payload = json.loads(
                agent._run_tool(
                    "run_research",
                    {
                        "question": "How should autoresearch stay reliable?",
                        "source_paths": [str(source)],
                        "provider": "local",
                        "include_memory": False,
                        "out_dir": str(root / "run"),
                        "db_path": str(root / "memory.sqlite"),
                    },
                )
            )
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["provenance"]["answer_author"], "local_extractive_synthesizer")
            self.assertFalse(payload["provenance"]["used_packaged_seed_corpus"])
            self.assertTrue(payload["sources"])
            self.assertTrue(payload["evidence"])
            self.assertIn("id", payload["claims"][0])
            evidence_ids = {item["id"] for item in payload["evidence"]}
            source_ids = {item["id"] for item in payload["sources"]}
            for claim in payload["claims"]:
                for citation in claim["citations"]:
                    self.assertIn(citation, evidence_ids)
            for chunk in payload["evidence"]:
                self.assertIn(chunk["source_id"], source_ids)
            self.assertIn("audit", payload)
            self.assertEqual(Path(payload["artifacts"]["json"]), root / "run" / "run.json")
            self.assertTrue((root / "run" / "run.json").exists())

    def test_list_skills_tool(self):
        payload = json.loads(agent._run_tool("list_skills", {}))
        self.assertTrue(any(s["name"] == "ioi-circuit" for s in payload))

    def test_unknown_tool_is_reported(self):
        payload = json.loads(agent._run_tool("nope", {}))
        self.assertFalse(payload["ok"])
        self.assertIn("error", payload)
        self.assertIn("tool_registered", payload["failed_checks"])

    def test_tool_dispatch_exceptions_are_structured(self):
        from mechferret import tools

        original = tools.HANDLERS.get("list_skills")
        try:
            tools.HANDLERS["list_skills"] = lambda _args: (_ for _ in ()).throw(RuntimeError("boom"))
            payload = json.loads(tools.run_tool("list_skills", {}))
        finally:
            if original is not None:
                tools.HANDLERS["list_skills"] = original
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["tool"], "list_skills")
        self.assertIn("RuntimeError: boom", payload["error"])
        self.assertIn("tool_exception", payload["failed_checks"])

    def test_permissions_and_cost(self):
        from mechferret import permissions
        from mechferret.costs import CostTracker

        self.assertEqual(permissions.decide("read_file", {}, read_only=True, permission_class="local", mode="auto").behavior, "allow")
        self.assertEqual(permissions.decide("write_file", {}, read_only=False, permission_class="write", mode="auto").behavior, "allow")
        self.assertEqual(permissions.decide("write_file", {}, read_only=False, permission_class="write", mode="plan").behavior, "ask")
        self.assertEqual(permissions.decide("bash", {"command": "rm -rf /tmp/x"}, read_only=False, permission_class="exec", mode="auto").behavior, "ask")
        c = CostTracker()
        c.add("claude-opus-4-8", {"input_tokens": 1_000_000, "output_tokens": 0})
        self.assertAlmostEqual(c.usd, 15.0, places=3)
        c.add(
            "gpt-5",
            {"input_tokens": "12", "output_tokens": "bad", "prompt_tokens": 999, "cache_read_input_tokens": -5},
        )
        c.add("gpt-5", {"input_tokens": float("inf"), "output_tokens": -7})
        c.add(None, None)
        c.add(123, {"input_tokens": 4, "completion_tokens": 2})
        self.assertEqual(c.input_tokens, 1_000_016)
        self.assertEqual(c.output_tokens, 2)
        self.assertEqual(c.cache_read_tokens, 0)
        self.assertEqual(c.by_model["unknown"]["input"], 4)
        self.assertEqual(c.by_model["unknown"]["output"], 2)

    def test_dispatch_denies_in_plan_mode_without_confirm(self):
        a = agent.Agent()
        a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
        a.permission_mode = "plan"
        out = json.loads(a._dispatch("write_file", {"path": "/tmp/should_not_exist_zzz", "content": "x"}))
        self.assertFalse(out["ok"])
        self.assertTrue(out["denied"])
        self.assertIn("tool_permission", out["failed_checks"])
        self.assertIn("write_file", a.denials)
        self.assertFalse(Path("/tmp/should_not_exist_zzz").exists())

    def test_agent_dispatch_rejects_unknown_tool_before_callback(self):
        fired = []
        a = agent.Agent(on_tool=lambda name, args: fired.append(name))
        a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "x"

        out = json.loads(a._dispatch("not_registered", {}))

        self.assertFalse(out["ok"])
        self.assertEqual(fired, [])
        self.assertIn("tool_registered", out["failed_checks"])

    def test_large_output_persisted(self):
        from mechferret import tools

        big = "x = 1\n" * 5000
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = tools.RESULTS_DIR
            tools.RESULTS_DIR = Path(tmp)
            try:
                out = tools._persist_if_large("bash", big)
            finally:
                tools.RESULTS_DIR = original_dir
            self.assertIn("saved to", out)
            self.assertLess(len(out), len(big))

    def test_large_json_tool_output_stays_parseable(self):
        from mechferret import tools

        payload = {
            "passed": True,
            "path": "/tmp/bundle.zip",
            "checks": [
                {"name": f"check_{index}", "passed": True, "observed": "x" * 100}
                for index in range(500)
            ],
            "next_actions": [],
        }
        raw = json.dumps(payload)
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = tools.RESULTS_DIR
            tools.RESULTS_DIR = Path(tmp)
            try:
                out = tools._persist_if_large("verify_bundle", raw)
            finally:
                tools.RESULTS_DIR = original_dir

            parsed = json.loads(out)
            self.assertTrue(parsed["passed"])
            self.assertTrue(parsed["tool_output_truncated"])
            self.assertEqual(parsed["checks"]["count"], 500)
            self.assertTrue(Path(parsed["full_output_path"]).exists())
            self.assertIn("Read the complete verify_bundle result", parsed["next_actions"][-1])
            self.assertLess(len(out), len(raw))

    def test_large_novelty_output_preserves_assessment_summary(self):
        from mechferret import tools

        payload = {
            "idea": "large novelty result",
            "related_papers": [{"title": f"paper {index}", "abstract": "x" * 800} for index in range(200)],
            "assessment": {
                "risk": "high_prior_art_risk",
                "verdict": "Closest retrieved evidence overlaps.",
                "coverage": {"web_source_types": {"benchmark": 2}, "web_results": 3},
                "closest_prior_art": [
                    {"title": "closest", "source_type": "benchmark", "score": 0.9, "evidence_excerpt": "x" * 300}
                ],
                "required_delta": ["Name the difference."],
            },
        }
        raw = json.dumps(payload)
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = tools.RESULTS_DIR
            tools.RESULTS_DIR = Path(tmp)
            try:
                out = tools._persist_if_large("verify_novelty", raw)
            finally:
                tools.RESULTS_DIR = original_dir

        parsed = json.loads(out)
        self.assertTrue(parsed["tool_output_truncated"])
        self.assertEqual(parsed["assessment"]["risk"], "high_prior_art_risk")
        self.assertEqual(parsed["assessment"]["coverage"]["web_source_types"]["benchmark"], 2)
        self.assertEqual(parsed["assessment"]["closest_prior_art"][0]["source_type"], "benchmark")

    def test_bounded_check_lists_stay_structured_when_compacted(self):
        from mechferret import tools

        checks = [
            {"name": f"check_{index}", "passed": True, "observed": "x" * 200, "threshold": "ok"}
            for index in range(20)
        ]
        compacted = tools._compact_json_value(checks)

        self.assertIsInstance(compacted, list)
        self.assertEqual(compacted[0]["name"], "check_0")
        self.assertTrue(compacted[0]["passed"])

    def test_tool_result_listing_and_cleanup_are_managed(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            original_dir = tools.RESULTS_DIR
            tools.RESULTS_DIR = Path(tmp)
            try:
                old = tools.RESULTS_DIR / "verify_bundle_old.txt"
                new = tools.RESULTS_DIR / "bash_new.txt"
                tools.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                old.write_text(json.dumps({"passed": True}), encoding="utf-8")
                new.write_text("plain output", encoding="utf-8")
                old_time = 1_700_000_000
                new_time = 1_700_100_000
                os.utime(old, (old_time, old_time))
                os.utime(new, (new_time, new_time))

                listed = json.loads(tools.run_tool("list_tool_results", {"limit": 5}))
                self.assertEqual(listed["count"], 2)
                self.assertEqual(listed["shown"], 2)
                self.assertTrue(next(row for row in listed["results"] if row["path"] == str(old))["is_json"])
                listed_actions = " ".join(listed["next_actions"])
                self.assertIn("mechferret tool-results --clean --json", listed_actions)
                self.assertIn("mechferret tool-results --clean --confirm", listed_actions)
                self.assertNotIn("clean_tool_results", listed_actions)
                self.assertNotIn("read_file", listed_actions)

                dry = json.loads(tools.run_tool("clean_tool_results", {"keep_latest": 1, "max_age_days": 0, "dry_run": True}))
                self.assertTrue(dry["dry_run"])
                self.assertTrue(old.exists())
                self.assertTrue(new.exists())
                self.assertEqual({row["path"] for row in dry["would_delete"]}, {str(old), str(new)})

                cleaned = json.loads(tools.run_tool("clean_tool_results", {"keep_latest": 1, "max_age_days": 10_000, "confirm": True}))
                self.assertFalse(cleaned["dry_run"])
                self.assertEqual([row["path"] for row in cleaned["deleted"]], [str(old)])
                self.assertFalse(old.exists())
                self.assertTrue(new.exists())
            finally:
                tools.RESULTS_DIR = original_dir

    def test_anthropic_tool_loop_executes_tool_then_replies(self):
        calls = {"n": 0}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                self.assertIn("tools", payload)
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "Let me list them."},
                        {"type": "tool_use", "id": "t1", "name": "list_skills", "input": {}},
                    ],
                }
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Here are the skills."}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("what skills exist?")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, ["list_skills"])
        self.assertIn("Here are the skills", reply)

    def test_anthropic_tool_loop_handles_mixed_text_content(self):
        calls = {"n": 0}
        streamed = []

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "stop_reason": "end_turn",
                "content": [
                    "Plain text block.",
                    {"type": "text", "text": ["not text"]},
                    {"type": "text", "text": "Structured text block."},
                ],
            }

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.on_text = streamed.append
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("answer directly")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(fired, [])
        self.assertIn("Plain text block.", reply)
        self.assertIn("Structured text block.", reply)
        self.assertEqual(streamed, [reply])

    def test_provider_usage_malformed_values_do_not_block_reply(self):
        calls = {"n": 0}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "usage": {"input_tokens": "bad", "output_tokens": float("inf"), "cache_read_input_tokens": -10},
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "reply despite usage"}],
            }

        original = agent._http_post
        agent._http_post = fake_post
        try:
            a = agent.Agent()
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("answer directly")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(reply, "reply despite usage")
        self.assertEqual(a.cost.input_tokens, 0)
        self.assertEqual(a.cost.output_tokens, 0)
        self.assertEqual(a.cost.cache_read_tokens, 0)

    def test_http_post_reports_invalid_json_body(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"<html>provider outage</html>"

        original = agent.urllib.request.urlopen
        agent.urllib.request.urlopen = lambda request, timeout=120: FakeResponse()
        try:
            with self.assertRaisesRegex(RuntimeError, "provider returned invalid JSON"):
                agent._http_post("https://example.invalid", {"ok": True}, {})
        finally:
            agent.urllib.request.urlopen = original

    def test_anthropic_tool_loop_reports_malformed_response_envelope(self):
        calls = {"n": 0}
        streamed = []

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {"stop_reason": "end_turn", "content": {"bad": "shape"}}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.on_text = streamed.append
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("answer directly")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(fired, [])
        self.assertIn("invalid response envelope", reply)
        self.assertIn("provider status", reply)
        self.assertEqual(streamed, [reply])
        self.assertEqual(a.messages[-1]["role"], "assistant")
        self.assertEqual(a.messages[-1]["content"], reply)

    def test_anthropic_tool_loop_reports_malformed_tool_use(self):
        calls = {"n": 0}
        observed_tool_result = {}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        "not a content block",
                        {"type": "tool_use", "id": "t_bad", "input": {}},
                    ],
                }
            tool_results = payload["messages"][-1]["content"]
            self.assertEqual(len(tool_results), 1)
            self.assertEqual(tool_results[0]["tool_use_id"], "t_bad")
            observed_tool_result.update(json.loads(tool_results[0]["content"]))
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Tool use fixed."}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("use a tool")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, [])
        self.assertFalse(observed_tool_result["ok"])
        self.assertIn("tool_call_envelope", observed_tool_result["failed_checks"])
        self.assertIn("fixed", reply.lower())

    def test_anthropic_tool_loop_reports_non_object_tool_input(self):
        calls = {"n": 0}
        observed_tool_result = {}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "id": "t_args", "name": "read_file", "input": []}],
                }
            tool_results = payload["messages"][-1]["content"]
            self.assertEqual(len(tool_results), 1)
            self.assertEqual(tool_results[0]["tool_use_id"], "t_args")
            observed_tool_result.update(json.loads(tool_results[0]["content"]))
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Arguments fixed."}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("read a file")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, [])
        self.assertFalse(observed_tool_result["ok"])
        self.assertIn("tool_arguments", observed_tool_result["failed_checks"])
        self.assertIn("fixed", reply.lower())

    def test_openai_tool_loop_handles_structured_text_content(self):
        calls = {"n": 0}
        streamed = []

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Structured reply."},
                                {"type": "output_text", "content": "Nested reply."},
                                {"type": "text", "text": ["not text"]},
                            ],
                        }
                    }
                ]
            }

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.on_text = streamed.append
            a.provider, a.model, a._key = "openai", "gpt-test", "fake"
            reply = a.send("answer directly")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(fired, [])
        self.assertIn("Structured reply.", reply)
        self.assertIn("Nested reply.", reply)
        self.assertEqual(streamed, [reply])

    def test_openai_tool_loop_reports_malformed_response_envelope(self):
        calls = {"n": 0}
        streamed = []

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {"choices": []}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.on_text = streamed.append
            a.provider, a.model, a._key = "openai", "gpt-test", "fake"
            reply = a.send("answer directly")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(fired, [])
        self.assertIn("invalid response envelope", reply)
        self.assertIn("response.choices", reply)
        self.assertEqual(streamed, [reply])
        self.assertEqual(a.messages[-1]["role"], "assistant")
        self.assertEqual(a.messages[-1]["content"], reply)

    def test_openai_tool_loop_reports_non_list_tool_calls(self):
        calls = {"n": 0}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I need a tool.",
                            "tool_calls": {"id": "call_1"},
                        }
                    }
                ]
            }

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "openai", "gpt-test", "fake"
            reply = a.send("use a tool")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(fired, [])
        self.assertIn("message.tool_calls", reply)
        self.assertEqual(a.messages[-1]["role"], "assistant")
        self.assertEqual(a.messages[-1]["content"], reply)

    def test_anthropic_tool_loop_rejects_duplicate_tool_ids(self):
        calls = {"n": 0}
        observed_tool_result = {}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "tool_use", "id": "same", "name": "list_skills", "input": {}},
                        {"type": "tool_use", "id": "same", "name": "read_file", "input": {"path": __file__}},
                    ],
                }
            tool_results = payload["messages"][-1]["content"]
            self.assertEqual(len(tool_results), 2)
            self.assertEqual([item["tool_use_id"] for item in tool_results], ["same", "same"])
            observed_tool_result.update(json.loads(tool_results[1]["content"]))
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Duplicate fixed."}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("use two tools")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, ["list_skills"])
        self.assertFalse(observed_tool_result["ok"])
        self.assertIn("tool_call_envelope", observed_tool_result["failed_checks"])
        self.assertIn("duplicate", observed_tool_result["error"])
        self.assertIn("fixed", reply.lower())

    def test_anthropic_tool_loop_reports_step_exhaustion(self):
        calls = {"n": 0}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": f"t{calls['n']}", "name": "list_skills", "input": {}}],
            }

        original_post = agent._http_post
        original_steps = agent.MAX_TOOL_STEPS
        agent._http_post = fake_post
        agent.MAX_TOOL_STEPS = 2
        fired = []
        streamed = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.on_text = streamed.append
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("keep going")
        finally:
            agent._http_post = original_post
            agent.MAX_TOOL_STEPS = original_steps
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, ["list_skills", "list_skills"])
        self.assertIn("Tool loop stopped", reply)
        self.assertIn("Tool loop stopped", streamed[-1])

    def test_openai_tool_loop_reports_malformed_arguments(self):
        calls = {"n": 0}
        observed_tool_result = {}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                self.assertIn("tools", payload)
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": "{"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
            self.assertEqual(len(tool_messages), 1)
            observed_tool_result.update(json.loads(tool_messages[0]["content"]))
            return {"choices": [{"message": {"role": "assistant", "content": "I will retry with valid arguments."}}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "openai", "gpt-test", "fake"
            reply = a.send("read a file")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, [])
        self.assertFalse(observed_tool_result["ok"])
        self.assertIn("tool_arguments", observed_tool_result["failed_checks"])
        self.assertIn("retry", reply.lower())

    def test_openai_tool_loop_reports_malformed_tool_envelope(self):
        calls = {"n": 0}
        observed_tool_result = {}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
            self.assertEqual(len(tool_messages), 1)
            self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
            observed_tool_result.update(json.loads(tool_messages[0]["content"]))
            return {"choices": [{"message": {"role": "assistant", "content": "Tool envelope fixed."}}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "openai", "gpt-test", "fake"
            reply = a.send("use a tool")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, [])
        self.assertFalse(observed_tool_result["ok"])
        self.assertIn("tool_call_envelope", observed_tool_result["failed_checks"])
        self.assertIn("fixed", reply.lower())

    def test_openai_tool_loop_rejects_duplicate_tool_ids(self):
        calls = {"n": 0}
        observed_tool_result = {}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "same",
                                        "type": "function",
                                        "function": {"name": "list_skills", "arguments": "{}"},
                                    },
                                    {
                                        "id": "same",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": json.dumps({"path": __file__})},
                                    },
                                ],
                            }
                        }
                    ]
                }
            tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
            self.assertEqual(len(tool_messages), 2)
            self.assertEqual([message["tool_call_id"] for message in tool_messages], ["same", "same"])
            observed_tool_result.update(json.loads(tool_messages[1]["content"]))
            return {"choices": [{"message": {"role": "assistant", "content": "Duplicate fixed."}}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "openai", "gpt-test", "fake"
            reply = a.send("use two tools")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, ["list_skills"])
        self.assertFalse(observed_tool_result["ok"])
        self.assertIn("tool_call_envelope", observed_tool_result["failed_checks"])
        self.assertIn("duplicate", observed_tool_result["error"])
        self.assertIn("fixed", reply.lower())

    def test_openai_tool_loop_reports_step_exhaustion(self):
        calls = {"n": 0}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": f"call_{calls['n']}",
                                    "type": "function",
                                    "function": {"name": "list_skills", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }

        original_post = agent._http_post
        original_steps = agent.MAX_TOOL_STEPS
        agent._http_post = fake_post
        agent.MAX_TOOL_STEPS = 2
        fired = []
        streamed = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.on_text = streamed.append
            a.provider, a.model, a._key = "openai", "gpt-test", "fake"
            reply = a.send("keep going")
        finally:
            agent._http_post = original_post
            agent.MAX_TOOL_STEPS = original_steps
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, ["list_skills", "list_skills"])
        self.assertIn("Tool loop stopped", reply)
        self.assertIn("Tool loop stopped", streamed[-1])

    def test_active_provider_empty_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            old = os.environ.get("MECHFERRET_CONFIG")
            old_keys = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
            os.environ["MECHFERRET_CONFIG"] = str(cfg)
            try:
                provider, model, key = agent.active_provider()
                self.assertEqual(provider, "")
                self.assertFalse(agent.is_configured())
            finally:
                if old is None:
                    os.environ.pop("MECHFERRET_CONFIG", None)
                else:
                    os.environ["MECHFERRET_CONFIG"] = old
                for k, v in old_keys.items():
                    if v is not None:
                        os.environ[k] = v

    def test_active_provider_requires_explicit_chat_model(self):
        from mechferret.config import MechFerretConfig, ProviderSettings, save_config

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            old = os.environ.get("MECHFERRET_CONFIG")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            old_openai_model = os.environ.get("MECHFERRET_OPENAI_MODEL")
            os.environ["MECHFERRET_CONFIG"] = str(cfg)
            os.environ["OPENAI_API_KEY"] = "env-key"
            os.environ.pop("MECHFERRET_OPENAI_MODEL", None)
            try:
                save_config(
                    MechFerretConfig(
                        default_provider="openai",
                        providers={"openai": ProviderSettings(api_key="", model="")},
                    ),
                    cfg,
                )
                self.assertEqual(agent.active_provider(), ("", "", ""))
                os.environ["MECHFERRET_OPENAI_MODEL"] = "env-model"
                self.assertEqual(agent.active_provider(), ("openai", "env-model", "env-key"))
            finally:
                if old is None:
                    os.environ.pop("MECHFERRET_CONFIG", None)
                else:
                    os.environ["MECHFERRET_CONFIG"] = old
                if old_openai_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                if old_openai_model is None:
                    os.environ.pop("MECHFERRET_OPENAI_MODEL", None)
                else:
                    os.environ["MECHFERRET_OPENAI_MODEL"] = old_openai_model

    def test_config_loading_tolerates_malformed_files(self):
        from mechferret.config import (
            MechFerretConfig,
            ProviderSettings,
            configure_provider,
            configured_api_key,
            configured_model,
            load_config,
            save_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.json"
            bad.write_text("{", encoding="utf-8")
            self.assertEqual(load_config(bad).default_provider, "local")

            malformed = root / "malformed.json"
            malformed.write_text(
                json.dumps(
                    {
                        "default_provider": "unknown",
                        "providers": {
                            "openai": {"api_key": ["bad"], "model": "  gpt-test  ", "extra": "ignored"},
                            "anthropic": "not settings",
                            "other": {"api_key": "x"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_config(malformed)
            self.assertEqual(cfg.default_provider, "local")
            self.assertEqual(cfg.providers["openai"].api_key, "")
            self.assertEqual(cfg.providers["openai"].model, "gpt-test")
            self.assertEqual(cfg.providers["anthropic"].api_key, "")
            self.assertNotIn("other", cfg.providers)
            self.assertEqual(configured_api_key("local", cfg), "")
            self.assertEqual(configured_model("local", cfg), "local")
            self.assertEqual(configured_model("openai", cfg, "  override-model  "), "override-model")
            with patch.dict(os.environ, {"MECHFERRET_OPENAI_MODEL": "env-chat-model"}):
                self.assertEqual(configured_model("openai", MechFerretConfig()), "env-chat-model")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(configured_model("openai", MechFerretConfig()), "")

            saved = save_config(
                MechFerretConfig(
                    default_provider="unknown",
                    providers={"openai": ProviderSettings(api_key="key", model="model"), "other": ProviderSettings(api_key="x")},
                ),
                root / "saved.json",
            )
            payload = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(payload["default_provider"], "local")
            self.assertEqual(set(payload["providers"]), {"openai"})

            configured = configure_provider("openai", "  key  ", model="  gpt-trim  ", path=root / "configured.json")
            loaded = load_config(configured)
            self.assertEqual(loaded.providers["openai"].api_key, "key")
            self.assertEqual(loaded.providers["openai"].model, "gpt-trim")


if __name__ == "__main__":
    unittest.main()
