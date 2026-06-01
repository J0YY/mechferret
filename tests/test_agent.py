import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from mechferret import agent


def _recent_window_label() -> str:
    year = datetime.now(UTC).year
    return f"{year - 2}-{year}"


def _option_threat_model():
    return [
        {
            "threat": "exact_phrase_overlap",
            "searched": True,
            "risk": "searched_no_strong_overlap",
            "evidence_count": 0,
            "strongest_score": 0.0,
            "representative_prior": {},
            "failure_mode": "A prior uses the same phrase.",
            "next_action": "Record why exact phrase search found no strong overlap.",
        },
        {
            "threat": "claim_collision",
            "searched": True,
            "risk": "needs_delta_review",
            "evidence_count": 1,
            "strongest_score": 0.34,
            "representative_prior": {"title": "Closest Paper", "url": "https://arxiv.org/abs/2501.0001", "source_type": "paper"},
            "failure_mode": "A prior claims the same core contribution.",
            "next_action": "Write the exact delta from the closest prior.",
        },
    ]


def _option_disqualifying_tests():
    return [
        {
            "test": "exact_phrase_overlap",
            "passed": True,
            "risk": "searched_no_strong_overlap",
            "representative_prior": {},
            "required_evidence": "Nearest exact-phrase prior and why its claim is materially different.",
        },
        {
            "test": "claim_collision",
            "passed": False,
            "risk": "needs_delta_review",
            "representative_prior": {"title": "Closest Paper", "url": "https://arxiv.org/abs/2501.0001", "source_type": "paper"},
            "required_evidence": "Closest paper/project claiming the same contribution and the specific delta.",
        },
    ]


def _option_search_audit():
    arxiv_focuses = [
        "core_relevance",
        "recent_submitted",
        "recent_updated",
        "method_relevance",
        "mechanism_evidence",
        "recent_evaluation",
        "recent_discovery",
        "architecture_variant",
        "replication_failure_modes",
        "evaluation_protocol",
    ]
    web_focuses = [
        "web_recent_method",
        "web_benchmark_evaluation",
        "web_code_prior",
        "web_exact_phrase",
        "web_claim_collision",
        "web_peer_review",
        "web_architecture_variant",
        "web_replication_results",
    ]
    focus_summary = [
        {
            "source": "arxiv",
            "focus": focus,
            "passes": 1,
            "failed_passes": 0,
            "retrieved": 50,
            "unique_added": 4 if i == 0 else 0,
            "requested_results_max": 50,
        }
        for i, focus in enumerate(arxiv_focuses)
    ]
    focus_summary.extend(
        {
            "source": "web",
            "focus": focus,
            "passes": 1,
            "failed_passes": 0,
            "retrieved": 24,
            "unique_added": 3 if i == 0 else 0,
            "requested_results_max": 24,
        }
        for i, focus in enumerate(web_focuses)
    )
    return {
        "pass_count": 18,
        "failed_passes": 0,
        "empty_search_passes": 2,
        "empty_arxiv_passes": 1,
        "empty_web_passes": 1,
        "duplicate_only_search_passes": 3,
        "focus_coverage": {
            "recency": True,
            "recent_discovery": True,
            "architecture": True,
            "method": True,
            "mechanism": True,
            "evaluation": True,
            "implementation": True,
            "replication": True,
            "failure_modes": True,
            "protocol": True,
            "exact_phrase": True,
            "claim_collision": True,
            "peer_review": True,
        },
        "missing_focus_coverage": [],
        "empty_focuses": [
            {"source": "arxiv", "focus": "peer_review_critique"},
            {"source": "web", "focus": "web_peer_review"},
        ],
        "failed_focuses": [],
        "focus_summary": focus_summary,
    }


def _validated_option(title: str = "Novelty audit") -> dict:
    return {
        "title": title,
        "summary": "Check the delta",
        "detail": "Use retrieved evidence before choosing.",
        "citations": ["Closest Paper https://arxiv.org/abs/2501.0001"],
        "novelty_risk": "medium_prior_art_risk",
        "novelty_verdict": "Related work exists.",
        "closest_prior_art": ["Closest Paper https://arxiv.org/abs/2501.0001"],
        "claim_readiness": {
            "status": "delta_review_required",
            "can_claim_high_novelty": False,
            "missing_checks": [],
            "next_actions": ["Write the delta."],
        },
        "comparison_matrix": [
            {"axis": "method", "covered": True, "evidence_count": 1, "next_action": "Compare method."},
            {"axis": "evaluation", "covered": False, "evidence_count": 0, "next_action": "Add benchmark."},
        ],
        "novelty_threat_model": _option_threat_model(),
        "disqualifying_overlap_tests": _option_disqualifying_tests(),
        "search_audit": _option_search_audit(),
        "recent_pressure": {
            "status": "recent_prior_present",
            "recent_window": _recent_window_label(),
            "recent_evidence_count": 1,
            "latest_year": datetime.now(UTC).year,
            "recent_prior_titles": ["Closest Paper"],
        },
        "required_delta": ["Show a causal ablation that differs from prior work."],
    }


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
        option_list_schema = next(tool for tool in tools.TOOL_SPECS if tool["name"] == "present_options")["parameters"]["properties"]["options"]
        self.assertEqual(option_list_schema["minItems"], 2)
        self.assertEqual(option_list_schema["maxItems"], 5)
        option_schema = option_list_schema["items"]
        self.assertIn("novelty_threat_model", option_schema["required"])
        self.assertIn("disqualifying_overlap_tests", option_schema["required"])
        self.assertIn("search_audit", option_schema["required"])

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
                            "claim_readiness": {
                                "status": "not_ready_needs_more_evidence",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["retrieved_prior_art"],
                                "next_actions": ["Collect more independent evidence."],
                            },
                            "required_delta": "Show a measurable delta.",
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_novelty["ok"])
        self.assertEqual(bad_novelty["expected"], "objects with novelty_risk from verify_novelty assessment")

        bad_readiness = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "missing readiness evidence",
                            "detail": "A direction without claim readiness should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "required_delta": "Show a measurable delta.",
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_readiness["ok"])
        self.assertEqual(bad_readiness["expected"], "objects with claim_readiness from verify_novelty assessment")

        bad_comparison = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "missing comparison matrix",
                            "detail": "A direction without per-axis novelty evidence should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "claim_readiness": {
                                "status": "not_ready_needs_more_evidence",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["focus_breadth"],
                                "next_actions": ["Run follow-up searches."],
                            },
                            "recent_pressure": {
                                "status": "recent_prior_present",
                                "recent_window": "2024-2026",
                                "recent_evidence_count": 1,
                            },
                            "required_delta": ["Show a measurable delta."],
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_comparison["ok"])
        self.assertEqual(bad_comparison["expected"], "objects with comparison_matrix from verify_novelty assessment")

        bad_threat_model = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "missing novelty threat model",
                            "detail": "A direction without threat-model novelty evidence should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "claim_readiness": {
                                "status": "not_ready_needs_more_evidence",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["focus_breadth"],
                                "next_actions": ["Run follow-up searches."],
                            },
                            "comparison_matrix": [
                                {"axis": "method", "covered": True, "evidence_count": 1, "next_action": "Compare method."},
                                {"axis": "evaluation", "covered": False, "evidence_count": 0, "next_action": "Add benchmark."},
                            ],
                            "recent_pressure": {
                                "status": "recent_prior_present",
                                "recent_window": "2024-2026",
                                "recent_evidence_count": 1,
                            },
                            "required_delta": ["Show a measurable delta."],
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_threat_model["ok"])
        self.assertEqual(bad_threat_model["expected"], "objects with novelty_threat_model from verify_novelty assessment")

        bad_disqualifying_tests = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "missing disqualifying overlap tests",
                            "detail": "A direction without disqualifying overlap tests should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "claim_readiness": {
                                "status": "not_ready_needs_more_evidence",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["focus_breadth"],
                                "next_actions": ["Run follow-up searches."],
                            },
                            "comparison_matrix": [
                                {"axis": "method", "covered": True, "evidence_count": 1, "next_action": "Compare method."},
                                {"axis": "evaluation", "covered": False, "evidence_count": 0, "next_action": "Add benchmark."},
                            ],
                            "novelty_threat_model": _option_threat_model(),
                            "recent_pressure": {
                                "status": "recent_prior_present",
                                "recent_window": "2024-2026",
                                "recent_evidence_count": 1,
                            },
                            "required_delta": ["Show a measurable delta."],
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_disqualifying_tests["ok"])
        self.assertEqual(
            bad_disqualifying_tests["expected"],
            "objects with disqualifying_overlap_tests from verify_novelty assessment",
        )

        bad_search_audit = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "missing search audit",
                            "detail": "A direction without per-query search audit evidence should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "claim_readiness": {
                                "status": "not_ready_needs_more_evidence",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["focus_breadth"],
                                "next_actions": ["Run follow-up searches."],
                            },
                            "comparison_matrix": [
                                {"axis": "method", "covered": True, "evidence_count": 1, "next_action": "Compare method."},
                                {"axis": "evaluation", "covered": False, "evidence_count": 0, "next_action": "Add benchmark."},
                            ],
                            "novelty_threat_model": _option_threat_model(),
                            "disqualifying_overlap_tests": _option_disqualifying_tests(),
                            "recent_pressure": {
                                "status": "recent_prior_present",
                                "recent_window": "2024-2026",
                                "recent_evidence_count": 1,
                            },
                            "required_delta": ["Show a measurable delta."],
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_search_audit["ok"])
        self.assertEqual(bad_search_audit["expected"], "objects with search_audit from verify_novelty assessment")

        shallow_search_audit = _option_search_audit()
        shallow_search_audit["pass_count"] = 4
        shallow_search_audit["focus_summary"] = shallow_search_audit["focus_summary"][:4]
        bad_shallow_search = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "shallow search audit",
                            "detail": "A direction with too few search passes should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "claim_readiness": {
                                "status": "not_ready_needs_more_evidence",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["deep_query_plan"],
                                "next_actions": ["Run follow-up searches."],
                            },
                            "comparison_matrix": [
                                {"axis": "method", "covered": True, "evidence_count": 1, "next_action": "Compare method."},
                                {"axis": "evaluation", "covered": False, "evidence_count": 0, "next_action": "Add benchmark."},
                            ],
                            "novelty_threat_model": _option_threat_model(),
                            "disqualifying_overlap_tests": _option_disqualifying_tests(),
                            "search_audit": shallow_search_audit,
                            "recent_pressure": {
                                "status": "recent_prior_present",
                                "recent_window": "2024-2026",
                                "recent_evidence_count": 1,
                            },
                            "required_delta": ["Show a measurable delta."],
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_shallow_search["ok"])
        self.assertEqual(bad_shallow_search["expected"], "objects with search_audit from verify_novelty assessment")

        unfocused_search_audit = _option_search_audit()
        for index, row in enumerate(unfocused_search_audit["focus_summary"]):
            row["focus"] = f"generic_focus_{index}"
        bad_unfocused_search = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "unfocused search audit",
                            "detail": "A direction with enough passes but no deep focus coverage should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "medium_prior_art_risk",
                            "novelty_verdict": "Related work exists.",
                            "closest_prior_art": [],
                            "claim_readiness": {
                                "status": "not_ready_needs_more_evidence",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["focus_breadth"],
                                "next_actions": ["Run focused follow-up searches."],
                            },
                            "comparison_matrix": [
                                {"axis": "method", "covered": True, "evidence_count": 1, "next_action": "Compare method."},
                                {"axis": "evaluation", "covered": False, "evidence_count": 0, "next_action": "Add benchmark."},
                            ],
                            "novelty_threat_model": _option_threat_model(),
                            "disqualifying_overlap_tests": _option_disqualifying_tests(),
                            "search_audit": unfocused_search_audit,
                            "recent_pressure": {
                                "status": "recent_prior_present",
                                "recent_window": "2024-2026",
                                "recent_evidence_count": 1,
                            },
                            "required_delta": ["Show a measurable delta."],
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_unfocused_search["ok"])
        self.assertEqual(bad_unfocused_search["expected"], "objects with search_audit from verify_novelty assessment")

        failed_search_audit = _option_search_audit()
        failed_search_audit["failed_passes"] = 1
        failed_search_audit["failed_focuses"] = [{"source": "web", "focus": "web_claim_collision"}]
        bad_failed_search = json.loads(
            tools.run_tool(
                "present_options",
                {
                    "options": [
                        {
                            "title": "Thin option",
                            "summary": "failed search audit",
                            "detail": "A direction with failed retrieval passes should be rejected.",
                            "citations": ["https://arxiv.org/abs/2501.0001"],
                            "novelty_risk": "unknown_search_incomplete",
                            "novelty_verdict": "Search incomplete.",
                            "closest_prior_art": [],
                            "claim_readiness": {
                                "status": "not_ready_search_incomplete",
                                "can_claim_high_novelty": False,
                                "missing_checks": ["search_completed"],
                                "next_actions": ["Retry failed retrieval passes."],
                            },
                            "comparison_matrix": [
                                {"axis": "method", "covered": True, "evidence_count": 1, "next_action": "Compare method."},
                                {"axis": "evaluation", "covered": False, "evidence_count": 0, "next_action": "Add benchmark."},
                            ],
                            "novelty_threat_model": _option_threat_model(),
                            "disqualifying_overlap_tests": _option_disqualifying_tests(),
                            "search_audit": failed_search_audit,
                            "recent_pressure": {
                                "status": "recent_prior_present",
                                "recent_window": "2024-2026",
                                "recent_evidence_count": 1,
                            },
                            "required_delta": ["Show a measurable delta."],
                        }
                    ]
                },
            )
        )
        self.assertFalse(bad_failed_search["ok"])
        self.assertEqual(bad_failed_search["expected"], "objects with search_audit from verify_novelty assessment")

        no_web_unique_search_audit = _option_search_audit()
        for row in no_web_unique_search_audit["focus_summary"]:
            if row["source"] == "web":
                row["unique_added"] = 0
        no_web_unique = _validated_option("No web evidence")
        no_web_unique["search_audit"] = no_web_unique_search_audit
        bad_no_web_unique = json.loads(
            tools.run_tool(
                "present_options",
                {"options": [no_web_unique]},
            )
        )
        self.assertFalse(bad_no_web_unique["ok"])
        self.assertEqual(bad_no_web_unique["expected"], "objects with search_audit from verify_novelty assessment")

        missing_recent = _validated_option("No recent evidence")
        missing_recent["recent_pressure"] = {
            "status": "missing_recent_prior_art",
            "recent_window": _recent_window_label(),
            "recent_evidence_count": 0,
            "latest_year": 0,
            "recent_prior_titles": [],
        }
        bad_missing_recent = json.loads(
            tools.run_tool(
                "present_options",
                {"options": [missing_recent]},
            )
        )
        self.assertFalse(bad_missing_recent["ok"])
        self.assertEqual(bad_missing_recent["expected"], "objects with recent_pressure from verify_novelty assessment")

        stale_recent = _validated_option("Stale evidence")
        stale_recent["recent_pressure"] = {
            "status": "recent_prior_present",
            "recent_window": "2020-2022",
            "recent_evidence_count": 1,
            "latest_year": 2022,
            "recent_prior_titles": ["Old Paper"],
        }
        bad_stale_recent = json.loads(
            tools.run_tool(
                "present_options",
                {"options": [stale_recent]},
            )
        )
        self.assertFalse(bad_stale_recent["ok"])
        self.assertEqual(bad_stale_recent["expected"], "objects with recent_pressure from verify_novelty assessment")

        too_few_options = json.loads(
            tools.run_tool(
                "present_options",
                {"options": [_validated_option("Only direction")]},
            )
        )
        self.assertFalse(too_few_options["ok"])
        self.assertEqual(too_few_options["expected"], "2-5 validated research direction objects")

        ok = json.loads(
            tools.run_tool(
                "present_options",
                {"options": [_validated_option("Run audit"), _validated_option("Run audit 2")]},
            )
        )
        self.assertEqual(ok["options"], ["Run audit", "Run audit 2"])
        self.assertEqual(ok["option_details"][0]["novelty_risk"], "medium_prior_art_risk")
        self.assertIn("Closest Paper", ok["option_details"][0]["citations"][0])
        self.assertIn("Closest Paper", ok["option_details"][0]["closest_prior_art"][0])
        self.assertEqual(ok["option_details"][0]["claim_readiness"]["status"], "delta_review_required")
        self.assertFalse(ok["option_details"][0]["claim_readiness"]["can_claim_high_novelty"])
        self.assertIn("causal ablation", ok["option_details"][0]["required_delta"])
        self.assertEqual(ok["option_details"][0]["comparison_matrix"][0]["axis"], "method")
        self.assertFalse(ok["option_details"][0]["comparison_matrix"][1]["covered"])
        self.assertEqual(ok["option_details"][0]["novelty_threat_model"][0]["threat"], "exact_phrase_overlap")
        self.assertEqual(ok["option_details"][0]["novelty_threat_model"][1]["representative_prior"]["source_type"], "paper")
        self.assertEqual(ok["option_details"][0]["disqualifying_overlap_tests"][1]["test"], "claim_collision")
        self.assertFalse(ok["option_details"][0]["disqualifying_overlap_tests"][1]["passed"])
        self.assertEqual(ok["option_details"][0]["search_audit"]["pass_count"], 18)
        self.assertEqual(ok["option_details"][0]["search_audit"]["empty_search_passes"], 2)
        self.assertEqual(ok["option_details"][0]["search_audit"]["focus_summary"][0]["requested_results_max"], 50)
        self.assertTrue(ok["option_details"][0]["search_audit"]["focus_coverage"]["claim_collision"])
        self.assertEqual(ok["option_details"][0]["recent_pressure"]["status"], "recent_prior_present")
        self.assertEqual(ok["option_details"][0]["recent_pressure"]["latest_year"], datetime.now(UTC).year)

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
        self.assertIn("model_required", payload["failed_checks"])
        self.assertTrue(any("model" in action.lower() for action in payload["next_actions"]))

    def test_run_discovery_requires_explicit_task_for_vague_prompt(self):
        from mechferret import tools

        with tempfile.TemporaryDirectory() as tmp:
            payload = json.loads(
                tools.run_tool(
                    "run_discovery",
                    {
                        "question": "Investigate an interesting model behavior",
                        "model": "gpt2",
                        "backend": "synthetic",
                        "out_dir": str(Path(tmp) / "run"),
                        "db_path": str(Path(tmp) / "memory.sqlite"),
                        "include_memory": False,
                    },
                )
            )
        self.assertFalse(payload["ok"])
        self.assertIn("could not infer", payload["error"])
        self.assertIn("task_required", payload["failed_checks"])
        self.assertNotIn("tool_exception", payload["failed_checks"])
        self.assertTrue(any("task" in action.lower() for action in payload["next_actions"]))

    def test_system_prompt_does_not_inject_memory_by_default(self):
        with patch.dict(os.environ, {"MECHFERRET_INCLUDE_MEMORY_CONTEXT": ""}):
            prompt = agent.build_system_prompt()

        self.assertNotIn("Previously confirmed mechanisms", prompt)
        self.assertNotIn("find the IOI circuit in gpt2", prompt)
        self.assertNotIn("press " + "enter", prompt.lower())
        self.assertIn("ask one targeted clarifying question", prompt)
        self.assertIn("Never fill that gap with benchmark-specific models", prompt)
        self.assertNotIn("GPT-2", agent.BASE_SYSTEM_PROMPT)
        self.assertNotIn("IOI", agent.BASE_SYSTEM_PROMPT)
        self.assertIn("novelty_risk", prompt)
        self.assertIn("closest_prior_art", prompt)
        self.assertIn("claim_readiness", prompt)
        self.assertIn("comparison_matrix", prompt)
        self.assertIn("novelty_threat_model", prompt)
        self.assertIn("disqualifying_overlap_tests", prompt)
        self.assertIn("search_audit", prompt)
        self.assertIn("recent_pressure", prompt)

    def test_assistant_text_sanitizes_stale_benchmark_scaffolds(self):
        stale = (
            "The minimal experiment should be:\n"
            "1. Target behavior: duplicate-token/name-mover behavior in GPT-2 small.\n"
            "2. Start modules: heads 5.0, 5.2, 5.6, 5.11, 4.8, 6.8, 4.11, 4.3.\n"
            "Next: " + "press " + "enter to proceed."
        )
        sanitized = agent._sanitize_assistant_text("Proceed with the next step.", stale)

        self.assertIn("which model and behavior/task", sanitized)
        self.assertNotIn("GPT-2", sanitized)
        self.assertNotIn("5.0", sanitized)
        self.assertNotIn("press " + "enter", sanitized.lower())

    def test_assistant_text_sanitizes_unrequested_benchmark_model_only(self):
        text = "I will use GPT-2 small as the model and run a compact probe first."
        sanitized = agent._sanitize_assistant_text("Proceed with the next step.", text)

        self.assertIn("which model and behavior/task", sanitized)
        self.assertNotIn("GPT-2", sanitized)

    def test_assistant_text_sanitizes_model_default_even_when_task_named(self):
        text = "For IOI, use GPT-2 small and start with the standard head screen."
        sanitized = agent._sanitize_assistant_text("Study IOI.", text)

        self.assertIn("which model and behavior/task", sanitized)
        self.assertNotIn("GPT-2", sanitized)

    def test_assistant_text_sanitizes_known_heads_without_model(self):
        text = "Start modules: heads 5.0, 5.2, 5.6, 5.11, then validate by patching."
        sanitized = agent._sanitize_assistant_text("Study duplicate-token behavior.", text)

        self.assertIn("which model and behavior/task", sanitized)
        self.assertNotIn("5.0", sanitized)

    def test_assistant_text_allows_explicit_benchmark_requests(self):
        text = "Run duplicate-token/name-mover tests in GPT-2 small."

        self.assertEqual(agent._sanitize_assistant_text("Use GPT-2 small for IOI.", text), text)

    def test_assistant_text_does_not_treat_benchmark_complaint_as_selection(self):
        text = "Run duplicate-token/name-mover tests in GPT-2 small."
        sanitized = agent._sanitize_assistant_text("Why are we still seeing gpt2 by default?", text)

        self.assertIn("which model and behavior/task", sanitized)
        self.assertNotIn("GPT-2", sanitized)

    def test_assistant_text_does_not_treat_benchmark_negation_as_selection(self):
        text = "I will use GPT-2 small as the model and run a compact probe first."
        sanitized = agent._sanitize_assistant_text("Do not use gpt2; choose no default model.", text)

        self.assertIn("which model and behavior/task", sanitized)
        self.assertNotIn("GPT-2", sanitized)

    def test_assistant_text_allows_space_separated_explicit_benchmark_request(self):
        text = "Run duplicate-token/name-mover tests in gpt2 small."

        self.assertEqual(agent._sanitize_assistant_text("Please run gpt2 small on IOI.", text), text)

    def test_openai_stale_benchmark_text_blocks_same_turn_tool_call(self):
        calls = {"n": 0}
        streamed = []

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "The minimal experiment should be:\n"
                                "1. Target behavior: duplicate-token/name-mover behavior in GPT-2 small.\n"
                                "2. Start modules: heads 5.0, 5.2, 5.6, 5.11, 4.8, 6.8, 4.11, 4.3.\n"
                                "Next: " + "press " + "enter to proceed."
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "list_skills", "arguments": "{}"},
                                }
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
            reply = a.send("Proceed with the next step.")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(fired, [])
        self.assertEqual(streamed, [reply])
        self.assertIn("which model and behavior/task", reply)
        self.assertNotIn("GPT-2", reply)
        self.assertNotIn("tool_calls", a.messages[-1])

    def test_anthropic_stale_benchmark_text_blocks_same_turn_tool_call(self):
        calls = {"n": 0}
        streamed = []

        def fake_post(url, payload, headers):
            calls["n"] += 1
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The minimal experiment should be:\n"
                            "1. Target behavior: duplicate-token/name-mover behavior in GPT-2 small.\n"
                            "2. Start modules: heads 5.0, 5.2, 5.6, 5.11, 4.8, 6.8, 4.11, 4.3.\n"
                            "Next: " + "press " + "enter to proceed."
                        ),
                    },
                    {"type": "tool_use", "id": "t1", "name": "list_skills", "input": {}},
                ],
            }

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.on_text = streamed.append
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("Proceed with the next step.")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(fired, [])
        self.assertEqual(streamed, [reply])
        self.assertIn("which model and behavior/task", reply)
        self.assertNotIn("GPT-2", reply)
        self.assertEqual(a.messages[-1]["content"], [{"type": "text", "text": reply}])

    def test_retrieval_tools_floor_shallow_result_counts(self):
        from mechferret import tools

        arxiv_calls = []
        web_calls = []

        def fake_arxiv(query, max_results=20, sort_by="relevance"):
            arxiv_calls.append({"query": query, "max_results": max_results, "sort_by": sort_by})
            return 0, []

        def fake_web(query, max_results=12):
            web_calls.append({"query": query, "max_results": max_results})
            return []

        with (
            patch("mechferret.knowledge.search_arxiv", side_effect=fake_arxiv),
            patch("mechferret.knowledge.web_search", side_effect=fake_web),
        ):
            shallow_count = 2 + 3
            tools.run_tool("arxiv_search", {"query": "sparse autoencoder", "max_results": shallow_count})
            tools.run_tool("web_search", {"query": "sparse autoencoder", "max_results": shallow_count})

        self.assertEqual(arxiv_calls[0]["max_results"], 50)
        self.assertEqual(web_calls[0]["max_results"], 24)

    def test_verify_novelty_runs_deep_recent_method_search(self):
        from mechferret import tools

        calls = []
        web_calls = []
        fetch_calls = []

        def fake_search(query, max_results=20, sort_by="relevance"):
            calls.append({"query": query, "max_results": max_results, "sort_by": sort_by})
            return 99, [
                {
                    "title": "Sparse autoencoder method paper",
                    "url": "https://arxiv.org/abs/2501.0001",
                    "published": "2025-01-01T00:00:00Z",
                    "abstract": "Sparse autoencoder method for vision language action policies and mechanism discovery.",
                    "authors": ["A. Researcher"],
                }
            ]

        def fake_web_search(query, max_results=12):
            web_calls.append({"query": query, "max_results": max_results})
            return [
                {
                    "title": "Sparse autoencoder method implementation for VLA mechanism discovery",
                    "url": "https://example.org/vla-sae",
                    "snippet": "Recent benchmark implementation for sparse autoencoder method in vision language action policy mechanisms.",
                },
                {
                    "title": "VLA sparse autoencoder leaderboard",
                    "url": "https://paperswithcode.com/task/vla-sae",
                    "snippet": "Benchmark leaderboard for sparse autoencoder method mechanism discovery.",
                },
                {
                    "title": "VLA SAE implementation",
                    "url": "https://github.com/example/vla-sae",
                    "snippet": "Repository with source code for sparse autoencoder probes on action policies.",
                },
            ]

        def fake_web_fetch(url, max_chars=2400):
            fetch_calls.append({"url": url, "max_chars": max_chars})
            return (
                "Fetched page text describing sparse autoencoder method for vision language action policies, "
                "mechanism discovery, benchmark evaluation, and implementation details."
            )

        with (
            patch("mechferret.knowledge.search_arxiv", side_effect=fake_search),
            patch("mechferret.knowledge.web_search", side_effect=fake_web_search),
            patch("mechferret.knowledge.web_fetch", side_effect=fake_web_fetch),
        ):
            payload = json.loads(
                tools.run_tool(
                    "verify_novelty",
                    {
                        "idea": "sparse autoencoder method for vision language action policies",
                        "queries": ["VLA sparse autoencoder mechanisms"],
                    },
                )
            )
        if payload.get("tool_output_truncated"):
            payload = json.loads(Path(payload["full_output_path"]).read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(calls), 10)
        self.assertTrue(all(call["max_results"] == 50 for call in calls))
        self.assertIn("submittedDate", {call["sort_by"] for call in calls})
        self.assertIn("lastUpdatedDate", {call["sort_by"] for call in calls})
        self.assertTrue(any("method" in call["query"].lower() for call in calls))
        self.assertTrue(any("mechanism" in call["query"].lower() for call in calls))
        self.assertTrue(any("evaluation" in call["query"].lower() for call in calls))
        self.assertTrue(any("recent discovery" in call["query"].lower() for call in calls))
        self.assertTrue(any("architecture variant" in call["query"].lower() for call in calls))
        self.assertTrue(any("replication" in call["query"].lower() for call in calls))
        self.assertTrue(any("same contribution" in call["query"].lower() for call in calls))
        self.assertTrue(any("independent replication critique" in call["query"].lower() for call in calls))
        self.assertEqual(len(payload["search_plan"]), len(calls))
        self.assertEqual(payload["arxiv_search_plan"], payload["search_plan"])
        self.assertGreaterEqual(len(web_calls), 8)
        self.assertTrue(all(call["max_results"] == 24 for call in web_calls))
        self.assertTrue(any('"we propose"' in call["query"].lower() for call in web_calls))
        self.assertTrue(any("openreview review rebuttal" in call["query"].lower() for call in web_calls))
        self.assertGreaterEqual(len(fetch_calls), 1)
        self.assertLessEqual(len(fetch_calls), 12)
        self.assertTrue(all(call["max_chars"] == 2400 for call in fetch_calls))
        self.assertIn("web_search_plan", payload)
        self.assertEqual(len(payload["search_audit"]), len(calls) + len(web_calls))
        arxiv_audit = [row for row in payload["search_audit"] if row["source"] == "arxiv"]
        web_audit = [row for row in payload["search_audit"] if row["source"] == "web"]
        self.assertEqual(len(arxiv_audit), len(calls))
        self.assertEqual(len(web_audit), len(web_calls))
        self.assertTrue(all(row["requested_results"] == 50 for row in arxiv_audit))
        self.assertTrue(all(row["requested_results"] == 24 for row in web_audit))
        self.assertTrue(all("retrieved" in row and "unique_added" in row for row in payload["search_audit"]))
        self.assertTrue(any(row["unique_added"] == 0 and row["retrieved"] > 0 for row in payload["search_audit"]))
        self.assertEqual(payload["web_results"][0]["source"], "web")
        self.assertEqual(payload["web_results"][0]["source_domain"], "example.org")
        web_source_types = {row["source_type"] for row in payload["web_results"]}
        self.assertIn("benchmark", web_source_types)
        self.assertIn("code_repository", web_source_types)
        self.assertIn("recent_papers", payload)
        self.assertIn("focused_papers", payload)
        self.assertIn("method_papers", payload)
        self.assertNotIn("architecture" + "_papers", payload)
        self.assertEqual(payload["assessment"]["risk"], "high_prior_art_risk")
        self.assertTrue(payload["assessment"]["closest_prior_art"])
        self.assertIn("sparse", payload["assessment"]["closest_prior_art"][0]["matched_terms"])
        self.assertIn("source_type", payload["assessment"]["closest_prior_art"][0])
        self.assertIn("source_credibility", payload["assessment"]["closest_prior_art"][0])
        self.assertIn("source_domain", payload["assessment"]["closest_prior_art"][0])
        self.assertIn("evidence_excerpt", payload["assessment"]["closest_prior_art"][0])
        self.assertTrue(any("Sparse autoencoder method paper" in item for item in payload["assessment"]["required_delta"]))
        axes = {row["axis"]: row for row in payload["assessment"]["comparison_matrix"]}
        self.assertIn("method", axes)
        self.assertIn("architecture", axes)
        self.assertIn("implementation", axes)
        self.assertTrue(axes["method"]["covered"])
        self.assertTrue(axes["implementation"]["covered"])
        self.assertTrue(axes["method"]["representative_prior"])
        self.assertEqual(payload["assessment"]["recent_pressure"]["latest_year"], 2025)
        self.assertEqual(payload["assessment"]["recent_pressure"]["status"], "recent_prior_present")
        self.assertTrue(any(row.get("fetched") for row in payload["web_results"]))
        self.assertIn(payload["assessment"]["evidence_strength"], {"strong_multi_source_overlap", "strong_but_narrow_overlap"})
        self.assertEqual(payload["assessment"]["source_diversity"], "broad_independent")
        self.assertIn("recent_window", payload["assessment"]["coverage"])
        self.assertEqual(payload["assessment"]["coverage"]["arxiv_results_per_query"], 50)
        self.assertEqual(payload["assessment"]["coverage"]["web_results_per_query"], 24)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["arxiv_query_count"], 10)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_query_count"], 8)
        self.assertIn("replication_failure_modes", payload["assessment"]["coverage"]["arxiv_focuses"])
        self.assertIn("web_replication_results", payload["assessment"]["coverage"]["web_focuses"])
        self.assertIn("recent_discovery", payload["assessment"]["coverage"]["arxiv_focuses"])
        self.assertIn("architecture_variant", payload["assessment"]["coverage"]["arxiv_focuses"])
        self.assertIn("web_recent_discovery", payload["assessment"]["coverage"]["web_focuses"])
        self.assertIn("web_architecture_variant", payload["assessment"]["coverage"]["web_focuses"])
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_results"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_results_with_snippets"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_pages_fetched"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_results_with_page_text"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_source_types"]["benchmark"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["web_source_types"]["code_repository"], 1)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["unique_source_domains"], 3)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["credible_source_count"], 2)
        self.assertIn("benchmark", payload["assessment"]["coverage"]["credible_source_types"])
        self.assertIn("github.com", payload["assessment"]["coverage"]["source_domain_counts"])
        self.assertGreaterEqual(payload["assessment"]["coverage"]["retrieved_evidence"], 2)
        self.assertEqual(payload["assessment"]["coverage"]["search_audit_rows"], len(payload["search_audit"]))
        self.assertGreater(payload["assessment"]["coverage"]["duplicate_only_search_passes"], 0)
        self.assertEqual(payload["assessment"]["coverage"]["empty_search_passes"], 0)
        self.assertIn("search_audit", payload["assessment"])
        self.assertEqual(payload["assessment"]["search_audit"]["pass_count"], len(payload["search_audit"]))
        self.assertGreater(payload["assessment"]["search_audit"]["duplicate_only_search_passes"], 0)
        self.assertTrue(payload["assessment"]["search_audit"]["focus_coverage"]["claim_collision"])
        self.assertEqual(payload["assessment"]["search_audit"]["missing_focus_coverage"], [])
        focus_audit = {
            (row["source"], row["focus"]): row
            for row in payload["assessment"]["search_audit"]["focus_summary"]
        }
        self.assertIn(("arxiv", "recent_discovery"), focus_audit)
        self.assertIn(("web", "web_recent_discovery"), focus_audit)
        self.assertGreaterEqual(focus_audit[("arxiv", "recent_discovery")]["retrieved"], 1)
        self.assertIn("claim_readiness", payload["assessment"])
        self.assertFalse(payload["assessment"]["claim_readiness"]["can_claim_high_novelty"])
        self.assertEqual(payload["assessment"]["claim_readiness"]["status"], "not_ready_prior_art_overlap")
        self.assertIn("focus_coverage", payload["assessment"]["coverage"])
        self.assertTrue(payload["assessment"]["coverage"]["focus_coverage"]["method"])
        self.assertTrue(payload["assessment"]["coverage"]["focus_coverage"]["recent_discovery"])
        self.assertTrue(payload["assessment"]["coverage"]["focus_coverage"]["architecture"])
        self.assertTrue(payload["assessment"]["coverage"]["focus_coverage"]["replication"])
        self.assertTrue(payload["assessment"]["coverage"]["focus_coverage"]["exact_phrase"])
        self.assertTrue(payload["assessment"]["coverage"]["focus_coverage"]["claim_collision"])
        self.assertTrue(payload["assessment"]["coverage"]["focus_coverage"]["peer_review"])
        self.assertIn("threat_model_coverage", payload["assessment"]["coverage"])
        self.assertTrue(payload["assessment"]["coverage"]["threat_model_coverage"]["exact_phrase_overlap"])
        self.assertTrue(payload["assessment"]["coverage"]["threat_model_coverage"]["claim_collision"])
        self.assertIn("novelty_threat_model", payload["assessment"])
        threats = {row["threat"]: row for row in payload["assessment"]["novelty_threat_model"]}
        self.assertIn("claim_collision", threats)
        self.assertIn("disqualifying_overlap_tests", payload["assessment"])
        self.assertTrue(any(row["test"] == "claim_collision" for row in payload["assessment"]["disqualifying_overlap_tests"]))
        self.assertIn("threat_model_depth", payload["assessment"]["claim_readiness"]["checks"])
        self.assertGreaterEqual(payload["assessment"]["coverage"]["recent_evidence"], 1)
        self.assertIn("claim-collision", payload["guidance"])
        self.assertIn("recent-discovery", payload["guidance"])
        self.assertIn("architecture-variant", payload["guidance"])

    def test_verify_novelty_search_plan_uses_idea_terms_without_fixed_architectures(self):
        from mechferret import tools

        idea = "adaptive receptor design for protein folding assays"
        arxiv_plan = tools._novelty_search_plan(idea, None)
        web_plan = tools._novelty_web_search_plan(idea, None)
        combined = " ".join(item["query"].lower() for item in [*arxiv_plan, *web_plan])

        self.assertIn("protein", combined)
        self.assertIn("method design", combined)
        self.assertIn("benchmark evaluation", combined)
        self.assertIn("implementation repository code", combined)
        self.assertIn("same contribution", combined)
        self.assertIn("we propose", combined)
        self.assertNotIn("transformer", combined)
        self.assertNotIn("sparse autoencoder", combined)
        self.assertNotIn("github " + "project", combined)

    def test_verify_novelty_reports_unknown_when_search_fails(self):
        from mechferret import tools

        with (
            patch("mechferret.knowledge.search_arxiv", side_effect=RuntimeError("network unavailable")),
            patch("mechferret.knowledge.web_search", side_effect=RuntimeError("network unavailable")),
        ):
            payload = json.loads(tools.run_tool("verify_novelty", {"idea": "adaptive probe routing for activation patches"}))
        if payload.get("tool_output_truncated"):
            payload = json.loads(Path(payload["full_output_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["assessment"]["risk"], "unknown_search_incomplete")
        self.assertEqual(payload["assessment"]["claim_readiness"]["status"], "not_ready_search_incomplete")
        self.assertFalse(payload["assessment"]["claim_readiness"]["can_claim_high_novelty"])
        self.assertIn("search_completed", payload["assessment"]["claim_readiness"]["missing_checks"])
        self.assertGreater(payload["assessment"]["coverage"]["failed_queries"], 0)
        self.assertGreater(payload["assessment"]["coverage"]["failed_arxiv_queries"], 0)
        self.assertGreater(payload["assessment"]["coverage"]["failed_web_queries"], 0)
        self.assertEqual(payload["assessment"]["coverage"]["search_audit_rows"], len(payload["search_audit"]))
        self.assertGreater(payload["assessment"]["coverage"]["empty_search_passes"], 0)
        self.assertGreater(payload["assessment"]["search_audit"]["failed_passes"], 0)
        self.assertTrue(all(row["failed"] for row in payload["search_audit"]))
        self.assertTrue(payload["assessment"]["search_audit"]["failed_focuses"])
        self.assertEqual(payload["related_papers"], [])
        self.assertEqual(payload["web_results"], [])

    def test_verify_novelty_reports_web_fetch_failures_without_losing_search_hits(self):
        from mechferret import tools

        def fake_search(query, max_results=20, sort_by="relevance"):
            return 0, []

        def fake_web_search(query, max_results=12):
            return [
                {
                    "title": "Sparse autoencoder project page",
                    "url": "https://example.org/project",
                    "snippet": "Project page for sparse autoencoder probes.",
                }
            ]

        with (
            patch("mechferret.knowledge.search_arxiv", side_effect=fake_search),
            patch("mechferret.knowledge.web_search", side_effect=fake_web_search),
            patch("mechferret.knowledge.web_fetch", side_effect=RuntimeError("fetch unavailable")),
        ):
            payload = json.loads(tools.run_tool("verify_novelty", {"idea": "sparse autoencoder probes"}))
        if payload.get("tool_output_truncated"):
            payload = json.loads(Path(payload["full_output_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["assessment"]["coverage"]["web_results"], 1)
        self.assertEqual(payload["assessment"]["coverage"]["web_pages_fetched"], 0)
        self.assertGreater(payload["assessment"]["coverage"]["failed_web_fetches"], 0)
        self.assertEqual(payload["assessment"]["search_audit"]["failed_passes"], 0)
        self.assertGreaterEqual(payload["assessment"]["coverage"]["empty_arxiv_passes"], 1)
        self.assertTrue(payload["web_results"])

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
            self.assertEqual(blocked["failed_checks"], ["source_material"])

            with patch("mechferret.controller.MechFerret.run", side_effect=ValueError("Live provider research failed for openai: provider down")):
                provider_failed = json.loads(
                    agent._run_tool(
                        "run_research",
                        {
                            "question": "What is new?",
                            "provider": "openai",
                            "include_memory": False,
                            "out_dir": str(root / "provider-failed"),
                            "db_path": str(root / "memory.sqlite"),
                        },
                    )
                )
            self.assertFalse(provider_failed["ok"])
            self.assertEqual(provider_failed["failed_checks"], ["provider_research"])
            self.assertIn("provider down", provider_failed["error"])

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

    def test_agent_dispatch_validates_options_before_picker(self):
        fired = []
        picked = []
        a = agent.Agent(on_tool=lambda name, args: fired.append(name))
        a.on_options = lambda options: picked.append(options) or "none"
        a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "x"

        invalid = json.loads(a._dispatch("present_options", {"options": [{"title": "thin"}]}))

        self.assertFalse(invalid["ok"])
        self.assertEqual(fired, [])
        self.assertEqual(picked, [])
        self.assertIn("options_argument", invalid["failed_checks"])

    def test_agent_dispatch_passes_normalized_options_to_picker(self):
        picked = []
        a = agent.Agent()
        a.on_options = lambda options: picked.append(options) or options[0]["title"]
        a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "x"
        args = {
            "options": [
                _validated_option("Novelty audit"),
                _validated_option("Second audit"),
            ]
        }

        selected = json.loads(a._dispatch("present_options", args))

        self.assertEqual(selected["user_selected"], "Novelty audit")
        self.assertEqual(selected["selected_option"]["title"], "Novelty audit")
        self.assertEqual(selected["selected_option"]["recent_pressure"]["status"], "recent_prior_present")
        self.assertEqual(selected["selected_option"]["comparison_matrix"][1]["axis"], "evaluation")
        self.assertEqual(selected["selected_option"]["novelty_threat_model"][1]["threat"], "claim_collision")
        self.assertEqual(selected["selected_option"]["disqualifying_overlap_tests"][0]["test"], "exact_phrase_overlap")
        self.assertEqual(selected["selected_option"]["search_audit"]["pass_count"], 18)
        self.assertIn("causal ablation", picked[0][0]["required_delta"])
        self.assertEqual(picked[0][0]["comparison_matrix"][1]["axis"], "evaluation")
        self.assertEqual(picked[0][0]["novelty_threat_model"][1]["risk"], "needs_delta_review")
        self.assertFalse(picked[0][0]["disqualifying_overlap_tests"][1]["passed"])
        self.assertEqual(picked[0][0]["search_audit"]["duplicate_only_search_passes"], 3)
        self.assertEqual(picked[0][0]["recent_pressure"]["status"], "recent_prior_present")

    def test_agent_dispatch_preserves_deferred_option_selection_payload(self):
        a = agent.Agent()
        a.on_options = lambda options: {
            "ok": False,
            "user_selected": "none",
            "selection_deferred": True,
            "failed_checks": ["interactive_selection_unavailable"],
            "option_details": options,
        }
        a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "x"
        args = {
            "options": [
                _validated_option("Novelty audit"),
                _validated_option("Second audit"),
            ]
        }

        selected = json.loads(a._dispatch("present_options", args))

        self.assertFalse(selected["ok"])
        self.assertTrue(selected["selection_deferred"])
        self.assertEqual(selected["failed_checks"], ["interactive_selection_unavailable"])
        self.assertEqual(selected["option_details"][0]["title"], "Novelty audit")
        self.assertEqual(selected["option_details"][0]["novelty_threat_model"][0]["threat"], "exact_phrase_overlap")
        self.assertEqual(selected["option_details"][0]["disqualifying_overlap_tests"][1]["test"], "claim_collision")
        self.assertEqual(selected["option_details"][0]["search_audit"]["empty_search_passes"], 2)
        self.assertNotIn("selected_option", selected)

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
                "evidence_strength": "strong_multi_source_overlap",
                "source_diversity": "broad_independent",
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
        self.assertEqual(parsed["assessment"]["evidence_strength"], "strong_multi_source_overlap")
        self.assertEqual(parsed["assessment"]["source_diversity"], "broad_independent")
        self.assertEqual(parsed["assessment"]["coverage"]["web_source_types"]["benchmark"], 2)
        self.assertEqual(parsed["assessment"]["closest_prior_art"][0]["source_type"], "benchmark")

    def test_minimal_large_novelty_output_preserves_nested_assessment(self):
        from mechferret import tools

        closest = [
            {
                "title": f"closest prior {index}",
                "url": f"https://arxiv.org/abs/2501.{index:04d}",
                "source": "arxiv",
                "source_type": "paper",
                "source_domain": "arxiv.org",
                "score": 0.91,
                "matched_terms": ["sparse", "autoencoder", "robot"],
                "evidence_excerpt": "overlap " * 200,
            }
            for index in range(120)
        ]
        payload = {
            "idea": "large novelty result",
            "related_papers": [{"title": f"paper {index}", "abstract": "x" * 800} for index in range(300)],
            "recent_papers": [{"title": f"recent {index}"} for index in range(80)],
            "focused_papers": [{"title": f"focused {index}"} for index in range(80)],
            "method_papers": [{"title": f"method {index}"} for index in range(80)],
            "web_results": [{"title": f"web {index}"} for index in range(80)],
            "assessment": {
                "risk": "high_prior_art_risk",
                "verdict": "Closest retrieved evidence overlaps.",
                "evidence_strength": "strong_multi_source_overlap",
                "source_diversity": "broad_independent",
                "coverage": {
                    "arxiv_query_count": 20,
                    "web_query_count": 12,
                    "arxiv_results_per_query": 50,
                    "web_results_per_query": 24,
                    "focus_coverage": {"recent_discovery": True, "architecture": True},
                    "long": "y" * 4000,
                },
                "claim_readiness": {
                    "status": "not_ready_prior_art_overlap",
                    "can_claim_high_novelty": False,
                    "missing_checks": [],
                    "next_actions": ["Write the exact delta."],
                },
                "closest_prior_art": closest,
                "required_delta": ["Name the difference."],
            },
        }
        raw = json.dumps(payload)
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = tools.RESULTS_DIR
            tools.RESULTS_DIR = Path(tmp)
            try:
                out = tools._persist_if_large("verify_novelty", raw)
                saved_path_exists = Path(json.loads(out)["full_output_path"]).exists()
            finally:
                tools.RESULTS_DIR = original_dir

        parsed = json.loads(out)
        self.assertTrue(parsed["tool_output_truncated"])
        self.assertEqual(parsed["assessment"]["risk"], "high_prior_art_risk")
        self.assertEqual(parsed["assessment"]["claim_readiness"]["status"], "not_ready_prior_art_overlap")
        self.assertEqual(parsed["assessment"]["coverage"]["arxiv_results_per_query"], 50)
        self.assertEqual(parsed["assessment"]["closest_prior_art_count"], 120)
        self.assertLessEqual(len(parsed["assessment"]["closest_prior_art"]), 5)
        self.assertEqual(parsed["related_papers_count"], 300)
        self.assertTrue(saved_path_exists)

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

    def test_active_provider_uses_env_only_when_config_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "missing.json"
            old = os.environ.get("MECHFERRET_CONFIG")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            old_openai_model = os.environ.get("MECHFERRET_OPENAI_MODEL")
            os.environ["MECHFERRET_CONFIG"] = str(cfg)
            os.environ["OPENAI_API_KEY"] = "env-key"
            os.environ["MECHFERRET_OPENAI_MODEL"] = "env-model"
            try:
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

    def test_active_provider_respects_saved_local_default_with_env_present(self):
        from mechferret.config import MechFerretConfig, save_config

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            old = os.environ.get("MECHFERRET_CONFIG")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            old_openai_model = os.environ.get("MECHFERRET_OPENAI_MODEL")
            os.environ["MECHFERRET_CONFIG"] = str(cfg)
            os.environ["OPENAI_API_KEY"] = "env-key"
            os.environ["MECHFERRET_OPENAI_MODEL"] = "env-model"
            try:
                save_config(MechFerretConfig(default_provider="local"), cfg)
                self.assertEqual(agent.active_provider(), ("", "", ""))
                self.assertFalse(agent.is_configured())
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
            self.assertEqual(loaded.default_provider, "openai")
            self.assertEqual(loaded.providers["openai"].api_key, "key")
            self.assertEqual(loaded.providers["openai"].model, "gpt-trim")

            with patch.dict(os.environ, {}, clear=True):
                partial = configure_provider("anthropic", "  key  ", path=root / "partial.json")
            partial_loaded = load_config(partial)
            self.assertEqual(partial_loaded.default_provider, "local")
            self.assertEqual(partial_loaded.providers["anthropic"].api_key, "key")
            self.assertEqual(partial_loaded.providers["anthropic"].model, "")


if __name__ == "__main__":
    unittest.main()
