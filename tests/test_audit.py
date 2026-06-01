import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mechferret.audit import audit_run_artifact, latest_run_json, load_run_artifact
from mechferret.cli import main


def _payload(readiness=0.8):
    return {
        "run_id": "run_audit",
        "question": "Can this become a paper?",
        "created_at": "2026-05-30T00:00:00+00:00",
        "mode": "discovery",
        "plan": {
            "question": "Can this become a paper?",
            "strategy": "test",
            "steps": [
                {"id": "p1", "question": "q", "intent": "retrieve", "status": "done", "notes": []},
            ],
        },
        "sources": [
            {"id": "s1", "title": "S1", "text": "text"},
            {"id": "s2", "title": "S2", "text": "text"},
            {"id": "s3", "title": "S3", "text": "text"},
        ],
        "evidence": [
            {"id": "e1", "source_id": "s1", "title": "S1", "text": "text", "score": 1.0},
            {"id": "e2", "source_id": "s2", "title": "S2", "text": "text", "score": 1.0},
            {"id": "e3", "source_id": "s3", "title": "S3", "text": "text", "score": 1.0},
        ],
        "claims": [
            {"id": f"c{i}", "text": f"claim {i}", "citations": ["e1"], "source_ids": ["s1"], "confidence": 0.8, "support_score": 0.8}
            for i in range(5)
        ],
        "contradictions": [],
        "gaps": [],
        "answer": "answer",
        "metrics": {
            "readiness_score": readiness,
            "source_diversity": 3,
            "citation_density": 1.0,
            "plan_coverage": 1.0,
            "contradiction_pressure": 0.0,
            "reproducibility_rate": 1.0,
        },
        "artifacts": {},
        "hypotheses": [
            {
                "id": "h1",
                "statement": "Head copies names.",
                "rationale": "r",
                "task": "ioi",
                "predicted_effect": "drops logit diff",
                "status": "confirmed",
                "confidence": 0.8,
            }
        ],
        "experiments": [
            {
                "id": "x1",
                "spec_id": "s1",
                "probe": "head_ablation",
                "status": "ran",
                "effect_size": 1.0,
                "baseline": 0.0,
                "per_seed": [0.9, 1.0, 1.1],
                "significant": True,
                "reproduced": True,
            },
            {
                "id": "x2",
                "spec_id": "s2",
                "probe": "activation_patching",
                "status": "ran",
                "effect_size": 0.8,
                "baseline": 0.0,
                "per_seed": [0.7, 0.8, 0.9],
                "significant": True,
                "reproduced": True,
            },
        ],
        "discoveries": [
            {
                "id": "d1",
                "statement": "Head copies names.",
                "confidence": 0.8,
                "novelty": 0.6,
                "effect_size": 1.0,
                "reproducibility": 1.0,
                "supporting_experiments": ["x1", "x2"],
                "hypothesis_id": "h1",
            }
        ],
    }


def _valid_paper_tex() -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\begin{document}",
            "\\section{Results}",
            "Recorded findings.",
            "\\section{Experiment Ledger}",
            "Recorded experiments.",
            "\\section{Evidence Ledger}",
            "Recorded evidence.",
            "\\section{Limitations}",
            "Recorded gaps.",
            "\\end{document}",
            "",
        ]
    )


class AuditTest(unittest.TestCase):
    def test_latest_run_json_finds_newest_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "a" / "run.json"
            newer = root / "b" / "run.json"
            older.parent.mkdir()
            newer.parent.mkdir()
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            newer.touch()
            self.assertEqual(latest_run_json(root), newer)

    def test_audit_reports_missing_paper_as_next_action_for_ready_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            result = audit_run_artifact(path)
            failed = {c["name"] for c in result["checks"] if not c["passed"]}
            self.assertIn("paper_artifact_exists", failed)
            self.assertIn("paper_artifact_structure", failed)
            self.assertIn("paper_artifact_exists", result["failed_checks"])
            paper_actions = [action for action in result["next_actions"] if "paper/main.tex" in action]
            self.assertEqual(len(paper_actions), 1)
            self.assertIn("mechferret paper --provider local", paper_actions[0])

    def test_audit_passes_seed_gate_when_all_ran_experiments_have_per_seed_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "run.json"
            path.write_text(json.dumps(_payload(readiness=0.5)), encoding="utf-8")
            (root / "paper").mkdir()
            (root / "paper" / "main.tex").write_text(_valid_paper_tex(), encoding="utf-8")
            result = audit_run_artifact(path)
            seed_gate = next(c for c in result["checks"] if c["name"] == "experiments_log_seed_values")
            self.assertTrue(seed_gate["passed"])

    def test_audit_advisories_do_not_fail_strict_gates(self):
        payload = _payload(readiness=0.5)
        payload["question"] = "Head copies names"
        payload["provenance"] = {
            "answer_author": "experiment_ledger_synthesizer",
            "backend_used": "synthetic",
            "used_packaged_seed_corpus": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            (root / "paper").mkdir()
            (root / "paper" / "main.tex").write_text(_valid_paper_tex(), encoding="utf-8")

            result = audit_run_artifact(path)
            self.assertTrue(result["passed"])
            names = {item["name"] for item in result["advisories"]}
            self.assertIn("local_synthesis_not_final", names)
            self.assertIn("synthetic_backend_not_final", names)
            self.assertIn("packaged_seed_corpus_used", names)
            self.assertTrue(result["advisory_actions"])

    def test_top_level_paper_does_not_satisfy_run_bound_paper_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "runs" / "demo" / "run.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            (root / "paper").mkdir()
            (root / "paper" / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
            result = audit_run_artifact(path)
            failed = {c["name"] for c in result["checks"] if not c["passed"]}
            self.assertIn("paper_artifact_exists", failed)

    def test_missing_artifact_gives_actionable_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_run_artifact(runs_root=tmp)
            self.assertFalse(result["passed"])
            self.assertIn("run_artifact_exists", result["checks"][0]["name"])
            self.assertEqual(result["failed_checks"], ["run_artifact_exists"])
            self.assertTrue(result["next_actions"])

    def test_audit_tolerates_malformed_nested_artifact_rows(self):
        payload = _payload(readiness="bad")
        payload["plan"] = {"steps": ["bad", {"id": 7}]}
        payload["sources"] = ["bad", {"id": "s1", "metadata": ["bad"]}]
        payload["evidence"] = [{"id": "e1", "score": "bad"}]
        payload["claims"] = ["bad", {"id": "c1", "citations": "e1", "confidence": "bad"}]
        payload["contradictions"] = [{"severity": "bad"}]
        payload["gaps"] = "bad"
        payload["metrics"] = {
            "readiness_score": "bad",
            "source_diversity": "3",
            "citation_density": "bad",
            "plan_coverage": None,
            "contradiction_pressure": "bad",
        }
        payload["hypotheses"] = [{"confidence": "bad"}]
        payload["experiments"] = [{"status": "ran", "effect_size": "bad", "baseline": "bad"}]
        payload["discoveries"] = [{"supporting_experiments": "bad", "reproducibility": "bad"}]
        payload["artifacts"] = ["bad"]
        payload["provenance"] = ["bad"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            run = load_run_artifact(path)
            self.assertEqual(len(run.plan.steps), 1)
            self.assertEqual(len(run.sources), 1)
            self.assertEqual(run.sources[0].metadata, {})
            self.assertEqual(run.claims[0].citations, [])
            result = audit_run_artifact(path)
            self.assertFalse(result["passed"])
            self.assertTrue(result["failed_checks"])

    def test_audit_flags_question_result_drift(self):
        payload = _payload()
        payload["question"] = "Find SAEs for OpenVLA"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = audit_run_artifact(path)
            alignment = next(c for c in result["checks"] if c["name"] == "question_result_alignment")
            self.assertFalse(alignment["passed"])
            self.assertIn("question_result_alignment", result["failed_checks"])
            self.assertIn("Rerun or narrow", " ".join(result["next_actions"]))

    def test_audit_cli_strict_exits_nonzero_on_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            out = StringIO()
            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(out):
                    main(["audit", str(path), "--strict", "--json"])
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("paper_artifact_exists", out.getvalue())

    def test_audit_cli_strict_passes_clean_artifact(self):
        payload = _payload(readiness=0.5)
        payload["question"] = "Head copies names"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            (root / "paper").mkdir()
            (root / "paper" / "main.tex").write_text(_valid_paper_tex(), encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                main(["audit", str(path), "--strict", "--json"])
            self.assertIn('"passed": true', out.getvalue())

    def test_audit_rejects_malformed_run_bound_paper(self):
        payload = _payload()
        payload["question"] = "Head copies names"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            (root / "paper").mkdir()
            (root / "paper" / "main.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
            result = audit_run_artifact(path)
            self.assertIn("paper_artifact_structure", result["failed_checks"])
            structure = next(c for c in result["checks"] if c["name"] == "paper_artifact_structure")
            self.assertFalse(structure["passed"])
            self.assertIn("Evidence Ledger", structure["threshold"])

    def test_audit_flags_manifest_tampering_when_manifest_exists(self):
        from mechferret.controller import MechFerret

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "Reliable autoresearch agents need planning, retrieval, citations, evidence, and critic loops.",
                encoding="utf-8",
            )
            MechFerret(root / "memory.sqlite").run(
                "How should autoresearch agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )
            (root / "run" / "report.md").write_text("tampered\n", encoding="utf-8")

            result = audit_run_artifact(root / "run" / "run.json")
            manifest_gate = next(c for c in result["checks"] if c["name"] == "manifest_integrity")
            self.assertFalse(manifest_gate["passed"])
            self.assertIn("manifest_integrity", result["failed_checks"])
            self.assertIn("artifact_sha256:markdown", manifest_gate["observed"])

    def test_cli_paper_default_satisfies_sibling_audit_gate(self):
        payload = _payload()
        payload["question"] = "Head copies names"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "runs" / "demo" / "run.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                main(["paper", str(path), "--provider", "local"])
            self.assertTrue((path.parent / "paper" / "main.tex").exists())
            audit_out = StringIO()
            with redirect_stdout(audit_out):
                try:
                    main(["audit", str(path), "--strict", "--json"])
                except SystemExit as exc:
                    self.fail(f"audit strict failed with exit {exc.code}: {audit_out.getvalue()}")


if __name__ == "__main__":
    unittest.main()
