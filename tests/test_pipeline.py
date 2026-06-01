import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mechferret.controller import MechFerret, dedupe_evidence
from mechferret.llm import _answer_prompt
from mechferret.models import Claim, EvidenceChunk
from mechferret.ops import verify_run_artifacts
from mechferret.provenance import refresh_run_manifest


class PipelineTest(unittest.TestCase):
    def test_run_creates_artifacts_and_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "# Agent Research\n"
                "Autoresearch agents need planning, retrieval, evidence citations, and critic loops. "
                "A reliable implementation tracks source diversity and contradiction pressure. "
                "Inspectable traces make agent failures easier to debug.",
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
            self.assertEqual(Path(payload["artifacts"]["json"]), root / "run" / "run.json")
            self.assertEqual(Path(payload["artifacts"]["html"]), root / "run" / "report.html")
            self.assertEqual(Path(payload["artifacts"]["manifest"]), root / "run" / "manifest.json")
            self.assertEqual(payload["provenance"]["engine"], "literature")
            self.assertEqual(payload["provenance"]["answer_author"], "local_extractive_synthesizer")
            self.assertIn("local extractive synthesis", (root / "run" / "report.md").read_text(encoding="utf-8"))
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], run.run_id)
            self.assertIn("run_ledger", manifest)
            self.assertIn("sha256", manifest["run_ledger"])
            self.assertIn("sources", manifest)
            self.assertIn("html", manifest["artifacts"])
            self.assertTrue(verify_run_artifacts(root / "run" / "run.json")["passed"])

            (root / "run" / "report.md").write_text("tampered\n", encoding="utf-8")
            verification = verify_run_artifacts(root / "run" / "run.json")
            self.assertFalse(verification["passed"])
            self.assertIn("artifact_sha256:markdown", verification["failed_checks"])
            self.assertFalse(verification["repairable"])
            self.assertIn("artifact_sha256:markdown", verification["repair_blockers"])
            blocked = verify_run_artifacts(root / "run" / "run.json", repair=True)
            self.assertFalse(blocked["passed"])
            self.assertTrue(blocked["repair_blocked"])
            self.assertIn("artifact_sha256:markdown", blocked["repair_blockers"])

    def test_verify_repair_blocks_run_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )
            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["question"] = "Changed after manifest creation?"
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("question_matches_manifest", verification["failed_checks"])
            self.assertFalse(verification["repairable"])
            self.assertIn("question_matches_manifest", verification["repair_blockers"])
            blocked = verify_run_artifacts(run_json, repair=True)
            self.assertFalse(blocked["passed"])
            self.assertTrue(blocked["repair_blocked"])
            self.assertFalse(blocked["repair_attempted"])
            self.assertIn("question_matches_manifest", blocked["repair_blockers"])

    def test_verify_repair_blocks_missing_or_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )
            run_json = root / "run" / "run.json"
            manifest = root / "run" / "manifest.json"
            manifest.unlink()

            missing = verify_run_artifacts(run_json)
            self.assertFalse(missing["passed"])
            self.assertFalse(missing["repairable"])
            self.assertIn("manifest_exists", missing["repair_blockers"])
            blocked_missing = verify_run_artifacts(run_json, repair=True)
            self.assertTrue(blocked_missing["repair_blocked"])
            self.assertIn("manifest_exists", blocked_missing["repair_blockers"])

            manifest.write_text("{not-json", encoding="utf-8")
            corrupt = verify_run_artifacts(run_json)
            self.assertFalse(corrupt["passed"])
            self.assertFalse(corrupt["repairable"])
            self.assertIn("manifest_parseable", corrupt["repair_blockers"])
            blocked_corrupt = verify_run_artifacts(run_json, repair=True)
            self.assertTrue(blocked_corrupt["repair_blocked"])
            self.assertIn("manifest_parseable", blocked_corrupt["repair_blockers"])

    def test_verify_requires_manifest_to_track_declared_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            paper = root / "run" / "paper" / "main.tex"
            paper.parent.mkdir()
            paper.write_text(
                "\\documentclass{article}\\begin{document}"
                "\\section{Results}ok"
                "\\section{Experiment Ledger}ok"
                "\\section{Evidence Ledger}ok"
                "\\section{Limitations}ok"
                "\\end{document}",
                encoding="utf-8",
            )
            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["artifacts"]["paper"] = str(paper)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("manifest_tracks_declared_artifact:paper", verification["failed_checks"])
            self.assertIn("same artifact set", " ".join(verification["next_actions"]))
            self.assertTrue(verification["repairable"])
            self.assertIn("--repair --strict", verification["repair_command"])
            self.assertIn("mechferret verify", " ".join(verification["next_actions"]))
            repaired = verify_run_artifacts(run_json, repair=True)
            self.assertTrue(repaired["passed"])
            self.assertTrue(repaired["repair_attempted"])
            self.assertTrue(repaired["repaired"])
            self.assertIn("manifest_tracks_declared_artifact:paper", repaired["before_failed_checks"])
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("paper", manifest["artifacts"])

    def test_verify_rejects_manifest_artifacts_not_declared_by_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["artifacts"].pop("html")
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("manifest_artifact_declared:html", verification["failed_checks"])

    def test_verify_can_repair_stale_manifest_header_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 0
            manifest["mode"] = "discovery"
            manifest["provenance"] = {"engine": "changed"}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertTrue(verification["repairable"])
            self.assertIn("manifest_schema_version_supported", verification["failed_checks"])
            self.assertIn("mode_matches_manifest", verification["failed_checks"])
            self.assertIn("provenance_matches_manifest", verification["failed_checks"])

            repaired = verify_run_artifacts(run_json, repair=True)
            self.assertTrue(repaired["passed"])
            self.assertTrue(repaired["repaired"])
            repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(repaired_manifest["schema_version"], 1)
            self.assertEqual(repaired_manifest["mode"], "literature")
            self.assertEqual(repaired_manifest["provenance"]["engine"], "literature")

    def test_verify_rejects_run_ledger_answer_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["answer"] = "A rewritten answer that was not in the original manifest."
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("run_ledger_sha256", verification["failed_checks"])
            self.assertIn("run_ledger_sha256", verification["repair_blockers"])
            self.assertIn("run ledger", " ".join(verification["next_actions"]))

            blocked = verify_run_artifacts(run_json, repair=True)
            self.assertFalse(blocked["passed"])
            self.assertTrue(blocked["repair_blocked"])
            self.assertIn("run_ledger_sha256", blocked["repair_blockers"])

    def test_manifest_refresh_refuses_to_bless_run_ledger_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["metrics"]["readiness_score"] = 1.0
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "run ledger changed"):
                refresh_run_manifest(run_json)

    def test_verify_can_repair_missing_run_ledger_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("run_ledger")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertTrue(verification["repairable"])
            self.assertIn("run_ledger_sha256_declared", verification["failed_checks"])

            repaired = verify_run_artifacts(run_json, repair=True)
            self.assertTrue(repaired["passed"])
            self.assertTrue(repaired["repaired"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("run_ledger", manifest)

    def test_verify_rejects_malformed_run_ledger_hash_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_ledger"]["sha256"] = "z" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("run_ledger_sha256_declared", verification["failed_checks"])
            self.assertNotIn("run_ledger_sha256", verification["failed_checks"])

    def test_verify_rejects_source_text_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            source_id = payload["sources"][0]["id"]
            payload["sources"][0]["text"] = "Rewritten source text after manifest creation."
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn(f"source_text_sha256_matches:{source_id}", verification["failed_checks"])
            self.assertIn(f"source_text_sha256_matches:{source_id}", verification["repair_blockers"])

            blocked = verify_run_artifacts(run_json, repair=True)
            self.assertFalse(blocked["passed"])
            self.assertTrue(blocked["repair_blocked"])
            self.assertIn(f"source_text_sha256_matches:{source_id}", blocked["repair_blockers"])

    def test_verify_rejects_malformed_source_hash_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            source_id = payload["sources"][0]["id"]
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["text_sha256"] = "z" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn(f"source_text_sha256_declared:{source_id}", verification["failed_checks"])
            self.assertNotIn(f"source_text_sha256_matches:{source_id}", verification["failed_checks"])

    def test_verify_rejects_malformed_source_byte_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            source_id = payload["sources"][0]["id"]
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["text_bytes"] = True
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn(f"source_text_bytes_declared:{source_id}", verification["failed_checks"])
            self.assertNotIn(f"source_text_bytes_matches:{source_id}", verification["failed_checks"])

    def test_verify_rejects_manifest_sources_not_declared_by_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            extra = dict(manifest["sources"][0])
            extra["id"] = "untracked_source"
            manifest["sources"].append(extra)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(root / "run" / "run.json")
            self.assertFalse(verification["passed"])
            self.assertIn("source_count_matches_manifest", verification["failed_checks"])
            self.assertIn("manifest_source_declared:untracked_source", verification["failed_checks"])

    def test_verify_rejects_evidence_chunks_with_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            evidence_id = payload["evidence"][0]["id"]
            payload["evidence"][0]["source_id"] = "missing_source"
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn(f"evidence_source_tracked:{evidence_id}", verification["failed_checks"])
            self.assertIn(f"evidence_source_tracked:{evidence_id}", verification["repair_blockers"])
            self.assertIn("run ledger", " ".join(verification["next_actions"]))

    def test_verify_rejects_claim_citations_with_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            claim_id = payload["claims"][0]["id"]
            payload["claims"][0]["citations"] = ["missing_evidence"]
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn(f"claim_citation_tracked:{claim_id}:missing_evidence", verification["failed_checks"])
            self.assertIn(f"claim_citation_tracked:{claim_id}:missing_evidence", verification["repair_blockers"])

    def test_verify_rejects_duplicate_evidence_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            duplicate = dict(payload["evidence"][0])
            payload["evidence"].append(duplicate)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("evidence_ids_unique", verification["failed_checks"])

    def test_verify_rejects_broken_discovery_graph_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["hypotheses"] = [
                {
                    "id": "h1",
                    "statement": "Citation discipline improves research-agent reliability.",
                    "rationale": "Grounding claims in the source ledger should reduce hallucinated support.",
                    "task": "agent_reliability",
                    "predicted_effect": "higher verified citation coverage",
                    "experiment_ids": ["missing_experiment"],
                    "source_ids": [payload["sources"][0]["id"]],
                }
            ]
            payload["experiments"] = [
                {
                    "id": "x1",
                    "spec_id": "spec_1",
                    "probe": "synthetic_probe",
                    "status": "ran",
                    "effect_size": 0.3,
                    "baseline": 0.1,
                }
            ]
            payload["discoveries"] = [
                {
                    "id": "d1",
                    "statement": "Research-agent citations are only shareable when the graph is closed.",
                    "confidence": 0.8,
                    "novelty": 0.5,
                    "effect_size": 0.3,
                    "reproducibility": 1.0,
                    "supporting_experiments": ["missing_experiment"],
                    "claim_ids": ["missing_claim"],
                    "hypothesis_id": "missing_hypothesis",
                }
            ]
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("hypothesis_experiment_tracked:h1:missing_experiment", verification["failed_checks"])
            self.assertIn("discovery_experiment_tracked:d1:missing_experiment", verification["failed_checks"])
            self.assertIn("discovery_claim_tracked:d1:missing_claim", verification["failed_checks"])
            self.assertIn("discovery_hypothesis_tracked:d1:missing_hypothesis", verification["failed_checks"])
            self.assertIn("discovery_claim_tracked:d1:missing_claim", verification["repair_blockers"])

    def test_verify_accepts_experiment_spec_references_in_discovery_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            claim_id = payload["claims"][0]["id"]
            source_id = payload["sources"][0]["id"]
            payload["hypotheses"] = [
                {
                    "id": "h1",
                    "statement": "Citation discipline improves research-agent reliability.",
                    "rationale": "Grounding claims in the source ledger should reduce hallucinated support.",
                    "task": "agent_reliability",
                    "predicted_effect": "higher verified citation coverage",
                    "experiment_ids": ["spec_1"],
                    "source_ids": [source_id],
                }
            ]
            payload["experiments"] = [
                {
                    "id": "x1",
                    "spec_id": "spec_1",
                    "probe": "synthetic_probe",
                    "status": "ran",
                    "effect_size": 0.3,
                    "baseline": 0.1,
                }
            ]
            payload["discoveries"] = [
                {
                    "id": "d1",
                    "statement": "Research-agent citations are only shareable when the graph is closed.",
                    "confidence": 0.8,
                    "novelty": 0.5,
                    "effect_size": 0.3,
                    "reproducibility": 1.0,
                    "supporting_experiments": ["spec_1"],
                    "claim_ids": [claim_id],
                    "hypothesis_id": "h1",
                }
            ]
            for artifact in ("markdown", "html", "graph", "evals"):
                payload.get("artifacts", {}).pop(artifact, None)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            (root / "run" / "manifest.json").unlink()
            refresh_run_manifest(run_json)

            verification = verify_run_artifacts(run_json)
            self.assertTrue(verification["passed"], verification["failed_checks"])

    def test_verify_rejects_malformed_discovery_graph_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["hypotheses"] = {"id": "h1"}
            payload["experiments"] = {"id": "x1"}
            payload["discoveries"] = {"id": "d1"}
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("run_hypotheses_parseable", verification["failed_checks"])
            self.assertIn("run_experiments_parseable", verification["failed_checks"])
            self.assertIn("run_discoveries_parseable", verification["failed_checks"])

    def test_verify_rejects_graph_and_evals_sidecar_drift_after_manifest_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            (root / "run" / "graph.json").write_text(
                json.dumps({"run_id": "changed", "question": "changed", "nodes": [], "edges": []}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (root / "run" / "evals.json").write_text(
                json.dumps({"run_id": "changed", "passed": True, "checks": [], "readiness_score": 1.0}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            refresh_run_manifest(run_json)

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("graph_sidecar_matches_run", verification["failed_checks"])
            self.assertIn("evals_sidecar_matches_run", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:graph", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:evals", verification["failed_checks"])

    def test_verify_rejects_report_sidecar_drift_after_manifest_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            (root / "run" / "report.md").write_text("# Rewritten report\n", encoding="utf-8")
            (root / "run" / "report.html").write_text("<!doctype html><title>rewritten</title>\n", encoding="utf-8")
            refresh_run_manifest(run_json)

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("markdown_sidecar_matches_run", verification["failed_checks"])
            self.assertIn("html_sidecar_matches_run", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:markdown", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:html", verification["failed_checks"])

    def test_verify_rejects_malformed_paper_artifact_after_manifest_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            paper = root / "run" / "paper" / "main.tex"
            paper.parent.mkdir()
            paper.write_text("\\documentclass{article}\\begin{document}ok\\end{document}", encoding="utf-8")
            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["artifacts"]["paper"] = str(paper)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            refresh_run_manifest(run_json)

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("paper_artifact_latex_structure", verification["failed_checks"])
            structure = next(c for c in verification["checks"] if c["name"] == "paper_artifact_latex_structure")
            self.assertIn("Evidence Ledger", structure["threshold"])
            self.assertNotIn("artifact_sha256:paper", verification["failed_checks"])

    def test_verify_rejects_malformed_review_and_pdf_artifacts_after_manifest_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            paper_dir = root / "run" / "paper"
            paper_dir.mkdir()
            review = paper_dir / "review.md"
            pdf = paper_dir / "main.pdf"
            review.write_text("Recommendation: Borderline\n", encoding="utf-8")
            pdf.write_bytes(b"%PDF fake\n")
            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["artifacts"]["review"] = str(review)
            payload["artifacts"]["pdf"] = str(pdf)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            refresh_run_manifest(run_json)

            review.write_text("\n", encoding="utf-8")
            pdf.write_bytes(b"not a pdf\n")
            refresh_run_manifest(run_json)

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("review_artifact_nonempty", verification["failed_checks"])
            self.assertIn("pdf_artifact_header", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:review", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:pdf", verification["failed_checks"])

    def test_verify_rejects_wrong_run_trace_after_manifest_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            trace = root / "run" / "trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "trace_id": "trace",
                        "run_id": "other_run",
                        "span_id": "span",
                        "phase": "event",
                        "name": "artifacts_written",
                        "time_unix_ms": 1,
                        "elapsed_ms": 0.0,
                        "attributes": {},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            run_json = root / "run" / "run.json"
            refresh_run_manifest(run_json)

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("trace_artifact_run_id", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:trace", verification["failed_checks"])

    def test_verify_rejects_empty_manifest_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["html"]["path"] = ""
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("artifact_path_declared:html", verification["failed_checks"])
            self.assertIn("artifact_exists:html", verification["failed_checks"])

    def test_verify_rejects_malformed_manifest_hash_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["html"]["sha256"] = "z" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("artifact_sha256_declared:html", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:html", verification["failed_checks"])

    def test_verify_rejects_malformed_manifest_byte_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["html"]["bytes"] = "large"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("artifact_bytes_declared:html", verification["failed_checks"])

    def test_verify_rejects_stale_mutable_manifest_byte_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["json"]["bytes"] += 1
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("artifact_bytes:json", verification["failed_checks"])

    def test_verify_rejects_non_string_manifest_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Reliable agents need citations and inspectable evidence.", encoding="utf-8")
            engine = MechFerret(root / "memory.sqlite")
            engine.run(
                "How should agents stay reliable?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )

            run_json = root / "run" / "run.json"
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["html"]["path"] = ["report.html"]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertIn("artifact_path_declared:html", verification["failed_checks"])

    def test_explicit_empty_corpus_does_not_fallback_to_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            (source_dir / "paper.pdf").write_bytes(b"%PDF fake")
            engine = MechFerret(root / "memory.sqlite")
            with self.assertRaisesRegex(ValueError, "No supported text sources"):
                engine.run(
                    "What is in my PDF?",
                    source_paths=[str(source_dir)],
                    out_dir=root / "run",
                    include_memory=False,
                )

    def test_explicit_unsupported_file_does_not_run_as_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paper.pdf"
            source.write_bytes(b"%PDF fake")
            engine = MechFerret(root / "memory.sqlite")
            with self.assertRaisesRegex(ValueError, "No supported text sources"):
                engine.run(
                    "What is in my PDF?",
                    source_paths=[str(source)],
                    out_dir=root / "run",
                    include_memory=False,
                )

    def test_run_without_sources_requires_provider_memory_or_explicit_seed_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = MechFerret(root / "memory.sqlite")
            with self.assertRaisesRegex(ValueError, "No source material"):
                engine.run(
                    "What should I build next?",
                    out_dir=root / "blocked",
                    provider="local",
                    include_memory=False,
                )

            seeded = engine.run(
                "What should I build next?",
                out_dir=root / "seeded",
                provider="local",
                include_memory=False,
                allow_seed_corpus=True,
            )
            self.assertTrue(seeded.provenance["used_packaged_seed_corpus"])
            self.assertTrue((root / "seeded" / "run.json").exists())

    def test_run_api_normalizes_malformed_boundary_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = MechFerret(root / "memory.sqlite")
            run = engine.run(
                b"What should I build next?",
                source_paths=None,
                urls=["", None],
                out_dir=root / "seeded",
                max_rounds="bad",
                provider=[],
                model={},
                use_openai="yes",
                include_memory=[],
                allow_seed_corpus=True,
            )
            self.assertEqual(run.provenance["max_rounds"], 2)
            self.assertTrue(run.provenance["included_memory"])
            self.assertEqual(run.provenance["provider_requested"], "auto")
            self.assertEqual(run.provenance["requested_urls"], [])
            self.assertTrue(run.provenance["used_packaged_seed_corpus"])

    def test_explicit_provider_can_author_final_synthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "Autoresearch reliability depends on retrieval, citations, critic loops, and inspectable traces.",
                encoding="utf-8",
            )
            with (
                patch("mechferret.llm.configured_api_key", return_value="test-key"),
                patch("mechferret.llm.configured_model", return_value="test-model"),
                patch("mechferret.llm.OpenAIWebResearch.search_summary", return_value=None),
                patch("mechferret.llm._call_openai", return_value="Provider-authored answer with evidence IDs."),
            ):
                run = MechFerret(root / "memory.sqlite").run(
                    "How should autoresearch stay reliable?",
                    source_paths=[str(source)],
                    out_dir=root / "run",
                    provider="openai",
                    include_memory=False,
                )
            self.assertEqual(run.answer, "Provider-authored answer with evidence IDs.")
            self.assertEqual(run.provenance["answer_author"], "provider_model")
            self.assertEqual(run.provenance["answer_provider"], "openai")
            report = (root / "run" / "report.md").read_text(encoding="utf-8")
            self.assertIn("model-authored synthesis (openai/test-model)", report)

    def test_provider_answer_prompt_sanitizes_malformed_ledger_rows(self):
        prompt = _answer_prompt(
            b"What happened?",
            [
                object(),
                SimpleNamespace(
                    id=b"c1",
                    text=b"Claim text",
                    citations="e1",
                    confidence="0.8",
                    support_score=float("nan"),
                    quality_flags=[" flag ", None],
                ),
                SimpleNamespace(id="c2", text="", confidence="bad"),
            ],
            [
                object(),
                SimpleNamespace(id="e1", source_id=b"s1", title=None, url=None, score="3.5", text=b"Evidence text"),
                SimpleNamespace(id="e2", source_id="s2", score="bad", text=""),
            ],
            ["gap", None, b"second gap"],
            [
                object(),
                SimpleNamespace(
                    id="d1",
                    statement=b"Discovery",
                    confidence="bad",
                    effect_size="1.2",
                    reproducibility=float("inf"),
                    novelty="0.4",
                    supporting_experiments="x1",
                ),
            ],
            [
                object(),
                SimpleNamespace(
                    id="x1",
                    probe=b"attention_pattern",
                    status="ran",
                    target={"layer": 6, "bad": object()},
                    effect_size="0.5",
                    baseline=None,
                    per_seed=["0.5", "bad"],
                    significant="yes",
                    reproduced="false",
                ),
                SimpleNamespace(id="x2", status="error", probe="bad"),
            ],
        )
        payload = json.loads(prompt.split("RUN LEDGER:\n", 1)[1])

        self.assertEqual(payload["question"], "What happened?")
        self.assertEqual(payload["claims"][0]["citations"], [])
        self.assertEqual(payload["claims"][0]["confidence"], 0.8)
        self.assertEqual(payload["claims"][0]["support_score"], 0.0)
        self.assertEqual(payload["evidence"][0]["score"], 3.5)
        self.assertEqual(payload["discoveries"][0]["supporting_experiments"], [])
        self.assertEqual(payload["experiments"][0]["target"]["layer"], 6)
        self.assertEqual(payload["experiments"][0]["per_seed"], [0.5, 0.0])
        self.assertFalse(payload["experiments"][0]["reproduced"])
        self.assertEqual(payload["gaps"], ["gap", "second gap"])

    def test_controller_sanitizes_malformed_sources_claims_and_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class FakeMemory:
                def __init__(self, path):
                    pass

                def recall_sources(self, question):
                    return [object(), SimpleNamespace(id=b"mem-src", title=None, text=b"Memory evidence text.")]

                def upsert_sources(self, sources):
                    self.sources = sources

                def record_run(self, run):
                    self.run = run

                def close(self):
                    pass

            class FakeAdapter:
                available = True

                def search_summary(self, question):
                    return SimpleNamespace(id=b"live-src", title=None, text=b"Live evidence text.", kind=b"live")

            class FakeExtractor:
                def extract(self, question, evidence, limit=24):
                    return [
                        object(),
                        Claim("claim-1", "Memory evidence supports the answer.", ["ev-1"], ["mem-src"], "0.8", "bad"),
                    ]

            class FakeCritic:
                def evaluate(self, question, plan, claims, evidence):
                    return [
                        b"gap one",
                        None,
                    ], [
                        SimpleNamespace(id=b"contradiction-1", claim_a=b"claim-1", claim_b=b"claim-2", reason=None, severity="bad"),
                        SimpleNamespace(id=b"bad"),
                    ], {"readiness_score": "0.7"}

            class FakeSynthesizer:
                def synthesize(self, question, claims, evidence, gaps):
                    return f"claims={len(claims)} gaps={','.join(gaps)}"

            with (
                patch("mechferret.controller.ResearchMemory", FakeMemory),
                patch("mechferret.controller.make_research_adapter", return_value=FakeAdapter()),
                patch("mechferret.controller.load_config", return_value=object()),
                patch("mechferret.controller.synthesize_answer_with_provider", return_value=("", {"reason": "skipped"})),
            ):
                engine = MechFerret(root / "memory.sqlite")
                engine.extractor = FakeExtractor()
                engine.critic = FakeCritic()
                engine.synthesizer = FakeSynthesizer()
                run = engine.run(
                    b"What does memory say?",
                    out_dir=root / "run",
                    max_rounds=1,
                    provider="openai",
                    include_memory=True,
                )

        self.assertEqual([source.id for source in run.sources], ["mem-src", "live-src"])
        self.assertEqual(run.gaps, ["gap one"])
        self.assertEqual([row.id for row in run.contradictions], ["contradiction-1"])
        self.assertEqual(run.answer, "claims=1 gaps=gap one")
        self.assertEqual(run.provenance["provider_source_added"], True)
        self.assertEqual(run.metrics["readiness_score"], 0.7)

    def test_dedupe_evidence_normalizes_malformed_rows(self):
        chunks = dedupe_evidence(
            [
                object(),
                SimpleNamespace(id=b"e1", source_id=b"s1", title=None, text=b"low", score="bad", highlights="bad"),
                EvidenceChunk("e1", "s1", "Title", "high", score="2.5", highlights=[" h ", None]),
                SimpleNamespace(id="", source_id="s2", text="skip", score=9),
            ]
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].id, "e1")
        self.assertEqual(chunks[0].score, 2.5)
        self.assertEqual(chunks[0].highlights, ["h"])

    def test_live_research_adapters_sanitize_malformed_boundaries(self):
        from mechferret.config import MechFerretConfig, ProviderSettings
        from mechferret.llm import AnthropicResearch, OpenAIWebResearch

        openai_calls = {}

        class FakeOpenAIClient:
            def __init__(self, api_key):
                self.responses = self

            def create(self, **kwargs):
                openai_calls.update(kwargs)
                return SimpleNamespace(output_text=b"OpenAI live finding")

        anthropic_calls = {}

        class FakeAnthropicClient:
            def __init__(self, api_key):
                self.messages = self

            def create(self, **kwargs):
                anthropic_calls.update(kwargs)
                return SimpleNamespace(content=[object(), SimpleNamespace(text=b"Anthropic brief"), SimpleNamespace(text=None)])

        config = MechFerretConfig(
            default_provider="openai",
            providers={
                "openai": ProviderSettings(api_key="openai-key", model="openai-model"),
                "anthropic": ProviderSettings(api_key="anthropic-key", model="anthropic-model"),
            },
        )
        with patch.dict(
            sys.modules,
            {
                "openai": SimpleNamespace(OpenAI=FakeOpenAIClient),
                "anthropic": SimpleNamespace(Anthropic=FakeAnthropicClient),
            },
        ):
            openai_source = OpenAIWebResearch(config=config).search_summary(
                b"What is new?", allowed_domains=[b"example.com", None, " docs.example "]
            )
            anthropic_source = AnthropicResearch(config=config).search_summary(b"What is new?")

        self.assertEqual(openai_source.text, "OpenAI live finding")
        self.assertEqual(openai_calls["tools"][0]["filters"]["allowed_domains"], ["example.com", "docs.example"])
        self.assertIn("Question: What is new?", openai_calls["input"])
        self.assertEqual(anthropic_source.text, "Anthropic brief")
        self.assertIn("Question: What is new?", anthropic_calls["messages"][0]["content"])

    def test_live_research_adapters_report_empty_or_provider_errors(self):
        from mechferret.config import MechFerretConfig, ProviderSettings
        from mechferret.llm import OpenAIWebResearch

        class FailingOpenAIClient:
            def __init__(self, api_key):
                self.responses = self

            def create(self, **kwargs):
                raise RuntimeError("provider down")

        config = MechFerretConfig(
            default_provider="openai",
            providers={"openai": ProviderSettings(api_key="openai-key", model="openai-model")},
        )
        adapter = OpenAIWebResearch(config=config)
        self.assertIsNone(adapter.search_summary([]))
        self.assertEqual(adapter.last_diagnostic["reason"], "empty question")
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FailingOpenAIClient)}):
            self.assertIsNone(adapter.search_summary("Need current context"))
        self.assertIn("provider down", adapter.last_diagnostic["reason"])
        self.assertEqual(adapter.last_diagnostic["provider"], "openai")

    def test_provider_research_failure_without_sources_fails_closed(self):
        from mechferret.config import MechFerretConfig, ProviderSettings, save_config

        class FailingOpenAIClient:
            def __init__(self, api_key):
                self.responses = self

            def create(self, **kwargs):
                raise RuntimeError("provider down")

        old_config = os.environ.get("MECHFERRET_CONFIG")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            os.environ["MECHFERRET_CONFIG"] = str(config_path)
            save_config(
                MechFerretConfig(
                    default_provider="openai",
                    providers={"openai": ProviderSettings(api_key="openai-key", model="openai-model")},
                ),
                config_path,
            )
            try:
                with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FailingOpenAIClient)}):
                    with self.assertRaisesRegex(ValueError, "Live provider research failed for openai: provider down"):
                        MechFerret(root / "memory.sqlite").run(
                            "Need current context",
                            provider="openai",
                            include_memory=False,
                            out_dir=root / "run",
                        )
            finally:
                if old_config is None:
                    os.environ.pop("MECHFERRET_CONFIG", None)
                else:
                    os.environ["MECHFERRET_CONFIG"] = old_config


if __name__ == "__main__":
    unittest.main()
