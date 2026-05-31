import unittest
import json
import os
import re
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from mechferret.agents import Synthesizer
from mechferret.models import Claim, Discovery, EvidenceChunk, ExperimentResult, ResearchPlan, ResearchRun, Source
from mechferret.ops import run_quickstart, verify_run_artifacts
from mechferret.paper import compile_tex, draft_latex_with_model, latex_from_run, review_paper, write_paper_from_artifact
from mechferret.report import claim_graph, html_report, markdown_report, run_evals, write_artifacts


class ReportingTest(unittest.TestCase):
    def test_citation_labels_are_unique_for_multiple_chunks_per_source(self):
        evidence = [
            EvidenceChunk("c1", "s1", "Long Source", "first"),
            EvidenceChunk("c2", "s1", "Long Source", "second"),
            EvidenceChunk("c3", "s2", "Short Source", "third"),
        ]
        labels = Synthesizer().citation_labels(evidence)
        self.assertEqual(labels["c1"], "S1.1")
        self.assertEqual(labels["c2"], "S1.2")
        self.assertEqual(labels["c3"], "S2")
        self.assertEqual(len(set(labels.values())), 3)

    def test_html_embeds_parseable_json_payload(self):
        run = ResearchRun(
            run_id="run_test",
            question="Can <script> break JSON?",
            created_at="2026-05-30T00:00:00+00:00",
            plan=ResearchPlan("Can <script> break JSON?", [], "test"),
            sources=[],
            evidence=[],
            claims=[],
            contradictions=[],
            gaps=[],
            answer="No </script> breakage.",
            metrics={},
        )
        rendered = html_report(run)
        payload = re.search(r'<script id="run-json" type="application/json">(.*?)</script>', rendered, re.S)
        self.assertIsNotNone(payload)
        parsed = json.loads(payload.group(1))
        self.assertEqual(parsed["run_id"], "run_test")

    def test_reports_tolerate_malformed_numeric_fields(self):
        run = ResearchRun(
            run_id="run_bad_numbers",
            question="Can bad scores break reports?",
            created_at="2026-05-30T00:00:00+00:00",
            plan=ResearchPlan("Can bad scores break reports?", [], "test"),
            sources=[],
            evidence=[EvidenceChunk("e1", "s1", "Source", "Evidence text", score="bad")],  # type: ignore[arg-type]
            claims=[Claim("c1", "Claim text", ["e1"], ["s1"], "bad", "bad")],  # type: ignore[arg-type]
            contradictions=[],
            gaps=[],
            answer="Answer",
            metrics={
                "readiness_score": "bad",
                "source_diversity": "3",
                "citation_density": "bad",
                "plan_coverage": None,
                "contradiction_pressure": "bad",
            },
            experiments=[
                ExperimentResult(
                    "x1",
                    "s1",
                    "head_ablation",
                    "ran",
                    "bad",  # type: ignore[arg-type]
                    "bad",  # type: ignore[arg-type]
                    significant=True,
                    reproduced=True,
                    target={"layer": object()},
                )
            ],
            discoveries=[
                Discovery("d1", "Discovery", "bad", "bad", "bad", "bad", ["x1"])  # type: ignore[arg-type]
            ],
            mode="discovery",
        )
        self.assertIn("**Readiness score:** 0.00", markdown_report(run))
        self.assertIn("run_bad_numbers", html_report(run))
        self.assertFalse(run_evals(run)["passed"])

    def test_report_outputs_tolerate_malformed_rows(self):
        run = ResearchRun(
            run_id="run_bad_rows",
            question=b"Can malformed rows break artifact output?",  # type: ignore[arg-type]
            created_at=b"2026-05-30T00:00:00+00:00",  # type: ignore[arg-type]
            plan=SimpleNamespace(steps=["bad", SimpleNamespace(intent=b"map", question=b"What changed?")]),  # type: ignore[arg-type]
            sources=[
                Source("s1", "Source", "Evidence text", url="https://example.com/a"),
                "bad",
            ],
            evidence=[
                SimpleNamespace(id="e1", source_id="s1", title=b"Evidence", text=b"Evidence text", url=object(), score=float("nan")),
                "bad",
            ],  # type: ignore[list-item]
            claims=[
                SimpleNamespace(
                    id="c1",
                    text=b"Claim text",
                    citations=["e1", object()],
                    source_ids="s1",
                    confidence=float("inf"),
                    support_score=object(),
                    quality_flags="bad",
                ),
                "bad",
            ],  # type: ignore[list-item]
            contradictions=[
                SimpleNamespace(id="k1", claim_a=b"c1", claim_b=object(), reason=b"mixed evidence", severity=float("nan")),
                "bad",
            ],  # type: ignore[list-item]
            gaps=[b"Need a stricter baseline.", object()],  # type: ignore[list-item]
            answer=b"Answer",  # type: ignore[arg-type]
            metrics={1: {"nested": object()}, "readiness_score": float("inf")},
            artifacts={"bad": {"not": "a path"}},  # type: ignore[dict-item]
            provenance=["bad"],  # type: ignore[arg-type]
            hypotheses=[SimpleNamespace(id="h1", statement=b"Hypothesis", status=b"open", confidence=float("nan"))],  # type: ignore[list-item]
            experiments=[
                SimpleNamespace(
                    status="ran",
                    probe=b"head_ablation",
                    target={"obj": object()},
                    effect_size=float("nan"),
                    baseline=None,
                    significant="false",
                    reproduced=True,
                    backend_used=b"synthetic",
                )
            ],  # type: ignore[list-item]
            discoveries=[
                SimpleNamespace(
                    id="d1",
                    statement=b"Discovery",
                    confidence=float("nan"),
                    novelty=object(),
                    effect_size=float("inf"),
                    reproducibility="1",
                    supporting_experiments="x1",
                    claim_ids="c1",
                    hypothesis_id=object(),
                ),
                "bad",
            ],  # type: ignore[list-item]
            mode="discovery",
        )

        rendered_md = markdown_report(run)
        rendered_html = html_report(run)
        graph = claim_graph(run)
        evals = run_evals(run)
        self.assertIn("run_bad_rows", rendered_html)
        self.assertIn("Claim text", rendered_md)
        self.assertEqual(graph["nodes"][0]["id"], "s1")
        self.assertFalse(evals["passed"])
        payload = re.search(r'<script id="run-json" type="application/json">(.*?)</script>', rendered_html, re.S)
        self.assertIsNotNone(payload)
        json.loads(payload.group(1))

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_artifacts(run, tmp)
            self.assertTrue(Path(artifacts["manifest"]).exists())
            json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))
            json.loads(Path(artifacts["graph"]).read_text(encoding="utf-8"))

    def test_latex_from_run_writes_paper_sections_and_escapes_text(self):
        run = ResearchRun(
            run_id="run_tex",
            question="Does layer_5 head copy A&B?",
            created_at="2026-05-30T00:00:00+00:00",
            plan=ResearchPlan("Does layer_5 head copy A&B?", [], "ablate and patch"),
            sources=[],
            evidence=[],
            claims=[
                Claim("c1", "Layer_5 head increases A&B logit diff by 20%.", ["e1"], ["s1"], 0.9, 0.8)
            ],
            contradictions=[],
            gaps=["Needs stronger baseline_2."],
            answer="a",
            metrics={"readiness_score": 0.8},
            experiments=[
                ExperimentResult(
                    "x1", "s1", "head_ablation", "ran", 1.2, 0.1,
                    per_seed=[1.1, 1.2], significant=True, reproduced=True,
                    target={"layer": 5, "head": 1},
                )
            ],
            discoveries=[
                Discovery("d1", "Layer_5 head copies A&B.", 0.9, 0.7, 1.2, 1.0, ["x1", "x2"])
            ],
            mode="discovery",
        )
        tex = latex_from_run(run)
        self.assertIn("\\section{Results}", tex)
        self.assertIn("\\section{Experiment Ledger}", tex)
        self.assertIn("\\section{Evidence Ledger}", tex)
        self.assertIn("layer\\_5", tex)
        self.assertIn("A\\&B", tex)
        self.assertIn("\\section{Limitations}", tex)
        self.assertIn("Layer\\_5 head copies", tex)
        self.assertIn("Needs stronger baseline\\_2.", tex)
        self.assertNotIn("TODO:", tex)

    def test_paper_generation_sanitizes_malformed_run_fields(self):
        from mechferret import paper as paper_mod

        captured = {}
        original_provider = paper_mod._paper_provider
        original_call = paper_mod._call_openai
        try:
            paper_mod._paper_provider = lambda provider, model: ("openai", "paper-model", "key")

            def fake_call(model, key, prompt):
                captured["prompt"] = prompt
                return "\\documentclass{article}\\begin{document}ok\\end{document}"

            paper_mod._call_openai = fake_call
            run = ResearchRun(
                run_id=b"run_paper_bad",  # type: ignore[arg-type]
                question=b"does layer_1 copy A&B?",  # type: ignore[arg-type]
                created_at="2026-05-30T00:00:00+00:00",
                plan=ResearchPlan("q", [], "s"),
                sources=[],
                evidence=[],
                claims=[
                    SimpleNamespace(
                        id=b"c1",
                        text=b"Claim",
                        citations=["e1", object()],
                        confidence=float("inf"),
                        quality_flags="bad",
                    ),
                    "bad",
                ],  # type: ignore[list-item]
                contradictions=[],
                gaps=[b"Needs stronger baseline.", object()],  # type: ignore[list-item]
                answer="a",
                metrics={1: {"nested": object()}, "readiness_score": float("nan")},
                experiments=[
                    SimpleNamespace(
                        id=b"x1",
                        status="ran",
                        probe=b"head_ablation",
                        target={"obj": object()},
                        effect_size=float("inf"),
                        baseline=None,
                        per_seed=[1.0, "bad", float("nan")],
                        significant="true",
                        reproduced="false",
                        backend_used=b"synthetic",
                    )
                ],  # type: ignore[list-item]
                discoveries=[
                    SimpleNamespace(
                        id=b"d1",
                        statement=b"Discovery",
                        confidence=float("nan"),
                        effect_size=object(),
                        reproducibility="1",
                        novelty=float("inf"),
                        supporting_experiments="x1",
                    )
                ],  # type: ignore[list-item]
                mode="discovery",
            )

            latex = draft_latex_with_model(run, provider=["openai"], model=object())  # type: ignore[arg-type]
            self.assertIn("\\documentclass", latex)
            json_blob = captured["prompt"].split("RUN EVIDENCE JSON:\n", 1)[1].split("\n\nSTRUCTURE", 1)[0]
            parsed = json.loads(json_blob)
            self.assertEqual(parsed["claims"][0]["confidence"], 0.0)
            self.assertTrue(parsed["experiments"][0]["target"]["obj"].startswith("<object object"))
            self.assertEqual(parsed["discoveries"][0]["supporting_experiments"], [])
            self.assertIn("layer\\_1", latex_from_run(run))
        finally:
            paper_mod._paper_provider = original_provider
            paper_mod._call_openai = original_call

    def test_paper_artifact_recording_repairs_bad_artifact_map(self):
        run = ResearchRun(
            run_id="run_file",
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "run.json"
            payload = run.to_dict()
            payload["artifacts"] = ["bad"]
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            result = write_paper_from_artifact(artifact, out_dir=object(), compile_pdf="false", provider=["local"])  # type: ignore[arg-type]
            written = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(Path(result["tex"]), root / "paper" / "main.tex")
            self.assertIsInstance(written["artifacts"], dict)
            self.assertEqual(Path(written["artifacts"]["paper"]), root / "paper" / "main.tex")
            self.assertTrue(Path(written["artifacts"]["manifest"]).exists())
            self.assertNotIn("compiled", result)

        self.assertFalse(compile_tex(object())["compiled"])

    def test_compile_tex_times_out_with_actionable_note(self):
        from mechferret import paper as paper_mod

        original_which = paper_mod.shutil.which
        original_run = paper_mod.subprocess.run
        try:
            paper_mod.shutil.which = lambda name: "/usr/bin/tectonic" if name == "tectonic" else None

            def fake_run(*args, **kwargs):
                self.assertEqual(kwargs["timeout"], 3)
                raise paper_mod.subprocess.TimeoutExpired(args[0], timeout=kwargs["timeout"], stderr="still compiling")

            paper_mod.subprocess.run = fake_run
            with tempfile.TemporaryDirectory() as tmp:
                tex = Path(tmp) / "main.tex"
                tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}", encoding="utf-8")
                result = compile_tex(tex, timeout=3)
            self.assertFalse(result["compiled"])
            self.assertIn("timed out after 3s", result["note"])
            self.assertIn("still compiling", result["stderr"])
        finally:
            paper_mod.shutil.which = original_which
            paper_mod.subprocess.run = original_run

    def test_write_paper_from_artifact_local_mode_writes_scaffold(self):
        run = ResearchRun(
            run_id="run_file",
            question="Does a head copy names?",
            created_at="2026-05-30T00:00:00+00:00",
            plan=ResearchPlan("Does a head copy names?", [], "ablate"),
            sources=[],
            evidence=[],
            claims=[],
            contradictions=[],
            gaps=[],
            answer="a",
            metrics={"readiness_score": 0.5},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "run.json"
            artifact.write_text(json.dumps(run.to_dict()), encoding="utf-8")
            result = write_paper_from_artifact(artifact, out_dir=root / "paper", provider="local")
            tex = Path(result["tex"])
            self.assertTrue(tex.exists())
            self.assertTrue(result["ok"])
            self.assertEqual(result["mode"], "scaffold")
            self.assertEqual(result["artifacts"]["paper"], str(tex))
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["artifacts"]["paper"]), tex)
            self.assertTrue(Path(payload["artifacts"]["manifest"]).exists())
            self.assertTrue(verify_run_artifacts(artifact)["passed"])
            self.assertIn("\\documentclass{article}", tex.read_text(encoding="utf-8"))

    def test_cli_paper_can_print_json_result(self):
        from mechferret.cli import main

        run = ResearchRun(
            run_id="run_file",
            question="Does a head copy names?",
            created_at="2026-05-30T00:00:00+00:00",
            plan=ResearchPlan("Does a head copy names?", [], "ablate"),
            sources=[],
            evidence=[],
            claims=[],
            contradictions=[],
            gaps=[],
            answer="a",
            metrics={"readiness_score": 0.5},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "run.json"
            artifact.write_text(json.dumps(run.to_dict()), encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                main(["paper", str(artifact), "--provider", "local", "--json"])
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(Path(payload["tex"]), root / "paper" / "main.tex")
            self.assertEqual(payload["artifacts"]["paper"], payload["tex"])

    def test_cli_paper_json_reports_missing_run_without_traceback(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = StringIO()
            err = StringIO()
            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(out), redirect_stderr(err):
                    main(["paper", "--runs-root", str(root / "missing-runs"), "--provider", "local", "--json"])
            self.assertEqual(ctx.exception.code, 1)
            payload = json.loads(out.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"], "no run artifact found")
            self.assertEqual(payload["runs_root"], str(root / "missing-runs"))
            self.assertTrue(payload["next_actions"])
            self.assertEqual(err.getvalue(), "")

    def test_cli_paper_json_reports_missing_explicit_run(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "run.json"
            out = StringIO()
            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(out):
                    main(["paper", str(missing), "--provider", "local", "--json"])
            self.assertEqual(ctx.exception.code, 1)
            payload = json.loads(out.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"], "run artifact not found")
            self.assertEqual(Path(payload["path"]), missing)

    def test_write_paper_defaults_to_run_sibling_paper_dir(self):
        run = ResearchRun(
            run_id="run_file",
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "runs" / "demo" / "run.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(json.dumps(run.to_dict()), encoding="utf-8")
            result = write_paper_from_artifact(artifact, provider="local")
            self.assertEqual(Path(result["tex"]), artifact.parent / "paper" / "main.tex")
            self.assertTrue((artifact.parent / "paper" / "main.tex").exists())
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["artifacts"]["paper"]), artifact.parent / "paper" / "main.tex")

    def test_write_paper_records_compiled_pdf_when_available(self):
        from mechferret import paper as paper_mod

        run = ResearchRun(
            run_id="run_file",
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
        original_compile = paper_mod.compile_tex
        try:
            def fake_compile(tex, *, timeout=60):
                self.assertEqual(timeout, 4)
                pdf = Path(tex).with_suffix(".pdf")
                pdf.write_bytes(b"%PDF fake")
                return {"pdf": str(pdf), "compiled": True, "stderr": ""}

            paper_mod.compile_tex = fake_compile
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifact = root / "runs" / "demo" / "run.json"
                artifact.parent.mkdir(parents=True)
                artifact.write_text(json.dumps(run.to_dict()), encoding="utf-8")
                result = write_paper_from_artifact(artifact, provider="local", compile_pdf=True, compile_timeout=4)
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                self.assertEqual(Path(payload["artifacts"]["paper"]), Path(result["tex"]))
                self.assertEqual(Path(payload["artifacts"]["pdf"]), Path(result["pdf"]))
                self.assertTrue(Path(result["pdf"]).exists())
        finally:
            paper_mod.compile_tex = original_compile

    def test_repl_paper_requires_run_artifact(self):
        from mechferret.repl import _paper

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                out = StringIO()
                with redirect_stdout(out):
                    _paper(SimpleNamespace(configured=False), SimpleNamespace(goal="g"), [])
                self.assertIn("no run artifact", out.getvalue())
                self.assertFalse((Path(tmp) / "paper" / "main.tex").exists())
            finally:
                os.chdir(cwd)

    def test_repl_paper_and_status_honor_run_selection(self):
        from mechferret.repl import _paper, _status

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

            with redirect_stdout(StringIO()) as paper_out:
                _paper(
                    SimpleNamespace(configured=False),
                    SimpleNamespace(goal="g"),
                    ["--runs-root", str(root / "runs"), "--select", "best", "--provider", "local"],
                )
            self.assertIn(f"Source run: {good}", paper_out.getvalue())

            with redirect_stdout(StringIO()) as status_out:
                _status(["--runs-root", str(root / "runs"), "--select", "best"])
            self.assertIn(f"Selected run: {good}", status_out.getvalue())

    def test_repl_review_paper_finds_latest_run_bound_paper(self):
        from mechferret.repl import _review_paper

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                paper = Path("runs/demo/paper/main.tex")
                paper.parent.mkdir(parents=True)
                paper.write_text("\\documentclass{article}\\begin{document}x\\end{document}", encoding="utf-8")
                (Path("runs/demo/run.json")).write_text(
                    json.dumps(
                        ResearchRun(
                            run_id="run_file",
                            question="Does a head copy names?",
                            created_at="2026-05-30T00:00:00+00:00",
                            plan=ResearchPlan("Does a head copy names?", [], "ablate"),
                            sources=[],
                            evidence=[],
                            claims=[],
                            contradictions=[],
                            gaps=[],
                            answer="a",
                            metrics={},
                        ).to_dict()
                    ),
                    encoding="utf-8",
                )
                out = StringIO()
                with redirect_stdout(out):
                    _review_paper(SimpleNamespace(configured=False), [])
                self.assertIn("review not available", out.getvalue())
                self.assertNotIn("no paper at", out.getvalue())
            finally:
                os.chdir(cwd)

    def test_review_paper_without_provider_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "main.tex"
            paper.write_text("\\documentclass{article}\\begin{document}x\\end{document}", encoding="utf-8")
            result = review_paper(paper, provider="local")
            self.assertFalse(result["ok"])
            self.assertIn("login", " ".join(result["next_actions"]))

    def test_review_paper_missing_target_preserves_selection_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = review_paper(None, runs_root=root / "runs", selection="ready", provider="openai")
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "paper artifact not found")
            self.assertEqual(result["runs_root"], str(root / "runs"))
            self.assertEqual(result["selection"], "ready")
            self.assertEqual(result["target"], "paper")
            self.assertTrue(result["next_actions"])

    def test_review_paper_missing_explicit_target_reports_requested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "paper" / "main.tex"
            result = review_paper(missing, provider="openai")
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "paper artifact not found")
            self.assertEqual(Path(result["requested_path"]), missing)

    def test_review_paper_writes_provider_review(self):
        from mechferret import paper as paper_mod
        from mechferret.ops import resolve_artifact

        original_provider = paper_mod._paper_provider
        original_call = paper_mod._call_openai
        try:
            paper_mod._paper_provider = lambda provider, model: ("openai", "test-model", "key")
            paper_mod._call_openai = lambda model, key, prompt: "Soundness: 7\nOverall: 7\nRecommendation: Borderline"
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_json = root / "runs" / "demo" / "run.json"
                tex = root / "runs" / "demo" / "paper" / "main.tex"
                tex.parent.mkdir(parents=True)
                run_json.write_text(
                    json.dumps(
                        ResearchRun(
                            run_id="run_file",
                            question="Does a head copy names?",
                            created_at="2026-05-30T00:00:00+00:00",
                            plan=ResearchPlan("Does a head copy names?", [], "ablate"),
                            sources=[],
                            evidence=[],
                            claims=[],
                            contradictions=[],
                            gaps=[],
                            answer="a",
                            metrics={},
                            artifacts={"paper": str(tex)},
                        ).to_dict()
                    ),
                    encoding="utf-8",
                )
                tex.write_text(
                    "\\documentclass{article}\\begin{document}"
                    "\\section{Results}Evidence."
                    "\\section{Experiment Ledger}Experiments."
                    "\\section{Evidence Ledger}Citations."
                    "\\section{Limitations}Gaps."
                    "\\end{document}",
                    encoding="utf-8",
                )
                result = review_paper(tex, provider="openai")
                self.assertTrue(result["ok"])
                self.assertEqual(result["model"], "test-model")
                self.assertTrue((tex.parent / "review.md").exists())
                self.assertEqual(Path(result["artifacts"]["review"]), tex.parent / "review.md")
                payload = json.loads(run_json.read_text(encoding="utf-8"))
                self.assertEqual(Path(payload["artifacts"]["review"]), tex.parent / "review.md")
                self.assertTrue(verify_run_artifacts(run_json)["passed"])
                resolved = resolve_artifact("review", runs_root=root / "runs")
                self.assertTrue(resolved["exists"])
                self.assertEqual(Path(resolved["path"]), tex.parent / "review.md")
                alias = resolve_artifact("review_md", runs_root=root / "runs")
                self.assertTrue(alias["exists"])
                self.assertEqual(Path(alias["path"]), tex.parent / "review.md")
                self.assertIn("Recommendation", result["review"])
        finally:
            paper_mod._paper_provider = original_provider
            paper_mod._call_openai = original_call

    def test_review_paper_selects_best_run_bound_paper(self):
        from mechferret import paper as paper_mod
        from mechferret.ops import run_quickstart

        original_provider = paper_mod._paper_provider
        original_call = paper_mod._call_openai
        try:
            paper_mod._paper_provider = lambda provider, model: ("openai", "test-model", "key")
            paper_mod._call_openai = lambda model, key, prompt: "Soundness: 8\nOverall: 8\nRecommendation: Accept"
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

                latest = review_paper(None, runs_root=root / "runs", selection="latest", provider="openai")
                best = review_paper(None, runs_root=root / "runs", selection="best", provider="openai")
                self.assertFalse(latest["ok"])
                self.assertTrue(best["ok"])
                self.assertEqual(Path(best["path"]), root / "runs" / "good" / "paper" / "main.tex")
                self.assertTrue((root / "runs" / "good" / "paper" / "review.md").exists())
        finally:
            paper_mod._paper_provider = original_provider
            paper_mod._call_openai = original_call

    def test_review_paper_cli_reports_missing_provider(self):
        from mechferret.cli import main
        from mechferret import paper as paper_mod

        original_provider = paper_mod._paper_provider
        with tempfile.TemporaryDirectory() as tmp:
            try:
                paper_mod._paper_provider = lambda provider, model: ("", "", "")
                tex = Path(tmp) / "main.tex"
                tex.write_text("\\documentclass{article}\\begin{document}Evidence.\\end{document}", encoding="utf-8")
                out = StringIO()
                with self.assertRaises(SystemExit) as ctx:
                    with redirect_stdout(out):
                        main(["review-paper", str(tex), "--provider", "openai", "--json"])
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn('"ok": false', out.getvalue())
            finally:
                paper_mod._paper_provider = original_provider


if __name__ == "__main__":
    unittest.main()
