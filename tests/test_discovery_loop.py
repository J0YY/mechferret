import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mechferret.discovery import DiscoveryController, request_alignment_issue
from mechferret.hooks import Budget, BudgetGuard
from mechferret.coordinator import Coordinator, default_workers
from mechferret.interp.critic import ExperimentCritic
from mechferret.interp.hypotheses import HypothesisGenerator, classify_head_role, update_hypotheses
from mechferret.interp.engine import InterpEngine
from mechferret.models import Claim, EvidenceChunk, ExperimentResult, ExperimentSpec, Hypothesis, Source
from mechferret.ops import verify_run_artifacts
from mechferret.provenance import refresh_run_manifest
from mechferret.skills import list_skills, load_skill


class CoordinatorHooksTest(unittest.TestCase):
    def test_coordinator_preserves_order_serial_and_parallel(self):
        items = list(range(20))
        self.assertEqual(Coordinator(1).map(lambda x: x * x, items), [x * x for x in items])
        self.assertEqual(Coordinator(8).map(lambda x: x * x, items), [x * x for x in items])

    def test_coordinator_tolerates_malformed_boundary_values(self):
        self.assertEqual(Coordinator("bad").max_workers, 1)
        self.assertEqual(Coordinator(0).max_workers, 1)
        self.assertEqual(Coordinator(True).max_workers, 1)
        self.assertEqual(Coordinator(4).map(lambda x: x + 1, "not items"), [])
        self.assertEqual(Coordinator(4).map("not callable", [1, 2, 3]), [])
        self.assertEqual(Coordinator(4).map(lambda x: x * 2, (1, 2)), [2, 4])
        self.assertEqual(default_workers([]), 1)
        self.assertEqual(default_workers(b"modal"), 8)

    def test_budget_guard_admits_and_exhausts(self):
        guard = BudgetGuard(Budget(max_experiments=5, max_rounds=2))
        admitted = guard.admit(list(range(10)))
        self.assertEqual(len(admitted), 5)
        self.assertTrue(guard.notices)
        guard.start_round()
        guard.start_round()
        exhausted, reason = guard.exhausted()
        self.assertTrue(exhausted)
        self.assertIn("max_rounds", reason)


class SkillTest(unittest.TestCase):
    def test_skills_load(self):
        skills = {s.name for s in list_skills()}
        self.assertIn("ioi-circuit", skills)
        skill = load_skill("ioi-circuit")
        self.assertEqual(skill.task, "ioi")
        self.assertEqual(skill.to_budget().max_experiments, 400)

    def test_unknown_skill_raises(self):
        with self.assertRaises(KeyError):
            load_skill("does-not-exist")

    def test_skill_loading_skips_malformed_playbooks_and_sanitizes_fields(self):
        from mechferret import skills as skills_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad-json.json").write_text("{", encoding="utf-8")
            (root / "missing-task.json").write_text(json.dumps({"name": "missing-task"}), encoding="utf-8")
            (root / "good.json").write_text(
                json.dumps(
                    {
                        "name": " good ",
                        "description": 123,
                        "task": " ioi ",
                        "model": "",
                        "question": ["bad"],
                        "max_screen_heads": 0,
                        "promote_top_k": "bad",
                        "seeds": [0, "2", -1, "bad"],
                        "budget": {
                            "max_experiments": "bad",
                            "max_rounds": 0,
                            "max_gpu_seconds": -1,
                            "max_wall_seconds": "12.5",
                            "allow_gpu": "false",
                            "allow_network": "yes",
                        },
                        "stop": {"min_confirmed_mechanisms": "bad", "min_rigor_score": "bad"},
                        "references": [" ref ", 3],
                    }
                ),
                encoding="utf-8",
            )
            old_dir = skills_mod.SKILLS_DIR
            skills_mod.SKILLS_DIR = root
            try:
                listed = skills_mod.list_skills()
                self.assertEqual([skill.name for skill in listed], ["good"])
                skill = skills_mod.load_skill("good")
                self.assertEqual(skill.task, "ioi")
                self.assertEqual(skill.model, "")
                self.assertEqual(skill.max_screen_heads, 96)
                self.assertEqual(skill.promote_top_k, 5)
                self.assertEqual(skill.seeds, [0, 2])
                self.assertEqual(skill.references, ["ref"])
                self.assertEqual(skill.min_confirmed, 1)
                self.assertEqual(skill.min_rigor, 0.6)
                budget = skill.to_budget()
                self.assertEqual(budget.max_experiments, 400)
                self.assertEqual(budget.max_rounds, 4)
                self.assertEqual(budget.max_gpu_seconds, 900.0)
                self.assertEqual(budget.max_wall_seconds, 12.5)
                self.assertFalse(budget.allow_gpu)
                self.assertTrue(budget.allow_network)
                with self.assertRaisesRegex(ValueError, "invalid skill"):
                    skills_mod.load_skill(str(root / "bad-json.json"))
            finally:
                skills_mod.SKILLS_DIR = old_dir


class HypothesisFlowTest(unittest.TestCase):
    def test_screen_then_promote_confirms_a_head(self):
        engine = InterpEngine("gpt2", "synthetic")
        gen = HypothesisGenerator("gpt2")
        hyps, specs = gen.screen("find ioi", "ioi")
        self.assertGreater(len(specs), 10)
        results = engine.run_specs(specs)
        by_id = {r.spec_id: r for r in results}
        new_hyps, tri = gen.promote(results, "ioi", top_k=5)
        self.assertTrue(new_hyps)
        by_id.update({r.spec_id: r for r in engine.run_specs(tri)})
        all_hyps = hyps + new_hyps
        update_hypotheses(all_hyps, by_id)
        confirmed = [h for h in all_hyps if h.status == "confirmed" and "head" in h.target]
        self.assertGreaterEqual(len(confirmed), 1)

    def test_hypothesis_generation_normalizes_malformed_boundary_values(self):
        gen = HypothesisGenerator(["bad"], seeds=["2", "bad", -1, True, 2])
        with self.assertRaisesRegex(ValueError, "model is required"):
            gen.screen(b"find ioi", [], max_heads="bad", source_ids=[" src ", None, b"bytes"])
        self.assertEqual(gen.model, "")
        self.assertEqual(gen.seeds, [2])

    def test_promote_skips_malformed_screen_results(self):
        gen = HypothesisGenerator("gpt2")
        good = ExperimentResult(
            id="good",
            spec_id="spec-good",
            probe="head_ablation",
            status="ran",
            effect_size="1.5",
            baseline=0.0,
            significant=True,
            reproduced=True,
            target={"layer": "6", "head": "3"},
        )
        bad_truthy_flags = SimpleNamespace(
            probe="head_ablation",
            status="ran",
            effect_size=9.0,
            significant="false",
            reproduced="true",
            target={"layer": 7, "head": 2},
        )
        bad_target = SimpleNamespace(
            probe="head_ablation",
            status="ran",
            effect_size=8.0,
            significant=True,
            reproduced=True,
            target=[],
        )

        hyps, specs = gen.promote([bad_truthy_flags, bad_target, good], "ioi", top_k="bad", source_ids=[" src "])

        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].target["layer"], 6)
        self.assertEqual(hyps[0].target["head"], 3)
        self.assertEqual(hyps[0].target["screen_effect"], 1.5)
        self.assertEqual(hyps[0].source_ids, ["src"])
        self.assertEqual(len(specs), 3)
        self.assertTrue(all(spec.task == "ioi" for spec in specs))

        with self.assertRaisesRegex(KeyError, "Unknown interpretability task"):
            gen.promote([good], [], top_k="bad", source_ids=[" src "])

    def test_update_and_classify_tolerate_malformed_rows(self):
        hyp = Hypothesis(
            id="hyp",
            statement="head",
            rationale="",
            task="ioi",
            predicted_effect="",
            target={"layer": 6, "head": 3},
            experiment_ids=["attn", "dla", None],
        )
        by_id = {
            "attn": SimpleNamespace(
                id="attn",
                probe="attention_pattern",
                status="ran",
                significant="yes",
                reproduced=True,
                metrics={"duplicate_token": "0.7", "induction": "bad"},
                effect_size=0.0,
            ),
            "dla": SimpleNamespace(
                id="dla",
                probe="direct_logit_attribution",
                status="ran",
                significant=True,
                reproduced="true",
                metrics=[],
                effect_size="-1.2",
            ),
        }

        update_hypotheses([None, hyp], by_id)

        self.assertEqual(hyp.status, "confirmed")
        self.assertEqual(classify_head_role(list(by_id.values())), "duplicate-token / name-mover head")
        self.assertEqual(
            classify_head_role([SimpleNamespace(probe="direct_logit_attribution", effect_size="-0.5")]),
            "suppressor (negative) head",
        )

    def test_experiment_critic_tolerates_malformed_rows(self):
        confirmed = Hypothesis(
            id="confirmed",
            statement="head",
            rationale="",
            task="ioi",
            predicted_effect="",
            target={"layer": 6, "head": 3},
            status="confirmed",
            experiment_ids=["attn", "dla"],
        )
        inconclusive = SimpleNamespace(
            id=b"inc",
            statement=b"needs another probe",
            status="inconclusive",
            target=[],
            experiment_ids="bad",
        )
        rows = [
            object(),
            SimpleNamespace(
                id="attn",
                spec_id="spec-attn",
                probe="attention_pattern",
                status="ran",
                baseline=None,
                significant="yes",
                reproduced=True,
            ),
            SimpleNamespace(
                id="dla",
                spec_id="spec-dla",
                probe="direct_logit_attribution",
                status="ran",
                baseline=0.0,
                significant=True,
                reproduced="true",
            ),
            SimpleNamespace(
                id="bad",
                spec_id="bad",
                probe="activation_patching",
                status="ran",
                baseline=0.0,
                significant="false",
                reproduced="true",
            ),
            SimpleNamespace(status="error", baseline=0.0, significant=True, reproduced=True),
        ]

        gaps, metrics = ExperimentCritic().evaluate([object(), confirmed, inconclusive], rows)

        self.assertEqual(metrics["confirmed_mechanisms"], 1.0)
        self.assertEqual(metrics["significant_effects"], 2.0)
        self.assertEqual(metrics["reproduced_effects"], 2.0)
        self.assertEqual(metrics["experiments_errored"], 1.0)
        self.assertTrue(any("hypothesis inc is inconclusive" in gap for gap in gaps))
        self.assertFalse(any("confirmed head hypothesis confirmed rests" in gap for gap in gaps))


class DiscoveryLoopTest(unittest.TestCase):
    def test_full_loop_finds_reproducible_mechanism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = DiscoveryController(root / "mem.sqlite").run(
                skill="ioi-circuit", model="gpt2", backend="synthetic", out_dir=root / "run", include_memory=False
            )
            self.assertEqual(run.mode, "discovery")
            self.assertGreaterEqual(len(run.discoveries), 1)
            for discovery in run.discoveries:
                self.assertGreaterEqual(len(discovery.supporting_experiments), 2)
            # artifacts present and serializable
            payload = json.loads((root / "run" / "discoveries.json").read_text())
            self.assertEqual(len(payload["discoveries"]), len(run.discoveries))
            run_payload = json.loads((root / "run" / "run.json").read_text())
            self.assertEqual(Path(run_payload["artifacts"]["json"]), root / "run" / "run.json")
            self.assertEqual(Path(run_payload["artifacts"]["discoveries"]), root / "run" / "discoveries.json")
            self.assertEqual(Path(run_payload["artifacts"]["experiments"]), root / "run" / "experiments.json")
            evals = json.loads((root / "run" / "evals.json").read_text())
            names = {c["name"] for c in evals["checks"]}
            self.assertIn("has_confirmed_mechanism", names)
            self.assertIn("discoveries_are_triangulated", names)
            self.assertFalse(run.provenance["allow_seed_corpus"])
            self.assertFalse(run.provenance["used_packaged_seed_corpus"])
            self.assertEqual(run.provenance["answer_author"], "experiment_ledger_synthesizer")
            self.assertEqual(run.provenance["source_count"], 1)
            self.assertEqual([source.kind for source in run.sources], ["experiment_log"])
            json.dumps(run.to_dict())

    def test_verify_rejects_discovery_sidecar_drift_after_manifest_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            DiscoveryController(root / "mem.sqlite").run(
                skill="ioi-circuit", model="gpt2", backend="synthetic", out_dir=root / "run", include_memory=False
            )
            run_json = root / "run" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            (root / "run" / "experiments.json").write_text("[]\n", encoding="utf-8")
            (root / "run" / "discoveries.json").write_text(
                json.dumps(
                    {
                        "run_id": payload["run_id"],
                        "question": "Changed sidecar question",
                        "discoveries": [],
                        "hypotheses": [],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            refresh_run_manifest(run_json)

            verification = verify_run_artifacts(run_json)
            self.assertFalse(verification["passed"])
            self.assertFalse(verification["repairable"])
            self.assertIn("experiments_sidecar_matches_run", verification["failed_checks"])
            self.assertIn("discoveries_sidecar_question_matches_run", verification["failed_checks"])
            self.assertIn("discoveries_sidecar_discoveries_matches_run", verification["failed_checks"])
            self.assertNotIn("artifact_sha256:experiments", verification["failed_checks"])

    def test_seed_corpus_prior_art_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = DiscoveryController(root / "mem.sqlite").run(
                skill="ioi-circuit",
                model="gpt2",
                backend="synthetic",
                out_dir=root / "run",
                include_memory=False,
                allow_seed_corpus=True,
                budget=Budget(max_experiments=20, max_rounds=2),
            )
            self.assertTrue(run.provenance["allow_seed_corpus"])
            self.assertTrue(run.provenance["used_packaged_seed_corpus"])
            self.assertGreater(run.provenance["source_count"], 1)
            self.assertIn("file", {source.kind for source in run.sources})
            payload = json.loads((root / "run" / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["provenance"]["used_packaged_seed_corpus"])

    def test_provider_auto_can_author_discovery_synthesis(self):
        from mechferret.config import MechFerretConfig, ProviderSettings

        config = MechFerretConfig(
            default_provider="openai",
            providers={"openai": ProviderSettings(api_key="test-key", model="test-model")},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("mechferret.discovery.load_config", return_value=config),
                patch("mechferret.llm.load_config", return_value=config),
                patch(
                    "mechferret.llm.OpenAIWebResearch.search_summary",
                    return_value=Source("src_live", "Live prior art", "IOI circuits need evidence.", kind="openai_web_search"),
                ),
                patch("mechferret.llm._call_openai", return_value="Provider-authored discovery synthesis."),
            ):
                run = DiscoveryController(root / "mem.sqlite").run(
                    skill="ioi-circuit",
                    model="gpt2",
                    backend="synthetic",
                    out_dir=root / "run",
                    include_memory=False,
                    budget=Budget(max_experiments=20, max_rounds=2),
                    provider="auto",
                )
            self.assertEqual(run.answer, "Provider-authored discovery synthesis.")
            self.assertEqual(run.provenance["answer_author"], "provider_model")
            self.assertEqual(run.provenance["answer_provider"], "openai")
            self.assertEqual(run.provenance["answer_model"], "test-model")

    def test_budget_caps_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = DiscoveryController(root / "mem.sqlite").run(
                skill="ioi-circuit",
                model="gpt2",
                backend="synthetic",
                out_dir=root / "run",
                budget=Budget(max_experiments=20, max_rounds=2),
                include_memory=False,
            )
            self.assertLessEqual(run.metrics.get("experiments_run", 999), 20)

    def test_discovery_api_normalizes_malformed_boundary_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = DiscoveryController(root / "mem.sqlite").run(
                question=b"Find IOI heads",
                skill="ioi-circuit",
                model="gpt2",
                backend=[],
                out_dir=root / "run",
                source_paths=None,
                urls=["", None],
                provider=[],
                llm_model={},
                include_memory=[],
                allow_seed_corpus="yes",
                budget=Budget(max_experiments=4, max_rounds="bad", allow_network="yes"),
            )
            self.assertEqual(run.provenance["backend_requested"], "auto")
            self.assertEqual(run.provenance["provider_requested"], "auto")
            self.assertTrue(run.provenance["included_memory"])
            self.assertFalse(run.provenance["allow_seed_corpus"])
            self.assertEqual(run.provenance["requested_urls"], [])
            self.assertEqual(run.provenance["budget"]["max_experiments"], 4)
            self.assertEqual(run.provenance["budget"]["max_rounds"], 4)

    def test_openvla_sae_prompt_does_not_fall_through_to_ioi(self):
        issue = request_alignment_issue("Find SAEs for OpenVLA", None, "ioi", "gpt2")
        self.assertIn("openvla", issue)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "not aligned"):
                DiscoveryController(root / "mem.sqlite").run(
                    "Find SAEs for OpenVLA",
                    model="gpt2",
                    backend="synthetic",
                    out_dir=root / "run",
                    include_memory=False,
                )

    def test_vague_discovery_prompt_requires_explicit_task(self):
        issue = request_alignment_issue("Investigate an interesting model behavior", None, "", "gpt2")
        self.assertIn("could not infer", issue)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "could not infer"):
                DiscoveryController(root / "mem.sqlite").run(
                    "Investigate an interesting model behavior",
                    model="gpt2",
                    backend="synthetic",
                    out_dir=root / "run",
                    include_memory=False,
                )

    def test_allow_mismatch_is_explicit_escape_hatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = DiscoveryController(root / "mem.sqlite").run(
                "Find SAEs for OpenVLA",
                model="gpt2",
                task="ioi",
                backend="synthetic",
                out_dir=root / "run",
                budget=Budget(max_experiments=4, max_rounds=1),
                include_memory=False,
                allow_mismatch=True,
            )
            self.assertEqual(run.mode, "discovery")

    def test_discovery_artifact_builders_tolerate_malformed_rows(self):
        controller = DiscoveryController(":memory:")
        hyp = Hypothesis(
            id="hyp",
            statement="head",
            rationale="",
            task="ioi",
            predicted_effect="",
            target={"layer": "6", "head": "3"},
            status="confirmed",
            confidence="0.82",
            experiment_ids=["attn", "dla", "bad", None],
        )
        results_by_spec = {
            "attn": SimpleNamespace(
                id="attn-exp",
                spec_id="attn",
                probe="attention_pattern",
                significant="yes",
                reproduced=True,
                effect_size="0.6",
                target={"layer": "6", "head": "3"},
                evidence_text=b"attention evidence",
                observations=["attention observation"],
                backend_used=None,
                metrics={"duplicate_token": "0.8"},
            ),
            "dla": SimpleNamespace(
                id="dla-exp",
                spec_id="dla",
                probe="direct_logit_attribution",
                significant=True,
                reproduced="true",
                effect_size="-1.2",
                target={"layer": "6", "head": "3"},
                evidence_text="dla evidence",
                observations=[None, "dla observation"],
                backend_used="synthetic",
                metrics=[],
            ),
            "bad": SimpleNamespace(
                id="bad-exp",
                spec_id="bad",
                probe="activation_patching",
                significant="false",
                reproduced="true",
                effect_size="9",
                target=[],
            ),
            "malformed": object(),
        }

        discoveries = controller._build_discoveries(
            [object(), SimpleNamespace(status="confirmed", target=[]), hyp],
            results_by_spec,
            [object(), SimpleNamespace(id="prior", text="Known duplicate-token head evidence.")],
            b"gpt2",
        )
        evidence, claims = controller._ledger(
            [hyp],
            results_by_spec,
            discoveries + [SimpleNamespace(statement="", supporting_experiments=[])],
            [object(), EvidenceChunk("prior-ev", "prior-src", "Prior", "Prior evidence.")],
            [object(), Claim("prior-claim", "Prior claim.", [], ["prior-src"], 0.5, 0.5)],
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].effect_size, 1.2)
        self.assertEqual(discoveries[0].supporting_experiments, ["attn-exp", "dla-exp"])
        self.assertGreaterEqual(len([chunk for chunk in evidence if chunk.source_id != "prior-src"]), 2)
        discovery_claims = [claim for claim in claims if claim.stance == "discovery"]
        self.assertEqual(len(discovery_claims), 1)
        self.assertEqual(discoveries[0].claim_ids, [discovery_claims[0].id])
        self.assertTrue(discovery_claims[0].citations)

    def test_discovery_synthesis_helpers_normalize_malformed_values(self):
        controller = DiscoveryController(":memory:")
        readiness = controller._readiness(
            {
                "rigor_score": "bad",
                "confirmed_mechanisms": "2",
                "mean_discovery_confidence": "0.75",
            }
        )
        answer = controller._synthesize(
            b"Find IOI heads",
            [],
            {},
            [
                object(),
                SimpleNamespace(
                    statement=b"Head 6.3 moves names.",
                    confidence="0.8",
                    effect_size="bad",
                    reproducibility="0.67",
                    novelty=None,
                ),
            ],
            ["gap one", None, b"gap two"],
        )

        self.assertEqual(readiness, 0.4)
        self.assertIn("Answer to: Find IOI heads", answer)
        self.assertIn("Confirmed mechanisms in model for the task task:", answer)
        self.assertIn("effect=0.00", answer)
        self.assertIn("- gap two", answer)

    def test_discovery_loop_normalizes_internal_worker_rows_and_metrics(self):
        class FakeGenerator:
            def __init__(self, *args, **kwargs):
                pass

            def screen(self, question, task_name, max_heads, source_ids):
                hyp = Hypothesis(
                    id="screen-hyp",
                    statement="screen",
                    rationale="",
                    task="ioi",
                    predicted_effect="",
                    target={"scope": "upper_layers"},
                    experiment_ids=["spec-screen"],
                )
                spec = ExperimentSpec(
                    id="spec-screen",
                    name="screen",
                    probe="head_ablation",
                    model="gpt2",
                    task="ioi",
                    target={"layer": 6, "head": 3},
                    hypothesis_id="screen-hyp",
                )
                return [hyp], [spec]

            def promote(self, results, task_name, top_k, source_ids):
                hyp = Hypothesis(
                    id="target-hyp",
                    statement="target",
                    rationale="",
                    task="ioi",
                    predicted_effect="",
                    target={"layer": "6", "head": "3"},
                    experiment_ids=["spec-target"],
                )
                spec = ExperimentSpec(
                    id="spec-target",
                    name="target",
                    probe="attention_pattern",
                    model="gpt2",
                    task="ioi",
                    target={"layer": 6, "head": 3},
                    hypothesis_id="target-hyp",
                )
                return [hyp, SimpleNamespace(id="bad", target=[])], [spec, SimpleNamespace(hypothesis_id=[])]

        class FakeCoordinator:
            calls = 0

            def __init__(self, max_workers=1):
                pass

            def map(self, fn, specs):
                FakeCoordinator.calls += 1
                spec_id = "spec-screen" if FakeCoordinator.calls == 1 else "spec-target"
                return [
                    ExperimentResult(
                        id=f"{spec_id}-result",
                        spec_id=spec_id,
                        probe="head_ablation",
                        status="ran",
                        effect_size=1.0,
                        baseline=0.0,
                        significant=True,
                        reproduced=True,
                        target={"layer": 6, "head": 3},
                    ),
                    SimpleNamespace(id="missing-spec-id"),
                ]

        def fake_prior_art(*args, **kwargs):
            return [Source("prior-src", "Prior", "Prior text.")], [], [], False, False

        def fake_evaluate(self, hypotheses, results):
            return [b"gap"], {"confirmed_mechanisms": "2", "rigor_score": "0.8"}

        FakeCoordinator.calls = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("mechferret.discovery.HypothesisGenerator", FakeGenerator),
                patch("mechferret.discovery.Coordinator", FakeCoordinator),
                patch("mechferret.discovery.ExperimentCritic.evaluate", fake_evaluate),
                patch.object(DiscoveryController, "_prior_art", side_effect=fake_prior_art),
            ):
                run = DiscoveryController(root / "mem.sqlite").run(
                    "Find IOI heads",
                    model="gpt2",
                    task="ioi",
                    backend="synthetic",
                    out_dir=root / "run",
                    budget=Budget(max_experiments=10, max_rounds=4),
                    include_memory=False,
                )

        self.assertEqual(FakeCoordinator.calls, 2)
        self.assertEqual(run.metrics["confirmed_mechanisms"], "2")
        self.assertEqual(run.metrics["rigor_score"], "0.8")
        self.assertEqual(run.provenance["experiment_count"], 2)
        self.assertEqual([source.id for source in run.sources[:1]], ["prior-src"])


if __name__ == "__main__":
    unittest.main()
