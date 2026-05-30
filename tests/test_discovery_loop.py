import json
import tempfile
import unittest
from pathlib import Path

from mechferret.discovery import DiscoveryController
from mechferret.hooks import Budget, BudgetGuard
from mechferret.coordinator import Coordinator
from mechferret.interp.hypotheses import HypothesisGenerator, update_hypotheses
from mechferret.interp.engine import InterpEngine
from mechferret.skills import list_skills, load_skill


class CoordinatorHooksTest(unittest.TestCase):
    def test_coordinator_preserves_order_serial_and_parallel(self):
        items = list(range(20))
        self.assertEqual(Coordinator(1).map(lambda x: x * x, items), [x * x for x in items])
        self.assertEqual(Coordinator(8).map(lambda x: x * x, items), [x * x for x in items])

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


class DiscoveryLoopTest(unittest.TestCase):
    def test_full_loop_finds_reproducible_mechanism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = DiscoveryController(root / "mem.sqlite").run(
                skill="ioi-circuit", backend="synthetic", out_dir=root / "run", include_memory=False
            )
            self.assertEqual(run.mode, "discovery")
            self.assertGreaterEqual(len(run.discoveries), 1)
            for discovery in run.discoveries:
                self.assertGreaterEqual(len(discovery.supporting_experiments), 2)
            # artifacts present and serializable
            payload = json.loads((root / "run" / "discoveries.json").read_text())
            self.assertEqual(len(payload["discoveries"]), len(run.discoveries))
            evals = json.loads((root / "run" / "evals.json").read_text())
            names = {c["name"] for c in evals["checks"]}
            self.assertIn("has_confirmed_mechanism", names)
            self.assertIn("discoveries_are_triangulated", names)
            json.dumps(run.to_dict())

    def test_budget_caps_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = DiscoveryController(root / "mem.sqlite").run(
                skill="ioi-circuit",
                backend="synthetic",
                out_dir=root / "run",
                budget=Budget(max_experiments=20, max_rounds=2),
                include_memory=False,
            )
            self.assertLessEqual(run.metrics.get("experiments_run", 999), 20)


if __name__ == "__main__":
    unittest.main()
