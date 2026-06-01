import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mechferret.interp.engine import InterpEngine
from mechferret.interp.synthetic import SyntheticBackend
from mechferret.interp.tasks import TASKS, get_task, infer_task
from mechferret.models import ExperimentSpec


def spec(probe, layer, head, task="ioi", **target):
    return ExperimentSpec(
        id=f"{probe}-{layer}-{head}",
        name=probe,
        probe=probe,
        model="gpt2",
        task=task,
        target={"layer": layer, "head": head, **target},
        metric="logit_diff",
    )


class InterpEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = InterpEngine("gpt2", "synthetic")
        self.backend = self.engine.backend_for()
        self.assertIsInstance(self.backend, SyntheticBackend)
        self.circuit = self.backend.circuit("ioi")

    def test_tasks_registered(self):
        self.assertEqual(set(TASKS), {"ioi", "induction", "greater_than", "factual_recall"})
        self.assertEqual(get_task("IOI").name, "ioi")
        self.assertEqual(infer_task("find induction heads"), "induction")
        self.assertEqual(infer_task("plan a vague research project"), "")

    def test_key_head_is_significant_and_reproduced(self):
        key = max(self.circuit.heads, key=lambda h: abs(h.magnitude))
        result = self.engine.run_spec(spec("head_ablation", key.layer, key.head))
        self.assertEqual(result.status, "ran")
        self.assertTrue(result.significant)
        self.assertTrue(result.reproduced)
        self.assertGreater(abs(result.effect_size), 0.5)

    def test_synthetic_circuit_positive_heads_are_screenable(self):
        for salt in range(30):
            circuit = SyntheticBackend("gpt2", run_salt=salt).circuit("ioi")
            positives = [head for head in circuit.heads if head.magnitude > 0]
            self.assertGreaterEqual(len(positives), 1)
            self.assertTrue(all(head.layer >= circuit.n_layers // 3 for head in positives))

    def test_control_head_is_not_significant(self):
        layer, head = self.backend.control_head("ioi", 0)
        result = self.engine.run_spec(spec("head_ablation", layer, head))
        self.assertFalse(result.significant)

    def test_backend_keeps_run_scope_stable(self):
        key = self.circuit.heads[0]
        a = self.engine.run_spec(spec("head_ablation", key.layer, key.head))
        b = self.engine.run_spec(spec("head_ablation", key.layer, key.head))
        self.assertEqual(a.effect_size, b.effect_size)
        self.assertEqual(a.per_seed, b.per_seed)

    def test_all_probes_run(self):
        key = self.circuit.heads[0]
        for probe in ("head_ablation", "activation_patching", "attention_pattern", "direct_logit_attribution", "logit_lens"):
            result = self.engine.run_spec(spec(probe, key.layer, key.head, pattern="duplicate_token"))
            self.assertEqual(result.status, "ran", probe)
            self.assertIn("separation", result.metrics)

    def test_unknown_task_and_probe_error(self):
        bad_task = ExperimentSpec(id="x", name="x", probe="head_ablation", model="gpt2", task="nope", target={"layer": 0, "head": 0})
        self.assertEqual(self.engine.run_spec(bad_task).status, "error")
        bad_probe = ExperimentSpec(id="y", name="y", probe="nope", model="gpt2", task="ioi", target={"layer": 0, "head": 0})
        self.assertEqual(self.engine.run_spec(bad_probe).status, "error")

    def test_run_spec_tolerates_malformed_spec_and_seeds(self):
        malformed = SimpleNamespace(id=[], probe=[], task=[], target=[], model=[], backend=[])
        result = self.engine.run_spec(malformed)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.spec_id, "spec")
        self.assertEqual(result.target, {})

        key = self.circuit.heads[0]
        bad_seeds = spec("head_ablation", key.layer, key.head)
        bad_seeds.seeds = ["0", True, "bad", 1, 1]  # type: ignore[list-item]
        result = self.engine.run_spec(bad_seeds)
        self.assertEqual(result.status, "ran")
        self.assertEqual(len(result.per_seed), 2)

    def test_run_spec_returns_error_for_bad_probe_reading(self):
        bad_spec = spec("head_ablation", 0, 0)

        def bad_probe(_backend, _task, _target, _seed):
            return SimpleNamespace(effect="nan", control=0.0, unit="logit_diff", observations=[], extra={})

        with patch("mechferret.interp.engine.get_probe", return_value=bad_probe):
            result = self.engine.run_spec(bad_spec)
        self.assertEqual(result.status, "error")
        self.assertIn("non-finite", result.error)

    def test_run_specs_tolerates_malformed_collection(self):
        self.assertEqual(self.engine.run_specs("not specs"), [])


if __name__ == "__main__":
    unittest.main()
