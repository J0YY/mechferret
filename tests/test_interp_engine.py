import unittest

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
        self.backend = SyntheticBackend("gpt2")
        self.circuit = self.backend.circuit("ioi")

    def test_tasks_registered(self):
        self.assertEqual(set(TASKS), {"ioi", "induction", "greater_than", "factual_recall"})
        self.assertEqual(get_task("IOI").name, "ioi")
        self.assertEqual(infer_task("find induction heads"), "induction")

    def test_key_head_is_significant_and_reproduced(self):
        key = max(self.circuit.heads, key=lambda h: abs(h.magnitude))
        result = self.engine.run_spec(spec("head_ablation", key.layer, key.head))
        self.assertEqual(result.status, "ran")
        self.assertTrue(result.significant)
        self.assertTrue(result.reproduced)
        self.assertGreater(abs(result.effect_size), 0.5)

    def test_control_head_is_not_significant(self):
        layer, head = self.backend.control_head("ioi", 0)
        result = self.engine.run_spec(spec("head_ablation", layer, head))
        self.assertFalse(result.significant)

    def test_determinism(self):
        key = self.circuit.heads[0]
        a = self.engine.run_spec(spec("head_ablation", key.layer, key.head))
        b = InterpEngine("gpt2", "synthetic").run_spec(spec("head_ablation", key.layer, key.head))
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


if __name__ == "__main__":
    unittest.main()
