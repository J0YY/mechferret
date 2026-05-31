import unittest
from types import SimpleNamespace

from mechferret.hooks import Budget, BudgetGuard


class HooksTest(unittest.TestCase):
    def test_budget_guard_normalizes_malformed_budget_fields(self):
        guard = BudgetGuard(
            Budget(
                max_experiments="bad",  # type: ignore[arg-type]
                max_rounds=0,
                max_gpu_seconds=float("nan"),
                max_wall_seconds=[],
                allow_gpu="yes",  # type: ignore[arg-type]
                allow_network=[],  # type: ignore[arg-type]
            ),
            experiments_run=-5,
            gpu_seconds="bad",  # type: ignore[arg-type]
            rounds_run="bad",  # type: ignore[arg-type]
            notices="bad",  # type: ignore[arg-type]
        )

        self.assertEqual(guard.budget.max_experiments, 400)
        self.assertEqual(guard.budget.max_rounds, 4)
        self.assertEqual(guard.budget.max_gpu_seconds, 900.0)
        self.assertEqual(guard.budget.max_wall_seconds, 1800.0)
        self.assertTrue(guard.permits("gpu"))
        self.assertTrue(guard.permits("network"))
        self.assertEqual(guard.remaining_experiments(), 400)
        self.assertEqual(guard.notices, [])

    def test_budget_guard_tolerates_malformed_specs_and_results(self):
        guard = BudgetGuard(Budget(max_experiments=2, max_rounds=1))

        self.assertEqual(guard.admit("not specs"), [])
        admitted = guard.admit([1, 2, 3])
        self.assertEqual(admitted, [1, 2])
        self.assertTrue(guard.notices)

        guard.record(
            [
                SimpleNamespace(gpu_seconds="3.5"),
                SimpleNamespace(gpu_seconds=-1),
                SimpleNamespace(gpu_seconds="bad"),
                object(),
            ]
        )
        self.assertEqual(guard.experiments_run, 4)
        self.assertEqual(guard.gpu_seconds, 3.5)
        usage = guard.usage()
        self.assertEqual(usage["experiments_run"], 4.0)
        self.assertEqual(usage["gpu_seconds"], 3.5)


if __name__ == "__main__":
    unittest.main()
