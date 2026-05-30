import tempfile
import unittest
from pathlib import Path

from mechferret.goal_loop import GoalLoop, estimate_acceptance_probability
from mechferret.models import ResearchPlan, ResearchRun


class GoalLoopTest(unittest.TestCase):
    def test_estimate_acceptance_probability_is_bounded(self):
        run = ResearchRun(
            run_id="run",
            question="q",
            created_at="now",
            plan=ResearchPlan("q", [], "s"),
            sources=[],
            evidence=[],
            claims=[],
            contradictions=[],
            gaps=["missing baselines"],
            answer="a",
            metrics={"readiness_score": 0.9, "source_diversity": 5, "citation_density": 1.3, "mean_confidence": 0.8},
        )
        probability = estimate_acceptance_probability(run, "NeurIPS main")
        self.assertGreater(probability, 0)
        self.assertLess(probability, 1)

    def test_goal_loop_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paper.md"
            source.write_text(
                "# Proposal\n"
                "The system has strong retrieval evidence, critic loops, baselines, ablations, and reviewer-facing evaluation. "
                "It includes independent datasets, source diversity, citation validation, and error analysis.",
                encoding="utf-8",
            )
            result = GoalLoop(root / "memory.sqlite").run(
                "Can this autoresearch system reach a conference bar?",
                venue="NeurIPS main",
                target=0.2,
                source_paths=[str(source)],
                out_dir=root / "goal",
                max_iterations=2,
                include_memory=False,
                provider="local",
            )
            self.assertTrue((root / "goal" / "goal.json").exists())
            self.assertGreaterEqual(len(result["iterations"]), 1)


if __name__ == "__main__":
    unittest.main()

