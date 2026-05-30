import unittest

from mechferret.agents import Critic
from mechferret.models import Claim


class CriticTest(unittest.TestCase):
    def test_detects_negation_contradiction(self):
        claims = [
            Claim("a", "The agent uses memory to validate evidence before synthesis.", ["e1"], ["s1"], 0.8, 0.8),
            Claim("b", "The agent does not use memory to validate evidence before synthesis.", ["e2"], ["s2"], 0.8, 0.8),
        ]
        contradictions = Critic()._contradictions(claims)
        self.assertEqual(len(contradictions), 1)


if __name__ == "__main__":
    unittest.main()

