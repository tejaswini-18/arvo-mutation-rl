"""Unit tests for the three agents (PRD build step 5). Stdlib only."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.qlearning_agent import QLearningAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from agents.ucb1_agent import UCB1Agent  # noqa: E402
from arvo_env import state_index  # noqa: E402

S0 = (8, 0, 0)
S1 = (3, 1, 1)


class TestInterface(unittest.TestCase):

    def test_all_agents_return_valid_actions_and_are_reproducible(self):
        for make in (lambda: RandomAgent(seed=3), lambda: UCB1Agent(seed=3),
                     lambda: QLearningAgent(T=100, seed=3)):
            runs = []
            for _ in range(2):
                agent, actions = make(), []
                for i in range(50):
                    a = agent.select_action(S0)
                    self.assertIn(a, range(8))
                    agent.update(S0, a, float(a == 2), S1)
                    actions.append(a)
                runs.append(actions)
            self.assertEqual(runs[0], runs[1])


class TestUCB1(unittest.TestCase):

    def test_tries_every_operator_once_first(self):
        agent = UCB1Agent(seed=0)
        first = []
        for _ in range(8):
            a = agent.select_action(None)
            first.append(a)
            agent.update(None, a, 0.0, None)
        self.assertEqual(sorted(first), list(range(8)))

    def test_score_formula(self):
        agent = UCB1Agent(seed=0)
        for a in range(8):
            agent.update(None, a, 1.0 if a == 5 else 0.0, None)
        # all n=1, t=8: operator 5 has the highest mean, same bonus
        self.assertEqual(agent.select_action(None), 5)
        # make 5 heavily tried with mean 0.5; an arm with mean 0 and n=1 then
        # wins because its bonus sqrt(2 ln t / 1) is larger
        agent.counts[5], agent.sums[5] = 100, 50.0
        agent.t = sum(agent.counts)
        bonus_5 = math.sqrt(2) * math.sqrt(math.log(agent.t) / 100)
        bonus_0 = math.sqrt(2) * math.sqrt(math.log(agent.t) / 1)
        self.assertGreater(bonus_0, 0.5 + bonus_5)
        self.assertNotEqual(agent.select_action(None), 5)

    def test_converges_to_best_operator(self):
        agent = UCB1Agent(seed=1)
        for _ in range(2000):
            a = agent.select_action(None)
            agent.update(None, a, 1.0 if a == 6 else 0.2, None)
        self.assertEqual(agent.counts.index(max(agent.counts)), 6)


class TestQLearning(unittest.TestCase):

    def test_q_table_shape_and_init(self):
        agent = QLearningAgent(T=5000)
        self.assertEqual(len(agent.Q), 54)
        self.assertTrue(all(len(r) == 8 and not any(r) for r in agent.Q))

    def test_epsilon_schedule(self):
        agent = QLearningAgent(T=5000)      # decays over 0.2 * 5000 = 1000 steps
        checks = {0: 1.0, 500: 0.5, 950: 0.05, 1000: 0.05, 4999: 0.05}
        for t, eps in checks.items():
            agent.t = t
            self.assertAlmostEqual(agent.epsilon(), eps)

    def test_update_rule(self):
        agent = QLearningAgent(T=100)
        s, s2 = state_index(S0), state_index(S1)
        agent.Q[s2] = [0.0, 2.0, 0, 0, 0, 0, 0, 0]
        agent.Q[s][4] = 1.0
        agent.update(S0, 4, 3.0, S1)
        # 1.0 + 0.1 * (3.0 + 0.9 * 2.0 - 1.0) = 1.38
        self.assertAlmostEqual(agent.Q[s][4], 1.38)

    def test_greedy_after_exploration_decays(self):
        agent = QLearningAgent(T=100, seed=0)
        agent.t = 1000                      # epsilon at its 0.05 floor
        agent.Q[state_index(S0)][7] = 1.0
        picks = [agent.select_action(S0) for _ in range(400)]
        self.assertGreater(picks.count(7) / 400.0, 0.9)

    def test_learns_rewarding_operator(self):
        agent = QLearningAgent(T=2000, seed=2)
        for _ in range(2000):
            a = agent.select_action(S0)
            agent.update(S0, a, 1.0 if a == 3 else 0.0, S0)
        self.assertEqual(agent.greedy_policy()[state_index(S0)], 3)


if __name__ == "__main__":
    unittest.main()
