"""Tabular Q-learning over the 54 states x 8 operators (PRD section 6).
Python 3.5 compatible (no numpy).

  Q[s,a] += alpha * (reward + gamma * max(Q[s',:]) - Q[s,a])
  alpha = 0.1, gamma = 0.9, Q initialised to 0
  epsilon-greedy with epsilon = max(0.05, 1 - t / (0.2 * T))

so exploration decays linearly to 0.05 over the first 20% of the budget.
"""

import random

from arvo_env import NUM_STATES, state_index

NUM_ACTIONS = 8


class QLearningAgent(object):
    name = "qlearning"

    def __init__(self, T, seed=0, alpha=0.1, gamma=0.9, eps_min=0.05,
                 decay_fraction=0.2):
        self.rng = random.Random(seed)
        self.T = T
        self.alpha = alpha
        self.gamma = gamma
        self.eps_min = eps_min
        self.decay_fraction = decay_fraction
        self.Q = [[0.0] * NUM_ACTIONS for _ in range(NUM_STATES)]
        self.t = 0           # number of actions selected so far

    def epsilon(self):
        return max(self.eps_min, 1.0 - self.t / (self.decay_fraction * self.T))

    def select_action(self, state):
        eps = self.epsilon()
        self.t += 1
        if self.rng.random() < eps:
            return self.rng.randrange(NUM_ACTIONS)
        row = self.Q[state_index(state)]
        best = max(row)
        # NOTE: ties (e.g. an all-zero row) are broken uniformly at random.
        return self.rng.choice([a for a in range(NUM_ACTIONS) if row[a] == best])

    def update(self, state, action, reward, next_state):
        # NOTE: the PRD's update always bootstraps from max Q[s',:], including
        # on the terminal (crash) step; the shared Agent interface has no
        # `done` flag. Each run is a single episode that ends at the first
        # crash, so this affects at most the last update of a run.
        s = state_index(state)
        s2 = state_index(next_state)
        target = reward + self.gamma * max(self.Q[s2])
        self.Q[s][action] += self.alpha * (target - self.Q[s][action])

    def greedy_policy(self):
        """Best operator per state (for RQ3); -1 where the row is all zero."""
        policy = []
        for row in self.Q:
            best = max(row)
            policy.append(row.index(best) if any(v != 0.0 for v in row) else -1)
        return policy
