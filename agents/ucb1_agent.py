"""UCB1 bandit over the 8 operators (PRD section 6). Python 3.5 compatible.

Stateless: ignores the state. Tries each operator once, then picks
argmax(mean_reward[a] + c * sqrt(ln(t) / n[a])) with c = sqrt(2), where t is
the total number of pulls so far.
"""

import math
import random

NUM_ACTIONS = 8


class UCB1Agent(object):
    name = "ucb1"

    def __init__(self, seed=0, c=math.sqrt(2)):
        self.rng = random.Random(seed)
        self.c = c
        self.counts = [0] * NUM_ACTIONS
        self.sums = [0.0] * NUM_ACTIONS
        self.t = 0

    def select_action(self, state):
        untried = [a for a in range(NUM_ACTIONS) if self.counts[a] == 0]
        if untried:
            # NOTE: the PRD says "try each once"; we try them in a seeded
            # random order rather than always 0..7.
            return self.rng.choice(untried)
        log_t = math.log(self.t)
        scores = [self.sums[a] / self.counts[a]
                  + self.c * math.sqrt(log_t / self.counts[a])
                  for a in range(NUM_ACTIONS)]
        best = max(scores)
        # NOTE: ties broken uniformly at random (seeded).
        return self.rng.choice([a for a in range(NUM_ACTIONS) if scores[a] == best])

    def update(self, state, action, reward, next_state):
        self.counts[action] += 1
        self.sums[action] += reward
        self.t += 1

    def means(self):
        return [self.sums[a] / self.counts[a] if self.counts[a] else 0.0
                for a in range(NUM_ACTIONS)]
