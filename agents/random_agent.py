"""Uniform random baseline (PRD section 6). Python 3.5 compatible."""

import random

NUM_ACTIONS = 8


class RandomAgent(object):
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select_action(self, state):
        return self.rng.randrange(NUM_ACTIONS)

    def update(self, state, action, reward, next_state):
        pass
