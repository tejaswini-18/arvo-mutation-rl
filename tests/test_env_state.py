"""The state's size bucket is the size of the seed selected for that step
(proposal Algorithm 1 / Eq. 1). Uses a fake target, so no container needed."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arvo_env  # noqa: E402
from arvo_env import ArvoEnv, size_bucket  # noqa: E402


class FakeEnv(ArvoEnv):
    """ArvoEnv with the process execution replaced by a deterministic fake."""

    def __init__(self, seeds, T=50):
        self.case_id = "fake"
        self.initial_seeds = [bytes(s) for s in seeds]
        self.rng_seed = 0
        self.T = T
        self.crash_dir = None
        self.use_coverage = True
        self.used_seeds = []

    def _run_input(self, data):
        # new "feature" for every distinct length, so many children are novel
        return {"crashed": False, "frames": [], "features": {str(len(data))},
                "returncode": 0, "timed_out": False, "seconds": 0.0, "output": ""}


class TestSelectedSeedState(unittest.TestCase):

    def test_state_bucket_is_size_of_seed_used_next(self):
        env = FakeEnv([b"a" * 10, b"b" * 2000])
        state = env.reset()
        for _ in range(40):
            # the seed the next step will mutate is the one the state describes
            self.assertEqual(state[2], size_bucket(len(env.current_seed)))
            state, _, done, _ = env.step(0)
            if done:
                break

    def test_seed_choice_is_uniform_over_pool(self):
        env = FakeEnv([b"x" * 5, b"y" * 5])
        env.reset()
        picks = {b"x" * 5: 0, b"y" * 5: 0}
        for _ in range(400):
            picks[env.current_seed] = picks.get(env.current_seed, 0) + 1
            env.step(2)
            if env.executions >= env.T:
                env.T += 400          # keep stepping; only sampling is checked
        self.assertGreater(picks[b"x" * 5], 50)
        self.assertGreater(picks[b"y" * 5], 50)


if __name__ == "__main__":
    unittest.main()
