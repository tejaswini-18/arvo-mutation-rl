"""Tests for the Setting B perturbation in run_experiment.py. Stdlib only."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiment import perturb  # noqa: E402


class TestPerturb(unittest.TestCase):

    def _changed(self, a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    def test_tiny_poc_changes_at_least_one_byte(self):
        poc = b"(?(n"
        for s in range(50):
            out, n = perturb(poc, random.Random(s))
            self.assertEqual(n, 1)
            self.assertEqual(len(out), 4)
            self.assertEqual(self._changed(poc, out), 1)

    def test_ten_percent_of_longer_input(self):
        poc = bytes(range(200))
        out, n = perturb(poc, random.Random(0))
        self.assertEqual(n, 20)
        self.assertEqual(self._changed(poc, out), 20)

    def test_reproducible(self):
        a = perturb(b"abcdefghij" * 5, random.Random(4))
        b = perturb(b"abcdefghij" * 5, random.Random(4))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
