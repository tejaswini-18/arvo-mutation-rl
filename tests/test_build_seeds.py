"""Tests for the seed-selection order in build_seeds.py. Stdlib only."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_seeds import spread_order  # noqa: E402


class TestSpreadOrder(unittest.TestCase):

    def test_small_list_unchanged(self):
        self.assertEqual(spread_order([1, 2, 3], k=10), [1, 2, 3])

    def test_first_k_span_the_range_then_rest(self):
        items = list(range(100))
        out = spread_order(items, k=10)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[9], 99)
        self.assertEqual(out[:10], sorted(out[:10]))
        self.assertEqual(sorted(out), items)          # nothing lost or repeated


if __name__ == "__main__":
    unittest.main()
