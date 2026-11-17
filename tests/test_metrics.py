"""Tests for the pure helpers in metrics.py. Stdlib only."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import a12, a12_magnitude, coverage_auc, crash_exec_or_censored, pad_curve  # noqa: E402


class TestCurves(unittest.TestCase):

    def test_pad_curve_holds_last_value(self):
        self.assertEqual(pad_curve([1, 3], 5), [1, 3, 3, 3, 3])
        self.assertEqual(pad_curve([1, 2, 3], 2), [1, 2])
        self.assertEqual(pad_curve([], 3), [0, 0, 0])

    def test_auc_is_mean_of_padded_curve(self):
        self.assertAlmostEqual(coverage_auc([2, 4], 4), (2 + 4 + 4 + 4) / 4.0)


class TestA12(unittest.TestCase):

    def test_extremes_and_ties(self):
        self.assertEqual(a12([5, 6], [1, 2]), 1.0)
        self.assertEqual(a12([1, 2], [5, 6]), 0.0)
        self.assertEqual(a12([3, 3], [3, 3]), 0.5)
        self.assertEqual(a12([1, 3], [2]), 0.5)

    def test_magnitude(self):
        self.assertEqual(a12_magnitude(0.52), "negligible")
        self.assertEqual(a12_magnitude(0.40), "small")
        self.assertEqual(a12_magnitude(0.68), "medium")
        self.assertEqual(a12_magnitude(0.0), "large")


class TestCensoring(unittest.TestCase):

    def test_unverified_or_missing_crash_is_T_plus_1(self):
        T = 100
        self.assertEqual(crash_exec_or_censored(
            {"verified": "True", "first_crash_exec": "7"}, T), 7)
        self.assertEqual(crash_exec_or_censored(
            {"verified": "False", "first_crash_exec": "7"}, T), 101)
        self.assertEqual(crash_exec_or_censored(
            {"verified": "False", "first_crash_exec": ""}, T), 101)


if __name__ == "__main__":
    unittest.main()
