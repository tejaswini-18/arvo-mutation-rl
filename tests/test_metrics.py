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


class TestCorrections(unittest.TestCase):

    def test_holm_and_bh_match_hand_computation(self):
        from metrics import bh_adjust, holm_adjust
        ps = [0.01, None, 0.04, 0.03]
        # Holm: sorted 0.01, 0.03, 0.04 -> 3*0.01, max(.03, 2*0.03), max(.06, 0.04)
        h = holm_adjust(ps)
        self.assertIsNone(h[1])
        for got, want in zip([h[0], h[3], h[2]], [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(got, want)
        # BH: 0.04*3/3=0.04, min(0.04, 0.03*3/2=0.045)=0.04, 0.01*3/1=0.03
        b = bh_adjust(ps)
        self.assertIsNone(b[1])
        for got, want in zip([b[0], b[3], b[2]], [0.03, 0.04, 0.04]):
            self.assertAlmostEqual(got, want)

    def test_adjusted_never_below_raw_and_capped(self):
        from metrics import bh_adjust, holm_adjust
        ps = [0.5, 0.9, 0.001]
        for adj in (holm_adjust(ps), bh_adjust(ps)):
            for p, a in zip(ps, adj):
                self.assertGreaterEqual(a, p)
                self.assertLessEqual(a, 1.0)


class TestSplitResultFiles(unittest.TestCase):
    """Per-job runs.<tag>.csv files (run_parallel.sh) load like one runs.csv."""

    def _write(self, d, name, rows):
        import csv
        fields = ["case", "agent", "setting", "seed", "T",
                  "coverage_curve", "operator_counts"]
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def test_tagged_files_are_combined_and_duplicates_dropped(self):
        import contextlib
        import io
        import shutil
        import tempfile
        from metrics import load_runs, result_files
        root = tempfile.mkdtemp()
        try:
            d = os.path.join(root, "12950")
            os.makedirs(d)
            base = {"case": "12950", "T": "10", "coverage_curve": "[1, 2]",
                    "operator_counts": "[0]"}
            row = lambda agent, setting, seed: dict(base, agent=agent,
                                                    setting=setting, seed=seed)
            self._write(d, "runs.A_random.csv", [row("random", "A", "0")])
            self._write(d, "runs.B_ucb1.csv", [row("ucb1", "B", "0"),
                                               row("ucb1", "B", "1")])
            # Same run as in runs.A_random.csv: must be loaded once only.
            self._write(d, "runs.csv", [row("random", "A", "0")])
            with open(os.path.join(d, "runs.A_random.jsonl"), "w") as f:
                f.write("{}\n".replace("{}", '{"agent": "random", "setting": "A", '
                                             '"seed": 0, "T": 10}'))
            self.assertEqual([os.path.basename(p) for p in result_files(d, "csv")],
                             ["runs.A_random.csv", "runs.B_ucb1.csv", "runs.csv"])
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rows = load_runs(root)
            keys = sorted((r["agent"], r["setting"], r["seed"]) for r in rows)
            self.assertEqual(keys, [("random", "A", "0"), ("ucb1", "B", "0"),
                                    ("ucb1", "B", "1")])
            self.assertIn("duplicate", err.getvalue())
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
