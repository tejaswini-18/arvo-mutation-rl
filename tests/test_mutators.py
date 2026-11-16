"""Unit tests for mutators.py (PRD build step 2).

Run from the project root:  python -m unittest discover -s tests -v
Uses only the standard library so it also runs inside the ARVO containers
(Python 3.5).
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mutators  # noqa: E402

# Inputs covering the size buckets and edge cases (empty, 1 byte, S, M, L).
SAMPLE_INPUTS = [
    b"",
    b"a",
    b"ab",
    b"(?<n>abc)",
    bytes(range(256)),
    os.urandom(3000),
]
POOL = [b"hello", b"(?P<x>y)+", bytes(range(100))]
TRIALS = 200


class TestEachOperator(unittest.TestCase):

    def test_returns_nonempty_bytes(self):
        rng = random.Random(0)
        for action in range(mutators.NUM_ACTIONS):
            for data in SAMPLE_INPUTS:
                for _ in range(TRIALS):
                    out = mutators.mutate(action, data, rng, POOL)
                    self.assertIsInstance(out, bytes, mutators.OPERATOR_NAMES[action])
                    self.assertGreater(len(out), 0, mutators.OPERATOR_NAMES[action])

    def test_does_not_modify_input(self):
        rng = random.Random(1)
        for action in range(mutators.NUM_ACTIONS):
            data = bytearray(b"unchanged input")
            before = bytes(data)
            mutators.mutate(action, data, rng, POOL)
            self.assertEqual(bytes(data), before)

    def test_reproducible_with_same_seed(self):
        for action in range(mutators.NUM_ACTIONS):
            a = [mutators.mutate(action, b"seed input", random.Random(7), POOL)
                 for _ in range(5)]
            b = [mutators.mutate(action, b"seed input", random.Random(7), POOL)
                 for _ in range(5)]
            self.assertEqual(a, b)

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            mutators.mutate(8, b"x", random.Random(0))
        with self.assertRaises(ValueError):
            mutators.mutate(-1, b"x", random.Random(0))


class TestOperatorSemantics(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(42)
        self.data = bytes(range(64))

    def _diff_positions(self, a, b):
        return [i for i in range(len(a)) if a[i] != b[i]]

    def test_bit_flip_changes_exactly_one_bit(self):
        for _ in range(TRIALS):
            out = mutators.bit_flip(self.data, self.rng, [])
            self.assertEqual(len(out), len(self.data))
            pos = self._diff_positions(self.data, out)
            self.assertEqual(len(pos), 1)
            self.assertEqual(bin(self.data[pos[0]] ^ out[pos[0]]).count("1"), 1)

    def test_byte_flip_xors_one_byte(self):
        for _ in range(TRIALS):
            out = mutators.byte_flip(self.data, self.rng, [])
            pos = self._diff_positions(self.data, out)
            self.assertEqual(len(pos), 1)
            self.assertEqual(self.data[pos[0]] ^ out[pos[0]], 0xFF)

    def test_random_byte_changes_one_byte(self):
        for _ in range(TRIALS):
            out = mutators.random_byte(self.data, self.rng, [])
            self.assertEqual(len(self._diff_positions(self.data, out)), 1)

    def test_insert_adds_1_to_4_bytes(self):
        for _ in range(TRIALS):
            out = mutators.insert_bytes(self.data, self.rng, [])
            self.assertIn(len(out) - len(self.data), (1, 2, 3, 4))

    def test_delete_removes_1_to_4_bytes(self):
        for _ in range(TRIALS):
            out = mutators.delete_bytes(self.data, self.rng, [])
            self.assertIn(len(self.data) - len(out), (1, 2, 3, 4))

    def test_delete_keeps_last_byte(self):
        self.assertEqual(mutators.delete_bytes(b"z", self.rng, []), b"z")

    def test_interesting_value_keeps_length(self):
        for _ in range(TRIALS):
            out = mutators.interesting_value(self.data, self.rng, [])
            self.assertEqual(len(out), len(self.data))
            self.assertLessEqual(len(self._diff_positions(self.data, out)), 4)

    def test_duplicate_chunk_grows_by_1_to_32(self):
        for _ in range(TRIALS):
            out = mutators.duplicate_chunk(self.data, self.rng, [])
            self.assertTrue(1 <= len(out) - len(self.data) <= 32)

    def test_splice_uses_prefix_of_input(self):
        other = b"\xAA" * 50
        for _ in range(TRIALS):
            out = mutators.splice(self.data, self.rng, [other])
            # Output = non-empty prefix of data + non-empty suffix of other.
            k = len(out)
            while k > 0 and out[:k] != self.data[:k]:
                k -= 1
            self.assertGreaterEqual(k, 1)
            self.assertTrue(other.endswith(out[k:]) or self.data.endswith(out[k:]))

    def test_splice_without_pool(self):
        out = mutators.splice(b"abc", self.rng, [])
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
