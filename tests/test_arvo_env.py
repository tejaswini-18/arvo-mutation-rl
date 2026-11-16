"""Tests for the parts of arvo_env.py that need no container (stdlib only)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arvo_env import parse_output, parse_summary, size_bucket, state_index  # noqa: E402

CLEAN = """MERGE-OUTER: writing coverage summary for 1 files to sum
MERGE-OUTER: 1 new files with 245 new features added
"""

# Format copied from a real summary written by /out/pcre2_fuzzer (case 12950).
SUMMARY = "/tmp/in/x size: 5 features:  0 18 28 30 a8 109 21d1\n"

CRASH = """MERGE-OUTER: attempt 1
==7==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000000b4
READ of size 1 at 0x6020000000b4 thread T0
    #0 0x5d4fa1 in read_name /src/pcre2/src/pcre2_compile.c:2249:5
    #1 0x5c8a10 in parse_regex /src/pcre2/src/pcre2_compile.c:3000:9
    #2 0x5c0001 in pcre2_compile_8 /src/pcre2/src/pcre2_compile.c:9500:3
    #3 0x5a1c00 in LLVMFuzzerTestOneInput /src/pcre2/src/pcre2_fuzzsupport.c:180:5

0x6020000000b4 is located 0 bytes to the right of 4-byte region
allocated by thread T0 here:
    #0 0x4f0000 in malloc
SUMMARY: AddressSanitizer: heap-buffer-overflow /src/pcre2/src/pcre2_compile.c:2249:5 in read_name
"""


class TestParseOutput(unittest.TestCase):

    def test_clean_run(self):
        r = parse_output(CLEAN)
        self.assertFalse(r["crashed"])
        self.assertEqual(r["frames"], [])

    def test_summary_features(self):
        self.assertEqual(parse_summary(SUMMARY),
                         {"0", "18", "28", "30", "a8", "109", "21d1"})
        self.assertEqual(parse_summary(""), set())

    def test_crash_takes_first_stack_only(self):
        r = parse_output(CRASH)
        self.assertTrue(r["crashed"])
        self.assertEqual(len(r["frames"]), 4)
        self.assertTrue(r["frames"][0].startswith("read_name "))
        self.assertNotIn("malloc", " ".join(r["frames"]))

    def test_garbage_is_not_a_crash(self):
        self.assertFalse(parse_output("garbage")["crashed"])


class TestState(unittest.TestCase):

    def test_size_buckets(self):
        self.assertEqual([size_bucket(n) for n in (0, 63, 64, 1024, 1025)],
                         [0, 0, 1, 1, 2])

    def test_state_index_covers_0_to_53_exactly_once(self):
        idx = [state_index((p, c, s)) for p in range(9) for c in (0, 1) for s in range(3)]
        self.assertEqual(sorted(idx), list(range(54)))


if __name__ == "__main__":
    unittest.main()
