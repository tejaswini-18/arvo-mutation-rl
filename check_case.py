"""Per-case check of ArvoEnv against one ARVO case (build step 7).

Run INSIDE the case's -vul container, from the project root:

    docker run --rm -v "$(pwd):/work" -w /work n132/arvo:<id>-vul \
        python3 check_case.py --case <id>

Checks, in our merge-mode runner (not the arvo script):
  1. the PoC is detected as a crash with a stack trace;
  2. a harmless input does not crash and yields coverage features;
  3. _verify accepts the PoC (reproduces on -vul, -fix exits 0);
  4. _verify rejects the harmless input.
Also prints exec speed. Python 3.5 compatible.
"""

import argparse
import sys
import time

import config
from arvo_env import ArvoEnv

HARMLESS = b"hello"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    args = ap.parse_args()
    case = config.CASES[args.case]
    with open(case["poc_path"], "rb") as f:
        poc = f.read()

    env = ArvoEnv(args.case, [HARMLESS], T=1)
    ok = True
    try:
        print("0. coverage mode: %s" % env.coverage_mode)
        ok &= env.coverage_mode is not None
        r = env._run_input(poc)
        print("1. PoC (%d bytes): crashed=%s frames=%s"
              % (len(poc), r["crashed"], r["frames"][:3]))
        ok &= bool(r["crashed"] and r["frames"])

        start = time.time()
        for _ in range(20):
            r = env._run_input(HARMLESS)
        per = (time.time() - start) / 20
        print("2. harmless: crashed=%s features=%d  (%.3f s/exec, %.0f exec/s)"
              % (r["crashed"], len(r["features"]), per, 1 / per))
        ok &= (not r["crashed"]) and len(r["features"]) > 0

        v = env._verify(poc)
        print("3. verify PoC: %s" % v)
        ok &= v["verified"] is True

        v = env._verify(HARMLESS)
        print("4. verify harmless: %s" % v)
        ok &= v["verified"] is False and not v["reproduces_on_vul"]
    finally:
        env.close()
    print("CASE %s: %s" % (args.case, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
