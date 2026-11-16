"""End-to-end check of ArvoEnv with the random agent (PRD build step 3).

Run inside the -vul container, from the project root:

    docker run --rm -v "$(pwd):/work" -w /work n132/arvo:12950-vul \
        python3 smoke_random.py --case 12950 --T 300

It checks that:
  1. the known PoC is detected as a crash, with a stack trace;
  2. a harmless seed is not a crash and its coverage is read;
  3. a random-agent episode runs, and prints coverage, speed and operator use.
Python 3.5 compatible.
"""

import argparse
import os
import sys
import time

import config
from arvo_env import ArvoEnv, state_index
from agents.random_agent import RandomAgent

# NOTE: placeholder seeds for this smoke test only. The real Setting A seed
# set is decided in build step 6/7.
DEFAULT_SEEDS = [b"hello", b"(a|b)*c", b"^[0-9]+$"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="12950")
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crash-dir", default="results/smoke_crashes")
    args = ap.parse_args()

    case = config.CASES[args.case]
    env = ArvoEnv(args.case, DEFAULT_SEEDS, rng_seed=args.seed, T=args.T,
                  crash_dir=args.crash_dir)
    ok = True
    try:
        print("== 1. PoC is detected as a crash")
        with open(case["poc_path"], "rb") as f:
            poc = f.read()
        r = env._run_input(poc)
        print("crashed=%s returncode=%s frames=%s" % (r["crashed"], r["returncode"], r["frames"][:3]))
        if not r["crashed"] or not r["frames"]:
            print("FAIL: PoC crash or its stack trace was not detected")
            ok = False

        print("== 2. Harmless seed: no crash, coverage read")
        r = env._run_input(DEFAULT_SEEDS[0])
        print("crashed=%s features=%d seconds=%.3f"
              % (r["crashed"], len(r["features"]), r["seconds"]))
        if r["crashed"] or not r["features"]:
            print("FAIL: unexpected crash or no coverage summary")
            ok = False

        print("== 3. Random-agent episode, T=%d, seed=%d" % (args.T, args.seed))
        agent = RandomAgent(seed=args.seed)
        state = env.reset()
        print("seed executions=%d start coverage=%d coverage_available=%s"
              % (env.seed_executions, len(env.covered), env.coverage_available))
        start = time.time()
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            assert 0 <= state_index(next_state) < 54
            agent.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            if info["executions"] % 100 == 0:
                print("  exec %d coverage %d pool %d"
                      % (info["executions"], info["total_coverage"], info["pool_size"]))
        elapsed = time.time() - start
        print("executions=%d crashed=%s final_coverage=%d pool=%d reward=%.2f"
              % (info["executions"], info["crashed"], info["total_coverage"],
                 info["pool_size"], total_reward))
        print("speed: %.1f exec/s (%.3f s/exec)"
              % (info["executions"] / elapsed, elapsed / info["executions"]))
        print("operator_counts=%s" % env.operator_counts)
        if info["crashed"]:
            print("crash saved to %s, frames=%s" % (info["crash_path"], info["frames"]))
        est = elapsed / info["executions"] * config.DEFAULT_T * 180 / 3600.0
        print("estimate for 180 runs x T=%d: %.1f hours (upper bound; runs stop at first crash)"
              % (config.DEFAULT_T, est))
    finally:
        env.close()
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
