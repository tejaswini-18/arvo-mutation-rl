"""End-to-end check of ArvoEnv and the agents (PRD build steps 3-5).

Run inside the -vul container, from the project root:

    docker run --rm -v "$(pwd):/work" -w /work n132/arvo:12950-vul \
        python3 smoke_random.py --case 12950 --T 300 [--agent ucb1] [--episode-only]

It checks that:
  1.  the known PoC is detected as a crash, with a stack trace;
  2.  a harmless seed is not a crash and its coverage is read;
  2b. _verify accepts the PoC and rejects the harmless seed;
  2c. a crash found through step() is verified;
  3.  an agent episode runs, and prints coverage, speed and operator use.
Python 3.5 compatible.
"""

import argparse
import sys
import time

import config
from arvo_env import ArvoEnv, state_index
from agents.qlearning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.ucb1_agent import UCB1Agent

# NOTE: placeholder seeds for this smoke test only. The real Setting A seed
# set is decided in build step 6/7.
DEFAULT_SEEDS = [b"hello", b"(a|b)*c", b"^[0-9]+$"]


def make_agent(name, T, seed):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "ucb1":
        return UCB1Agent(seed=seed)
    if name == "qlearning":
        return QLearningAgent(T=T, seed=seed)
    raise ValueError(name)


def env_checks(env, args, poc):
    """Checks 1-2c. Returns True if all pass."""
    ok = True
    print("== 1. PoC is detected as a crash")
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

    print("== 2b. _verify: PoC verified, harmless seed not")
    v = env._verify(poc)
    print("PoC: %s" % v)
    if v["verified"] is not True:
        print("FAIL: PoC should reproduce on -vul and be clean on -fix")
        ok = False
    v = env._verify(DEFAULT_SEEDS[0])
    print("harmless: %s" % v)
    if v["verified"] is not False or v["reproduces_on_vul"]:
        print("FAIL: a harmless input must not verify")
        ok = False

    print("== 2c. step() crash path: start from the PoC, mutate until a crash")
    # The 12950 PoC is 4 bytes; roughly 1 in 10 single mutations of it
    # still crashes, so 200 steps is plenty.
    penv = ArvoEnv(args.case, [poc], rng_seed=args.seed, T=200,
                   crash_dir=args.crash_dir)
    pagent = RandomAgent(seed=args.seed)
    try:
        penv.reset()
        done = False
        while not done:
            _, preward, done, pinfo = penv.step(pagent.select_action(None))
        print("crashed=%s at exec %d, reward=%.1f, verified=%s, fix_rc=%s, frames=%s"
              % (pinfo["crashed"], pinfo["executions"], preward,
                 pinfo["verified"], pinfo["fix_returncode"], pinfo["frames"]))
        if not pinfo["crashed"] or pinfo["verified"] is not True:
            print("FAIL: expected a verified crash within 200 mutations of the PoC")
            ok = False
    finally:
        penv.close()
    return ok


def run_episode(env, agent, T):
    print("== 3. %s episode, T=%d" % (agent.name, T))
    state = env.reset()
    print("seed executions=%d start coverage=%d coverage_available=%s"
          % (env.seed_executions, len(env.covered), env.coverage_available))
    start = time.time()
    total_reward = 0.0
    done = False
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
        print("crash saved to %s, verified=%s, frames=%s"
              % (info["crash_path"], info["verified"], info["frames"]))
    if hasattr(agent, "means"):
        print("ucb1 mean reward per operator=%s"
              % ["%.3f" % m for m in agent.means()])
    if hasattr(agent, "greedy_policy"):
        print("qlearning epsilon at end=%.3f, states visited=%d/54"
              % (agent.epsilon(), sum(1 for p in agent.greedy_policy() if p >= 0)))
    est = elapsed / info["executions"] * config.DEFAULT_T * 180 / 3600.0
    print("estimate for 180 runs x T=%d: %.1f hours (upper bound; runs stop at first crash)"
          % (config.DEFAULT_T, est))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="12950")
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crash-dir", default="results/smoke_crashes")
    ap.add_argument("--agent", default="random",
                    choices=["random", "ucb1", "qlearning"])
    ap.add_argument("--episode-only", action="store_true",
                    help="skip checks 1-2c and only run the agent episode")
    args = ap.parse_args()

    case = config.CASES[args.case]
    with open(case["poc_path"], "rb") as f:
        poc = f.read()
    env = ArvoEnv(args.case, DEFAULT_SEEDS, rng_seed=args.seed, T=args.T,
                  crash_dir=args.crash_dir)
    ok = True
    try:
        if not args.episode_only:
            ok = env_checks(env, args, poc)
        run_episode(env, make_agent(args.agent, args.T, args.seed), args.T)
    finally:
        env.close()
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
