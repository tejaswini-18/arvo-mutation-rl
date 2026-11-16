"""Experiment runner (PRD sections 8-9, build steps 6-7).

Loops agents x settings x seeds for ONE case, one episode per run, and appends
one row per run to results/<out>/runs.csv (plus full details to runs.jsonl).
Runs INSIDE the -vul container for the case, from the project root:

    docker run --rm -v "$(pwd):/work" -w /work n132/arvo:12950-vul \
        python3 run_experiment.py --case 12950 --settings A B --seeds 0 1 2 3 4

(run_all.sh loops this over the six cases, one container per case.)

Re-running the same command skips runs already in runs.csv, so an
interrupted experiment can be resumed. Python 3.5 compatible.

Setting B (perturbed PoC)
-------------------------
Start from the PoC with 10% of its bytes changed, re-checked so the perturbed
file does not already crash.
# NOTE: 10% of a short PoC rounds to zero bytes (the 12950 PoC is 4 bytes),
# so we change max(1, round(0.1 * len)) bytes. Each changed byte gets a new
# value different from the old one. The perturbation is seeded by the run
# seed only, so all three agents start from the same input for a given seed.
# If a draw still crashes we redraw (up to MAX_PERTURB_TRIES) and record how
# many tries it took.

Setting A (from seeds)
----------------------
Start from the seed set in seeds/<case>/seed_* (built by build_seeds.py; see
seeds/<case>/SOURCES.txt for where each file came from).
"""

import argparse
import binascii
import csv
import json
import os
import random
import sys
import time

import config
from arvo_env import ArvoEnv
from agents.qlearning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.ucb1_agent import UCB1Agent

AGENTS = ["random", "ucb1", "qlearning"]
PERTURB_FRACTION = 0.10
MAX_PERTURB_TRIES = 100

CSV_FIELDS = [
    "case", "agent", "setting", "seed", "T", "executions_used",
    "first_crash_exec", "verified", "reproduces_on_vul", "fix_returncode",
    "crash_frames", "final_coverage", "coverage_available", "seed_executions",
    "operator_counts", "start_input_hex", "perturbed_bytes", "perturb_tries",
    "wall_seconds", "coverage_curve",
]


def make_agent(name, T, seed):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "ucb1":
        return UCB1Agent(seed=seed)
    if name == "qlearning":
        return QLearningAgent(T=T, seed=seed)
    raise ValueError(name)


def perturb(data, rng, fraction=PERTURB_FRACTION):
    """Change max(1, round(fraction * len)) distinct bytes to new values."""
    buf = bytearray(data)
    n = max(1, int(round(fraction * len(buf))))
    for pos in rng.sample(range(len(buf)), n):
        new = rng.randrange(255)
        buf[pos] = new if new < buf[pos] else new + 1
    return bytes(buf), n


def setting_b_start(env, poc, seed):
    """Perturbed PoC that does not crash. Returns (input, n_changed, tries)."""
    rng = random.Random(1000 + seed)   # separate stream from the env/agent RNGs
    for tries in range(1, MAX_PERTURB_TRIES + 1):
        candidate, n = perturb(poc, rng)
        if not env._run_input(candidate)["crashed"]:
            return candidate, n, tries
    raise RuntimeError("no non-crashing perturbation of the PoC in %d tries"
                       % MAX_PERTURB_TRIES)


def load_seed_set(case_id):
    """Setting A inputs: seeds/<case>/seed_*, in name order."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds", case_id)
    names = sorted(n for n in os.listdir(d) if n.startswith("seed_"))
    if not names:
        raise RuntimeError("no seeds in %s; run build_seeds.py --case %s" % (d, case_id))
    seeds = []
    for n in names:
        with open(os.path.join(d, n), "rb") as f:
            seeds.append(f.read())
    return seeds


def run_one(case_id, agent_name, setting, seed, T, crash_dir, start_inputs):
    env = ArvoEnv(case_id, start_inputs, rng_seed=seed, T=T,
                  crash_dir=crash_dir)
    agent = make_agent(agent_name, T, seed)
    start = time.time()
    try:
        state = env.reset()
        done = False
        info = {}
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
        crash = env.crashes[0] if env.crashes else None
        row = {
            "case": case_id, "agent": agent_name, "setting": setting,
            "seed": seed, "T": T,
            "executions_used": env.executions,
            "first_crash_exec": crash["exec"] if crash else "",
            "verified": crash["verified"] if crash else False,
            "reproduces_on_vul": crash["reproduces_on_vul"] if crash else "",
            "fix_returncode": crash["fix_returncode"] if crash else "",
            "crash_frames": json.dumps(crash["frames"]) if crash else "",
            "final_coverage": len(env.covered),
            "coverage_available": env.coverage_available,
            "seed_executions": env.seed_executions,
            "operator_counts": json.dumps(env.operator_counts),
            # Setting B: the perturbed PoC; Setting A: "seeds/<case>" (N files).
            "start_input_hex": (binascii.hexlify(start_inputs[0]).decode("ascii")
                                if setting == "B" else
                                "seeds/%s (%d files)" % (case_id, len(start_inputs))),
            "wall_seconds": round(time.time() - start, 1),
            "coverage_curve": json.dumps(env.coverage_curve),
        }
        extra = {"crash_path": crash["path"] if crash else None}
        if hasattr(agent, "greedy_policy"):
            extra["q_greedy_policy"] = agent.greedy_policy()
            extra["q_table"] = agent.Q
        if hasattr(agent, "means"):
            extra["ucb1_means"] = agent.means()
            extra["ucb1_counts"] = agent.counts
        return row, extra
    finally:
        env.close()


def load_done(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add((r["case"], r["agent"], r["setting"], r["seed"], r["T"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="12950", choices=sorted(config.CASES))
    ap.add_argument("--settings", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--agents", nargs="+", default=AGENTS, choices=AGENTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--T", type=int, default=config.DEFAULT_T)
    ap.add_argument("--out", default="results/experiment")
    args = ap.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    csv_path = os.path.join(args.out, "runs.csv")
    jsonl_path = os.path.join(args.out, "runs.jsonl")
    crash_dir = os.path.join(args.out, "crashes")
    csv.field_size_limit(sys.maxsize)          # coverage curves are long
    already = load_done(csv_path)

    case = config.CASES[args.case]
    with open(case["poc_path"], "rb") as f:
        poc = f.read()

    for setting, seed in [(s, k) for s in args.settings for k in args.seeds]:
        # Same start input(s) for every agent with this seed (fair comparison).
        if setting == "A":
            start_inputs = load_seed_set(args.case)
            n_changed = tries = ""
            print("case %s setting A seed %d: %d seed files"
                  % (args.case, seed, len(start_inputs)))
        else:
            probe = ArvoEnv(args.case, [poc], rng_seed=seed, T=1)
            try:
                start_input, n_changed, tries = setting_b_start(probe, poc, seed)
            finally:
                probe.close()
            start_inputs = [start_input]
            print("case %s setting B seed %d: perturbed PoC (%d of %d bytes changed, %d tries)"
                  % (args.case, seed, n_changed, len(poc), tries))
        sys.stdout.flush()

        for agent_name in args.agents:
            key = (args.case, agent_name, setting, str(seed), str(args.T))
            if key in already:
                print("  skip %s (already in %s)" % (agent_name, csv_path))
                continue
            # One folder per run so crash files from different agents with
            # the same seed and crash step cannot overwrite each other.
            run_crash_dir = os.path.join(
                crash_dir, "%s_%s_%s_seed%d_T%d" % (args.case, agent_name,
                                                   setting, seed, args.T))
            row, extra = run_one(args.case, agent_name, setting, seed,
                                 args.T, run_crash_dir, start_inputs)
            row["perturbed_bytes"] = n_changed
            row["perturb_tries"] = tries

            new_file = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow(row)
            full = dict(row)
            full.update(extra)
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(full) + "\n")

            print("  %-9s execs=%-5d crash_exec=%-5s verified=%-5s coverage=%-5d %.0fs"
                  % (agent_name, row["executions_used"],
                     row["first_crash_exec"], row["verified"],
                     row["final_coverage"], row["wall_seconds"]))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
