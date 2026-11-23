"""Reference run: plain libFuzzer (in-process, its own mutators) on one case.

Not one of the experiment's agents. It answers "can this bug be found at all
from these starting inputs with a much larger budget, and after how many
executions?", and gives a mature-fuzzer reference point. Runs INSIDE the
case's -vul container, from the project root:

    docker run --rm -v "$(pwd):/work" -w /work n132/arvo:3848-vul \
        python3 -u run_libfuzzer.py --case 3848 --seed 0 --minutes 60 \
            --out results/libfuzzer

Writes <out>/<case>_<label>_seed<N>.json (one file per job, so parallel jobs
never share a file), the libFuzzer log next to it, and any crashing input to
<out>/crashes/. A crash counts only if it reproduces on -vul and the -fix
build exits cleanly, the same check (ArvoEnv._verify) as the agents' crashes.
Python 3.5 compatible.

# NOTE: libFuzzer is given the same ASAN_OPTIONS and -timeout as our runner
# (config.py). It uses its default -rss_limit_mb (2048): an out-of-memory or
# timeout artifact is recorded as such and is not counted as finding the bug.
# NOTE: executions are libFuzzer's own count (stat::number_of_executed_units,
# else the last "#N" status line). Its "ft:" feature count is the same kind of
# number our runner records, but libFuzzer also keeps every input that adds a
# feature, so its coverage is a reference, not a like-for-like comparison.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import config
from arvo_env import ArvoEnv, parse_output


def last_status(text):
    """(execs, cov, ft) from the last '#N ... cov: X ft: Y' libFuzzer line."""
    execs = cov = ft = None
    for line in text.splitlines():
        m = re.match(r"#(\d+)\s", line)
        if not m:
            continue
        execs = int(m.group(1))
        c = re.search(r"\bcov: (\d+)", line)
        f = re.search(r"\bft: (\d+)", line)
        if c:
            cov = int(c.group(1))
        if f:
            ft = int(f.group(1))
    return execs, cov, ft


def final_stat(text, name):
    m = re.search(r"^stat::%s:\s+(\d+)" % name, text, re.M)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(config.CASES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--seeds-dir", default=None,
                    help="starting inputs (default seeds/<case>)")
    ap.add_argument("--label", default="seeds",
                    help="name for this seed set in the output file names")
    ap.add_argument("--out", default="results/libfuzzer")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    seeds_src = args.seeds_dir or os.path.join(root, "seeds", args.case)
    names = sorted(n for n in os.listdir(seeds_src) if n.startswith("seed_"))
    if not names:
        sys.exit("no seed_* files in %s" % seeds_src)
    tag = "%s_%s_seed%d" % (args.case, args.label, args.seed)
    if not os.path.isdir(os.path.join(args.out, "crashes")):
        try:
            os.makedirs(os.path.join(args.out, "crashes"))
        except OSError:
            pass                                  # another job made it first

    case = config.CASES[args.case]
    scratch = tempfile.mkdtemp(prefix="libfz_")
    try:
        seeds_dir = os.path.join(scratch, "seeds")
        corpus_dir = os.path.join(scratch, "corpus")   # libFuzzer writes here
        art_dir = os.path.join(scratch, "artifacts")
        for d in (seeds_dir, corpus_dir, art_dir):
            os.makedirs(d)
        for n in names:
            shutil.copy(os.path.join(seeds_src, n), seeds_dir)

        env = dict(os.environ)
        env["ASAN_OPTIONS"] = config.ASAN_OPTIONS
        env["TMPDIR"] = scratch
        seconds = int(args.minutes * 60)
        cmd = [case["fuzz_binary"], "-seed=%d" % args.seed,
               "-max_total_time=%d" % seconds,
               "-timeout=%d" % config.EXEC_TIMEOUT,
               "-print_final_stats=1",
               "-artifact_prefix=" + art_dir + "/",
               corpus_dir, seeds_dir]
        log_path = os.path.join(args.out, tag + ".log")
        print("%s: %s" % (tag, " ".join(cmd)))
        sys.stdout.flush()
        start = time.time()
        with open(log_path, "wb") as log:
            # Generous wall-clock guard on top of -max_total_time.
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, timeout=seconds + 600)
        wall = time.time() - start
        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "replace")

        execs, cov, ft = last_status(text)
        execs = final_stat(text, "number_of_executed_units") or execs
        artifacts = sorted(os.listdir(art_dir))
        result = {
            "case": args.case, "seeds_label": args.label,
            "seeds_dir": os.path.relpath(seeds_src, root), "n_seeds": len(names),
            "seed": args.seed, "minutes": args.minutes,
            "returncode": proc.returncode, "wall_seconds": round(wall, 1),
            "executions": execs, "execs_per_second":
                round(execs / wall, 1) if execs and wall else None,
            "final_cov": cov, "final_ft": ft,
            "corpus_size": len(os.listdir(corpus_dir)),
            "peak_rss_mb": final_stat(text, "peak_rss_mb"),
            "artifact": artifacts[0] if artifacts else None,
            "artifact_kind": artifacts[0].split("-")[0] if artifacts else None,
            "crashed": False, "reproduces_on_vul": None, "fix_returncode": None,
            "verified": False, "crash_frames": None, "crash_path": None,
        }

        crash_files = [a for a in artifacts if a.startswith("crash-")]
        if crash_files:
            with open(os.path.join(art_dir, crash_files[0]), "rb") as f:
                data = f.read()
            dest = os.path.join(args.out, "crashes", tag + "_" + crash_files[0])
            with open(dest, "wb") as f:
                f.write(data)
            checker = ArvoEnv(args.case, [b"x"], use_coverage=False)
            try:
                check = checker._verify(data)
                vul = checker._run_plain(checker.binary, data)
            finally:
                checker.close()
            result.update({
                "crashed": True, "crash_path": os.path.relpath(dest, root),
                "reproduces_on_vul": check["reproduces_on_vul"],
                "fix_returncode": check["fix_returncode"],
                "verified": bool(check["verified"]),
                "crash_frames": vul.get("frames"),
            })

        with open(os.path.join(args.out, tag + ".json"), "w") as f:
            json.dump(result, f, indent=1)
        print("%s: execs=%s (%s/s) ft=%s artifact=%s verified=%s %.0fs"
              % (tag, execs, result["execs_per_second"], ft,
                 result["artifact"], result["verified"], wall))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
