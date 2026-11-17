"""Re-check every saved crash against the REAL -vul and -fix Docker images.

Inside the experiment, ArvoEnv._verify runs an extracted copy of the fix
binary (see extract_fix_binary.sh). The proposal says crashes are re-run "on
the fixed image"; this host-side script does exactly that, as an independent
check of the in-container result:

    python verify_on_fix_image.py --results results/full_T5000

For each crash file under <results>/parts/*/crashes/*/crash_<case>_*, it runs
    docker run n132/arvo:<case>-vul <fuzz binary> <file>   -> expect an ASan crash
    docker run n132/arvo:<case>-fix <fuzz binary> <file>   -> expect exit 0, no ASan
and writes <results>/fix_image_check.csv. Runs on the host (Python 3.6+).
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

import config


def run_image(image, binary, path):
    cmd = ["docker", "run", "--rm", "-e", "ASAN_OPTIONS=" + config.ASAN_OPTIONS,
           "-v", os.path.abspath(path) + ":/tmp/crash_input:ro",
           image, binary, "-artifact_prefix=/tmp/", "/tmp/crash_input"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=300)
    text = p.stdout.decode("utf-8", "replace")
    return p.returncode, "ERROR: AddressSanitizer" in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "parts", "*", "crashes",
                                          "*", "crash_*")))
    rows = []
    for path in files:
        case = os.path.basename(path).split("_")[1]
        binary = config.CASES[case]["fuzz_binary"]
        vul_rc, vul_asan = run_image("n132/arvo:%s-vul" % case, binary, path)
        fix_rc, fix_asan = run_image("n132/arvo:%s-fix" % case, binary, path)
        ok = bool(vul_asan and fix_rc == 0 and not fix_asan)
        rows.append({"case": case, "crash_file": os.path.relpath(path, args.results),
                     "vul_asan_crash": vul_asan, "fix_returncode": fix_rc,
                     "fix_asan_crash": fix_asan, "verified_on_images": ok})
        print("%s %-60s vul_crash=%s fix_rc=%s -> %s"
              % (case, os.path.basename(os.path.dirname(path)), vul_asan, fix_rc,
                 "VERIFIED" if ok else "not verified"))
        sys.stdout.flush()

    out = os.path.join(args.results, "fix_image_check.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    n_ok = sum(1 for r in rows if r["verified_on_images"])
    print("%d crash files; %d verified on the real -vul/-fix images -> %s"
          % (len(rows), n_ok, out))


if __name__ == "__main__":
    main()
