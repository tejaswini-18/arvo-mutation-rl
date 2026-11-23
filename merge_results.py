"""Merge results/<out>/parts/<case>_<setting>/ into results/<out>/<case>/ so
metrics.py can read them (it expects one runs.csv + runs.jsonl per case).

    python merge_results.py --out results/full
"""

import argparse
import csv
import glob
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/full")
    args = ap.parse_args()
    csv.field_size_limit(sys.maxsize)

    cases = sorted(set(os.path.basename(p).split("_")[0]
                       for p in glob.glob(os.path.join(args.out, "parts", "*_*"))))
    for case in cases:
        dest = os.path.join(args.out, case)
        os.makedirs(dest, exist_ok=True)
        rows, fields = [], None
        for part in sorted(glob.glob(os.path.join(args.out, "parts", case + "_*"))):
            p = os.path.join(part, "runs.csv")
            if os.path.exists(p):
                with open(p, newline="") as f:
                    r = csv.DictReader(f)
                    fields = r.fieldnames
                    rows.extend(r)
        with open(os.path.join(dest, "runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(dest, "runs.jsonl"), "w") as out:
            for part in sorted(glob.glob(os.path.join(args.out, "parts", case + "_*"))):
                p = os.path.join(part, "runs.jsonl")
                if os.path.exists(p):
                    with open(p) as f:
                        shutil.copyfileobj(f, out)
        print("case %s: %d runs" % (case, len(rows)))


if __name__ == "__main__":
    main()
