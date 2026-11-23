"""Build the Setting A seed set for one case (PRD section 8, build step 7).

Run INSIDE the case's -vul container, from the project root:

    docker run --rm -v "$(pwd):/work" -w /work n132/arvo:12950-vul \
        python3 build_seeds.py --case 12950

Writes seeds/<case>/ (committed, so every run uses the same files).

# NOTE: where an image ships an OSS-Fuzz seed corpus (/out/<harness>_seed_corpus.zip)
# we use it; otherwise (12950 pcre2, 10304 libxml2) three small hand-made valid
# files, as in the proposal (P4). Each case has a deterministic rule (SOURCES
# below) that lists candidates in a fixed order; we keep the first MAX_SEEDS
# that are non-empty, at most MAX_SEED_BYTES, and do NOT crash the -vul build
# (a seed that already crashes would make the experiment meaningless). For
# file collections the order is "spread": candidates sorted by size, and the
# first picks evenly spaced across that range, so the seed set is neither
# all-tiny nor all-large. The rule and kept files are listed in
# seeds/<case>/SOURCES.txt.
Python 3.5 compatible.
"""

import argparse
import os
import shutil
import zipfile

from arvo_env import ArvoEnv

MAX_SEEDS = 10
# 8 KiB so all five of yara's PE corpus files (2-7.5 KiB) are eligible.
MAX_SEED_BYTES = 8192


def spread_order(items, k=MAX_SEEDS):
    """items sorted by size; return k evenly spaced ones first, then the rest."""
    n = len(items)
    if n <= k:
        return list(items)
    picks = sorted(set(int(round(i * (n - 1) / float(k - 1))) for i in range(k)))
    rest = [i for i in range(n) if i not in set(picks)]
    return [items[i] for i in picks] + [items[i] for i in rest]


def _eligible_sorted(named):
    """(name, data) pairs within the size cap, sorted by (size, name)."""
    keep = [(n, d) for n, d in named if 0 < len(d) <= MAX_SEED_BYTES]
    return sorted(keep, key=lambda x: (len(x[1]), x[0]))


def corpus_zip(path):
    def gen():
        z = zipfile.ZipFile(path)
        named = [(i.filename, z.read(i.filename)) for i in z.infolist()
                 if not i.filename.endswith("/") and 0 < i.file_size <= MAX_SEED_BYTES]
        for item in spread_order(_eligible_sorted(named)):
            yield item
    return gen


def handmade(files):
    """A fixed list of small, valid, hand-written inputs (proposal P4)."""
    def gen():
        for i, data in enumerate(files):
            yield ("handmade_%d" % i, data)
    return gen


# Proposal P4: "the project's OSS-Fuzz seed corpus if one is shipped in the
# image, otherwise three small hand-made valid files of the target's format."
HANDMADE_PCRE2 = [
    b"a+b",
    b"(foo|bar)[0-9]{2,5}",
    br"^(?<name>\w+)\s*$",
]
HANDMADE_LIBXML2 = [
    b'<?xml version="1.0"?><root><a x="1">text</a></root>',
    b'<doc xmlns:p="urn:x"><p:item id="1"/><!-- c --></doc>',
    b'<!DOCTYPE r [<!ENTITY e "v">]><r>&e;</r>',
]


# case -> (human-readable rule, candidate generator)
SOURCES = {
    "12950": ("three hand-made valid regexes (no seed corpus in the image)",
              handmade(HANDMADE_PCRE2)),
    "13222": ("OSS-Fuzz corpus /out/magic_fuzzer_seed_corpus.zip, spread by size",
              corpus_zip("/out/magic_fuzzer_seed_corpus.zip")),
    "3848": ("OSS-Fuzz corpus /out/pe_fuzzer_seed_corpus.zip, spread by size",
             corpus_zip("/out/pe_fuzzer_seed_corpus.zip")),
    "10304": ("three hand-made valid XML documents (no seed corpus in the image)",
              handmade(HANDMADE_LIBXML2)),
    "3522": ("OSS-Fuzz corpus /out/block_round_trip_seed_corpus.zip, spread by size",
             corpus_zip("/out/block_round_trip_seed_corpus.zip")),
    "16091": ("OSS-Fuzz corpus /out/spng_read_fuzzer_seed_corpus.zip, spread by size, "
              "each prefixed with 2 harness option bytes 00 01",
              corpus_zip("/out/spng_read_fuzzer_seed_corpus.zip")),
}

# NOTE: 16091's harness (tests/spng_read_fuzzer.cc) reads data[0] as decode
# flags and data[1] as output format + stream bit, and parses the PNG from
# data[2:]. OSS-Fuzz's build zips the raw test PNGs, so without a prefix every
# seed loses its first two PNG bytes and is rejected at the signature check
# (all seeds reached the same 34 features as "hello"). We prepend a fixed,
# neutral prefix: flags 0, format 1 (SPNG_FMT_RGBA8), buffer mode. We do NOT
# copy the PoC's own prefix, so Setting A carries no information from the PoC.
PREFIX = {"16091": b"\x00\x01"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    args = ap.parse_args()

    rule, gen = SOURCES[args.case]
    out = os.path.join("seeds", args.case)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    env = ArvoEnv(args.case, [b"x"], T=1)
    kept, seen, skipped_crash = [], set(), 0
    try:
        for name, data in gen():
            data = PREFIX.get(args.case, b"") + data
            if not data or len(data) > MAX_SEED_BYTES or data in seen:
                continue
            seen.add(data)
            if env._run_input(data)["crashed"]:
                skipped_crash += 1
                continue
            kept.append((name, data))
            if len(kept) == MAX_SEEDS:
                break
    finally:
        env.close()

    lines = ["case %s" % args.case, "rule: %s" % rule,
             "kept %d (max %d, <= %d bytes, non-crashing); skipped %d that crashed"
             % (len(kept), MAX_SEEDS, MAX_SEED_BYTES, skipped_crash), ""]
    for i, (name, data) in enumerate(kept):
        fname = "seed_%02d" % i
        with open(os.path.join(out, fname), "wb") as f:
            f.write(data)
        lines.append("%s  %5d bytes  from %s" % (fname, len(data), name))
    with open(os.path.join(out, "SOURCES.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
