#!/usr/bin/env bash
# Run the full experiment matrix (PRD section 8): 6 cases x 2 settings x
# 3 agents x 5 seeds = 180 runs. One -vul container per case; each writes to
# results/<out>/<case>/. Safe to re-run: finished runs are skipped.
#
#   bash run_all.sh                      # T=5000, results/full
#   T=1000 OUT=results/trial bash run_all.sh 12950 13222
#
# Needs: the -vul images, fix_builds/<id>/ (extract_fix_binary.sh) and
# seeds/<id>/ (build_seeds.py) for every case.
set -u
export MSYS_NO_PATHCONV=1   # stop Git Bash on Windows rewriting /work paths
T="${T:-5000}"
OUT="${OUT:-results/full}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CASES="${*:-12950 13222 3848 10304 3522 16091}"

for id in $CASES; do
  echo "=== case $id  T=$T  $(date '+%Y-%m-%d %H:%M:%S')"
  mkdir -p "$OUT/$id"
  docker run --rm -v "$(pwd):/work" -w /work "n132/arvo:${id}-vul" \
    python3 -u run_experiment.py --case "$id" --settings A B \
      --seeds $SEEDS --T "$T" --out "$OUT/$id" \
    || echo "!!! case $id exited with an error"
done
echo "=== all done $(date '+%Y-%m-%d %H:%M:%S')"
