#!/usr/bin/env bash
# Run the experiment matrix with one container per (case, setting, seed group),
# all at once. Same runs as run_all.sh, using more of the machine's cores.
#
#   T=5000 OUT=results/full_T5000 bash run_parallel.sh
#
# Seeds are split into two groups ("0 1 2" and "3 4") so 6 cases x 2 settings
# x 2 groups = 24 containers. Each container writes to
# $OUT/parts/<case>_<setting>_g<group>/ (separate files, so no two containers
# append to the same CSV). Afterwards run
#   python merge_results.py --out results/full_T5000
# to build $OUT/<case>/runs.csv for metrics.py. Safe to re-run: finished runs
# are skipped.
set -u
export MSYS_NO_PATHCONV=1
T="${T:-5000}"
OUT="${OUT:-results/full_T5000}"
GROUPS_SEEDS=("0 1 2" "3 4")
CASES="${CASES:-12950 13222 3848 10304 3522 16091}"

mkdir -p "$OUT/parts"
for id in $CASES; do
  for s in A B; do
    g=0
    for seeds in "${GROUPS_SEEDS[@]}"; do
      g=$((g + 1))
      part="$OUT/parts/${id}_${s}_g${g}"
      mkdir -p "$part"
      docker run --rm --name "arvo_${id}_${s}_g${g}" -v "$(pwd):/work" -w /work \
        "n132/arvo:${id}-vul" \
        python3 -u run_experiment.py --case "$id" --settings "$s" \
          --seeds $seeds --T "$T" --out "$part" > "$part/log.txt" 2>&1 &
    done
  done
done
echo "started $(jobs -p | wc -l) containers, T=$T, out=$OUT  $(date '+%H:%M:%S')"
wait
echo "all done $(date '+%H:%M:%S')"
