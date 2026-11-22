#!/usr/bin/env bash
# Run the experiment matrix in parallel: one container per
# case x setting x agent (6 x 2 x 3 = 36 jobs), each running its seeds.
# Results are identical to run_all.sh (every run seeds its own RNGs); only
# wall_seconds can rise a little when all cores are busy.
#
#   bash run_parallel.sh                          # T=5000, results/full, 24 jobs at once
#   JOBS=12 T=1000 OUT=results/trial bash run_parallel.sh 12950 13222
#   SEEDS="0 1 2 3 4 5 6 7 8 9" T=20000 OUT=results/big bash run_parallel.sh
#
# Each job writes <OUT>/<case>/runs.<setting>_<agent>.csv/.jsonl (metrics.py
# reads all of them) and logs to <OUT>/logs/<case>_<setting>_<agent>.log.
# Safe to re-run: runs already in any runs*.csv for the case are skipped.
#
# Needs: the -vul images, fix_builds/<id>/ and seeds/<id>/ for every case.
set -u
export MSYS_NO_PATHCONV=1   # stop Git Bash on Windows rewriting /work paths
T="${T:-5000}"
OUT="${OUT:-results/full}"
SEEDS="${SEEDS:-0 1 2 3 4}"
JOBS="${JOBS:-24}"
SETTINGS="${SETTINGS:-A B}"
AGENT_LIST="${AGENT_LIST:-random ucb1 qlearning}"
# Default order: slowest cases first (measured wall time in results/full).
CASES="${*:-3848 10304 3522 16091 13222 12950}"
export T OUT SEEDS

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not responding; start Docker Desktop first." >&2
  exit 1
fi

mkdir -p "$OUT/logs"
for id in $CASES; do mkdir -p "$OUT/$id"; done

run_job() {
  id="$1"; setting="$2"; agent="$3"
  tag="${setting}_${agent}"
  log="$OUT/logs/${id}_${tag}.log"
  echo "start $id $tag  $(date '+%H:%M:%S')"
  if docker run --rm -v "$(pwd):/work" -w /work "n132/arvo:${id}-vul" \
       python3 -u run_experiment.py --case "$id" --settings "$setting" \
         --agents "$agent" --seeds $SEEDS --T "$T" --out "$OUT/$id" \
         --tag "$tag" >"$log" 2>&1; then
    echo "done  $id $tag  $(date '+%H:%M:%S')"
  else
    echo "FAIL  $id $tag  (see $log)"
  fi
}
export -f run_job

echo "=== $(date '+%Y-%m-%d %H:%M:%S')  T=$T  seeds=[$SEEDS]  jobs=$JOBS  out=$OUT"
# Jobs start in this order, so the long cases are not left running alone at the end.
for id in $CASES; do
  for s in $SETTINGS; do
    for a in $AGENT_LIST; do
      echo "$id $s $a"
    done
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_job "$0" "$1" "$2"'
echo "=== all done $(date '+%Y-%m-%d %H:%M:%S')"
