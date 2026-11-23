#!/usr/bin/env bash
# Reference runs of plain libFuzzer (run_libfuzzer.py) on the cases our agents
# never crashed, all jobs at once. Not part of the agent comparison.
#
#   bash run_libfuzzer.sh                               # 60 min, seeds 0-2
#   MINUTES=5 FZSEEDS="0" OUT=results/lf_trial bash run_libfuzzer.sh 3848
#
# EXTRA="<case>:<label>:<seeds dir>" adds jobs with another seed set, e.g.
#   EXTRA="10304:handmade:results/libfuzzer/seeds_10304_handmade"
# Each job writes <OUT>/<case>_<label>_seed<N>.json and .log.
set -u
export MSYS_NO_PATHCONV=1   # stop Git Bash on Windows rewriting /work paths
MINUTES="${MINUTES:-60}"
FZSEEDS="${FZSEEDS:-0 1 2}"
OUT="${OUT:-results/libfuzzer}"
EXTRA="${EXTRA:-}"
CASES="${*:-3848 10304 3522 16091}"
export MINUTES OUT

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not responding; start Docker Desktop first." >&2
  exit 1
fi
mkdir -p "$OUT"

run_job() {
  id="$1"; label="$2"; dir="$3"; seed="$4"
  extra_args=""
  [ "$dir" != "-" ] && extra_args="--seeds-dir $dir"
  echo "start ${id}_${label}_seed${seed}  $(date '+%H:%M:%S')"
  if docker run --rm -v "$(pwd):/work" -w /work "n132/arvo:${id}-vul" \
       python3 -u run_libfuzzer.py --case "$id" --seed "$seed" \
         --minutes "$MINUTES" --label "$label" --out "$OUT" $extra_args \
       >"$OUT/${id}_${label}_seed${seed}.stdout" 2>&1; then
    echo "done  $(tail -1 "$OUT/${id}_${label}_seed${seed}.stdout")"
  else
    echo "FAIL  ${id}_${label}_seed${seed} (see $OUT/${id}_${label}_seed${seed}.stdout)"
  fi
}
export -f run_job

{
  for id in $CASES; do for s in $FZSEEDS; do echo "$id seeds - $s"; done; done
  for e in $EXTRA; do
    IFS=: read -r id label dir <<<"$e"
    for s in $FZSEEDS; do echo "$id $label $dir $s"; done
  done
} > "$OUT/jobs.txt"
n=$(wc -l < "$OUT/jobs.txt")
echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $n jobs  ${MINUTES} min each  out=$OUT"
xargs -P "$n" -L 1 bash -c 'run_job "$0" "$1" "$2" "$3"' < "$OUT/jobs.txt"
echo "=== all done $(date '+%Y-%m-%d %H:%M:%S')"
