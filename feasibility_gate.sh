#!/usr/bin/env bash
# ARVO feasibility gate for the RL-fuzzing project.
# Run on your own machine (Linux, macOS, or Windows WSL2) with Docker running:
#   bash feasibility_gate.sh 12950
# Default case is 12950 (pcre2). Each image is several GB; check free disk first.
set -u
ID="${1:-12950}"
IMG_VUL="n132/arvo:${ID}-vul"
IMG_FIX="n132/arvo:${ID}-fix"

echo "== Check 1: Docker works"
docker version --format '{{.Server.Version}}' || { echo "FAIL: Docker not running"; exit 1; }

echo "== Check 2: pull images (this can take a while)"
docker pull "$IMG_VUL" || { echo "FAIL: cannot pull $IMG_VUL"; exit 1; }
docker pull "$IMG_FIX" || echo "WARN: cannot pull $IMG_FIX (crash verification will be unavailable)"

echo "== Check 3: PoC crashes the vulnerable build (expect an AddressSanitizer report)"
docker run --rm "$IMG_VUL" arvo 2>&1 | grep -m3 -E "ERROR: AddressSanitizer|SUMMARY" \
  && echo "PASS: vulnerable build crashes" || echo "FAIL: no ASan crash seen"

echo "== Check 4: PoC does NOT crash the fixed build"
docker run --rm "$IMG_FIX" arvo 2>&1 | grep -q "ERROR: AddressSanitizer" \
  && echo "FAIL: fixed build still crashes" || echo "PASS: fixed build is clean"

echo "== Check 5: where the fuzz binary, PoC, and arvo script live"
docker run --rm "$IMG_VUL" bash -c 'echo "--- /out:"; ls /out | head -20; echo "--- arvo script:"; cat "$(command -v arvo)" | head -40'

echo "== Check 6: harness reports coverage (needed for the RL reward)"
docker run --rm "$IMG_VUL" bash -c '
  BIN=$(find /out -maxdepth 1 -type f -perm -u+x -name "*fuzz*" | head -1)
  echo "using binary: $BIN"
  mkdir -p /tmp/seeds && printf "hello" > /tmp/seeds/s1
  "$BIN" -runs=0 /tmp/seeds 2>&1 | grep -m2 -E "INITED|cov:" \
    && echo "PASS: coverage readout works" || echo "FAIL: no coverage line (use crash-only reward fallback)"
'

echo "== Check 7: python available inside the container (to run the agent in-container)"
docker run --rm "$IMG_VUL" bash -c 'command -v python3 && python3 --version || echo "WARN: no python3 in image"'

echo "== Disk used by ARVO images"
docker images n132/arvo --format '{{.Tag}}  {{.Size}}'
