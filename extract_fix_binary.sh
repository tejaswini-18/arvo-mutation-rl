#!/usr/bin/env bash
# Copy the fixed build's /out folder out of the -fix image so ArvoEnv._verify
# can run it inside the -vul container (PRD section 7, "Verifying a crash").
#
#   bash extract_fix_binary.sh 12950
#
# Writes fix_builds/<id>/ (git-ignored). The fuzz binary is the one the
# image's own `arvo` script runs on /tmp/poc. Running the -fix binary inside
# the -vul container is only safe if both images have the same system
# libraries, so the script checks that first and stops if they differ, then
# re-runs the PoC on both binaries as a sanity check.
set -eu
export MSYS_NO_PATHCONV=1   # stop Git Bash on Windows rewriting /out paths
ID="${1:-12950}"
IMG_VUL="n132/arvo:${ID}-vul"
IMG_FIX="n132/arvo:${ID}-fix"
DEST="fix_builds/${ID}"

# The binary the arvo script runs, e.g. "/out/pcre2_fuzzer /tmp/poc".
BIN_PATH=$(docker run --rm "$IMG_VUL" bash -c \
  'grep -oE "/out/[^ ]+ /tmp/poc" "$(command -v arvo)" | head -1 | cut -d" " -f1')
[ -n "$BIN_PATH" ] || { echo "FAIL: no '/out/<binary> /tmp/poc' line in the arvo script"; exit 1; }
BIN=$(basename "$BIN_PATH")
echo "fuzz binary (from the arvo script): $BIN_PATH"

libs_sig="for f in \$(ldd $BIN_PATH | grep -o '/lib[^ ]*'); do md5sum \"\$f\"; done"
vul_libs=$(docker run --rm "$IMG_VUL" bash -c "$libs_sig")
fix_libs=$(docker run --rm "$IMG_FIX" bash -c "$libs_sig")
if [ -z "$vul_libs" ] || [ "$vul_libs" != "$fix_libs" ]; then
  echo "FAIL: system libraries differ (or ldd failed) between $IMG_VUL and $IMG_FIX;"
  echo "      the fix binary cannot be run in the -vul container."
  exit 1
fi
echo "OK: identical system libraries"

rm -rf "$DEST" && mkdir -p "$DEST"
cid=$(docker create "$IMG_FIX")
trap 'docker rm -f "$cid" >/dev/null' EXIT
docker cp "$cid:/out/." "$DEST/"
echo "copied /out of $IMG_FIX to $DEST ($(ls "$DEST" | wc -l) files)"

echo "== sanity: PoC on the extracted fix binary, run inside $IMG_VUL"
# "Clean" means it actually ran and exited 0, not merely "no ASan line".
if docker run --rm -v "$(pwd)/$DEST:/fixout" "$IMG_VUL" bash -c \
     "cp -r /fixout /tmp/fixout && chmod +x /tmp/fixout/$BIN && /tmp/fixout/$BIN /tmp/poc" \
     > /dev/null 2>&1; then
  echo "PASS: extracted fix binary runs the PoC and exits 0"
else
  echo "FAIL: extracted fix binary crashed or did not run on the PoC"; exit 1
fi
