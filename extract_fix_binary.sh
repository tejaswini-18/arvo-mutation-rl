#!/usr/bin/env bash
# Copy the fixed build's fuzz binary out of the -fix image so ArvoEnv._verify
# can run it inside the -vul container (PRD section 7, "Verifying a crash").
#
#   bash extract_fix_binary.sh 12950
#
# Writes fix_builds/<id>/ (git-ignored). Safe because both images share the
# same OS and system libraries; checked for 12950 (identical libc/libm md5,
# same ldd list). The script re-checks the libraries for every case and stops
# if they differ, and then re-runs the PoC on both binaries as a sanity check.
set -eu
export MSYS_NO_PATHCONV=1   # stop Git Bash on Windows rewriting /out paths
ID="${1:-12950}"
IMG_VUL="n132/arvo:${ID}-vul"
IMG_FIX="n132/arvo:${ID}-fix"
DEST="fix_builds/${ID}"

libs_sig='for f in $(ldd "$(find /out -maxdepth 1 -type f -perm -u+x -name "*fuzz*" | head -1)" | grep -o "/lib[^ ]*"); do md5sum "$f"; done'
vul_libs=$(docker run --rm "$IMG_VUL" bash -c "$libs_sig")
fix_libs=$(docker run --rm "$IMG_FIX" bash -c "$libs_sig")
if [ "$vul_libs" != "$fix_libs" ]; then
  echo "FAIL: system libraries differ between $IMG_VUL and $IMG_FIX;"
  echo "      the fix binary cannot be run in the -vul container."
  exit 1
fi
echo "OK: identical system libraries"

rm -rf "$DEST" && mkdir -p "$DEST"
cid=$(docker create "$IMG_FIX")
trap 'docker rm -f "$cid" >/dev/null' EXIT
docker cp "$cid:/out/." "$DEST/"
echo "copied /out of $IMG_FIX to $DEST:"
ls -la "$DEST"

echo "== sanity: PoC on the extracted fix binary, run inside $IMG_VUL"
BIN=$(basename "$(docker run --rm "$IMG_VUL" bash -c 'find /out -maxdepth 1 -type f -perm -u+x -name "*fuzz*" | head -1')")
# "Clean" means it actually ran and exited 0, not merely "no ASan line".
if docker run --rm -v "$(pwd)/$DEST:/fixout" "$IMG_VUL" bash -c \
     "cp /fixout/$BIN /tmp/fixbin && chmod +x /tmp/fixbin && /tmp/fixbin /tmp/poc" \
     > /dev/null 2>&1; then
  echo "PASS: extracted fix binary runs the PoC and exits 0"
else
  echo "FAIL: extracted fix binary crashed or did not run on the PoC"; exit 1
fi
