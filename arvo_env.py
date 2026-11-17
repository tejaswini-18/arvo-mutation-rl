"""ArvoEnv: wraps one ARVO fuzz target as an RL environment (PRD sections 5 and 7).

Runs INSIDE the -vul container (Python 3.5.2), with the project mounted at
/work. Each step runs the fuzz binary once on one input, in libFuzzer's merge
mode so it writes out the input's exact feature set:

    <fuzz_binary> -merge=1 -save_coverage_summary=<file> <empty dir> <dir with the input>

(newer libFuzzer builds use -merge_control_file=<file> instead; see
_detect_coverage_mode).

Coverage
--------
# NOTE: the PRD says "new coverage means the cumulative ft: rose since the
# last best", using `-runs=0`. Every execution is a fresh process, so ft:
# counts only what that ONE input reached, not a running total; comparing it
# to the best so far would reward inputs that are merely bigger, not inputs
# that reach new code. `-print_pcs=1` prints nothing in this libFuzzer
# version (checked on 12950). Merge mode's coverage summary (or, in newer
# builds, its control file) lists the IDs of the features ft: counts. For
# the input "hello" the count equals -runs=0's ft: on 5 of the 6 cases; on
# 10304 (libxml2) merge mode reports 508 vs ft: 455, consistently, so novelty
# within a run is still measured on one fixed feature definition.
# We keep their union over the episode in
# Python. "New coverage features" = feature IDs never seen before in this
# episode; the per-input ft is logged too. (Loading and saving the summary
# in the same file is broken in this libFuzzer, hence the union in Python.)
# If a target writes no summary we fall back to the crash-only reward (PRD
# section 7) and record coverage_available=False.

Crashes are detected from "ERROR: AddressSanitizer" in the output.

Verification (_verify)
----------------------
A crash counts only if (1) it reproduces on the -vul binary and (2) the -fix
binary runs the same input and exits 0 with no ASan report (PRD section 7).
# NOTE: the -fix binary is copied out of the -fix image by
# extract_fix_binary.sh and run inside this -vul container, instead of
# starting a -fix container per crash (we cannot run docker from in here).
# This is sound because both images have identical system libraries; the
# script checks that for each case and re-runs the PoC on both binaries.
# If the fix binary is missing, verified is recorded as None (unknown), never
# as True.
"""

import os
import random
import re
import shutil
import subprocess
import tempfile
import time

import config
import mutators

NONE_ACTION = 8                      # "no previous action yet"
SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE = 0, 1, 2
NUM_STATES = 9 * 2 * 3               # 54

_FRAME_RE = re.compile(r"^\s*#(\d+) 0x[0-9a-fA-F]+ in (.+)$")
_ASAN_MARKER = "ERROR: AddressSanitizer"


def size_bucket(n):
    if n < 64:
        return SIZE_SMALL
    if n <= 1024:
        return SIZE_MEDIUM
    return SIZE_LARGE


def state_index(state):
    """Map (previous_action, got_new_coverage, size_bucket) to 0..53."""
    prev, new_cov, size = state
    return (prev * 2 + new_cov) * 3 + size


def crash_signature(frames):
    """Two crashes are the same bug if their top three ASan frames match."""
    return tuple(frames[:3])


def dedup_crashes(crashes):
    """Keep the first crash of each signature (crashes: dicts with 'frames')."""
    seen = set()
    kept = []
    for c in crashes:
        sig = crash_signature(c["frames"])
        if sig not in seen:
            seen.add(sig)
            kept.append(c)
    return kept


def parse_summary(text):
    """Feature IDs from a coverage summary: lines like 'in/x size: 5 features:  0 18 28 ...'."""
    features = set()
    for line in text.splitlines():
        if "features:" in line:
            features.update(line.split("features:", 1)[1].split())
    return features


def parse_control_file(text):
    """Feature IDs from a merge control file (newer libFuzzer).

    Feature lines are 'FT <file index> f1 f2 ...' (newest builds) or
    'DONE <file index> f1 f2 ...' (2019-era builds). 'COV' lines (edge PCs)
    are ignored: features are what ft: counts.
    """
    features = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("FT", "DONE"):
            features.update(parts[2:])
    return features


def parse_output(text):
    """Pull crash and stack info out of one libFuzzer run's output."""
    crashed = _ASAN_MARKER in text
    frames = []
    if crashed:
        after = text[text.index(_ASAN_MARKER):]
        for line in after.splitlines():
            m = _FRAME_RE.match(line)
            if m:
                frames.append(m.group(2).strip())
            elif frames:
                break   # end of the first stack trace
    return {"crashed": crashed, "frames": frames}


class ArvoEnv(object):

    def __init__(self, case_id, seeds, rng_seed=0, T=config.DEFAULT_T,
                 crash_dir=None, use_coverage=True):
        """
        case_id:   key into config.CASES, e.g. "12950"
        seeds:     list of bytes, the initial seed pool
        rng_seed:  seeds the env's RNG (seed choice and mutations)
        T:         execution budget per episode
        crash_dir: where crashing inputs are saved (None = don't save)
        """
        if not seeds:
            raise ValueError("need at least one seed")
        self.case = config.CASES[case_id]
        self.case_id = case_id
        self.binary = self.case["fuzz_binary"]
        self.initial_seeds = [bytes(s) for s in seeds]
        self.rng_seed = rng_seed
        self.T = T
        self.crash_dir = crash_dir
        self.use_coverage = use_coverage

        self.env = dict(os.environ)
        self.env["ASAN_OPTIONS"] = config.ASAN_OPTIONS
        # Scratch dir in the container's own filesystem: much faster than the
        # Windows bind mount at /work.
        self.scratch = tempfile.mkdtemp(prefix="arvoenv_")
        self.input_dir = os.path.join(self.scratch, "in")
        self.merge_out_dir = os.path.join(self.scratch, "out")
        os.makedirs(self.input_dir)
        os.makedirs(self.merge_out_dir)
        self.input_path = os.path.join(self.input_dir, "input")
        self.summary_path = os.path.join(self.scratch, "summary")
        # libFuzzer's merge writes its control file to $TMPDIR; keep it here.
        self.env["TMPDIR"] = self.scratch
        self.verify_input_path = os.path.join(self.scratch, "verify_input")
        self.fix_binary = self._install_fix_binary()
        self.coverage_mode = self._detect_coverage_mode()

    def _detect_coverage_mode(self):
        """Which file libFuzzer can write the input's feature list to.

        # NOTE: the six ARVO builds span three libFuzzer versions. Older ones
        # only have -save_coverage_summary, the newest only have
        # -merge_control_file. Both list the same features that ft: counts.
        """
        proc = subprocess.run([self.binary, "-help=1"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, env=self.env, timeout=60)
        flags = proc.stdout.decode("utf-8", "replace")
        if re.search(r"^\s*save_coverage_summary\b", flags, re.M):
            return "summary"
        if re.search(r"^\s*merge_control_file\b", flags, re.M):
            return "control"
        return None   # no per-input features: crash-only reward

    def _install_fix_binary(self):
        """Copy the -fix /out folder into scratch and return its fuzz binary.

        The whole folder is copied because some harnesses load files next to
        themselves (file's magic_fuzzer reads magic.mgc); copying also restores
        the exec bit, which the Windows bind mount may drop.
        """
        rel = self.case.get("fix_dir")
        if not rel:
            return None
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
        name = os.path.basename(self.binary)
        if not os.path.isfile(os.path.join(src_dir, name)):
            print("WARNING: %s/%s not found; run `bash extract_fix_binary.sh %s` "
                  "on the host. Crashes will be recorded as verified=None."
                  % (src_dir, name, self.case_id))
            return None
        dst_dir = os.path.join(self.scratch, "fix_out")
        shutil.copytree(src_dir, dst_dir)
        dst = os.path.join(dst_dir, name)
        os.chmod(dst, 0o755)
        return dst

    def close(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    # ------------------------------------------------------------------ RL API

    def reset(self):
        self.rng = random.Random(self.rng_seed)
        self.pool = list(self.initial_seeds)
        self.covered = set()
        self.executions = 0
        self.crashes = []
        self.coverage_curve = []
        self.operator_counts = [0] * mutators.NUM_ACTIONS
        self.coverage_available = False

        # NOTE: the seeds are run once to establish baseline coverage. These
        # runs do not count toward the budget T; their number is in info.
        self.seed_executions = 0
        for seed in self.pool:
            result = self._run_input(seed)
            self.seed_executions += 1
            self.covered |= result["features"]
            if result["features"]:
                self.coverage_available = True
        if not self.use_coverage:
            self.coverage_available = False

        # Proposal Algorithm 1 / Eq. 1: the seed for step t is drawn BEFORE the
        # agent chooses, and the state's size bucket z_t is that seed's size.
        self.current_seed = self.rng.choice(self.pool)   # uniform for all agents
        return (NONE_ACTION, 0, size_bucket(len(self.current_seed)))

    def step(self, action):
        seed = self.current_seed
        child = mutators.mutate(action, seed, self.rng, self.pool)
        result = self._run_input(child)
        self.executions += 1
        self.operator_counts[action] += 1

        new_features = len(result["features"] - self.covered)
        self.covered |= result["features"]
        got_new = 1 if new_features > 0 else 0
        if got_new:
            self.pool.append(child)
        self.coverage_curve.append(len(self.covered))

        crashed = 1 if result["crashed"] else 0
        if self.coverage_available:
            reward = min(new_features, 10) / 10.0 + 10 * crashed
        else:
            reward = 10.0 * crashed               # crash-only fallback

        crash_path = None
        check = {"verified": None, "reproduces_on_vul": None,
                 "fix_returncode": None}
        if crashed:
            crash_path = self._save_crash(child)
            check = self._verify(child)
            self.crashes.append({"exec": self.executions, "path": crash_path,
                                 "frames": result["frames"][:3],
                                 "verified": check["verified"],
                                 "reproduces_on_vul": check["reproduces_on_vul"],
                                 "fix_returncode": check["fix_returncode"]})

        # Draw the seed for the NEXT step now (after the pool update, so a new
        # child can be picked) so the next state carries its size bucket.
        self.current_seed = self.rng.choice(self.pool)
        next_state = (action, got_new, size_bucket(len(self.current_seed)))
        # NOTE: per the PRD the episode ends at the first crash, verified or
        # not; an unverified first crash is recorded as such, not skipped.
        done = bool(crashed) or self.executions >= self.T
        info = {
            "executions": self.executions,
            "new_features": new_features,
            "total_coverage": len(self.covered),
            "input_ft": len(result["features"]),
            "crashed": bool(crashed),
            "crash_path": crash_path,
            "frames": result["frames"][:3],
            "verified": check["verified"],
            "reproduces_on_vul": check["reproduces_on_vul"],
            "fix_returncode": check["fix_returncode"],
            "pool_size": len(self.pool),
            "exec_seconds": result["seconds"],
        }
        return next_state, reward, done, info

    # --------------------------------------------------------------- internals

    def _run_input(self, data):
        """Run the fuzz binary on one input; return parsed coverage and crash info."""
        with open(self.input_path, "wb") as f:
            f.write(data)
        for name in os.listdir(self.merge_out_dir):
            os.remove(os.path.join(self.merge_out_dir, name))
        if os.path.exists(self.summary_path):
            os.remove(self.summary_path)
        # -artifact_prefix stops libFuzzer dropping crash-<sha1> files into the
        # current directory (the project folder); we save crashes ourselves.
        cmd = [self.binary, "-merge=1",
               "-artifact_prefix=" + self.scratch + "/"]
        if self.coverage_mode == "summary":
            cmd.append("-save_coverage_summary=" + self.summary_path)
        elif self.coverage_mode == "control":
            cmd.append("-merge_control_file=" + self.summary_path)
        cmd += [self.merge_out_dir, self.input_dir]
        start = time.time()
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=self.env,
                                  timeout=config.EXEC_TIMEOUT)
            text = proc.stdout.decode("utf-8", "replace")
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            # NOTE: a hang is not an ASan crash; treat it as "no crash, no
            # coverage" and record it.
            text, returncode = "", None
        result = parse_output(text)
        # NOTE: a crashing input has no (complete) feature line, so its
        # coverage counts as empty.
        result["features"] = set()
        if self.coverage_mode and os.path.exists(self.summary_path):
            with open(self.summary_path) as f:
                text = f.read()
            if self.coverage_mode == "summary":
                result["features"] = parse_summary(text)
            else:
                result["features"] = parse_control_file(text)
        result["returncode"] = returncode
        result["timed_out"] = returncode is None
        result["seconds"] = time.time() - start
        result["output"] = text
        return result

    def _run_plain(self, binary, data):
        """Run a binary once on one input file, the way the arvo script does."""
        with open(self.verify_input_path, "wb") as f:
            f.write(data)
        cmd = [binary, "-artifact_prefix=" + self.scratch + "/",
               self.verify_input_path]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=self.env,
                                  timeout=config.EXEC_TIMEOUT)
            text = proc.stdout.decode("utf-8", "replace")
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            text, returncode = "", None
        result = parse_output(text)
        result["returncode"] = returncode
        return result

    def _verify(self, data):
        """PRD section 7: keep a crash only if it reproduces on -vul and the
        -fix build exits cleanly (exit code 0, no ASan report, no timeout)."""
        vul = self._run_plain(self.binary, data)
        out = {"reproduces_on_vul": vul["crashed"], "fix_returncode": None,
               "verified": None}
        if self.fix_binary is None:
            return out                        # unknown, never claimed True
        fix = self._run_plain(self.fix_binary, data)
        out["fix_returncode"] = fix["returncode"]
        fix_clean = fix["returncode"] == 0 and not fix["crashed"]
        out["verified"] = bool(vul["crashed"] and fix_clean)
        return out

    def _save_crash(self, data):
        if not self.crash_dir:
            return None
        if not os.path.isdir(self.crash_dir):
            os.makedirs(self.crash_dir)
        name = "crash_%s_seed%d_exec%d" % (self.case_id, self.rng_seed,
                                           self.executions)
        path = os.path.join(self.crash_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path
