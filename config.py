"""Per-case settings discovered during feasibility (PRD section 7).

Only cases that have passed feasibility_gate.sh should be listed here.
Python 3.5 compatible.
"""

CASES = {
    "12950": {
        "project": "pcre2",
        "input_format": "regex text",
        # From feasibility_12950.log, check 5.
        "fuzz_binary": "/out/pcre2_fuzzer",
        # The arvo script runs `/out/pcre2_fuzzer /tmp/poc`.
        "poc_path": "/tmp/poc",
        # Fixed build, copied out by `bash extract_fix_binary.sh 12950`.
        # Relative to the project root.
        "fix_binary": "fix_builds/12950/pcre2_fuzzer",
    },
}

# Copied from the arvo script inside the image (feasibility_12950.log), so our
# runs use the same sanitizer settings the dataset used to reproduce the bug.
ASAN_OPTIONS = (
    "alloc_dealloc_mismatch=0:allocator_may_return_null=1:"
    "allocator_release_to_os_interval_ms=500:check_malloc_usable_size=0:"
    "detect_container_overflow=1:detect_odr_violation=0:detect_leaks=0:"
    "detect_stack_use_after_return=1:fast_unwind_on_fatal=0:handle_abort=1:"
    "handle_segv=1:handle_sigill=1:max_uar_stack_size_log=16:"
    "print_scariness=1:quarantine_size_mb=10:strict_memcmp=1:"
    "strip_path_prefix=/workspace/:symbolize=1:use_sigaltstack=1:"
    "dedup_token_length=3"
)

# Seconds before one execution is abandoned (the arvo script uses -timeout=25).
EXEC_TIMEOUT = 25

# Default episode budget (PRD section 5).
DEFAULT_T = 5000
