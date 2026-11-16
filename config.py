"""Per-case settings discovered during feasibility (PRD section 7).

Only cases that have passed feasibility_gate.sh should be listed here.
"fuzz_binary" is the program the image's own `arvo` script runs on /tmp/poc
(NOT simply the first /out/*fuzz* file: for 3848 and 16091 that picks a
different harness, and zstd's harnesses have no "fuzz" in their names).
"fix_dir" is the -fix image's /out, copied by `bash extract_fix_binary.sh <id>`
(relative to the project root); the whole folder is used because some
harnesses load data files next to themselves (file's magic.mgc).
Python 3.5 compatible.
"""

CASES = {
    "12950": {
        "project": "pcre2",
        "input_format": "regex text",
        "fuzz_binary": "/out/pcre2_fuzzer",
        "poc_path": "/tmp/poc",
        "fix_dir": "fix_builds/12950",
    },
    "13222": {
        "project": "file",
        "input_format": "any bytes",
        "fuzz_binary": "/out/magic_fuzzer",
        "poc_path": "/tmp/poc",
        "fix_dir": "fix_builds/13222",
    },
    "3848": {
        "project": "yara",
        "input_format": "PE binary",
        "fuzz_binary": "/out/pe_fuzzer",
        "poc_path": "/tmp/poc",
        "fix_dir": "fix_builds/3848",
    },
    "10304": {
        "project": "libxml2",
        "input_format": "XML",
        "fuzz_binary": "/out/libxml2_xml_read_memory_fuzzer",
        "poc_path": "/tmp/poc",
        "fix_dir": "fix_builds/10304",
    },
    "3522": {
        "project": "zstd",
        "input_format": "raw bytes",
        "fuzz_binary": "/out/block_round_trip",
        "poc_path": "/tmp/poc",
        "fix_dir": "fix_builds/3522",
    },
    "16091": {
        "project": "libspng",
        "input_format": "PNG",
        "fuzz_binary": "/out/spng_read_fuzzer",
        "poc_path": "/tmp/poc",
        "fix_dir": "fix_builds/16091",
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
