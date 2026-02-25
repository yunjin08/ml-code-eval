"""
PaC (Semgrep) setup verification — run before trusting any PaC results.

Runs known-bad C snippets that Semgrep rules MUST detect. If any required
check fails, exit non-zero so CI/pipeline or manual runs know the setup is broken.

Usage:
  python src/verify_pac_setup.py          # exit 0 = OK, exit 1 = broken
  python src/phase3_experiment.py --verify_pac  # Phase 3 runs this first and aborts if fail
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Snippets that Semgrep p/c (or our configs) MUST detect. (name, code, min_expected_count)
REQUIRED_CHECKS = [
    ("gets() - CWE-676", "int main() { char buf[64]; gets(buf); return 0; }", 1),
]


def run_verification() -> tuple[bool, list[str]]:
    """
    Run all REQUIRED_CHECKS via our Semgrep wrapper.
    Returns (all_passed, list of failure messages).
    """
    from utils.semgrep_runner import run_semgrep_on_code

    failures = []
    for name, code, min_count in REQUIRED_CHECKS:
        count, score = run_semgrep_on_code(code, ".c")
        if count < min_count:
            failures.append(f"PaC check '{name}': expected at least {min_count} finding, got {count} (score={score})")
        else:
            print(f"  OK: {name} -> count={count}, score={score}")

    return len(failures) == 0, failures


def main() -> int:
    print("PaC (Semgrep) verification: known-bad patterns must be detected.")
    passed, failures = run_verification()
    if passed:
        print("Verification PASSED. PaC setup is trusted for Phase 3.")
        return 0
    print("Verification FAILED. Do not trust PaC results until this passes.")
    for msg in failures:
        print(f"  - {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
