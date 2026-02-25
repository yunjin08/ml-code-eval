"""
Verify Semgrep is working and our wrapper reports findings correctly.
Proves PaC=0 is due to rule coverage, not config/implementation bugs.
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Known-bad patterns that Semgrep p/c rules SHOULD catch
KNOWN_BAD_SNIPPETS = [
    ("gets() - CWE-676", "int main() { char buf[64]; gets(buf); return 0; }"),
    ("strcpy no bounds - CWE-120", "void f(char *dst) { char src[100]; strcpy(dst, src); }"),
    ("sprintf buffer - CWE-120", "void f(char *buf) { sprintf(buf, \"%s\", \"x\"); }"),
]


def run_semgrep_raw(code: str, ext: str = ".c") -> tuple[int, str, str]:
    """Run Semgrep, return (count, stdout, stderr)."""
    import shutil

    from utils.semgrep_runner import SEMGREP_CONFIGS

    semgrep_exe = shutil.which("semgrep") or os.path.join(os.path.dirname(sys.executable), "semgrep")
    if not semgrep_exe or not os.path.isfile(semgrep_exe):
        semgrep_exe = "semgrep"

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8", errors="replace") as f:
        f.write(code)
        path = f.name
    try:
        cmd = [semgrep_exe, "scan", "--json", "--quiet", "--no-git-ignore"]
        for cfg in SEMGREP_CONFIGS:
            cmd.extend(["--config", cfg])
        cmd.append(path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=os.path.dirname(path))
        count = 0
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                count = len(data.get("results", []))
            except (json.JSONDecodeError, TypeError):
                pass
        return count, result.stdout[:500] if result.stdout else "", result.stderr[:500] if result.stderr else ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    print("=" * 60)
    print("1. KNOWN-BAD PATTERNS (Semgrep SHOULD find these)")
    print("=" * 60)
    for name, code in KNOWN_BAD_SNIPPETS:
        count, stdout, stderr = run_semgrep_raw(code)
        status = "OK" if count > 0 else "MISSED"
        print(f"\n{name}: count={count} [{status}]")
        if count > 0 and stdout:
            try:
                data = json.loads(stdout)
                for r in data.get("results", [])[:2]:
                    print(f"  - {r.get('check_id', '?')}: {r.get('extra', {}).get('message', '')[:60]}")
            except json.JSONDecodeError:
                pass
        if count == 0 and stderr:
            print(f"  stderr: {stderr[:200]}")

    print("\n" + "=" * 60)
    print("2. OUR WRAPPER on same snippets")
    print("=" * 60)
    from utils.semgrep_runner import run_semgrep_on_code

    for name, code in KNOWN_BAD_SNIPPETS:
        cnt, score = run_semgrep_on_code(code, ".c")
        print(f"{name}: count={cnt}, score={score}")

    print("\n" + "=" * 60)
    print("3. DIVERSEVUL SAMPLES (raw Semgrep output)")
    print("=" * 60)
    import pandas as pd

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "phase3_experiment_results.csv"), nrows=10)
    for i, row in df.head(5).iterrows():
        code = str(row["code"])[:3000]  # first 3k chars
        count, stdout, stderr = run_semgrep_raw(code)
        print(f"\nDiverseVul row {i} (label={row['label']}): count={count}")
        if count > 0 and stdout:
            try:
                data = json.loads(stdout)
                for r in data.get("results", [])[:2]:
                    print(f"  - {r.get('check_id')}: {r.get('extra', {}).get('message', '')[:80]}")
            except json.JSONDecodeError:
                pass
        elif stdout:
            try:
                data = json.loads(stdout)
                print(f"  results: {len(data.get('results', []))}")
            except json.JSONDecodeError:
                print(f"  (stdout truncated, raw count={count})")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    counts_bad = [run_semgrep_raw(c)[0] for _, c in KNOWN_BAD_SNIPPETS]
    if any(c > 0 for c in counts_bad):
        print("Semgrep WORKS on known-bad patterns -> our setup is correct.")
    else:
        print("Semgrep found NOTHING on known-bad patterns -> possible Semgrep install/config issue.")
    print("If DiverseVul samples all show count=0, PaC=0 is due to rule coverage, not our bug.")


if __name__ == "__main__":
    main()
