"""
Run Semgrep on a single C/C++ code snippet (written to a temp file).
Return violation count and optional normalized score in [0, 1].
"""

import os
import sys
import subprocess
import tempfile


# Configs per thesis 3.2.3: p/c, p/owasp-top-ten, p/cwe-top-25
SEMGREP_CONFIGS = ["p/c", "p/owasp-top-ten", "p/cwe-top-25"]

# Cap for normalizing violation count to [0,1]: V_PaC = min(1, count / K)
NORMALIZE_K = 10.0


def _semgrep_exe() -> str:
    """Resolve semgrep: prefer PATH (Kaggle/CI), then same dir as Python (venv)."""
    try:
        import shutil
        found = shutil.which("semgrep")
        if found and os.path.isfile(found):
            return found
    except Exception:
        pass
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(exe_dir, "semgrep")
    if os.path.isfile(candidate):
        return candidate
    return "semgrep"


def run_semgrep_on_code(code: str, ext: str = ".c") -> tuple[int, float]:
    """
    Write code to a temp file, run Semgrep, return (violation_count, normalized_score).
    normalized_score = min(1.0, violation_count / NORMALIZE_K).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=ext, delete=False, encoding="utf-8", errors="replace"
    ) as f:
        f.write(code)
        path = f.name
    try:
        # Use multiple configs for broader C/C++ coverage (p/c, cwe-top-25, owasp)
        cmd = [_semgrep_exe(), "scan", "--json", "--quiet", "--no-git-ignore"]
        for cfg in SEMGREP_CONFIGS:
            cmd.extend(["--config", cfg])
        cmd.append(path)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(path),
        )
        count = 0
        # Semgrep returns exit 1 when it finds issues; JSON is still in stdout
        if result.stdout:
            import json
            try:
                data = json.loads(result.stdout)
                count = len(data.get("results", []))
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: exit 1 + "findings" in stderr when JSON parse fails
        if count == 0 and result.returncode != 0 and result.stderr and "findings" in result.stderr.lower():
            count = 1
        score = min(1.0, count / NORMALIZE_K)
        return count, score
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 0, 0.0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
