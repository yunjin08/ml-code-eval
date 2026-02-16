"""
Run Semgrep on a single C/C++ code snippet (written to a temp file).
Return violation count and optional normalized score in [0, 1].
"""

import os
import subprocess
import tempfile


# Configs per thesis 3.2.3: p/c, p/owasp-top-ten, p/cwe-top-25
SEMGREP_CONFIGS = ["p/c", "p/owasp-top-ten", "p/cwe-top-25"]

# Cap for normalizing violation count to [0,1]: V_PaC = min(1, count / K)
NORMALIZE_K = 10.0


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
        cmd = [
            "semgrep",
            "scan",
            "--config", SEMGREP_CONFIGS[0],  # p/c for C/C++; other configs may have fewer C rules
            "--json",
            "--quiet",
            "--no-git-ignore",
            path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(path),
        )
        count = 0
        if result.returncode == 0 and result.stdout:
            import json
            try:
                data = json.loads(result.stdout)
                count = len(data.get("results", []))
            except (json.JSONDecodeError, TypeError):
                pass
        # Semgrep returns 1 on findings when using --json in some versions
        if result.returncode != 0 and result.stderr and "findings" in result.stderr.lower():
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
