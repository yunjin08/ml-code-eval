"""
Run Semgrep on a single C/C++ code snippet (written to a temp file).
Return violation count and optional normalized score in [0, 1].
"""

import os
import sys
import subprocess
import tempfile


# Single config for speed; same C/C++ coverage. (Multiple configs were 3x slower, no extra findings on DiverseVul.)
SEMGREP_CONFIG = "p/c"
# Legacy list for verify script / backward compat
SEMGREP_CONFIGS = [SEMGREP_CONFIG]

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
        cmd = [_semgrep_exe(), "scan", "--json", "--quiet", "--no-git-ignore", "--config", SEMGREP_CONFIG, path]
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
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0, 0.0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_semgrep_batch(codes: list, ext: str = ".c", timeout_per_batch: int = 120) -> list[tuple[int, float]]:
    """
    Run Semgrep once on a directory of files (batch). Much faster than N separate runs.
    Returns list of (count, score) in same order as codes.
    """
    import json
    import shutil

    if not codes:
        return []

    n = len(codes)
    tmpdir = tempfile.mkdtemp(prefix="semgrep_batch_")
    try:
        for i, code in enumerate(codes):
            fpath = os.path.join(tmpdir, f"{i:05d}{ext}")
            with open(fpath, "w", encoding="utf-8", errors="replace") as f:
                f.write(code)

        cmd = [_semgrep_exe(), "scan", "--json", "--quiet", "--no-git-ignore", "--config", SEMGREP_CONFIG, tmpdir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_per_batch, cwd=tmpdir)

        counts = [0] * n
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                for r in data.get("results", []):
                    path = r.get("path", "")
                    # path can be absolute or relative; basename gives e.g. 00042.c
                    base = os.path.basename(path)
                    idx_str = base[:5]  # "00042"
                    try:
                        idx = int(idx_str)
                        if 0 <= idx < n:
                            counts[idx] += 1
                    except ValueError:
                        pass
            except (json.JSONDecodeError, TypeError):
                pass

        return [(c, min(1.0, c / NORMALIZE_K)) for c in counts]
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except OSError:
            pass
