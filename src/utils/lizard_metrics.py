"""
Extract static code metrics (cyclomatic complexity, LOC, etc.) from C/C++ code
using lizard. Used for Random Forest baseline features.
"""

import tempfile
import os


def extract_metrics(code: str, language: str = "c"):
    """
    Return dict with cyclomatic_complexity, nloc, token_count, parameter_count
    (aggregated over all functions in the snippet). If lizard fails, return defaults.
    """
    try:
        import lizard
    except ImportError:
        return {
            "cyclomatic_complexity": 0,
            "nloc": len(code.splitlines()),
            "token_count": len(code.split()),
            "parameter_count": 0,
        }
    ext = ".c" if language == "c" else ".cpp"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=ext, delete=False, encoding="utf-8", errors="replace"
    ) as f:
        f.write(code)
        path = f.name
    try:
        results = lizard.analyze_file(path, languages=[language])
        # Aggregate over all functions
        ccn = 0
        nloc = 0
        token_count = 0
        params = 0
        for fn in results.function_list:
            ccn += fn.cyclomatic_complexity
            nloc += fn.length
            token_count += fn.token_count
            params += fn.parameter_count
        if not results.function_list:
            nloc = len(code.splitlines())
            token_count = len(code.split())
        return {
            "cyclomatic_complexity": ccn,
            "nloc": nloc,
            "token_count": token_count,
            "parameter_count": params,
        }
    except Exception:
        return {
            "cyclomatic_complexity": 0,
            "nloc": len(code.splitlines()),
            "token_count": len(code.split()),
            "parameter_count": 0,
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
