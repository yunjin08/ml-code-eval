"""
Thesis Demo API — Live vulnerability analysis via Semgrep (PaC).
POST /analyze   — accepts code text or file upload, returns findings + decision.
GET  /health    — health check.
"""
import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Hybrid Code Review — Thesis Demo API",
    description="Run Semgrep (PaC) on submitted C/C++ code and return findings + hybrid risk.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your Netlify URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Config (matches Phase 3) ─────────────────────────────────────────────────
HYBRID_CFG = {
    "alpha": 0.75,
    "beta": 0.25,
    "t_block": 0.4,
    "t_review": 0.1,
    "val_ml_min": 0.09322,
    "val_ml_max": 0.53684,
    "val_pac_min": 0.0,
    "val_pac_max": 1.0,
}
NORMALIZE_K = 10.0

# Semgrep configs relative to project root (mounted in Railway)
_HERE = Path(__file__).parent.parent          # repo root
_CUSTOM_RULES = _HERE / "config" / "custom_rules.yaml"

SEMGREP_CONFIGS = ["p/c", "p/cwe-top-25"]
if _CUSTOM_RULES.exists():
    SEMGREP_CONFIGS.append(str(_CUSTOM_RULES))


def _semgrep_exe() -> str:
    found = shutil.which("semgrep")
    if found:
        return found
    # venv fallback
    candidate = Path(os.path.dirname(os.sys.executable)) / "semgrep"
    if candidate.exists():
        return str(candidate)
    return "semgrep"


def _norm(x: float, mn: float, mx: float, eps: float = 1e-9) -> float:
    return max(0.0, min(1.0, (x - mn) / (mx - mn + eps)))


def _decision(score: float) -> str:
    if score >= HYBRID_CFG["t_block"]:
        return "Block"
    if score >= HYBRID_CFG["t_review"]:
        return "Review"
    return "Approve"


def run_semgrep(path: str) -> tuple[list[dict], float]:
    """Run Semgrep on a file/dir. Returns (findings_list, pac_score)."""
    cmd = [_semgrep_exe(), "scan", "--json", "--quiet", "--no-git-ignore"]
    for cfg in SEMGREP_CONFIGS:
        cmd += ["--config", cfg]
    cmd.append(path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, cwd=str(_HERE)
        )
        findings = []
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                findings = data.get("results", [])
            except json.JSONDecodeError:
                pass
        pac_score = min(1.0, len(findings) / NORMALIZE_K)
        return findings, pac_score
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Semgrep timed out (>60s). Try a smaller snippet.")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Semgrep not found. Ensure it is installed.")


def build_response(findings: list[dict], pac_score: float, filename: str) -> dict:
    """Build the analysis response including hybrid risk (no live ML — heuristic placeholder)."""
    pac_norm = _norm(pac_score, HYBRID_CFG["val_pac_min"], HYBRID_CFG["val_pac_max"])
    pac_dec = _decision(pac_score)

    # Group findings by rule id
    grouped: dict[str, list] = {}
    for f in findings:
        rule = f.get("check_id", "unknown").split(".")[-1]
        grouped.setdefault(rule, []).append({
            "line": f.get("start", {}).get("line"),
            "message": f.get("extra", {}).get("message", ""),
            "severity": f.get("extra", {}).get("severity", ""),
        })

    # Heuristic ML estimate (in a real deployment, replace with a lightweight model call)
    ml_heuristic = min(0.95, 0.12 + pac_score * 0.4)
    ml_norm = _norm(ml_heuristic, HYBRID_CFG["val_ml_min"], HYBRID_CFG["val_ml_max"])
    hybrid_risk = HYBRID_CFG["alpha"] * ml_norm + HYBRID_CFG["beta"] * pac_norm
    hybrid_dec = _decision(hybrid_risk)

    return {
        "filename": filename,
        "pac_score": round(pac_score, 4),
        "pac_norm": round(pac_norm, 4),
        "pac_decision": pac_dec,
        "ml_heuristic": round(ml_heuristic, 4),
        "ml_norm": round(ml_norm, 4),
        "hybrid_risk": round(hybrid_risk, 4),
        "hybrid_decision": hybrid_dec,
        "findings_count": len(findings),
        "findings_by_rule": grouped,
        "semgrep_configs": SEMGREP_CONFIGS,
        "hybrid_config": HYBRID_CFG,
        "note": (
            "ML score is a heuristic estimate (CodeBERT inference requires GPU). "
            "PaC score and findings are from live Semgrep analysis."
        ),
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str
    filename: Optional[str] = "snippet.c"


@app.post("/analyze")
async def analyze_code(req: CodeRequest):
    """Analyze a code string (paste). Returns PaC findings + hybrid decision."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="code must not be empty.")
    ext = ".cpp" if req.filename.endswith(".cpp") else ".c"
    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False,
                                     encoding="utf-8", errors="replace") as f:
        f.write(req.code)
        tmppath = f.name
    try:
        findings, pac_score = run_semgrep(tmppath)
        return build_response(findings, pac_score, req.filename)
    finally:
        try:
            os.unlink(tmppath)
        except OSError:
            pass


@app.post("/analyze/upload")
async def analyze_upload(
    files: list[UploadFile] = File(...),
):
    """Analyze uploaded C/C++ files. Max 20 files, 512 KB each."""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 files per request.")

    tmpdir = tempfile.mkdtemp(prefix="thesis_upload_")
    saved = []
    try:
        for uf in files:
            fname = Path(uf.filename).name
            if not fname.endswith((".c", ".cpp", ".h", ".hpp")):
                continue
            dest = Path(tmpdir) / fname
            content = await uf.read()
            if len(content) > 512 * 1024:
                raise HTTPException(status_code=400, detail=f"{fname} exceeds 512 KB limit.")
            dest.write_bytes(content)
            saved.append(fname)

        if not saved:
            raise HTTPException(status_code=400, detail="No valid C/C++ files (.c .cpp .h .hpp) found.")

        findings, pac_score = run_semgrep(tmpdir)
        result = build_response(findings, pac_score, f"{len(saved)} file(s): {', '.join(saved[:5])}")
        result["files_analyzed"] = saved
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/health")
async def health():
    semgrep_ok = bool(shutil.which("semgrep"))
    custom_rules_ok = _CUSTOM_RULES.exists()
    return {
        "status": "ok",
        "semgrep_available": semgrep_ok,
        "custom_rules_available": custom_rules_ok,
        "configs": SEMGREP_CONFIGS,
    }


@app.get("/")
async def root():
    return {
        "name": "Hybrid Code Review — Thesis Demo API",
        "docs": "/docs",
        "endpoints": {
            "POST /analyze": "Analyze pasted C/C++ code",
            "POST /analyze/upload": "Analyze uploaded C/C++ files",
            "GET /health": "Health check",
        },
    }
