"""
Thesis Demo API — Live vulnerability analysis via Semgrep (PaC).
POST /analyze   — accepts code text or file upload, returns findings + decision.
GET  /health    — health check.

Locally: set USE_REAL_ML=1 and have src/models/codebert/ to use real CodeBERT instead of heuristic.
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

# ── Optional CodeBERT for local runs ───────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CODEBERT_PATH = REPO_ROOT / "src" / "models" / "codebert"
USE_REAL_ML_ENV = os.environ.get("USE_REAL_ML", "").strip().lower() in ("1", "true", "yes")

_ml_model = None
_ml_tokenizer = None
_ml_device = None


def _load_codebert_if_available():
    """Load CodeBERT once when USE_REAL_ML=1 and model dir exists."""
    global _ml_model, _ml_tokenizer, _ml_device
    if not USE_REAL_ML_ENV or not CODEBERT_PATH.is_dir():
        return
    if _ml_model is not None:
        return
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _ml_tokenizer = AutoTokenizer.from_pretrained(str(CODEBERT_PATH))
        _ml_model = AutoModelForSequenceClassification.from_pretrained(str(CODEBERT_PATH))
        _ml_model.eval()
        _ml_device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        _ml_model.to(_ml_device)
    except Exception as e:
        import warnings
        warnings.warn(f"CodeBERT load failed: {e}. Using heuristic ML.")


def get_ml_score(code: str | list[str] | None, pac_score: float) -> tuple[float, bool]:
    """
    Return (ml_score, used_real_ml). If real CodeBERT is loaded and non-empty code is provided, run inference;
    otherwise return heuristic from pac_score.
    """
    _load_codebert_if_available()
    heuristic = (min(0.95, 0.12 + pac_score * 0.4), False)
    if _ml_model is not None and code is not None:
        texts = [code] if isinstance(code, str) else code
        texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if texts:
            import torch
            max_len = 512
            enc = _ml_tokenizer(
                texts,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            enc = {k: v.to(_ml_device) for k, v in enc.items()}
            with torch.no_grad():
                out = _ml_model(**enc)
                probs = torch.softmax(out.logits, dim=-1)
                p1 = probs[:, 1].cpu().numpy()
            ml = float(max(p1)) if len(p1) > 1 else float(p1[0])
            return (ml, True)
    return heuristic


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

# Semgrep configs relative to project root
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


def build_response(
    findings: list[dict],
    pac_score: float,
    filename: str,
    code_for_ml: Optional[str | list[str]] = None,
) -> dict:
    """Build the analysis response. Uses real CodeBERT when USE_REAL_ML=1 and code_for_ml provided."""
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

    ml_score, used_real_ml = get_ml_score(code_for_ml if code_for_ml is not None else "", pac_score)
    ml_norm = _norm(ml_score, HYBRID_CFG["val_ml_min"], HYBRID_CFG["val_ml_max"])
    hybrid_risk = HYBRID_CFG["alpha"] * ml_norm + HYBRID_CFG["beta"] * pac_norm
    hybrid_dec = _decision(hybrid_risk)

    note = (
        "ML score from live CodeBERT. PaC and findings from Semgrep."
        if used_real_ml
        else (
            "ML score is a heuristic estimate (CodeBERT not loaded; set USE_REAL_ML=1 locally with src/models/codebert). "
            "PaC score and findings are from live Semgrep analysis."
        )
    )

    return {
        "filename": filename,
        "pac_score": round(pac_score, 4),
        "pac_norm": round(pac_norm, 4),
        "pac_decision": pac_dec,
        "ml_heuristic": round(ml_score, 4),
        "ml_norm": round(ml_norm, 4),
        "hybrid_risk": round(hybrid_risk, 4),
        "hybrid_decision": hybrid_dec,
        "findings_count": len(findings),
        "findings_by_rule": grouped,
        "semgrep_configs": SEMGREP_CONFIGS,
        "hybrid_config": HYBRID_CFG,
        "note": note,
        "ml_from_codebert": used_real_ml,
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
        return build_response(findings, pac_score, req.filename, code_for_ml=req.code)
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
    code_contents: list[str] = []
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
            try:
                code_contents.append(content.decode("utf-8", errors="replace"))
            except Exception:
                code_contents.append("")

        if not saved:
            raise HTTPException(status_code=400, detail="No valid C/C++ files (.c .cpp .h .hpp) found.")

        findings, pac_score = run_semgrep(tmpdir)
        result = build_response(
            findings, pac_score, f"{len(saved)} file(s): {', '.join(saved[:5])}",
            code_for_ml=code_contents if code_contents else None,
        )
        result["files_analyzed"] = saved
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/health")
async def health():
    _load_codebert_if_available()
    semgrep_ok = bool(shutil.which("semgrep"))
    custom_rules_ok = _CUSTOM_RULES.exists()
    return {
        "status": "ok",
        "semgrep_available": semgrep_ok,
        "custom_rules_available": custom_rules_ok,
        "configs": SEMGREP_CONFIGS,
        "ml_from_codebert": _ml_model is not None,
        "use_real_ml_env": USE_REAL_ML_ENV,
        "codebert_path_exists": CODEBERT_PATH.is_dir(),
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
