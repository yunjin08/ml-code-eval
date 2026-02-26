"""
Phase 3: Controlled Governance Experiment
Run ML-Only, PaC-Only, and Hybrid governance on the test set.
Output: ml_confidence, pac_score, hybrid_risk, decision_ml, decision_pac, decision_hybrid, label.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# PaC progress (for --resume): save every N test samples so run can resume if interrupted
PHASE3_PAC_PROGRESS_PATH = os.path.join(RESULTS_DIR, "phase3_pac_progress.npz")
PAC_SAVE_EVERY_N = 5000


def _push_phase3_results_to_kaggle_dataset(results_dir: str, dataset_slug: str, message: str) -> bool:
    """Zip Phase 3 results and push as new version of Kaggle Dataset. Returns True if ok."""
    staging = "/kaggle/working/phase3_results_backup"
    if not os.path.isdir("/kaggle/working"):
        return False
    try:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)
        zip_path = os.path.join(staging, "phase3_results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("phase3_experiment_results.csv", "phase3_hybrid_config.json", "phase2_validation_report.json"):
                src = os.path.join(results_dir, name)
                if os.path.isfile(src):
                    zf.write(src, name)
        meta = {
            "title": "Phase 3 experiment results",
            "id": dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(os.path.join(staging, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        r = subprocess.run(
            ["kaggle", "datasets", "version", "-p", staging, "-m", message],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0:
            print(f"[Phase 3] Kaggle backup warning: {r.stderr or r.stdout}")
            return False
        print(f"[Phase 3] Results backed up to Kaggle dataset: {dataset_slug}")
        return True
    except Exception as e:
        print(f"[Phase 3] Kaggle backup failed: {e}")
        return False


def load_splits_and_test():
    with open(os.path.join(DATA_DIR, "splits.json")) as f:
        splits = json.load(f)
    df = pd.read_csv(os.path.join(DATA_DIR, "curated_cpp.csv"))
    test_df = df[df["id"].isin(splits["test_ids"])].copy()
    val_df = df[df["id"].isin(splits["validation_ids"])].copy()
    return test_df, val_df, splits


def load_selected_model():
    report_path = os.path.join(RESULTS_DIR, "phase2_validation_report.json")
    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"Run phase2_train.py first. Missing {report_path}")
    with open(report_path) as f:
        report = json.load(f)
    return report.get("selected_model", "codebert")


def get_ml_confidence_codebert(test_df, batch_size=16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from tqdm import tqdm

    model_path = os.path.join(MODELS_DIR, "codebert")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"CodeBERT checkpoint not found at {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    max_length = 512
    confidences = []
    n_batches = (len(test_df) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(test_df), batch_size), total=n_batches, desc="ML (CodeBERT)"):
        batch = test_df.iloc[start : start + batch_size]
        enc = tokenizer(
            batch["code"].tolist(),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)
            # class 1 = vulnerable
            p1 = probs[:, 1].cpu().numpy()
        confidences.extend(p1.tolist())
    return np.array(confidences)


def get_ml_confidence_rf(test_df):
    import joblib
    from utils.lizard_metrics import extract_metrics

    rf_path = os.path.join(MODELS_DIR, "rf.pkl")
    if not os.path.isfile(rf_path):
        raise FileNotFoundError(f"Random Forest not found at {rf_path}")
    rf = joblib.load(rf_path)
    rows = []
    for _, row in test_df.iterrows():
        m = extract_metrics(row["code"], "c")
        rows.append([m["cyclomatic_complexity"], m["nloc"], m["token_count"], m["parameter_count"]])
    X = np.array(rows)
    probs = rf.predict_proba(X)[:, 1]  # P(vulnerable)
    return probs


def _pac_one(args):
    """Worker for parallel PaC: (code, ext) -> score. Must be top-level for multiprocessing."""
    code, ext = args
    from utils.semgrep_runner import run_semgrep_on_code
    _, score = run_semgrep_on_code(code, ext)
    return score


def get_pac_scores(test_df, workers=1, pac_batch_size=0):
    from tqdm import tqdm
    from utils.semgrep_runner import run_semgrep_batch

    # Batched mode: one Semgrep run per N files (much faster than 1 run per file)
    if pac_batch_size and int(pac_batch_size) > 1:
        batch_size = int(pac_batch_size)
        codes = test_df["code"].tolist()
        scores = []
        n_batches = (len(codes) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(codes), batch_size), total=n_batches, desc="PaC"):
            batch = codes[start : start + batch_size]
            batch_results = run_semgrep_batch(batch, ".c")
            scores.extend(score for _, score in batch_results)
        return np.array(scores)

    if workers is not None and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(row["code"], ".c") for _, row in test_df.iterrows()]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            scores = list(tqdm(
                ex.map(_pac_one, tasks, chunksize=min(10, max(1, len(tasks) // (workers * 4)))),
                total=len(tasks),
                desc="PaC",
            ))
        return np.array(scores)
    from utils.semgrep_runner import run_semgrep_on_code
    scores = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="PaC"):
        _, score = run_semgrep_on_code(row["code"], ".c")
        scores.append(score)
    return np.array(scores)


def decisions_from_thresholds(score: np.ndarray, t_block: float, t_review: float) -> np.ndarray:
    """Return 0=Approve, 1=Review, 2=Block. Block if score >= t_block, Review if >= t_review."""
    out = np.zeros(len(score), dtype=int)
    out[score >= t_review] = 1
    out[score >= t_block] = 2
    return out


def tune_hybrid_on_val(ml_scores, pac_scores, labels):
    """Grid search alpha in [0, 0.25, 0.5, 0.75, 1.0], beta = 1-alpha; thresholds to maximize F1(Block)."""
    from sklearn.metrics import f1_score

    best_f1 = -1
    best_alpha = 0.5
    best_t_block = 0.5
    best_t_review = 0.25
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        beta = 1.0 - alpha
        hybrid = alpha * ml_scores + beta * pac_scores
        for t_block in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for t_review in [0.1, 0.2, 0.25, 0.3]:
                if t_review >= t_block:
                    continue
                block_pred = (hybrid >= t_block).astype(int)
                f1 = f1_score(labels, block_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = alpha
                    best_t_block = t_block
                    best_t_review = t_review
    return best_alpha, 1.0 - best_alpha, best_t_block, best_t_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_test", type=int, default=None, help="Cap test set size")
    parser.add_argument("--max_val", type=int, default=2000, help="Cap val size for hybrid tuning")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for PaC (Semgrep). Use 1 for sequential.")
    parser.add_argument("--resume", action="store_true", help="Resume PaC from last saved progress if interrupted (saves every 5k samples).")
    parser.add_argument("--no_verify_pac", action="store_true", help="Skip PaC setup verification (not recommended; use only if you already verified).")
    parser.add_argument("--pac_batch_size", type=int, default=100, help="Run Semgrep on N files per process (default 100). Set 0 for per-file mode (slower).")
    parser.add_argument("--backup_to_kaggle_dataset", type=str, default=None, help="Push results to Kaggle Dataset when done (e.g. USERNAME/code-reviewer-thesis-phase3-results). Saves phase3 CSV + config so you can find them after session reset.")
    args = parser.parse_args()

    # Verify PaC (Semgrep) works before trusting any PaC results
    if not args.no_verify_pac:
        if SCRIPT_DIR not in sys.path:
            sys.path.insert(0, SCRIPT_DIR)
        try:
            from verify_pac_setup import run_verification
            passed, failures = run_verification()
            if not passed:
                print("PaC verification FAILED. Fix Semgrep setup or run: python src/verify_pac_setup.py")
                for msg in failures:
                    print(f"  {msg}")
                sys.exit(1)
            print("PaC verification passed.")
        except Exception as e:
            print(f"PaC verification error: {e}")
            sys.exit(1)

    test_df, val_df, _ = load_splits_and_test()
    if args.max_test:
        test_df = test_df.sample(n=args.max_test, random_state=42)
    selected = load_selected_model()
    print(f"Using ML model: {selected}")

    # ML scores on test
    if selected == "codebert":
        ml_confidence = get_ml_confidence_codebert(test_df)
    else:
        ml_confidence = get_ml_confidence_rf(test_df)
    test_df = test_df.copy()
    test_df["ml_confidence"] = ml_confidence

    # PaC scores on test (with optional resume: save progress every PAC_SAVE_EVERY_N, resume from file if --resume)
    n_total = len(test_df)
    pac_scores_partial = None
    n_done = 0
    if args.resume and os.path.isfile(PHASE3_PAC_PROGRESS_PATH):
        try:
            with np.load(PHASE3_PAC_PROGRESS_PATH, allow_pickle=False) as data:
                pac_scores_partial = np.array(data["scores"], dtype=np.float64)
                file_n_total = int(data["n_total"])
            if file_n_total == n_total and len(pac_scores_partial) <= n_total:
                n_done = len(pac_scores_partial)
                if n_done == n_total:
                    print(f"Resumed: PaC scores already complete ({n_total} samples).")
                else:
                    print(f"Resuming PaC from {n_done}/{n_total} samples.")
            else:
                pac_scores_partial = None
                n_done = 0
        except (Exception, OSError):
            pac_scores_partial = None
            n_done = 0

    if pac_scores_partial is None:
        pac_scores_partial = np.array([], dtype=np.float64)
        n_done = 0

    if n_done < n_total:
        remaining_df = test_df.iloc[n_done:]
        for start in range(0, len(remaining_df), PAC_SAVE_EVERY_N):
            end = min(start + PAC_SAVE_EVERY_N, len(remaining_df))
            chunk_df = remaining_df.iloc[start:end]
            chunk_scores = get_pac_scores(chunk_df, workers=args.workers, pac_batch_size=args.pac_batch_size)
            pac_scores_partial = np.concatenate([pac_scores_partial, chunk_scores])
            if args.resume:
                np.savez(PHASE3_PAC_PROGRESS_PATH, scores=pac_scores_partial, n_total=n_total)
                print(f"PaC progress saved: {len(pac_scores_partial)}/{n_total}")

    pac_scores = pac_scores_partial
    if args.resume and os.path.isfile(PHASE3_PAC_PROGRESS_PATH):
        try:
            os.remove(PHASE3_PAC_PROGRESS_PATH)
        except OSError:
            pass

    test_df["pac_score"] = pac_scores

    # Tune hybrid on validation (small subset)
    val_sub = val_df.sample(n=min(args.max_val, len(val_df)), random_state=42) if args.max_val else val_df
    if selected == "codebert":
        val_ml = get_ml_confidence_codebert(val_sub)
    else:
        val_ml = get_ml_confidence_rf(val_sub)
    val_pac = get_pac_scores(val_sub, workers=args.workers, pac_batch_size=args.pac_batch_size)
    alpha, beta, t_block, t_review = tune_hybrid_on_val(val_ml, val_pac, val_sub["label"].values)

    # Hybrid risk and decisions on test
    test_df["hybrid_risk"] = alpha * test_df["ml_confidence"] + beta * test_df["pac_score"]
    test_df["decision_ml"] = decisions_from_thresholds(test_df["ml_confidence"].values, t_block, t_review)
    test_df["decision_pac"] = decisions_from_thresholds(test_df["pac_score"].values, t_block, t_review)
    test_df["decision_hybrid"] = decisions_from_thresholds(test_df["hybrid_risk"].values, t_block, t_review)

    # Block = 2; for evaluation we need binary Block vs non-Block
    out_path = os.path.join(RESULTS_DIR, "phase3_experiment_results.csv")
    test_df.to_csv(out_path, index=False)
    config_path = os.path.join(RESULTS_DIR, "phase3_hybrid_config.json")
    with open(config_path, "w") as f:
        json.dump({"alpha": alpha, "beta": beta, "t_block": t_block, "t_review": t_review}, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Hybrid config: alpha={alpha}, beta={beta}, t_block={t_block}, t_review={t_review}")

    if args.backup_to_kaggle_dataset:
        print("Backing up results to Kaggle Dataset...")
        _push_phase3_results_to_kaggle_dataset(
            RESULTS_DIR,
            args.backup_to_kaggle_dataset,
            "Phase 3 experiment results (auto-save)",
        )


if __name__ == "__main__":
    main()
