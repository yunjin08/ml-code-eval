"""
Phase 3: Controlled Governance Experiment
Run ML-Only, PaC-Only, and Hybrid governance on the test set.
Output: ml_confidence, pac_score, hybrid_risk, decision_ml, decision_pac, decision_hybrid, label.
"""

import argparse
import json
import os
import sys

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
    for start in range(0, len(test_df), batch_size):
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


def get_pac_scores(test_df):
    from utils.semgrep_runner import run_semgrep_on_code
    from tqdm import tqdm

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
    args = parser.parse_args()

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

    # PaC scores on test
    pac_scores = get_pac_scores(test_df)
    test_df["pac_score"] = pac_scores

    # Tune hybrid on validation (small subset)
    val_sub = val_df.sample(n=min(args.max_val, len(val_df)), random_state=42) if args.max_val else val_df
    if selected == "codebert":
        val_ml = get_ml_confidence_codebert(val_sub)
    else:
        val_ml = get_ml_confidence_rf(val_sub)
    from utils.semgrep_runner import run_semgrep_on_code as _semgrep_run
    val_pac = np.array([_semgrep_run(row["code"], ".c")[1] for _, row in val_sub.iterrows()])
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


if __name__ == "__main__":
    main()
