"""
Phase 4: Evaluation and Statistical Analysis
Compute Precision/Recall/F1 for Block decision, McNemar's test, odds ratio, ROC/AUC.
Validate hypotheses H1, H2, H3 and write results report.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from scipy.stats import chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_results():
    path = os.path.join(RESULTS_DIR, "phase3_experiment_results.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Run phase3_experiment.py first. Missing {path}")
    return pd.read_csv(path)


def block_prediction(decision_col: pd.Series) -> np.ndarray:
    """Block = 2 in our encoding; binary 1 = Block, 0 = non-Block."""
    return (decision_col == 2).astype(int).values


def primary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def mcnemar_and_odds_ratio(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
    """pred_a, pred_b are binary (1 = Block). Returns p-value and odds ratio with 95% CI."""
    # 2x2 table of (pred_a, pred_b): table[i,j] = count where pred_a=i, pred_b=j
    t00 = ((pred_a == 0) & (pred_b == 0)).sum()
    t01 = ((pred_a == 0) & (pred_b == 1)).sum()
    t10 = ((pred_a == 1) & (pred_b == 0)).sum()
    t11 = ((pred_a == 1) & (pred_b == 1)).sum()
    table = np.array([[t00, t01], [t10, t11]])
    # Discordant pairs for correctness: only_a = A correct & B wrong, only_b = A wrong & B correct
    only_a = ((pred_a == y_true) & (pred_b != y_true)).sum()
    only_b = ((pred_a != y_true) & (pred_b == y_true)).sum()
    # McNemar's test: discordant pairs b=table[0,1], c=table[1,0]; chi2 = (b-c)^2/(b+c), df=1
    b, c = int(table[0, 1]), int(table[1, 0])
    try:
        if b + c > 0:
            chi2_val = (b - c) ** 2 / (b + c)
            p_value = float(1 - chi2.cdf(chi2_val, 1))
        else:
            p_value = 1.0
    except Exception:
        p_value = 1.0
    # Odds ratio: (only_b / only_a) when only_a > 0; else undefined
    if only_a > 0:
        or_val = only_b / only_a
    else:
        or_val = float("inf") if only_b > 0 else 1.0
    # Approximate 95% CI (log scale)
    n_disc = only_a + only_b
    if n_disc > 0 and only_a > 0 and only_b > 0:
        log_or = np.log(or_val)
        se = np.sqrt(1 / only_a + 1 / only_b)
        ci_low = np.exp(log_or - 1.96 * se)
        ci_high = np.exp(log_or + 1.96 * se)
    else:
        ci_low = ci_high = or_val
    return {"mcnemar_p": p_value, "odds_ratio": or_val, "ci_95": [float(ci_low), float(ci_high)], "contingency": table.tolist(), "only_a_correct": int(only_a), "only_b_correct": int(only_b)}


def roc_auc(scores: np.ndarray, y_true: np.ndarray) -> tuple:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(auc(fpr, tpr)), fpr, tpr


def main():
    df = load_results()
    y_true = df["label"].values
    pred_ml = block_prediction(df["decision_ml"])
    pred_pac = block_prediction(df["decision_pac"])
    pred_hybrid = block_prediction(df["decision_hybrid"])

    # 4.1 Primary metrics (Block)
    metrics_ml = primary_metrics(y_true, pred_ml)
    metrics_pac = primary_metrics(y_true, pred_pac)
    metrics_hybrid = primary_metrics(y_true, pred_hybrid)

    # 4.2 McNemar and 4.3 Odds ratio
    hybrid_vs_ml = mcnemar_and_odds_ratio(y_true, pred_hybrid, pred_ml)
    hybrid_vs_pac = mcnemar_and_odds_ratio(y_true, pred_hybrid, pred_pac)

    # 4.4 ROC/AUC
    auc_ml, fpr_ml, tpr_ml = roc_auc(df["ml_confidence"].values, y_true)
    auc_hybrid, fpr_hybrid, tpr_hybrid = roc_auc(df["hybrid_risk"].values, y_true)
    try:
        auc_pac, fpr_pac, tpr_pac = roc_auc(df["pac_score"].values, y_true)
    except Exception:
        auc_pac = 0.5

    # 4.5 Hypothesis validation
    h1 = {
        "statement": "Hybrid F1 > ML-Only and > PaC-Only",
        "hybrid_f1": metrics_hybrid["f1"],
        "ml_f1": metrics_ml["f1"],
        "pac_f1": metrics_pac["f1"],
        "supported": metrics_hybrid["f1"] > metrics_ml["f1"] and metrics_hybrid["f1"] > metrics_pac["f1"],
        "mcnemar_hybrid_vs_ml_p": hybrid_vs_ml["mcnemar_p"],
        "mcnemar_hybrid_vs_pac_p": hybrid_vs_pac["mcnemar_p"],
    }
    h2 = {
        "statement": "Hybrid Recall > PaC-Only Recall",
        "hybrid_recall": metrics_hybrid["recall"],
        "pac_recall": metrics_pac["recall"],
        "supported": metrics_hybrid["recall"] > metrics_pac["recall"],
        "mcnemar_p": hybrid_vs_pac["mcnemar_p"],
        "odds_ratio_95ci": hybrid_vs_pac["ci_95"],
    }
    h3 = {
        "statement": "Hybrid Precision > ML-Only Precision",
        "hybrid_precision": metrics_hybrid["precision"],
        "ml_precision": metrics_ml["precision"],
        "supported": metrics_hybrid["precision"] > metrics_ml["precision"],
        "mcnemar_p": hybrid_vs_ml["mcnemar_p"],
        "odds_ratio_95ci": hybrid_vs_ml["ci_95"],
    }

    report = {
        "primary_metrics": {
            "ML_Only": metrics_ml,
            "PaC_Only": metrics_pac,
            "Hybrid": metrics_hybrid,
        },
        "mcnemar": {
            "Hybrid_vs_ML": hybrid_vs_ml,
            "Hybrid_vs_PaC": hybrid_vs_pac,
        },
        "roc_auc": {"ML_Only": auc_ml, "PaC_Only": auc_pac, "Hybrid": auc_hybrid},
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3},
    }

    report_path = os.path.join(RESULTS_DIR, "phase4_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    summary_path = os.path.join(RESULTS_DIR, "phase4_results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Phase 4: Evaluation & Statistical Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write("Primary metrics (Block decision):\n")
        for name, m in report["primary_metrics"].items():
            f.write(f"  {name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}\n")
        f.write("\nMcNemar (Hybrid vs others):\n")
        f.write(f"  vs ML-Only: p={report['mcnemar']['Hybrid_vs_ML']['mcnemar_p']:.4f}\n")
        f.write(f"  vs PaC-Only: p={report['mcnemar']['Hybrid_vs_PaC']['mcnemar_p']:.4f}\n")
        f.write("\nROC AUC: ML={:.4f} PaC={:.4f} Hybrid={:.4f}\n".format(
            report["roc_auc"]["ML_Only"], report["roc_auc"]["PaC_Only"], report["roc_auc"]["Hybrid"]))
        f.write("\nHypotheses:\n")
        f.write(f"  H1 (Hybrid F1 higher): {h1['supported']}\n")
        f.write(f"  H2 (Hybrid Recall > PaC): {h2['supported']}\n")
        f.write(f"  H3 (Hybrid Precision > ML): {h3['supported']}\n")
    print(f"Report saved to {report_path}")
    print(f"Summary saved to {summary_path}")
    print("Primary metrics (Block): ML F1={:.4f}, PaC F1={:.4f}, Hybrid F1={:.4f}".format(
        metrics_ml["f1"], metrics_pac["f1"], metrics_hybrid["f1"]))


if __name__ == "__main__":
    main()
