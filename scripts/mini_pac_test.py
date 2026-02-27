"""
Mini test: run Semgrep (PaC) on a sample of the test set to see if we still get all zeros.
Uses the same runner as Phase 3. Reports % of samples with PaC > 0 by label.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pandas as pd

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
N_VULN = 200   # vulnerable samples (label=1)
N_SAFE = 200   # non-vulnerable samples (label=0)
BATCH_SIZE = 50  # Semgrep batch size


def main():
    csv_path = os.path.join(DATA_DIR, "curated_cpp.csv")
    splits_path = os.path.join(DATA_DIR, "splits.json")
    fallback_path = os.path.join(PROJECT_ROOT, "results", "phase3_experiment_results.csv")

    if os.path.isfile(csv_path) and os.path.isfile(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        df = pd.read_csv(csv_path)
        test_df = df[df["id"].isin(splits["test_ids"])].copy()
    elif os.path.isfile(fallback_path):
        print("Using results/phase3_experiment_results.csv (no data/curated_cpp.csv).")
        df = pd.read_csv(fallback_path, nrows=50_000)
        test_df = df[["code", "label"]].copy()
    else:
        print("Missing data/curated_cpp.csv + splits.json and results/phase3_experiment_results.csv.")
        return 1

    vuln = test_df[test_df["label"] == 1]
    safe = test_df[test_df["label"] == 0]
    n_v = min(N_VULN, len(vuln))
    n_s = min(N_SAFE, len(safe))
    sample = pd.concat([
        vuln.sample(n=n_v, random_state=42),
        safe.sample(n=n_s, random_state=42),
    ], ignore_index=True)
    codes = sample["code"].tolist()
    labels = sample["label"].tolist()

    from utils.semgrep_runner import run_semgrep_batch

    print(f"Running PaC (Semgrep) on {len(codes)} test samples ({n_v} vuln, {n_s} safe), batch_size={BATCH_SIZE}...")
    scores = []
    for start in range(0, len(codes), BATCH_SIZE):
        batch = codes[start : start + BATCH_SIZE]
        batch_results = run_semgrep_batch(batch, ".c", timeout_per_batch=90)
        for _, score in batch_results:
            scores.append(score)
        if (start // BATCH_SIZE + 1) % 4 == 0:
            print(f"  ... {min(start + BATCH_SIZE, len(codes))}/{len(codes)}")

    n_pac_gt_0 = sum(1 for s in scores if s > 0)
    n_vuln = sum(1 for l in labels if l == 1)
    n_safe = sum(1 for l in labels if l == 0)
    vuln_pac_gt_0 = sum(1 for s, l in zip(scores, labels) if l == 1 and s > 0)
    safe_pac_gt_0 = sum(1 for s, l in zip(scores, labels) if l == 0 and s > 0)

    print()
    print("=" * 50)
    print("MINI PaC TEST RESULTS")
    print("=" * 50)
    print(f"  Total samples:     {len(scores)}")
    print(f"  PaC > 0:           {n_pac_gt_0} ({100 * n_pac_gt_0 / len(scores):.1f}%)")
    print(f"  Mean pac_score:    {sum(scores) / len(scores):.4f}")
    print(f"  By label:")
    print(f"    Vulnerable (1):  {vuln_pac_gt_0}/{n_vuln} with PaC>0 ({100 * vuln_pac_gt_0 / n_vuln if n_vuln else 0:.1f}%)")
    print(f"    Safe (0):       {safe_pac_gt_0}/{n_safe} with PaC>0 ({100 * safe_pac_gt_0 / n_safe if n_safe else 0:.1f}%)")
    print("=" * 50)
    if n_pac_gt_0 == 0:
        print("Still all zeros on this sample -> full Phase 3 would likely show PaC=0.")
    else:
        print("PaC is firing on this sample -> full Phase 3 should show non-zero PaC signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
