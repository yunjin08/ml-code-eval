# 3. Final Evaluation

This document describes the governance experiment (Phase 3) and the statistical evaluation (Phase 4): what is computed, how to run it, and how to interpret the results. Iterate these steps after full model training and full test-set runs.

## 3.1 Governance experiment (Phase 3)

### 3.1.1 Inputs

- **Test set:** Rows in `data/curated_cpp.csv` whose `id` is in `data/splits.json` → `test_ids`.  
- **Selected ML model:** From `results/phase2_validation_report.json` → `selected_model` (CodeBERT or Random Forest).  
- **Validation set (for tuning):** Used to choose Hybrid weights (α, β) and decision thresholds (Block / Review).

### 3.1.2 What Phase 3 does

1. **ML-Only:** For each test sample, run the selected model → `ml_confidence` in [0, 1].  
2. **PaC-Only:** For each test sample, write code to a temp file, run Semgrep (config `p/c`), count findings → normalized **PaC score** in [0, 1] (e.g. min(1, count/10)).  
3. **Hybrid:**  
   - **Risk formula:** `hybrid_risk = α * ml_confidence + β * pac_score` with α + β = 1.  
   - **Tuning:** Grid search over (α, β) and thresholds on the **validation** set to maximize F1 for the **Block** decision.  
   - **Test:** Apply the chosen α, β and thresholds to the test set.  
4. **Decisions:** For each approach, map scores to three outcomes: **Approve** (0), **Review** (1), **Block** (2) using the tuned thresholds (e.g. Block if score ≥ θ_block, Review if score ≥ θ_review).

### 3.1.3 Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| Experiment results | `results/phase3_experiment_results.csv` | Per test sample: `id`, `code`, `label`, `split`, `ml_confidence`, `pac_score`, `hybrid_risk`, `decision_ml`, `decision_pac`, `decision_hybrid`. |
| Hybrid config | `results/phase3_hybrid_config.json` | Chosen `alpha`, `beta`, `t_block`, `t_review`. |

**Command (full test set):**

```bash
python src/phase3_experiment.py
```

**Optional (faster, not for final thesis):** `--max_test N`, `--max_val N`.

---

## 3.2 Statistical evaluation (Phase 4)

Phase 4 reads `results/phase3_experiment_results.csv` and computes all metrics and tests below. **Ground truth for Block:** `label == 1` (critically defective).

### 3.2.1 Primary metrics (Block decision)

- For **ML-Only**, **PaC-Only**, and **Hybrid:**  
  - Binary prediction: **Block** if decision = 2, else non-Block.  
  - **Precision(Block):** Among predicted Block, proportion that are truly defective.  
  - **Recall(Block):** Among truly defective, proportion that were Blocked.  
  - **F1(Block):** Harmonic mean of Precision and Recall.

These are the main performance numbers for the thesis (Section 3.5.1).

### 3.2.2 McNemar’s test

- **Pairs:** Hybrid vs ML-Only; Hybrid vs PaC-Only (same test samples).  
- **Purpose:** Test whether the difference in correct/incorrect Block decisions is statistically significant (paired nominal outcomes).  
- **Implementation:** 2×2 table of (Hybrid prediction, Other prediction); chi-squared statistic from discordant pairs; p-value from chi-squared distribution (df=1).  
- **Reported:** p-value for each comparison in `results/phase4_evaluation_report.json` under `mcnemar`.

### 3.2.3 Effect size (odds ratio)

- **Quantity:** Odds of correct Block decision for Hybrid vs the other approach.  
- **Reported:** Odds ratio and 95% confidence interval (approximate, from 2×2 table).  
- **Location:** In `phase4_evaluation_report.json` under `mcnemar` → `Hybrid_vs_ML` and `Hybrid_vs_PaC` (e.g. `odds_ratio`, `ci_95`).

### 3.2.4 ROC and AUC

- **ML-Only:** ROC curve and AUC from `ml_confidence` vs binary `label`.  
- **Hybrid:** ROC curve and AUC from `hybrid_risk` vs `label`.  
- **PaC-Only:** AUC from `pac_score` vs `label` (when applicable).  
- **Reported:** AUC values in `phase4_evaluation_report.json` under `roc_auc`.

### 3.2.5 Hypothesis validation

- **H1:** Hybrid F1(Block) > ML-Only F1 and > PaC-Only F1.  
- **H2:** Hybrid Recall(Block) > PaC-Only Recall.  
- **H3:** Hybrid Precision(Block) > ML-Only Precision.  

For each, we report whether the comparison supports the hypothesis and the relevant McNemar p-value (and odds ratio where useful). Stored under `hypotheses` in the report.

### 3.2.6 Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| Full report | `results/phase4_evaluation_report.json` | All metrics, McNemar, odds ratios, AUCs, hypothesis flags. |
| Summary | `results/phase4_results_summary.txt` | Human-readable summary of metrics and H1–H3. |

**Command:**

```bash
python src/phase4_evaluation.py
```

(No arguments; reads Phase 3 results from the default path.)

---

## 3.3 Run order for final thesis

1. **Phase 1:** Curate data (already done). Optionally drop the one row with missing `code` and re-save.  
2. **Phase 2:** Full training (no `--max_train` / `--max_val`). Run and record validation F1 for both models.  
3. **Phase 3:** Full test set (no `--max_test`). Produces the final `phase3_experiment_results.csv` and `phase3_hybrid_config.json`.  
4. **Phase 4:** Run on that Phase 3 output. Produces the final evaluation report and summary.

Then update this document (and Section 2.7 in the training doc) with the final numbers and whether H1–H3 are supported.

## 3.4 Iteration notes

- After each full run, paste the key results (P/R/F1 per approach, McNemar p-values, AUCs, H1–H3) into this doc or into a short “Results” subsection.  
- If you change the Block/Review thresholds or the Hybrid formula, document the change here and re-run Phase 3 and Phase 4.
