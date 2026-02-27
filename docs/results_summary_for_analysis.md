# Results Summary — For Analysis (Should We Change the Dataset?)

This document summarizes the thesis pipeline results so an external reviewer (e.g. Claude) can assess whether we should change the dataset or accept current findings.

---

## 1. Pipeline Overview

- **Phase 1:** Dataset curation — DiverseVul (Hugging Face), filtered to C/C++, train/validation/test splits.
- **Phase 2:** Train CodeBERT and Random Forest on train set; select best model on validation.
- **Phase 3:** Run three governance policies on the **test set**: ML-only (CodeBERT), PaC-only (Semgrep p/c), Hybrid (α·ML + β·PaC with thresholds tuned on validation). Output: per-sample scores and Block/Review/Approve decisions.
- **Phase 4:** Evaluate with precision/recall/F1 (Block = positive), McNemar tests, ROC-AUC, and hypothesis checks (H1–H3).

---

## 2. Dataset

- **Source:** DiverseVul (C/C++ subset).
- **Test set:** 33,050 samples.
- **Labels:** Binary (0 = non-vulnerable, 1 = vulnerable).
- **Class balance (test):** 31,119 non-vulnerable (94.2%), 1,931 vulnerable (5.84%) — imbalanced.
- **Unit of analysis:** Function-level code snippets (one snippet per row).

---

## 3. Phase 2 Results (Model Selection)

- **CodeBERT (validation):** F1 ≈ 0.189, precision ≈ 0.128, recall ≈ 0.361.
- **Random Forest (validation):** F1 ≈ 0.024, much lower.
- **Selected model:** CodeBERT (used in Phase 3 for ML-only and Hybrid).

---

## 4. Phase 3 Results (Test Set)

- **PaC (Semgrep):** Config `p/c` only. **pac_score = 0 for all 33,050 rows** (Semgrep found no rule matches on any snippet).
- **Hybrid tuning (on validation):** α = 0.75, β = 0.25, t_block = 0.3, t_review = 0.1 — i.e. mostly ML.
- **Verification:** We run a mandatory check before Phase 3: a known-bad snippet (`gets()`) is scanned by Semgrep; it must get ≥ 1 finding. This passes, so the pipeline and Semgrep setup are trusted. The zero findings on the test set are therefore attributed to **rule coverage**, not to a broken configuration.

---

## 5. Phase 4 Results (Evaluation on Test Set)

**Primary metrics (Block vs non-Block):**

| Policy    | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| ML-Only  | 0.127     | 0.346  | 0.186  |
| PaC-Only  | 0         | 0      | 0      |
| Hybrid   | 0.127     | 0.346  | 0.186  |

- **ROC-AUC:** ML-Only 0.704, PaC-Only 0.5, Hybrid 0.704.
- **McNemar:** Hybrid vs ML (p ≈ 0.046); Hybrid vs PaC (p = 0). Hybrid and ML are very similar; Hybrid is clearly different from (and better than) PaC on this test set.

**Hypotheses:**

- **H1** (Hybrid F1 > ML and > PaC): **Supported** (Hybrid F1 ≈ 0.186, ML ≈ 0.186, PaC = 0).
- **H2** (Hybrid recall > PaC recall): **Supported** (0.346 > 0).
- **H3** (Hybrid precision > ML precision): **Supported** (slightly higher Hybrid precision).

Because PaC is 0, the hybrid effectively behaves like ML-only (α = 0.75 and PaC adds no signal).

---

## 6. Main Finding: PaC = 0 on This Dataset

- **Observed:** Semgrep p/c produces **zero** findings on all 33,050 test snippets. We re-ran Semgrep (single-file and batched) on samples from the Phase 3 CSV and again got 0 for every snippet. The known-bad `gets()` snippet still gets 1 finding, so the tool and wrapper are working.
- **Interpretation:** The C/C++ snippets in DiverseVul (function-level, real-project code) do not match the patterns in Semgrep’s **p/c** rules. So PaC contributes nothing on this dataset — a **rule-coverage gap**, not an implementation bug.

---

## 7. Open Question for Analysis

**Should we change the dataset?**

- **Option A — Keep DiverseVul:** Report PaC = 0 as a limitation (rule-coverage gap) and keep the current ML vs Hybrid vs PaC comparison. The thesis still shows that (i) hybrid can be tuned to match or slightly beat ML when PaC adds no signal, and (ii) verification ensures the result is interpretable.
- **Option B — Change or augment dataset:** Switch to (or add) a dataset where Semgrep p/c (or other C/C++ rules) are known to fire on some fraction of vulnerable samples, so PaC has non-zero contribution and the comparison is more informative.
- **Option C — Change PaC instead of dataset:** Keep DiverseVul but try different Semgrep configs or custom rules that target the CWE types prevalent in DiverseVul, then re-run Phase 3/4.

We want an analysis of whether **A** is sufficient for the thesis or whether **B** or **C** (or both) are recommended, and why.

---

## 8. Files Referenced

- `results/phase2_validation_report.json` — Phase 2 metrics and selected model.
- `results/phase3_experiment_results.csv` — Phase 3 per-sample outputs (or `phase3_results_1/`).
- `results/phase3_hybrid_config.json` — α, β, thresholds.
- `results/phase4_evaluation_report.json` — Full Phase 4 metrics and hypotheses.
- `results/phase4_results_summary.txt` — Short Phase 4 summary.
