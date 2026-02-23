# Verification Attestation: Methodology & Results

**Date:** 2026-02-23  
**Scope:** Phase 3 experiment results and Phase 4 evaluation correctness.

---

## 1. Data Verified

| Check | Result |
|-------|--------|
| **Test set size** | 150 rows (current `phase3_experiment_results.csv`) |
| **Class balance** | 143 non-vulnerable (95.3%), 7 vulnerable (4.67%) — severe imbalance |
| **Note** | Full Kaggle run was 33,050 samples; local file may be from `--max_test 150` run |

---

## 2. Primary Metrics — Correct

| Policy | TP | FP | FN | TN | Precision | Recall | F1 |
|--------|----|----|----|----|-----------|--------|-----|
| ML-Only | 1 | 4 | 6 | 139 | 0.20 | 0.143 | 0.167 |
| PaC-Only | 0 | 0 | 7 | 143 | 0.00 | 0.00 | 0.00 |
| Hybrid | 1 | 4 | 6 | 139 | 0.20 | 0.143 | 0.167 |

- **Precision** = TP/(TP+FP): ML = 1/5 = 0.20 ✓  
- **Recall** = TP/(TP+FN): ML = 1/7 ≈ 0.143 ✓  
- **F1** = 2PR/(P+R) ≈ 0.167 ✓  
- Phase 4 report matches these values. **Metrics computed correctly.**

---

## 3. PaC Coverage — Verified

- **pac_score > 0:** 0 / 150 (0.0%)
- Semgrep `p/c` rules did not fire on any sample in this test set.
- PaC-Only F1=0 is therefore correct: no true positives, no false positives.

---

## 4. Hybrid = ML — Expected

- **hybrid_risk** = `alpha * ml_confidence + beta * pac_score`
- With **pac_score = 0** for all rows: `hybrid_risk = alpha * ml_confidence`
- Tuned config: `alpha=0.75` (Kaggle) or `alpha=1` (local) → hybrid collapses to ML.
- **decision_ml == decision_hybrid** for all 150 rows. Correct.

---

## 5. Phase 4 Logic — Correct

- Block = decision 2; binary pred = 1 for Block, 0 otherwise.
- McNemar: discordant pairs (only_a, only_b) computed correctly.
- ROC-AUC: uses continuous scores (ml_confidence, pac_score, hybrid_risk) vs y_true. PaC AUC=0.5 (no discrimination) is correct when all pac_scores=0.

---

## 6. Attestation Summary

| Item | Status |
|------|--------|
| Class imbalance | Confirmed (4.67% vulnerable) |
| Primary metrics (P/R/F1) | Correct |
| PaC zero coverage | Confirmed — rules did not fire |
| Hybrid = ML when PaC=0 | Correct behavior |
| Phase 4 evaluation logic | Correct |
| McNemar, ROC-AUC | Correct |

**Conclusion:** The pipeline and evaluation were implemented correctly. The observed results (low F1, PaC=0, hybrid=ML) are consistent with: (1) severe class imbalance, (2) Semgrep p/c rules not matching the CWE types in this subset, and (3) threshold tuning favoring ML when PaC adds no signal.
