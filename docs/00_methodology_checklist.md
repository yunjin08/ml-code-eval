# Methodology Checklist – What’s Done vs What’s Left

Use this as your run order and checklist. Details are in [01_data_gathering_and_exploration.md](01_data_gathering_and_exploration.md), [02_model_training.md](02_model_training.md), and [03_evaluation.md](03_evaluation.md).

---

## Phase 1: Data gathering and exploration

| Step | What to do | Status |
|------|------------|--------|
| 1.1 | **Data source:** Load DiverseVul (HF `bstee615/diversevul`). | **Done** – Implemented in `phase1_dataset.py` and run once. |
| 1.2 | **Curation:** Use HF train/val/test splits; standardize to `id`, `code`, `label`, `split`; write `data/curated_cpp.csv` and `data/splits.json`. | **Done** – Script run; CSV and splits exist. |
| 1.3 | **Exploration:** Check missing data and class balance; document in doc 01. | **Done** – Checked and documented (1 missing `code`, ~94% / 6% balance). |
| 1.4 | **Fix missing row:** Drop any row with missing or empty `code` so every sample is valid. | **Done** – Implemented in `phase1_dataset.py` via `drop_missing_code()`. |


---

## Phase 2: Model training

| Step | What to do | Status |
|------|------------|--------|
| 2.1 | **Train CodeBERT** on full train set (264k), validate on full validation set (33k); save best checkpoint to `src/models/codebert/`. | **Not done** – Code is ready; only partial or skipped runs so far. |
| 2.2 | **Train Random Forest** on full train set; validate on full validation set; save `src/models/rf.pkl`. | **Not done** – Code is ready; only a dev run with `--max_train 5000 --max_val 1000` was run. |
| 2.3 | **Benchmark:** Compute validation Precision, Recall, F1 for both models; choose the model with higher F1. | **Not done** – Need full training first; then report will have final comparison. |
| 2.4 | **Record:** Ensure `results/phase2_validation_report.json` has `selected_model` and both models’ metrics. | **Not done** – Will be produced by the full Phase 2 run. |

**Next action:** Run full Phase 2 (no `--max_train`, no `--max_val`, no `--skip_codebert` / `--skip_rf`). After it finishes, update doc 02 with final validation F1 and hyperparameters.

---

## Phase 3: Governance experiment

| Step | What to do | Status |
|------|------------|--------|
| 3.1 | **ML-Only:** Run selected model on full test set (33k); get `ml_confidence` and Block/Review/Approve decisions. | **Not done** – Need Phase 2 full run first; then run Phase 3 on full test set. |
| 3.2 | **PaC-Only:** Run Semgrep on each test sample; get normalized PaC score and decisions. | **Not done** – Implemented; only run on a small `--max_test` subset so far. |
| 3.3 | **Hybrid:** Tune α, β and thresholds on validation; compute hybrid risk and decisions on full test set. | **Not done** – Same as above; full test run pending. |
| 3.4 | **Outputs:** Produce `results/phase3_experiment_results.csv` and `results/phase3_hybrid_config.json` for the full test set. | **Not done** – Current CSV is from a small or synthetic run. |

**Next action:** After Phase 2 is complete, run Phase 3 with **no** `--max_test` (and no `--max_val` if you want full tuning). This will take time (e.g. Semgrep per sample). Then you will have the real experiment results for Phase 4.

---

## Phase 4: Final evaluation

| Step | What to do | Status |
|------|------------|--------|
| 4.1 | **Primary metrics:** Precision, Recall, F1 for Block decision (ML-Only, PaC-Only, Hybrid). | **Done** – Implemented and run once on current Phase 3 output (small/synthetic). |
| 4.2 | **McNemar’s test:** Hybrid vs ML-Only, Hybrid vs PaC-Only; report p-values. | **Done** – Implemented and in report. |
| 4.3 | **Effect size:** Odds ratio with 95% CI for those comparisons. | **Done** – In report. |
| 4.4 | **ROC/AUC:** For ML, PaC, Hybrid risk scores. | **Done** – In report. |
| 4.5 | **Hypotheses:** Evaluate H1, H2, H3 and document in report/summary. | **Done** – In report and summary. |
| 4.6 | **Final run:** Run Phase 4 **after** Phase 3 full run; then update doc 03 with final numbers and hypothesis outcomes. | **Not done** – Need to re-run Phase 4 on the real Phase 3 results and document. |

**Next action:** After Phase 3 (full test set) is done, run Phase 4 once. Then paste the final metrics and H1–H3 results into doc 03 (or a “Results” subsection).

---

## Summary: what’s done vs what’s left

**Done**

- Phase 1: Data source, curation script, exploration, and docs.
- Phase 2: Training code for both models and selection logic.
- Phase 3: Experiment code (ML / PaC / Hybrid) and tuning.
- Phase 4: All evaluation code and a test run on small/synthetic data.
- Docs 01, 02, 03 and this checklist.

**Not done (required for thesis)**

1. **Phase 1 re-run:** Re-run Phase 1 so the curated CSV has zero missing data (drop step is now in the script).
2. **Phase 2 full run:** Train both CodeBERT and RF on full train/validation; get `phase2_validation_report.json` and the selected model.
3. **Phase 3 full run:** Run governance experiment on the **full test set**; get real `phase3_experiment_results.csv` and `phase3_hybrid_config.json`.
4. **Phase 4 final run:** Run evaluation on that Phase 3 output.
5. **Document:** Update docs 02 and 03 with final validation metrics, experiment summary, and hypothesis results.

---

## Timeframe and storage (rough guide)

### Timeframe

| Phase | What runs | Rough duration (full data) |
|-------|-----------|----------------------------|
| **Phase 1** | Load HF dataset, curate, write CSV | ~1–2 min (network-bound) |
| **Phase 2** | CodeBERT fine-tuning (264k train, 3 epochs) | **~15–45 h** on GPU; **3–10 days** on CPU only |
| | Random Forest: lizard features (264k + 33k) then train | **~15–25 h** (CPU-bound; lizard per sample) |
| | Phase 2 total (sequential) | **~1.5–3 days** with GPU; **~1–2 weeks** CPU-only |
| **Phase 3** | ML inference on 33k test | Minutes–~1 h |
| | Semgrep on 33k test samples (~2–3 s each) | **~20–25 h** |
| | Phase 3 total | **~1 day** |
| **Phase 4** | Metrics, McNemar, ROC, hypotheses | **&lt; 5 min** |

**End-to-end (full pipeline after Phase 1):** about **2–4 days** with a GPU (Phase 2 + Phase 3 dominate). Without a GPU, Phase 2 alone can be a week or more.

### Storage

| Item | Rough size |
|------|------------|
| `data/curated_cpp.csv` (330k rows, long `code`) | **~2–6 GB** |
| `data/splits.json` | &lt; 10 MB |
| CodeBERT model + checkpoints (`src/models/codebert*`) | **~1–2 GB** |
| Random Forest `rf.pkl` | ~10–50 MB |
| `results/phase3_experiment_results.csv` (33k rows + code) | **~0.5–2 GB** |
| Other results (reports, configs) | &lt; 50 MB |
| Hugging Face cache (dataset + CodeBERT) | **~1–2 GB** |

**Total:** plan for **~5–12 GB** free disk (plus OS/tools). The heavy parts are the curated CSV, CodeBERT weights, and (if you keep code in it) the Phase 3 results CSV.

---

## Recommended run order (to finish the methodology)

```text
1. Re-run Phase 1 (curated set will have no missing code):
   python src/phase1_dataset.py

2. Full model training (Phase 2):
   python src/phase2_train.py
   → Then update docs/02_model_training.md with final validation F1 and settings.

3. Full governance experiment (Phase 3):
   python src/phase3_experiment.py
   → No --max_test. Expect long runtime for Semgrep on 33k samples.

4. Final evaluation (Phase 4):
   python src/phase4_evaluation.py
   → Then update docs/03_evaluation.md with final metrics and H1–H3.
```
