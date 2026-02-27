# Notebooks

- **`findings.ipynb`** — Full pipeline findings (Phase 1–4): **Phase 1** dataset splits (`data/splits.json`), **Phase 2** validation report and model choice (`phase2_validation_report.json`), **Phase 3** hybrid config, **Phase 4** primary metrics, ROC AUC, McNemar, hypotheses, and optional experiment snapshot. Run from repo root or from this folder. Requires: `pandas`, `matplotlib`. Looks for files in `results/` or `phase3_results/` and `data/splits.json`.
