# Notebooks

- **`findings.ipynb`** — Full pipeline findings (Phase 1–4): **Phase 1** dataset splits (`data/splits.json`), **Phase 2** validation report and model choice (`phase2_validation_report.json`), **Phase 3** hybrid config, **Phase 4** primary metrics, ROC AUC, McNemar, hypotheses, and optional experiment snapshot. Looks for files in `results/` or `phase3_results/` and `data/splits.json`.

## How to run

**Option A — From terminal (repo root)**

```bash
cd /path/to/code-reviewer-thesis
pip install jupyter pandas matplotlib   # if needed
jupyter notebook notebooks/findings.ipynb
```

Or with JupyterLab: `jupyter lab notebooks/findings.ipynb`

**Option B — From Cursor / VS Code**

1. Open `notebooks/findings.ipynb`.
2. Pick the kernel: click “Select Kernel” (top right) → choose your `.venv` or the Python that has `pandas` and `matplotlib`.
3. Run all: “Run All” in the notebook toolbar, or run cells one by one (Shift+Enter).

**Tip:** Run from the **repo root** (e.g. open the folder `code-reviewer-thesis` in Cursor). The notebook detects `notebooks/` vs root and sets paths so `results/` and `phase3_results/` are found.
