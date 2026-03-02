# Notebooks

- **`findings.ipynb`** — Full pipeline findings (Phase 1–4): Phase 1 splits, Phase 2 validation & model choice, Phase 3 hybrid config (α, β, thresholds), Phase 4 primary metrics, ROC AUC, McNemar, and hypotheses (H1–H3). Reads from `results/` or `phase3_results/` and `data/splits.json`.

## How to run

**Prerequisites:** Phase 1–4 must have been run (or results copied into the repo). Required files:
- `data/splits.json`
- `results/phase2_validation_report.json`
- `results/phase3_hybrid_config.json`
- `results/phase4_evaluation_report.json`
- (optional) `results/phase3_experiment_results.csv` for the snapshot table

**Option A — Terminal (from repo root)**

```bash
cd /path/to/code-reviewer-thesis
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install jupyter pandas matplotlib   # if not already installed
jupyter notebook notebooks/findings.ipynb
```

Or JupyterLab: `jupyter lab notebooks/findings.ipynb`

**Option B — Cursor / VS Code**

1. Open the repo folder `code-reviewer-thesis` (so paths resolve correctly).
2. Open `notebooks/findings.ipynb`.
3. **Select kernel:** Click “Select Kernel” (top right) → choose the repo’s `.venv` (or any Python with `pandas`, `matplotlib`).
4. **Run:** “Run All” in the toolbar, or run cells with Shift+Enter.

**Note:** The notebook infers the project root from the current working directory. If you open the file from inside the `notebooks/` folder, it still finds `results/` and `data/` relative to the repo root.
