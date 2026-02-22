# Run Phase 3 on Kaggle (Full Test Recommended)

For the **full test set** (~33k samples), running Phase 3 on Kaggle is **recommended over your Mac**:

- **GPU** speeds up CodeBERT inference (large speedup vs CPU/MPS).
- **Multiple CPUs** let you run many PaC workers (`--workers 8` or more) without heating the laptop.
- **No local battery/heat**; session runs in the cloud (~1–2 hours for full run).
- **Reproducible**: same environment and inputs.

On a Mac, the full run can take **3–5+ hours** and tie up your machine; on Kaggle it’s faster and hands-off.

---

## What you need before starting

1. **Phase 2 outputs** (from your Kaggle run or local):
   - `results/phase2_validation_report.json`
   - `src/models/codebert/` (folder with `config.json`, `model.safetensors`, `tokenizer.json`, etc.)
2. **Data** (same as Phase 2): `data/curated_cpp.csv`, `data/splits.json`.

If you already have a **Phase 2 results Dataset** on Kaggle (the zip you saved with `results/` and `src/models/`), you can use that. Otherwise create one (see Step 1b).

---

## Step 1: Kaggle Datasets

### 1a. Data dataset (if you don’t have it from Phase 2)

- **Datasets** → **New dataset** → e.g. `code-reviewer-thesis-phase3-data`.
- Upload:
  - `data/curated_cpp.csv`
  - `data/splits.json`
- Note the path, e.g. `/kaggle/input/code-reviewer-thesis-phase3-data`.

(If you already created a dataset with `curated_cpp.csv` and `splits.json` for Phase 2, reuse it.)

### 1b. Phase 2 results dataset (model + report)

You need the trained CodeBERT model and the validation report on Kaggle.

**Option A – You already have a Phase 2 output zip on Kaggle**

- Use the Dataset where you saved `phase2_results.zip` (or the zip that contains `results/` and `src/models/`).
- After unzipping in the notebook, you’ll have `results/phase2_validation_report.json` and `src/models/codebert/`.

**Option B – Create a new Dataset from your Mac**

1. On your Mac, create a zip of the Phase 2 outputs (no need to include the full repo):
   ```bash
   cd /path/to/code-reviewer-thesis
   zip -r phase2_results.zip results/phase2_validation_report.json src/models/codebert
   ```
   (Or zip the `results` and `src/models` folders from your unzipped `kaggle-ml-results` if that’s where they live.)
2. **Datasets** → **New dataset** → e.g. `code-reviewer-thesis-phase2-results`.
3. Upload `phase2_results.zip`.
4. Note the path, e.g. `/kaggle/input/code-reviewer-thesis-phase2-results`.

---

## Step 2: Create a new Notebook

1. **Code** → **New notebook**.
2. **Settings** (right panel):
   - **Accelerator:** **GPU P100** (or **GPU T4**) — recommended for fast CodeBERT inference.
   - **Internet:** **On** (for `pip` and transformers).
3. **Add data:**
   - Your **data** dataset (e.g. `code-reviewer-thesis-phase3-data` or your Phase 2 data).
   - Your **Phase 2 results** dataset (e.g. `code-reviewer-thesis-phase2-results`).

Check the exact input paths (they may include a version number):

```python
!ls /kaggle/input/
```

---

## Step 3: Clone or upload repo and install dependencies

In the first cell (clone **or** upload, not both):

**Option A – Clone from GitHub**

```python
!git clone https://github.com/YOUR_USERNAME/ml-code-eval.git /kaggle/working/code-reviewer-thesis
# or: git clone https://github.com/YOUR_USERNAME/code-reviewer-thesis.git ...
```

**Option B – Upload zip**

1. On your Mac: `zip -r phase3-code.zip src requirements.txt` (from repo root).
2. In Kaggle: **File** → **Upload** the zip.
3. In a cell:
   ```python
   !cd /kaggle/working && unzip -o phase3-code.zip -d code-reviewer-thesis
   ```

Then add a cell to install dependencies:

```python
%cd /kaggle/working/code-reviewer-thesis
!pip install -q transformers tokenizers torch scikit-learn lizard semgrep tqdm accelerate pandas numpy scipy statsmodels
```

---

## Step 4: Copy data and Phase 2 model into the project

Set paths to **your** dataset names (check `/kaggle/input/` for exact names and version suffixes).

```python
import os

WORK_DIR = "/kaggle/working/code-reviewer-thesis"
# Data (curated_cpp.csv, splits.json)
DATA_IN = "/kaggle/input/code-reviewer-thesis-phase3-data"   # or your Phase 2 data dataset path
# Phase 2 results (phase2_results.zip or folder with results/ and src/models/)
PHASE2_IN = "/kaggle/input/code-reviewer-thesis-phase2-results"

os.makedirs(f"{WORK_DIR}/data", exist_ok=True)
os.makedirs(f"{WORK_DIR}/results", exist_ok=True)
os.makedirs(f"{WORK_DIR}/src/models", exist_ok=True)

# Copy data
!cp "{DATA_IN}/curated_cpp.csv" "{WORK_DIR}/data/"
!cp "{DATA_IN}/splits.json" "{WORK_DIR}/data/"

# Unzip Phase 2 results if you uploaded a zip
# If the zip has results/ and src/models/ at top level:
!cd "{WORK_DIR}" && unzip -o "{PHASE2_IN}/phase2_results.zip" -d .
# If the zip structure is different, adjust paths. You need:
#   WORK_DIR/results/phase2_validation_report.json
#   WORK_DIR/src/models/codebert/  (with config.json, model.safetensors, tokenizer*.json)
```

If your Phase 2 Dataset is a **folder** (not a zip) with `results/` and `src/models/`:

```python
!cp "{PHASE2_IN}/phase2_validation_report.json" "{WORK_DIR}/results/"
!cp -r "{PHASE2_IN}/codebert" "{WORK_DIR}/src/models/"
# Adjust PHASE2_IN paths to match your dataset layout
```

Verify:

```python
!ls -la "{WORK_DIR}/results/"
!ls -la "{WORK_DIR}/src/models/codebert/"
```

You should see `phase2_validation_report.json` and in `codebert/`: `config.json`, `model.safetensors`, `tokenizer.json`, etc.

---

## Step 5: Run Phase 3 (full test)

No `--max_test` or `--max_val` cap: use the full test set and default validation sample for hybrid tuning. Use multiple workers so PaC (Semgrep) runs in parallel.

```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase3_experiment.py --workers 8
```

- **Full test:** ~33k samples (no `--max_test`).
- **Default `--max_val`** is 2000 (enough for hybrid tuning).
- **`--workers 8`** uses 8 parallel Semgrep processes (adjust down if you get resource errors).

Expect roughly **1–2 hours** total on a Kaggle GPU notebook (CodeBERT on GPU + 8 PaC workers).

---

## Step 6: Download results

Outputs are under `/kaggle/working/code-reviewer-thesis/results/`:

- `phase3_experiment_results.csv` — main experiment results (ml_confidence, pac_score, hybrid_risk, decisions, label).
- `phase3_hybrid_config.json` — chosen alpha, beta, t_block, t_review.

**Option A – Zip and download from UI**

```python
!cd /kaggle/working/code-reviewer-thesis && zip -r phase3_results.zip results/
```

In the left **Output** panel, find `phase3_results.zip` under `/kaggle/working/code-reviewer-thesis/`, right‑click → **Download**.

**Option B – Save as Dataset**

Right‑click `phase3_results.zip` (or the `results` folder) → **Save as Dataset** (e.g. `code-reviewer-thesis-phase3-results`) for later use or to run Phase 4 on Kaggle.

---

## Quick checklist

| Step | What to do |
|------|------------|
| 1 | Data dataset (curated_cpp.csv, splits.json); Phase 2 results dataset (report + codebert model) |
| 2 | New notebook, **GPU** on, add both datasets |
| 3 | Clone repo or upload zip; `pip install` deps |
| 4 | Copy data and Phase 2 model/report into project; verify paths |
| 5 | `python src/phase3_experiment.py --workers 8` |
| 6 | Download or save `phase3_results.zip` |

---

## After Phase 3: Phase 4 (evaluation)

You can run Phase 4 on Kaggle in the same notebook (after Phase 3) or in a new one:

- Ensure `results/phase3_experiment_results.csv` is in `WORK_DIR/results/`.
- Run: `!python src/phase4_evaluation.py`
- Download `results/phase4_results_summary.txt` and `results/phase4_evaluation_report.json`.

If you saved Phase 3 results as a Dataset, create a new notebook, add that dataset + repo, copy `phase3_experiment_results.csv` into `results/`, then run Phase 4.
