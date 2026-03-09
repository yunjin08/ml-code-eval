# Full Kaggle re-run: commands in order

Use these commands in Kaggle notebooks to re-run the full pipeline. Replace placeholders: `YOUR_USERNAME`, `YOUR_KAGGLE_USERNAME`, and dataset paths (check `/kaggle/input/` after adding data).

---

## Prerequisites (once per Kaggle account)

- **Account:** [kaggle.com](https://www.kaggle.com), **Phone verification** (Settings → Account) for GPU.
- **Notebook settings:** **Accelerator: GPU P100** (or T4), **Internet: On**.

---

## Phase 1 — Dataset (optional; skip if you already have a data dataset)

Run in a **new notebook** (no GPU required; Internet On).

```python
# Clone repo (or upload zip and unzip to /kaggle/working/code-reviewer-thesis)
!git clone https://github.com/YOUR_USERNAME/code-reviewer-thesis.git /kaggle/working/code-reviewer-thesis
```

```python
%cd /kaggle/working/code-reviewer-thesis
!pip install -q datasets pandas
!python src/phase1_dataset.py
```

Then **Save as Dataset**: zip `data/curated_cpp.csv` and `data/splits.json`, create dataset e.g. `code-reviewer-thesis-phase1-data`. Use this dataset in Phase 2 (and 3).

---

## Phase 2 — Model training (CodeBERT + RF + KNN)

**Add data:** Your Phase 1 (or existing) data dataset → e.g. `/kaggle/input/code-reviewer-thesis-phase2-data`.

```python
# Clone repo
!git clone https://github.com/YOUR_USERNAME/code-reviewer-thesis.git /kaggle/working/code-reviewer-thesis
```

```python
import os
DATA_IN = "/kaggle/input/code-reviewer-thesis-phase2-data"  # check path: !ls /kaggle/input/
WORK_DIR = "/kaggle/working/code-reviewer-thesis"
os.makedirs(f"{WORK_DIR}/data", exist_ok=True)
!cp "{DATA_IN}/curated_cpp.csv" "{WORK_DIR}/data/"
!cp "{DATA_IN}/splits.json" "{WORK_DIR}/data/"
!pip install -q transformers datasets tokenizers torch scikit-learn lizard semgrep tqdm accelerate pandas numpy
```

```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase2_train.py --epochs 3 --batch_size 16
```

(If GPU fails: `!pip install -q torch --index-url https://download.pytorch.org/whl/cpu` then add `--cpu` to the command.)

**Resume from checkpoint if run stopped:**
```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase2_train.py --epochs 3 --batch_size 16 --resume_from_checkpoint auto
```

**KNN-only run (add KNN to existing report without re-running CodeBERT):**  
Ensure `results/phase2_validation_report.json` exists (from a previous full run). Then:
```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase2_train.py --skip_codebert --skip_rf
```
This loads the existing report, trains only KNN, writes KNN metrics into the report, and keeps `selected_model` and CodeBERT/RF metrics unchanged.

**Save Phase 2 outputs as Dataset (so Phase 3 can use them):**

Use the **slim** zip (excludes checkpoints; ~500MB instead of ~3GB). Phase 3 only needs the final model and report.
```python
%cd /kaggle/working/code-reviewer-thesis
!zip -r phase2_results.zip results src/models/codebert src/models/rf.pkl src/models/knn.pkl
```
Then right‑click `phase2_results.zip` in Output → **Save as Dataset** (e.g. `code-reviewer-thesis-phase2-results`).

(Full zip including checkpoints, if you need them for resume: `zip -r phase2_results_full.zip results src/models` — large ~3GB.)

---

## Phase 3 — Governance experiment (full test)

**Add data:** (1) Same data dataset (curated_cpp.csv, splits.json), (2) Phase 2 results dataset (phase2_results.zip or folder with `results/` and `src/models/`).

```python
!git clone https://github.com/YOUR_USERNAME/code-reviewer-thesis.git /kaggle/working/code-reviewer-thesis
```

```python
%cd /kaggle/working/code-reviewer-thesis
!pip install -q transformers tokenizers torch scikit-learn lizard semgrep tqdm accelerate pandas numpy scipy statsmodels
```

```python
import os
WORK_DIR = "/kaggle/working/code-reviewer-thesis"
DATA_IN = "/kaggle/input/code-reviewer-thesis-phase3-data"      # or your data dataset path
PHASE2_IN = "/kaggle/input/phase2-results-2"  # Phase 2 results dataset (phase2_results-2)

os.makedirs(f"{WORK_DIR}/data", exist_ok=True)
os.makedirs(f"{WORK_DIR}/results", exist_ok=True)
os.makedirs(f"{WORK_DIR}/src/models", exist_ok=True)
!cp "{DATA_IN}/curated_cpp.csv" "{WORK_DIR}/data/"
!cp "{DATA_IN}/splits.json" "{WORK_DIR}/data/"
!cd "{WORK_DIR}" && unzip -o "{PHASE2_IN}/phase2_results.zip" -d .
```

```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase3_experiment.py --workers 8 --resume --backup_to_kaggle_dataset YOUR_KAGGLE_USERNAME/code-reviewer-thesis-phase3-results
```

Replace `YOUR_KAGGLE_USERNAME` with your Kaggle username (e.g. `donairejededison`). Creates/updates dataset `code-reviewer-thesis-phase3-results` with the CSV and config when the run finishes.

---

## Phase 4 — Evaluation

Run **in the same notebook after Phase 3** (so `results/phase3_experiment_results.csv` is already there):

```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase4_evaluation.py
```

Then download from Output: `results/phase4_evaluation_report.json`, `results/phase4_results_summary.txt`.

**If Phase 4 runs in a new notebook** (e.g. you only have the Phase 3 results dataset):

1. Add data: Phase 3 results dataset (with `phase3_experiment_results.csv`).
2. Clone repo, pip install (same as Phase 3 setup).
3. Copy Phase 3 CSV into project:
   ```python
   import os
   WORK_DIR = "/kaggle/working/code-reviewer-thesis"
   PHASE3_IN = "/kaggle/input/code-reviewer-thesis-phase3-results"
   os.makedirs(f"{WORK_DIR}/results", exist_ok=True)
   !cp "{PHASE3_IN}/phase3_experiment_results.csv" "{WORK_DIR}/results/"
   ```
4. Run:
   ```python
   %cd /kaggle/working/code-reviewer-thesis
   !python src/phase4_evaluation.py
   ```

---

## One-page command summary

| Phase | Main command (Kaggle notebook) |
|-------|---------------------------------|
| **1** | `!python src/phase1_dataset.py` |
| **2** | `!python src/phase2_train.py --epochs 3 --batch_size 16` |
| **2 resume** | `!python src/phase2_train.py --epochs 3 --batch_size 16 --resume_from_checkpoint auto` |
| **3** | `!python src/phase3_experiment.py --workers 8 --resume --backup_to_kaggle_dataset YOUR_KAGGLE_USERNAME/code-reviewer-thesis-phase3-results` |
| **4** | `!python src/phase4_evaluation.py` |

Before each phase: set `%cd /kaggle/working/code-reviewer-thesis`, ensure data and (for 3) Phase 2 results are in place. See sections above for dataset paths and zip/setup steps.
