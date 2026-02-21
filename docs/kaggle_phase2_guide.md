# Run Phase 2 on Kaggle (Free GPU)

Use Kaggle’s free GPU (P100) to run Phase 2 in ~5–7 hours instead of days on CPU.

**Important:** Kaggle’s `/kaggle/working/` is **temporary**. If the session dies (timeout, refresh, disconnect), everything there—including checkpoints—is **gone**. Save checkpoints to a **Dataset** (see below) so you can resume in a new session instead of starting over.

---

## Step 1: Create a Kaggle account

1. Go to [kaggle.com](https://www.kaggle.com).
2. Sign up or log in (Google account is fine).
3. In **Settings** → **Account**, enable **Phone verification** if required (needed for GPU).

---

## Step 2: Upload your data as a Dataset

1. Click **Datasets** in the left menu → **New dataset**.
2. **Dataset name:** e.g. `code-reviewer-thesis-phase2-data`.
3. Upload these two files from your computer:
   - `data/curated_cpp.csv`
   - `data/splits.json`
   (Drag and drop or click to upload. May take a few minutes for the CSV.)
4. Click **Create**. Keep visibility **Private** if you prefer.
5. Note the dataset URL, e.g. `kaggle.com/datasets/YOUR_USERNAME/code-reviewer-thesis-phase2-data`. You’ll add this to your notebook.

---

## Step 3: Create a new Notebook

1. Click **Code** in the left menu → **New notebook**.
2. **Settings** (right panel):
   - **Accelerator:** **GPU P100** (or **GPU T4** if P100 isn’t available).
   - Leave **Internet** **On** (to install packages and load CodeBERT).
3. **Add data:** Click **+ Add data** → search for your dataset (`code-reviewer-thesis-phase2-data`) → **Add**. It will mount at `/kaggle/input/code-reviewer-thesis-phase2-data/` (path may include a version number; check the path shown after adding).

---

## Step 4: Add your code to the notebook

You can either **clone from GitHub** or **upload a zip** of the repo.

### Option A: Clone from GitHub (if your repo is on GitHub)

In the first notebook cell, run:

```python
!git clone https://github.com/YOUR_USERNAME/code-reviewer-thesis.git /kaggle/working/code-reviewer-thesis
```

Replace `YOUR_USERNAME` with your GitHub username. If the repo is private, you’ll need to use a token or upload a zip instead.

### Option B: Upload project zip

1. On your Mac, create a zip **without** the large data files (they’re already in the Kaggle dataset):
   ```bash
   cd /path/to/code-reviewer-thesis
   zip -r phase2-code.zip src requirements.txt
   ```
2. In the Kaggle notebook: **File** → **Upload** (or drag the zip into the file browser on the left).
3. In a notebook cell:
   ```python
   !cd /kaggle/working && unzip -o phase2-code.zip -d code-reviewer-thesis
   ```
   (Adjust the zip filename if different.)

---

## Step 5: Set up data path and install dependencies

Add a new cell and run:

```python
import os

# Path where Kaggle mounted your dataset (check the path after "Add data")
DATA_IN = "/kaggle/input/code-reviewer-thesis-phase2-data"  # or .../code-reviewer-thesis-phase2-data/version/1
WORK_DIR = "/kaggle/working/code-reviewer-thesis"
os.makedirs(f"{WORK_DIR}/data", exist_ok=True)

# Copy data into project (Kaggle input is read-only)
!cp "{DATA_IN}/curated_cpp.csv" "{WORK_DIR}/data/"
!cp "{DATA_IN}/splits.json" "{WORK_DIR}/data/"

# Install dependencies (no --cpu; we use GPU)
!pip install -q transformers datasets tokenizers torch scikit-learn lizard semgrep tqdm accelerate pandas numpy scipy statsmodels

# Optional: install CPU-only torch if you get CUDA errors (then use --cpu in the run command)
# !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
```

If your dataset path has a version suffix (e.g. `.../code-reviewer-thesis-phase2-data/1`), set `DATA_IN` to that full path. You can check with:

```python
!ls /kaggle/input/
```

---

## Step 6: Run Phase 2 training

Add another cell:

```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase2_train.py --epochs 3 --batch_size 16
```

Do **not** pass `--cpu`; the notebook is using the GPU. If you see a CUDA/GPU error, you can fall back to CPU by installing CPU-only torch (see comment in Step 5) and running:

```python
!python src/phase2_train.py --epochs 3 --batch_size 16 --cpu
```

Run the cell. Training will take about **5–7 hours** with GPU. You can leave the tab open or check back later (Kaggle may disconnect after idle; long-running sessions usually keep going).

**As soon as at least one epoch has finished**, run the next section so checkpoints are saved to a Dataset. If you can’t run another cell while training (UI stuck), do it **as soon as the run stops or finishes**—before you refresh or leave the session.

---

## Step 6b: Save checkpoints to a Dataset (do this so you don’t lose them)

Checkpoints only exist in `/kaggle/working/`. If the session dies or you refresh, they’re gone. Save them to a **Kaggle Dataset** so you can resume in a new session.

**When to run this:** After training finishes, or after you stop the run (so whatever checkpoints exist are saved). Run this in a **new cell** while you’re still in the same session.

**1. Zip the checkpoint folder** (run in a cell):

```python
%cd /kaggle/working/code-reviewer-thesis
!zip -r /kaggle/working/codebert_checkpoints.zip src/models/codebert_checkpoint
```

**2. Save that zip as a Dataset** so it persists:

- **Option A (UI):** In the right-hand **Output** panel, find `codebert_checkpoints.zip` under `/kaggle/working/`. Right-click it → **Save as Dataset** (or use the **Save version** dropdown and choose to save output). Name it e.g. `code-reviewer-thesis-phase2-checkpoints`.
- **Option B (API):** If you have Kaggle API set up in the notebook (e.g. API key in `/kaggle/input/` or env), you can create a dataset from the zip; for most users, Option A is enough.

**3. Note the Dataset name** (e.g. `YOUR_USERNAME/code-reviewer-thesis-phase2-checkpoints`). You’ll add it as input when you want to resume.

**Repeat after more progress if you want:** If you stop the run again later (e.g. after epoch 2), run the zip cell again and save a new version of the Dataset so the latest checkpoints are backed up.

---

## Step 7: Save the results

When training finishes, the model and report are under `/kaggle/working/code-reviewer-thesis/`:

- `src/models/codebert/` (and possibly `src/models/rf.pkl`)
- `results/phase2_validation_report.json`

To download them:

1. In the left **Files** panel, open `kaggle/working/code-reviewer-thesis/`.
2. Right-click `results` and `src/models` → **Download** (or zip them first in a cell):
   ```python
   !cd /kaggle/working/code-reviewer-thesis && zip -r phase2_results.zip results src/models
   ```
   Then download `phase2_results.zip` from the file browser.

---

## Quick checklist

| Step | What to do |
|------|------------|
| 1 | Kaggle account + phone verification (for GPU) |
| 2 | New dataset: upload `curated_cpp.csv` + `splits.json` |
| 3 | New notebook, set **Accelerator: GPU P100**, add your dataset |
| 4 | Clone repo or upload zip; unzip to `/kaggle/working/code-reviewer-thesis` |
| 5 | Set `DATA_IN` to your dataset path; copy data into `.../data/`; `pip install` deps |
| 6 | `python src/phase2_train.py --epochs 3 --batch_size 16` |
| 7 | Download `results/` and `src/models/` (or the zip) |

---

## If the session disconnects or you stop the run

Kaggle notebooks can time out after long idle. To reduce the chance:

- Keep the browser tab open and avoid putting the machine to sleep.
- Run in one go; don’t leave the notebook idle for hours between cells.

**Resuming from the last checkpoint:** If training stops partway (e.g. at 91%), you can continue from the last saved checkpoint instead of starting over. Phase 2 saves checkpoints at the end of each epoch under `src/models/codebert_checkpoint/` (e.g. `checkpoint-16525`, `checkpoint-33050`). Run the same command with `--resume_from_checkpoint auto`:

```python
%cd /kaggle/working/code-reviewer-thesis
!python src/phase2_train.py --epochs 3 --batch_size 16 --resume_from_checkpoint auto
```

`auto` picks the latest checkpoint in `src/models/codebert_checkpoint/`. To use a specific folder:

```python
!python src/phase2_train.py --epochs 3 --batch_size 16 --resume_from_checkpoint src/models/codebert_checkpoint/checkpoint-33050
```

(Replace `checkpoint-33050` with the latest you see from `!ls src/models/codebert_checkpoint/`.)

**Resuming in a new session (after you lost the previous one):** If you had saved checkpoints to a Dataset (Step 6b), you can resume without starting from 0:

1. **New notebook** (or same notebook after session was reset). Add **two** datasets: your phase2 data (e.g. `code-reviewer-thesis-phase2-data`) and your checkpoints dataset (e.g. `code-reviewer-thesis-phase2-checkpoints`).
2. Do **Step 4** (clone or unzip repo) and **Step 5** (data path, copy data, pip install) as usual. Set the checkpoints input path, e.g.:
   ```python
   CHECKPOINTS_IN = "/kaggle/input/code-reviewer-thesis-phase2-checkpoints"  # or .../version/N
   ```
3. **Unzip the checkpoints** into the project so the script can find them:
   ```python
   !cd /kaggle/working/code-reviewer-thesis && unzip -o "$CHECKPOINTS_IN/codebert_checkpoints.zip" -d .
   ```
   (If the zip contains `src/models/codebert_checkpoint/`, this puts them in the right place.)
4. Run Phase 2 **with resume**:
   ```python
   %cd /kaggle/working/code-reviewer-thesis
   !python src/phase2_train.py --epochs 3 --batch_size 16 --resume_from_checkpoint auto
   ```
5. After this run, **save checkpoints to the Dataset again** (Step 6b) so you don’t lose them if the session dies again.

Using `--epochs 1` or `--max_train 50000` can shorten the run for a quick test.
