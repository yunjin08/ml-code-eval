# 2. Model Training

This document describes how we train and select the ML model used in the governance experiment (Phase 3). Two models are trained; the one with higher validation F1 is selected.

## 2.1 Data used

- **Training:** Rows in `data/curated_cpp.csv` whose `id` is in `data/splits.json` → `train_ids`.  
- **Validation:** Rows whose `id` is in `validation_ids`.  
- **Load logic:** Implemented in `src/phase2_train.py` via `load_curated()` (reads CSV + splits.json).  
- **Note:** For a clean run, drop the single row with missing `code` before training (see [01_data_gathering_and_exploration.md](01_data_gathering_and_exploration.md)).

## 2.2 Model 1: CodeBERT

### 2.2.1 Setup

- **Base model:** `microsoft/codebert-base` (Hugging Face).  
- **Task:** Binary sequence classification (0: non-vulnerable, 1: vulnerable).  
- **Input:** Raw function source in `code`; tokenized with max length 512, padding to max length.  
- **Output:** Probability of class 1 (vulnerable) → used as ML confidence score in Phase 3.

### 2.2.2 Training details

- **Loss:** Cross-entropy with **class weights** to account for imbalance (weight for class 1 ∝ ratio of non-vulnerable to vulnerable in training set).  
- **Optimizer:** AdamW (Trainer default).  
- **Hyperparameters (defaults):**  
  - Epochs: 3 (configurable via `--epochs`).  
  - Batch size: 16 (configurable via `--batch_size`).  
  - Evaluation and checkpointing every epoch; best checkpoint by validation **F1**.  
- **Checkpoints:** Saved under `src/models/codebert_checkpoint/` during training; best model and tokenizer saved to `src/models/codebert/`.

### 2.2.3 Optional caps (development only)

- `--max_train N`, `--max_val N`: Limit number of training/validation samples for quick runs.  
- **For thesis results:** Do not use these; use the full train/validation sets.

### 2.2.4 Apple Silicon (M1/M2/M3) and MPS OOM

- On Macs with Metal, PyTorch may use MPS (GPU). CodeBERT can hit **out-of-memory** (e.g. `kIOGPUCommandBufferCallbackErrorOutOfMemory`).  
- **Options:**  
  - **`--cpu`**: Force CPU for CodeBERT. Slower but avoids Metal OOM.  
  - **Smaller batch size:** e.g. `--batch_size 4` or `--batch_size 2` to reduce memory use if staying on MPS.  
- Example short run on M3:  
  `python src/phase2_train.py --max_train 10000 --max_val 2000 --epochs 1 --cpu`  
  or with GPU: `... --batch_size 4` (and omit `--cpu` if 4 fits in memory).

## 2.3 Model 2: Random Forest (baseline)

### 2.3.1 Features

- **Source:** Static code metrics per function, computed with **lizard** (C/C++).  
- **Features per sample:**  
  - Cyclomatic complexity (sum over functions in snippet)  
  - NLOC (lines of code)  
  - Token count  
  - Parameter count  
- **Implementation:** `src/utils/lizard_metrics.py` (writes code to temp file, runs lizard, aggregates). Missing or failed parses get fallback values (e.g. 0 complexity, line/token counts from the raw string).

### 2.3.2 Training details

- **Model:** `sklearn.ensemble.RandomForestClassifier` (n_estimators=100, max_depth=20, random_state=42).  
- **Data:** Same train/validation split as CodeBERT; feature matrix built from train, validated on validation.  
- **Saved artifact:** `src/models/rf.pkl` (joblib).

## 2.4 Model selection

- **Criterion:** Validation **F1** for the positive class (label = 1).  
- **Selection:** The model with higher validation F1 is chosen for Phase 3 (ML-Only and Hybrid).  
- **Record:** Selected model name (`"codebert"` or `"random_forest"`) is written to `results/phase2_validation_report.json` under `"selected_model"`.  
- Phase 3 reads this file to decide whether to load CodeBERT or the Random Forest.

## 2.5 Artifacts produced

| Artifact | Path | Description |
|----------|------|-------------|
| CodeBERT model + tokenizer | `src/models/codebert/` | Used when `selected_model` is `"codebert"`. |
| Random Forest pickle | `src/models/rf.pkl` | Used when `selected_model` is `"random_forest"`. |
| Validation report | `results/phase2_validation_report.json` | Validation metrics for both models and `selected_model`. |

## 2.6 Commands

**Full training (both models, full data):**

```bash
python src/phase2_train.py
```

**CodeBERT only (e.g. if RF already trained):**

```bash
python src/phase2_train.py --skip_rf
```

**Random Forest only (faster, for pipeline testing):**

```bash
python src/phase2_train.py --skip_codebert
```

**Quick dev run (not for final thesis):**

```bash
python src/phase2_train.py --max_train 5000 --max_val 1000
```

## 2.7 Iteration notes

- After full training, update this section with final hyperparameters (epochs, batch size, any tuning) and validation F1/Precision/Recall for both models.  
- If you add more features for the RF or change the CodeBERT architecture, document the changes here.
