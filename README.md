# Hybrid ML and Policy-as-Code Approach for Software Governance

Empirical evaluation of vulnerability detection using a hybrid model combining fine-tuned CodeBERT and Semgrep (Policy-as-Code) on the DiverseVul C/C++ dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Methodology (Four Phases)

1. **Phase 1 – Dataset Curation:** Load DiverseVul, filter C/C++, produce train/validation/test with `id`, `code`, `label`, `split`. Semgrep rule set: `p/c`, `p/owasp-top-ten`, `p/cwe-top-25`.
2. **Phase 2 – Model Training:** Fine-tune CodeBERT and train Random Forest baseline; benchmark on validation set and select ML model for Phase 3.
3. **Phase 3 – Controlled Experiment:** Run ML-Only, PaC-Only, and Hybrid governance on the test set; output scores and Block/Review/Approve decisions.
4. **Phase 4 – Evaluation:** Precision/Recall/F1 for Block, McNemar's test, odds ratio, ROC/AUC, and hypothesis validation (H1–H3).

## Usage

```bash
# Phase 1: Curate dataset (requires network for HF download)
python src/phase1_dataset.py

# Phase 2: Train CodeBERT and Random Forest (GPU recommended for CodeBERT)
python src/phase2_train.py
# Optional: --skip_codebert (RF only), --skip_rf (CodeBERT only), --max_train N, --max_val N

# Phase 3: Run governance experiment on test set
python src/phase3_experiment.py
# Optional: --max_test N, --max_val N (for faster runs; Semgrep is slow per sample)

# Phase 4: Evaluation and statistical analysis (reads results/phase3_experiment_results.csv)
python src/phase4_evaluation.py
```

## Repository Layout

- `config/` – Semgrep and experiment config
- `data/` – Cached dataset and curated CSVs
- `docs/` – **Methodology execution docs:** [data gathering & exploration](docs/01_data_gathering_and_exploration.md), [model training](docs/02_model_training.md), [evaluation](docs/03_evaluation.md)
- `src/` – Phase scripts and utils
- `results/` – Metrics, figures, and report
