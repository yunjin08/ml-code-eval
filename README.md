# Hybrid CodeBERT–Semgrep Governance for Vulnerability Detection

**A Machine Learning and Policy-as-Code Evaluation**
*UP Cebu Special Problem — Final Manuscript*

> **Author:** Jed Edison J. Donaire
> **Repository:** [github.com/yunjin08/ml-code-eval](https://github.com/yunjin08/ml-code-eval)
> **Live Demo:** [code-eval.jed-edison.com](https://code-eval.jed-edison.com/)

---

## Overview

This study empirically evaluates a **hybrid governance framework** for software vulnerability detection that combines a fine-tuned **CodeBERT** machine learning model with **Semgrep** Policy-as-Code (PaC) rules. The framework produces a three-tier governance decision — **Block**, **Review**, or **Pass** — for C/C++ code functions.

The experiment is conducted on the **DiverseVul** dataset (~330,000 labeled C/C++ functions from real-world GitHub projects), comparing three governance approaches:

| Approach | Description |
|---|---|
| **ML-Only** | CodeBERT fine-tuned for binary vulnerability classification |
| **PaC-Only** | Semgrep with `p/c`, `p/cwe-top-25`, and custom CWE rules |
| **Hybrid** | Weighted combination: risk score = 0.75 × ML + 0.25 × PaC |

---

## Repository Layout

```
ml-code-eval/
├── src/                    # Phase pipeline scripts
│   ├── phase1_dataset.py   # Dataset curation and train/val/test splits
│   ├── phase2_train.py     # CodeBERT fine-tuning + Random Forest baseline
│   ├── phase3_experiment.py# Governance experiment (ML-Only / PaC-Only / Hybrid)
│   ├── phase4_evaluation.py# Statistical analysis (precision, recall, F1, AUC)
│   ├── pac_sensitivity_sweep.py
│   └── utils/              # Semgrep runner, Lizard metrics helpers
├── config/
│   ├── custom_rules.yaml   # Custom Semgrep rules targeting top CWEs
│   └── semgrep_rules.md    # Rule documentation and rationale
├── api/
│   ├── main.py             # FastAPI backend for the live demo tool
│   └── requirements.txt
├── notebooks/
│   └── findings.ipynb      # Key findings and visualization notebook
├── figures/                # Generated figures (ROC/AUC, interaction plots)
├── tex/                    # LaTeX constants and tables (auto-generated)
├── docs/                   # Methodology execution guides
│   ├── 01_data_gathering_and_exploration.md
│   ├── 02_model_training.md
│   ├── 03_evaluation.md
│   └── pac_verification.md
├── manuscript.tex          # Full thesis manuscript (LaTeX)
├── references.bib          # BibTeX references
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.9+, Git, Semgrep CLI

```bash
git clone https://github.com/yunjin08/ml-code-eval.git
cd ml-code-eval

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install Semgrep separately (required for PaC phases):

```bash
pip install semgrep
# or: brew install semgrep
```

---

## Reproducing the Experiment

The pipeline runs in four sequential phases:

### Phase 1 — Dataset Curation

Downloads DiverseVul from Hugging Face, filters to C/C++, and produces deterministic train/validation/test splits.

```bash
python src/phase1_dataset.py
```

Output: `data/splits.json`, curated CSVs (gitignored; re-run to regenerate).

### Phase 2 — Model Training

Fine-tunes CodeBERT and trains a Random Forest baseline. GPU is recommended for CodeBERT.

```bash
python src/phase2_train.py
# Flags: --skip_codebert  --skip_rf  --max_train N  --max_val N
```

Output: `src/models/codebert/` checkpoint (gitignored).

### Phase 3 — Governance Experiment

Runs all three governance approaches on the test set and records decisions and scores.

```bash
python src/phase3_experiment.py
# Flags: --max_test N  (Semgrep is slow; limit for fast iteration)
```

Output: `results/phase3_experiment_results.csv` (gitignored).

### Phase 4 — Statistical Evaluation

Computes precision, recall, F1, AUC, McNemar's test, odds ratios, and bootstrap confidence intervals.

```bash
python src/phase4_evaluation.py
```

Output: `results/phase4_evaluation_report.json`, figures in `figures/`.

---

## Key Results

All metrics are computed on the held-out test split (33,050 C/C++ functions from DiverseVul).

| Approach | Precision (Block) | Recall (Block) | F1 (Block) | ROC-AUC |
|---|---|---|---|---|
| ML-Only (CodeBERT) | 0.1272 | 0.3459 | 0.1860 | 0.7044 |
| PaC-Only (Semgrep) | 0.2059 | 0.0326 | 0.0563 | 0.5515 |
| **Hybrid** | **0.1272** | **0.3459** | **0.1860** | **0.7086** |

**Hypothesis outcomes:**
- **H1** (Hybrid F1 > ML-Only and PaC-Only) — *Partially accepted*: Hybrid F1 > PaC-only; Hybrid F1 = ML-only (no significant difference, McNemar p ≈ 0.157)
- **H2** (Hybrid Recall > PaC-Only) — *Accepted*: 0.346 vs 0.033, McNemar p < 0.0001
- **H3** (Hybrid Precision > ML-Only) — *Rejected*: both ≈ 0.127, McNemar p ≈ 0.157

The Hybrid's marginal AUC gain (+0.004) over ML-Only reflects that the continuous risk score benefits from PaC signal, even when the fixed Block threshold produces identical binary decisions. See `notebooks/findings.ipynb` for the full statistical analysis.

---

## Live Demo

A web-based testing tool is deployed at **[code-eval.jed-edison.com](https://code-eval.jed-edison.com/)** where you can paste any C/C++ snippet and receive a real-time governance decision (Block / Review / Pass) along with the ML confidence score, PaC flag count, and hybrid risk score.

The API source is in `api/main.py` (FastAPI). To run locally:

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

---

## Semgrep Custom Rules

The custom rule set (`config/custom_rules.yaml`) targets the top CWEs found in DiverseVul:

| CWE | Description | Pattern Examples |
|---|---|---|
| CWE-787 | Out-of-bounds Write | `strcpy`, `sprintf`, `memcpy` |
| CWE-125 | Out-of-bounds Read | Array access, pointer arithmetic |
| CWE-119 | Improper Memory Bounds | `gets`, unsafe string functions |
| CWE-416 | Use After Free | Double-free, post-free dereference |
| CWE-476 | NULL Pointer Dereference | `realloc` without NULL check |
| CWE-190 | Integer Overflow | Overflow in allocation size |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/01_data_gathering_and_exploration.md`](docs/01_data_gathering_and_exploration.md) | Dataset download, filtering, split rationale |
| [`docs/02_model_training.md`](docs/02_model_training.md) | CodeBERT fine-tuning details and hyperparameters |
| [`docs/03_evaluation.md`](docs/03_evaluation.md) | Evaluation metrics, statistical tests, results |
| [`docs/pac_verification.md`](docs/pac_verification.md) | Semgrep setup verification and rule coverage rationale |

---

## Citation

If you use this code or findings in your work, please cite the manuscript:

```
Donaire, J. E. J. (2025). Hybrid CodeBERT–Semgrep Governance for Vulnerability Detection:
A Machine Learning and Policy-as-Code Evaluation. Special Problem, University of the Philippines Cebu.
```

---

## License

This repository is maintained for academic purposes. Contact the author for reuse permissions.
