# 1. Data Gathering and Exploration

## 1.1 Data source

- **Dataset:** DiverseVul (Chen et al., 2023).  
  - Paper: *DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection* (RAID 2023).  
  - Official: [wagner-group/diversevul](https://github.com/wagner-group/diversevul) (metadata and Google Drive links).
- **Acquisition:** Loaded via **Hugging Face** from `bstee615/diversevul` to avoid manual download.  
  - Script: `src/phase1_dataset.py` → `datasets.load_dataset("bstee615/diversevul")`.  
  - The HF dataset provides the same DiverseVul content with pre-defined train / validation / test splits.

## 1.2 Curation steps (Phase 1)

1. **Load** the dataset from Hugging Face (`bstee615/diversevul`).
2. **Use existing splits:** train (264,393), validation (33,049), test (33,050). No custom split applied; the HF dataset already provides these.
3. **Filter by language:** The script supports filtering to C/C++ if a language column exists. The `bstee615/diversevul` version does not expose a language column; the dataset is C/C++-oriented, so all rows are kept.
4. **Standardize columns:**  
   - Input columns: `func` (code), `target` (0/1).  
   - Output columns: `id`, `code`, `label`, `split`.  
   - Global numeric `id` across splits for traceability.
5. **Drop missing/empty code:** Remove any row where `code` is null or empty (so the curated set has no missing data or outliers).  
6. **Write artifacts:**  
   - `data/curated_cpp.csv` – one row per function.  
   - `data/splits.json` – lists of `id`s for train, validation, and test.

**Command run:**

```bash
python src/phase1_dataset.py
```

## 1.3 Data exploration

### 1.3.1 Scope

- **Total samples:** 330,492.  
- **Splits:** train 264,393 | validation 33,049 | test 33,050.

### 1.3.2 Missing data

Phase 1 now **drops** any row with missing or empty `code` before writing the curated CSV. After re-running Phase 1:

| Column | Missing count |
|--------|----------------|
| `id`   | 0 |
| `label`| 0 |
| `split`| 0 |
| `code` | 0 |

**Conclusion:** No missing data in the curated set; the single row with missing `code` is removed by `drop_missing_code()` in `phase1_dataset.py`.

### 1.3.3 Class balance

- **Label 0 (non-vulnerable):** 311,547 (94.3%).  
- **Label 1 (vulnerable):** 18,945 (5.7%).  
- **Ratio:** ~16 : 1 (non-vulnerable : vulnerable).

**Per split:**

| Split      | Size    | Label 1 count | % vulnerable |
|------------|---------|--------------|--------------|
| Train      | 264,393 | 15,145       | 5.73%        |
| Validation | 33,049  | 1,869        | 5.66%        |
| Test       | 33,050  | 1,931        | 5.84%        |

**Conclusion:** Strong class imbalance; typical for vulnerability datasets. Model training uses class weighting (e.g. CodeBERT) and/or appropriate metrics (Precision, Recall, F1 for the positive “Block” class).

### 1.3.4 Ground truth

- **Label:** `label = 1` → critically defective (vulnerable) → should map to **Block**.  
- **Label:** `label = 0` → not critically defective → Approve or Review.  
- All evaluation (Phase 4) uses this definition of ground truth for the Block decision.

## 1.4 Artifacts produced

| Artifact | Path | Description |
|----------|------|-------------|
| Curated dataset | `data/curated_cpp.csv` | Columns: `id`, `code`, `label`, `split`. |
| Split indices | `data/splits.json` | `train_ids`, `validation_ids`, `test_ids`. |
| Semgrep rule set | `config/semgrep_rules.md` | Documented configs: `p/c`, `p/owasp-top-ten`, `p/cwe-top-25`. |

## 1.5 Iteration notes

- After full training runs, re-run the same exploration (e.g. missing counts, label distribution) on `data/curated_cpp.csv` if the curation script or data source changes.  
- If you add language or CWE filters later, document them here and update the exploration stats.
