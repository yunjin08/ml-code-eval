# Thesis: Findings, Context, Trials, and Results

This document gathers the full context of the hybrid code-reviewer thesis: what was done, what was corrected, trials and realizations, and the final results and reasoning.

---

## 1. Overview and context

### 1.1 Goal

Empirical evaluation of **hybrid vulnerability detection** combining:

- **ML (CodeBERT):** Fine-tuned on DiverseVul for binary classification (vulnerable vs not). Outputs a confidence score in [0,1].
- **PaC (Semgrep):** Policy-as-Code rules (C/C++) producing a normalized finding-based score.
- **Hybrid:** Fusion of the two with tuned weights (α·norm(ML) + β·norm(PaC)) and thresholds → **Approve / Review / Block**.

Dataset: **DiverseVul** (C/C++ function-level snippets), with train / validation / test splits from Hugging Face (`bstee615/diversevul`).

### 1.2 Four-phase pipeline

| Phase | Purpose | Main outputs |
|-------|---------|--------------|
| **Phase 1** | Dataset curation | `data/curated_cpp.csv`, `data/splits.json` (train 264k, val 33k, test 33k) |
| **Phase 2** | Model training & selection | CodeBERT + Random Forest; best by validation F1 → CodeBERT |
| **Phase 3** | Governance experiment | ML-only, PaC-only, Hybrid on test set; `phase3_experiment_results.csv`, `phase3_hybrid_config.json` |
| **Phase 4** | Evaluation | Precision/Recall/F1, McNemar, ROC-AUC, hypotheses H1–H3 |

Ground truth for “Block”: `label == 1` (vulnerable). Class balance is highly imbalanced (~94% non-vulnerable, ~6% vulnerable).

---

## 2. What we did (methodology and implementation)

### 2.1 Data (Phase 1)

- Loaded DiverseVul from Hugging Face; used provided splits.
- Standardized columns to `id`, `code`, `label`, `split`.
- **Dropped rows with missing or empty `code`** so the curated set has no missing data.
- Confirmed CWE presence in the dataset (e.g. top CWEs: 787, 125, 119, 416, 476, 190) for aligning PaC rules later.

### 2.2 Model training (Phase 2)

- **CodeBERT:** `microsoft/codebert-base`, fine-tuned for binary classification; max length 512; class weights for imbalance; best checkpoint by validation F1; saved to `src/models/codebert/`.
- **Random Forest:** Lizard-based static metrics (complexity, NLOC, tokens, params); trained on same train set.
- **Selection:** Higher validation F1 → CodeBERT selected (F1 ≈ 0.189 vs RF ≈ 0.024).

### 2.3 Semgrep (PaC) setup and expansion

**Initial setup (trial):**

- Semgrep config: **`p/c` only**.
- **Finding:** On the full test set, **PaC score was 0 for all 33,050 samples** — no rule matches.
- **Verification:** A known-bad snippet (e.g. `gets()`) was scanned; it correctly produced ≥ 1 finding. So the **tool and wrapper were correct**; the issue was **rule coverage**, not a broken config.

**Correction — expanded Semgrep:**

- **Configs:** `p/c` + **`p/cwe-top-25`** + **custom rules** (`config/custom_rules.yaml`).
- **Custom rules** target top DiverseVul-relevant CWEs:
  - CWE-119/787/125: dangerous functions (`strcpy`, `gets`, `sprintf`), `memcpy`/`memmove`, `strncpy`/`snprintf`.
  - CWE-476: NULL deref, `realloc`-then-deref without null check.
  - CWE-416: use-after-free, double-free patterns.
  - CWE-190: integer overflow in allocation (e.g. `n * elem` in `malloc`).
- **Result:** PaC now yields **non-zero findings on a fraction of the test set** (e.g. ~11% of samples with PaC > 0 in verification), so the hybrid can use both channels.

### 2.4 Hybrid tuning (Phase 3) and min_pac_weight

- **Score normalization:** ML and PaC scores are min-max normalized to [0,1] using the **validation** set so both channels are on a comparable scale.
- **Risk:** `hybrid_risk = α·norm(ML) + β·norm(PaC)` with α + β = 1.
- **Thresholds:** `t_block`, `t_review` tuned on validation (e.g. Block if risk ≥ 0.4, Review if ≥ 0.1).
- **Trial without constraint:** Grid search over α sometimes chose **β = 0** (ML-only), which is not desirable for a “hybrid” or for governance where policy should contribute.
- **Correction — `--min_pac_weight`:** Added a lower bound on β (e.g. `--min_pac_weight 0.2`). Tuning only considers α such that β ≥ 0.2, so the chosen config is a **true hybrid** (e.g. α = 0.75, β = 0.25). Final config: `alpha=0.75`, `beta=0.25`, `t_block=0.4`, `t_review=0.1`, plus validation min/max for normalization.

### 2.5 Verification before trusting PaC

- **Mandatory check:** `python src/verify_pac_setup.py` (or Phase 3’s built-in verification) runs known-bad snippets through the same Semgrep wrapper used in the experiment.
- **Interpretation:** If verification **passes** and PaC is still 0 on the dataset, we attribute that to **rule coverage**, not to a broken setup. After expanding rules, PaC is non-zero on a subset of test samples.

---

## 3. Trials, corrections, and realizations

### 3.1 PaC = 0 on full test set (p/c only)

- **Trial:** Run Phase 3 with Semgrep `p/c` only.
- **Result:** PaC score = 0 for every test sample.
- **Realization:** DiverseVul function-level C/C++ snippets do not match the patterns in `p/c` alone; this is a **rule-coverage gap**, not an implementation bug (verified by gets()-style check).
- **Correction:** Add `p/cwe-top-25` and custom CWE-focused rules; re-run Phase 3. PaC then contributes non-zero signal on a subset of samples.

### 3.2 Hybrid collapsing to ML-only (β = 0)

- **Trial:** Tune α, β on validation without constraints.
- **Result:** Optimizer often chose β = 0 (pure ML).
- **Realization:** When PaC has low or sparse signal, F1 is maximized by ignoring PaC; that undermines the thesis’s “hybrid” and governance story.
- **Correction:** Introduce `--min_pac_weight` (e.g. 0.2); re-run Phase 3 so β ≥ 0.2. Final config has β = 0.25.

### 3.3 Live demo: ML as heuristic vs real CodeBERT

- **Context:** The showcase has a “Live Analysis” section that calls an API to analyze pasted or uploaded code. The API runs Semgrep (PaC) but cannot run CodeBERT in a typical serverless/small deployment (model size, GPU).
- **Trial:** Use a **heuristic** for the “ML” score in the API: e.g. `ml_heuristic = min(0.95, 0.12 + pac_score * 0.4)` so the UI can still show a hybrid risk and decision.
- **Realization:** “ML” in the live demo is a placeholder when CodeBERT is not available; only PaC is “live” there unless the API runs with real ML.
- **Correction (local):** When running the API **locally**, set `USE_REAL_ML=1` and place the trained CodeBERT at `src/models/codebert/`. The API then loads CodeBERT and uses real ML scores for hybrid; response includes `ml_from_codebert: true/false` and a note. Deployed (e.g. serverless) continues to use the heuristic only.

### 3.4 Deployment: Railway removed; Netlify build failure

- **Trial:** Deploy API to Railway; deploy static showcase to Netlify.
- **Result (Netlify):** Build tried to install full repo dependencies (including `torch`, `transformers`, CUDA wheels) and failed with **“No space left on device”**.
- **Realization:** Netlify was building from repo root and saw `requirements.txt` (or API requirements), triggering a heavy Python install.
- **Correction:** Netlify should only see the static showcase. In `netlify.toml`: set **`base = "showcase"`** and **`publish = "."`** so the build context is only the `showcase/` folder. No Python, no pip; only HTML/CSS/JS are deployed. Railway was removed; live analysis is **local-only** (or GPU-backed server) with a note in the UI.

### 3.5 Folder / file drop in Live Analysis

- **Trial:** Allow users to drop a **folder** of C/C++ files in the upload zone.
- **Result:** Using `entry.file()` from the File System Access API (e.g. for dropped files) caused **EncodingError** (“Data URL has exceeded the URL length limitations”) in some browsers, and **0 files** were collected.
- **Realization:** The browser’s `entry.file()` path can throw for certain drops; we should avoid it for top-level file items.
- **Correction:** For **file** entries (not directories), use **`item.getAsFile()`** instead of walking into `collectFilesFromEntry` and calling `entry.file()`. Only use the entry API for **directories** (folder walk). Added try/catch and logging (upload log + console) so users can see “Loaded N file(s)” and debug. Fallback to `dataTransfer.files` when the entry path fails.

### 3.6 Mobile and UX

- **Trial:** Use the showcase on phones/tablets.
- **Correction:** Added responsive CSS (e.g. breakpoints at 768px and 480px): reduced padding, stacked comparison cards and result grid, wrapped pipeline steps and config strip, touch-friendly button height (min 44px), API URL row stacked on small screens. Summary table uses `data-label` and stacked layout on very small screens. Pipeline panel and status bar already worked; ensured the result grid uses a class (e.g. `live-result-grid`) so it stacks to one column on mobile.

### 3.7 Default API URL and pipeline feedback

- **Trial:** User had to type the API URL each time.
- **Correction:** Default API URL set to **`http://localhost:8000`** (and stored in localStorage when changed). When user clicks “Analyze”, show a **pipeline flow** (Send → PaC → ML → Hybrid → Done) and a **status bar** that advances during the request so the user sees progress.

---

## 4. Final results (Phase 4)

### 4.1 Primary metrics (Block decision)

| Policy    | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| ML-Only   | 0.1272    | 0.3459 | 0.1860 |
| PaC-Only  | 0.2059    | 0.0326 | 0.0563 |
| Hybrid    | 0.1272    | 0.3459 | 0.1860 |

(Values from `results/phase4_evaluation_report.json`.)

### 4.2 ROC-AUC

- ML-Only: **0.7044**
- PaC-Only: **0.5515**
- Hybrid: **0.7086**

Hybrid has a slight AUC gain over ML-only; PaC-only has much lower discriminative power on this test set.

### 4.3 McNemar (Hybrid vs others)

- **Hybrid vs ML-Only:** p ≈ 0.157 — not statistically significant; Hybrid and ML Block decisions are very similar (only 2 discordant pairs where one is correct and the other is not).
- **Hybrid vs PaC-Only:** p = 0.0000 — significant; Hybrid is clearly different from (and better than) PaC in terms of Block correctness. Odds ratio (Hybrid correct vs PaC correct) ≈ 6.13 [95% CI 5.67–6.63].

### 4.4 Formal hypotheses: accepted, rejected, partially accepted

| Hypothesis | Statement | Result | Verdict |
|------------|-----------|--------|--------|
| **H1** | The Hybrid approach will achieve a **higher F1** than either ML-only or PaC-only for Block decisions. | Hybrid F1 ≈ 0.186, ML F1 ≈ 0.186 (same), PaC F1 ≈ 0.056. McNemar Hybrid vs ML: p ≈ 0.16 (no significant difference). | **Partially accepted:** Hybrid F1 > PaC-only ✓; Hybrid F1 is **not** higher than ML-only (they match). So H1 holds vs PaC but not vs ML. |
| **H2** | The Hybrid approach will demonstrate **higher Recall** than PaC-only, flagging defects detectable by ML but not covered by static rules. | Hybrid recall 0.346 vs PaC 0.033; McNemar p = 0.0000. | **Accepted.** |
| **H3** | The Hybrid approach will demonstrate **higher Precision** than ML-only, by reducing false positives through the contextual filter of static analysis. | Hybrid and ML precision both ≈ 0.127; McNemar p ≈ 0.16 (no significant difference). | **Rejected.** |

**Summary:**

- **Accepted:** H2 (Hybrid recall > PaC-only).
- **Rejected:** H3 (Hybrid precision is not higher than ML-only; static analysis did not improve precision over ML on this test set).
- **Partially accepted:** H1 (Hybrid F1 is higher than PaC-only but not higher than ML-only; Hybrid matches ML on F1).

### 4.5 Reasoning and findings

- **Hybrid ≈ ML on this test set:** With α = 0.75 and β = 0.25, and PaC having low recall (0.033), the hybrid risk is dominated by the ML term for most samples. So Block/Review/Approve decisions align closely with ML-only, and F1/Precision are effectively the same. The small AUC improvement (0.7086 vs 0.7044) suggests a marginal gain when using the continuous risk score rather than a single threshold.
- **Hybrid >> PaC:** Hybrid recall and AUC are much higher than PaC-only; McNemar strongly supports that Hybrid is better than PaC for correct Block decisions. So the hybrid preserves ML’s recall while keeping the option to inject policy (PaC) where it fires.
- **PaC role:** PaC adds high-precision, low-recall signal. With min_pac_weight, the system is forced to combine both channels; in deployments where PaC rules are strengthened or data matches rules better, the hybrid could show larger gains over ML-only.

### 4.6 Conclusion: Hybrid vs ML

- **On the Block decision (binary):** Hybrid is **not better than ML**. Precision, Recall, and F1 are effectively the same (McNemar p ≈ 0.16 for Hybrid vs ML), so we do **not** conclude that the hybrid outperforms ML on this test set for the tuned Block/Review/Approve thresholds.
- **The slight difference we observed:** **ROC-AUC** — Hybrid **0.7086** vs ML-Only **0.7044**. That is a small but real improvement in how well the **continuous** risk score ranks vulnerable vs non-vulnerable samples (across all possible thresholds). So:
  - For a **single fixed threshold** (our tuned Block decision): hybrid and ML behave the same.
  - For **ranking** or **alternative thresholds**: the hybrid risk score has marginally better discrimination (AUC +0.004). In practice this means the hybrid score could be slightly more useful if you later change thresholds or use the score for prioritization rather than a single Block/Approve cut.
- **Summary:** We do not claim “hybrid is better than ML” on the primary decision metrics; we do report the **slight AUC gain** as a positive signal that adding PaC did not hurt and may add a small amount of information when using the score as a continuous signal.

### 4.7 Main finding and recommendation

- **Main finding:** ML and Hybrid are **strongly connected** on this dataset: because PaC has low recall, the hybrid risk is dominated by the ML term (α = 0.75), so Hybrid ≈ ML. The **lever that would make Hybrid more powerful** is a **more powerful PaC** — better rules, broader coverage, or data where static rules fire on a larger fraction of vulnerable samples.
- **Recommendation / future work:** Investigate setups where PaC has **comparable signal to ML** — e.g. stronger or more targeted Semgrep rules, or a dataset where rules match more often — so that tuning yields something closer to **50–50 (or more) PaC weight** instead of 25% PaC. Under those conditions we would expect:
  - **Higher AUC** for Hybrid (and possibly better F1) if PaC adds discriminative signal.
  - A real chance to **accept the alternative hypotheses** we currently reject: **H1** (Hybrid F1 > ML) and **H3** (Hybrid precision > ML) could be supported when PaC is strong enough to improve over ML-only. So the recommendation is to **strengthen PaC** (or choose contexts where PaC is strong) and re-run the experiment to see if Hybrid then shows a clear gain and the hypotheses are accepted.

---

## 5. Artifacts and references

### 5.1 Key files

| Artifact | Path |
|----------|------|
| Phase 1 splits | `data/splits.json` |
| Phase 2 validation & selected model | `results/phase2_validation_report.json` |
| Phase 3 config (α, β, thresholds, norm min/max) | `results/phase3_hybrid_config.json` |
| Phase 3 per-sample results | `results/phase3_experiment_results.csv` (or `phase3_results/`) |
| Phase 4 evaluation | `results/phase4_evaluation_report.json`, `results/phase4_results_summary.txt` |
| Semgrep custom rules | `config/custom_rules.yaml` |
| Methodology docs | `docs/01_data_gathering_and_exploration.md`, `02_model_training.md`, `03_evaluation.md` |
| PaC verification | `docs/pac_verification.md`; `src/verify_pac_setup.py` |
| Findings notebook | `notebooks/findings.ipynb` |

### 5.2 Showcase and API (summary)

- **Showcase:** Static single-page app (`showcase/index.html`): pipeline diagram, tuned config, code scenarios (S1–S5), summary table, and **Live Analysis** (paste code / upload files, API URL, pipeline + status bar, result panel). Deployed with **Netlify** with `base = "showcase"` so only static assets are built.
- **API:** FastAPI (`api/main.py`): `POST /analyze` (paste), `POST /analyze/upload` (files), `GET /health`. Runs Semgrep with `p/c`, `p/cwe-top-25`, and custom rules; returns findings, pac_score, ml score (heuristic or real CodeBERT if `USE_REAL_ML=1` locally), hybrid_risk, and decisions. **Local only** (or GPU server); no Railway. Default API URL in the UI is `http://localhost:8000`; note in UI that live analysis with real ML works only on local or GPU-backed deployment.
- **“Should block” repo:** A separate repo (`thesis-should-block-repo`) with example C/C++ files (buffer overflow, use-after-free, null deref, integer overflow, dangerous functions) for testing the live analysis; users can drop files or paste code to see Block/Review/Approve.

---

## 6. Summary table: trials and corrections

| Issue | Trial / observation | Correction / realization |
|-------|----------------------|---------------------------|
| PaC = 0 on all test samples | Semgrep `p/c` only | Rule-coverage gap. Expanded to `p/cwe-top-25` + custom CWE rules; PaC non-zero on a subset. |
| Hybrid tuning picks β = 0 | Unconstrained grid search | Added `--min_pac_weight 0.2` so β ≥ 0.2; true hybrid (α=0.75, β=0.25). |
| Live demo “ML” not real | API cannot run CodeBERT in cloud | Heuristic ML in API; optional real CodeBERT when `USE_REAL_ML=1` and model present locally. |
| Netlify build OOM | Build from repo root installed torch/transformers | `base = "showcase"`, `publish = "."` so Netlify only sees static site. |
| Folder/file drop → 0 files | `entry.file()` used for dropped files | Use `item.getAsFile()` for files; entry API only for directories; added logging. |
| Mobile usability | Desktop-only layout | Responsive CSS, stacked layout, touch targets, default API URL, pipeline + status bar. |

This document is the single place for context, methodology, trials, corrections, realizations, and final results for the thesis and the showcase/API work.
