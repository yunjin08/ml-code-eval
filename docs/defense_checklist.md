# Thesis Defense Checklist
**Title:** A Hybrid ML and Policy-as-Code Approach for Software Governance: An Empirical Evaluation of Vulnerability Detection  
**Author:** Jed Edison J. Donaire  
**Panel Score:** 7.2/10 | **Pass Probability:** HIGH  

---

## Red Flags — Critical Issues

- [x] **Red Flag 1 — Empty ANOVA Tables (Tables 4.10 & 4.11)**  
  Tables 4.10 (Precision ANOVA) and 4.11 (Recall ANOVA) had `---` placeholders instead of actual values.  
  **Fix:** Populated with computed values from notebook (cells 45 & 48). Values match the interaction plots already in the manuscript.  
  _Status: FIXED in manuscript.tex_

- [x] **Red Flag 2 — Unexplained α and β Weights**  
  The hybrid formula `Hybrid Risk = α·C_ML + β·V_PaC` uses fixed weights (α=0.75, β=0.25) with no previously stated method for selection.  
  **Fix:** Expanded §3 Hybrid Governance Setup paragraph to document the exact grid search: α ∈ {0.00, 0.25, 0.50, 0.75, 1.00}, t_block ∈ {0.30–0.70}, t_review ∈ {0.10–0.30}, joint optimization of Block F1 on validation set. Sensitivity sweep (Figure pac_sensitivity, Table pac_gain_peaks) was already in manuscript.  
  _Status: FIXED in manuscript.tex_

- [x] **Red Flag 3 — Threshold Optimization Not Described**  
  The decision thresholds were stated but not justified.  
  **Fix:** Covered in the same Red Flag 2 edit — the paragraph now explicitly states thresholds were selected via grid search over t_block ∈ {0.30, 0.40, 0.50, 0.60, 0.70} and t_review ∈ {0.10, 0.20, 0.25, 0.30} jointly with α/β, maximizing validation Block F1.  
  _Status: FIXED in manuscript.tex_

- [x] **Red Flag 4 — No Real Codebase Validation**  
  All evaluation is on DiverseVul (a research dataset). No real-world repository tested.  
  **Already addressed:** §5 External Validity Considerations (lines ~1269–1279) explicitly covers: (a) within-distribution generalization only, (b) function-level evaluation limitations for PaC, (c) C/C++ scope, (d) temporal validity not assessed. Cites Chen et al. (2023) showing F1 drops from 38% → 11% under cross-project holdout.  
  **Defense prep:** Be ready to say — "Expanding to real-world repo validation is a prioritized next step; the within-distribution evaluation was a deliberate scope boundary for the thesis."  
  _Status: Already in manuscript — defend confidently_

- [x] **Red Flag 5 — Semgrep Rule Source Not Attributed**  
  The PaC rules were mentioned but not formally documented.  
  **Fix:** Added Table (tab:semgrep_rules) in §3 Policy-as-Code Rule Selection documenting all three rule sources (p/c, p/cwe-top-25, custom_rules.yaml), their purpose, and target CWEs. Semgrep version (≥1.45.0) and execution year (2024) noted in table footnote.  
  _Status: FIXED in manuscript.tex_

- [ ] **Red Flag 6 — No Computational Cost Analysis** _(Cannot fix without re-running experiments)_  
  No per-function timing data was collected during Phase 3. The phase3_full_run.log has no wall-clock timing per approach.  
  **What's already there:** §5 Practical Deployment Considerations (lines ~1285–1291) discusses GPU vs CPU latency qualitatively, scalability, and operational cost — but no measured numbers.  
  **Defense prep:** Acknowledge timing was not instrumented; note that CodeBERT on GPU runs ~50ms/function per published benchmarks, Semgrep is ~100ms/file; combined Hybrid would be ~150ms/function. Flag explicit timing benchmarking as future work.  
  _Status: Cannot fix in manuscript without new data — prepare verbal answer_

- [x] **Red Flag 7 — Dataset Imbalance Not Addressed**  
  Class imbalance (~5.8% vulnerable) was mentioned but handling was not specified.  
  **Fix:** Expanded §3 Model Fine-Tuning to specify the exact inverse-frequency weighting formula: w1 = n0/n1 ≈ 15.7 for vulnerable class, w0 = 1.0 for non-vulnerable, applied via `CrossEntropyLoss(weight=[1.0, 15.7])`. The class imbalance context (94%/6% split) was already in §3 Data Selection.  
  _Status: FIXED in manuscript.tex_

- [x] **Red Flag 8 — Generalizability Limited to C/C++**  
  Applicability beyond C/C++ was not discussed.  
  **Already addressed:** §5 External Validity line ~1277 — "specific weights, thresholds, and PaC rule sets derived here were calibrated to C/C++ vulnerability patterns and were not validated for other languages. Deployment in Python, Java, or JavaScript environments would require independent calibration." Also in Limitation 3 (~line 1303) covering the single-dataset, single-language scope.  
  **Defense prep:** Emphasize the architecture is language-agnostic in principle (both CodeBERT and Semgrep support multiple languages); only the calibration is language-specific.  
  _Status: Already in manuscript — defend confidently_

---

## Strengths (Defend These Confidently)

- [ ] Review and be ready to explain the statistical rigor:
  - [ ] McNemar's test for pairwise AUC comparison
  - [ ] Paired bootstrap CI (B=5000) for ΔAUC
  - [ ] 10-fold stratified cross-validation (avoids data leakage)
  - [ ] Shapiro–Wilk + Levene's test before ANOVA (assumption checks)
  - [ ] Two-Way ANOVA with interaction (Type II SS) for F1, Precision, Recall

- [ ] Be ready to explain the DiverseVul dataset choice:
  - [ ] 330k C/C++ functions from real GitHub repos
  - [ ] CVE-labeled — not synthetic vulnerabilities
  - [ ] Available via Hugging Face (`bstee615/diversevul`) — reproducible

- [ ] Be ready to explain the hybrid formula and its behavior:
  - [ ] Hybrid Risk = 0.7·C_ML + 0.3·V_PaC
  - [ ] Why ML gets higher weight (broader coverage, trained on patterns)
  - [ ] Why PaC contributes (deterministic, interpretable, CWE-specific)

- [ ] Be ready to explain why PaC-Only has high Block Precision but low Block Recall:
  - [ ] Semgrep rules are conservative — only fires when pattern matches exactly
  - [ ] High precision = when it fires, it's almost always right
  - [ ] Low recall = it misses vulnerabilities that don't match static patterns

---

## Anticipated Panel Questions — Prepare Answers

- [ ] "Why α=0.7, β=0.3? Have you tested other weight combinations?"  
  → Prepare: justification + sensitivity analysis (or acknowledge as limitation + future work)

- [ ] "Did you consider cross-validation leakage? How did you ensure the test set was truly held out?"  
  → Prepare: explain stratified 10-fold, no overlap between train/test per fold

- [ ] "How does your system handle zero-day vulnerabilities not covered by Semgrep rules?"  
  → Prepare: ML component provides probabilistic coverage; PaC complements with known-CWE precision

- [ ] "How would this system perform on a live CI/CD pipeline? What is the latency?"  
  → Prepare: acknowledge this was not measured; note it as future work

- [ ] "What does the interaction effect in the ANOVA actually tell us practically?"  
  → Prepare: approach effectiveness is decision-class-dependent — the Hybrid advantage is strongest on Block decisions where PaC raises precision without sacrificing ML recall

- [ ] "Why not use a more recent model than CodeBERT (e.g., CodeLlama, StarCoder)?"  
  → Prepare: CodeBERT was state-of-the-art at time of study; it's a 125M-param model feasible on limited hardware; comparison with newer models is future work

- [ ] "Your dataset is from 2022 — are the vulnerabilities still representative?"  
  → Prepare: CVEs are historical by nature; DiverseVul includes vulnerabilities through ~2022; newer CVEs not included is an acknowledged limitation

- [ ] "The Hybrid approach doesn't significantly outperform ML-Only on all metrics — is the added complexity justified?"  
  → Prepare: Hybrid shows statistically significant improvement on AUC (McNemar p<0.05) and the interaction effect shows specific gains on Block Precision — justify complexity for high-stakes governance contexts

---

## Manuscript Fixes — Revision Tracker

| # | Item | Location | Status |
|---|------|----------|--------|
| 1 | Fill Tables 4.10 & 4.11 with ANOVA values | §4.4, lines 1141–1176 | **DONE** |
| 2 | Document α/β weight grid search method | §3 Hybrid Governance Setup | **DONE** |
| 3 | Document threshold grid search method | §3 Hybrid Governance Setup (same edit) | **DONE** |
| 4 | External validity / real-codebase discussion | §5 External Validity (was already there) | **DONE** |
| 5 | Semgrep rules attribution table | §3 Policy-as-Code Rule Selection | **DONE** |
| 6 | Inference latency comparison | §5 Practical Deployment (qualitative only) | **N/A — no data** |
| 7 | Class imbalance handling with formula | §3 Model Fine-Tuning | **DONE** |
| 8 | C/C++ generalizability scope in limitations | §5 External Validity + Limitation 3 (was already there) | **DONE** |

---

## Why the Interaction Plots Appeared but Tables Were Empty

The figures (`interaction_plot_approach_precision.pdf`, `interaction_plot_approach_recall.pdf`) were generated automatically by notebook cells 46 and 49 via `plt.savefig()` — the files were written to `figures/` and LaTeX just included them with `\includegraphics`.

The ANOVA table values had **no equivalent auto-generation mechanism**. The notebook printed the values to stdout (cells 45 and 48), but nothing wrote them to a `.tex` file. The manuscript had manual placeholder `---` text with a note "Values filled after notebook execution" — this step was never completed before the PDF was submitted.

**Lesson for future work:** Use `statsmodels` or a custom script to write ANOVA results directly to a `.tex` input file, similar to how other constants are written to `tex/` in this project.
