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

- [ ] **Red Flag 2 — Unexplained α and β Weights**  
  The hybrid formula `Hybrid Risk = α·C_ML + β·V_PaC` uses fixed weights (α=0.7, β=0.3) with no principled justification or sensitivity analysis.  
  **Fix needed:** Add a paragraph in the methodology explaining how weights were chosen (grid search, domain rationale, or ablation). Consider adding a small sensitivity table showing how F1 changes at α={0.5, 0.6, 0.7, 0.8}.

- [ ] **Red Flag 3 — Threshold Optimization Not Described**  
  The decision threshold for ML classification is not clearly defined or justified.  
  **Fix needed:** State explicitly which threshold was used (e.g., 0.5 default, or Youden's J), and why. A ROC curve with the chosen operating point marked would strengthen this.

- [ ] **Red Flag 4 — No Real Codebase Validation**  
  All evaluation is on DiverseVul (a research dataset). No real-world repository (e.g., Linux kernel, OpenSSL) was tested.  
  **Fix needed:** Either run a small real-world pilot and report qualitative findings, or add a dedicated "Threats to Validity: External Validity" section addressing this gap explicitly.

- [ ] **Red Flag 5 — Semgrep Rule Source Not Attributed**  
  The PaC rules (`p/c`, `p/cwe-top-25`, custom CWE rules) are not fully documented — version, registry snapshot date, and custom rule count are absent.  
  **Fix needed:** Add a table listing each rule source, version/date accessed, and rule count. This is needed for reproducibility.

- [ ] **Red Flag 6 — No Computational Cost Analysis**  
  The thesis does not compare inference latency or computational overhead between ML-Only, PaC-Only, and Hybrid approaches.  
  **Fix needed:** Add a table with mean inference time per function for each approach. Even rough wall-clock numbers improve practical credibility.

- [ ] **Red Flag 7 — Dataset Imbalance Not Addressed**  
  DiverseVul is highly imbalanced (~5.8% vulnerable). No class weighting, oversampling, or threshold adjustment is discussed.  
  **Fix needed:** Acknowledge in methodology that class imbalance was present. Describe how (or whether) it was handled. Add a note on how this affects precision/recall interpretation.

- [ ] **Red Flag 8 — Generalizability Limited to C/C++**  
  All findings are C/C++ specific due to DiverseVul. Applicability to Python, Java, etc. is unaddressed.  
  **Fix needed:** Add to the Limitations section: scope is explicitly C/C++; language-specific vulnerability patterns mean results may not transfer without retraining.

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
| 2 | Add α/β weight justification + sensitivity | §3.x Methodology | Pending |
| 3 | State and justify decision threshold | §3.x or §4.x | Pending |
| 4 | Add external validity / real-codebase discussion | §5 Threats to Validity | Pending |
| 5 | Document Semgrep rules (version, date, count) | §3.x or Appendix | Pending |
| 6 | Add inference latency comparison table | §4.x or Appendix | Pending |
| 7 | Address class imbalance in methodology | §3.x | Pending |
| 8 | Expand limitations to address C/C++ scope | §5 Limitations | Pending |

---

## Why the Interaction Plots Appeared but Tables Were Empty

The figures (`interaction_plot_approach_precision.pdf`, `interaction_plot_approach_recall.pdf`) were generated automatically by notebook cells 46 and 49 via `plt.savefig()` — the files were written to `figures/` and LaTeX just included them with `\includegraphics`.

The ANOVA table values had **no equivalent auto-generation mechanism**. The notebook printed the values to stdout (cells 45 and 48), but nothing wrote them to a `.tex` file. The manuscript had manual placeholder `---` text with a note "Values filled after notebook execution" — this step was never completed before the PDF was submitted.

**Lesson for future work:** Use `statsmodels` or a custom script to write ANOVA results directly to a `.tex` input file, similar to how other constants are written to `tex/` in this project.
