# PaC (Semgrep) verification before trusting results

PaC results are **not trustworthy** until Semgrep is confirmed to be working. This project uses a mandatory check-in step so that a PaC=0 outcome can be attributed to **rule coverage** (rules don’t match the dataset), not to a broken setup.

---

## 1. Run verification (required before Phase 3 / reporting)

From the project root:

```bash
python src/verify_pac_setup.py
```

- **Exit 0:** Semgrep detected the known-bad pattern(s). PaC setup is trusted; you can run Phase 3 and interpret PaC=0 on DiverseVul as a rule-coverage finding.
- **Exit 1:** At least one required check failed. Do **not** trust PaC results; fix Semgrep (path, exit-code handling, config) and re-run until this passes.

---

## 2. What the verification does

- Runs **known-bad C snippets** (e.g. `gets()`) through the same Semgrep wrapper used in Phase 3 (`utils.semgrep_runner.run_semgrep_on_code`).
- **Requires** that at least one such snippet gets at least one finding (e.g. `gets()` → count ≥ 1).
- If that fails, the script exits with code 1 and prints what was expected vs obtained.

So: if verification **passes**, Semgrep and our wrapper are working; if PaC is still 0 on your dataset, it is because the rules do not match the data, not because of a configuration bug.

---

## 3. Phase 3 integration

Phase 3 runs this verification **by default** before computing PaC scores:

```bash
python src/phase3_experiment.py --workers 8
```

If verification fails, Phase 3 exits with code 1 and does not run the experiment.

To skip verification (only if you have already run `verify_pac_setup.py` and it passed):

```bash
python src/phase3_experiment.py --no_verify_pac --workers 8
```

---

## 4. Checklist before reporting PaC results

| Step | Action |
|------|--------|
| 1 | Run `python src/verify_pac_setup.py` and ensure **exit 0**. |
| 2 | Run Phase 3 (with or without `--no_verify_pac`; if you skip, step 1 must have passed recently). |
| 3 | If PaC=0 on the full run, state in the thesis that verification passed and PaC=0 is due to **rule coverage**, not setup error. |

This gives a clear, repeatable way to show that PaC results are trustworthy (or that the setup was broken and has been fixed).
