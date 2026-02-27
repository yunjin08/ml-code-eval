# Semgrep Rule Selection (Policy-as-Code)

Per thesis Section 3.2.3, the Semgrep policy set is based on OWASP Top 10 and CWE for C/C++.  
Updated to target top CWEs in the DiverseVul subset (CWE-787, 125, 119, 416, 476, 190, etc.).

## Config Used

```bash
semgrep scan --config "p/c" --config "p/cwe-top-25" --config "config/custom_rules.yaml" <path>
```

- **p/c** – Semgrep community C/C++ rules (includes dangerous functions, memory safety)
- **p/cwe-top-25** – CWE Top 25 (rules applicable to C/C++ are used)
- **config/custom_rules.yaml** – Custom rules for CWE-119/787/125 (dangerous functions), CWE-476 (NULL deref), CWE-416 (use-after-free), CWE-190 (integer overflow in alloc)

Rule counts and language applicability are determined at runtime by Semgrep. Document the exact rule count after first run in `results/semgrep_rule_summary.txt` if needed.

**Note:** The registry config `p/c-memory-safety` does not exist (404). We use `p/c` + `p/cwe-top-25` + custom rules to cover the top DiverseVul CWEs (119/787/125, 476, 416, 190, 415).

**Custom rules (expanded):** In addition to dangerous functions (strcpy, gets, etc.), NULL deref, use-after-free, and integer overflow in alloc, `config/custom_rules.yaml` includes: memcpy/memmove, strncpy/snprintf, realloc-deref without null check, double free, and malloc with addition/shift (CWE-190). Mini test on 400 samples: PaC>0 went from ~4% to ~11% (vulnerable 14%, safe 8%).
