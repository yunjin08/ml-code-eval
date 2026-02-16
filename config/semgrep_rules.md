# Semgrep Rule Selection (Policy-as-Code)

Per thesis Section 3.2.3, the Semgrep policy set is based on OWASP Top 10 and CWE for C/C++.

## Config Used

```bash
semgrep scan --config "p/c" --config "p/owasp-top-ten" --config "p/cwe-top-25" <path>
```

- **p/c** – Semgrep community C/C++ rules
- **p/owasp-top-ten** – OWASP Top 10 (rules applicable to C/C++ are used)
- **p/cwe-top-25** – CWE Top 25 (rules applicable to C/C++ are used)

Rule counts and language applicability are determined at runtime by Semgrep. Document the exact rule count after first run in `results/semgrep_rule_summary.txt` if needed.
