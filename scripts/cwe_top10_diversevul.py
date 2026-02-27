"""
One-off script: Load DiverseVul from Hugging Face, inspect CWE column, report top 10 CWEs by frequency.
Requires: pip install datasets pandas
"""
import sys
from collections import Counter

def main():
    from datasets import load_dataset
    import pandas as pd

    import os
    # Use project-local cache to avoid sandbox permission issues
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = os.path.join(proj_root, "data", "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    print("Loading DiverseVul from Hugging Face (bstee615/diversevul)...")
    ds = load_dataset("bstee615/diversevul", cache_dir=cache_dir)
    print("Dataset features:", ds["train"].features)
    print()

    # Collect all CWE values across splits (only for vulnerable samples, label=1)
    cwe_counter = Counter()
    total_with_cwe = 0
    total_vuln = 0
    sample_cwe = None

    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        part = ds[split_name]
        for i, row in enumerate(part):
            target = row.get("target")
            if target != 1:
                continue
            total_vuln += 1
            cwe = row.get("cwe")
            if sample_cwe is None and cwe is not None:
                sample_cwe = (type(cwe), cwe)
            if cwe is None:
                continue
            total_with_cwe += 1
            # Handle list/sequence of CWEs per sample
            if isinstance(cwe, (list, tuple)):
                for c in cwe:
                    if c is not None and str(c).strip():
                        cwe_counter[str(c).strip()] += 1
            else:
                cwe_counter[str(cwe).strip()] += 1

    print("Sample CWE value (type, value):", sample_cwe)
    print("Total vulnerable samples (label=1):", total_vuln)
    print("Vulnerable samples with non-null CWE:", total_with_cwe)
    print()

    # Normalize CWE to "CWE-XXX" form if they're raw numbers
    def normalize_cwe(s):
        s = str(s).strip()
        if s.isdigit():
            return f"CWE-{s}"
        if s.upper().startswith("CWE"):
            return s.upper() if not s.startswith("CWE-") else "CWE-" + s[3:].lstrip("-")
        return f"CWE-{s}"

    normalized_counter = Counter()
    for cwe_val, count in cwe_counter.items():
        key = normalize_cwe(cwe_val)
        normalized_counter[key] += count

    top10 = normalized_counter.most_common(10)
    print("Top 10 CWEs by frequency (vulnerable samples only):")
    print("-" * 40)
    for rank, (cwe, count) in enumerate(top10, 1):
        print(f"  {rank:2}. {cwe}: {count}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
