"""
Phase 1: Dataset Curation
Load DiverseVul, filter to C/C++ if language column exists, produce train/validation/test
with columns: id, code, label, split. Save curated dataset and document Semgrep config.
"""

import os
import sys

import pandas as pd

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_diversevul():
    """Load DiverseVul from Hugging Face (bstee615/diversevul)."""
    from datasets import load_dataset

    print("Loading DiverseVul from Hugging Face (bstee615/diversevul)...")
    try:
        ds = load_dataset("bstee615/diversevul")
    except Exception as e:
        print(f"HF load failed: {e}. Trying claudios/DiverseVul as fallback...")
        try:
            ds = load_dataset("claudios/DiverseVul")
        except Exception as e2:
            raise RuntimeError(
                "Could not load DiverseVul. Ensure network access and try again."
            ) from e2
    return ds


def get_splits(ds):
    """
    Return train, validation, test DataFrames.
    bstee615/diversevul has train, validation, test. Convert to pandas.
    """
    if "train" in ds and "validation" in ds and "test" in ds:
        return (
            ds["train"].to_pandas(),
            ds["validation"].to_pandas(),
            ds["test"].to_pandas(),
        )
    if "train" in ds and "test" in ds:
        train_df = ds["train"].to_pandas()
        from sklearn.model_selection import train_test_split

        train_sub, val_sub = train_test_split(
            train_df, test_size=0.1, stratify=train_df["target"], random_state=42
        )
        return train_sub, val_sub, ds["test"].to_pandas()
    # Single split: assume 'train' only
    full = ds["train"].to_pandas()
    from sklearn.model_selection import train_test_split

    train_df, rest = train_test_split(
        full, test_size=0.2, stratify=full["target"], random_state=42
    )
    val_df, test_df = train_test_split(
        rest, test_size=0.5, stratify=rest["target"], random_state=42
    )
    return train_df, val_df, test_df


def standardize_columns(df, split_name):
    """Standardize to id, code, label, split."""
    # bstee615/diversevul: func, target
    rename = {"func": "code", "target": "label"}
    df = df.rename(columns=rename)
    if "code" not in df.columns:
        raise ValueError("Dataset must have 'func' or 'code' column.")
    if "label" not in df.columns:
        raise ValueError("Dataset must have 'target' or 'label' column.")
    df["id"] = range(len(df))  # simple numeric id per split
    df["split"] = split_name
    return df[["id", "code", "label", "split"]]


def drop_missing_code(df):
    """Drop rows with missing or empty code so we have no missing data / outliers."""
    if "code" not in df.columns:
        return df
    before = len(df)
    # drop null/NaN and empty string (or whitespace-only)
    df = df[df["code"].notna() & df["code"].astype(str).str.strip().str.len().gt(0)]
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} row(s) with missing or empty code.")
    return df.reset_index(drop=True)


def filter_c_cpp(df, lang_column="lang"):
    """Filter to C and C++ if language column exists."""
    if lang_column not in df.columns:
        return df
    if "language" in df.columns:
        lang_column = "language"
    if lang_column not in df.columns:
        return df
    allowed = {"c", "cpp", "c++", "c/c++"}
    mask = df[lang_column].astype(str).str.lower().str.strip().isin(allowed)
    return df[mask].copy()


def main():
    ds = load_diversevul()
    train_df, val_df, test_df = get_splits(ds)

    # Filter C/C++ if language column present (bstee615/diversevul has no lang column)
    if "lang" in train_df.columns or "language" in train_df.columns:
        train_df = filter_c_cpp(train_df)
        val_df = filter_c_cpp(val_df)
        test_df = filter_c_cpp(test_df)

    train_s = standardize_columns(train_df, "train")
    val_s = standardize_columns(val_df, "validation")
    test_s = standardize_columns(test_df, "test")

    # Remove rows with missing or empty code (no missing data / outliers)
    print("Dropping rows with missing or empty code...")
    train_s = drop_missing_code(train_s)
    val_s = drop_missing_code(val_s)
    test_s = drop_missing_code(test_s)

    # Global id across splits
    base = 0
    train_s["id"] = range(base, base + len(train_s))
    base += len(train_s)
    val_s["id"] = range(base, base + len(val_s))
    base += len(val_s)
    test_s["id"] = range(base, base + len(test_s))

    curated = pd.concat([train_s, val_s, test_s], ignore_index=True)

    out_path = os.path.join(DATA_DIR, "curated_cpp.csv")
    curated.to_csv(out_path, index=False)
    print(f"Curated dataset saved to {out_path}")
    print(f"  Train: {len(train_s)}, Validation: {len(val_s)}, Test: {len(test_s)}")
    print(f"  Label distribution (train): {train_s['label'].value_counts().to_dict()}")

    # Save split indices for reproducible loading
    splits_path = os.path.join(DATA_DIR, "splits.json")
    import json

    with open(splits_path, "w") as f:
        json.dump(
            {
                "train_ids": train_s["id"].tolist(),
                "validation_ids": val_s["id"].tolist(),
                "test_ids": test_s["id"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Splits saved to {splits_path}")
    return curated


if __name__ == "__main__":
    main()
