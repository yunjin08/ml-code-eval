#!/usr/bin/env python3
"""
Dry-run script to verify Kaggle Dataset upload is set up correctly.
Run this on Kaggle (or anywhere with kaggle CLI) before a long Phase 2 run.
Creates a minimal test version and pushes to your checkpoint dataset.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Staging dir must be under /kaggle/working when on Kaggle
STAGING = "/kaggle/working/phase2_backup_dry_run"


def main():
    parser = argparse.ArgumentParser(description="Dry run: test Kaggle dataset upload setup")
    parser.add_argument(
        "dataset_slug",
        nargs="?",
        default="donairejededison/code-reviewer-thesis-phase2-checkpoints",
        help="Dataset slug, e.g. USERNAME/code-reviewer-thesis-phase2-checkpoints",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Only check environment and create staging; do not run kaggle datasets version",
    )
    args = parser.parse_args()
    dataset_slug = args.dataset_slug

    print("=== Kaggle backup dry run ===\n")

    # 1. Check we're on Kaggle
    if not os.path.isdir("/kaggle/working"):
        print("FAIL: /kaggle/working not found. Run this script on Kaggle (e.g. in a Kaggle notebook).")
        sys.exit(1)
    print("OK: Running on Kaggle (/kaggle/working exists)")

    # 2. Check kaggle CLI
    try:
        r = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            print("FAIL: kaggle CLI returned non-zero. Install with: pip install kaggle")
            sys.exit(1)
    except FileNotFoundError:
        print("FAIL: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    print("OK: kaggle CLI is available")

    # 3. Create staging with only files (no folders) so API accepts it
    if os.path.isdir(STAGING):
        import shutil
        shutil.rmtree(STAGING)
    os.makedirs(STAGING, exist_ok=True)

    meta = {
        "title": "Phase 2 CodeBERT checkpoints",
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(STAGING, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Single small file (no directories)
    dry_run_path = os.path.join(STAGING, "dry_run.txt")
    with open(dry_run_path, "w") as f:
        f.write(f"Dry run at {datetime.now(timezone.utc).isoformat()}\nDataset: {dataset_slug}\n")

    print(f"OK: Staging dir created at {STAGING} with dataset-metadata.json + dry_run.txt")

    if args.no_upload:
        print("\n--no-upload: skipping kaggle datasets version")
        print("To test upload, run without --no-upload")
        sys.exit(0)

    # 4. Push new version
    message = "dry run test"
    print(f"\nPushing new version to {dataset_slug} (message: {message})...")
    r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", STAGING, "-m", message],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        print("FAIL: kaggle datasets version failed")
        if r.stdout:
            print("stdout:", r.stdout)
        if r.stderr:
            print("stderr:", r.stderr)
        print("\nCommon fixes:")
        print("  - Create the dataset on Kaggle first (Datasets -> New dataset)")
        print("  - Use exact slug: USERNAME/dataset-name (e.g. donairejededison/code-reviewer-thesis-phase2-checkpoints)")
        print("  - On Kaggle notebook, credentials are usually already set")
        sys.exit(1)

    print("OK: New version pushed successfully")
    print("\n=== Dry run passed: upload setup is correct ===\n")


if __name__ == "__main__":
    main()
