#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shamdose.metadata import load_metadata, print_metadata_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cohort metadata and print included subject counts."
    )

    parser.add_argument(
        "--metadata",
        default=str(PROJECT_ROOT / "config" / "cohort_metadata.csv"),
        help="Path to private cohort_metadata.csv file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[INFO] Checking metadata file:")
    print(f"       {Path(args.metadata).expanduser().resolve()}")

    df = load_metadata(args.metadata)
    print("[OK] Metadata validation passed.")

    print_metadata_summary(df)


if __name__ == "__main__":
    main()
