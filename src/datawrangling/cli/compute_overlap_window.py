"""CLI: compute the date-range overlap between the transcripts and stock datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datawrangling.overlap import DatasetOverlapAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-path", default="data/sets/motley-fool-data.pkl")
    parser.add_argument("--stock-path", default="data/sets/stock_details_5_years.csv")
    parser.add_argument("--output-path", default="data/generated/overlap_window.json")
    args = parser.parse_args()

    window = DatasetOverlapAnalyzer(args.transcripts_path, args.stock_path).compute()
    window.to_json(args.output_path)

    for field, value in vars(window).items():
        print(f"{field}: {value}")
    print(f"\nWrote overlap window to {args.output_path}")


if __name__ == "__main__":
    main()
