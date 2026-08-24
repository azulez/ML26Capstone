"""CLI: merge transcripts with stock/S&P data and compute the excess-return label."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datawrangling.merge import TranscriptStockMerger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-path", default="data/sets/motley-fool-data.pkl")
    parser.add_argument("--stock-path", default="data/sets/stock_details_5_years.csv")
    parser.add_argument("--sp500-path", default="data/streamed/sp500_daily.csv")
    parser.add_argument("--lag-days", type=int, default=10)
    parser.add_argument("--output-path", default="data/generated/labeled_transcripts.parquet")
    args = parser.parse_args()

    result = TranscriptStockMerger().run(
        args.transcripts_path, args.stock_path, args.sp500_path, lag_days=args.lag_days,
    )
    result.to_parquet(args.output_path)

    print(f"Funnel: {result.funnel}")
    print(f"\nexcess_return_label distribution:\n{result.data['excess_return_label'].describe()}")
    print(f"\nWrote {len(result.data)} labeled rows to {args.output_path}")


if __name__ == "__main__":
    main()
