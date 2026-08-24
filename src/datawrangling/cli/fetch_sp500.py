"""CLI: fetch and archive S&P 500 daily data for the range needed by the merge step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datawrangling.overlap import OverlapWindow
from datawrangling.sp500 import SP500Fetcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-window-path", default="data/generated/overlap_window.json")
    parser.add_argument("--index-ticker", default="^GSPC")
    parser.add_argument("--output-path", default="data/streamed/sp500_daily.csv")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if output-path already exists")
    args = parser.parse_args()

    window = OverlapWindow.from_json(args.overlap_window_path)
    fetcher = SP500Fetcher(index_ticker=args.index_ticker)
    output_path = fetcher.fetch_and_archive(
        window.sp500_fetch_start, window.sp500_fetch_end, args.output_path, force=args.force,
    )

    import pandas as pd
    df = pd.read_csv(output_path)
    print(f"Archived {len(df)} rows to {output_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")


if __name__ == "__main__":
    main()
