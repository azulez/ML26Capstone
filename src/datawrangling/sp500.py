"""Fetches and archives S&P 500 (or another index) daily price data via yfinance."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


class SP500Fetcher:
    """Wraps a yfinance download + normalization for an index ticker."""

    def __init__(self, index_ticker: str = "^GSPC"):
        self._index_ticker = index_ticker

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        # yfinance's `end` is exclusive, so add a day to include the requested end date.
        raw = yf.download(
            self._index_ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {self._index_ticker} in [{start}, {end}]")

        # yfinance returns a MultiIndex column (field, ticker); flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df

    def fetch_and_archive(self, start: date, end: date, output_path: str | Path, force: bool = False) -> Path:
        output_path = Path(output_path)
        if output_path.exists() and not force:
            return output_path

        df = self.fetch(start, end)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path
