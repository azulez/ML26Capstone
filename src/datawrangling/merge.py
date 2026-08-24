"""Orchestrates the full transcript + stock-performance merge/labeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .dateparsing import TranscriptDateParser
from .labeling import RelativeReturnLabeler
from .trading_calendar import TradingCalendar

_OUTPUT_COLUMNS = [
    "ticker", "exchange", "q", "call_datetime_raw", "call_date", "call_time_parsed_ok",
    "anchor_day", "lag_days", "lag_day", "price_day0", "price_dayN", "stock_return_pct",
    "sp500_close_day0", "sp500_close_dayN", "market_return_pct", "excess_return_label",
    "transcript",
]


@dataclass
class FunnelStats:
    raw_rows: int
    deduped_rows: int
    date_parsed_rows: int
    ticker_matched_rows: int
    labeled_rows: int
    split_affected_rows: int

    def __str__(self) -> str:
        return (
            f"raw={self.raw_rows} -> deduped={self.deduped_rows} "
            f"-> date_parsed={self.date_parsed_rows} -> ticker_matched={self.ticker_matched_rows} "
            f"-> labeled={self.labeled_rows} (split_affected={self.split_affected_rows})"
        )


@dataclass
class MergeResult:
    data: pd.DataFrame
    funnel: FunnelStats

    def to_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(path, engine="pyarrow", index=False)


class TranscriptStockMerger:
    """Loads transcripts + stock + S&P data, dedupes, and computes the excess-return label."""

    def __init__(self):
        self._date_parser = TranscriptDateParser()
        self.last_raw_row_count = 0

    def load_transcripts(self, path: str | Path) -> pd.DataFrame:
        transcripts = pd.read_pickle(path)
        self.last_raw_row_count = len(transcripts)
        # `date` is occasionally a list (unhashable), so drop_duplicates can't
        # operate on the raw column directly; dedupe against a stringified copy instead.
        is_duplicate = transcripts.assign(date=transcripts["date"].map(str)).duplicated()
        return transcripts[~is_duplicate]

    def _load_stock(self, path: str | Path) -> pd.DataFrame:
        stock = pd.read_csv(path)
        stock["Date"] = pd.to_datetime(stock["Date"], utc=True).dt.date
        return stock

    def _load_sp500(self, path: str | Path) -> pd.DataFrame:
        sp500 = pd.read_csv(path)
        sp500["Date"] = pd.to_datetime(sp500["Date"]).dt.date
        return sp500

    def run(
        self,
        transcripts_path: str | Path,
        stock_path: str | Path,
        sp500_path: str | Path,
        lag_days: int = 10,
    ) -> MergeResult:
        transcripts = self.load_transcripts(transcripts_path)
        raw_rows = self.last_raw_row_count
        deduped_rows = len(transcripts)

        stock_df = self._load_stock(stock_path)
        sp500_df = self._load_sp500(sp500_path)

        parsed = self._date_parser.parse_series(transcripts["date"])
        transcripts = transcripts.join(parsed)
        transcripts["call_datetime_raw"] = transcripts["date"].map(str)
        transcripts = transcripts[transcripts["call_date"].notna()].copy()
        date_parsed_rows = len(transcripts)

        known_tickers = set(stock_df["Company"].unique())
        transcripts = transcripts[transcripts["ticker"].isin(known_tickers)].copy()
        ticker_matched_rows = len(transcripts)

        calendar = TradingCalendar(sp500_df["Date"])
        labeler = RelativeReturnLabeler(calendar, stock_df, sp500_df)
        label_columns = labeler.label_frame(transcripts, lag_days)

        merged = transcripts.join(label_columns, how="inner")
        merged["lag_days"] = lag_days

        funnel = FunnelStats(
            raw_rows=raw_rows,
            deduped_rows=deduped_rows,
            date_parsed_rows=date_parsed_rows,
            ticker_matched_rows=ticker_matched_rows,
            labeled_rows=len(merged),
            split_affected_rows=int(merged["split_adjusted"].sum()) if len(merged) else 0,
        )

        return MergeResult(data=merged[_OUTPUT_COLUMNS].reset_index(drop=True), funnel=funnel)
