"""Computes the date-range overlap between the transcripts and stock datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .dateparsing import TranscriptDateParser


@dataclass
class OverlapWindow:
    transcript_date_min: date
    transcript_date_max: date
    stock_date_min: date
    stock_date_max: date
    overlap_start: date
    overlap_end: date
    sp500_fetch_start: date
    sp500_fetch_end: date
    n_transcript_rows: int
    n_stock_rows: int
    generated_at: str

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(self).items()}
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "OverlapWindow":
        payload = json.loads(Path(path).read_text())
        date_fields = {
            "transcript_date_min", "transcript_date_max",
            "stock_date_min", "stock_date_max",
            "overlap_start", "overlap_end",
            "sp500_fetch_start", "sp500_fetch_end",
        }
        for field in date_fields:
            payload[field] = date.fromisoformat(payload[field])
        return cls(**payload)


class DatasetOverlapAnalyzer:
    """Loads the two raw sources and computes the date-range overlap between them."""

    def __init__(self, transcripts_path: str | Path, stock_path: str | Path):
        self._transcripts_path = Path(transcripts_path)
        self._stock_path = Path(stock_path)
        self._date_parser = TranscriptDateParser()

    def compute(self) -> OverlapWindow:
        transcripts = pd.read_pickle(self._transcripts_path)
        stock = pd.read_csv(self._stock_path)

        parsed_dates = self._date_parser.parse_series(transcripts["date"])
        transcript_dates = parsed_dates.loc[parsed_dates["call_time_parsed_ok"], "call_date"]

        stock_dates = pd.to_datetime(stock["Date"], utc=True).dt.date

        transcript_date_min = transcript_dates.min()
        transcript_date_max = transcript_dates.max()
        stock_date_min = stock_dates.min()
        stock_date_max = stock_dates.max()

        overlap_start = max(transcript_date_min, stock_date_min)
        overlap_end = min(transcript_date_max, stock_date_max)

        return OverlapWindow(
            transcript_date_min=transcript_date_min,
            transcript_date_max=transcript_date_max,
            stock_date_min=stock_date_min,
            stock_date_max=stock_date_max,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            # Fetch S&P data through the stock CSV's actual end (not overlap_end):
            # transcripts near the tail of the overlap window need trading days
            # *beyond* overlap_end to compute a forward-looking lag label, and
            # the stock CSV itself already has price data that far out.
            sp500_fetch_start=overlap_start,
            sp500_fetch_end=stock_date_max,
            n_transcript_rows=len(transcripts),
            n_stock_rows=len(stock),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
