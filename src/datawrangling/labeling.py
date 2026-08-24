"""Computes the market-relative (excess) return label for each transcript row."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta

import pandas as pd

from .trading_calendar import TradingCalendar

_MARKET_CLOSE_ET = time(16, 0)


@dataclass
class LabelResult:
    anchor_day: date
    lag_day: date
    price_day0: float
    price_dayN: float
    stock_return_pct: float
    sp500_close_day0: float
    sp500_close_dayN: float
    market_return_pct: float
    excess_return_label: float
    split_adjusted: bool


class RelativeReturnLabeler:
    """Computes anchor days and excess-return labels for (ticker, call date/time) rows.

    `stock_df` must have columns: Company (ticker), Date (date), Close, and
    "Stock Splits". `sp500_df` must have columns: Date (date), Close. The
    trading calendar is built from the fetched S&P data (a single continuous
    series with no per-ticker gaps) rather than any individual stock's own
    dates, so both legs of the excess-return subtraction are measured over
    identical calendar days.
    """

    def __init__(self, calendar: TradingCalendar, stock_df: pd.DataFrame, sp500_df: pd.DataFrame):
        self._calendar = calendar
        self._price_lookup: dict[tuple[str, date], float] = (
            stock_df.set_index(["Company", "Date"])["Close"].to_dict()
        )
        self._split_cumprod: dict[str, pd.Series] = {}
        for ticker, group in stock_df.sort_values("Date").groupby("Company"):
            ratios = group["Stock Splits"].where(group["Stock Splits"] != 0, 1.0)
            self._split_cumprod[ticker] = pd.Series(ratios.to_numpy().cumprod(), index=group["Date"].to_numpy())
        self._sp500_lookup: dict[date, float] = sp500_df.set_index("Date")["Close"].to_dict()
        self.split_affected_count = 0

    def anchor_day(self, call_date: date, call_time: time | None, parsed_ok: bool) -> date | None:
        """Day 0: same day if the call was before 4pm ET, else the next day; then
        snapped forward to the next actual trading day. An unparseable time
        defaults to after-hours, since most calls happen in the evening."""
        if parsed_ok and call_time is not None and call_time < _MARKET_CLOSE_ET:
            candidate = call_date
        else:
            candidate = call_date + timedelta(days=1)
        return self._calendar.snap_forward(candidate)

    def _split_factor(self, ticker: str, day0: date, dayN: date) -> float:
        cumprod = self._split_cumprod.get(ticker)
        if cumprod is None:
            return 1.0
        factor_dayN = cumprod.asof(dayN)
        factor_day0 = cumprod.asof(day0)
        if pd.isna(factor_dayN) or pd.isna(factor_day0) or factor_day0 == 0:
            return 1.0
        return float(factor_dayN / factor_day0)

    def label_row(self, ticker: str, anchor_day: date, lag_days: int) -> LabelResult | None:
        lag_day = self._calendar.offset(anchor_day, lag_days)
        if lag_day is None:
            return None

        price_day0 = self._price_lookup.get((ticker, anchor_day))
        price_dayN = self._price_lookup.get((ticker, lag_day))
        if price_day0 is None or price_dayN is None:
            return None

        sp500_day0 = self._sp500_lookup.get(anchor_day)
        sp500_dayN = self._sp500_lookup.get(lag_day)
        if sp500_day0 is None or sp500_dayN is None:
            return None

        split_factor = self._split_factor(ticker, anchor_day, lag_day)
        split_adjusted = split_factor != 1.0
        if split_adjusted:
            self.split_affected_count += 1
        price_day0_adjusted = price_day0 / split_factor

        stock_return_pct = (price_dayN - price_day0_adjusted) / price_day0_adjusted
        market_return_pct = (sp500_dayN - sp500_day0) / sp500_day0

        return LabelResult(
            anchor_day=anchor_day,
            lag_day=lag_day,
            price_day0=price_day0,
            price_dayN=price_dayN,
            stock_return_pct=stock_return_pct,
            sp500_close_day0=sp500_day0,
            sp500_close_dayN=sp500_dayN,
            market_return_pct=market_return_pct,
            excess_return_label=stock_return_pct - market_return_pct,
            split_adjusted=split_adjusted,
        )

    def label_frame(self, df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
        """Batched entry point: expects columns ticker, call_date, call_time, call_time_parsed_ok.
        Returns a DataFrame of label columns aligned to df's index; rows that
        can't be labeled (missing price/calendar data) are dropped."""
        results: dict[int, LabelResult] = {}
        for idx, row in df.iterrows():
            anchor = self.anchor_day(row["call_date"], row["call_time"], row["call_time_parsed_ok"])
            if anchor is None:
                continue
            result = self.label_row(row["ticker"], anchor, lag_days)
            if result is not None:
                results[idx] = result

        columns = [
            "anchor_day", "lag_day", "price_day0", "price_dayN", "stock_return_pct",
            "sp500_close_day0", "sp500_close_dayN", "market_return_pct",
            "excess_return_label", "split_adjusted",
        ]
        if not results:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            [vars(result) for result in results.values()],
            index=list(results.keys()),
            columns=columns,
        )
