"""A trading-day calendar built from an index's daily price series."""

from __future__ import annotations

import bisect
from datetime import date

import pandas as pd


class TradingCalendar:
    """Sorted set of trading days, with forward-snap and N-day-offset lookups."""

    def __init__(self, trading_days: pd.Series | list[date]):
        self._days: list[date] = sorted(set(trading_days))

    def __len__(self) -> int:
        return len(self._days)

    def snap_forward(self, d: date) -> date | None:
        """The first trading day >= d, or None if d is past the calendar's end."""
        i = bisect.bisect_left(self._days, d)
        if i >= len(self._days):
            return None
        return self._days[i]

    def offset(self, d: date, n: int) -> date | None:
        """The trading day n sessions after the trading day snapped from d.

        d itself does not need to be a trading day; it is snapped forward
        first. Returns None if the offset runs past the end of the calendar.
        """
        snapped = self.snap_forward(d)
        if snapped is None:
            return None
        i = bisect.bisect_left(self._days, snapped)
        j = i + n
        if j < 0 or j >= len(self._days):
            return None
        return self._days[j]

    def cross_check(self, other_dates: pd.Series) -> dict:
        """Diagnostic comparison against another date series (e.g. union of a stock CSV's dates)."""
        other = set(other_dates)
        calendar = set(self._days)
        return {
            "calendar_only": len(calendar - other),
            "other_only": len(other - calendar),
            "shared": len(calendar & other),
        }
