"""Parsing for the Motley Fool transcript ``date`` field.

The raw field is inconsistent in two ways found by direct inspection of the
data: most rows are a single string like ``"Aug 27, 2020, 9:00 p.m. ET"``, but
~2% are a two-element list ``[title_and_ticker_string, date_string]`` where
the trailing date string itself sometimes drops the comma before the time
(``"Jan. 31, 2019 11:00 a.m. ET"``). A handful of single-string rows also have
the company title crammed directly in front of the date with no separator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, time

from dateutil import parser as dateutil_parser

import pandas as pd

_MONTH_ALIASES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)

# Month names spelled out explicitly (rather than a generic [A-Za-z]+) so the
# match doesn't greedily swallow an adjoining word when the date is crammed
# directly against a title string with no separator, e.g. "...CallApril 30, 2021".
_MONTH_ALTERNATION = "|".join(_MONTH_NAMES) + "|" + "|".join(m.capitalize() for m in _MONTH_ALIASES)
# full names listed first so the alternation doesn't need to backtrack out of
# a 3-letter abbreviation match (e.g. "Jun") before trying "June".
_DATE_PATTERN = re.compile(
    rf"(?P<month>{_MONTH_ALTERNATION})\.?\s+(?P<day>\d{{1,2}}),\s+(?P<year>\d{{4}}),?\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+(?P<ampm>[AaPp]\.?[Mm]\.?)\s+(?P<tz>[A-Za-z]{2,4})\.?$"
)


@dataclass(frozen=True)
class ParsedCallDatetime:
    call_date: date | None
    call_time: time | None
    parsed_ok: bool


class TranscriptDateParser:
    """Parses the messy transcript ``date`` field into a date + time-of-day."""

    def _extract_raw_string(self, raw: object) -> str:
        if isinstance(raw, list):
            return str(raw[-1]).strip()
        return str(raw).strip()

    def parse(self, raw: object) -> ParsedCallDatetime:
        text = self._extract_raw_string(raw)
        match = _DATE_PATTERN.search(text)
        if match is None:
            return self._parse_fallback(text)

        month_key = match.group("month")[:3].lower()
        month = _MONTH_ALIASES.get(month_key)
        if month is None:
            return self._parse_fallback(text)

        ampm = match.group("ampm").replace(".", "").upper()
        hour = int(match.group("hour"))
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0

        try:
            call_date = date(int(match.group("year")), month, int(match.group("day")))
            call_time = time(hour, int(match.group("minute")))
        except ValueError:
            return self._parse_fallback(text)

        return ParsedCallDatetime(call_date=call_date, call_time=call_time, parsed_ok=True)

    def _parse_fallback(self, text: str) -> ParsedCallDatetime:
        cleaned = text.replace("ET", "").strip()
        try:
            parsed = dateutil_parser.parse(cleaned, fuzzy=True)
        except (ValueError, OverflowError):
            return ParsedCallDatetime(call_date=None, call_time=None, parsed_ok=False)
        return ParsedCallDatetime(call_date=parsed.date(), call_time=parsed.time(), parsed_ok=True)

    def parse_series(self, raw: pd.Series) -> pd.DataFrame:
        parsed = raw.map(self.parse)
        return pd.DataFrame(
            {
                "call_date": parsed.map(lambda p: p.call_date),
                "call_time": parsed.map(lambda p: p.call_time),
                "call_time_parsed_ok": parsed.map(lambda p: p.parsed_ok),
            },
            index=raw.index,
        )
