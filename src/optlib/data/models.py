"""Vendor-neutral market data records.

Every data source (Upstox now, NSE bhavcopy and Bloomberg later) returns these types,
so the pricing tools never see vendor formats.

Conventions:
- Prices in index points (rupees per unit); missing or zero bid/ask -> NaN.
- Timestamps are timezone-aware, in IST.
- NSE index options stop trading at 15:30 IST on expiry day; time to expiry is
  measured to that instant on Actual/365 Fixed.
- ``vendor_iv`` / ``vendor_delta`` are what the vendor reports, kept only for comparison.
  Pricing never uses them.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30), name="IST")
NSE_CLOSE = time(15, 30)
_SECONDS_PER_YEAR = 365.0 * 24 * 3600


def nse_expiry_instant(expiry: date) -> datetime:
    """Moment an NSE index option stops trading: 15:30 IST on its expiry date."""
    return datetime.combine(expiry, NSE_CLOSE, tzinfo=IST)


def year_fraction(start: datetime, expiry: date) -> float:
    """Actual/365 Fixed year fraction from ``start`` to the 15:30 IST expiry close (>= 0)."""
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    seconds = (nse_expiry_instant(expiry) - start).total_seconds()
    return max(seconds, 0.0) / _SECONDS_PER_YEAR


@dataclass(frozen=True)
class OptionQuote:
    instrument_key: str
    expiry: date
    strike: float
    option_type: str  # "call" | "put"
    bid: float = math.nan
    ask: float = math.nan
    bid_qty: int = 0
    ask_qty: int = 0
    last: float = math.nan
    prev_close: float = math.nan
    volume: int = 0
    oi: float = 0.0
    prev_oi: float = 0.0
    vendor_iv: float = math.nan
    vendor_delta: float = math.nan

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask) if self.two_sided else math.nan

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.two_sided else math.nan

    @property
    def two_sided(self) -> bool:
        return not (math.isnan(self.bid) or math.isnan(self.ask)) and self.ask >= self.bid


@dataclass(frozen=True)
class OptionChain:
    underlying: str
    underlying_key: str
    expiry: date
    spot: float
    as_of: datetime
    source: str
    quotes: tuple[OptionQuote, ...] = field(default_factory=tuple)

    @property
    def strikes(self) -> list[float]:
        return sorted({q.strike for q in self.quotes})

    def calls(self) -> list[OptionQuote]:
        return sorted((q for q in self.quotes if q.option_type == "call"), key=lambda q: q.strike)

    def puts(self) -> list[OptionQuote]:
        return sorted((q for q in self.quotes if q.option_type == "put"), key=lambda q: q.strike)

    def quote(self, strike: float, option_type: str) -> Optional[OptionQuote]:
        for q in self.quotes:
            if q.strike == strike and q.option_type == option_type:
                return q
        return None

    def time_to_expiry(self, at: Optional[datetime] = None) -> float:
        """Years (Act/365F) from ``at`` (default: snapshot time) to the 15:30 IST expiry close."""
        return year_fraction(at or self.as_of, self.expiry)

    def to_records(self) -> list[dict]:
        """Flat rows, one per option, ready for pandas or Snowflake."""
        base = {"underlying": self.underlying, "spot": self.spot, "as_of": self.as_of.isoformat(),
                "source": self.source}
        return [{**base, **asdict(q), "expiry": q.expiry.isoformat()} for q in
                sorted(self.quotes, key=lambda q: (q.strike, q.option_type))]

    def to_frame(self):
        """pandas DataFrame of ``to_records()`` (pandas is optional)."""
        import pandas as pd

        return pd.DataFrame(self.to_records())


@dataclass(frozen=True)
class FutureQuote:
    instrument_key: str
    underlying: str
    expiry: date
    last: float
    bid: float = math.nan
    ask: float = math.nan
    oi: float = 0.0
    volume: int = 0
    as_of: Optional[datetime] = None

    @property
    def mid(self) -> float:
        if math.isnan(self.bid) or math.isnan(self.ask):
            return math.nan
        return 0.5 * (self.bid + self.ask)
