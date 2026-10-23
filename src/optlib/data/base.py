"""The interface every market data source implements.

Tools depend on this protocol only, so switching from Upstox to Bloomberg (or a
recorded snapshot in tests) is a configuration change, not a code change.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from .models import FutureQuote, OptionChain


@runtime_checkable
class MarketDataSource(Protocol):
    name: str

    def expiries(self, underlying: str) -> list[date]:
        """Listed option expiries for an underlying, nearest first."""

    def option_chain(self, underlying: str, expiry: date | str) -> OptionChain:
        """Full chain (all strikes, calls and puts) for one expiry."""

    def spot(self, underlying: str) -> float:
        """Current index level."""

    def futures(self, underlying: str) -> list[FutureQuote]:
        """Listed index futures, nearest expiry first."""
