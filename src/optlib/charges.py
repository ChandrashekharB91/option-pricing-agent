"""Transaction charges for NSE index options and futures (brokerage, STT, exchange, SEBI, stamp, GST).

Rates live in ``config/charges.json`` with their sources, so a budget or circular changes a file, not code.
All amounts are in rupees for the given quantity (units, i.e. lots x lot size).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

_CRORE = 1e7


@dataclass(frozen=True)
class Charges:
    brokerage_per_order: float = 20.0
    stt_option_sell_premium: float = 0.0015
    stt_option_exercise_intrinsic: float = 0.0015
    stt_future_sell: float = 0.0005
    exchange_option_premium: float = 0.0003503
    exchange_future_turnover: float = 0.000019
    sebi_per_crore: float = 10.0
    ipft_per_crore: float = 10.0
    stamp_option_buy: float = 0.00003
    stamp_future_buy: float = 0.00002
    gst: float = 0.18

    @classmethod
    def load(cls, path: Path) -> "Charges":
        raw = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in raw.items() if k in names})

    def order(self, instrument: str, side: int, price: float, quantity: float) -> dict:
        """Charges for one order: ``side`` +1 buy, -1 sell; ``price`` premium (options) or futures price."""
        value = abs(price * quantity)
        is_opt = instrument != "future"
        brokerage = self.brokerage_per_order if quantity else 0.0
        stt = value * ((self.stt_option_sell_premium if is_opt else self.stt_future_sell) if side < 0 else 0.0)
        exchange = value * (self.exchange_option_premium if is_opt else self.exchange_future_turnover)
        sebi = value * self.sebi_per_crore / _CRORE
        ipft = value * self.ipft_per_crore / _CRORE
        stamp = value * ((self.stamp_option_buy if is_opt else self.stamp_future_buy) if side > 0 else 0.0)
        gst = self.gst * (brokerage + exchange + sebi + ipft)
        total = brokerage + stt + exchange + sebi + ipft + stamp + gst
        return {"brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi + ipft, "stamp": stamp,
                "gst": gst, "total": total}

    def exercise_stt(self, intrinsic: float, quantity: float) -> float:
        """STT on long options that expire in the money (charged on the settlement value)."""
        return self.stt_option_exercise_intrinsic * max(intrinsic, 0.0) * abs(quantity)
