"""Build synthetic Upstox responses in the documented v2 format.

The chain is priced with Black-76 on a known forward and smile, so tests can check
that parsing preserves prices exactly and that implied vols come back as built.
Run from the repo root:  python tests/fixtures/build_upstox_fixtures.py
"""
import json
import math
from datetime import datetime
from pathlib import Path

from optlib import black76_price
from optlib.data.models import IST, year_fraction

HERE = Path(__file__).parent
AS_OF = datetime(2026, 10, 23, 11, 0, tzinfo=IST)
EXPIRY = "2026-10-27"
SPOT = 25712.35
RATE = 0.065
CARRY = 0.052  # r - q for the synthetic index
STRIKES = [25200.0 + 50.0 * i for i in range(21)]  # 25200 .. 26200
TICK = 0.05


def smile(k_over_f: float) -> float:
    x = math.log(k_over_f)
    return 0.118 - 0.35 * x + 2.2 * x * x


def tick(x: float) -> float:
    return round(round(x / TICK) * TICK, 2)


def side(key: str, mid: float, vol: float, delta: float, deep_otm: bool) -> dict:
    half = max(0.05, 0.0025 * mid)
    bid, ask = (0.0, tick(mid + half)) if deep_otm else (tick(mid - half), tick(mid + half))
    return {
        "instrument_key": key,
        "market_data": {"ltp": tick(mid), "close_price": tick(mid * 1.04), "volume": 125000, "oi": 845000.0,
                        "bid_price": bid, "bid_qty": 0 if deep_otm else 650, "ask_price": ask, "ask_qty": 975,
                        "prev_oi": 812000.0},
        "option_greeks": {"vega": 9.8, "theta": -14.2, "gamma": 0.0009, "delta": round(delta, 4),
                          "iv": round(vol * 100, 2), "pop": 41.5},
    }


def build():
    T = year_fraction(AS_OF, datetime.fromisoformat(EXPIRY).date())
    fwd = SPOT * math.exp(CARRY * T)
    rows = []
    for i, k in enumerate(STRIKES):
        vol = smile(k / fwd)
        call = black76_price(fwd, k, T, RATE, vol, "call")
        put = black76_price(fwd, k, T, RATE, vol, "put")
        d1 = (math.log(fwd / k) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
        nd1 = 0.5 * math.erfc(-d1 / math.sqrt(2))
        row = {
            "expiry": EXPIRY, "pcr": 1.12, "strike_price": k, "underlying_key": "NSE_INDEX|Nifty 50",
            "underlying_spot_price": SPOT,
            "call_options": side(f"NSE_FO|{50000 + i}", call, vol, nd1, deep_otm=(k == STRIKES[-1])),
            "put_options": side(f"NSE_FO|{51000 + i}", put, vol, nd1 - 1, deep_otm=(k == STRIKES[0])),
        }
        rows.append(row)
    # one strike with no put listed, as Upstox returns for partially listed strikes
    rows[10]["put_options"] = {}
    chain = {"status": "success", "data": rows}

    quotes = {"status": "success", "data": {
        "NSE_INDEX:Nifty 50": {
            "ohlc": {"open": 25650.1, "high": 25748.9, "low": 25611.0, "close": 25640.55},
            "depth": {"buy": [], "sell": []}, "timestamp": "2026-10-23T11:00:00.125+05:30",
            "instrument_token": "NSE_INDEX|Nifty 50", "symbol": "NIFTY", "last_price": SPOT, "volume": 0,
            "average_price": 0.0, "oi": 0, "net_change": 71.8, "total_buy_quantity": 0, "total_sell_quantity": 0,
            "lower_circuit_limit": 0, "upper_circuit_limit": 0, "last_trade_time": "1792738800125",
            "oi_day_high": 0, "oi_day_low": 0},
        "NSE_FO:NIFTY26OCTFUT": {
            "ohlc": {"open": 25700.0, "high": 25790.0, "low": 25660.0, "close": 25689.9},
            "depth": {"buy": [{"quantity": 650, "price": 25745.1, "orders": 4}],
                      "sell": [{"quantity": 325, "price": 25746.5, "orders": 2}]},
            "timestamp": "2026-10-23T11:00:00.311+05:30", "instrument_token": "NSE_FO|48704",
            "symbol": "NIFTY26OCTFUT", "last_price": 25745.8, "volume": 1834500, "average_price": 25721.4,
            "oi": 14250000.0, "net_change": 55.9, "total_buy_quantity": 410000, "total_sell_quantity": 398000,
            "lower_circuit_limit": 23120.9, "upper_circuit_limit": 28258.8, "last_trade_time": "1792738800311",
            "oi_day_high": 14400000, "oi_day_low": 14100000},
    }}

    error_401 = {"status": "error", "errors": [{
        "error_code": "UDAPI100050", "message": "Invalid token used to access API",
        "property_path": None, "invalid_value": None}]}
    error_400 = {"status": "error", "errors": [{
        "error_code": "UDAPI100036", "message": "Invalid Input", "property_path": "expiry_date",
        "invalid_value": "2026-13-01"}]}

    meta = {"as_of": AS_OF.isoformat(), "expiry": EXPIRY, "spot": SPOT, "rate": RATE, "carry": CARRY,
            "forward": fwd, "time_to_expiry": T}
    for name, obj in [("upstox_option_chain.json", chain), ("upstox_quotes.json", quotes),
                      ("upstox_error_401.json", error_401), ("upstox_error_400.json", error_400),
                      ("upstox_chain_meta.json", meta)]:
        (HERE / name).write_text(json.dumps(obj, indent=1) + "\n")


if __name__ == "__main__":
    build()
