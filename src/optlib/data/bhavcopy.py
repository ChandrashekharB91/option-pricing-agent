"""NSE F&O end-of-day "bhavcopy" (UDiFF format) for index options and futures.

Source: https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
(public exchange archive). Used for P&L explain on real consecutive trading days.

Prices used: ``ClsPric`` (NSE closing price) for options and futures, ``UndrlygPric`` for the index
close. Closing prices are not simultaneous to the second, so implied vols from them are slightly noisier
than from live mid quotes; P&L explain reports what is left unexplained.
"""
from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Optional

URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


@dataclass(frozen=True)
class BhavRow:
    trade_date: date
    symbol: str
    kind: str  # "option" | "future"
    expiry: date
    strike: float
    option_type: Optional[str]  # "call" | "put" | None for futures
    close: float
    settle: float
    underlying: float
    oi: float
    volume: float
    lot_size: int


def parse_rows(text: str, symbols: Iterable[str] = INDEX_SYMBOLS) -> list[BhavRow]:
    symbols = set(symbols)
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        if r["TckrSymb"] not in symbols or r["FinInstrmTp"] not in ("IDO", "IDF"):
            continue
        is_opt = r["FinInstrmTp"] == "IDO"
        out.append(BhavRow(
            trade_date=date.fromisoformat(r["TradDt"]), symbol=r["TckrSymb"],
            kind="option" if is_opt else "future", expiry=date.fromisoformat(r["XpryDt"]),
            strike=float(r["StrkPric"] or 0.0) if is_opt else 0.0,
            option_type={"CE": "call", "PE": "put"}.get(r["OptnTp"]) if is_opt else None,
            close=float(r["ClsPric"] or 0.0), settle=float(r["SttlmPric"] or 0.0),
            underlying=float(r["UndrlygPric"] or 0.0), oi=float(r["OpnIntrst"] or 0.0),
            volume=float(r["TtlTradgVol"] or 0.0), lot_size=int(float(r["NewBrdLotQty"] or 0))))
    return out


def _download(url: str) -> bytes:
    import requests

    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    if resp.status_code != 200:
        raise RuntimeError(f"bhavcopy: HTTP {resp.status_code} for {url} (holiday or not yet published?)")
    return resp.content


def load_bhavcopy(day: date, cache_dir: Path, symbols: Iterable[str] = INDEX_SYMBOLS,
                  fetch: Optional[Callable[[str], bytes]] = None) -> list[BhavRow]:
    """Index F&O rows for one trading day, from cache or NSE's archive."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"bhav_fo_{day:%Y%m%d}.csv"
    if not path.is_file():
        blob = (fetch or _download)(URL.format(d=day))
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode("utf-8")
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return parse_rows(path.read_text(encoding="utf-8"), symbols)


def load_symbol_rows(day: date, cache_dir: Path, symbol: str = "NIFTY",
                     fetch: Optional[Callable[[str], bytes]] = None) -> list[BhavRow]:
    """One symbol's index F&O rows for a day, cached as a small gzip file (for multi-year backtests)."""
    import gzip

    cache_dir = Path(cache_dir)
    path = cache_dir / f"bhav_{symbol.lower()}_{day:%Y%m%d}.csv.gz"
    if not path.is_file():
        blob = (fetch or _download)(URL.format(d=day))
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode("utf-8")
        lines = text.splitlines()
        keep = [lines[0]] + [l for l in lines[1:] if f",{symbol}," in l]
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write("\n".join(keep) + "\n")
        tmp.replace(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return parse_rows(f.read(), {symbol})
