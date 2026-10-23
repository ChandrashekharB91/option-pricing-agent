"""Upstox market data adapter (read-only, Analytics Token).

Uses three Upstox v2 endpoints plus the public instrument master:
    GET /v2/option/chain          full chain for one expiry
    GET /v2/market-quote/quotes   index spot and futures quotes (<= 500 keys per call)
    GET /v2/option/contract       listed contracts (fallback for expiries)
    assets.upstox.com .../NSE.json.gz   daily instrument master: expiries, futures keys, lot sizes

The Analytics Token is read-only (no orders), valid one year, and needs no static IP
for these endpoints. It is sent only to api.upstox.com, never to the public assets host.
"""
from __future__ import annotations

import gzip
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .env import get_secret, mask
from .models import IST, FutureQuote, OptionChain, OptionQuote

API_BASE = "https://api.upstox.com"
INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
TOKEN_ENV_VAR = "UPSTOX_ANALYTICS_TOKEN"

# Exchange symbol -> Upstox instrument key (verified against the NSE instrument master).
UNDERLYINGS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
    "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
    "NIFTYNXT50": "NSE_INDEX|Nifty Next 50",
}
_KEY_TO_SYMBOL = {v: k for k, v in UNDERLYINGS.items()}
_MAX_QUOTE_KEYS = 500
_RETRY_STATUS = {429, 500, 502, 503, 504}


# ------------------------------------------------------------------ errors

class UpstoxError(Exception):
    """Any failure talking to Upstox."""


class UpstoxAuthError(UpstoxError):
    """Token missing, invalid or expired (HTTP 401 / UDAPI100050)."""


class UpstoxAPIError(UpstoxError):
    def __init__(self, status: int, errors: list[dict], url: str):
        self.status = status
        self.errors = errors
        self.url = url
        detail = "; ".join(f"{e.get('error_code') or e.get('errorCode')}: {e.get('message')}" for e in errors) or "no detail"
        super().__init__(f"Upstox HTTP {status} for {url}: {detail}")

    @property
    def codes(self) -> list[str]:
        return [str(e.get("error_code") or e.get("errorCode")) for e in self.errors]


# ------------------------------------------------------------------ HTTP client

class UpstoxClient:
    """Thin authenticated GET client with retries on 429/5xx and network errors."""

    def __init__(self, token: str, *, session=None, timeout: float = 10.0, max_retries: int = 3,
                 backoff: float = 1.0, sleep: Callable[[float], None] = time.sleep, base_url: str = API_BASE):
        if not token:
            raise UpstoxAuthError("empty Upstox token")
        if session is None:
            import requests

            session = requests.Session()
        self._token = token
        self._session = session
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls, search_dirs: Optional[Iterable[Path]] = None, **kwargs) -> "UpstoxClient":
        return cls(get_secret(TOKEN_ENV_VAR, search_dirs), **kwargs)

    def __repr__(self) -> str:
        return f"UpstoxClient(token={mask(self._token)}, base_url={self.base_url!r})"

    def _headers(self) -> dict:
        return {"Accept": "application/json", "Authorization": f"Bearer {self._token}"}

    def get(self, path: str, params: dict) -> Any:
        """GET ``path`` and return the ``data`` field of a successful response."""
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
            except Exception as exc:  # network failure: requests.ConnectionError, Timeout, ...
                if attempt > self.max_retries:
                    raise UpstoxError(f"network error calling {url}: {type(exc).__name__}") from exc
                self._sleep(self.backoff * 2 ** (attempt - 1))
                continue

            status = resp.status_code
            if status in _RETRY_STATUS and attempt <= self.max_retries:
                self._sleep(self._retry_delay(resp, attempt))
                continue

            body = self._json(resp)
            if status == 401 or "UDAPI100050" in _error_codes(body):
                raise UpstoxAuthError(
                    "Upstox rejected the token (invalid or expired). Generate a new Analytics Token "
                    f"in Upstox Developer Apps and update {TOKEN_ENV_VAR} in .env.")
            if status >= 400 or not isinstance(body, dict) or body.get("status") != "success":
                errors = body.get("errors", []) if isinstance(body, dict) else []
                raise UpstoxAPIError(status, errors, url)
            return body.get("data")

    def _retry_delay(self, resp, attempt: int) -> float:
        retry_after = getattr(resp, "headers", {}).get("Retry-After")
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            return self.backoff * 2 ** (attempt - 1)

    @staticmethod
    def _json(resp) -> Any:
        try:
            return resp.json()
        except ValueError:
            return None

    # endpoints -----------------------------------------------------------
    def option_chain(self, instrument_key: str, expiry: str) -> list[dict]:
        return self.get("/v2/option/chain", {"instrument_key": instrument_key, "expiry_date": expiry}) or []

    def option_contracts(self, instrument_key: str, expiry: Optional[str] = None) -> list[dict]:
        params = {"instrument_key": instrument_key}
        if expiry:
            params["expiry_date"] = expiry
        return self.get("/v2/option/contract", params) or []

    def quotes(self, instrument_keys: list[str]) -> dict[str, dict]:
        """Full market quotes keyed by instrument key (pipe form), batched by 500."""
        out: dict[str, dict] = {}
        for i in range(0, len(instrument_keys), _MAX_QUOTE_KEYS):
            batch = instrument_keys[i:i + _MAX_QUOTE_KEYS]
            data = self.get("/v2/market-quote/quotes", {"instrument_key": ",".join(batch)}) or {}
            for colon_key, q in data.items():
                # response keys use "SEG:SYMBOL"; instrument_token carries the pipe key we asked for
                out[q.get("instrument_token") or colon_key] = q
        return out


def _error_codes(body) -> list[str]:
    if not isinstance(body, dict):
        return []
    return [str(e.get("error_code") or e.get("errorCode")) for e in body.get("errors") or []]


# ------------------------------------------------------------------ instrument master

@dataclass(frozen=True)
class Contract:
    instrument_key: str
    underlying: str
    instrument_type: str  # FUT | CE | PE
    expiry: date
    strike: float
    lot_size: int
    tick_size: float  # rupees (Upstox publishes paise)
    weekly: bool


def _expiry_from_epoch_ms(ms) -> date:
    # Upstox stamps expiry as 23:59:59 IST on the expiry date
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=IST).date()


def _parse_expiry(value) -> date:
    if isinstance(value, (int, float)):
        return _expiry_from_epoch_ms(value)
    return date.fromisoformat(str(value)[:10])


class InstrumentMaster:
    """NSE F&O contracts for the supported index underlyings, from Upstox's daily file."""

    def __init__(self, contracts: list[Contract]):
        self.contracts = contracts

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "InstrumentMaster":
        contracts = []
        for r in records:
            if r.get("segment") != "NSE_FO" or r.get("underlying_key") not in _KEY_TO_SYMBOL:
                continue
            if r.get("instrument_type") not in ("FUT", "CE", "PE"):
                continue
            contracts.append(Contract(
                instrument_key=r["instrument_key"],
                underlying=_KEY_TO_SYMBOL[r["underlying_key"]],
                instrument_type=r["instrument_type"],
                expiry=_parse_expiry(r["expiry"]),
                strike=float(r.get("strike_price") or 0.0),
                lot_size=int(r.get("lot_size") or 0),
                tick_size=float(r.get("tick_size") or 0.0) / 100.0,
                weekly=bool(r.get("weekly")),
            ))
        return cls(contracts)

    @classmethod
    def load(cls, cache_dir: Path, *, url: str = INSTRUMENT_MASTER_URL, today: Optional[date] = None,
             fetch: Optional[Callable[[str], bytes]] = None) -> "InstrumentMaster":
        """Load today's file from ``cache_dir`` or download it (public URL, no token sent)."""
        today = today or datetime.now(IST).date()
        cache_dir = Path(cache_dir)
        path = cache_dir / f"upstox_NSE_{today.isoformat()}.json.gz"
        if not path.is_file():
            cache_dir.mkdir(parents=True, exist_ok=True)
            payload = (fetch or _download)(url)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(path)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls.from_records(json.load(fh))

    def expiries(self, underlying: str) -> list[date]:
        return sorted({c.expiry for c in self.contracts if c.underlying == underlying and c.instrument_type != "FUT"})

    def futures(self, underlying: str) -> list[Contract]:
        return sorted((c for c in self.contracts if c.underlying == underlying and c.instrument_type == "FUT"),
                      key=lambda c: c.expiry)

    def lot_size(self, underlying: str) -> int:
        sizes = {c.lot_size for c in self.contracts if c.underlying == underlying}
        if not sizes:
            raise KeyError(f"no contracts for {underlying}")
        return max(sizes)


def _download(url: str) -> bytes:
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


# ------------------------------------------------------------------ data source

def _num(x) -> float:
    return math.nan if x is None else float(x)


def _parse_time(value) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        return datetime.fromtimestamp(int(value) / 1000.0, tz=IST)
    ts = datetime.fromisoformat(str(value))
    return ts if ts.tzinfo else ts.replace(tzinfo=IST)


def _option_quote(side: dict, expiry: date, strike: float, option_type: str) -> Optional[OptionQuote]:
    if not side or not side.get("instrument_key"):
        return None
    md = side.get("market_data") or {}
    greeks = side.get("option_greeks") or {}
    return OptionQuote(
        instrument_key=side["instrument_key"],
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        bid=_nan_or_positive(md.get("bid_price")),
        ask=_nan_or_positive(md.get("ask_price")),
        bid_qty=int(md.get("bid_qty") or 0),
        ask_qty=int(md.get("ask_qty") or 0),
        last=_nan_or_positive(md.get("ltp")),
        prev_close=_nan_or_positive(md.get("close_price")),
        volume=int(md.get("volume") or 0),
        oi=float(md.get("oi") or 0.0),
        prev_oi=float(md.get("prev_oi") or 0.0),
        vendor_iv=_num(greeks.get("iv")),
        vendor_delta=_num(greeks.get("delta")),
    )


def _nan_or_positive(x) -> float:
    if x is None:
        return math.nan
    x = float(x)
    return x if x > 0 else math.nan


class UpstoxSource:
    """``MarketDataSource`` backed by Upstox."""

    name = "upstox"

    def __init__(self, client: UpstoxClient, *, cache_dir: Path | str = "data/cache",
                 instruments: Optional[InstrumentMaster] = None, clock: Callable[[], datetime] = None):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self._instruments = instruments
        self._clock = clock or (lambda: datetime.now(IST))

    @classmethod
    def from_env(cls, search_dirs: Optional[Iterable[Path]] = None, **kwargs) -> "UpstoxSource":
        return cls(UpstoxClient.from_env(search_dirs), **kwargs)

    @property
    def instruments(self) -> InstrumentMaster:
        if self._instruments is None:
            self._instruments = InstrumentMaster.load(self.cache_dir)
        return self._instruments

    @staticmethod
    def resolve(underlying: str) -> tuple[str, str]:
        """Return (symbol, instrument_key) for 'NIFTY', 'nifty' or 'NSE_INDEX|Nifty 50'."""
        if "|" in underlying:
            return _KEY_TO_SYMBOL.get(underlying, underlying), underlying
        symbol = underlying.strip().upper()
        if symbol not in UNDERLYINGS:
            raise KeyError(f"unknown underlying {underlying!r}; supported: {', '.join(UNDERLYINGS)}")
        return symbol, UNDERLYINGS[symbol]

    def expiries(self, underlying: str) -> list[date]:
        symbol, key = self.resolve(underlying)
        today = self._clock().date()
        try:
            listed = self.instruments.expiries(symbol)
        except Exception:  # master unavailable: fall back to the contracts API
            listed = sorted({_parse_expiry(c["expiry"]) for c in self.client.option_contracts(key)})
        return [d for d in listed if d >= today]

    def option_chain(self, underlying: str, expiry: date | str) -> OptionChain:
        symbol, key = self.resolve(underlying)
        expiry_str = expiry.isoformat() if isinstance(expiry, date) else str(expiry)
        as_of = self._clock()
        rows = self.client.option_chain(key, expiry_str)
        if not rows:
            raise UpstoxError(f"empty option chain for {symbol} {expiry_str}; check the expiry is listed")
        quotes: list[OptionQuote] = []
        spot = math.nan
        chain_expiry = None
        for row in rows:
            strike = float(row["strike_price"])
            chain_expiry = _parse_expiry(row.get("expiry", expiry_str))
            spot = _num(row.get("underlying_spot_price")) if math.isnan(spot) else spot
            for side, kind in (("call_options", "call"), ("put_options", "put")):
                q = _option_quote(row.get(side), chain_expiry, strike, kind)
                if q is not None:
                    quotes.append(q)
        return OptionChain(underlying=symbol, underlying_key=key, expiry=chain_expiry, spot=spot,
                           as_of=as_of, source=self.name, quotes=tuple(quotes))

    def spot(self, underlying: str) -> float:
        _, key = self.resolve(underlying)
        q = self.client.quotes([key]).get(key)
        if not q or q.get("last_price") is None:
            raise UpstoxError(f"no quote returned for {key}")
        return float(q["last_price"])

    def futures(self, underlying: str) -> list[FutureQuote]:
        symbol, _ = self.resolve(underlying)
        today = self._clock().date()
        contracts = [c for c in self.instruments.futures(symbol) if c.expiry >= today]
        if not contracts:
            return []
        raw = self.client.quotes([c.instrument_key for c in contracts])
        out = []
        for c in contracts:
            q = raw.get(c.instrument_key)
            if not q:
                continue
            depth = q.get("depth") or {}
            best_bid = (depth.get("buy") or [{}])[0].get("price")
            best_ask = (depth.get("sell") or [{}])[0].get("price")
            out.append(FutureQuote(
                instrument_key=c.instrument_key, underlying=symbol, expiry=c.expiry,
                last=_num(q.get("last_price")), bid=_nan_or_positive(best_bid), ask=_nan_or_positive(best_ask),
                oi=float(q.get("oi") or 0.0), volume=int(q.get("volume") or 0),
                as_of=_parse_time(q.get("timestamp")),
            ))
        return out
