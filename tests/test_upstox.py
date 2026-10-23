"""Upstox adapter tests on recorded-format fixtures; no network access."""
import gzip
import json
import math
from datetime import date, datetime
from pathlib import Path

import pytest

from optlib import implied_vol_black76
from optlib.data import (
    IST,
    InstrumentMaster,
    MarketDataSource,
    UpstoxAPIError,
    UpstoxAuthError,
    UpstoxClient,
    UpstoxError,
    UpstoxSource,
    year_fraction,
)
from optlib.data.env import get_secret, mask, read_env_file

FIX = Path(__file__).parent / "fixtures"
TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGc.fake.token-for-tests-1234"


def load(name):
    return json.loads((FIX / name).read_text())


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    """Returns queued responses per path; records every call."""

    def __init__(self, routes):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        path = url.split("api.upstox.com", 1)[1]
        queue = self.routes[path]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def ok(name):
    return FakeResponse(200, load(name))


@pytest.fixture
def master():
    return InstrumentMaster.from_records(load("upstox_instruments_sample.json"))


def make_source(routes, master, clock_at=datetime(2026, 10, 23, 11, 0, tzinfo=IST)):
    session = FakeSession(routes)
    client = UpstoxClient(TOKEN, session=session, sleep=lambda s: None)
    return UpstoxSource(client, instruments=master, clock=lambda: clock_at), session


# ------------------------------------------------------------------ option chain

def test_option_chain_parsing(master):
    src, session = make_source({"/v2/option/chain": [ok("upstox_option_chain.json")]}, master)
    chain = src.option_chain("nifty", date(2026, 10, 27))

    call = session.calls[0]
    assert call["params"] == {"instrument_key": "NSE_INDEX|Nifty 50", "expiry_date": "2026-10-27"}
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"

    assert chain.underlying == "NIFTY" and chain.source == "upstox"
    assert chain.expiry == date(2026, 10, 27)
    assert chain.spot == 25712.35
    assert len(chain.strikes) == 21
    assert len(chain.calls()) == 21 and len(chain.puts()) == 20  # strike 25700 has no put listed
    assert chain.quote(25700.0, "put") is None

    q = chain.quote(25650.0, "call")
    assert (q.bid, q.ask) == (172.65, 173.5)
    assert q.mid == pytest.approx(173.075) and q.two_sided
    assert q.instrument_key == "NSE_FO|50009"
    assert q.oi == 845000.0 and q.volume == 125000
    assert 0 < q.vendor_iv < 100  # vendor reports percent; kept for comparison only


def test_zero_bid_becomes_nan(master):
    src, _ = make_source({"/v2/option/chain": [ok("upstox_option_chain.json")]}, master)
    chain = src.option_chain("NIFTY", "2026-10-27")
    deep_call = chain.quote(26200.0, "call")
    assert math.isnan(deep_call.bid) and not deep_call.two_sided and math.isnan(deep_call.mid)
    assert deep_call.ask == 8.95


def test_chain_round_trips_through_optlib_iv(master):
    """Prices survive parsing: implied vols from the mids recover the fixture smile."""
    meta = load("upstox_chain_meta.json")
    src, _ = make_source({"/v2/option/chain": [ok("upstox_option_chain.json")]}, master)
    chain = src.option_chain("NIFTY", "2026-10-27")
    T = chain.time_to_expiry()
    assert T == pytest.approx(meta["time_to_expiry"], rel=1e-12)
    F, r = meta["forward"], meta["rate"]
    for q in chain.quotes:
        if not q.two_sided or (q.option_type == "call") != (q.strike >= F):
            continue  # OTM side only
        vol = implied_vol_black76(q.mid, F, q.strike, T, r, q.option_type)
        x = math.log(q.strike / F)
        expected = 0.118 - 0.35 * x + 2.2 * x * x
        assert vol == pytest.approx(expected, abs=0.004), q.strike  # tick rounding of quotes


def test_to_records(master):
    src, _ = make_source({"/v2/option/chain": [ok("upstox_option_chain.json")]}, master)
    rows = src.option_chain("NIFTY", "2026-10-27").to_records()
    assert len(rows) == 41
    assert rows[0]["strike"] == 25200.0 and rows[0]["option_type"] == "call"
    assert rows[0]["expiry"] == "2026-10-27" and rows[0]["as_of"].endswith("+05:30")


def test_empty_chain_raises(master):
    src, _ = make_source({"/v2/option/chain": [FakeResponse(200, {"status": "success", "data": []})]}, master)
    with pytest.raises(UpstoxError, match="empty option chain"):
        src.option_chain("NIFTY", "2026-10-28")


# ------------------------------------------------------------------ quotes, futures, expiries

def test_spot_and_futures(master):
    routes = {"/v2/market-quote/quotes": [ok("upstox_quotes.json")]}
    src, session = make_source(routes, master)
    assert src.spot("NIFTY") == 25712.35

    futs = src.futures("NIFTY")
    nearest = futs[0]
    assert nearest.instrument_key == "NSE_FO|48704" and nearest.expiry == date(2026, 10, 27)
    assert (nearest.bid, nearest.ask, nearest.last) == (25745.1, 25746.5, 25745.8)
    assert nearest.mid == pytest.approx(25745.8)
    assert nearest.as_of.utcoffset().total_seconds() == 19800
    # all three listed NIFTY futures were requested in one comma-separated call
    keys = session.calls[-1]["params"]["instrument_key"].split(",")
    assert len(keys) == 3 and "NSE_FO|48704" in keys


def test_expiries_from_master_include_holiday_shifts(master):
    src, _ = make_source({}, master)
    exp = src.expiries("NIFTY")
    assert exp[0] == date(2026, 10, 27)
    assert exp == sorted(exp) and all(d >= date(2026, 10, 23) for d in exp)
    # Nov monthly expires Monday 23 Nov (Tuesday 24 Nov is a holiday): taken from exchange data, not a weekday rule
    assert date(2026, 11, 23) in exp and date(2026, 11, 24) not in exp


def test_master_parsing(master):
    futs = master.futures("NIFTY")
    assert len(futs) == 3 and futs[0].lot_size == 65 and futs[0].tick_size == pytest.approx(0.10)
    assert master.lot_size("NIFTY") == 65
    assert {c.underlying for c in master.contracts} == {"NIFTY", "BANKNIFTY"}
    assert all(c.instrument_type in ("FUT", "CE", "PE") for c in master.contracts)


def test_master_cache(tmp_path):
    payload = gzip.compress((FIX / "upstox_instruments_sample.json").read_bytes())
    fetched = []

    def fetch(url):
        fetched.append(url)
        return payload

    day = date(2026, 10, 23)
    m1 = InstrumentMaster.load(tmp_path, today=day, fetch=fetch)
    m2 = InstrumentMaster.load(tmp_path, today=day, fetch=fetch)
    assert len(fetched) == 1 and len(m1.contracts) == len(m2.contracts) > 0
    assert (tmp_path / "upstox_NSE_2026-10-23.json.gz").is_file()


def test_resolve():
    assert UpstoxSource.resolve("banknifty") == ("BANKNIFTY", "NSE_INDEX|Nifty Bank")
    assert UpstoxSource.resolve("NSE_INDEX|NIFTY MID SELECT") == ("MIDCPNIFTY", "NSE_INDEX|NIFTY MID SELECT")
    with pytest.raises(KeyError):
        UpstoxSource.resolve("SENSEX")


def test_source_satisfies_protocol(master):
    src, _ = make_source({}, master)
    assert isinstance(src, MarketDataSource)


# ------------------------------------------------------------------ errors and retries

def test_401_raises_auth_error_without_retry(master):
    src, session = make_source({"/v2/option/chain": [FakeResponse(401, load("upstox_error_401.json"))]}, master)
    with pytest.raises(UpstoxAuthError, match="Analytics Token"):
        src.option_chain("NIFTY", "2026-10-27")
    assert len(session.calls) == 1


def test_400_raises_api_error_with_codes(master):
    src, _ = make_source({"/v2/option/chain": [FakeResponse(400, load("upstox_error_400.json"))]}, master)
    with pytest.raises(UpstoxAPIError) as info:
        src.option_chain("NIFTY", "2026-13-01")
    assert info.value.codes == ["UDAPI100036"] and info.value.status == 400


def test_429_then_success_retries_and_honours_retry_after(master):
    sleeps = []
    session = FakeSession({"/v2/option/chain": [
        FakeResponse(429, {"status": "error", "errors": [{"error_code": "UDAPI10005"}]}, {"Retry-After": "2"}),
        FakeResponse(503, ValueError("not json")),
        ok("upstox_option_chain.json"),
    ]})
    client = UpstoxClient(TOKEN, session=session, sleep=sleeps.append, backoff=0.5)
    src = UpstoxSource(client, instruments=master, clock=lambda: datetime(2026, 10, 23, 11, tzinfo=IST))
    chain = src.option_chain("NIFTY", "2026-10-27")
    assert len(chain.quotes) == 41
    assert sleeps == [2.0, 1.0]  # Retry-After honoured, then exponential backoff


def test_gives_up_after_max_retries(master):
    session = FakeSession({"/v2/option/chain": [FakeResponse(500, None)]})
    client = UpstoxClient(TOKEN, session=session, sleep=lambda s: None, max_retries=2)
    src = UpstoxSource(client, instruments=master)
    with pytest.raises(UpstoxAPIError):
        src.option_chain("NIFTY", "2026-10-27")
    assert len(session.calls) == 3


def test_network_error_retries_then_raises(master):
    session = FakeSession({"/v2/option/chain": [ConnectionError("boom")]})
    client = UpstoxClient(TOKEN, session=session, sleep=lambda s: None, max_retries=1)
    src = UpstoxSource(client, instruments=master)
    with pytest.raises(UpstoxError, match="network error"):
        src.option_chain("NIFTY", "2026-10-27")
    assert len(session.calls) == 2


def test_token_never_in_repr_or_errors(master):
    client = UpstoxClient(TOKEN, session=FakeSession({"/v2/option/chain": [FakeResponse(401, {})]}))
    assert TOKEN not in repr(client)
    with pytest.raises(UpstoxAuthError) as info:
        UpstoxSource(client, instruments=master).option_chain("NIFTY", "2026-10-27")
    assert TOKEN not in str(info.value)


def test_empty_token_rejected():
    with pytest.raises(UpstoxAuthError):
        UpstoxClient("", session=FakeSession({}))


# ------------------------------------------------------------------ .env handling

def test_env_file_and_precedence(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "﻿# comment\n\nexport OTHER=1\nUPSTOX_ANALYTICS_TOKEN = \"abc123def456ghi789\"\n", encoding="utf-8")
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    assert read_env_file(tmp_path / ".env")["OTHER"] == "1"
    assert get_secret("UPSTOX_ANALYTICS_TOKEN", [tmp_path]) == "abc123def456ghi789"
    monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "from-env-var")
    assert get_secret("UPSTOX_ANALYTICS_TOKEN", [tmp_path]) == "from-env-var"


def test_missing_secret_message(tmp_path, monkeypatch):
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    with pytest.raises(KeyError, match=".env"):
        get_secret("UPSTOX_ANALYTICS_TOKEN", [tmp_path])


def test_mask():
    assert mask(TOKEN) == "eyJ0...1234" and TOKEN not in mask(TOKEN)
    assert mask("short") == "****"


# ------------------------------------------------------------------ time conventions

def test_year_fraction_to_1530_ist():
    start = datetime(2026, 10, 27, 9, 15, tzinfo=IST)
    assert year_fraction(start, date(2026, 10, 27)) == pytest.approx(6.25 / 24 / 365)
    assert year_fraction(datetime(2026, 10, 27, 16, 0, tzinfo=IST), date(2026, 10, 27)) == 0.0
    with pytest.raises(ValueError):
        year_fraction(datetime(2026, 10, 27, 9, 15), date(2026, 10, 27))
