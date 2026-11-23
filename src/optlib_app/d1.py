"""Cloudflare D1 (hosted SQLite) as the store for accounts, subscribers and the feed's history and send log.

The laptop app, the hosted app and the scheduled GitHub Actions job all read and write the same database, so
the daily email does not depend on any one computer being on. D1 speaks SQLite, so the store's SQL is unchanged:
this module only provides a small connection object with the parts of sqlite3 the store uses.

Settings:
    CLOUDFLARE_API_TOKEN   API token with "Account > D1 > Edit" (dash.cloudflare.com > My Profile > API Tokens)
    D1_DATABASE_NAME       default "option-pricing-agent" (or D1_DATABASE_ID)
    D1_ACCOUNT_ID          optional; found from the token when the token covers a single account
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import requests

API = "https://api.cloudflare.com/client/v4"
DEFAULT_NAME = "option-pricing-agent"


class D1Error(RuntimeError):
    pass


class _Cursor:
    def __init__(self, rows: list[dict], changes: int):
        self._rows = rows
        self.rowcount = changes

    def fetchone(self) -> Optional[dict]:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict]:
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class D1Connection:
    """execute(sql, params) like sqlite3, over D1's HTTP query API. Rows are dicts."""

    def __init__(self, account_id: str, database_id: str, token: str, session: Optional[requests.Session] = None,
                 timeout: float = 30):
        self.url = f"{API}/accounts/{account_id}/d1/database/{database_id}/query"
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.timeout = timeout

    def execute(self, sql: str, params: Iterable = ()) -> _Cursor:
        body = {"sql": sql, "params": [p for p in params]}
        try:
            r = self.session.post(self.url, json=body, timeout=self.timeout)
            data = r.json()
        except (requests.RequestException, ValueError) as exc:
            raise D1Error(f"Cloudflare D1 not reachable: {exc}") from exc
        if not data.get("success"):
            msg = "; ".join(e.get("message", "") for e in data.get("errors", [])) or f"HTTP {r.status_code}"
            raise D1Error(f"Cloudflare D1 refused the query: {msg}")
        res = (data.get("result") or [{}])[0]
        return _Cursor(res.get("results") or [], int((res.get("meta") or {}).get("changes") or 0))

    # sqlite3.Connection compatibility for the store's context manager
    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


def _get(session: requests.Session, url: str, token: str, **params) -> list:
    r = session.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)
    data = r.json()
    if not data.get("success"):
        msg = "; ".join(e.get("message", "") for e in data.get("errors", [])) or f"HTTP {r.status_code}"
        raise D1Error(f"Cloudflare API: {msg}")
    return data.get("result") or []


def connect_from_settings(get: Callable[[str, str], str], session: Optional[requests.Session] = None
                          ) -> Optional[D1Connection]:
    """A D1 connection when CLOUDFLARE_API_TOKEN is set, else None (the store then uses a local SQLite file)."""
    token = get("CLOUDFLARE_API_TOKEN", "")
    if not token:
        return None
    session = session or requests.Session()
    account = get("D1_ACCOUNT_ID", "")
    if not account:
        accounts = _get(session, f"{API}/accounts", token)
        if len(accounts) != 1:
            raise D1Error("set D1_ACCOUNT_ID: the API token can see "
                          f"{len(accounts)} Cloudflare accounts" if accounts else "the API token sees no account")
        account = accounts[0]["id"]
    db_id = get("D1_DATABASE_ID", "")
    if not db_id:
        name = get("D1_DATABASE_NAME", "") or DEFAULT_NAME
        found = [d for d in _get(session, f"{API}/accounts/{account}/d1/database", token, name=name)
                 if d.get("name") == name]
        if not found:
            raise D1Error(f"no D1 database named {name!r} in the Cloudflare account")
        db_id = found[0]["uuid"]
    return D1Connection(account, db_id, token, session=session)
