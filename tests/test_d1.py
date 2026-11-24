"""The store over Cloudflare D1: the HTTP client is exercised against a fake D1 that answers like the real API
(result[0].results rows as objects, meta.changes) from a local SQLite database."""
import sqlite3

import pytest

from optlib_app import d1
from optlib_app.subscribers import SubscriberStore


class FakeD1Session:
    """Stands in for requests.Session: routes D1 API calls to an in-memory SQLite database."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.headers = {}
        self.calls = []

    class _Resp:
        def __init__(self, payload, status=200):
            self._p, self.status_code = payload, status

        def json(self):
            return self._p

    def post(self, url, json, timeout):
        self.calls.append(url)
        assert self.headers["Authorization"] == "Bearer tok"
        try:
            cur = self.db.execute(json["sql"], json["params"])
            rows = [dict(r) for r in cur.fetchall()]
            self.db.commit()
        except sqlite3.Error as exc:
            return self._Resp({"success": False, "errors": [{"message": str(exc)}]}, 400)
        return self._Resp({"success": True, "result": [{"results": rows, "meta": {"changes": max(cur.rowcount, 0)}}]})

    def get(self, url, headers, params, timeout):
        if url.endswith("/accounts"):
            return self._Resp({"success": True, "result": [{"id": "acc1"}]})
        return self._Resp({"success": True, "result": [{"name": "option-pricing-agent", "uuid": "db1"}]})


def test_store_on_d1(tmp_path):
    s = FakeD1Session()
    conn = d1.connect_from_settings(lambda k, default="": {"CLOUDFLARE_API_TOKEN": "tok"}.get(k, default), session=s)
    assert conn.url.endswith("/accounts/acc1/d1/database/db1/query")  # account and database found from the token
    st = SubscriberStore(tmp_path / "unused.db", remote=conn)
    assert not (tmp_path / "unused.db").exists() and st.where.startswith("Cloudflare D1")
    a = st.create_account("Asha@Example.com", "Asha", "secret123", consent=True)
    assert a.status == "pending" and st.authenticate("asha@example.com", "secret123")
    st.confirm(a.token)
    assert [x.email for x in st.active()] == ["Asha@example.com"]
    tok = st.new_session("asha@example.com")
    assert st.session_user(tok).email == "Asha@example.com"
    st.save_history({"date": "2026-11-20", "spot": 1.0})
    st.save_history({"date": "2026-11-20", "spot": 2.0})  # same day replaces
    assert st.previous_history("2026-11-23") == {"date": "2026-11-20", "spot": 2.0}
    st.add_log({"at": "t", "date": "2026-11-23", "kind": "daily", "sent": 3})
    assert st.logs()[-1]["sent"] == 3
    assert st.delete("asha@example.com") and st.get("asha@example.com") is None


def test_d1_errors_are_clear():
    s = FakeD1Session()
    conn = d1.D1Connection("a", "b", "tok", session=s)
    with pytest.raises(d1.D1Error, match="refused the query"):
        conn.execute("SELECT * FROM missing_table")
    assert d1.connect_from_settings(lambda k, default="": default) is None  # no token: local SQLite


def test_cloud_setup_migrates_local_list(tmp_path):
    import importlib.util

    from optlib_app.subscribers import SubscriberStore as Store

    spec = importlib.util.spec_from_file_location("cloud_setup", "scripts/cloud_setup.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    local = Store(tmp_path / "local.db")
    acct = local.create_account("me@example.com", "Me", "secret123", consent=True)
    local.confirm(acct.token)
    s = FakeD1Session()
    remote = Store(tmp_path / "x.db", remote=d1.D1Connection("a", "b", "tok", session=s))
    assert cs.migrate(tmp_path / "local.db", remote) == (1, 0)
    assert cs.migrate(tmp_path / "local.db", remote) == (0, 1)  # second run changes nothing
    moved = remote.get("me@example.com")
    assert moved.status == "active" and remote.authenticate("me@example.com", "secret123")  # password kept


def test_d1_only_token_needs_account_id():
    """A token limited to D1 cannot list accounts (Cloudflare returns none): ask for the ID, then it works."""
    s = FakeD1Session()
    s.get = lambda url, headers, params, timeout: FakeD1Session._Resp(
        {"success": True, "result": []} if url.endswith("/accounts") else
        {"success": True, "result": [{"name": "option-pricing-agent", "uuid": "db1"}]})
    with pytest.raises(d1.NeedAccountId, match="32-character code"):
        d1.connect_from_settings(lambda k, default="": {"CLOUDFLARE_API_TOKEN": "tok"}.get(k, default), session=s)
    conn = d1.connect_from_settings(lambda k, default="": {"CLOUDFLARE_API_TOKEN": "tok",
                                                          "D1_ACCOUNT_ID": "a" * 32}.get(k, default), session=s)
    assert conn.url.endswith(f"/accounts/{'a' * 32}/d1/database/db1/query")
