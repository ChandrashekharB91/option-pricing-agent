"""Regression suite: every frozen output must match tests/regression/golden_phase1.json within tolerance,
and the pricer must pass the QuantLib gates. To accept a deliberate change run
``python scripts/regression.py --update --reason "..."`` and get the changelog entry reviewed."""
import json

import pytest

from optlib_app import regression as reg


@pytest.fixture(scope="module")
def golden():
    return reg.load_golden()


@pytest.fixture(scope="module")
def current():
    return reg.compute()


def test_golden_file_is_stamped(golden):
    meta = golden["meta"]
    assert meta["reason"] and meta["updated"]
    assert set(golden["values"]) == set(reg.SECTIONS)
    assert reg.CHANGELOG.exists()


@pytest.mark.parametrize("section", list(reg.SECTIONS))
def test_section_matches_golden(golden, current, section):
    rows = reg.compare({section: golden["values"][section]}, {section: current[section]})
    bad = [f"{r['status']} {r['key']}: golden {r['golden']}, now {r['current']} (tol {r['tolerance']})"
           for r in rows if r["status"] != "PASS"]
    assert not bad, f"{len(bad)} of {len(rows)} values changed:\n" + "\n".join(bad[:25])


@pytest.mark.parametrize("gate", reg.gates(), ids=lambda g: g["gate"])
def test_benchmark_gate(gate):
    assert gate["passed"], gate["detail"]


def test_compare_flags_changes():
    gold = {"pricing": {"ATM_call/price": 100.0, "ATM_call/vol_source": "market", "ATM_call/released": True}}
    cur = {"pricing": {"ATM_call/price": 100.0005, "ATM_call/vol_source": "svi", "extra/price": 1.0}}
    by = {r["key"]: r["status"] for r in reg.compare(gold, cur)}
    assert by == {"ATM_call/price": "PASS", "ATM_call/vol_source": "FAIL", "ATM_call/released": "MISSING",
                  "extra/price": "NEW"}
    assert reg.compare({"pricing": {"x/price": 100.0}}, {"pricing": {"x/price": 100.002}})[0]["status"] == "FAIL"


def test_every_value_has_a_tolerance(current):
    for section, values in current.items():
        for key, v in values.items():
            if isinstance(v, float):
                reg.tolerance(section, key)
    json.dumps(current)  # serialisable
