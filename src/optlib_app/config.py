"""Settings for the API, agents and dashboard. Read from environment variables or the repo's .env.

    DATA_MODE                 replay (default: the recorded 23 Oct 2026 NIFTY snapshot) | live (Upstox + FBIL)
    UPSTOX_ANALYTICS_TOKEN    needed for live mode
    LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, VALIDATOR_MODEL
                              the LLM the agents run on (see optlib_app.llm)
    PRICING_API_URL           where the dashboard finds the API (default http://127.0.0.1:8000)
    SURFACE_TTL_SECONDS       live-mode cache lifetime for surfaces (default 60)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from optlib.data.env import read_env_file

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"


def get_setting(name: str, default: str = "") -> str:
    """Environment variable first, then the repo's .env, then the default."""
    value = os.environ.get(name, "").strip()
    return value or read_env_file(ROOT / ".env").get(name, "").strip() or default


_get = get_setting  # backwards-compatible name


@dataclass
class Settings:
    data_mode: str = field(default_factory=lambda: get_setting("DATA_MODE", "replay").lower())
    snapshot_path: Path = FIX / "upstox_live_nifty_20261023_full.json"
    instruments_path: Path = FIX / "upstox_instruments_sample.json"
    fbil_ois_path: Path = FIX / "fbil_miborois_20261001_20261023.json"
    fbil_overnight_path: Path = FIX / "fbil_ovnmibor_20261001_20261023.json"
    history_path: Path = FIX / "index_history_20150101_20261023.csv"
    bhavcopy_files: dict = field(default_factory=lambda: {
        "2026-10-22": FIX / "nse_fo_bhavcopy_nifty_20261022.csv",
        "2026-10-23": FIX / "nse_fo_bhavcopy_nifty_20261023.csv"})
    limits_path: Path = ROOT / "config" / "risk_limits.json"
    charges_path: Path = ROOT / "config" / "charges.json"
    margin_path: Path = ROOT / "config" / "margin.json"
    audit_log_path: Path = ROOT / "data" / "recommendations.jsonl"
    cache_dir: Path = ROOT / "data" / "cache"
    surface_ttl_seconds: float = field(default_factory=lambda: float(get_setting("SURFACE_TTL_SECONDS", "60")))
    api_url: str = field(default_factory=lambda: get_setting("PRICING_API_URL", "http://127.0.0.1:8000"))

    def __post_init__(self):
        if self.data_mode not in ("replay", "live"):
            raise ValueError(f"DATA_MODE must be 'replay' or 'live', got {self.data_mode!r}")


def llm_status() -> dict:
    """What the agents will run on, without exposing the key."""
    from .llm import LLMNotConfigured, make_llm

    try:
        llm = make_llm(get_setting)
        return {"configured": True, "provider": llm.provider, "model": llm.model, "message": ""}
    except LLMNotConfigured as exc:
        return {"configured": False, "provider": get_setting("LLM_PROVIDER", "") or None, "model": None,
                "message": str(exc)}
