"""Settings for the API, agents and dashboard. Read from environment variables or the repo's .env.

    DATA_MODE                 replay (default: the recorded 23 Oct 2026 NIFTY snapshot) | live (Upstox + FBIL)
    UPSTOX_ANALYTICS_TOKEN    needed for live mode
    ANTHROPIC_API_KEY         needed for the agents
    ORCHESTRATOR_MODEL        default claude-sonnet-5-5
    VALIDATOR_MODEL           default claude-sonnet-5-5
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


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or read_env_file(ROOT / ".env").get(name, "").strip() or default


@dataclass
class Settings:
    data_mode: str = field(default_factory=lambda: _get("DATA_MODE", "replay").lower())
    snapshot_path: Path = FIX / "upstox_live_nifty_20261023_full.json"
    instruments_path: Path = FIX / "upstox_instruments_sample.json"
    fbil_ois_path: Path = FIX / "fbil_miborois_20261001_20261023.json"
    fbil_overnight_path: Path = FIX / "fbil_ovnmibor_20261001_20261023.json"
    cache_dir: Path = ROOT / "data" / "cache"
    surface_ttl_seconds: float = field(default_factory=lambda: float(_get("SURFACE_TTL_SECONDS", "60")))
    orchestrator_model: str = field(default_factory=lambda: _get("ORCHESTRATOR_MODEL", "claude-sonnet-5-5"))
    validator_model: str = field(default_factory=lambda: _get("VALIDATOR_MODEL", "claude-sonnet-5-5"))
    api_url: str = field(default_factory=lambda: _get("PRICING_API_URL", "http://127.0.0.1:8000"))

    def __post_init__(self):
        if self.data_mode not in ("replay", "live"):
            raise ValueError(f"DATA_MODE must be 'replay' or 'live', got {self.data_mode!r}")

    @staticmethod
    def anthropic_api_key() -> str:
        return _get("ANTHROPIC_API_KEY")
