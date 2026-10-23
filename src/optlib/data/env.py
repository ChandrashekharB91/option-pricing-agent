"""Credential loading. Secrets come from the environment or a local ``.env`` file only;
they are never logged, printed or committed."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignores blanks, comments and surrounding quotes."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def get_secret(name: str, search_dirs: Optional[Iterable[Path]] = None) -> str:
    """Environment variable first, then ``.env`` in the given directories (default: cwd)."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    for directory in search_dirs or [Path.cwd()]:
        value = read_env_file(Path(directory) / ".env").get(name, "").strip()
        if value:
            return value
    raise KeyError(
        f"{name} not found. Add a line '{name}=<value>' to the .env file in the repo root, "
        f"or set it as an environment variable.")


def mask(secret: str) -> str:
    """Safe display form: first and last 4 characters only."""
    return f"{secret[:4]}...{secret[-4:]}" if len(secret) > 12 else "****"
