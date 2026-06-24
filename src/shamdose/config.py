from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """
    Return the root folder of the cerebellar-sham-dose project.

    This file lives at:
        src/shamdose/config.py

    Therefore:
        parents[0] = src/shamdose
        parents[1] = src
        parents[2] = project root
    """
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the local YAML configuration file.

    By default, this reads:
        config/paths_local.yaml

    paths_local.yaml is private and should not be uploaded to GitHub.
    """
    project_root = get_project_root()

    if config_path is None:
        config_path = project_root / "config" / "paths_local.yaml"
    else:
        config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Create it from config/paths_template.yaml."
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file did not load as a dictionary: {config_path}")

    return cfg


def require_config_key(cfg: dict[str, Any], *keys: str) -> Any:
    """
    Safely retrieve nested config values.

    Example:
        require_config_key(cfg, "cohorts", "HC", "headmodels_root")
    """
    value: Any = cfg
    path_so_far: list[str] = []

    for key in keys:
        path_so_far.append(key)

        if not isinstance(value, dict) or key not in value:
            joined = " -> ".join(path_so_far)
            raise KeyError(f"Missing required config key: {joined}")

        value = value[key]

    return value
