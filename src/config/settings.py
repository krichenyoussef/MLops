from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class Settings:
    """
    Runtime settings loaded from environment variables.
    Keep secrets out of params.yaml and out of Git.
    """
    mlflow_tracking_uri: Optional[str]
    mlflow_username: Optional[str]
    mlflow_password: Optional[str]


def load_params(path: str = "params.yaml") -> Dict[str, Any]:
    """
    Loads non-secret configuration (paths, training params, etc.)
    from params.yaml.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"params file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings() -> Settings:
    """
    Loads secret/runtime environment variables.
    """
    return Settings(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME"),
        mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD"),
    )
