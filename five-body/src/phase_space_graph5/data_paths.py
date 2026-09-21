"""Default persistent data locations for the five-body demo.

Generated data is kept outside the Git working tree by default.  The
``PSF_DATA_ROOT`` environment variable is an explicit escape hatch for a
different data volume or a reproducible isolated run.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path.home() / ".psf-data"
DEMO_NAME = "5体"


def data_root() -> Path:
    """Return the configured cross-demo persistent data root."""

    value = os.environ.get("PSF_DATA_ROOT")
    return Path(value).expanduser() if value else DEFAULT_DATA_ROOT


def demo_data_root() -> Path:
    """Return the five-body data directory on the configured volume."""

    return data_root() / "Demo案例" / DEMO_NAME


def default_output(*parts: str) -> Path:
    """Build a default five-body output path without creating it."""

    return demo_data_root().joinpath(*parts)


def active_catalog_root(*parts: str) -> Path:
    """Return the relocated five-body catalog root."""

    return demo_data_root().joinpath("repo_outputs", "candidate_catalog", *parts)


__all__ = [
    "DEFAULT_DATA_ROOT", "DEMO_NAME", "data_root", "demo_data_root",
    "default_output", "active_catalog_root",
]
