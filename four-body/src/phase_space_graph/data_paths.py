"""Default persistent data locations for the four-body demo."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path.home() / ".psf-data"
DEMO_NAME = "4体"


def data_root() -> Path:
    """Return the configured cross-demo persistent data root.

    ``PSF_DATA_ROOT`` is preferred for shared or reproducible data volumes;
    the user-local default keeps generated files out of the source checkout.
    """

    value = os.environ.get("PSF_DATA_ROOT")
    return Path(value).expanduser() if value else DEFAULT_DATA_ROOT


def demo_data_root() -> Path:
    """Return the four-body data directory on the configured volume."""

    return data_root() / "Demo案例" / DEMO_NAME


def default_output(*parts: str) -> Path:
    """Build a default four-body output path without creating it."""

    return demo_data_root().joinpath(*parts)


__all__ = ["DEFAULT_DATA_ROOT", "DEMO_NAME", "data_root", "demo_data_root", "default_output"]
