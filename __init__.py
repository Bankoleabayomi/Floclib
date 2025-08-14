# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:39:05 2025

@author: banko
"""

# floclab/__init__.py
"""floclab package: core access points."""

#__version__ = "0.1.0"   # match pyproject.toml version
# floclab/__init__.py
"""floclab package."""

try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    # for older Python or environments, the backport can be used
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("floclab")
except PackageNotFoundError:
    __version__ = "0+unknown"


# expose core functions at package-level if you like
from .asd import compute_beta_from_features
from .fit import fit_ka_kb
from .cstr import simulate_retention_times

__all__ = ["compute_beta_from_features", "fit_ka_kb", "simulate_retention_times"]
