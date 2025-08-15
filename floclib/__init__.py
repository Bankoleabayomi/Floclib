# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:39:05 2025

@author: banko
"""

# floclib/__init__.py
"""floclib package: core access points."""

#__version__ = "0.1.0"   # match pyproject.toml version
# floclib/__init__.py
"""floclib package."""

try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    # for older Python or environments, the backport can be used
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("floclib")
except PackageNotFoundError:
    __version__ = "0+unknown"


# expose core functions at package-level if you like
from .asd import compute_beta
from .fit import fit_ka_kb
from .cstr import simulate_retention_times

__all__ = ["compute_beta", "fit_ka_kb", "simulate_retention_times"]
