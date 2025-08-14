# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:39:05 2025

@author: banko
"""

# floclab/__init__.py
"""floclab package: core access points."""

__version__ = "0.1.0"   # match pyproject.toml version

# expose core functions at package-level if you like
from .asd import compute_beta_from_features
from .fit import fit_ka_kb
from .cstr import simulate_retention_times

__all__ = ["compute_beta_from_features", "fit_ka_kb", "simulate_retention_times"]
