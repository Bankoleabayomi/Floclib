# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:14:02 2025

@author: banko
"""

# floclib/cstr.py
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Union, Optional
import math
import csv
from datetime import datetime

def reactor_ratio_product(T: float, Gf: np.ndarray, Ka: np.ndarray, Kb: np.ndarray, m: int) -> float:
    """
    Compute the product_term (R_calculated) for the m-tank system as in your script.
    """
    product_term = 1.0
    n_prev = 1.0
    for i in range(m):
        # ratio = (1 + Ka[i] * Gf[i] * T / m) / (1 + n_prev * Kb[i] * Gf[i]**2 * T / m)
        numerator = 1.0 + Ka[i] * Gf[i] * T / m
        denominator = 1.0 + n_prev * Kb[i] * (Gf[i] ** 2) * T / m
        ratio = numerator / denominator
        product_term *= ratio
        n_prev = ratio
    return product_term

def f_for_R(T: float, R_specified: float, Gf: np.ndarray, Ka: np.ndarray, Kb: np.ndarray, m: int) -> float:
    return R_specified - reactor_ratio_product(T, Gf, Ka, Kb, m)

def secant_method(T0: float, T1: float, R_specified: float, Gf: np.ndarray, Ka: np.ndarray, Kb: np.ndarray, m: int,
                  tol: float = 1e-6, max_iter: int = 1000) -> Optional[float]:
    for iteration in range(max_iter):
        f_T0 = f_for_R(T0, R_specified, Gf, Ka, Kb, m)
        f_T1 = f_for_R(T1, R_specified, Gf, Ka, Kb, m)
        denom = (f_T1 - f_T0)
        if abs(denom) < 1e-12:
            return None
        T_new = T1 - f_T1 * (T1 - T0) / denom
        # convergence check
        if abs(T_new - T1) < tol:
            return T_new
        T0, T1 = T1, T_new
    return None

def newton_raphson(T0: float, R_specified: float, Gf: np.ndarray, Ka: np.ndarray, Kb: np.ndarray, m: int,
                   tol: float = 1e-6, max_iter: int = 1000, epsilon: float = 1e-6) -> Optional[float]:
    T = T0
    for iteration in range(max_iter):
        f_val = f_for_R(T, R_specified, Gf, Ka, Kb, m)
        # numeric derivative
        f_prime = (f_for_R(T + epsilon, R_specified, Gf, Ka, Kb, m) - f_val) / epsilon
        if abs(f_val) < tol:
            return T
        if abs(f_prime) < 1e-12:
            return None
        T = T - f_val / f_prime
    return None

def _is_scalar(x) -> bool:
    """True for a plain number (int/float/numpy scalar), False for sequences/arrays."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return False
    return np.isscalar(x) or isinstance(x, (int, float, np.number))


def _broadcast_param(value, m: int, name: str):
    """Coerce a scalar or sequence to a length-m float numpy array.

    Scalars broadcast to ``m`` identical compartments; sequences must already be
    length ``m`` or a ``ValueError`` is raised.
    """
    if _is_scalar(value):
        return np.full(m, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:  # 0-d array from np.asarray(scalar) edge case
        return np.full(m, float(arr))
    if len(arr) != m:
        raise ValueError(
            f"{name} has length {len(arr)} but Gf has {m} compartments."
        )
    return arr


def simulate_retention_times(
    Gf: Union[float, Sequence[float]],
    Ka: Union[float, Sequence[float]],
    Kb: Union[float, Sequence[float]],
    R_values: Sequence[float] = (2, 3, 10),
    m: Optional[int] = None,
    T0: float = 50.0,
    T1: float = 100.0,
) -> pd.DataFrame:
    """
    Simulate the Total Hydraulic Retention Time (THRT) T for each specified
    treatment-efficiency ratio R, using both Newton-Raphson and Secant methods
    over the m-compartment Chambers-in-Series (CSTR) reactor model.

    Parameters
    ----------
    Gf, Ka, Kb : scalar or per-compartment sequence
        Shear velocity and the fitted aggregation/breakage coefficients. When
        scalars are passed they are broadcast across all ``m`` compartments
        (single-shear design). When sequences/arrays are passed they declare a
        *varying-Gf* design, one value per compartment, and ``m`` is taken as
        ``len(Gf)``.
    m : int, optional
        Number of CSTR compartments. For a scalar ``Gf`` it defaults to 5. For an
        array ``Gf`` it must be ``None`` or equal to ``len(Gf)``. Note that a
        one-element array ``Gf=[18]`` yields ``m=1``, which is distinct from the
        scalar ``Gf=18`` that yields ``m=5``.

    Returns
    -------
    pd.DataFrame
        Columns ``Date, R, m, Gf, Ka, Kb, Newton_T, Newton_T_min, Secant_T,
        Secant_T_min``. For a varying-Gf design the ``Gf``/``Ka``/``Kb`` cells
        hold the per-compartment numpy array; for the scalar case they hold the
        scalar value.
    """
    # Resolve the compartment count and per-compartment arrays.
    if _is_scalar(Gf):
        if m is None:
            m = 5
        m = int(m)
        Gf_arr = np.full(m, float(Gf))
        Ka_arr = _broadcast_param(Ka, m, "Ka")
        Kb_arr = _broadcast_param(Kb, m, "Kb")
        gf_cell, ka_cell, kb_cell = float(Gf), float(Ka_arr[0]), float(Kb_arr[0])
    else:
        Gf_arr = np.asarray(Gf, dtype=float)
        if Gf_arr.ndim == 0:
            raise ValueError("Gf array must have at least one element.")
        if m is not None and int(m) != len(Gf_arr):
            raise ValueError(
                f"m={m} does not match len(Gf)={len(Gf_arr)}; "
                f"pass m=None or omit it when Gf is an array."
            )
        m = len(Gf_arr)
        Ka_arr = _broadcast_param(Ka, m, "Ka")
        Kb_arr = _broadcast_param(Kb, m, "Kb")
        gf_cell, ka_cell, kb_cell = Gf_arr, Ka_arr, Kb_arr

    rows = []
    for R_specified in R_values:
        T_newton = newton_raphson(T0, R_specified, Gf_arr, Ka_arr, Kb_arr, m)
        T_secant = secant_method(T0, T1, R_specified, Gf_arr, Ka_arr, Kb_arr, m)
        rows.append({
            "Date": datetime.now().isoformat(),
            "R": R_specified,
            "m": m,
            "Gf": gf_cell,
            "Ka": ka_cell,
            "Kb": kb_cell,
            "Newton_T": T_newton,
            "Newton_T_min": (T_newton / 60.0) if T_newton else None,
            "Secant_T": T_secant,
            "Secant_T_min": (T_secant / 60.0) if T_secant else None,
        })
    return pd.DataFrame(rows)
