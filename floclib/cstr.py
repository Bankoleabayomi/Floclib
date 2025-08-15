# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:14:02 2025

@author: banko
"""

# floclib/cstr.py
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional
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

def simulate_retention_times(
    Gf_val: float,
    Ka_fitted: float,
    Kb_fitted: float,
    R_values: Sequence[float] = (2,3,10),
    m: int = 5,
    T0: float = 50.0,
    T1: float = 100.0
) -> pd.DataFrame:
    """
    Simulate retention time T for each specified R using both Newton and Secant methods.
    Returns a DataFrame with columns ['R','m','Gf','Ka','Kb','Newton_T','Newton_T_min','Secant_T','Secant_T_min'].
    """
    Gf = np.array([Gf_val] * m)
    Ka = np.array([Ka_fitted] * m)
    Kb = np.array([Kb_fitted] * m)

    rows = []
    for R_specified in R_values:
        T_newton = newton_raphson(T0, R_specified, Gf, Ka, Kb, m)
        T_secant = secant_method(T0, T1, R_specified, Gf, Ka, Ka*0+Kb, m)  # Kb array is Kb
        rows.append({
            "Date": datetime.now().isoformat(),
            "R": R_specified,
            "m": m,
            "Gf": Gf_val,
            "Ka": Ka_fitted,
            "Kb": Kb_fitted,
            "Newton_T": T_newton,
            "Newton_T_min": (T_newton/60.0) if T_newton else None,
            "Secant_T": T_secant,
            "Secant_T_min": (T_secant/60.0) if T_secant else None
        })
    return pd.DataFrame(rows)
