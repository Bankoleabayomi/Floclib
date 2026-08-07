# -*- coding: utf-8 -*-
"""Varying-Gf THRT: simulate_retention_times scalar/array + Pipeline.simulate(Gf=) lookup.

These tests exercise the 0.3.0 numeric-layer changes directly (no images, no [seg]
extra required) plus the Pipeline.fit-table lookup path (fit_results_df set by hand
so the tests stay fast and image-free).

The Ka/Kb magnitudes are chosen so the THRT root sits near T0=50..100 (Newton and
Secant converge); the real fitted values in the image tests are smaller and only used
where convergence is not asserted.
"""
import numpy as np
import pandas as pd
import pytest

from floclib.cstr import (
    simulate_retention_times,
    reactor_ratio_product,
    newton_raphson,
)


def test_varying_gf_matches_manual():
    """Per-compartment arrays reach the solver unchanged: the returned Newton_T
    satisfies the ARRAY reactor product equation and matches a hand-rolled
    newton_raphson call with the same arrays; a scalar-broadcast of compartment 0
    would give a different product (so the design is genuinely per-compartment)."""
    Gf = np.array([18.0, 30.0])
    Ka = np.array([1.0e-3, 8.0e-4])
    Kb = np.array([1.0e-6, 1.2e-6])
    R_values = [2.0, 3.0]
    m = len(Gf)

    df = simulate_retention_times(Gf, Ka, Kb, R_values=R_values, T0=50.0, T1=100.0)
    assert len(df) == len(R_values)
    assert (df["m"] == m).all()

    for R, row in zip(R_values, df.itertuples(index=False)):
        T = row.Newton_T
        assert T is not None and np.isfinite(T), f"Newton did not converge for R={R}"
        # the per-compartment ARRAY product equation is satisfied at T
        assert np.isclose(reactor_ratio_product(T, Gf, Ka, Kb, m), R, rtol=1e-5), (
            f"array product != R at R={R}: {reactor_ratio_product(T, Gf, Ka, Kb, m)}"
        )
        # matches a hand-rolled newton_raphson with the same arrays
        assert np.isclose(T, newton_raphson(50.0, R, Gf, Ka, Kb, m), rtol=1e-8)

    # scalar-broadcast of compartment 0 would NOT satisfy the same equation:
    # proves the design uses the per-compartment arrays, not a single broadcast value
    T0 = df.iloc[0].Newton_T
    scalar_prod = reactor_ratio_product(
        T0, np.full(m, Gf[0]), np.full(m, Ka[0]), np.full(m, Kb[0]), m
    )
    assert not np.isclose(scalar_prod, 2.0, rtol=1e-3), (
        f"scalar-broadcast product {scalar_prod} matches R=2; arrays collapsed?"
    )


def test_scalar_backward_compat():
    """Scalar Gf/Ka/Kb broadcast across m=5 — the original single-shear path."""
    df = simulate_retention_times(18, 1.0e-3, 1.0e-6, R_values=[2, 3], m=5)
    assert len(df) == 2
    assert (df["m"] == 5).all()
    assert df["Newton_T_min"].notna().all()
    # scalar cells hold the scalar value, not an array
    assert df["Gf"].iloc[0] == 18.0
    assert np.isclose(df["Ka"].iloc[0], 1.0e-3)


def test_m_mismatch_raises():
    """Array Gf with a conflicting m raises a clear ValueError."""
    with pytest.raises(ValueError, match="does not match"):
        simulate_retention_times([18, 30], 1.0e-3, 1.0e-6, R_values=[2, 3], m=5)


def test_1element_array_m_is_1():
    """A one-element array Gf=[18] yields m=1, distinct from scalar Gf=18 (m=5)."""
    df = simulate_retention_times([18], [2.0e-5], [3.0e-7], R_values=[2])
    assert len(df) == 1
    assert df["m"].iloc[0] == 1


def test_pipeline_simulate_missing_gf_raises():
    """simulate(Gf=[50]) when 50 is not in the fit table lists the available Gf."""
    from floclib import Pipeline

    pipe = Pipeline("/nonexistent-root", bins=(0.5, 4.5, 0.5))
    # hand-build a fit table (image-free) with Gf 18 and 30 only
    pipe.fit_results_df = pd.DataFrame(
        [
            {"Condition": "Gf_18", "Gf": 18.0, "Ka": 2e-5, "Kb": 3e-7, "Skipped": False},
            {"Condition": "Gf_30", "Gf": 30.0, "Ka": 1e-5, "Kb": 5e-7, "Skipped": False},
        ]
    )
    with pytest.raises(ValueError, match=r"Available Gf values.*18.*30"):
        pipe.simulate(Gf=[50], R_values=[2, 3])


def test_pipeline_simulate_no_fit_all_raises():
    """simulate(Gf=...) before fit_all (fit_results_df is None) raises RuntimeError."""
    from floclib import Pipeline

    pipe = Pipeline("/nonexistent-root", bins=(0.5, 4.5, 0.5))
    with pytest.raises(RuntimeError, match="fit_results_df is None"):
        pipe.simulate(Gf=[18], R_values=[2, 3])