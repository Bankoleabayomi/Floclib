# -*- coding: utf-8 -*-
"""PSO seed determinism for fit_ka_kb."""
import os
import numpy as np
import pytest

_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "testing.csv")
)


def _load_series():
    from floclib.io import load_features, build_beta
    from floclib.asd import compute_beta
    df = load_features(_CSV)
    beta_df = compute_beta(df, size_col="longest_length", folder_col="Folder",
                          method="delta", min_size=0.02, max_size=2.375, interval=0.10)
    Tf_arr, Bo_B_obs, _ = build_beta(beta_df)
    return Tf_arr, Bo_B_obs


@pytest.mark.skipif(not os.path.exists(_CSV), reason="examples/testing.csv missing")
def test_same_seed_gives_same_fit():
    from floclib.fit import fit_ka_kb
    Tf_arr, Bo_B_obs = _load_series()
    res1 = fit_ka_kb(Tf_arr, Bo_B_obs, Gf=18, seed=123,
                     run_grid_search=False, pso_iters=30, plot=False)
    res2 = fit_ka_kb(Tf_arr, Bo_B_obs, Gf=18, seed=123,
                     run_grid_search=False, pso_iters=30, plot=False)
    assert np.isclose(res1["Ka_fit"], res2["Ka_fit"], rtol=1e-12), (
        f"Ka_seed mismatch: {res1['Ka_fit']} vs {res2['Ka_fit']}"
    )
    assert np.isclose(res1["Kb_fit"], res2["Kb_fit"], rtol=1e-12), (
        f"Kb_seed mismatch: {res1['Kb_fit']} vs {res2['Kb_fit']}"
    )


@pytest.mark.skipif(not os.path.exists(_CSV), reason="examples/testing.csv missing")
def test_seed_recorded_in_result():
    from floclib.fit import fit_ka_kb
    Tf_arr, Bo_B_obs = _load_series()
    res = fit_ka_kb(Tf_arr, Bo_B_obs, Gf=18, seed=7,
                    run_grid_search=False, pso_iters=20, plot=False)
    assert res.get("seed") == 7
