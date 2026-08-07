# -*- coding: utf-8 -*-
"""Backward-compat: the existing examples/testing.csv tabular path still works."""
import os

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(REPO_ROOT, "examples", "testing.csv")


@pytest.mark.skipif(not os.path.exists(CSV), reason="examples/testing.csv missing")
def test_tabular_csv_pipeline_still_works():
    from floclib.io import load_features, build_beta
    from floclib.asd import compute_beta
    from floclib.fit import fit_ka_kb
    from floclib.cstr import simulate_retention_times

    df = load_features(CSV)
    beta_df = compute_beta(
        df, size_col="longest_length", folder_col="Folder",
        method="delta", min_size=0.02, max_size=2.375, interval=0.10,
    )
    assert not beta_df.empty
    Tf_arr, Bo_B_obs, _ = build_beta(beta_df)
    res = fit_ka_kb(
        Tf_arr, Bo_B_obs, Gf=18,
        run_grid_search=False, pso_iters=30, loss_for_pso="mse", plot=False,
    )
    assert np.isfinite(res["Ka_fit"]) and np.isfinite(res["Kb_fit"])
    assert res["Ka_fit"] > 0
    sim = simulate_retention_times(18, res["Ka_fit"], res["Kb_fit"])
    assert len(sim) == 3
    assert "Newton_T_min" in sim.columns


def test_compute_beta_folder_col_unchanged_signature():
    """compute_beta still accepts folder_col and returns Tf/Beta/Intercept/n_points/r2."""
    from floclib.asd import compute_beta
    import inspect, pandas as pd
    sig = inspect.signature(compute_beta)
    assert "folder_col" in sig.parameters
    assert sig.parameters["size_col"].default == "longest_length"
    # quick smoke on a tiny frame
    df = pd.DataFrame({"Folder": [1, 1, 2, 2, 2],
                       "longest_length": [0.5, 0.6, 0.5, 0.6, 0.7]})
    out = compute_beta(df, size_col="longest_length", folder_col="Folder",
                      min_size=0.4, max_size=1.0, interval=0.2)
    for col in ("Tf", "Beta", "Intercept", "n_points", "r2"):
        assert col in out.columns
