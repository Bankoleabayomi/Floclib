# -*- coding: utf-8 -*-
"""Pipeline.fit_all: per-condition fit table + varying-Gf simulate lookup.

Builds a synthetic image tree with THREE Tf points per condition (fit_all skips
conditions with fewer than three Tf points). Reuses the skewed-exponential circle
generator from test_pipeline.py so Beta decreases over time and fit_ka_kb has signal.
"""
import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
skimage = pytest.importorskip("skimage")

PX = 0.1
BINS = (0.5, 4.5, 0.5)
# three Tf points, flocs growing with time -> Beta decreases -> Bo_B increases
TF_SCALES = {"2": 3.0, "5": 5.0, "10": 8.0}

_FIT_KW = dict(run_grid_search=False, pso_iters=20, plot=False)


def _write_circle_png(path, r_scale=4.0, n_target=40, seed=0):
    """Image with ~n_target non-overlapping circles whose radii follow an
    exponential distribution with mean ~r_scale (many small, few large)."""
    img = np.zeros((320, 320), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    occupied = np.zeros(img.shape, dtype=bool)
    placed = 0
    attempts = 0
    while placed < n_target and attempts < 400:
        r = int(rng.exponential(r_scale)) + 2
        r = min(max(r, 2), 28)
        cx = int(rng.integers(r + 2, img.shape[1] - r - 2))
        cy = int(rng.integers(r + 2, img.shape[0] - r - 2))
        pad = r + 1
        if occupied[cy - pad:cy + pad, cx - pad:cx + pad].any():
            attempts += 1
            continue
        cv2.circle(img, (cx, cy), r, 220, -1)
        occupied[cy - pad:cy + pad, cx - pad:cx + pad] = True
        placed += 1
        attempts += 1
    cv2.imwrite(path, img)


def _seed_for(cond, tf):
    """Deterministic (PYTHONHASHSEED-independent) per-(condition, Tf) seed.

    Python randomizes str hashing per process, so ``hash(...)`` would make the
    synthetic images and therefore the fits differ on every pytest run. This
    fold is stable across runs and platforms.
    """
    h = 0
    for ch in f"{cond}/{tf}":
        h = (h * 31 + ord(ch)) % 100000
    return h


def _build_tree(root, conditions, gf_map):
    for cond in conditions:
        for tf, scale in TF_SCALES.items():
            d = root / cond / tf
            d.mkdir(parents=True)
            _write_circle_png(str(d / "img.png"), r_scale=scale,
                              seed=_seed_for(cond, tf))
    return str(root)


@pytest.fixture
def two_cond_tree(tmp_path):
    """root/Cond_18/{2,5,10} and Cond_30/{2,5,10}, Gf from a dict."""
    return _build_tree(tmp_path / "FlocsData", ("Cond_18", "Cond_30"),
                       {"Cond_18": 18.0, "Cond_30": 30.0})


@pytest.fixture
def replicate_tree(tmp_path):
    """Three replicate conditions all at Gf=18, each with 3 Tf points."""
    return _build_tree(tmp_path / "FlocsData", ("Rep_A", "Rep_B", "Rep_C"),
                       {"Rep_A": 18.0, "Rep_B": 18.0, "Rep_C": 18.0})


def _make_pipe(root, gf):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        root, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS, size_col="longest_length", gf=gf, seed=42,
    )
    pipe.analyze()
    return pipe


def test_fit_all_returns_per_condition_table(two_cond_tree):
    pipe = _make_pipe(two_cond_tree, {"Cond_18": 18.0, "Cond_30": 30.0})
    fits = pipe.fit_all(seed=42, **_FIT_KW)

    assert isinstance(fits, pd.DataFrame)
    assert len(fits) == 2
    expected = {"Condition", "Gf", "Ka", "Kb", "Ka/Kb", "RMSE", "AIC", "BIC",
                "Ka_se", "Kb_se", "Ka_CI_low", "Ka_CI_high",
                "Kb_CI_low", "Kb_CI_high", "n", "seed",
                "pso_best_score", "Skipped", "Skip_Reason"}
    assert expected.issubset(set(fits.columns))
    assert set(fits["Condition"]) == {"Cond_18", "Cond_30"}
    assert (fits["Skipped"] == False).all()
    assert fits["Ka"].notna().all() and fits["Kb"].notna().all()
    # fit_results_df stored on the instance too
    assert pipe.fit_results_df is fits


def test_fit_all_replicate_averaging(replicate_tree):
    pipe = _make_pipe(replicate_tree,
                      {"Rep_A": 18.0, "Rep_B": 18.0, "Rep_C": 18.0})
    fits = pipe.fit_all(seed=42, **_FIT_KW)
    assert len(fits) == 3
    assert (fits["Gf"] == 18.0).all()
    assert (fits["Skipped"] == False).all()

    sim = pipe.simulate(Gf=[18, 18], R_values=[2, 3, 10])
    assert (sim["m"] == 2).all()  # m = len(Gf) = 2

    # the per-compartment Ka is the mean of the three replicates' Ka
    ka_cell = sim["Ka"].iloc[0]
    ka_mean = fits["Ka"].mean()
    assert np.isclose(float(ka_cell[0]), float(ka_mean), rtol=1e-8), (
        f"replicate-averaged Ka {ka_cell[0]} != fit-table mean {ka_mean}"
    )


def test_fit_all_skips_non_positive_beta(two_cond_tree):
    pipe = _make_pipe(two_cond_tree, {"Cond_18": 18.0, "Cond_30": 30.0})
    # corrupt Cond_30's Beta so build_beta raises "non-positive values"
    pipe.beta_df.loc[pipe.beta_df["Condition"] == "Cond_30", "Beta"] = 0.0

    fits = pipe.fit_all(seed=42, **_FIT_KW)  # must NOT raise
    assert len(fits) == 2

    row18 = fits[fits["Condition"] == "Cond_18"].iloc[0]
    row30 = fits[fits["Condition"] == "Cond_30"].iloc[0]
    assert row18["Skipped"] == False
    assert np.isfinite(row18["Ka"])
    assert row30["Skipped"] == True
    assert "non-positive" in row30["Skip_Reason"].lower()


def test_fit_all_skips_nan_gf(two_cond_tree):
    """No gf / no condition_pattern -> Gf NaN -> every condition skipped."""
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        two_cond_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS, size_col="longest_length", seed=42,  # no gf, no pattern
    )
    pipe.analyze()
    assert pipe.beta_df["Gf"].isna().all()

    fits = pipe.fit_all(seed=42, **_FIT_KW)
    assert len(fits) == 2
    assert (fits["Skipped"] == True).all()
    assert all("nan" in str(r).lower() for r in fits["Skip_Reason"])


def test_fit_all_conditions_arg_filters(two_cond_tree):
    pipe = _make_pipe(two_cond_tree, {"Cond_18": 18.0, "Cond_30": 30.0})
    fits = pipe.fit_all(conditions=["Cond_18"], seed=42, **_FIT_KW)
    assert len(fits) == 1
    assert fits["Condition"].iloc[0] == "Cond_18"
    assert fits["Skipped"].iloc[0] == False


def test_fit_all_unknown_condition_raises(two_cond_tree):
    pipe = _make_pipe(two_cond_tree, {"Cond_18": 18.0, "Cond_30": 30.0})
    with pytest.raises(ValueError, match="Unknown condition"):
        pipe.fit_all(conditions=["Cond_99"], seed=42, **_FIT_KW)