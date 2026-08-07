# -*- coding: utf-8 -*-
"""End-to-end Pipeline tests on a temp image tree."""
import os
import re

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
skimage = pytest.importorskip("skimage")

# synthetic image params: px=0.1 um/px.
# Circles use a SKEWED (exponential) radius distribution — many small, few
# large — like a real floc size distribution, and the scale GROWS with Tf so
# Beta decreases over time (Bo_B increases) -> fit_ka_kb has a real signal.
# bins (0.5, 4.5, 0.5) span the resulting longest_length range.
PX = 0.1
BINS = (0.5, 4.5, 0.5)


def _write_circle_png(path, r_scale=4.0, n_target=40, seed=0):
    """Image with ~`n_target` circles whose radii follow an exponential
    distribution with mean ~`r_scale` (many small, few large).  Non-overlapping."""
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
        # crude non-overlap check on a padded bounding box
        pad = r + 1
        if occupied[cy - pad:cy + pad, cx - pad:cx + pad].any():
            attempts += 1
            continue
        cv2.circle(img, (cx, cy), r, 220, -1)
        occupied[cy - pad:cy + pad, cx - pad:cx + pad] = True
        placed += 1
        attempts += 1
    cv2.imwrite(path, img)


@pytest.fixture
def image_tree(tmp_path):
    """Build root/Cond_18/{2,10}/img.png and Cond_30/{2,10}/img.png.

    Tf=2 -> small flocs (r_scale=3); Tf=10 -> larger flocs (r_scale=8).
    """
    root = tmp_path / "FlocsData"
    tf_scales = {"2": 3.0, "10": 8.0}
    for cond in ("Cond_18", "Cond_30"):
        for tf, scale in tf_scales.items():
            d = root / cond / tf
            d.mkdir(parents=True)
            _write_circle_png(str(d / "img.png"), r_scale=scale,
                              seed=_seed_for(cond, tf))
    return str(root)


def _seed_for(cond, tf):
    """Deterministic (PYTHONHASHSEED-independent) per-(condition, Tf) seed.

    Python randomizes str hashing per process, so ``hash(...)`` would make the
    synthetic images and therefore the Beta values and fits differ on every
    run. On CI this caused test_fit_simulate_chain to intermittently raise
    "Beta contains non-positive values". This fold is stable across runs and
    platforms.
    """
    h = 0
    for ch in f"{cond}/{tf}":
        h = (h * 31 + ord(ch)) % 100000
    return h


def test_analyze_two_dataframes(image_tree):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS, size_col="longest_length",
    )
    particles_df, beta_df = pipe.analyze()
    assert {"Condition", "Tf", "Image", "Particle_num", "longest_length"}.issubset(particles_df.columns)
    assert not particles_df.empty
    for cond in ("Cond_18", "Cond_30"):
        assert cond in set(particles_df["Condition"])
    assert {"Condition", "Tf", "Beta"}.issubset(beta_df.columns)
    assert len(beta_df) == 4  # 2 conditions x 2 timesteps


def test_tf_sorted_numeric_ascending(image_tree):
    """Tf=2 must appear before Tf=10 (no lexicographic sort)."""
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS,
    )
    _, beta_df = pipe.analyze()
    for cond, g in beta_df.groupby("Condition"):
        tfs = g["Tf"].tolist()
        nums = [float(re.search(r"\d+", str(t)).group()) for t in tfs]
        assert nums == sorted(nums), f"Tf not ascending for {cond}: {tfs}"
        assert float(tfs[0]) == 2


def test_single_gf_default(image_tree):
    """No gf declared -> Gf NaN until fit() assigns one."""
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS,
    )
    particles_df, _ = pipe.analyze()
    assert set(particles_df["Condition"]) == {"Cond_18", "Cond_30"}
    assert particles_df["Gf"].isna().all()


def test_gf_dict_mapping(image_tree):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS,
        gf={"Cond_18": 18.0, "Cond_30": 30.0},
    )
    particles_df, _ = pipe.analyze()
    sub18 = particles_df[particles_df["Condition"] == "Cond_18"]
    sub30 = particles_df[particles_df["Condition"] == "Cond_30"]
    assert (sub18["Gf"] == 18.0).all()
    assert (sub30["Gf"] == 30.0).all()


def test_gf_dict_missing_condition_raises(image_tree):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu()]),
        bins=BINS,
        gf={"Cond_18": 18.0},  # missing Cond_30
    )
    with pytest.raises(ValueError, match="missing conditions"):
        pipe.analyze()


def test_condition_pattern_parses_gf(image_tree):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    root = image_tree
    os.rename(os.path.join(root, "Cond_18"), os.path.join(root, "Gf_18"))
    os.rename(os.path.join(root, "Cond_30"), os.path.join(root, "Gf_30"))
    pipe = Pipeline.from_images(
        root, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS,
        condition_pattern=r"Gf_(\d+)",
    )
    particles_df, _ = pipe.analyze()
    assert set(particles_df["Condition"]) == {"Gf_18", "Gf_30"}
    assert (particles_df[particles_df["Condition"] == "Gf_18"]["Gf"] == 18.0).all()


def test_fit_simulate_chain(image_tree):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS,
        gf={"Cond_18": 18.0, "Cond_30": 30.0},
        seed=42,
    )
    pipe.analyze()
    fit = pipe.fit(condition="Cond_18", seed=42, run_grid_search=False, pso_iters=20, plot=False)
    assert np.isfinite(fit["Ka_fit"]) and np.isfinite(fit["Kb_fit"])
    # new metrics present
    for m in ("RMSE", "AIC", "BIC", "Ka/Kb"):
        assert m in fit
    assert np.isfinite(fit["RMSE"])
    sim = pipe.simulate(fit, R_values=[2, 3, 10], m=5)
    assert len(sim) == 3
    assert "Newton_T_min" in sim.columns


def test_segments_arg_can_be_single_op(image_tree):
    from floclib import Pipeline
    from floclib.segment import ThresholdOtsu
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=ThresholdOtsu(),  # bare Op, not Compose
        bins=BINS,
    )
    particles_df, _ = pipe.analyze()
    assert not particles_df.empty


def test_callable_escape_hatch_in_segment(image_tree):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu

    def my_norm(img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        preprocess=Compose([my_norm]),
        segment=Compose([ThresholdOtsu()]),
        bins=BINS,
    )
    particles_df, _ = pipe.analyze()
    assert not particles_df.empty


def test_save_particles_and_beta(image_tree, tmp_path):
    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, RemoveSmallObjects
    pipe = Pipeline.from_images(
        image_tree, pixels_to_um=PX,
        segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=20)]),
        bins=BINS,
    )
    pipe.analyze()
    pipe.save_particles(str(tmp_path / "p.parquet"))
    pipe.save_beta(str(tmp_path / "b.parquet"))
    assert (tmp_path / "p.parquet").exists()
    assert (tmp_path / "b.parquet").exists()