# Changelog

All notable changes to **floclib** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-08-07

### Added
- PyPI trove classifiers (`Development Status`, `Intended Audience`, `License`, `Operating System`, `Programming Language :: Python :: 3.12`, `Topic :: Scientific/Engineering`) so the shields.io Python-versions badge reads the supported interpreter from PyPI metadata rather than returning "missing".
- README shields.io badges for uv, pip, conda, and Ruff (with logos), alongside the existing PyPI version, Python versions, downloads, license, DOI, tests, issues, and stars badges.
- `[tool.ruff]` configuration in `pyproject.toml` (`line-length = 88`, `target-version = "py312"`).

### Changed
- README: the varying-Gf THRT feature (the headline change of the 0.3.0 series) is now the prominent, physically motivated section it warrants. A new "Why this matters" lead explains that a CSTR train can run a different velocity gradient per compartment and that each compartment must use the Ka/Kb fitted at that compartment's Gf; the quick-start example now demonstrates `fit_all` plus `simulate(Gf=[18, 18, 50], ...)`; and a Key-features bullet highlights the per-condition fitting and varying-Gf design.
- README: the image-pipeline attribution now reads "new in 0.3.1" so the documentation points at the current release.

### Fixed
- **Ka/Kb parity with the reference single-file script.** The root cause was in the image measurement step, not the fit. `measure_particles` labelled every connected region and kept all of it, whereas the reference `beta_MultipleGf.py` keeps only flocs with `perimeter * pixels_to_um > 0`. On real floc images the dropped regions are sub-resolution single-pixel specks with zero perimeter -- on this dataset, 2,616,240 of 4,600,733 detected regions (57%), and 84% of the particles in some early-Tf groups. Their `equivalent_diameter_area` (~0.305 µm) is noise that piles mass into the smallest size bin, flattening the Beta power-law slope; the shifted Bo_B curve then drives the PSO + curve_fit fit into a different region (e.g. Gf=30 landing at an interior Ka instead of the Ka=1e-3 bound the reference reaches). `measure_particles` now drops zero-perimeter regions, matching the reference; floclib then reproduces the reference Ka/Kb to six significant figures on the same dataset. The `lb` and binning changes below were necessary supporting alignments but were not sufficient on their own -- the perimeter filter is what closes the gap.
- `fit_ka_kb` default lower bound aligned to the reference: `lb=(1e-13, 1e-13)` became `lb=(1e-7, 1e-7)` (the upper bound already matched at `1e-3`). The six-orders-wider search space could let PSO reach different optima on near-bound data. Pass explicit `lb`/`ub` to `fit`/`fit_all` to override.
- `compute_beta` binning now uses `pd.cut(bins, labels, include_lowest=True)` plus `groupby(observed=False).size()`, matching the reference exactly. Previously it used `np.histogram`, whose half-open `[left, right)` intervals assign boundary values to the opposite bin from `pd.cut`'s `(left, right]`. The `include_lowest` parameter (previously declared but unused -- dead code) is now functional. Effect is negligible for continuous float sizes but guarantees byte-for-byte Beta parity with the reference.
- Flaky CI test `test_fit_simulate_chain` (and the `fit_all` tests): synthetic image seeds were derived from `hash(condition + tf)`, which Python randomizes per process via `PYTHONHASHSEED`, so the generated Beta values and fits differed on every run and occasionally produced non-positive Beta. Replaced with a deterministic per-(condition, Tf) seed fold in `tests/test_pipeline.py` and `tests/test_fit_all.py`. CI now passes deterministically across all `PYTHONHASHSEED` values.

### Behavior notes
- The `fit_ka_kb` default lower bound changed from `1e-13` to `1e-7` (consistency with the reference). Existing scripts that relied on Ka/Kb below `1e-7` should pass `lb` explicitly.

## [0.3.0] - 2026-08-07

**Headline:** varying-Gf (per-compartment velocity gradient) THRT. A flocculation CSTR
train can run a different shear in each compartment, and the aggregation/breakage
kinetics `Ka`/`Kb` depend on that local shear. 0.3.0 makes the velocity gradient a
per-compartment design input: each compartment of the THRT model uses the `Ka`/`Kb`
fitted at that compartment's Gf, instead of one pair broadcast across every tank.
This is the major change of the 0.3.0 series.

### Added
- `Pipeline.fit_all(*, conditions=None, seed=None, time_multiplier=60.0, plot=False, **fit_kwargs)` fits Ka/Kb for **every condition** independently and returns a per-condition fit table (`fit_results_df`) with one row per condition: `Condition, Gf, Ka, Kb, Ka/Kb, RMSE, AIC, BIC, Ka_se, Kb_se, Ka_CI_low, Ka_CI_high, Kb_CI_low, Kb_CI_high, n, seed, pso_best_score, Skipped, Skip_Reason`. This replaces the mathematically incorrect single-fit-for-all-conditions pattern and is the fit table that the varying-Gf simulate path looks up against.
- **Varying-Gf THRT (the headline change).** `simulate_retention_times` now accepts either scalars (broadcast across `m` compartments, the original single-shear path) or per-compartment arrays (`Gf`, `Ka`, `Kb`), with `m = len(Gf)` when an array is passed. Each compartment is solved with its own Gf/Ka/Kb, so a design like `Gf=[18, 18, 50]` uses the Gf=18 kinetics in the first two tanks and the Gf=50 kinetics in the third. The low-level `reactor_ratio_product` / `newton_raphson` / `secant_method` solvers already supported per-compartment arrays; only the public wrapper and `Pipeline.simulate` needed to expose them.
- `Pipeline.simulate(Gf=...)` looks up the per-compartment Ka/Kb from the `fit_all` table by Gf, **averaging replicates** that share a Gf, and raises a clear `ValueError` listing the available Gf values when a requested Gf is not present.
- `floclib seg --Gf-design 18,18,50` CLI argument declares a per-compartment Gf design and triggers `fit_all` plus varying-Gf THRT. New artifact `<out>_fits.parquet` (the per-condition fit table).
- README shields.io badges (PyPI version, Python versions, PyPI downloads, license, DOI, tests, issues, stars).
- `CHANGELOG.md`.
- GitHub Actions workflow `.github/workflows/tests.yml` runs the test suite on push and pull request to `main` (Python 3.12, `[seg,yaml,dev]` extras, `MPLBACKEND=Agg`).

### Changed
- `simulate_retention_times` signature: `Gf_val/Ka_fitted/Kb_fitted` (scalars) became `Gf/Ka/Kb` (scalar or sequence); `m` is now `Optional[int]` defaulting to `None` (resolved to `5` for scalar input, `len(Gf)` for array input). Fully backward compatible: positional scalar calls behave exactly as before.
- `floclib seg` now calls `fit_all` and writes `<out>_fits.parquet` instead of fitting a single condition.

### Fixed
- Secant solver call inside `simulate_retention_times` now passes `Kb` directly instead of `Ka*0 + Kb` (cosmetic; values were already identical).

### Behavior notes
- Conditions that cannot be fit (NaN Gf, non-positive Beta, or fewer than three Tf points) are kept in the table as NaN rows with `Skipped=True` and a human-readable `Skip_Reason`, so the fit table always lines up with the conditions you ran.
- A one-element array `Gf=[18]` yields `m=1`, which is distinct from the scalar `Gf=18` that yields `m=5` (broadcast across five compartments). For the original broadcast-to-5 behaviour, pass a `fit_result` to `Pipeline.simulate`.

## [0.2.0] - 2026-08-07

### Added
- **Image segmentation pipeline.** `Pipeline.from_images(...)` runs raw floc images through a composable segmentation stage, measures particles, reduces them to Beta, fits Ka/Kb, and simulates THRT. Requires the optional `[seg]` extra (scikit-image, opencv-python-headless, imageio).
- **Composable image operations.** Seventeen curated preprocessing, segmentation, and post-processing operations built on OpenCV and scikit-image, with monotonic `preprocess` -> `segment` -> `post` kind ordering enforced by `Compose`. The catalog is extended through the `@register` decorator or a plain-callable escape hatch.
- Optional extras `[seg]`, `[yaml]`, and `[all]`. Image dependencies are lazy-imported, so `import floclib.segment` succeeds without `[seg]` and only raises a clear `ImportError` (with an install hint) when an operation is actually invoked.
- PSO `seed` parameter for reproducible initial particle positions; two runs with the same `seed` yield identical `Ka_fit`/`Kb_fit`.
- Fit-quality metrics from `fit_ka_kb`: `RMSE`, `MSE`, `AIC`, `BIC`, `Ka/Kb`, `Ka_se`, `Kb_se`, and 95 percent confidence intervals `Ka_CI_low/high`, `Kb_CI_low/high`.
- `floclib seg` CLI subcommand for the end-to-end image pipeline.
- Gf metadata resolution on conditions: scalar, per-condition dict, `condition_pattern` regex, or undeclared.
- `uv` installation instructions and expanded test suite.

### Changed
- Core runtime dependencies cover the tabular pipeline only; heavy image deps moved to the `[seg]` extra so `pip install floclib` stays light.

### Fixed
- setuptools package discovery restricted to `include = ["floclib", "floclib.*"]` so a bare `floclib*` glob no longer bundles the local virtualenv directory into the wheel.