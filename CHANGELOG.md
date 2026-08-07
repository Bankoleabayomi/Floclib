# Changelog

All notable changes to **floclib** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-07

### Added
- `Pipeline.fit_all(*, conditions=None, seed=None, time_multiplier=60.0, plot=False, **fit_kwargs)` fits Ka/Kb for **every condition** independently and returns a per-condition fit table (`fit_results_df`) with one row per condition: `Condition, Gf, Ka, Kb, Ka/Kb, RMSE, AIC, BIC, Ka_se, Kb_se, Ka_CI_low, Ka_CI_high, Kb_CI_low, Kb_CI_high, n, seed, pso_best_score, Skipped, Skip_Reason`. This replaces the mathematically incorrect single-fit-for-all-conditions pattern.
- Varying-Gf THRT: `simulate_retention_times` now accepts either scalars (broadcast across `m` compartments, the original single-shear path) or per-compartment arrays (`Gf`, `Ka`, `Kb`), with `m = len(Gf)` when an array is passed.
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