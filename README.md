
# Floclib

[![PyPI version](https://img.shields.io/pypi/v/floclib.svg)](https://pypi.org/project/floclib/)
[![Python versions](https://img.shields.io/pypi/pyversions/floclib.svg)](https://pypi.org/project/floclib/)
[![PyPI downloads](https://img.shields.io/pypi/dm/floclib.svg)](https://pypi.org/project/floclib/#files)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jwpe.2026.110892-blue.svg)](https://doi.org/10.1016/j.jwpe.2026.110892)
[![Tests](https://img.shields.io/github/actions/workflow/status/Bankoleabayomi/Floclib/tests.yml?branch=main&label=tests)](https://github.com/Bankoleabayomi/Floclib/actions/workflows/tests.yml)
[![GitHub issues](https://img.shields.io/github/issues/Bankoleabayomi/Floclib.svg)](https://github.com/Bankoleabayomi/Floclib/issues)
[![GitHub stars](https://img.shields.io/github/stars/Bankoleabayomi/Floclib.svg)](https://github.com/Bankoleabayomi/Floclib)
[![uv](https://img.shields.io/badge/uv-supported-2665DE?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![pip](https://img.shields.io/badge/pip-supported-3775A9?logo=pypi&logoColor=white)](https://pip.pypa.io/)
[![conda](https://img.shields.io/badge/conda-supported-44A833?logo=anaconda&logoColor=white)](https://docs.conda.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22759941-blue.svg)](https://doi.org/10.5281/zenodo.22759941)

Floclib is a Python toolkit for analyzing flocculation kinetics from image and feature data. It derives the Power Law Slope (Beta) from the Aggregate Size Distribution (ASD), fits the aggregation and breakage coefficients (Ka, Kb) using Swarm Intelligence (SI) combined with non-linear least squares (NLS), and simulates the Total Hydraulic Retention Time (THRT) for an array of treatment efficiencies across Completely Stirred Tank Reactors (CSTR) in series, also known as the Chambers-in-Series model.

Floclib supports two complementary workflows:

- **Image pipeline (new in 0.3.1):** raw floc images are segmented, measured, reduced to Beta, fit for Ka and Kb, and simulated through to THRT in roughly three lines of code. This workflow requires the optional `[seg]` extra (scikit-image, OpenCV, imageio).
- **Tabular pipeline (original):** pre-computed feature tables in CSV, Parquet, or NumPy format are consumed directly and carried through the same Beta, fit, and simulate stages. This workflow needs only the lightweight core dependencies.

---

## Key features

- **Image to THRT in three lines:** a composable segmentation stage (`Compose` together with `@register` ops) feeds the existing Beta, fit, and simulate chain, so no external segmentation step is required.
- **Pluggable image operations:** seventeen curated preprocessing, segmentation, and post-processing operations (built on OpenCV and scikit-image) are shipped out of the box. The catalog is extended without limit through the `@register` decorator or a plain callable escape hatch for user-defined steps. The heavy image dependencies are an optional `[seg]` extra and are lazy-imported, so the tabular path remains light.
- **Two ASD methods:** the legacy `delta` method (dN equals previous bin count minus current bin count) and the standard `density` method (counts divided by bin width).
- **Robust and reproducible fitting:** Particle Swarm Optimization performs a global search over a configurable hyperparameter grid with an optional Huber loss, followed by Levenberg-Marquardt refinement through `scipy.curve_fit`. A `seed` parameter fixes the PSO initial particle positions so that repeated runs are reproducible. Fit quality is reported through RMSE, AIC, BIC, the Ka/Kb ratio, standard errors, and 95 percent confidence intervals.
- **Retention time solvers:** Secant and Newton-Raphson methods simulate THRT for a multi-compartment CSTR system.
- **Per-condition fitting and varying-Gf THRT (0.3.0):** `fit_all` fits Ka/Kb for every condition independently, and `simulate(Gf=[18, 18, 50], ...)` runs a different velocity gradient per CSTR compartment, looking up the Ka/Kb fitted at each compartment's Gf from the `fit_all` table.
- **CLI and Python API:** both a command-line interface and an interactive Python API are provided for scriptable and exploratory use.

---

## Image segmentation → THRT in 3 lines

Organize your images as `root/Condition/Tf/images` (3 levels): one folder per experimental condition, each containing one folder per timestep (`Tf`), each containing the images captured at that time.

```
FlocsData/
  Gf_30/            # condition (any unique ID: Gf_30, Impeller_C, pH_7, ...)
    2/              # Tf (time folder; sorted numerically, so 2 precedes 10)
      img000.tif
      img001.tif
    10/
      img000.tif
  Gf_60/
    2/ ...
    10/ ...
```

Then:

```python
from floclib import Pipeline
from floclib.segment import Compose, MedianBlur, ThresholdOtsu, RemoveSmallObjects

pipe = Pipeline.from_images("FlocsData", pixels_to_um=0.27,
        segment=Compose([MedianBlur(ksize=3), ThresholdOtsu(), RemoveSmallObjects(min_size=50)]),
        bins=(0.02, 2.375, 0.1), size_col="longest_length",
        condition_pattern=r"Gf_(\d+)")   # parse Gf from each condition folder name
particles_df, beta_df = pipe.analyze()                       # two DataFrames
fits = pipe.fit_all(seed=42)                                 # one Ka/Kb per condition
thrt = pipe.simulate(Gf=[18, 18, 50], R_values=[2, 3, 10])   # varying-Gf THRT, m = 3
```

- **`particles_df`:** one row per detected floc, with every linear measurement physically scaled by `pixels_to_um` and areas scaled by its square. Columns: `Gf, Condition, Tf, Image, Particle_num, area, equivalent_diameter_area, longest_length, axis_minor_length, perimeter, aspect_ratio, eccentricity`.
- **`beta_df`:** one row per `(Condition, Tf)` pair, containing `Beta, Intercept, n_points, r2, Df` together with the per-group arithmetic mean and geometric mean of each floc property. Rows are sorted chronologically so that `Tf=2` always precedes `Tf=10`.

### Gf metadata on conditions

`Gf` (shear velocity) is metadata on each condition, resolved in this order:
1. `gf` scalar → all conditions share that Gf;
2. `gf` dict keyed by condition name → mapped per condition (missing conditions raise);
3. `condition_pattern` regex (e.g. `r"Gf_(\d+)"`) → parsed from each condition folder name;
4. undeclared → Gf is `NaN` until you pass `Gf=` to `pipe.fit()` (single-Gf datasets).

### Compose and operations

`Compose([...])` runs its operations in order and enforces monotonic kind ordering `preprocess`, then `segment`, then `post`. A `preprocess` operation placed after a `segment` operation raises a clear `TypeError`, which prevents silent type mismatches between grayscale images and integer label arrays. A bare `Op` instance or a plain Python callable is accepted anywhere a list is expected, which serves as the callable escape hatch for user-defined steps.

| Op | Kind | Parameters | What it does |
|----|------|------------|--------------|
| `GaussianBlur` | preprocess | `ksize=5` | Gaussian blur (OpenCV) |
| `MedianBlur` | preprocess | `ksize=5` | Median blur (OpenCV) |
| `CLAHE` | preprocess | `clip_limit=2.0, tile_size=8` | Contrast-limited adaptive histogram equalization |
| `BilateralFilter` | preprocess | `d=9, sigma_color=75, sigma_space=75` | Edge-preserving bilateral filter |
| `Grayscale` | preprocess | none | Force single-channel (passthrough if already 2D) |
| `MorphOpenGray` | preprocess | `kernel_size=3` | Grayscale morphological opening |
| `MorphCloseGray` | preprocess | `kernel_size=3` | Grayscale morphological closing |
| `ThresholdManual` | segment | `value=128, invert=False` | Fixed-threshold binarization plus 8-connectivity labelling |
| `ThresholdOtsu` | segment | `invert=False` | Otsu auto-threshold plus labelling |
| `ThresholdAdaptive` | segment | `block_size=15, C=5, invert=False` | Adaptive (local Gaussian) threshold plus labelling |
| `ThresholdTriangle` | segment | `invert=False` | Triangle auto-threshold plus labelling |
| `Watershed` | segment | `min_distance=20, peak_footprint=20` | Marker-controlled watershed from distance-transform peaks |
| `RemoveSmallObjects` | post | `min_size=50` | Drop objects smaller than `min_size` pixels (skimage) |
| `FillHoles` | post | none | Fill interior holes, then re-label |
| `ClearBorder` | post | none | Remove objects touching the image border |
| `MorphOpen` | post | `kernel_size=3` | Binary morphological opening, then re-label |
| `MorphClose` | post | `kernel_size=3` | Binary morphological closing, then re-label |

```python
from floclib.segment import register, list_ops, get_op

list_ops()                 # returns ['gaussian_blur', 'median_blur', ..., 'morph_close']
get_op("threshold_otsu")   # returns an Op instance you can call on an image

@register("my_denoise", kind="preprocess")        # register your own Op subclass
class MyDenoise(Op): ...

pipe = Pipeline.from_images(..., segment=lambda img: my_label_fn(img))  # callable escape hatch
```

> Image operations lazy-import `cv2`, `scikit-image`, and `scipy.ndimage` inside their `__call__` methods. As a result, `import floclib.segment` succeeds even when the `[seg]` extra is not installed, and the operation only raises a clear `ImportError` (with an install hint) at the moment it is actually invoked.

---

## Per-condition fitting and varying-Gf THRT (0.3.0)

**Why this matters.** A CSTR train can run a different velocity gradient in each compartment, and the aggregation and breakage kinetics (Ka, Kb) depend on that local shear. Each compartment must therefore use the Ka/Kb fitted at that compartment's Gf, not one pair broadcast across every tank. Applying a single Ka/Kb to every condition is physically wrong when conditions differ in shear (Gf). Floclib 0.3.0 fixes this with two additions that sit on top of the image and tabular pipelines without changing the existing API.

### `fit_all` returns one fit per condition

`Pipeline.fit_all(...)` loops every condition in `beta_df` and fits Ka/Kb independently, returning a tidy DataFrame with one row per condition. Each row carries the full fit-quality block:

```python
fits = pipe.fit_all(seed=42)        # one row per condition
# columns: Condition, Gf, Ka, Kb, Ka/Kb, RMSE, AIC, BIC,
#          Ka_se, Kb_se, Ka_CI_low, Ka_CI_high, Kb_CI_low, Kb_CI_high,
#          n, seed, pso_best_score, Skipped, Skip_Reason
```

Conditions that cannot be fit are not dropped: they appear as NaN rows with `Skipped=True` and a human-readable `Skip_Reason` (NaN Gf, non-positive Beta, or fewer than three Tf points), so the table always lines up with the conditions you ran.

### Varying-Gf THRT from the fit table

Real CSTR designs vary the shear per compartment (for example `Gf=[18, 18, 50]`). Pass the per-compartment Gf design to `Pipeline.simulate` and floclib looks up the Ka/Kb fitted at each compartment's Gf from the `fit_all` table, averaging replicates that share a Gf. The compartment count `m` is `len(Gf)`:

```python
pipe.fit_all(seed=42)
thrt = pipe.simulate(Gf=[18, 18, 50], R_values=[2, 3, 10])   # m = 3, per-compartment Ka/Kb
```

- `Gf=18` (a scalar) declares a one-compartment design (`m=1`). For the original "one shear broadcast across five tanks" behaviour, pass a `fit_result` instead (see below).
- Replicate averaging: if three conditions were all run at Gf=18, their Ka/Kb are mean-averaged before THRT, while the fit table still reports each replicate separately.
- A Gf that is not in the fit table raises a clear `ValueError` listing the available Gf values, so a typo cannot silently produce a wrong retention time.

The low-level numeric solver `simulate_retention_times` now accepts either scalars (broadcast across `m` compartments, the original single-shear path) or per-compartment arrays (`m = len(Gf)`). This is the layer `Pipeline.simulate` builds on, and it is available directly for users who manage their own Ka/Kb arrays:

```python
from floclib.cstr import simulate_retention_times
simulate_retention_times(Gf=[18, 30, 40], Ka=[ka18, ka30, ka40],
                         Kb=[kb18, kb30, kb40], R_values=[2, 3, 10])  # m = 3
```

### Backward compatibility

The single-condition path is unchanged. `pipe.fit(condition="Gf_30", seed=42)` still returns one fit dict, and `pipe.simulate(fit_result, R_values=[2, 3, 10], m=5)` still broadcasts that condition's Gf/Ka/Kb across `m` identical compartments. Existing scripts and the three-line example above continue to work without changes.

### CLI

`floclib seg` now runs `fit_all` and writes a per-condition fit table alongside the other artifacts:

```bash
python -m floclib.cli seg --root FlocsData \
  --pixels-to-um 0.01 --segment "median_blur[ksize=3]|threshold_otsu" \
  --post "remove_small_objects[min_size=50]" --bins 0.02:2.375:0.1 \
  --condition-pattern "Gf_(\d+)" --Gf-design 18,18,50 --seed 42 --out run_seg.json
```

- `--Gf-design` : comma-separated per-compartment Gf design (for example `18,18,50`). Triggers varying-Gf THRT from the fit table. When omitted, the CLI falls back to broadcasting the first condition's fit across `--m` compartments.
- New artifact: `<out>_fits.parquet`, the per-condition fit table.

---

## Installation

> Do not install into base/system Python. Create a virtual environment first.

floclib has one optional extra:
- `pip install floclib` installs the tabular pipeline only (numpy, pandas, scipy, scikit-learn, pyswarms, matplotlib). This is the lightweight core.
- `pip install floclib[seg]` additionally installs the image-segmentation dependencies (scikit-image, opencv-python-headless, imageio). This is required for `Pipeline.from_images(...)`.

### uv (recommended)

```bash
uv venv --python 3.12 floclib_env
# Linux/macOS:  source floclib_env/bin/activate
# Windows:      floclib_env\Scripts\activate
uv pip install -e ".[seg,yaml]"      # dev install with image + YAML config extras
```

### pip

Linux / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install floclib[seg]
```
Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install floclib[seg]
```

### Conda (binary-safe fallback)

```bash
conda env create -f environment.yml
conda activate floclib
pip install -e ".[seg]"
```

### Quick verification

```bash
# tabular path imports (no [seg] deps required)
python -c "from floclib.asd import compute_beta; from floclib.fit import fit_ka_kb; print('OK')"
# image path imports (requires [seg])
python -c "from floclib import Pipeline; from floclib.segment import Compose, ThresholdOtsu; print('seg OK')"
# CLI help
python -m floclib.cli --help
```

---

## Input data format

Minimum required columns in feature table (rows = detected particles):

- `Folder`: grouping key (one folder per G or Tf).
- `longest_length`: particle size measure (units must be consistent across the dataset).

Optional useful columns: `Particle_num`, `Area_px`, `Equivalent_diameter_px`, `Perimeter_px`, `Major_axis_length_px`, `Minor_axis_length_px`, `Threshold_val`, `Timestamp`.

Supported filetypes: `.csv`, `.parquet`, `.feather`, `.npy`, `.npz`.

---

## Python API reference

Import:

```py
from floclib.io import load_features, build_beta, save_results
from floclib.asd import compute_beta
from floclib.fit import fit_ka_kb
from floclib.cstr import simulate_retention_times
```

### `compute_beta`

Calculate Beta per folder/group.

**Signature (key args):**
```py
compute_beta(
    features: pd.DataFrame,
    *,
    size_col: str = "longest_length",
    folder_col: str = "Folder",
    method: str = "delta",            # "delta" or "density"
    bins: Optional[Sequence[float]] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    interval: Optional[float] = None,
    midpoint_type: str = "geom",      # "geom" or "mid"
    min_points_for_fit: int = 3,
    include_lowest: bool = True,
    verbose: bool = False
) -> pd.DataFrame
```

**Notes:**
- `method="delta"` reproduces legacy `dN = prev − current` and fits `log(dN/dp)` vs `log(size)`.
- `method="density"` fits `log(counts/dp)` vs `log(size)`.
- Provide either `bins` or (`min_size`, `max_size`, `interval`).
- Returns DataFrame with `Tf`, `Beta`, `Intercept`, `n_points`, `r2`.

---

### `fit_ka_kb`

Fit Ka and Kb using PSO + NLS.

**Signature (key args):**
```py
fit_ka_kb(
    Tf: np.ndarray,
    Bo_B_obs: np.ndarray,
    Gf: float,
    *,
    lb: Tuple[float,float] = (1e-13, 1e-13),
    ub: Tuple[float,float] = (1e-3,  1e-3),
    param_grid: Optional[dict] = None,
    pso_iters: int = 100,
    loss_for_pso: str = "huber",   # "huber" or "mse"
    huber_delta: float = 0.01,
    run_grid_search: bool = True,
    verbose: bool = False,
    plot: bool = True,
    plot_title: Optional[str] = None,
    seed: Optional[int] = None,    # reproducible PSO initial positions
) -> Dict[str, Any]
```

**Behavior:**
- Default replicates the legacy workflow: PSO hyperparameter grid (w, c1, c2, swarm sizes) + Huber loss → select best PSO result → `curve_fit` refine.
- `seed` (new) fixes PSO initial particle positions via `np.random.default_rng(seed)`; two runs with the same `seed` yield identical `Ka_fit`/`Kb_fit`. `seed=None` keeps the original stochastic behaviour.
- Returns `Ka_pso_init`, `Kb_pso_init`, `pso_best_score`, `pso_best_opts`, `Ka_fit`, `Kb_fit`, `Bo_B_fit`, `pcov`, plus the fit-quality metrics below.

**Fit-quality metrics (returned keys):**
- `RMSE`, `MSE`: root-mean-square and mean-square error of the fit.
- `AIC`, `BIC`: Akaike and Bayesian information criteria (`n·log(MSE) + 2k` and `n·log(MSE) + k·log(n)`, with `k=2`).
- `Ka/Kb`: fitted ratio.
- `Ka_se`, `Kb_se`: standard errors derived from the `curve_fit` covariance.
- `Ka_CI_low`, `Ka_CI_high`, `Kb_CI_low`, `Kb_CI_high`: 95 percent confidence intervals (plus or minus 1.96 times the standard error).
- `n`, `k`, `seed`: sample size, parameter count, and the seed used.

**Tuning tips:**
- `run_grid_search=True` gives more robust PSO starting guesses (slower).
- `loss_for_pso="huber"` is robust to outliers; `huber_delta` controls sensitivity.

---

### `simulate_retention_times(...)`

Simulate retention times T for specified R values over the m-compartment CSTR model.

**Signature:**
```py
simulate_retention_times(
    Gf: float | Sequence[float],
    Ka: float | Sequence[float],
    Kb: float | Sequence[float],
    R_values: Sequence[float] = (2, 3, 10),
    m: Optional[int] = None,
    T0: float = 50.0,
    T1: float = 100.0
) -> pd.DataFrame
```

**Behavior:**
- `Gf`, `Ka`, `Kb` accept either scalars or per-compartment sequences.
  - Scalars are broadcast across all `m` compartments (the single-shear design). `m` defaults to `5` when `None`, preserving the original behaviour: `simulate_retention_times(18, ka, kb)` yields `m=5`.
  - Sequences declare a varying-Gf design, one value per compartment, and `m` is taken as `len(Gf)`. Passing `m` together with an array `Gf` is an error unless `m == len(Gf)`. Note that a one-element array `Gf=[18]` yields `m=1`, which is distinct from the scalar `Gf=18` that yields `m=5`.
- Uses Secant and Newton-Raphson methods to find THRT, solving the reactor product equation per compartment.
- Returns DataFrame with `Date`, `R`, `m`, `Gf`, `Ka`, `Kb`, `Newton_T`, `Newton_T_min`, `Secant_T`, `Secant_T_min`. For a varying-Gf design the `Gf`/`Ka`/`Kb` cells hold the per-compartment array; for the scalar case they hold the scalar value.

---

## IO helpers

- `load_features(path)`: loads CSV, Parquet, or NumPy arrays into a DataFrame.
- `build_beta(beta_df, tf_col="Tf", beta_col="Beta", time_multiplier=60)`: constructs the `Tf_arr` and `Bo_B_obs` arrays used for fitting.
- `save_results(obj, out_path)`: saves a DataFrame or dict to JSON, CSV, or Parquet as appropriate.

---

## CLI usage (example with tabular image features)

Run end-to-end feature → Beta → fit → simulate:
(Activate the environment first before the following).
(Windows, macOS, Linux)
```bash
python -m floclib.cli -i examples/testing.csv --Gf 18 --method delta --min-size 0.02 --max-size 2.375 --interval 0.10 --loss huber --pso-grid --pso-iters 100 --out run_results.json
```
---
Optional (Multi-line; Linux and macOS; bash, zsh)
```bash
python -m floclib.cli \
  -i examples/testing.csv \
  --Gf 18 \
  --method delta \
  --min-size 0.02 \
  --max-size 2.375 \
  --interval 0.10 \
  --loss huber \
  --pso-grid \
  --pso-iters 100 \
  --out run_results.json
```


**Key CLI options:**
- `-i, --input` : feature file path (csv/parquet/npy)
- `--Gf` : shear velocity (scalar)
- `--method` : ASD method (`delta` or `density`)
- `--bins` or (`--min-size`, `--max-size`, `--interval`) : bin specification
- `--loss` : `huber` or `mse` for PSO objective
- `--pso-grid` : toggle PSO hyperparameter grid search
- `--pso-iters` : iterations per PSO run
- `--plot` : show observed vs fitted curve

Outputs: JSON summary and companion Parquet files: `<out>_beta.parquet`, `<out>_cstr.parquet`.

---

## CLI usage: image pipeline (`floclib seg`)

Run images → Beta → Ka/Kb → THRT end-to-end (requires `floclib[seg]`):

```bash
python -m floclib.cli seg --root FlocsData \
  --pixels-to-um 0.01 \
  --segment "median_blur[ksize=3]|threshold_otsu" \
  --post "remove_small_objects[min_size=50]" \
  --bins 0.02:2.375:0.1 \
  --condition-pattern "Gf_(\d+)" \
  --seed 42 --out run_seg.json
```

**Key `seg` options:**
- `--root` : image root folder (`Condition/Tf/images`).
- `--pixels-to-um` : pixel size in µm (or mm) for physical scaling.
- `--segment` / `--preprocess` / `--post` : `|`-separated op specs, each `name[k=v,...]`.
- `--bins` : `min:max:step` or comma-separated edges.
- `--Gf` : scalar shear velocity (overrides condition Gf); or use `--condition-pattern` to parse it from folder names.
- `--Gf-design` : comma-separated per-compartment Gf design (e.g. `18,18,50`). Triggers per-condition `fit_all` plus varying-Gf THRT; overrides `--Gf` for the simulate stage. Omit it to broadcast the first condition's fit across `--m` compartments (the original single-shear behaviour).
- `--condition-pattern` : regex with one group parsed as Gf from each condition dir (e.g. `Gf_(\d+)`).
- `--seed` : reproducible PSO; `--no-fit` to stop after Beta.

Outputs: `<out>.json` summary, `<out>_particles.parquet`, `<out>_beta.parquet`, `<out>_fits.parquet` (per-condition Ka/Kb fit table), `<out>_cstr.parquet`.

---

## Output artifacts

- `<out>.json`: summary containing fit metadata, the Beta table, and simulation results.
- `<out>_beta.parquet`: Beta table with `Time` and `Bo_B`.
- `<out>_cstr.parquet`: retention time results.

---

## Notes & recommendations

- **Units consistency:** ensure particle-size units and bin edges use the same unit (mm or µm). Unit changes and bin size/intervals alter fitted slopes.
- **ASD method selection:** use `delta` to reproduce legacy behaviour; `density` is the standard alternative.
- **PSO performance:** grid search improves robustness but increases runtime. Adjust `pso_iters` and swarm sizes for faster iteration during heavy simulation.
- **Reproducibility:** PSO is stochastic. Pass `seed=` to `fit_ka_kb` / `Pipeline.fit(...)` for deterministic PSO initial positions (two runs with the same seed give identical `Ka_fit`/`Kb_fit`).
- **Error handling:** input validation checks for required columns; ensure `Folder` is declared for the corresponding column for Tf accordingly.

---

## Contributing & license

Contributions are welcome. Include tests for algorithmic changes.
License: MIT
Copyright (c) 2025 Bankoleabayomi.

---
## Citation
```bash
@article{bankole_automatic_2026,
	title = {Automatic image-based flocculation modelling and sustainable water treatment process optimisation using {FlocLib}},
	volume = {93},
	issn = {2214-7144},
	url = {https://www.sciencedirect.com/science/article/pii/S2214714426014509},
	doi = {10.1016/j.jwpe.2026.110892},
	urldate = {2026-09-13},
	journal = {Journal of Water Process Engineering},
	author = {Bankole, Abayomi O. and Moruzzi, Rodrigo and Negri, Rogério G. and Sharifi, Soroosh},
	month = nov,
	year = {2026},
	keywords = {Automatic modelling, Completely stirred tank reactor, Flocculation kinetics, Swarm intelligence},
	pages = {110892},
	file = {ScienceDirect Full Text PDF:C\:\\Users\\banko\\Zotero\\storage\\64PSLP9R\\Bankole et al. - 2026 - Automatic image-based flocculation modelling and sustainable water treatment process optimisation us.pdf:application/pdf;ScienceDirect Snapshot:C\:\\Users\\banko\\Zotero\\storage\\59GA6IIR\\S2214714426014509.html:text/html},
}

@article{bankole_novel_2025,
	title = {A novel open-source framework for automatic flocculation kinetics and retention time modelling using image analysis and swarm intelligence},
	volume = {74},
	rights = {All rights reserved},
	issn = {2214-7144},
	url = {https://www.sciencedirect.com/science/article/pii/S2214714425009432},
	doi = {10.1016/j.jwpe.2025.107871},
	pages = {107871},
	journaltitle = {Journal of Water Process Engineering},
	author = {Bankole, Abayomi O. and Moruzzi, Rodrigo and Negri, Rogério G. and Campos, Luiza C.},
	urldate = {2025-05-05},
	date = {2025-05-01},
}
```
---
## Contact

For questions, issues, or feature requests, open an issue in the project repository with a reproducible example and expected vs. actual behavior.
