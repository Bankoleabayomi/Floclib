
# Floclib

Floclib is a Python toolkit for analyzing flocculation image feature data.  
It computes flocculation kinetics from Aggregate Size Distribution (ASD) to derive the Power Law Slope (Beta), fits aggregation/breakage coefficients (Ka, Kb) using Swarm Intelligence (SI) + NLS, and simulates the Total Hydraulic Retention Time (THRT) for an array of treatment efficiency and Completely Stirred Tank Reactors (CSTR) in series - Chambers-in-Series. Floclib is designed for reproducible, offline use with feature tables exported from segmentation tools.

---

## Key features

- **Two ASD methods:** legacy `delta` (dN = previous − current) and standard `density` (counts / bin_width).
- **Robust fitting:** PSO global search (configurable grid) with optional Huber loss, followed by Levenberg–Marquardt refinement (`scipy.curve_fit`).
- **Retention time solvers:** Secant and Newton–Raphson methods for simulating THRT for multi-compartment CSTR system.
- **Feature-first workflow:** accepts CSV / Parquet / NumPy feature tables from an upstream floc image segmentation (version including direct image segmentation will be released soon).
- **CLI + Python API:** scriptable and interactive usage.

---

## Installation
#Note: Do not pip install into base/system Python. It is advisable to create a virtual environment using either "conda env create -f environment.yml" or "python -m venv .venv" before installing floclib. 

Install runtime dependencies (Linux / macOS):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_ranges.txt
pip install floclib

```
Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_ranges.txt
pip install floclib
```
---

Using Conda (recommended; Binary-safe)
Linux / macOS / Windows (Anaconda/Miniconda)
```bash
# from repo root (where environment.yml is)
conda env create -f environment.yml
conda activate floclib

# install your package in editable mode (dev)
pip install floclib
```
---

Quick verification (after installation)
Run these to confirm core imports and CLI show help:
```bash
# basic import checks
python -c "import sys; from floclib.asd import compute_beta; print('ASD OK'); from floclib.fit import fit_ka_kb; print('FIT OK')"

# CLI help
python -m floclib.cli --help
```
If these succeed, the install is good.

---

## Input data format

Minimum required columns in feature table (rows = detected particles):

- `Folder` — grouping key (one folder per G or Tf).
- `longest_length` — particle size measure (units must be consistent across the dataset).

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

### `compute_beta_from_features(...)`

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

### `fit_ka_kb(...)`

Fit Ka and Kb using PSO + NLS.

**Signature (key args):**
```py
fit_ka_kb(
    Tf: np.ndarray,
    Bo_B_obs: np.ndarray,
    Gf: float,
    *,
    lb: Tuple[float,float] = (1e-13, 1e-13),
    ub: Tuple[float,float] = (1e-3, 1e-3),
    param_grid: Optional[dict] = None,
    pso_iters: int = 100,
    loss_for_pso: str = "huber",   # "huber" or "mse"
    huber_delta: float = 0.01,
    run_grid_search: bool = True,
    verbose: bool = False,
    plot: bool = True,
    plot_title: Optional[str] = None
) -> Dict[str, Any]
```

**Behavior:**
- Default replicates the legacy workflow: PSO hyperparameter grid (w, c1, c2, swarm sizes) + Huber loss → select best PSO result → `curve_fit` refine.
- Returns `Ka_pso_init`, `Kb_pso_init`, `pso_best_score`, `pso_best_opts`, `Ka_fit`, `Kb_fit`, `Bo_B_fit`, `pcov`.

**Tuning tips:**
- `run_grid_search=True` gives more robust PSO starting guesses (slower).
- `loss_for_pso="huber"` is robust to outliers; `huber_delta` controls sensitivity.

---

### `simulate_retention_times(...)`

Simulate retention times T for specified R values.

**Signature:**
```py
simulate_retention_times(
    Gf_val: float,
    Ka_fitted: float,
    Kb_fitted: float,
    R_values: Sequence[float] = (2,3,10),
    m: int = 5,
    T0: float = 50.0,
    T1: float = 100.0
) -> pd.DataFrame
```

**Behavior:**
- Repeats the provided scalars to build arrays for `m` identical compartments against the reciprocal of efficiency (R).
- Uses Secant and Newton–Raphson methods to find THRT, solving the reactor product equation.
- Returns DataFrame with `Date`, `R`, `m`, `Gf`, `Ka`, `Kb`, `Newton_T`, `Newton_T_min`, `Secant_T`, `Secant_T_min`.

---

## IO helpers

- `load_features(path)` — loads CSV / Parquet / NumPy arrays into a DataFrame.
- `build_beta(beta_df, tf_col="Tf", beta_col="Beta", time_multiplier=60)` — constructs `Tf_arr` and `Bo_B_obs` used for fitting.
- `save_results(obj, out_path)` — saves DataFrame/dict to JSON / CSV / Parquet as appropriate.

---

## CLI usage (example)

Run end-to-end feature → Beta → fit → simulate:
(Activate the environment first before the following).
(Windows, macOS, Linux)
```bash
python -m floclib.cli -i examples/testing.csv --Gf 18 --method delta --min-size 0.02 --max-size 2.375 --interval 0.10 --loss huber --pso-grid --pso-iters 100 --out run_results.json
```
---
Optional (Multi-line — Linux / macOS; bash, zsh)
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

## Output artifacts

- `<out>.json` — summary (fit metadata, Beta table, simulation results).
- `<out>_beta.parquet` — Beta table with `Time` and `Bo_B`.
- `<out>_cstr.parquet` — retention time results.

---

## Notes & recommendations

- **Units consistency:** ensure particle-size units and bin edges use the same unit (mm or µm). Unit changes and bin size/intervals alter fitted slopes.
- **ASD method selection:** use `delta` to reproduce legacy behaviour; `density` is the standard alternative.
- **PSO performance:** grid search improves robustness but increases runtime. Adjust `pso_iters` and swarm sizes for faster iteration during heavy simulation.
- **Reproducibility:** PSO is stochastic. Add a seed option (if deterministic results are required) before large-scale production runs.
- **Error handling:** input validation checks for required columns; ensure `Folder` is declared for the corresponding column for Tf accordingly.

---

## Contributing & license

Contributions are welcome. Include tests for algorithmic changes.
License: MIT
Copyright (c) 2025 Bankoleabayomi.

---
## Citation
```bash
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
