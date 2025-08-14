
# Floclab

FlocLab — a lightweight, local-first Python toolkit for analyzing flocculation image feature data.  
It computes Aggregate Size Distribution (ASD) to derive the Power Law Slope (Beta), fits aggregation/breakage coefficients (Ka, Kb) using Swarm Intelligence (SI) + NLS, and simulates the Total Hydraulic Retention Time (THRT) for an array of treatment efficiency and Completely Stirred Tank Reactor (CSTR) in series - Chambers-in-Series. Floclab is designed for reproducible, offline use with feature tables exported from segmentation tools.

---

## Key features

- **Two ASD methods:** legacy `delta` (dN = previous − current) and standard `density` (counts / bin_width).
- **Robust fitting:** PSO global search (configurable grid) with optional Huber loss, followed by Levenberg–Marquardt refinement (`scipy.curve_fit`).
- **Retention time solvers:** Newton–Raphson and Secant method for multi-compartment CSTR arrays.
- **Feature-first workflow:** accepts CSV / Parquet / NumPy feature tables from an upstream floc image segmentation (version including direct image segmentation will be released soon).
- **CLI + Python API:** scriptable and interactive usage.

---

## Installation (brief)

Install runtime dependencies (Linux / macOS):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_ranges.txt
pip install floclab

```
Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_ranges.txt
pip install floclab
```
---

Using Conda (recommended)
Linux / macOS / Windows (Anaconda/Miniconda)
```bash
# from repo root (where environment.yml is)
conda env create -f environment.yml
conda activate floclab

# install your package in editable mode (dev)
pip install floclab
```
---

Quick verification (after install)
Run these to confirm core imports and CLI show help:
```bash
# basic import checks
python -c "import sys; from floclab.asd import compute_beta_from_features; print('ASD OK'); from floclab.fit import fit_ka_kb; print('FIT OK')"

# CLI help
python -m floclab.cli --help
```
If these succeed, the install is good.


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
from floclab.io import load_features, build_time_and_bobeta, save_results
from floclab.asd import compute_beta_from_features
from floclab.fit import fit_ka_kb
from floclab.cstr import simulate_retention_times
```

### `compute_beta_from_features(...)`

Calculate Beta per folder/group.

**Signature (key args):**
```py
compute_beta_from_features(
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

Fit Ka and Kb using PSO + refinement.

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
- PSO is stochastic — consider controlled seeding for repeatability if needed.

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
- Repeats the provided scalars to build arrays for `m` identical compartments.
- Uses Newton–Raphson and Secant methods to find T solving the reactor product equation.
- Returns DataFrame with `Date`, `R`, `m`, `Gf`, `Ka`, `Kb`, `Newton_T`, `Newton_T_min`, `Secant_T`, `Secant_T_min`.

---

## IO helpers

- `load_features(path)` — loads CSV / Parquet / NumPy arrays into a DataFrame.
- `build_time_and_bobeta(beta_df, tf_col="Tf", beta_col="Beta", time_multiplier=60)` — constructs `Tf_arr` and `Bo_B_obs` used for fitting.
- `save_results(obj, out_path)` — saves DataFrame/dict to JSON / CSV / Parquet as appropriate.

---

## CLI usage (example)

Run end-to-end feature → Beta → fit → simulate:

```bash
python -m floclab.cli \
  -i examples/P2_gf_50_120.csv \
  --Gf 50 \
  --method delta \
  --min-size 0.02 --max-size 2.375 --interval 0.10 \
  --loss huber --pso-grid --pso-iters 100 \
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

- **Units consistency:** ensure particle-size units and bin edges use the same unit (mm or µm). Unit changes alter fitted slopes.
- **ASD method selection:** use `delta` to reproduce legacy behaviour; `density` is the standard alternative.
- **PSO performance:** grid search improves robustness but increases runtime. Adjust `pso_iters` and swarm sizes for faster iteration during development.
- **Reproducibility:** PSO is stochastic. Add a seed option (if deterministic results are required) before large-scale production runs.
- **Error handling:** input validation checks for required columns; ensure `Folder` groups map cleanly to numeric Tf values or preprocess accordingly.

---

## Contributing & license

Contributions are welcome. Include tests for algorithmic changes. Choose an open-source license (e.g., MIT) for broad use.

---

## Contact

For questions, issues, or feature requests, open an issue in the project repository with a reproducible example and expected vs. actual behavior.
