# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:13:18 2025

@author: banko
"""


# floclib/asd.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional, Sequence, Union, Dict, Any

def compute_beta(
    features: pd.DataFrame,
    *,
    size_col: str = "longest_length",
    folder_col: str = "Folder",
    method: str = "delta",            # "delta" (previous - current) or "density" (counts/dp)
    bins: Optional[Union[Sequence[float], np.ndarray]] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    interval: Optional[float] = None,
    midpoint_type: str = "geom",      # "geom" or "mid"
    min_points_for_fit: int = 2,
    include_lowest: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compute Beta per folder from features using either:
      - method="delta": dN = previous_bin_count - current_bin_count (original method)
      - method="density": number_density = counts / dp (standard)
    Parameters:
      features: DataFrame with size and folder columns.
      size_col: column name for particle size.
      folder_col: grouping column (e.g., Folder or Tf).
      bins: optional explicit bin edges (sequence). If None, min_size/max_size/interval must be provided.
      min_size, max_size, interval: used to construct bins if bins is None.
      midpoint_type: geometric mean ("geom") or midpoint ("mid")
      min_points_for_fit: minimum valid points (positive dN/dp or density) needed to perform regression.
      include_lowest: passed to pd.cut when mapping, if used.
    Returns:
      DataFrame with rows: ['Tf', 'Beta', 'n_points', 'intercept', 'r2']
    """
    df = features.copy()

    if bins is None:
        if min_size is None or max_size is None or interval is None:
            raise ValueError("If bins is not provided, you must give min_size, max_size and interval.")
        bins = np.arange(min_size, max_size + interval, interval)
        if len(bins) < 2:
            raise ValueError("Constructed bins are too short. Check min_size/max_size/interval.")
    else:
        bins = np.asarray(bins)

    # compute dp (bin widths)
    dp = np.diff(bins)  # length len(bins)-1

    # compute bin labels: geometric mean or midpoint
    if midpoint_type == "geom":
        bin_labels = np.sqrt(bins[:-1] * bins[1:])
    elif midpoint_type == "mid":
        bin_labels = 0.5 * (bins[:-1] + bins[1:])
    else:
        raise ValueError("midpoint_type must be 'geom' or 'mid'")

    results = []

    # iterate folders
    grouped = df.groupby(folder_col)
    for folder, g in grouped:
        sizes = g[size_col].dropna().astype(float)
        sizes = sizes[sizes > 0]
        if sizes.empty:
            if verbose:
                print(f"[compute_beta] folder={folder}: no valid sizes, skipping.")
            continue

        # Counts per bin, matching the reference (beta_MultipleGf.py): pd.cut with
        # include_lowest, then groupby(observed=False).size() so every bin
        # (including empty ones) appears in ascending bin order. This is
        # boundary-identical to the reference -- its intervals are (left, right]
        # with the first bin closed on the left, which np.histogram's
        # [left, right) convention is NOT (a value on an interior edge lands in
        # opposite bins). Using pd.cut makes the include_lowest parameter
        # functional and guarantees Beta parity with the single-file reference.
        binned = pd.cut(sizes, bins=bins.tolist(), labels=bin_labels.tolist(),
                        include_lowest=include_lowest)
        counts = (binned.groupby(binned, observed=False).size()
                  .reindex(bin_labels.tolist(), fill_value=0).to_numpy(dtype=float))

        if method == "delta":
            # previous - current as in your original code:
            # create array previous shifted by 1 (first previous = NaN -> treat as 0 for dN)
            prev = np.empty_like(counts)
            prev[0] = 0
            prev[1:] = counts[:-1]
            dN = prev - counts
            # map to dN/dp
            with np.errstate(divide='ignore', invalid='ignore'):
                number_density = dN / dp
        elif method == "density":
            # standard counts per bin width
            with np.errstate(divide='ignore', invalid='ignore'):
                number_density = counts / dp
        else:
            raise ValueError("method must be 'delta' or 'density'")

        # keep only bins with positive number_density and positive midpoints
        mask = (number_density > 0) & (bin_labels > 0)

        if np.sum(mask) < min_points_for_fit:
            if verbose:
                print(f"[compute_beta] folder={folder}: insufficient valid points ({np.sum(mask)}).")
            continue

        x = np.log(bin_labels[mask]).reshape(-1, 1)
        y = np.log(number_density[mask])

        # linear regression on log-log
        model = LinearRegression()
        model.fit(x, y)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        beta = -slope

        # R^2 diagnostic
        y_pred = model.predict(x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

        results.append({
            "Tf": folder,
            "Beta": float(beta),
            "Intercept": float(intercept),
            "n_points": int(np.sum(mask)),
            "r2": r2
        })

    out_df = pd.DataFrame(results)
    if out_df.empty and verbose:
        print("[compute_beta] No results produced. Check your bin settings or data.")
    return out_df


#---------Aggregate floc stats per (Condition, Tf)--------------

def aggregate_floc_stats(
    particles_df: pd.DataFrame,
    group_cols: Sequence[str] = ("Condition", "Tf"),
    size_col: str = "longest_length",
    area_col: str = "area",
    props: Sequence[str] = ("area", "equivalent_diameter_area", "longest_length",
                            "perimeter", "aspect_ratio", "eccentricity"),
    compute_df: bool = True,
) -> pd.DataFrame:
    """Per-group summary of floc properties for the image pipeline.

    For each ``(Condition, Tf)`` group this returns a row with:
      * the particle count (``n_particles``),
      * the arithmetic mean of each property (``mean_<prop>``),
      * the geometric mean of each property (``geomean_<prop>``),
      * the 2D fractal dimension ``Df`` (slope of log area vs log size).

    ``Df`` matches the relationship used in the reference script
    ``beta_MultipleGf.py`` (``np.polyfit(log length, log area, 1)``).

    Parameters
    ----------
    particles_df : DataFrame of per-particle measurements (as produced by
        :func:`floclib.segment.measure_particles`).
    group_cols : columns defining each timestep group.
    size_col, area_col : columns used for the fractal-dimension fit.
    props : columns to average.  Missing columns are skipped silently.
    compute_df : set False to skip the (sometimes noisy) Df fit.

    Returns
    -------
    pd.DataFrame with one row per group.
    """
    import numpy as np
    from .segment.measures import fractal_dimension

    df = particles_df.copy()
    available_props = [p for p in props if p in df.columns]
    if group_cols[0] not in df.columns or group_cols[1] not in df.columns:
        raise ValueError(f"particles_df must contain group columns {group_cols}")

    # natural/numeric key for Tf so output sorts chronologically (2 before 10)
    df["_tf_key"] = df[group_cols[1]].map(_natural_key)
    df = df.sort_values([group_cols[0], "_tf_key"])

    rows = []
    for key, g in df.groupby(list(group_cols), sort=False):
        row: dict[str, Any] = {
            group_cols[0]: key[0] if isinstance(key, tuple) else key,
            group_cols[1]: key[1] if isinstance(key, tuple) else key,
            "n_particles": len(g),
        }
        for p in available_props:
            vals = g[p].dropna().astype(float)
            vals = vals[vals > 0]
            row[f"mean_{p}"] = float(vals.mean()) if not vals.empty else float("nan")
            row[f"geomean_{p}"] = (
                float(np.exp(np.mean(np.log(vals)))) if not vals.empty else float("nan")
            )
        if compute_df and area_col in g.columns and size_col in g.columns:
            row["Df"] = fractal_dimension(g[area_col], g[size_col])
        else:
            row["Df"] = float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_tf_key"] = out[group_cols[1]].map(_natural_key)
        out = out.sort_values([group_cols[0], "_tf_key"]).drop(columns="_tf_key")
    return out.reset_index(drop=True)


def _natural_key(x: Any):
    """Natural-sort key so '10' sorts after '2' (chronological Tf ordering).

    Returns a *tuple* (hashable) so it works both as a ``key=`` for
    ``sorted()`` and as a value mapped onto a pandas column for
    ``sort_values`` (lists are unhashable and break pandas factorize).
    """
    import re
    s = str(x)
    parts = re.findall(r"\d+|\D+", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p), ""))
        else:
            key.append((1, 0, p.lower()))
    return tuple(key)


#---------Density only Method--------------
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression

# def compute_beta(
#     features: pd.DataFrame,
#     size_col: str = "longest_length",
#     folder_col: str = "Folder",
#     bins: np.ndarray = None,
#     min_points_for_fit: int = 3,
#     verbose: bool = False
# ) -> pd.DataFrame:
#     """
#     Compute Beta per folder from a features DataFrame.

#     - features: DataFrame with at least columns [folder_col, size_col].
#     - size_col: particle size measure (same units across data).
#     - folder_col: grouping column (e.g., Tf or folder name).
#     - bins: array of bin edges in same units as size_col. If None, auto-create.
#     Returns: DataFrame with columns ['Tf','Beta','n_points','r2'] (Tf is folder value).
#     """
#     df = features.copy()

#     # create default bins if not provided (auto scale around data)
#     if bins is None:
#         smin = df[size_col].min()
#         smax = df[size_col].max()
#         # choose a reasonable bin width - 20-30 bins
#         nbins = 30
#         bins = np.linspace(smin, smax, nbins + 1)

#     # bin midpoints (geometric mean is often used for log fits; use midpoints if zeros possible)
#     midpoints = np.sqrt(bins[:-1] * bins[1:])

#     results = []
#     # group by folder
#     for folder, g in df.groupby(folder_col):
#         # drop NaNs and non-positive sizes
#         sizes = g[size_col].dropna()
#         sizes = sizes[sizes > 0]
#         if sizes.empty:
#             if verbose:
#                 print(f"[compute_beta] Folder {folder} has no valid sizes. Skipping.")
#             continue

#         counts, _edges = np.histogram(sizes, bins=bins)
#         dp = np.diff(bins)  # widths
#         # number density per unit size: count / bin_width / total_volume_like (we use count/dp)
#         # if you prefer normalized density, divide by sizes.size
#         number_density = counts / dp

#         # keep only bins with positive number_density and positive midpoints
#         mask = (number_density > 0) & (midpoints > 0)
#         if mask.sum() < min_points_for_fit:
#             if verbose:
#                 print(f"[compute_beta] Folder {folder} has insufficient density bins ({mask.sum()}). Skipping.")
#             continue

#         x = np.log(midpoints[mask]).reshape(-1, 1)
#         y = np.log(number_density[mask])

#         model = LinearRegression()
#         model.fit(x, y)
#         slope = model.coef_[0]
#         intercept = model.intercept_
#         # in N(dp) ∝ dp^{-B} => log(N) = -B log(dp) + C => slope = -B
#         beta = -slope

#         # compute R^2 for diagnostics
#         y_pred = model.predict(x)
#         ss_res = np.sum((y - y_pred) ** 2)
#         ss_tot = np.sum((y - np.mean(y)) ** 2)
#         r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

#         results.append({
#             "Tf": folder,
#             "Beta": float(beta),
#             "Intercept": float(intercept),
#             "n_points": int(mask.sum()),
#             "r2": float(r2)
#         })
#     return pd.DataFrame(results)
