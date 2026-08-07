# -*- coding: utf-8 -*-
"""
floclib/segment/measures.py
===========================

Per-particle measurement from a labelled image, and fractal-dimension helper.

This replaces the hand-rolled ``for floc_props in flocs`` loop in the
reference script ``beta_MultipleGf.py`` with the vectorized
``skimage.measure.regionprops_table`` API, which returns a dict of arrays
that converts directly to a :class:`pandas.DataFrame`.

All linear measurements are scaled from pixels to physical units
(micrometre or millimetre) using ``pixels_to_um``; areas scale by its
square, matching the reference script.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_PROPS", "measure_particles", "fractal_dimension"]

# regionprops_table property names we request. ``label`` becomes Particle_num.
DEFAULT_PROPS = [
    "label",
    "area",
    "equivalent_diameter_area",
    "axis_major_length",
    "axis_minor_length",
    "perimeter",
    "eccentricity",
]

#: canonical column name for the major axis length
LONGEST_LENGTH = "longest_length"

# measure-only columns (group cols Gf/Condition/Tf/Image are inserted separately)
_MEASURE_COLS = [
    "Particle_num",
    "area",
    "equivalent_diameter_area",
    LONGEST_LENGTH,
    "axis_minor_length",
    "perimeter",
    "aspect_ratio",
    "eccentricity",
]


def measure_particles(
    label_image: np.ndarray,
    pixels_to_um: float,
    *,
    image: str = "",
    condition: str = "",
    tf=None,
    gf=None,
) -> pd.DataFrame:
    """Measure per-particle properties of a labelled image.

    Parameters
    ----------
    label_image : np.ndarray
        Integer label array (0 = background, 1..N = objects).
    pixels_to_um : float
        Pixel size in micrometre (or millimetre). Linear measurements multiply
        by this; areas by this squared.
    image, condition, tf, gf
        Optional grouping metadata prepended to every row.

    Returns
    -------
    pd.DataFrame
        One row per detected object with canonical lowercase columns:
        ``Particle_num, area, equivalent_diameter_area, longest_length,
        perimeter, aspect_ratio, eccentricity`` plus the supplied group cols
        (``Gf, Condition, Tf, Image``).
    """
    skimage = _require_skimage()
    px = float(pixels_to_um)

    table = skimage.measure.regionprops_table(
        label_image.astype(int), properties=DEFAULT_PROPS
    )
    if len(table.get("label", [])) == 0:
        df = pd.DataFrame(columns=_MEASURE_COLS)
    else:
        df = pd.DataFrame(table)

    # physical scaling (matches reference script)
    if not df.empty:
        df["area"] = df["area"] * (px ** 2)
        df["equivalent_diameter_area"] = df["equivalent_diameter_area"] * px
        df["axis_major_length"] = df["axis_major_length"] * px
        df["axis_minor_length"] = df["axis_minor_length"] * px
        perimeter = df.get("perimeter")
        if perimeter is not None:
            df["perimeter"] = perimeter * px

    # reshape canonical columns
    if not df.empty:
        df = df.rename(columns={
            "label": "Particle_num",
            "axis_major_length": LONGEST_LENGTH,
        })
        with np.errstate(divide="ignore", invalid="ignore"):
            major = df[LONGEST_LENGTH].astype(float)
            minor = df["axis_minor_length"].astype(float)
            df["aspect_ratio"] = np.where(major != 0, minor / major, 0.0)
    else:
        df = pd.DataFrame(columns=_MEASURE_COLS)

    # grouping metadata
    df.insert(0, "Image", image)
    df.insert(0, "Tf", tf)
    df.insert(0, "Condition", condition)
    df.insert(0, "Gf", gf)
    return df.reset_index(drop=True)


def _canonical_columns():
    return ["Gf", "Condition", "Tf", "Image"] + _MEASURE_COLS


def fractal_dimension(
    area_um2: "pd.Series | np.ndarray",
    longest_length_um: "pd.Series | np.ndarray",
    eps: float = 1e-6,
) -> float:
    """2D fractal dimension Df from log(area) vs log(longest_length).

    Matches the ``np.polyfit(log_length, log_area, 1)`` relation used in the
    reference script ``beta_MultipleGf.py``.  Returns ``nan`` if fewer than
    two finite, positive points are available.
    """
    a = np.asarray(area_um2, dtype=float)
    L = np.asarray(longest_length_um, dtype=float)
    mask = np.isfinite(a) & np.isfinite(L) & (a > 0) & (L > 0)
    if mask.sum() < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(L[mask] + eps), np.log(a[mask] + eps), 1)
    return float(coeffs[0])


def _require_skimage():
    try:
        import skimage
        return skimage
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "measuring particles requires scikit-image. "
            "Install it with:  pip install floclib[seg]"
        ) from e
