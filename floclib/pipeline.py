# -*- coding: utf-8 -*-
"""
floclib/pipeline.py
===================

The high-level orchestrator that runs the whole flocculation-kinetics
pipeline from **images** to **Beta** to **Ka/Kb** to **THRT** in a few lines.

Folder convention
-----------------
``root/`` contains one directory per experimental *condition*; each condition
directory contains one directory per timestep (*Tf*); each Tf directory
contains the images captured at that time::

    root/
      Gf_30/            <- condition (unique ID)
        2/              <- Tf (time, sorted numerically)
          img000.tif
          img001.tif
        10/
          img000.tif
      Gf_60/
        2/ ...
        10/ ...

A "condition" is any experimental grouping — it may encode shear velocity
(``Gf_30``), an impeller speed (``Impeller_C``), a chemistry (``pH_7``), or
simply a folder label.  The shear velocity ``Gf`` is **metadata on each
condition**: supplied as a scalar (all conditions share it), a dict keyed by
condition name, or parsed from the condition name via ``condition_pattern``.
When ``Gf`` is not declared, a single implicit condition is used and Gf is
assigned at :meth:`Pipeline.fit` time — so single-Gf datasets work too.

Tf folder names are sorted **numerically / natural-sort ascending**, so
``Tf=2`` always precedes ``Tf=10`` (lexicographic sort is the bug this avoids).

Example
-------
::

    from floclib import Pipeline
    from floclib.segment import Compose, ThresholdOtsu, MedianBlur

    pipe = Pipeline.from_images("FlocsData", pixels_to_um=0.27,
            segment=Compose([MedianBlur(ksize=3), ThresholdOtsu()]),
            bins=(0.3, 2.375, 0.125), size_col="longest_length")
    particles_df, beta_df = pipe.analyze()
    fit = pipe.fit(Gf=18)
    thrt = pipe.simulate(R_values=[2, 3, 10], m=5)
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .asd import compute_beta, aggregate_floc_stats
from .io import build_beta, save_results
from .fit import fit_ka_kb
from .cstr import simulate_retention_times
from .segment import Compose, Op


__all__ = ["Pipeline"]

_DEFAULT_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _natural_key(x: Any):
    """Natural-sort key so '10' sorts after '2' (chronological Tf ordering).

    Returns a *tuple* (hashable) so it works both as a ``key=`` for
    ``sorted()`` and as a value mapped onto a pandas column for
    ``sort_values`` (lists are unhashable and break pandas factorize).
    """
    s = str(x)
    parts = re.findall(r"\d+|\D+", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p), ""))
        else:
            key.append((1, 0, p.lower()))
    return tuple(key)


def _tf_numeric(tf: Any) -> float:
    """Extract a leading number from a Tf label for sorting/plotting."""
    m = re.search(r"-?\d+\.?\d*", str(tf))
    return float(m.group()) if m else float("inf")


class Pipeline:
    """Image → Beta → Ka/Kb → THRT pipeline orchestrator."""

    def __init__(
        self,
        root: str,
        *,
        pixels_to_um: float = 1.0,
        segment: Optional[Union[Compose, Op, Sequence]] = None,
        preprocess: Optional[Union[Compose, Sequence]] = None,
        post: Optional[Union[Compose, Sequence]] = None,
        size_col: str = "longest_length",
        bins: Optional[Sequence[float]] = None,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
        interval: Optional[float] = None,
        image_ext: Sequence[str] = _DEFAULT_IMAGE_EXT,
        condition_pattern: Optional[str] = None,
        gf: Optional[Union[float, dict, "pd.Series"]] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.root = root
        self.pixels_to_um = pixels_to_um
        self.size_col = size_col
        # `bins` accepts either explicit edges (>3 values) or a 3-tuple
        # (min, max, step) matching the README CLI / reference script convention.
        self.bins, self.min_size, self.max_size, self.interval = _normalize_bins(
            bins, min_size, max_size, interval
        )
        self.image_ext = tuple(e.lower() for e in image_ext)
        self.condition_pattern = condition_pattern
        self.gf = gf
        self.seed = seed
        self.verbose = verbose

        # coerce segment/preprocess/post into Compose (or None)
        self.preprocess = _coerce_compose(preprocess, default_kind="preprocess")
        self.post = _coerce_compose(post, default_kind="post")
        # segment may be a single Op, a Compose, or a list
        self.segment = _coerce_segment(segment)

        # outputs (populated by analyze())
        self.particles_df: Optional[pd.DataFrame] = None
        self.beta_df: Optional[pd.DataFrame] = None
        self._fit_result: Optional[dict] = None
        # populated by fit_all() — one row per condition + full per-condition dicts
        self.fit_results_df: Optional[pd.DataFrame] = None
        self._all_fits: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    #  Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_images(cls, root: str, **kwargs: Any) -> "Pipeline":
        """Convenience constructor — identical to ``Pipeline(root, **kwargs)``."""
        return cls(root, **kwargs)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root: str) -> "Pipeline":
        """Rebuild a :class:`Pipeline` from a config dict (see :mod:`floclib.segment.config`)."""
        from .segment.config import config_to_compose

        bins = cfg.get("bins")
        return cls(
            root,
            preprocess=config_to_compose(cfg.get("preprocess"), default_kind="preprocess"),
            segment=config_to_compose(cfg.get("segment"), default_kind="segment"),
            post=config_to_compose(cfg.get("post"), default_kind="post"),
            pixels_to_um=cfg.get("pixels_to_um", 1.0),
            size_col=cfg.get("size_col", "longest_length"),
            image_ext=cfg.get("image_ext", list(_DEFAULT_IMAGE_EXT)),
            bins=list(bins) if bins is not None else None,
            condition_pattern=cfg.get("condition_pattern"),
            seed=cfg.get("seed"),
        )

    # ------------------------------------------------------------------ #
    #  Folder discovery + Gf resolution
    # ------------------------------------------------------------------ #
    def discover(self) -> list[tuple[str, str, str, float]]:
        """Walk root → condition → Tf → images.

        Returns a list of ``(condition, tf, image_path, gf)`` tuples, with
        conditions natural-sorted and Tf folders sorted numerically ascending.
        """
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Image root not found: {self.root}")

        condition_dirs = sorted(
            (d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))),
            key=_natural_key,
        )
        if not condition_dirs:
            raise ValueError(f"No condition folders found under {self.root}")

        gf_map = self._resolve_gf(condition_dirs)

        tasks: list[tuple[str, str, str, float]] = []
        for cond in condition_dirs:
            cond_path = os.path.join(self.root, cond)
            tf_dirs = sorted(
                (d for d in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, d))),
                key=_natural_key,
            )
            for tf in tf_dirs:
                tf_path = os.path.join(cond_path, tf)
                for fname in sorted(os.listdir(tf_path)):
                    if fname.lower().endswith(self.image_ext):
                        tasks.append((cond, tf, os.path.join(tf_path, fname), gf_map[cond]))
        if not tasks:
            raise ValueError(
                f"No images (extensions {self.image_ext}) found under {self.root}"
            )
        return tasks

    def _resolve_gf(self, condition_dirs: list[str]) -> dict[str, float]:
        """Map each condition name to its Gf value.

        Precedence: explicit ``gf`` mapping (scalar / dict / sequence) first,
        then ``condition_pattern`` parsing, then the single-implicit default
        (NaN — Gf assigned at :meth:`fit` time).
        """
        if self.condition_pattern is not None and not _gf_is_explicit(self.gf):
            pat = re.compile(self.condition_pattern)
            out = {}
            for c in condition_dirs:
                m = pat.search(c)
                if not m:
                    raise ValueError(
                        f"condition_pattern {self.condition_pattern!r} did not match "
                        f"condition name {c!r}"
                    )
                out[c] = float(m.group(1))
            return out

        if self.gf is None:
            # single implicit condition path: Gf assigned at fit() time
            return {c: float("nan") for c in condition_dirs}

        if isinstance(self.gf, (int, float)):
            g = float(self.gf)
            return {c: g for c in condition_dirs}

        if hasattr(self.gf, "items"):  # dict-like
            missing = [c for c in condition_dirs if c not in self.gf]
            if missing:
                raise ValueError(
                    f"gf dict is missing conditions: {missing}. "
                    f"Provided keys: {list(self.gf)}"
                )
            return {c: float(self.gf[c]) for c in condition_dirs}

        # Series/sequence aligned to condition_dirs — require equal length
        vals = list(self.gf)
        if len(vals) != len(condition_dirs):
            raise ValueError(
                f"gf has {len(vals)} values but {len(condition_dirs)} conditions found: "
                f"{condition_dirs}"
            )
        return {c: float(v) for c, v in zip(condition_dirs, vals)}

    # ------------------------------------------------------------------ #
    #  Image I/O (lazy)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        """Read an image as grayscale, with an imageio fallback for tricky formats."""
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        except ImportError:
            pass
        # fallback for formats cv2 rejects (some TIFFs) or when cv2 missing
        import imageio.v3 as iio  # lazy; part of [seg] extra
        img = iio.imread(path)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=-1)  # naive luminance
        return img.astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  analyze() — the big one
    # ------------------------------------------------------------------ #
    def analyze(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run segmentation + measurement → ``(particles_df, beta_df)``.

        ``particles_df`` — one row per detected floc (physical units).
        ``beta_df`` — one row per ``(Condition, Tf)`` with Beta, Df, geomeans.
        """
        tasks = self.discover()
        frames: list[pd.DataFrame] = []
        if self.verbose:
            print(f"[floclib] segmenting {len(tasks)} image(s) across "
                  f"{len(set(t[0] for t in tasks))} condition(s)...")

        for cond, tf, path, gf in tasks:
            image = self._read_image(path)
            if self.preprocess is not None:
                image = self.preprocess(image)
            if self.segment is None:
                raise ValueError(
                    "Pipeline.analyze() requires a `segment` op/Compose. "
                    "Provide one via Pipeline(segment=...) or from_images(segment=...)."
                )
            labels = self.segment(image)
            if self.post is not None:
                labels = self.post(labels)
            from .segment.measures import measure_particles
            frames.append(
                measure_particles(
                    labels, self.pixels_to_um,
                    image=os.path.basename(path), condition=cond, tf=tf, gf=gf,
                )
            )

        # drop empty (no-particle) frames before concat to avoid pandas'
        # "concatenation with empty or all-NA entries" FutureWarning and to
        # keep dtypes clean; empty frames contribute no rows anyway.
        non_empty = [f for f in frames if not f.empty]
        particles = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        # drop background-only images' empty rows cleanly
        if not particles.empty:
            particles = particles[particles["Particle_num"].notna()].reset_index(drop=True)
        self.particles_df = particles

        if particles.empty:
            if self.verbose:
                print("[floclib] no particles detected.")
            self.beta_df = pd.DataFrame()
            return particles, self.beta_df

        beta_df = self._compute_beta_df(particles)
        self.beta_df = beta_df
        if self.verbose:
            print(f"[floclib] beta_df has {len(beta_df)} rows; "
                  f"particles_df has {len(particles)} rows.")
        return particles, beta_df

    def _compute_beta_df(self, particles: pd.DataFrame) -> pd.DataFrame:
        """Per-condition Beta (grouped by Tf) + per-group Df/geomeans, merged chronologically."""
        from .segment.measures import fractal_dimension

        per_condition_frames = []
        for cond, g_cond in particles.groupby("Condition", sort=False):
            if self.bins is not None:
                beta = compute_beta(
                    g_cond,
                    size_col=self.size_col,
                    folder_col="Tf",
                    bins=self.bins,
                    min_points_for_fit=2,
                    verbose=self.verbose,
                )
            else:
                beta = compute_beta(
                    g_cond,
                    size_col=self.size_col,
                    folder_col="Tf",
                    min_size=self.min_size,
                    max_size=self.max_size,
                    interval=self.interval,
                    min_points_for_fit=2,
                    verbose=self.verbose,
                )
            if beta.empty:
                if self.verbose:
                    print(f"[floclib] no Beta computed for condition {cond!r}.")
                continue
            beta = beta.rename(columns={"Tf": "Tf"}).copy()
            beta["Condition"] = cond
            # carry Gf (single value for this condition)
            gf_vals = g_cond["Gf"].dropna().unique()
            beta["Gf"] = gf_vals[0] if len(gf_vals) else float("nan")
            per_condition_frames.append(beta)

        if not per_condition_frames:
            return pd.DataFrame()

        beta_df = pd.concat(per_condition_frames, ignore_index=True)

        # per-(Condition,Tf) Df + geometric/arithmetic means
        agg = aggregate_floc_stats(
            particles,
            group_cols=("Condition", "Tf"),
            size_col=self.size_col,
            area_col="area",
            compute_df=True,
        )

        # merge on (Condition, Tf)
        agg_cols = agg.columns.tolist()
        beta_df = beta_df.merge(agg, on=["Condition", "Tf"], how="left")

        # sort chronologically: (Condition natural, Tf numeric ascending)
        beta_df["_cond_key"] = beta_df["Condition"].map(_natural_key)
        beta_df["_tf_num"] = beta_df["Tf"].map(_tf_numeric)
        beta_df = beta_df.sort_values(["_cond_key", "_tf_num"]).drop(columns=["_cond_key", "_tf_num"])
        return beta_df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  fit() / simulate() — downstream chain (reuse existing functions)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        Gf: Optional[float] = None,
        *,
        condition: Optional[str] = None,
        seed: Optional[int] = None,
        time_multiplier: float = 60.0,
        plot: bool = False,
        **fit_kwargs: Any,
    ) -> dict[str, Any]:
        """Fit Ka/Kb for a condition's beta series using :func:`fit_ka_kb`.

        ``Gf`` overrides the condition's recorded shear velocity.  If
        ``condition`` is None, the first (or only) condition is used.
        """
        if self.beta_df is None:
            raise RuntimeError("Call analyze() before fit().")
        if self.beta_df.empty:
            raise RuntimeError("beta_df is empty — no Beta values to fit (check segmentation/binning).")

        if condition is None:
            conditions = list(self.beta_df["Condition"].unique())
            condition = conditions[0]
        sub = self.beta_df[self.beta_df["Condition"] == condition].copy()
        if sub.empty:
            raise ValueError(f"No Beta rows for condition {condition!r}.")

        gf_val = float(Gf) if Gf is not None else float(sub["Gf"].iloc[0])
        if not np.isfinite(gf_val):
            raise ValueError(
                f"Gf is not set for condition {condition!r}. "
                f"Pass Gf=... to fit(), or declare gf when building the Pipeline."
            )

        Tf_arr, Bo_B_obs, _ = build_beta(
            sub, tf_col="Tf", beta_col="Beta", time_multiplier=time_multiplier
        )
        used_seed = seed if seed is not None else self.seed
        res = fit_ka_kb(
            Tf_arr, Bo_B_obs, gf_val,
            plot=plot,
            seed=used_seed,
            **fit_kwargs,
        )
        res["Condition"] = condition
        res["Gf"] = gf_val
        self._fit_result = res
        return res

    # ------------------------------------------------------------------ #
    #  fit_all() — per-condition fit table (one row per condition)
    # ------------------------------------------------------------------ #
    # columns reported in the per-condition fit table (flat view)
    _FIT_TABLE_COLUMNS = [
        "Condition", "Gf", "Ka", "Kb", "Ka/Kb", "RMSE", "AIC", "BIC",
        "Ka_se", "Kb_se", "Ka_CI_low", "Ka_CI_high",
        "Kb_CI_low", "Kb_CI_high", "n", "seed", "pso_best_score",
        "Skipped", "Skip_Reason",
    ]

    def fit_all(
        self,
        *,
        conditions: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        time_multiplier: float = 60.0,
        plot: bool = False,
        **fit_kwargs: Any,
    ) -> pd.DataFrame:
        """Fit Ka/Kb for **every** condition, returning a per-condition fit table.

        One row per condition with ``Condition, Gf, Ka, Kb, Ka/Kb, RMSE, AIC, BIC,
        Ka_se, Kb_se, Ka_CI_low/high, Kb_CI_low/high, n, seed, pso_best_score,
        Skipped, Skip_Reason``.  Conditions whose Beta series is non-positive,
        whose Gf is NaN, or that have fewer than three Tf points are **skipped
        gracefully** (a NaN row with ``Skipped=True`` and a ``Skip_Reason``)
        rather than aborted.

        The full per-condition fit dicts (with ``Ka_fit``/``Kb_fit`` and all
        diagnostic keys) are kept in ``self._all_fits`` keyed by condition name;
        the flat table is stored in ``self.fit_results_df`` and returned.

        ``self._fit_result`` (the single-condition fit) is left untouched.
        """
        if self.beta_df is None:
            raise RuntimeError("Call analyze() before fit_all().")
        if self.beta_df.empty:
            raise RuntimeError(
                "beta_df is empty — no Beta values to fit (check segmentation/binning)."
            )

        available = list(self.beta_df["Condition"].unique())
        if conditions is None:
            cond_list = available
        else:
            cond_list = list(conditions)
            for c in cond_list:
                if c not in available:
                    raise ValueError(
                        f"Unknown condition {c!r}; available: {available}."
                    )

        used_seed = seed if seed is not None else self.seed
        rows: list[dict] = []
        self._all_fits = {}

        for c in cond_list:
            sub = self.beta_df[self.beta_df["Condition"] == c]
            gf_val = float(sub["Gf"].iloc[0]) if "Gf" in sub.columns else float("nan")

            if not np.isfinite(gf_val):
                reason = ("Gf is NaN. Declare gf or condition_pattern when building "
                          "the Pipeline, or pass Gf= per condition via fit().")
                if self.verbose:
                    print(f"[floclib] Skipping condition {c!r}: {reason}")
                rows.append(self._skipped_row(c, float("nan"), reason))
                continue

            try:
                Tf_arr, Bo_B_obs, _ = build_beta(
                    sub, tf_col="Tf", beta_col="Beta",
                    time_multiplier=time_multiplier,
                )
            except ValueError as e:
                reason = str(e)
                if self.verbose:
                    print(f"[floclib] Skipping condition {c!r}: {reason}")
                rows.append(self._skipped_row(c, gf_val, reason))
                continue

            if len(Tf_arr) < 3:
                reason = (f"only {len(Tf_arr)} Tf points "
                          f"(need >= 3 for curve_fit with 2 parameters).")
                if self.verbose:
                    print(f"[floclib] Skipping condition {c!r}: {reason}")
                rows.append(self._skipped_row(c, gf_val, reason))
                continue

            res = fit_ka_kb(
                Tf_arr, Bo_B_obs, gf_val,
                plot=plot,
                seed=used_seed,
                **fit_kwargs,
            )
            res["Condition"] = c
            res["Gf"] = gf_val
            self._all_fits[c] = res
            rows.append(self._fit_row(c, res))

        self.fit_results_df = pd.DataFrame(rows, columns=self._FIT_TABLE_COLUMNS)
        return self.fit_results_df

    @staticmethod
    def _skipped_row(condition: str, gf_val: float, reason: str) -> dict:
        return {
            "Condition": condition, "Gf": gf_val,
            "Ka": np.nan, "Kb": np.nan, "Ka/Kb": np.nan,
            "RMSE": np.nan, "AIC": np.nan, "BIC": np.nan,
            "Ka_se": np.nan, "Kb_se": np.nan,
            "Ka_CI_low": np.nan, "Ka_CI_high": np.nan,
            "Kb_CI_low": np.nan, "Kb_CI_high": np.nan,
            "n": np.nan, "seed": np.nan, "pso_best_score": np.nan,
            "Skipped": True, "Skip_Reason": reason,
        }

    def _fit_row(self, condition: str, res: dict) -> dict:
        return {
            "Condition": condition, "Gf": res["Gf"],
            "Ka": res["Ka_fit"], "Kb": res["Kb_fit"], "Ka/Kb": res["Ka/Kb"],
            "RMSE": res["RMSE"], "AIC": res["AIC"], "BIC": res["BIC"],
            "Ka_se": res["Ka_se"], "Kb_se": res["Kb_se"],
            "Ka_CI_low": res["Ka_CI_low"], "Ka_CI_high": res["Ka_CI_high"],
            "Kb_CI_low": res["Kb_CI_low"], "Kb_CI_high": res["Kb_CI_high"],
            "n": res["n"], "seed": res["seed"],
            "pso_best_score": res["pso_best_score"],
            "Skipped": False, "Skip_Reason": "",
        }

    # ------------------------------------------------------------------ #
    #  simulate() — single-condition (fit_result) OR varying-Gf (Gf=)
    # ------------------------------------------------------------------ #
    def _lookup_ka_kb_by_gf(
        self, gf_design: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Look up per-compartment Ka/Kb from ``fit_results_df`` by Gf.

        Conditions sharing the same Gf (replicates) are averaged.  Skipped rows
        and rows with NaN Ka/Kb are excluded from the mean.  Raises a clear
        ``ValueError`` when a requested Gf is absent from the fit table.
        """
        if self.fit_results_df is None:
            raise RuntimeError(
                "fit_results_df is None. Call fit_all() before simulate(Gf=...)."
            )
        df = self.fit_results_df
        usable = df[df.get("Skipped", False) == False].dropna(subset=["Ka", "Kb"])
        if usable.empty:
            raise ValueError(
                "No usable (non-skipped) fits in fit_results_df; cannot look up Ka/Kb."
            )
        available_gf = sorted(float(g) for g in usable["Gf"].unique())
        ka_out, kb_out = [], []
        for g in gf_design:
            matches = usable[np.isclose(usable["Gf"], float(g), rtol=1e-6, atol=1e-6)]
            if matches.empty:
                raise ValueError(
                    f"Gf={float(g)} not found in fit_results_df. "
                    f"Available Gf values: {available_gf}. "
                    f"Either run fit_all with conditions at Gf={float(g)}, "
                    f"or check the Gf design array."
                )
            ka_out.append(float(matches["Ka"].mean()))
            kb_out.append(float(matches["Kb"].mean()))
        return np.array(ka_out), np.array(kb_out)

    def simulate(
        self,
        fit_result: Optional[dict] = None,
        *,
        Gf: Optional[Union[float, Sequence[float]]] = None,
        R_values: Sequence[float] = (2, 3, 10),
        m: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Simulate THRT via :func:`simulate_retention_times`.

        Two paths, both backward compatible:

        * **Single condition** — pass ``fit_result=`` (a dict from :meth:`fit`
          or a per-condition dict from ``self._all_fits``).  The condition's
          ``Gf``/``Ka_fit``/``Kb_fit`` are broadcast across ``m`` compartments
          (``m`` defaults to 5).  This is the original path.

        * **Varying-Gf design** — pass ``Gf=`` as a scalar or per-compartment
          array.  Ka/Kb per compartment are looked up from :attr:`fit_results_df`
          by Gf (averaging replicates at the same Gf), so :meth:`fit_all` must
          have run first.  ``m`` is taken as ``len(Gf)`` for an array.

        Passing neither raises a ``RuntimeError``; passing both favours ``Gf``.
        """
        if Gf is not None:
            if fit_result is not None and self.verbose:
                print("[floclib] simulate(): Gf= takes precedence; fit_result ignored.")
            gf_arr = np.atleast_1d(np.asarray(Gf, dtype=float))
            ka_arr, kb_arr = self._lookup_ka_kb_by_gf(gf_arr)
            return simulate_retention_times(
                gf_arr, ka_arr, kb_arr,
                R_values=R_values, m=(len(gf_arr) if m is None else m),
                **kwargs,
            )

        fr = fit_result if fit_result is not None else self._fit_result
        if fr is None:
            raise RuntimeError(
                "Call fit_all() (and pass Gf=) or pass fit_result= to simulate()."
            )
        return simulate_retention_times(
            fr.get("Gf"), fr["Ka_fit"], fr["Kb_fit"],
            R_values=R_values, m=(5 if m is None else m), **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  persistence
    # ------------------------------------------------------------------ #
    def save_particles(self, path: str) -> None:
        if self.particles_df is None:
            raise RuntimeError("Nothing to save — call analyze() first.")
        save_results(self.particles_df, path)

    def save_beta(self, path: str) -> None:
        if self.beta_df is None:
            raise RuntimeError("Nothing to save — call analyze() first.")
        save_results(self.beta_df, path)

    def to_config(self) -> dict[str, Any]:
        from .segment.config import pipeline_to_config
        return pipeline_to_config(self)

    def save_config(self, path: str) -> None:
        from .segment.config import to_yaml, pipeline_to_config
        cfg = pipeline_to_config(self)
        if path.lower().endswith((".yaml", ".yml")):
            to_yaml(path, cfg)
        else:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        n_cond = len(set(t[0] for t in (self.discover() if os.path.isdir(self.root) else [])))
        return (f"Pipeline(root={self.root!r}, conditions={n_cond}, "
                f"size_col={self.size_col!r})")


# ---------------------------------------------------------------------- #
#  module-level helpers
# ---------------------------------------------------------------------- #
def _coerce_compose(value, *, default_kind: str) -> Optional[Compose]:
    if value is None:
        return None
    if isinstance(value, Compose):
        return value
    if isinstance(value, (list, tuple)):
        return Compose(list(value), default_kind=default_kind)
    if isinstance(value, Op):
        return Compose([value], default_kind=default_kind)
    raise TypeError(
        f"Expected Compose, Op, or list of ops; got {type(value).__name__}"
    )


def _coerce_segment(value) -> Optional[Compose]:
    if value is None:
        return None
    if isinstance(value, Compose):
        return value
    if isinstance(value, Op):
        return Compose([value], default_kind="segment")
    if isinstance(value, (list, tuple)):
        return Compose(list(value), default_kind="segment")
    raise TypeError(
        f"segment must be a Compose, Op, or list; got {type(value).__name__}"
    )


def _gf_is_explicit(gf) -> bool:
    """True when gf is a scalar, dict-like, or sequence (not None)."""
    if gf is None:
        return False
    if isinstance(gf, (int, float)):
        return True
    if hasattr(gf, "items"):
        return True
    try:
        list(gf)
        return True
    except TypeError:
        return False


def _normalize_bins(bins, min_size, max_size, interval):
    """Split a bins spec into (edges, min_size, max_size, interval).

    * ``bins`` with >3 values  -> explicit edges (passed straight to compute_beta).
    * ``bins`` with exactly 3   -> interpreted as ``(min, max, step)`` and unpacked
      into min_size/max_size/interval (compute_beta builds the edges via np.arange).
    * ``bins`` is None          -> use the supplied min_size/max_size/interval.
    """
    if bins is None:
        return None, min_size, max_size, interval
    bins = list(bins)
    if len(bins) == 3:
        return None, float(bins[0]), float(bins[1]), float(bins[2])
    return bins, None, None, None
