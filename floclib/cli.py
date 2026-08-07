# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:48:12 2025

@author: banko
"""

# floclib/cli.py
import argparse
import sys
import os
import pandas as pd
import numpy as np
from .io import load_features, validate_features, build_beta, save_results
from .asd import compute_beta
from .fit import fit_ka_kb
from .cstr import simulate_retention_times

def parse_bins_arg(bins_str: str):
    # expected "min:max:step" or comma-separated list
    if ":" in bins_str:
        parts = [float(p) for p in bins_str.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("bins must be min:max:step when using ':' format")
        mn, mx, step = parts
        return list(np.arange(mn, mx + step, step))
    else:
        parts = [float(p) for p in bins_str.split(",")]
        return parts


def _parse_op_spec(spec: str):
    """Parse an op spec like 'name[k1=v1,k2=v2]' into (name, params dict)."""
    name = spec
    params = {}
    if "[" in spec and spec.endswith("]"):
        name, rest = spec.split("[", 1)
        rest = rest[:-1]
        for kv in rest.split(","):
            if not kv:
                continue
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip()] = _coerce(v.strip())
    return name, params


def _coerce(v: str):
    # light type coercion for op-spec param strings
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _build_segment(spec: str):
    """Build a Compose from a '|'-separated op-spec string."""
    from .segment import Compose, get_op
    if not spec:
        return None
    steps = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        name, params = _parse_op_spec(part)
        steps.append(get_op(name, **params))
    return Compose(steps)


def main_seg(argv=None):
    """`floclib seg` — images -> Beta -> Ka/Kb -> THRT end-to-end."""
    parser = argparse.ArgumentParser(
        prog="floclib seg",
        description="Image segmentation -> Beta -> Ka/Kb -> THRT pipeline",
    )
    parser.add_argument("--root", required=True, help="Image root folder (Condition/Tf/images)")
    parser.add_argument("--pixels-to-um", type=float, required=True, help="Pixel size in micrometre/mm")
    parser.add_argument("--segment", required=True,
                        help="Op spec: 'name[k=v,...]|name2[...]'. e.g. 'median_blur[ksize=3]|threshold_otsu'")
    parser.add_argument("--preprocess", default=None,
                        help="Optional preprocess op spec (same format as --segment)")
    parser.add_argument("--post", default=None,
                        help="Optional post op spec (e.g. 'remove_small_objects[min_size=50]')")
    parser.add_argument("--size-col", default="longest_length", help="Particle size column")
    parser.add_argument("--bins", default=None,
                        help="Bins as 'min:max:step' or comma-separated edges")
    parser.add_argument("--Gf", type=float, default=None, help="Shear velocity (scalar; overrides condition Gf)")
    parser.add_argument("--condition-pattern", default=None,
                        help="Regex with one group to parse Gf from condition dir names, e.g. 'Gf_(\\d+)'")
    parser.add_argument("--image-ext", default=None, help="Comma-separated image extensions (default many)")
    parser.add_argument("--seed", type=int, default=None, help="PSO seed for reproducibility")
    parser.add_argument("--out", default="floclib_seg_results.json", help="Output results file")
    # fit options
    parser.add_argument("--pso-iters", type=int, default=100)
    parser.add_argument("--loss", choices=["huber", "mse"], default="mse", help="Loss used in PSO")
    parser.add_argument("--no-fit", action="store_true", help="Skip Ka/Kb fit; just segment + Beta")
    parser.add_argument("--R", default="2,3,10", help="Comma-separated R values for THRT")
    parser.add_argument("--m", type=int, default=5, help="CSTR compartments")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    from .pipeline import Pipeline

    bins = parse_bins_arg(args.bins) if args.bins else None
    image_ext = tuple(args.image_ext.split(",")) if args.image_ext else None

    pipe = Pipeline.from_images(
        args.root,
        pixels_to_um=args.pixels_to_um,
        segment=_build_segment(args.segment),
        preprocess=_build_segment(args.preprocess) if args.preprocess else None,
        post=_build_segment(args.post) if args.post else None,
        size_col=args.size_col,
        bins=bins,
        condition_pattern=args.condition_pattern,
        gf=args.Gf,
        seed=args.seed,
        verbose=args.verbose,
        **({"image_ext": image_ext} if image_ext else {}),
    )

    particles_df, beta_df = pipe.analyze()
    base, _ = os.path.splitext(args.out)
    save_results(particles_df, base + "_particles.parquet")
    save_results(beta_df, base + "_beta.parquet")

    summary = {
        "n_particles": len(particles_df),
        "n_beta_rows": len(beta_df),
        "beta": beta_df.to_dict(orient="records") if not beta_df.empty else [],
    }

    if not args.no_fit and not beta_df.empty:
        R_values = [float(r) for r in args.R.split(",")]
        fit = pipe.fit(Gf=args.Gf, seed=args.seed, pso_iters=args.pso_iters,
                       loss_for_pso=args.loss, run_grid_search=False, plot=False)
        sim_df = pipe.simulate(fit, R_values=R_values, m=args.m)
        save_results(sim_df, base + "_cstr.parquet")
        summary["fit"] = {
            "Condition": fit.get("Condition"),
            "Gf": fit.get("Gf"),
            "Ka_fit": fit.get("Ka_fit"),
            "Kb_fit": fit.get("Kb_fit"),
            "seed": fit.get("seed"),
        }
        summary["cstr"] = sim_df.to_dict(orient="records")

    save_results(summary, args.out)
    print(f"Done. Particles: {summary['n_particles']}, Beta rows: {summary['n_beta_rows']}.")
    print("Saved:", args.out, base + "_particles.parquet", base + "_beta.parquet")


def main(argv=None):
    # Dispatch `floclib seg ...` to the image pipeline; everything else is the
    # existing tabular CLI (kept verbatim for backward compatibility).
    raw = argv if argv is not None else sys.argv[1:]
    if raw and raw[0] == "seg":
        return main_seg(raw[1:])

    parser = argparse.ArgumentParser(description="Floclib CLI - feature->Beta->Ka/Kb->CSTR")
    parser.add_argument("-i", "--input", required=True, help="Feature file (csv/parquet/npy)")
    parser.add_argument("--Gf", type=float, required=True, help="Shear velocity Gf (scalar)")
    parser.add_argument("--out", default="floclib_results.json", help="Output results file (json recommended)")
    # ASD bin options (optional)
    parser.add_argument("--method", choices=["delta", "density"], default="delta",
                        help="ASD method: 'delta' (your original) or 'density' (counts/dp)")
    parser.add_argument("--min-size", type=float, default=None, help="Min particle size for bins (unit as in file)")
    parser.add_argument("--max-size", type=float, default=None, help="Max particle size for bins")
    parser.add_argument("--interval", type=float, default=None, help="Bin interval (if min/max provided)")
    parser.add_argument("--bins", type=str, default=None,
                        help="Explicit bins: 'min:max:step' or comma-separated edges (overrides min/max/interval)")
    parser.add_argument("--midpoint", choices=["geom", "mid"], default="geom", help="Bin midpoint type")
    parser.add_argument("--min-points", type=int, default=2, help="Min points per folder to fit Beta")
    # PSO / fit options
    parser.add_argument("--pso-iters", type=int, default=100)
    parser.add_argument("--pso-grid", action="store_true", help="Run PSO hyperparam grid search (slower, more robust)")
    parser.add_argument("--loss", choices=["huber", "mse"], default="mse", help="Loss used in PSO")
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--plot", action="store_true", help="Show fit plot (interactive)")
    parser.add_argument("--no-save-plots", action="store_true", help="Don't save plot files")
    args = parser.parse_args(argv)

    # load features
    features = load_features(args.input)
    ok, missing = validate_features(features, required_columns=("longest_length", "Folder"))
    if not ok:
        print(f"Input features missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # bins handling
    bins = None
    if args.bins:
        try:
            bins = parse_bins_arg(args.bins)
            bins = np.asarray(bins, dtype=float)
        except Exception as e:
            print(f"Failed to parse --bins: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        if args.min_size is not None and args.max_size is not None and args.interval is not None:
            bins = np.arange(args.min_size, args.max_size + args.interval, args.interval)

    # compute Beta (ASD)
    beta_df = compute_beta(
        features,
        size_col="longest_length",
        folder_col="Folder",
        method=args.method,
        bins=bins,
        min_size=args.min_size,
        max_size=args.max_size,
        interval=args.interval,
        midpoint_type=args.midpoint,
        min_points_for_fit=args.min_points,
        verbose=True
    )
    if beta_df.empty:
        print("No Beta values computed. Exiting.", file=sys.stderr)
        sys.exit(1)

    # build Tf_arr and Bo_B_obs
    Tf_arr, Bo_B_obs, beta_with_time = build_beta(beta_df, tf_col="Tf", beta_col="Beta", time_multiplier=60.0)

    # Fit Ka and Kb
    res = fit_ka_kb(
        Tf_arr,
        Bo_B_obs,
        args.Gf,
        pso_iters=args.pso_iters,
        loss_for_pso=args.loss,
        huber_delta=args.huber_delta,
        run_grid_search=args.pso_grid,
        plot=args.plot,
        plot_title=f"Fit Gf={args.Gf}"
    )

    # Simulate retention times
    sim_df = simulate_retention_times(args.Gf, res["Ka_fit"], res["Kb_fit"], R_values=[2,3,10], m=5)

    # Save outputs: write a results JSON and a summary parquet
    summary = {
        "fit": {
            "Ka_pso_init": res.get("Ka_pso_init"),
            "Kb_pso_init": res.get("Kb_pso_init"),
            "Ka_fit": res.get("Ka_fit"),
            "Kb_fit": res.get("Kb_fit"),
            "pso_best_score": res.get("pso_best_score"),
            "pso_best_opts": res.get("pso_best_opts")
        },
        "beta_summary": beta_with_time.to_dict(orient="records"),
        "sim_retention": sim_df.to_dict(orient="records")
    }

    out_path = args.out
    save_results(summary, out_path)
    # also write a machine-friendly parquet summary
    base, _ = os.path.splitext(out_path)
    save_results(pd.DataFrame(summary["beta_summary"]), base + "_beta.parquet")
    save_results(sim_df, base + "_cstr.parquet")

    print("Done. Results saved to:", out_path)
    print("Ka_fit, Kb_fit:", res["Ka_fit"], res["Kb_fit"])

if __name__ == "__main__":
    main()
