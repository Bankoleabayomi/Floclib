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


def main(argv=None):
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
