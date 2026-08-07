# -*- coding: utf-8 -*-
"""Real-image trial of the floclib image pipeline on a single-shear dataset.

Expected structure:  <ROOT>/{P0_Gf_18,P1_Gf_18,P2_Gf_18}/{2,10,12,...}/Img*.tif
           (root = ROOT; conditions = P0/P1/P2; Tf = numeric folders)

Example settings: pixels_to_um = 0.01, bins = (0.02, 2.375, 0.1).
Set ROOT and OUT below to your own dataset and output locations before running.
"""
import time, os, json
import numpy as np

from floclib import Pipeline
from floclib.segment import Compose, ThresholdOtsu, MedianBlur, RemoveSmallObjects, ClearBorder

# NOTE: replace these with your own dataset and output locations.
ROOT = r"/path/to/your/FlocsData/Gf_18"   # e.g. a folder of P{0,1,2}_Gf_18/{2,10,...}/Img*.tif
OUT = r"trial_out"
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
pipe = Pipeline.from_images(
    ROOT,
    pixels_to_um=0.01,
    preprocess=Compose([MedianBlur(ksize=3)]),
    segment=Compose([ThresholdOtsu(), RemoveSmallObjects(min_size=50), ClearBorder()]),
    bins=(0.02, 2.375, 0.1),
    size_col="longest_length",
    condition_pattern=r"Gf_(\d+)",
    verbose=True,
)
print("Pipeline:", pipe)
print("Discovered tasks:", len(pipe.discover()))

particles_df, beta_df = pipe.analyze()
t1 = time.time()
print(f"\n[analyze] {t1-t0:.1f}s  particles={len(particles_df)}  beta_rows={len(beta_df)}")
print("conditions:", list(particles_df['Condition'].unique()))
print("Tf order:", list(beta_df['Tf'].unique()))
print("\nbeta_df head:")
print(beta_df.head(12).to_string())

pipe.save_particles(os.path.join(OUT, "particles_Gf18.parquet"))
pipe.save_beta(os.path.join(OUT, "beta_Gf18.parquet"))

# Fit one condition
cond = "P0_Gf_18"
print(f"\n[fit] condition={cond}")
fit = pipe.fit(condition=cond, seed=42, run_grid_search=False, pso_iters=120, plot=False)
t2 = time.time()
print(f"[fit] {t2-t1:.1f}s")
keys = ["Ka_fit", "Kb_fit", "Ka/Kb", "RMSE", "AIC", "BIC",
        "Ka_se", "Kb_se", "Ka_CI_low", "Ka_CI_high", "Kb_CI_low", "Kb_CI_high",
        "pso_best_score", "seed"]
print({k: fit.get(k) for k in keys})

sim = pipe.simulate(fit, R_values=[2, 3, 10], m=5)
print("\nTHRT simulation:")
print(sim.to_string())
sim.to_csv(os.path.join(OUT, "thrt_Gf18.csv"), index=False)

summary = {
    "condition": cond, "Gf": float(fit["Gf"]),
    "analyze_seconds": round(t1-t0, 1), "fit_seconds": round(t2-t1, 1),
    "n_particles": int(len(particles_df)), "n_beta_rows": int(len(beta_df)),
    "fit": {k: (float(fit[k]) if isinstance(fit.get(k), (int, float, np.floating)) else None)
            for k in keys},
}
with open(os.path.join(OUT, "trial_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDONE in {t2-t0:.1f}s. Outputs in {OUT}")