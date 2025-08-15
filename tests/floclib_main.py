# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:55:09 2025

@author: banko
"""

from floclib.io import load_features, build_beta
from floclib.asd import compute_beta
from floclib.fit import fit_ka_kb
from floclib.cstr import simulate_retention_times

df = load_features("examples/testing.csv")
beta_df = compute_beta(df, size_col="longest_length", folder_col="Folder",
                            method="delta", min_size=0.02, max_size=2.375, interval=0.10)
Tf_arr, Bo_B_obs, beta_with_time = build_beta(beta_df)
res = fit_ka_kb(Tf_arr, Bo_B_obs, Gf=18, run_grid_search=True, loss_for_pso="huber")
sim_df = simulate_retention_times(18, res["Ka_fit"], res["Kb_fit"])
print(res["Ka_fit"], res["Kb_fit"])
print(sim_df)



# #Example: computing Beta from CSV of features
# import pandas as pd
# import numpy as np
# from asd import compute_beta
# from fit import fit_ka_kb
# from cstr import simulate_retention_times
# from plots import plot_observed_vs_fitted

# features = pd.read_csv("P1_gf_18_120.csv")
# #beta_df = compute_beta_from_features(features, size_col="longest_length", folder_col="Folder")
# beta_df = compute_beta(
#     features,
#     size_col="longest_length",
#     folder_col="Folder",
#     method="delta",
#     min_size=0.02,
#     max_size=2.375,
#     interval=0.10,
#     midpoint_type="geom",
#     min_points_for_fit=2,
#     verbose=True
# )


#  #--Or pass the bin ranges
# bins = np.arange(0.02, 2.375 + 0.10, 0.10)
# beta_df = compute_beta(features, size_col="longest_length", folder_col="Folder", method="delta", bins=bins)
 
 
# # For one folder (Tf) choose Tf row and fit Ka/Kb
# row = beta_df.iloc[0]
# # Suppose you have mapping from Tf -> time and Beta -> Bo/B as your script did:
# # Build Tf_arr and Bo_B_obs from your original data frame (see your script)
# # Example placeholders:
# Tf_arr = beta_df.Tf *60   # numpy array of times (seconds)
# Bo_B_obs = beta_df.Beta # numpy array of observed Bo/B values
# Gf_val = 18
# Gf = Gf_val

# res = fit_ka_kb(Tf_arr, Bo_B_obs, Gf, plot=False, pso_iters=150, loss_for_pso= 'mse', run_grid_search=False)
# print("PSO best opts:", res["pso_best_opts"])
# print("PSO best score:", res["pso_best_score"])
# print("Final Ka/Kb:", res["Ka_fit"], res["Kb_fit"])

# # simulate retention times:
# df_T = simulate_retention_times(Gf, res["Ka_fit"], res["Kb_fit"], R_values=[2,3,10], m=5)
# print(df_T)
