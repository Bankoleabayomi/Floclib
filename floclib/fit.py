# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:13:38 2025

@author: banko
"""

# floclib/fit.py
import numpy as np
import warnings
from typing import Tuple, Dict, Any, Optional
from scipy.optimize import curve_fit
from pyswarms.single import GlobalBestPSO
import matplotlib.pyplot as plt

def _A_K(Tf, Gf, Ka, Kb):
    """Model A_K as in your original script."""
    with np.errstate(over="ignore", invalid="ignore"):
        val = ((Kb / Ka) * Gf + (1 - (Kb / Ka) * Gf) * np.exp(-Ka * Gf * Tf)) ** -1
    return val

def _huber_loss_vec(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 0.01) -> float:
    err = y_true - y_pred
    abs_err = np.abs(err)
    small_mask = abs_err < delta
    loss = np.empty_like(err, dtype=float)
    loss[small_mask] = 0.5 * (err[small_mask] ** 2)
    loss[~small_mask] = delta * (abs_err[~small_mask] - 0.5 * delta)
    return float(np.mean(loss))

def _mse_loss_vec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.mean(err ** 2))

def fit_ka_kb(
    Tf: np.ndarray,
    Bo_B_obs: np.ndarray,
    Gf: float,
    *,
    lb: Tuple[float,float] = (1e-13, 1e-13),
    ub: Tuple[float,float] = (1e-3,  1e-3),
    param_grid: Optional[dict] = None,
    pso_iters: int = 100,
    loss_for_pso: str = "huber",   # default to match your original script
    huber_delta: float = 0.01,
    run_grid_search: bool = True,  # run the w/c1/c2/s grid like your code
    verbose: bool = False,
    plot: bool = True,
    plot_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fit Ka and Kb using PSO (grid search across hyperparameters optionally) then curve_fit.
    - Tf: 1D array (time)
    - Bo_B_obs: 1D array (observed Beta0/Beta)
    - Gf: scalar
    Returns a dict with Ka/Kb (PSO init and curve_fit refined), pso diagnostics,
    and Bo_B_fit array. By default it matches your original script:
      grid-search over (w,c1,c2,s) with Huber loss in PSO, then curve_fit refine.
    """

    Tf_arr = np.asarray(Tf)
    #B_B = (Bo_B_obs.max())/Bo_B_obs
    Bo_B = np.asarray(Bo_B_obs)
    

    lb_arr = np.asarray(lb)
    ub_arr = np.asarray(ub)
    bounds_for_ps = (lb_arr, ub_arr)

    # default param grid if not provided (matches your original)
    if param_grid is None:
        param_grid = {
            'w':  [0.5, 0.7, 0.9],
            'c1': [0.2, 0.5, 0.8],
            'c2': [0.2, 0.5, 0.8],
            's':  [30, 50, 200]
        }

    # Objective used by PSO optimize call: x is (n_particles, 2)
    def pso_objective(x):
        costs = []
        for Ka_c, Kb_c in x:
            ypred = _A_K(Tf_arr, Gf, Ka_c, Kb_c)
            if loss_for_pso.lower() == "huber":
                costs.append(_huber_loss_vec(Bo_B, ypred, delta=huber_delta))
            else:
                costs.append(_mse_loss_vec(Bo_B, ypred))
        return np.array(costs)

    best_score = np.inf
    best_opts = None
    best_cost_history = None

    if run_grid_search:
        # iterate grid exactly as your script
        for w in param_grid['w']:
            for c1 in param_grid['c1']:
                for c2 in param_grid['c2']:
                    for s in param_grid['s']:
                        options = {'w': w, 'c1': c1, 'c2': c2}
                        optimizer = GlobalBestPSO(
                            n_particles=s,
                            dimensions=2,
                            options=options,
                            bounds=bounds_for_ps
                        )
                        try:
                            cost, pos = optimizer.optimize(
                                pso_objective,
                                iters=pso_iters,
                                verbose=False
                            )
                        except Exception as e:
                            if verbose:
                                print(f"PSO run failed for w={w},c1={c1},c2={c2},s={s}: {e}")
                            cost = np.inf
                            pos = np.array([np.nan, np.nan])

                        if cost < best_score:
                            best_score = cost
                            best_opts = (w, c1, c2, s, pos)
                            # try to capture optimizer cost history (if attribute exists)
                            best_cost_history = getattr(optimizer, "cost_history", None)

        if best_opts is None:
            raise RuntimeError("PSO grid search failed to produce any valid result.")
        w_best, c1_best, c2_best, s_best, p_best = best_opts
    else:
        # single PSO run with default options (choose middle grid option or user-provided param_grid)
        # pick first available options if grid provided else fallback
        w_default = param_grid['w'][0] if 'w' in param_grid else 0.7
        c1_default = param_grid['c1'][0] if 'c1' in param_grid else 0.5
        c2_default = param_grid['c2'][0] if 'c2' in param_grid else 0.5
        s_default = param_grid['s'][0] if 's' in param_grid else 50
        options = {'w': w_default, 'c1': c1_default, 'c2': c2_default}
        optimizer = GlobalBestPSO(
            n_particles=s_default,
            dimensions=2,
            options=options,
            bounds=bounds_for_ps
        )
        try:
            best_score, p_best = optimizer.optimize(pso_objective, iters=pso_iters, verbose=False)
            best_cost_history = getattr(optimizer, "cost_history", None)
            w_best, c1_best, c2_best, s_best = (w_default, c1_default, c2_default, s_default)
        except Exception as e:
            raise RuntimeError(f"PSO failed: {e}")

    # Unpack best PSO position
    Ka_init, Kb_init = float(p_best[0]), float(p_best[1])

    # Use PSO best as initial guess for curve_fit
    try:
        popt, pcov = curve_fit(
            lambda t, Ka, Kb: _A_K(t, Gf, Ka, Kb),
            Tf_arr,
            Bo_B,
            p0=[Ka_init, Kb_init],
            bounds=(lb_arr, ub_arr),
            maxfev=10000
        )
        Ka_fit, Kb_fit = float(popt[0]), float(popt[1])
    except Exception as e:
        # if curve_fit fails, fall back to PSO initial positions
        warnings.warn(f"curve_fit refinement failed: {e}")
        Ka_fit, Kb_fit = Ka_init, Kb_init
        pcov = None

    Bo_B_fit = _A_K(Tf_arr, Gf, Ka_fit, Kb_fit)
    T_m = Tf_arr/60

    # Plot (replicates your final plotting block)
    if plot:
        try:
            fig, ax = plt.subplots(figsize=(8,5), dpi=150)
            ax.plot(T_m, Bo_B, 'bo', label='Observed')
            ax.plot(T_m, Bo_B_fit, 'r-', label='Fitted')
            ax.set_xlabel('Time (Minutes)')
            ax.set_ylabel('β₀/β')
            if plot_title:
                ax.set_title(plot_title)
            else:
                ax.set_title('Flocculation Kinetic- β')
            ax.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            warnings.warn(f"Plotting failed: {e}")

    result = {
        "Ka_pso_init": Ka_init,
        "Kb_pso_init": Kb_init,
        "pso_best_score": float(best_score),
        "pso_best_opts": {"w": w_best, "c1": c1_best, "c2": c2_best, "swarm": s_best},
        "pso_cost_history_best_run": best_cost_history,
        "Ka_fit": Ka_fit,
        "Kb_fit": Kb_fit,
        "Bo_B_fit": Bo_B_fit,
        "pcov": pcov
    }
    return result
