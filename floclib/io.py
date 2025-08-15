# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 17:03:22 2025

@author: banko
"""

# floclib/io.py
import os
import json
from typing import Tuple, Optional, Sequence, Dict, Any
import pandas as pd
import numpy as np

SUPPORTED_EXT = (".csv", ".parquet", ".pq", ".feather", ".npy", ".npz")

def load_features(path: str) -> pd.DataFrame:
    """
    Load feature table from CSV, parquet, feather, or numpy (.npy/.npz).
    Returns a pandas DataFrame.

    Accepts:
      - CSV (.csv)
      - Parquet (.parquet/.pq)
      - Feather (.feather)
      - NumPy saved arrays (.npy or .npz) - will convert to DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        df = _numpy_to_dataframe(arr)
    elif ext == ".npz":
        arr = np.load(path, allow_pickle=True)
        # if a single array saved with key 'arr_0' or such, try to load it; else load dict
        if len(arr.files) == 1:
            df = _numpy_to_dataframe(arr[arr.files[0]])
        else:
            # convert named arrays to columns if lengths match
            try:
                dict_data = {k: arr[k] for k in arr.files}
                df = pd.DataFrame(dict_data)
            except Exception as e:
                raise ValueError(f"Cannot interpret .npz contents as table: {e}")
    else:
        raise ValueError(f"Unsupported extension: {ext}. Supported: {SUPPORTED_EXT}")
    return df

def _numpy_to_dataframe(arr: np.ndarray) -> pd.DataFrame:
    """
    Convert numpy array (structured or 2D) to DataFrame.
    """
    if hasattr(arr, "dtype") and arr.dtype.names:
        # structured array
        return pd.DataFrame({name: arr[name] for name in arr.dtype.names})
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # single column vector -> make column 'value'
        return pd.DataFrame({"value": arr})
    elif arr.ndim == 2:
        # create column names col0, col1...
        cols = [f"col{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
    else:
        raise ValueError("Unsupported numpy array shape for conversion to DataFrame")

def validate_features(
    df: pd.DataFrame,
    required_columns: Sequence[str] = ("longest_length", "Folder")
) -> Tuple[bool, Sequence[str]]:
    """
    Quick validation to check required columns exist. Returns (ok, missing_cols).
    """
    missing = [c for c in required_columns if c not in df.columns]
    return (len(missing) == 0, missing)

def build_beta(
    beta_df: pd.DataFrame,
    tf_col: str = "Tf",
    beta_col: str = "Beta",
    time_multiplier: float = 60.0
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    From beta_df (one row per Tf with Beta), build Tf_arr (time in minutes)
    and Bo_B_obs arrays as used in fitting:
        Time = Tf * time_multiplier
        Bo_B = Beta.max()/Beta
    Returns (Tf_arr, Bo_B_obs, beta_df_with_time)
    """
    if tf_col not in beta_df.columns or beta_col not in beta_df.columns:
        raise ValueError(f"beta_df must contain columns '{tf_col}' and '{beta_col}'")

    df = beta_df.copy().sort_values(tf_col)
    df["Time"] = df[tf_col].astype(float) * float(time_multiplier)
    # guard: avoid division by zero
    if (df[beta_col] <= 0).any():
        raise ValueError("Beta contains non-positive values; cannot compute Bo_B")
    df["Bo_B"] = df[beta_col].max() / df[beta_col].astype(float)
    Tf_arr = df["Time"].values
    Bo_B_obs = df["Bo_B"].values
    return Tf_arr, Bo_B_obs, df

def save_results(obj: Any, out_path: str) -> None:
    """
    Save results to disk. Behavior depends on extension:
      - .json: saves JSON (dicts or DataFrame -> to_dict)
      - .parquet: saves DataFrame
      - .csv: saves DataFrame
      - otherwise: try JSON dump
    """
    base, ext = os.path.splitext(out_path)
    ext = ext.lower()
    if isinstance(obj, pd.DataFrame):
        if ext in (".parquet", ".pq"):
            obj.to_parquet(out_path, index=False)
        elif ext == ".csv" or ext == ".txt":
            obj.to_csv(out_path, index=False)
        elif ext == ".json":
            obj.to_json(out_path, orient="records", date_format="iso")
        else:
            # default to parquet if no known ext
            obj.to_parquet(out_path + ".parquet", index=False)
    elif isinstance(obj, dict):
        if ext == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, default=_json_fallback, indent=2)
        else:
            # save JSON by default
            with open(out_path if ext == ".json" else out_path + ".json", "w", encoding="utf-8") as f:
                json.dump(obj, f, default=_json_fallback, indent=2)
    else:
        # fallback: try json
        with open(out_path if ext == ".json" else out_path + ".json", "w", encoding="utf-8") as f:
            json.dump(obj, f, default=_json_fallback, indent=2)

def _json_fallback(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)
