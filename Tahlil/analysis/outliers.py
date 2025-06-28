"""Outlier detection helpers used by the spreadsheet application.

This module centralises the algorithms so that GUI and models remain thin.
It currently supports three methods:

* z   – z-score > 3
* iqr – Tukey IQR rule
* knn – K-nearest-neighbours (k=5) mean distance > mean+3·std
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

_Method = Literal["z", "iqr", "knn"]


def mask_outliers(series: pd.Series, method: _Method = "z", *, k: int = 5) -> pd.Series:
    """Return a boolean mask marking outliers in *series*.

    NaNs are ignored. The index of the returned Boolean Series matches *series*.
    """
    series = pd.to_numeric(series, errors="coerce")
    if series.isna().all():
        return pd.Series(False, index=series.index)

    if method == "z":
        mu = series.mean(); sigma = series.std()
        if sigma == 0 or math.isnan(sigma):
            return pd.Series(False, index=series.index)
        return (series - mu).abs() > 3 * sigma

    if method == "iqr":
        q1 = series.quantile(0.25); q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr; upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)

    if method == "knn":
        arr = series.dropna().to_numpy(float)
        if len(arr) < k + 1:
            # not enough data
            return pd.Series(False, index=series.index)
        if len(arr) <= 2000:
            diff = np.abs(arr[:, None] - arr[None, :])
            part = np.partition(diff, k, axis=1)[:, 1 : k + 1]
            dists = part.mean(axis=1)
        else:
            dists = []
            for v in arr:
                nearest = np.partition(np.abs(arr - v), k + 1)[: k + 1]
                dists.append(nearest[1:].mean())
            dists = np.array(dists)
        mu = dists.mean(); sigma = dists.std()
        thresh = mu + 3 * sigma
        mask_vals = dists > thresh
        mask = pd.Series(False, index=series.index)
        mask.loc[series.dropna().index] = mask_vals
        return mask

    raise ValueError(f"Unknown outlier detection method: {method}")
