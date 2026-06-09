"""
Standardised evaluation metrics for the urban entropy prediction project.

All group members report the same metrics from the same held-out test set
so results can be compared in a single table.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    """
    Return the standard metric dict used across all protocol runs.

    Parameters
    ----------
    y_true, y_pred : array-like of float

    Returns
    -------
    dict with keys: rmse, mae, r2
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {'rmse': round(rmse, 6), 'mae': round(mae, 6), 'r2': round(r2, 6)}


def save_results(metrics, model_name, member, feature_set, results_dir=None):
    """
    Save evaluation metrics to results/model_results/<member>/<model>_<set>.json.

    Parameters
    ----------
    metrics : dict
        Output of compute_metrics(), optionally extended with cv_rmse / cv_std.
    model_name : str
        e.g. 'random_forest', 'gradient_boosting', 'ridge'
    member : str
        Team member identifier used as the subfolder (e.g. 'edu', 'eva').
    feature_set : str
        'core' or 'full'.
    results_dir : Path or str, optional
        Root directory for results. Defaults to <project_root>/results/model_results/.

    Returns
    -------
    Path to the saved JSON file.
    """
    if results_dir is None:
        root = Path(__file__).parent.parent.parent
        results_dir = root / 'results' / 'model_results'

    out_dir = Path(results_dir) / member
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        'model': model_name,
        'member': member,
        'feature_set': feature_set,
        **metrics,
    }

    out_path = out_dir / f"{model_name}_{feature_set}.json"
    with open(out_path, 'w') as fh:
        json.dump(record, fh, indent=2)

    print(f"Saved -> {out_path}")
    print(f"  RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
    return out_path
