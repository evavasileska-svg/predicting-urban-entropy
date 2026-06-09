"""
Shared training protocol constants for the urban entropy prediction project.

All group members must use these exact settings so results are comparable.
Do NOT change RANDOM_SEED, TEST_SIZE, N_CV_FOLDS, or the feature lists
between runs — any change breaks cross-member comparability.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR

# ── reproducibility ────────────────────────────────────────────────
RANDOM_SEED = 42

# ── train/test split ───────────────────────────────────────────────
TEST_SIZE = 0.20   # 20% held-out test set

# ── cross-validation ───────────────────────────────────────────────
N_CV_FOLDS = 5

# ── target ─────────────────────────────────────────────────────────
TARGET = 'entropy_normalised'

# ── feature groups ─────────────────────────────────────────────────
FEATURES_GRAPH = [
    'n_4way', 'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_degree', 'mean_edge_length', 'total_edge_length',
    'meshedness', 'intersection_density', 'street_density',
]
FEATURES_CIRCUITY = ['circuity']
FEATURES_BLOCK = [
    'n_blocks', 'mean_block_area', 'std_block_area', 'cv_block_area',
    'mean_block_eri', 'mean_block_compact', 'mean_block_elong', 'mean_block_convex',
]
FEATURES_BUILDING = [
    'n_buildings', 'building_density',
    'mean_building_footprint', 'std_building_footprint',
    'mean_nn_distance', 'std_nn_distance',
]

# Three standard feature sets used across all runs:
#   GRAPH — graph + circuity only (100% patch coverage)
#   CORE  — graph + circuity + block (~84% patch coverage)
#   FULL  — all features (~69% patch coverage, requires step 12 building data)
FEATURE_SET_GRAPH = FEATURES_GRAPH + FEATURES_CIRCUITY
FEATURE_SET_CORE  = FEATURES_GRAPH + FEATURES_CIRCUITY + FEATURES_BLOCK
FEATURE_SET_FULL  = FEATURE_SET_CORE + FEATURES_BUILDING


def load_and_split(csv_path=None, feature_set='core'):
    """
    Load the training CSV and apply the fixed protocol train/test split.

    Parameters
    ----------
    csv_path : Path or str, optional
        Path to the training CSV. Defaults to
        data/processed/patch_training_data_clean.csv.
    feature_set : {'core', 'full'}
        'core' uses graph + circuity + block features (~82% coverage).
        'full' also includes building features (~30% coverage, drops more rows).

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names : tuple
        Arrays are numpy float64; feature_names is the ordered list of column
        names matching the columns of X_train / X_test.
    """
    if csv_path is None:
        csv_path = PROCESSED_DIR / 'patch_training_data_full.csv'

    df = pd.read_csv(csv_path)

    SETS = {'graph': FEATURE_SET_GRAPH, 'core': FEATURE_SET_CORE, 'full': FEATURE_SET_FULL}
    if feature_set not in SETS:
        raise ValueError(f"feature_set must be 'graph', 'core', or 'full', got {feature_set!r}")

    features = [f for f in SETS[feature_set] if f in df.columns]
    missing = [f for f in SETS[feature_set] if f not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} expected feature(s) not found in CSV: {missing}")

    cols_needed = features + [TARGET]
    df_clean = df[['patch_id', 'city_code'] + cols_needed].dropna(subset=cols_needed)

    print(f"Dataset: {len(df):,} rows loaded, "
          f"{len(df_clean):,} complete for '{feature_set}' features "
          f"({len(df) - len(df_clean):,} dropped due to NaN)")

    X = df_clean[features].to_numpy(dtype=float)
    y = df_clean[TARGET].to_numpy(dtype=float)

    # stratify on entropy decile so the full entropy range is represented in
    # both train and test sets regardless of dataset size
    entropy_decile = pd.qcut(df_clean[TARGET], q=10, labels=False, duplicates='drop')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=entropy_decile,
    )

    print(f"Split:   {len(X_train):,} train / {len(X_test):,} test")
    return X_train, X_test, y_train, y_test, features
