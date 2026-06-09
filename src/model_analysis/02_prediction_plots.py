"""
Prediction visualisation for the urban entropy prediction model.

Trains all three models and produces:
  - prediction_scatter_<feature_set>.png : predicted vs actual entropy for all 3 models
  - prediction_errors_<feature_set>.png  : error distribution histogram for all 3 models
  - prediction_by_city_<feature_set>.png : median absolute error per city (top 20 worst)

Saved to results/figures/predictions/.

Usage
-----
    python src/model_analysis/02_prediction_plots.py --feature_set graph
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import FIGURES_DIR, PROCESSED_DIR
from src.model_training.protocol import RANDOM_SEED, TARGET, TEST_SIZE, load_and_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


MODELS = {
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=1.0)),
    ]),
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_SEED,
    ),
}


def run_prediction_plots(feature_set='graph'):
    print(f"\n{'=' * 60}")
    print(f"Prediction Plots — feature set: {feature_set}")
    print(f"{'=' * 60}\n")

    X_train, X_test, y_train, y_test, features = load_and_split(
        feature_set=feature_set
    )

    # reload full CSV to recover city_code for the test indices
    df = pd.read_csv(PROCESSED_DIR / 'patch_training_data_full.csv')
    from src.model_training.protocol import FEATURE_SET_GRAPH, FEATURE_SET_CORE, FEATURE_SET_FULL
    SETS = {'graph': FEATURE_SET_GRAPH, 'core': FEATURE_SET_CORE, 'full': FEATURE_SET_FULL}
    cols_needed = [f for f in SETS[feature_set] if f in df.columns] + [TARGET]
    df_clean = df[['patch_id', 'city_code'] + cols_needed].dropna(subset=cols_needed)
    entropy_decile = pd.qcut(df_clean[TARGET], q=10, labels=False, duplicates='drop')
    _, test_idx = train_test_split(
        df_clean.index, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=entropy_decile
    )
    city_codes_test = df_clean.loc[test_idx, 'city_code'].values

    out_dir = FIGURES_DIR / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for name, model in MODELS.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    # ── scatter: predicted vs actual ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Predicted vs Actual Entropy — {feature_set} features', fontsize=13)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel('Actual entropy')
        ax.set_ylabel('Predicted entropy')
        ax.set_title(f'{name}\nR²={r2:.3f}')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    scatter_path = out_dir / f'prediction_scatter_{feature_set}.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {scatter_path}")

    # ── error histograms ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Prediction Error Distribution — {feature_set} features', fontsize=13)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        errors = y_pred - y_test
        ax.hist(errors, bins=30, color='steelblue', edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Prediction error (predicted − actual)')
        ax.set_ylabel('Count')
        ax.set_title(name)

    plt.tight_layout()
    error_path = out_dir / f'prediction_errors_{feature_set}.png'
    plt.savefig(error_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {error_path}")

    # ── per-city median absolute error (Random Forest only) ───────────
    rf_pred = predictions['Random Forest']
    abs_errors = np.abs(rf_pred - y_test)
    city_error_df = pd.DataFrame({'city': city_codes_test, 'abs_error': abs_errors})
    city_summary = city_error_df.groupby('city')['abs_error'].median().sort_values(ascending=False)

    top_n = min(20, len(city_summary))
    fig, ax = plt.subplots(figsize=(10, 6))
    city_summary.head(top_n).plot(kind='barh', ax=ax, color='steelblue')
    ax.invert_yaxis()
    ax.set_xlabel('Median absolute error')
    ax.set_title(f'Worst predicted cities — Random Forest / {feature_set} (top {top_n})')
    plt.tight_layout()
    city_path = out_dir / f'prediction_by_city_{feature_set}.png'
    plt.savefig(city_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {city_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction plots for urban entropy model')
    parser.add_argument(
        '--feature_set', choices=['graph', 'core', 'full'], default='graph',
    )
    args = parser.parse_args()
    run_prediction_plots(feature_set=args.feature_set)
