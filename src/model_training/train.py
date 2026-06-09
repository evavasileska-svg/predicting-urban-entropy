"""
Shared training entry-point for the urban entropy prediction project.

Each team member runs this script with their own dataset path to produce
results that are directly comparable across members.

Usage
-----
    python src/model_training/train.py --member edu --feature_set core
    python src/model_training/train.py --member eva --feature_set full
    python src/model_training/train.py --member edu --feature_set core \\
        --data path/to/custom_training_data.csv

Results are saved to results/model_results/<member>/.
"""

import argparse
import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model_training.evaluate import compute_metrics, save_results
from src.model_training.protocol import N_CV_FOLDS, RANDOM_SEED, load_and_split

# ── model registry ─────────────────────────────────────────────────
# Fixed hyperparameters for all runs — do not tune per-dataset.
# The goal is dataset comparison, not model optimisation.
MODELS = {
    'ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=1.0)),
    ]),
    'random_forest': RandomForestRegressor(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_SEED,
    ),
    'xgboost': XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    ),
}


def train_and_evaluate(member, feature_set='core', data_path=None):
    print(f"\n{'=' * 60}")
    print(f"Urban Entropy — Shared Training Protocol")
    print(f"  Member:      {member}")
    print(f"  Feature set: {feature_set}")
    print(f"  Seed:        {RANDOM_SEED}")
    print(f"{'=' * 60}\n")

    X_train, X_test, y_train, y_test, features = load_and_split(
        csv_path=data_path, feature_set=feature_set
    )
    print(f"\nFeatures ({len(features)}): {', '.join(features)}\n")

    all_results = {}

    for model_name, model in MODELS.items():
        print(f"-- {model_name} {'-' * (40 - len(model_name))}")

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=N_CV_FOLDS,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
        )
        cv_rmse = float(-cv_scores.mean())
        cv_std  = float(cv_scores.std())
        print(f"CV RMSE ({N_CV_FOLDS}-fold): {cv_rmse:.4f} ± {cv_std:.4f}")

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred       = model.predict(X_test)
        train_metrics = compute_metrics(y_train, y_pred_train)
        metrics = compute_metrics(y_test, y_pred)
        metrics['r2_train'] = train_metrics['r2']
        metrics['cv_rmse']  = round(cv_rmse, 6)
        metrics['cv_std']   = round(cv_std, 6)
        print(f"Train R²: {train_metrics['r2']:.4f}  |  Test R²: {metrics['r2']:.4f}  |  Gap: {train_metrics['r2'] - metrics['r2']:.4f}")

        save_results(metrics, model_name, member, feature_set)
        all_results[model_name] = metrics
        print()

    # ── summary table ──────────────────────────────────────────────
    print(f"{'=' * 60}")
    print(f"SUMMARY — {member} / {feature_set}")
    print(f"{'=' * 60}")
    print(f"{'Model':<24} {'R²-train':>10} {'R²-test':>9} {'Gap':>7} {'RMSE':>8} {'CV-RMSE':>10}")
    print(f"{'-' * 70}")
    for name, m in all_results.items():
        gap = m['r2_train'] - m['r2']
        print(
            f"{name:<24} {m['r2_train']:>10.4f} {m['r2']:>9.4f} "
            f"{gap:>7.4f} {m['rmse']:>8.4f} {m['cv_rmse']:>10.4f}"
        )
    print()
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Urban Entropy — Shared Training Protocol'
    )
    parser.add_argument(
        '--member', required=True,
        help='Your identifier, used as the results subfolder (e.g. edu, eva)'
    )
    parser.add_argument(
        '--feature_set', choices=['graph', 'core', 'full'], default='core',
        help="'graph' = graph+circuity only; 'core' = graph+circuity+block; 'full' = core + building features"
    )
    parser.add_argument(
        '--data', default=None, dest='data_path',
        help='Path to training CSV (default: data/processed/patch_training_data_clean.csv)'
    )
    args = parser.parse_args()

    train_and_evaluate(
        member=args.member,
        feature_set=args.feature_set,
        data_path=args.data_path,
    )
