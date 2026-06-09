"""
SHAP analysis for the urban entropy prediction model.

Trains Random Forest on the selected feature set and computes SHAP values
to reveal which features drive entropy predictions most.

Produces two plots saved to results/figures/shap/:
  - shap_bar_<feature_set>.png     : mean absolute SHAP value per feature (ranking)
  - shap_beeswarm_<feature_set>.png: per-sample SHAP values (direction + magnitude)

Usage
-----
    python src/model_analysis/01_shap_analysis.py --feature_set graph
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import shap

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import FIGURES_DIR
from src.model_training.protocol import RANDOM_SEED, load_and_split
from sklearn.ensemble import RandomForestRegressor


def run_shap(feature_set='graph'):
    print(f"\n{'=' * 60}")
    print(f"SHAP Analysis — feature set: {feature_set}")
    print(f"{'=' * 60}\n")

    X_train, X_test, y_train, y_test, features = load_and_split(
        feature_set=feature_set
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    print("Training Random Forest...")
    model.fit(X_train, y_train)

    print("Computing SHAP values on test set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    out_dir = FIGURES_DIR / 'shap'
    out_dir.mkdir(parents=True, exist_ok=True)

    # bar plot — feature importance ranking
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=features,
        plot_type='bar',
        show=False,
    )
    plt.title(f'SHAP Feature Importance — {feature_set}')
    plt.tight_layout()
    bar_path = out_dir / f'shap_bar_{feature_set}.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {bar_path}")

    # beeswarm — direction and spread of each feature's effect
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=features,
        show=False,
    )
    plt.title(f'SHAP Beeswarm — {feature_set}')
    plt.tight_layout()
    beeswarm_path = out_dir / f'shap_beeswarm_{feature_set}.png'
    plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {beeswarm_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHAP analysis for urban entropy model')
    parser.add_argument(
        '--feature_set', choices=['graph', 'core', 'full'], default='graph',
    )
    args = parser.parse_args()
    run_shap(feature_set=args.feature_set)
