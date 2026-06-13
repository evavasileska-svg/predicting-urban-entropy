"""
Correlation matrix of the 13 graph + circuity features plus entropy target.

Features are reordered using hierarchical clustering so that correlated
features appear next to each other.

Outputs:
  - results/figures/correlation_matrix.png
  - Prints the correlation matrix to terminal
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

FEATURES = [
    'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_edge_length', 'total_edge_length',
    'circuity',
    'elev_mean', 'elev_std', 'elev_range', 'mean_slope',
    'distance_to_center_km',
]


def compute_correlation_matrix(df):
    """Compute correlation matrix for features + target."""
    cols = FEATURES + [TARGET]
    return df[cols].corr(method='pearson')


def reorder_by_clustering(corr_matrix):
    """Reorder matrix using hierarchical clustering of features.

    Uses correlation distance (1 - |r|) and average linkage.
    """
    # convert correlation to distance: high |r| → low distance
    distance = 1 - np.abs(corr_matrix.values)

    # ensure diagonal is exactly zero (avoids floating-point issues)
    np.fill_diagonal(distance, 0.0)

    # condensed distance vector (upper triangle) for linkage
    condensed = squareform(distance, checks=False)

    # hierarchical clustering with average linkage
    linkage_matrix = linkage(condensed, method='average')

    # get the leaf order
    order = leaves_list(linkage_matrix)

    # reorder the correlation matrix
    feature_names = corr_matrix.index.tolist()
    ordered_names = [feature_names[i] for i in order]
    return corr_matrix.loc[ordered_names, ordered_names]


def plot_correlation_matrix(corr_matrix):
    """Heatmap of the correlation matrix with annotations."""
    n_features = len(corr_matrix)

    fig, ax = plt.subplots(figsize=(13, 12))

    # heatmap
    im = ax.imshow(corr_matrix.values,
                   cmap='RdBu_r',
                   vmin=-1, vmax=1,
                   aspect='auto')

    # ticks and labels
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(corr_matrix.columns, rotation=45,
                       ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.index, fontsize=10)

    # annotate each cell with the correlation value
    for i in range(n_features):
        for j in range(n_features):
            value = corr_matrix.iloc[i, j]
            # bold if strong correlation (excluding diagonal)
            is_strong = abs(value) >= 0.7 and i != j
            text_color = 'white' if abs(value) > 0.5 else 'black'
            weight = 'bold' if is_strong else 'normal'
            ax.text(j, i, f"{value:+.2f}",
                    ha='center', va='center',
                    color=text_color,
                    fontsize=8,
                    fontweight=weight)

    # highlight the target row and column
    target_idx = list(corr_matrix.index).index(TARGET)
    # row outline
    ax.add_patch(plt.Rectangle(
        (-0.5, target_idx - 0.5),
        n_features, 1,
        fill=False, edgecolor='black', linewidth=2,
    ))
    # column outline
    ax.add_patch(plt.Rectangle(
        (target_idx - 0.5, -0.5),
        1, n_features,
        fill=False, edgecolor='black', linewidth=2,
    ))

    # colorbar
    cbar = plt.colorbar(im, ax=ax, label='Pearson correlation',
                        shrink=0.85)
    cbar.ax.tick_params(labelsize=10)

    # title
    ax.set_title(
        "Feature correlation matrix (clustered order)\n"
        "Black outline highlights target variable",
        fontsize=13, pad=15,
    )

    plt.tight_layout()
    output_path = FIGURES_DIR / "correlation_matrix.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_correlation_summary(corr_matrix):
    """Print key findings from the correlation matrix."""
    print(f"\n{'=' * 60}")
    print(f"CORRELATION MATRIX (clustered order)")
    print(f"{'=' * 60}\n")

    # print the full matrix
    print(corr_matrix.round(2).to_string())

    # find strongly correlated feature pairs
    print(f"\n{'=' * 60}")
    print(f"STRONGLY CORRELATED FEATURE PAIRS (|r| >= 0.7)")
    print(f"{'=' * 60}\n")

    strong_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= 0.7:
                strong_pairs.append({
                    'feature_1': cols[i],
                    'feature_2': cols[j],
                    'r':         r,
                })

    if not strong_pairs:
        print(f"  No pairs with |r| >= 0.7")
    else:
        for p in sorted(strong_pairs, key=lambda x: -abs(x['r'])):
            sign = '+' if p['r'] > 0 else '-'
            print(f"  {p['feature_1']:<22} <-> "
                  f"{p['feature_2']:<22} "
                  f"r = {sign}{abs(p['r']):.3f}")

    # correlations with the target
    print(f"\n{'=' * 60}")
    print(f"CORRELATIONS WITH {TARGET}")
    print(f"{'=' * 60}\n")

    target_corrs = corr_matrix[TARGET].drop(TARGET)
    sorted_corrs = target_corrs.reindex(
        target_corrs.abs().sort_values(ascending=False).index
    )

    for feature, r in sorted_corrs.items():
        sign = '+' if r > 0 else '-'
        print(f"  {feature:<25} r = {sign}{abs(r):.3f}")


def main():
    csv_path = PROCESSED_DIR / "dataset_dropped_features_clean.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Run 14a_merge_features.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches from {csv_path.name}")
    print(f"Computing correlation matrix for {len(FEATURES)} features "
          f"+ target\n")

    # check feature availability
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features: {missing}")
        return

    # compute and reorder
    corr = compute_correlation_matrix(df)
    corr_clustered = reorder_by_clustering(corr)

    # print findings
    print_correlation_summary(corr_clustered)

    # plot
    plot_correlation_matrix(corr_clustered)


if __name__ == "__main__":
    main()