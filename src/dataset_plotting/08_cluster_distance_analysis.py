"""
Analyze how distance to city center varies across the 5 K-Means clusters.

Workflow:
  1. Load the merged training dataset (with distance_to_center_km)
  2. Re-run K-Means clustering with the same 13 features (NOT including distance)
  3. For each cluster, compute distance statistics
  4. Test whether distance differs significantly between clusters
  5. Plot distance distributions per cluster
  6. Report findings

Outputs:
  - results/figures/cluster_distance_analysis.png
  - Terminal output with statistics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

# same 13 features used for clustering before
FEATURES = [
    'n_4way', 'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_degree',
    'mean_edge_length', 'total_edge_length',
    'meshedness',
    'intersection_density', 'street_density',
    'circuity',
]

# the new feature we want to analyze across clusters
DISTANCE_FEATURE = 'distance_to_center_km'

# clustering settings (same as before)
K = 5
RANDOM_SEED = 42


def run_clustering(df):
    """Re-run K-Means with the same 13 features."""
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=20)
    labels = km.fit_predict(X_scaled)
    return labels


def analyze_distance_per_cluster(df, labels):
    """Compute distance statistics per cluster."""
    df = df.copy()
    df['cluster'] = labels

    print(f"\n{'=' * 70}")
    print(f"DISTANCE TO CITY CENTER PER CLUSTER")
    print(f"{'=' * 70}\n")

    print(f"  {'Cluster':<10} {'N':<6} {'Mean':<10} {'Median':<10} "
          f"{'Std':<10} {'Min':<8} {'Max':<8}")
    print(f"  {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 10} "
          f"{'-' * 10} {'-' * 8} {'-' * 8}")

    cluster_stats = []
    for cluster_idx in range(K):
        cluster_df = df[df['cluster'] == cluster_idx]
        distances = cluster_df[DISTANCE_FEATURE].dropna()

        if len(distances) == 0:
            continue

        stats_dict = {
            'cluster':  cluster_idx,
            'n':        len(distances),
            'mean':     distances.mean(),
            'median':   distances.median(),
            'std':      distances.std(),
            'min':      distances.min(),
            'max':      distances.max(),
        }
        cluster_stats.append(stats_dict)

        print(f"  C{cluster_idx:<9} "
              f"{stats_dict['n']:<6} "
              f"{stats_dict['mean']:<10.2f} "
              f"{stats_dict['median']:<10.2f} "
              f"{stats_dict['std']:<10.2f} "
              f"{stats_dict['min']:<8.2f} "
              f"{stats_dict['max']:<8.2f}")

    return cluster_stats


def statistical_test(df, labels):
    """ANOVA test: do clusters differ significantly in distance?"""
    print(f"\n{'=' * 70}")
    print(f"STATISTICAL TEST (ANOVA)")
    print(f"{'=' * 70}\n")

    df = df.copy()
    df['cluster'] = labels

    # group distances by cluster
    groups = [
        df[df['cluster'] == c][DISTANCE_FEATURE].dropna().values
        for c in range(K)
    ]

    # F-test (ANOVA)
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"  Null hypothesis: clusters have the same mean distance")
    print(f"  F-statistic:     {f_stat:.4f}")
    print(f"  p-value:         {p_value:.6f}")

    if p_value < 0.001:
        print(f"  Result: REJECT null (p < 0.001) — "
              f"clusters differ significantly in distance")
    elif p_value < 0.05:
        print(f"  Result: REJECT null (p < 0.05) — "
              f"clusters differ in distance")
    else:
        print(f"  Result: FAIL to reject null — "
              f"no significant difference in distance")
    print()


def plot_distance_per_cluster(df, labels):
    """Boxplot and histogram of distance per cluster."""
    df = df.copy()
    df['cluster'] = labels

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    cluster_cmap = cm.tab10

    # ── Left: boxplot ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0, axis='y')
    ax.set_axisbelow(True)

    box_data = []
    box_positions = []
    box_colors = []
    box_labels = []

    for cluster_idx in range(K):
        cluster_distances = df[df['cluster'] == cluster_idx][
            DISTANCE_FEATURE
        ].dropna().values
        if len(cluster_distances) == 0:
            continue
        box_data.append(cluster_distances)
        box_positions.append(cluster_idx)
        box_colors.append(cluster_cmap(cluster_idx))
        box_labels.append(f"C{cluster_idx}\n(n={len(cluster_distances)})")

    bplot = ax.boxplot(
        box_data, positions=box_positions,
        patch_artist=True, widths=0.6,
        medianprops={'color': 'black', 'linewidth': 1.5},
        zorder=2,
    )

    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    ax.set_xticks(box_positions)
    ax.set_xticklabels(box_labels, fontsize=10)
    ax.set_xlabel('K-Means cluster', fontsize=11)
    ax.set_ylabel(f'{DISTANCE_FEATURE}', fontsize=11)
    ax.set_title('Distance distribution per cluster (boxplot)',
                 fontsize=13, pad=10)

    # ── Right: overlapping histograms ─────────────────────────────
    ax = axes[1]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # determine common bin edges
    all_distances = df[DISTANCE_FEATURE].dropna()
    bin_edges = np.linspace(all_distances.min(), all_distances.max(), 40)

    for cluster_idx in range(K):
        cluster_distances = df[df['cluster'] == cluster_idx][
            DISTANCE_FEATURE
        ].dropna().values
        if len(cluster_distances) == 0:
            continue
        ax.hist(
            cluster_distances,
            bins=bin_edges,
            color=cluster_cmap(cluster_idx),
            alpha=0.5,
            edgecolor='black',
            linewidth=0.3,
            label=f'C{cluster_idx} (n={len(cluster_distances)})',
            zorder=2,
        )

    ax.set_xlabel(f'{DISTANCE_FEATURE}', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distance distribution per cluster (histograms)',
                 fontsize=13, pad=10)
    ax.legend(loc='upper right', fontsize=10, frameon=True,
              framealpha=0.9)

    fig.suptitle(
        f"How does {DISTANCE_FEATURE} differ between K-Means clusters?",
        fontsize=14, y=1.02,
    )

    plt.tight_layout()
    output_path = FIGURES_DIR / "cluster_distance_analysis.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}\n")


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Run 14_merge_features.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches from {csv_path.name}\n")

    # check that distance column exists
    if DISTANCE_FEATURE not in df.columns:
        print(f"ERROR: {DISTANCE_FEATURE} not found in training data.")
        print(f"Make sure 13_compute_distance_city_center.py ran successfully")
        print(f"and 14_merge_features.py was rerun afterward.")
        return

    # drop rows with NaN in features (shouldn't happen)
    n_nan = df[FEATURES].isna().any(axis=1).sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} rows have NaN in features. Dropping.")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)

    # also drop rows with NaN distance
    n_nan_dist = df[DISTANCE_FEATURE].isna().sum()
    if n_nan_dist > 0:
        print(f"WARNING: {n_nan_dist} rows have NaN distance. Dropping.")
        df = df.dropna(subset=[DISTANCE_FEATURE]).reset_index(drop=True)

    # run clustering
    print(f"Running K-Means clustering with k={K} on {len(FEATURES)} "
          f"features...\n")
    labels = run_clustering(df)

    # analyze distance per cluster
    cluster_stats = analyze_distance_per_cluster(df, labels)

    # statistical test
    statistical_test(df, labels)

    # visualizations
    plot_distance_per_cluster(df, labels)

    print(f"{'=' * 70}")
    print(f"INTERPRETATION")
    print(f"{'=' * 70}\n")

    # sort clusters by mean distance
    sorted_stats = sorted(cluster_stats, key=lambda x: x['mean'])

    print(f"  Clusters ordered by mean distance to city center:")
    print()
    for rank, s in enumerate(sorted_stats, 1):
        print(f"  {rank}. Cluster {s['cluster']:>1}: mean = {s['mean']:>6.1f} km, "
              f"median = {s['median']:>6.1f} km, n = {s['n']}")


if __name__ == "__main__":
    main()