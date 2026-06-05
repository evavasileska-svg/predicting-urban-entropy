"""
K-Means clustering of patches based on graph + circuity features.

Workflow:
  1. Standardize the 13 features
  2. Test k = 2 to 10, compute inertia and silhouette score
  3. Identify optimal k from elbow + silhouette
  4. Run final clustering with chosen k
  5. Describe each cluster (centroids, distinguishing features, mean entropy)
  6. Show per-city distribution across clusters
  7. Visualize clusters in PC1-PC2 space

Outputs:
  - results/figures/clustering_k_selection.png (elbow + silhouette)
  - results/figures/clustering_pca_comparison.png (cluster vs entropy in PC space)
  - Terminal output with cluster descriptions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

FEATURES = [
    'n_4way', 'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_degree',
    'mean_edge_length', 'total_edge_length',
    'meshedness',
    'intersection_density', 'street_density',
    'circuity',
]

K_RANGE = range(2, 11)   # test k = 2 to 10
N_ENTROPY_BINS = 10
RANDOM_SEED = 42


def compute_bin_edges(target_values):
    """Reproduce stratification bin edges."""
    entropy_path = PROCESSED_DIR / "patch_entropy.csv"
    if entropy_path.exists():
        full = pd.read_csv(entropy_path)
        ho_min = full[TARGET].min()
        ho_max = full[TARGET].max()
    else:
        ho_min = target_values.min()
        ho_max = target_values.max()
    return np.linspace(ho_min, ho_max + 1e-6, N_ENTROPY_BINS + 1)


def evaluate_k_values(X_scaled):
    """Compute inertia and silhouette score for each k value."""
    results = []
    print(f"Evaluating k = {K_RANGE.start} to {K_RANGE.stop - 1}:\n")
    print(f"  {'k':<4} {'Inertia':<15} {'Silhouette':<12}")
    print(f"  {'-' * 4} {'-' * 15} {'-' * 12}")

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette': sil,
        })
        print(f"  {k:<4} {inertia:<15.1f} {sil:<12.4f}")

    return pd.DataFrame(results)


def plot_k_selection(eval_df):
    """Plot elbow curve and silhouette scores."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Elbow plot ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(eval_df['k'], eval_df['inertia'], 'o-',
            color='steelblue', linewidth=2, markersize=8, zorder=2)
    ax.set_xlabel('Number of clusters (k)', fontsize=11)
    ax.set_ylabel('Inertia (within-cluster SSE)', fontsize=11)
    ax.set_title('Elbow plot\n(look for the bend)',
                 fontsize=12, pad=10)
    ax.set_xticks(eval_df['k'])

    # ── Silhouette plot ────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(eval_df['k'], eval_df['silhouette'], 'o-',
            color='indianred', linewidth=2, markersize=8, zorder=2)

    # highlight best k by silhouette
    best_idx = eval_df['silhouette'].idxmax()
    best_k = int(eval_df.loc[best_idx, 'k'])
    best_sil = eval_df.loc[best_idx, 'silhouette']
    ax.axvline(best_k, color='green', linestyle='--', alpha=0.5,
               label=f'Best k = {best_k}\n(silhouette = {best_sil:.3f})')

    ax.set_xlabel('Number of clusters (k)', fontsize=11)
    ax.set_ylabel('Silhouette score', fontsize=11)
    ax.set_title('Silhouette score\n(higher = better separated)',
                 fontsize=12, pad=10)
    ax.set_xticks(eval_df['k'])
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = FIGURES_DIR / "clustering_k_selection.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    return best_k


def describe_clusters(df, labels, k):
    """Print descriptive statistics for each cluster."""
    df = df.copy()
    df['cluster'] = labels

    print(f"\n{'=' * 70}")
    print(f"CLUSTER DESCRIPTIONS (k = {k})")
    print(f"{'=' * 70}\n")

    for cluster_idx in range(k):
        cluster_df = df[df['cluster'] == cluster_idx]
        n = len(cluster_df)
        mean_entropy = cluster_df[TARGET].mean()
        std_entropy = cluster_df[TARGET].std()

        # compute feature means for this cluster vs overall
        cluster_means = cluster_df[FEATURES].mean()
        overall_means = df[FEATURES].mean()
        overall_stds = df[FEATURES].std()

        # z-score: how distinctive is each feature in this cluster
        z_scores = (cluster_means - overall_means) / overall_stds
        z_scores_sorted = z_scores.reindex(
            z_scores.abs().sort_values(ascending=False).index
        )

        print(f"  CLUSTER {cluster_idx}: {n} patches "
              f"({100*n/len(df):.1f}% of total)")
        print(f"  Mean entropy: {mean_entropy:.4f} ± {std_entropy:.4f}")
        print(f"  Top distinctive features (z-score vs overall):")
        for feature, z in z_scores_sorted.head(5).items():
            direction = '↑' if z > 0 else '↓'
            print(f"    {feature:<25} {direction} z = {z:+.2f}")
        print()


def per_city_clusters(df, labels):
    """Show how each city's patches distribute across clusters."""
    df = df.copy()
    df['cluster'] = labels

    print(f"{'=' * 70}")
    print(f"PER-CITY CLUSTER DISTRIBUTION (top 20 cities by patch count)")
    print(f"{'=' * 70}\n")

    # get top cities by patch count
    city_patch_counts = df['city_code'].value_counts().head(20)

    for city in city_patch_counts.index:
        city_df = df[df['city_code'] == city]
        n_total = len(city_df)

        # count patches per cluster for this city
        counts = city_df['cluster'].value_counts().sort_values(ascending=False)

        dist_str = ' | '.join([
            f"C{int(c)}: {int(n)} ({100*int(n)/n_total:.0f}%)"
            for c, n in counts.items()
        ])
        print(f"  {city:<15} (n={n_total:>3}):  {dist_str}")


def plot_cluster_pca_comparison(X_pca, labels, df, bin_edges, k):
    """Side-by-side: PC1-PC2 colored by cluster vs by entropy."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    pc1_label = 'PC1'
    pc2_label = 'PC2'

    # ── Left: colored by cluster ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # use tab10 for clusters
    cluster_cmap = cm.tab10
    for cluster_idx in range(k):
        mask = labels == cluster_idx
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=cluster_cmap(cluster_idx),
                   s=14, alpha=0.6, edgecolors='none',
                   label=f"Cluster {cluster_idx} (n={mask.sum()})",
                   zorder=2)

    ax.set_xlabel(pc1_label, fontsize=11)
    ax.set_ylabel(pc2_label, fontsize=11)
    ax.set_title(f'Patches colored by K-Means cluster (k = {k})',
                 fontsize=13, pad=10)
    ax.legend(loc='upper right', fontsize=9, frameon=True,
              framealpha=0.9)

    # ── Right: colored by entropy bin ─────────────────────────────
    ax = axes[1]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    entropy_bins = pd.cut(
        df[TARGET], bins=bin_edges,
        labels=False, include_lowest=True,
    ).astype(int)

    entropy_cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=N_ENTROPY_BINS - 1)

    for bin_idx in range(N_ENTROPY_BINS):
        mask = entropy_bins == bin_idx
        if mask.sum() == 0:
            continue
        bin_low = bin_edges[bin_idx]
        bin_high = bin_edges[bin_idx + 1]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=entropy_cmap(norm(bin_idx)),
                   s=14, alpha=0.6, edgecolors='none',
                   label=f"Bin {bin_idx}: [{bin_low:.2f}, {bin_high:.2f}]",
                   zorder=2)

    ax.set_xlabel(pc1_label, fontsize=11)
    ax.set_ylabel(pc2_label, fontsize=11)
    ax.set_title('Patches colored by entropy bin (reference)',
                 fontsize=13, pad=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8, frameon=True, framealpha=0.9)

    fig.suptitle(
        f"K-Means clusters vs entropy bins in PCA space",
        fontsize=14, y=1.02,
    )

    plt.tight_layout()
    output_path = FIGURES_DIR / "clustering_pca_comparison.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches from {csv_path.name}\n")

    # check features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features: {missing}")
        return

    # drop any rows with NaN (shouldn't happen with graph features)
    n_nan = df[FEATURES].isna().any(axis=1).sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} rows have NaN. Dropping.")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)

    # standardize features
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%\n")

    # evaluate k values
    eval_df = evaluate_k_values(X_scaled)

    # plot k selection
    best_k_silhouette = plot_k_selection(eval_df)

    print(f"\nBest k by silhouette score: {best_k_silhouette}")

    # use k=5 (chosen based on elbow method and interpretability)
    chosen_k = 5
    print(f"Running final clustering with k = {chosen_k}")
    print(f"(Note: silhouette would suggest k = {best_k_silhouette}, "
          f"but k = {chosen_k} chosen for finer archetypes)\n")

    # final clustering
    km = KMeans(n_clusters=chosen_k, random_state=RANDOM_SEED, n_init=20)
    labels = km.fit_predict(X_scaled)

    # describe clusters
    describe_clusters(df, labels, chosen_k)

    # per-city distribution
    per_city_clusters(df, labels)

    # bin edges for entropy comparison
    bin_edges = compute_bin_edges(df[TARGET])

    # visualization
    plot_cluster_pca_comparison(X_pca, labels, df, bin_edges, chosen_k)


if __name__ == "__main__":
    main()