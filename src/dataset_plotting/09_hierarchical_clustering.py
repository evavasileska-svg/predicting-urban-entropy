import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

FEATURES = [
    'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_edge_length', 'total_edge_length',
    'circuity',
    'distance_to_center_km',
]

# how many clusters to extract for the colored visualization
N_CLUSTERS = 5

# how many top-level branches to show in the dendrogram
N_TRUNCATE = 30

RANDOM_SEED = 42


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches\n")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features: {missing}")
        return

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"After dropping NaN: {len(df):,} patches\n")

    # standardize features
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Computing hierarchical clustering...")
    print(f"(this may take 30-60 seconds for {len(df)} patches)\n")

    # Ward linkage minimizes within-cluster variance
    # similar in spirit to K-Means but produces a hierarchy
    Z = linkage(X_scaled, method='ward')

    print(f"Linkage matrix computed: shape {Z.shape}\n")

    # extract cluster labels for N_CLUSTERS
    labels = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')
    # convert to 0-indexed
    labels = labels - 1

    print(f"Extracted {N_CLUSTERS} clusters from the hierarchy:\n")
    for c in range(N_CLUSTERS):
        n = (labels == c).sum()
        mean_ent = df.loc[labels == c, TARGET].mean()
        print(f"  Cluster {c}: {n} patches ({100*n/len(df):.1f}%), "
              f"mean entropy = {mean_ent:.3f}")
    print()

    # ── Plot 1: dendrogram (truncated) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_facecolor('white')

    dendrogram(
        Z,
        ax=ax,
        truncate_mode='lastp',
        p=N_TRUNCATE,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        color_threshold=Z[-(N_CLUSTERS - 1), 2],
    )

    ax.set_title(
        f"Hierarchical clustering dendrogram "
        f"(Ward linkage, showing top {N_TRUNCATE} merges)\n"
        f"colored at the cut for {N_CLUSTERS} clusters",
        fontsize=13, pad=15,
    )
    ax.set_xlabel('Cluster index (or number of patches in parentheses)',
                  fontsize=11)
    ax.set_ylabel('Distance (Ward)', fontsize=11)

    # mark the cut line
    cut_height = Z[-(N_CLUSTERS - 1), 2]
    ax.axhline(cut_height, color='red', linestyle='--', alpha=0.5,
               label=f'Cut at distance {cut_height:.1f} → {N_CLUSTERS} clusters')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = FIGURES_DIR / "hierarchical_dendrogram.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}\n")

    # ── Plot 2: clusters in PCA space ─────────────────────────────
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA for visualization: "
          f"PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%\n")

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    cluster_cmap = cm.tab10
    for c in range(N_CLUSTERS):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mean_ent = df.loc[mask, TARGET].mean()
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=cluster_cmap(c),
                   s=14, alpha=0.6, edgecolors='none',
                   label=f"H-Cluster {c} (n={mask.sum()}, "
                         f"mean Ho={mean_ent:.2f})",
                   zorder=2)

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(
        f'Hierarchical clusters (k = {N_CLUSTERS}, Ward linkage) in PCA space',
        fontsize=13, pad=10,
    )
    ax.legend(loc='upper right', fontsize=9, frameon=True,
              framealpha=0.9)

    plt.tight_layout()
    output_path = FIGURES_DIR / "hierarchical_clusters_pca.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}\n")

    # ── Cluster descriptions ──────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"HIERARCHICAL CLUSTER DESCRIPTIONS (k = {N_CLUSTERS})")
    print(f"{'=' * 70}\n")

    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels

    for c in range(N_CLUSTERS):
        cluster_df = df_with_labels[df_with_labels['cluster'] == c]
        n = len(cluster_df)
        mean_ent = cluster_df[TARGET].mean()
        std_ent = cluster_df[TARGET].std()

        cluster_means = cluster_df[FEATURES].mean()
        overall_means = df_with_labels[FEATURES].mean()
        overall_stds = df_with_labels[FEATURES].std()

        z_scores = (cluster_means - overall_means) / overall_stds
        z_scores_sorted = z_scores.reindex(
            z_scores.abs().sort_values(ascending=False).index
        )

        print(f"  H-CLUSTER {c}: {n} patches ({100*n/len(df_with_labels):.1f}%)")
        print(f"  Mean entropy: {mean_ent:.4f} +/- {std_ent:.4f}")
        if 'distance_to_center_km' in cluster_df.columns:
            mean_dist = cluster_df['distance_to_center_km'].mean()
            print(f"  Mean dist to center: {mean_dist:.2f} km")
        print(f"  Top distinctive features (z-score vs overall):")
        for feature, z in z_scores_sorted.head(5).items():
            direction = 'up' if z > 0 else 'down'
            print(f"    {feature:<25} {direction} z = {z:+.2f}")
        print()


if __name__ == "__main__":
    main()