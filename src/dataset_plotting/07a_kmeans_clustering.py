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
from src.config import PROCESSED_DIR, FIGURES_DIR, CITIES


TARGET = 'entropy_normalised'
DISTANCE_FEATURE = 'distance_to_center_km'

# the 9 features after dropping redundancies
FEATURES = [
    'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_edge_length', 'total_edge_length',
    'circuity',
    'distance_to_center_km',
]

K_RANGE = range(2, 11)
N_ENTROPY_BINS = 10
RANDOM_SEED = 42

REGION_ORDER = [
    'North America',
    'Latin America',
    'Europe',
    'Asia',
    'Africa',
    'Middle East',
    'Oceania',
]

REGION_COLORS = {
    'North America': '#1f77b4',
    'Latin America': '#ff7f0e',
    'Europe':        '#2ca02c',
    'Asia':          '#d62728',
    'Africa':        '#9467bd',
    'Middle East':   '#8c564b',
    'Oceania':       '#e377c2',
    'Other':         '#7f7f7f',
}


def map_city_to_region(city_code):
    label = CITIES.get(city_code, '').lower()

    if any(k in label for k in ['usa', 'united states', 'canada']):
        return 'North America'

    if any(k in label for k in [
        'argentina', 'colombia', 'venezuela', 'ecuador',
        'brazil', 'chile', 'peru', 'mexico', 'uruguay',
    ]):
        return 'Latin America'

    if any(k in label for k in [
        'morocco', 'egypt', 'tunisia', 'algeria', 'libya',
        'south africa', 'nigeria', 'kenya', 'ethiopia',
    ]):
        return 'Africa'

    if any(k in label for k in [
        'turkey', 'iran', 'israel', 'palestine', 'lebanon',
        'syria', 'jordan', 'iraq', 'saudi arabia', 'uae',
        'georgia',
    ]):
        return 'Middle East'

    if any(k in label for k in [
        'india', 'japan', 'china', 'south korea', 'vietnam',
        'thailand', 'indonesia', 'malaysia', 'singapore',
        'philippines', 'taiwan', 'uzbekistan', 'kazakhstan',
        'mongolia', 'macau',
    ]):
        return 'Asia'

    if any(k in label for k in ['australia', 'new zealand']):
        return 'Oceania'

    if any(k in label for k in [
        'france', 'spain', 'italy', 'germany', 'uk',
        'united kingdom', 'belgium', 'poland', 'romania',
        'slovakia', 'hungary', 'netherlands', 'czech',
        'sweden', 'finland', 'ireland', 'norway', 'austria',
        'switzerland', 'greece', 'estonia', 'lithuania',
        'latvia', 'serbia', 'bosnia', 'macedonia', 'ukraine',
        'belarus', 'denmark', 'portugal', 'bulgaria',
        'croatia', 'slovenia', 'iceland', 'malta',
    ]):
        return 'Europe'

    return 'Other'


def compute_bin_edges(target_values):
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

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

    ax = axes[1]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(eval_df['k'], eval_df['silhouette'], 'o-',
            color='indianred', linewidth=2, markersize=8, zorder=2)

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
    output_path = FIGURES_DIR / "clustering_k_selection_v2.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    return best_k


def describe_clusters(df, labels, k):
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

        cluster_means = cluster_df[FEATURES].mean()
        overall_means = df[FEATURES].mean()
        overall_stds = df[FEATURES].std()

        z_scores = (cluster_means - overall_means) / overall_stds
        z_scores_sorted = z_scores.reindex(
            z_scores.abs().sort_values(ascending=False).index
        )

        print(f"  CLUSTER {cluster_idx}: {n} patches "
              f"({100*n/len(df):.1f}% of total)")
        print(f"  Mean entropy: {mean_entropy:.4f} +/- {std_entropy:.4f}")

        if DISTANCE_FEATURE in cluster_df.columns:
            mean_dist = cluster_df[DISTANCE_FEATURE].mean()
            median_dist = cluster_df[DISTANCE_FEATURE].median()
            print(f"  Mean dist to center: {mean_dist:.2f} km "
                  f"(median {median_dist:.2f} km)")

        print(f"  Top distinctive features (z-score vs overall):")
        for feature, z in z_scores_sorted.head(5).items():
            direction = 'up' if z > 0 else 'down'
            print(f"    {feature:<25} {direction} z = {z:+.2f}")
        print()


def per_city_clusters(df, labels):
    df = df.copy()
    df['cluster'] = labels

    print(f"{'=' * 70}")
    print(f"PER-CITY CLUSTER DISTRIBUTION (top 20 cities by patch count)")
    print(f"{'=' * 70}\n")

    city_patch_counts = df['city_code'].value_counts().head(20)

    for city in city_patch_counts.index:
        city_df = df[df['city_code'] == city]
        n_total = len(city_df)
        counts = city_df['cluster'].value_counts().sort_values(ascending=False)

        dist_str = ' | '.join([
            f"C{int(c)}: {int(n)} ({100*int(n)/n_total:.0f}%)"
            for c, n in counts.items()
        ])
        print(f"  {city:<15} (n={n_total:>3}):  {dist_str}")


def per_region_summary(df, labels):
    df = df.copy()
    df['cluster'] = labels

    print(f"\n{'=' * 70}")
    print(f"PER-REGION SUMMARY")
    print(f"{'=' * 70}\n")

    for region in REGION_ORDER:
        region_df = df[df['region'] == region]
        n = len(region_df)
        if n == 0:
            continue
        mean_entropy = region_df[TARGET].mean()
        cities = region_df['city_code'].unique()

        cluster_dist = region_df['cluster'].value_counts().sort_values(
            ascending=False
        )
        cluster_str = ' | '.join([
            f"C{int(c)}: {int(n)} ({100*int(n)/len(region_df):.0f}%)"
            for c, n in cluster_dist.items()
        ])

        print(f"  {region:<15} n={n:>4}, mean Ho={mean_entropy:.3f}, "
              f"{len(cities)} cities")
        print(f"    Clusters: {cluster_str}")
        print()


def plot_cluster_pca_comparison(X_pca, labels, df, bin_edges, k):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # Panel 1: clusters
    ax = axes[0, 0]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

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

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(f'K-Means clusters (k = {k})',
                 fontsize=13, pad=10)
    ax.legend(loc='upper right', fontsize=9, frameon=True,
              framealpha=0.9)

    # Panel 2: entropy bins
    ax = axes[0, 1]
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

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Entropy bins',
                 fontsize=13, pad=10)
    ax.legend(loc='upper right', fontsize=7, frameon=True,
              framealpha=0.9, ncol=1)

    # Panel 3: region
    ax = axes[1, 0]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    regions = df['region'].fillna('Other')

    for region in REGION_ORDER:
        mask = regions == region
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=REGION_COLORS[region],
                   s=14, alpha=0.7, edgecolors='none',
                   label=f"{region} (n={mask.sum()})",
                   zorder=2)

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Region of origin',
                 fontsize=13, pad=10)
    ax.legend(loc='upper right', fontsize=9, frameon=True,
              framealpha=0.9)

    # Panel 4: distance to city center
    ax = axes[1, 1]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    if DISTANCE_FEATURE in df.columns:
        distances = df[DISTANCE_FEATURE].values
        dist_min = np.nanmin(distances)
        dist_max = np.nanpercentile(distances, 95)

        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=distances,
                        cmap='plasma',
                        vmin=dist_min,
                        vmax=dist_max,
                        s=14, alpha=0.7, edgecolors='none',
                        zorder=2)

        cbar = plt.colorbar(sc, ax=ax, label='Distance to city center (km)',
                            shrink=0.8)
        cbar.ax.tick_params(labelsize=9)

        ax.set_title(
            f'Distance to city center\n'
            f'(colormap capped at 95th percentile = {dist_max:.1f} km)',
            fontsize=13, pad=10,
        )
    else:
        ax.text(0.5, 0.5, f'{DISTANCE_FEATURE}\nnot in dataset',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)

    fig.suptitle(
        f"PCA space: K-Means clusters, entropy, region, distance "
        f"({len(FEATURES)} features)",
        fontsize=14, y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    output_path = FIGURES_DIR / "clustering_pca_comparison_v2.png"
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

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features: {missing}")
        print(f"Did you run 14a_merge_features.py first?")
        return

    print(f"Using {len(FEATURES)} features for clustering:")
    for f in FEATURES:
        print(f"  - {f}")
    print()

    n_nan = df[FEATURES].isna().any(axis=1).sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} rows have NaN. Dropping.")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if 'city_code' in df.columns:
        df['region'] = df['city_code'].apply(map_city_to_region)
    else:
        df['region'] = 'Unknown'

    print(f"Region distribution:")
    for region in REGION_ORDER:
        n = (df['region'] == region).sum()
        if n > 0:
            print(f"  {region:<15} {n:>4} patches")
    print()

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%\n")

    eval_df = evaluate_k_values(X_scaled)
    best_k_silhouette = plot_k_selection(eval_df)

    print(f"\nBest k by silhouette score: {best_k_silhouette}")

    chosen_k = 4
    print(f"Running final clustering with k = {chosen_k}")
    print(f"(Note: silhouette would suggest k = {best_k_silhouette}, "
          f"but k = {chosen_k} chosen for finer archetypes)\n")

    km = KMeans(n_clusters=chosen_k, random_state=RANDOM_SEED, n_init=20)
    labels = km.fit_predict(X_scaled)

    describe_clusters(df, labels, chosen_k)
    per_city_clusters(df, labels)
    per_region_summary(df, labels)

    bin_edges = compute_bin_edges(df[TARGET])
    plot_cluster_pca_comparison(X_pca, labels, df, bin_edges, chosen_k)


if __name__ == "__main__":
    main()