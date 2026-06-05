"""
PCA analysis on the 13 graph + circuity features.

Produces a single image with 4 panels:
  1. Scree plot (% variance explained per component)
  2. Cumulative variance plot
  3. Patches in PC1-PC2 space, colored by entropy bin
  4. Feature loadings heatmap (which features load on which PCs)

Also prints to terminal:
  - Variance explained per PC
  - Top features loading on PC1, PC2, PC3
  - Number of PCs needed for 80%, 90%, 95% variance

Outputs:
  - results/figures/pca_analysis.png
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

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

# graph + circuity features (all 100% complete)
FEATURES = [
    'n_4way', 'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_degree',
    'mean_edge_length', 'total_edge_length',
    'meshedness',
    'intersection_density', 'street_density',
    'circuity',
]

N_ENTROPY_BINS = 10


def compute_bin_edges(target_values):
    """Reproduce stratification bin edges from script 08."""
    entropy_path = PROCESSED_DIR / "patch_entropy.csv"
    if entropy_path.exists():
        full = pd.read_csv(entropy_path)
        ho_min = full[TARGET].min()
        ho_max = full[TARGET].max()
    else:
        ho_min = target_values.min()
        ho_max = target_values.max()

    return np.linspace(ho_min, ho_max + 1e-6, N_ENTROPY_BINS + 1)


def run_pca(df):
    """Standardize features and fit PCA."""
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=len(FEATURES))
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, X_scaled


def print_pca_summary(pca):
    """Print variance explained and key findings."""
    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)

    print(f"\n{'=' * 60}")
    print(f"PCA SUMMARY")
    print(f"{'=' * 60}\n")
    print(f"  {'PC':<6} {'Var explained':<18} {'Cumulative':<12}")
    print(f"  {'-' * 6} {'-' * 18} {'-' * 12}")
    for i, (v, c) in enumerate(zip(var_explained, cum_var), 1):
        print(f"  PC{i:<5} {v*100:>6.2f}%           {c*100:>6.2f}%")

    print(f"\n  Number of PCs needed for:")
    for threshold in [0.80, 0.90, 0.95]:
        n_pcs = np.argmax(cum_var >= threshold) + 1
        print(f"    {threshold*100:.0f}% variance: {n_pcs} components")


def print_top_loadings(pca, n_top=5):
    """Print top features loading on each of the first 3 PCs."""
    print(f"\n{'=' * 60}")
    print(f"TOP FEATURE LOADINGS (first 3 PCs)")
    print(f"{'=' * 60}\n")

    loadings = pca.components_

    for pc_idx in range(min(3, loadings.shape[0])):
        pc_loadings = loadings[pc_idx]
        sorted_idx = np.argsort(-np.abs(pc_loadings))

        print(f"  PC{pc_idx + 1} (explains "
              f"{pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance):")
        for i in sorted_idx[:n_top]:
            sign = '+' if pc_loadings[i] > 0 else '-'
            print(f"    {FEATURES[i]:<25} "
                  f"{sign}{abs(pc_loadings[i]):.3f}")
        print()


def plot_pca_panels(pca, X_pca, df, bin_edges):
    """Combined image with 4 PCA panels."""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

    # ── Panel 1: Scree plot ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    var_explained = pca.explained_variance_ratio_ * 100
    pc_numbers = np.arange(1, len(var_explained) + 1)

    ax1.set_facecolor('white')
    ax1.grid(color='lightgrey', linestyle='-',
             linewidth=0.5, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)

    ax1.bar(pc_numbers, var_explained,
            color='steelblue', edgecolor='black', linewidth=0.5,
            zorder=2)

    for x, v in zip(pc_numbers, var_explained):
        ax1.text(x, v + 0.5, f"{v:.1f}%",
                 ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(pc_numbers)
    ax1.set_xlabel('Principal component', fontsize=11)
    ax1.set_ylabel('Variance explained (%)', fontsize=11)
    ax1.set_title('Scree plot - variance per PC',
                  fontsize=12, pad=10)
    ax1.set_ylim(0, var_explained.max() * 1.15)

    # ── Panel 2: Cumulative variance ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cum_var = np.cumsum(var_explained)

    ax2.set_facecolor('white')
    ax2.grid(color='lightgrey', linestyle='-',
             linewidth=0.5, alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)

    ax2.plot(pc_numbers, cum_var, 'o-',
             color='steelblue', linewidth=2,
             markersize=8, zorder=2)

    ax2.axhline(80, color='green', linestyle='--',
                alpha=0.5, zorder=1, label='80% variance')
    ax2.axhline(90, color='orange', linestyle='--',
                alpha=0.5, zorder=1, label='90% variance')
    ax2.axhline(95, color='red', linestyle='--',
                alpha=0.5, zorder=1, label='95% variance')

    ax2.set_xticks(pc_numbers)
    ax2.set_xlabel('Principal component', fontsize=11)
    ax2.set_ylabel('Cumulative variance (%)', fontsize=11)
    ax2.set_title('Cumulative variance explained',
                  fontsize=12, pad=10)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right', fontsize=9)

    # ── Panel 3: PC1 vs PC2 scatter ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('white')
    ax3.grid(color='lightgrey', linestyle='-',
             linewidth=0.5, alpha=0.5, zorder=0)
    ax3.set_axisbelow(True)

    bins = pd.cut(
        df[TARGET],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    ).astype(int)

    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=N_ENTROPY_BINS - 1)

    for bin_idx in range(N_ENTROPY_BINS):
        bin_mask = bins == bin_idx
        if bin_mask.sum() == 0:
            continue

        bin_low  = bin_edges[bin_idx]
        bin_high = bin_edges[bin_idx + 1]
        label = f"Bin {bin_idx}: [{bin_low:.2f}, {bin_high:.2f}]"

        ax3.scatter(
            X_pca[bin_mask, 0],
            X_pca[bin_mask, 1],
            color=cmap(norm(bin_idx)),
            s=12, alpha=0.6, edgecolors='none',
            label=label, zorder=2,
        )

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    ax3.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
    ax3.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
    ax3.set_title('Patches in PC1-PC2 space',
                  fontsize=12, pad=10)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=8, title='Entropy bin', title_fontsize=9,
               frameon=True, framealpha=0.9)

    # ── Panel 4: Feature loadings heatmap ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    n_pcs_show = min(5, len(FEATURES))
    loadings = pca.components_[:n_pcs_show]

    im = ax4.imshow(loadings.T,
                    cmap='RdBu_r',
                    vmin=-0.7, vmax=0.7,
                    aspect='auto')

    ax4.set_xticks(range(n_pcs_show))
    ax4.set_xticklabels([f'PC{i+1}' for i in range(n_pcs_show)],
                        fontsize=10)
    ax4.set_yticks(range(len(FEATURES)))
    ax4.set_yticklabels(FEATURES, fontsize=9)
    ax4.set_title(f'Feature loadings (first {n_pcs_show} PCs)',
                  fontsize=12, pad=10)

    for i in range(len(FEATURES)):
        for j in range(n_pcs_show):
            val = loadings[j, i]
            text_color = 'white' if abs(val) > 0.4 else 'black'
            ax4.text(j, i, f"{val:+.2f}",
                     ha='center', va='center',
                     color=text_color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax4, label='Loading',
                        shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"PCA on {len(FEATURES)} graph + circuity features "
        f"({len(df):,} patches)",
        fontsize=14,
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_path = FIGURES_DIR / "pca_analysis.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Run 13_merge_features.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches from {csv_path.name}")
    print(f"Running PCA on {len(FEATURES)} graph + circuity features\n")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features: {missing}")
        return

    n_nan_rows = df[FEATURES].isna().any(axis=1).sum()
    if n_nan_rows > 0:
        print(f"WARNING: {n_nan_rows} patches have NaN in features. "
              f"Dropping them.")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
        print(f"  Remaining: {len(df):,} patches\n")

    bin_edges = compute_bin_edges(df[TARGET])
    pca, X_pca, X_scaled = run_pca(df)
    print_pca_summary(pca)
    print_top_loadings(pca)
    plot_pca_panels(pca, X_pca, df, bin_edges)


if __name__ == "__main__":
    main()