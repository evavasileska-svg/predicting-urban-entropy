"""
Plot the distribution of all 13 graph + circuity features as a combined image.

For each feature:
  - Histogram with 40 bins
  - Stacked by entropy bin (10 colors, viridis colormap)
  - Title shows feature name and key statistics

Single image with 5×3 grid layout (13 features + 2 empty cells).

Outputs:
  - results/figures/feature_distributions_combined.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

# graph + circuity features (all 100% complete in the dataset)
FEATURES_TO_PLOT = [
    'n_4way', 'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_degree',
    'mean_edge_length', 'total_edge_length',
    'meshedness',
    'intersection_density', 'street_density',
    'circuity',
]

N_BINS_HIST = 40   # histogram bins
N_ENTROPY_BINS = 10
GRID_ROWS = 5
GRID_COLS = 3


def compute_bin_edges(target_values):
    """Reproduce the stratification bin edges from script 08."""
    entropy_path = PROCESSED_DIR / "patch_entropy.csv"
    if entropy_path.exists():
        full = pd.read_csv(entropy_path)
        ho_min = full[TARGET].min()
        ho_max = full[TARGET].max()
    else:
        ho_min = target_values.min()
        ho_max = target_values.max()

    return np.linspace(ho_min, ho_max + 1e-6, N_ENTROPY_BINS + 1)


def plot_distributions(df, bin_edges):
    """Create the combined distribution plot."""
    # set up figure
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(18, 22),
    )
    axes_flat = axes.flatten()

    # assign each patch to an entropy bin
    df = df.copy()
    df['bin'] = pd.cut(
        df[TARGET],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    ).astype(int)

    # viridis colormap for entropy bins
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=N_ENTROPY_BINS - 1)
    bin_colors = [cmap(norm(i)) for i in range(N_ENTROPY_BINS)]

    # plot each feature
    for idx, feature in enumerate(FEATURES_TO_PLOT):
        ax = axes_flat[idx]
        ax.set_facecolor('white')
        ax.grid(color='lightgrey', linestyle='-',
                linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        if feature not in df.columns:
            ax.text(0.5, 0.5, f"{feature}\nNOT IN DATASET",
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # split feature values by entropy bin
        values_by_bin = [
            df.loc[df['bin'] == b, feature].dropna().values
            for b in range(N_ENTROPY_BINS)
        ]

        # determine common histogram bins across all values
        all_values = df[feature].dropna()
        if len(all_values) == 0:
            ax.text(0.5, 0.5, "no data",
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            continue
        hist_bins = np.linspace(
            all_values.min(), all_values.max(), N_BINS_HIST + 1
        )

        # stacked histogram
        ax.hist(
            values_by_bin,
            bins=hist_bins,
            stacked=True,
            color=bin_colors,
            edgecolor='black',
            linewidth=0.3,
            zorder=2,
        )

        # title and labels
        feat_mean = all_values.mean()
        feat_std  = all_values.std()
        feat_min  = all_values.min()
        feat_max  = all_values.max()

        ax.set_title(
            f"{feature}\n"
            f"μ = {feat_mean:.3g}, σ = {feat_std:.3g}, "
            f"range = [{feat_min:.3g}, {feat_max:.3g}]",
            fontsize=10,
        )
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('count', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    # hide unused subplots
    for idx in range(len(FEATURES_TO_PLOT), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # add a shared legend at the bottom
    legend_handles = []
    legend_labels = []
    for i in range(N_ENTROPY_BINS):
        bin_low  = bin_edges[i]
        bin_high = bin_edges[i + 1]
        patch = plt.Rectangle((0, 0), 1, 1,
                              facecolor=bin_colors[i],
                              edgecolor='black',
                              linewidth=0.3)
        legend_handles.append(patch)
        legend_labels.append(
            f"Bin {i}: [{bin_low:.2f}, {bin_high:.2f}]"
        )

    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        ncol=5,
        fontsize=10,
        title="Entropy bin (Ho range)",
        title_fontsize=11,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        framealpha=0.9,
    )

    # super title
    fig.suptitle(
        "Distribution of street network features, stacked by entropy bin",
        fontsize=14,
        y=0.995,
    )

    # layout - leave space at the bottom for the legend
    plt.tight_layout(rect=[0, 0.04, 1, 0.985])

    output_path = FIGURES_DIR / "feature_distributions_combined.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Run 13_merge_features.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches from {csv_path.name}\n")

    # compute stratification bin edges
    bin_edges = compute_bin_edges(df[TARGET])
    print(f"Using stratification bin edges:")
    print(f"  Ho range: [{bin_edges[0]:.4f}, {bin_edges[-1]:.4f}]\n")

    plot_distributions(df, bin_edges)

    print(f"\nDone.")


if __name__ == "__main__":
    main()