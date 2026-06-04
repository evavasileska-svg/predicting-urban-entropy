"""
Plot each feature against the entropy target.

For each of the 13 graph + circuity features:
  - Scatter plot of feature vs entropy_normalised
  - Points colored by entropy bin (10 bins matching stratification)
  - Title shows Pearson correlation
  - One PNG per feature

Outputs:
  - results/figures/feature_vs_entropy/feature_vs_entropy_{name}.png
    (one PNG per feature)
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

# output directory for individual feature plots
OUTPUT_DIR = FIGURES_DIR / "feature_vs_entropy"


def compute_bin_edges(target_values):
    """
    Reproduce the stratification bin edges from script 08.
    The stratification used the full patch_entropy.csv range.
    """
    entropy_path = PROCESSED_DIR / "patch_entropy.csv"
    if entropy_path.exists():
        full = pd.read_csv(entropy_path)
        ho_min = full[TARGET].min()
        ho_max = full[TARGET].max()
    else:
        # fallback: use the sample's range
        ho_min = target_values.min()
        ho_max = target_values.max()

    return np.linspace(ho_min, ho_max + 1e-6, 11)   # 10 bins → 11 edges


def plot_feature_vs_target(df, feature, bin_edges):
    """Create one scatter plot of feature vs entropy."""
    # subset to non-null rows for this feature
    valid = df.dropna(subset=[feature, TARGET])
    if len(valid) == 0:
        print(f"  WARNING: no valid data for {feature}, skipping")
        return

    # assign each patch to an entropy bin
    bins = pd.cut(
        valid[TARGET],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    ).astype(int)

    # compute Pearson correlation
    corr = valid[feature].corr(valid[TARGET])

    # set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # use viridis colormap — low entropy = purple/blue, high = yellow
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=9)

    # plot each bin separately so we can add a legend
    for bin_idx in range(10):
        bin_mask = bins == bin_idx
        if bin_mask.sum() == 0:
            continue

        bin_low  = bin_edges[bin_idx]
        bin_high = bin_edges[bin_idx + 1]
        label = f"Bin {bin_idx}: [{bin_low:.2f}, {bin_high:.2f}]"

        ax.scatter(
            valid.loc[bin_mask, feature],
            valid.loc[bin_mask, TARGET],
            color=cmap(norm(bin_idx)),
            s=18,
            alpha=0.6,
            edgecolors='none',
            label=label,
            zorder=2,
        )

    # entropy band reference lines
    ax.axhline(0.80, color='green', linestyle='--', alpha=0.4, zorder=3)
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.4, zorder=3)

    # labels and title
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel(TARGET, fontsize=12)
    ax.set_title(
        f"{feature} vs {TARGET}\n"
        f"Pearson r = {corr:.3f}  (n = {len(valid):,})",
        fontsize=13,
    )

    # legend on the side, outside the plot area
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=9,
        title="Entropy bin",
        title_fontsize=10,
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"feature_vs_entropy_{feature}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}  (r = {corr:+.3f})")


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Run 13_merge_features.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches from {csv_path.name}\n")

    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # compute stratification bin edges
    bin_edges = compute_bin_edges(df[TARGET])
    print(f"Using stratification bin edges:")
    print(f"  Ho range: [{bin_edges[0]:.4f}, {bin_edges[-1]:.4f}]")
    print(f"  10 bins, width {(bin_edges[-1] - bin_edges[0]) / 10:.4f}\n")

    # plot each feature
    print(f"Generating {len(FEATURES_TO_PLOT)} feature-vs-target plots:\n")
    for feature in FEATURES_TO_PLOT:
        if feature not in df.columns:
            print(f"  WARNING: {feature} not found in dataset, skipping")
            continue
        plot_feature_vs_target(df, feature, bin_edges)

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()