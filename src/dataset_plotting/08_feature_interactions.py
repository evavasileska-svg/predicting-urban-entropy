import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

# the 9 features after dropping redundancies
FEATURES = [
    'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_edge_length', 'total_edge_length',
    'circuity',
    'distance_to_center_km',
]

# focus on the most interesting feature pairs to plot
# (ordered by expected interaction interest)
INTERESTING_PAIRS = [
    ('proportion_4way',     'circuity'),               # straight grid vs winding grid
    ('proportion_4way',     'mean_edge_length'),       # grid + block size
    ('proportion_4way',     'distance_to_center_km'),  # grid + position
    ('proportion_deadend',  'distance_to_center_km'),  # suburb + position
    ('total_edge_length',   'distance_to_center_km'),  # density + position
    ('mean_edge_length',    'distance_to_center_km'),  # block size + position
    ('proportion_3way',     'proportion_deadend'),     # organic types
    ('circuity',            'mean_edge_length'),       # winding + block size
    ('total_edge_length',   'proportion_4way'),        # density + grid
]

N_BINS = 10
MIN_PATCHES_PER_BIN = 3  # bins with fewer patches get masked out


def plot_pair_heatmap(df, feature_x, feature_y, ax):
    """Plot one 2D heatmap of feature_x vs feature_y, colored by mean entropy."""

    # use quantile-based binning so each bin has approximately equal patches
    df_clean = df.dropna(subset=[feature_x, feature_y, TARGET])

    try:
        x_bins = pd.qcut(
            df_clean[feature_x], q=N_BINS,
            duplicates='drop', retbins=True,
        )
        x_bin_idx = x_bins[0].cat.codes
        x_edges = x_bins[1]
    except Exception:
        x_bin_idx = pd.cut(df_clean[feature_x], bins=N_BINS, labels=False)
        x_edges = np.linspace(
            df_clean[feature_x].min(),
            df_clean[feature_x].max(),
            N_BINS + 1,
        )

    try:
        y_bins = pd.qcut(
            df_clean[feature_y], q=N_BINS,
            duplicates='drop', retbins=True,
        )
        y_bin_idx = y_bins[0].cat.codes
        y_edges = y_bins[1]
    except Exception:
        y_bin_idx = pd.cut(df_clean[feature_y], bins=N_BINS, labels=False)
        y_edges = np.linspace(
            df_clean[feature_y].min(),
            df_clean[feature_y].max(),
            N_BINS + 1,
        )

    # compute mean entropy per (x_bin, y_bin)
    actual_nx = int(x_bin_idx.max()) + 1
    actual_ny = int(y_bin_idx.max()) + 1
    heatmap = np.full((actual_ny, actual_nx), np.nan)
    counts = np.zeros((actual_ny, actual_nx), dtype=int)

    for i in range(actual_ny):
        for j in range(actual_nx):
            mask = (x_bin_idx == j) & (y_bin_idx == i)
            n_patches = mask.sum()
            counts[i, j] = n_patches
            if n_patches >= MIN_PATCHES_PER_BIN:
                heatmap[i, j] = df_clean.loc[mask, TARGET].mean()

    # plot
    ax.set_facecolor('white')
    im = ax.imshow(heatmap, cmap='viridis', origin='lower',
                   aspect='auto', vmin=0.4, vmax=1.0)

    # annotate each cell with count and entropy
    for i in range(actual_ny):
        for j in range(actual_nx):
            if not np.isnan(heatmap[i, j]):
                ent = heatmap[i, j]
                n = counts[i, j]
                # choose text color based on background brightness
                text_color = 'white' if ent < 0.7 else 'black'
                ax.text(j, i, f'{ent:.2f}\n(n={n})',
                        ha='center', va='center',
                        color=text_color, fontsize=6.5)

    # ticks with bin edges
    # need (n_bins + 1) tick positions to show all (n_bins + 1) edges
    ax.set_xticks(np.arange(actual_nx + 1) - 0.5)
    ax.set_xticklabels([f'{e:.2f}' for e in x_edges[:actual_nx + 1]],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticks(np.arange(actual_ny + 1) - 0.5)
    ax.set_yticklabels([f'{e:.2f}' for e in y_edges[:actual_ny + 1]],
                       fontsize=7)

    ax.set_xlabel(feature_x, fontsize=9)
    ax.set_ylabel(feature_y, fontsize=9)
    ax.set_title(
        f'{feature_y} vs {feature_x}\n'
        f'(colored by mean entropy)',
        fontsize=10, pad=8,
    )

    return im


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches\n")

    missing_pairs = [
        (fx, fy) for fx, fy in INTERESTING_PAIRS
        if fx not in df.columns or fy not in df.columns
    ]
    if missing_pairs:
        print(f"ERROR: missing features for these pairs: {missing_pairs}")
        return

    n_pairs = len(INTERESTING_PAIRS)
    print(f"Generating {n_pairs} feature interaction heatmaps...\n")

    # arrange in 3x3 grid
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes_flat = axes.flatten()

    for idx, (fx, fy) in enumerate(INTERESTING_PAIRS):
        ax = axes_flat[idx]
        im = plot_pair_heatmap(df, fx, fy, ax)
        print(f"  Plotted: {fy} vs {fx}")

    # hide unused subplots
    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].axis('off')

    # single colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.7, pad=0.02,
                        label='Mean entropy_normalised')
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Feature interaction heatmaps: how feature pairs jointly predict entropy",
        fontsize=14, y=0.995,
    )

    output_path = FIGURES_DIR / "feature_interactions.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}\n")

    # also print interesting findings per pair
    print(f"{'=' * 70}")
    print(f"PER-PAIR INTERACTION STRENGTH")
    print(f"{'=' * 70}\n")
    print(f"  (variation in mean entropy across bins suggests interaction)")
    print()
    print(f"  {'Pair':<55} {'Entropy range':<15}")
    print(f"  {'-' * 55} {'-' * 15}")

    for fx, fy in INTERESTING_PAIRS:
        df_clean = df.dropna(subset=[fx, fy, TARGET])
        try:
            x_bin_idx = pd.qcut(df_clean[fx], q=N_BINS,
                                duplicates='drop', labels=False)
            y_bin_idx = pd.qcut(df_clean[fy], q=N_BINS,
                                duplicates='drop', labels=False)
        except Exception:
            continue

        actual_nx = int(x_bin_idx.max()) + 1
        actual_ny = int(y_bin_idx.max()) + 1

        bin_means = []
        for i in range(actual_ny):
            for j in range(actual_nx):
                mask = (x_bin_idx == j) & (y_bin_idx == i)
                if mask.sum() >= MIN_PATCHES_PER_BIN:
                    bin_means.append(df_clean.loc[mask, TARGET].mean())

        if len(bin_means) > 0:
            ent_range = max(bin_means) - min(bin_means)
            print(f"  {fy + ' x ' + fx:<55} {ent_range:.4f}")


if __name__ == "__main__":
    main()