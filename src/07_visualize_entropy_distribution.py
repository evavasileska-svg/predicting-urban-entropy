"""
Visualize the patch-level entropy distribution.

Produces four figures to understand the entropy data:
  1. Overall histogram of Ho across all patches
  2. Per-city entropy boxplots (sorted by median)
  3. Comparison of Ho, Hw, and phi distributions
  4. Spatial map of entropy for representative cities

All figures saved to results/figures/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    RAW_DIR, PROCESSED_DIR, FIGURES_DIR,
    PATCH_SIZE_M,
)


# representative cities for spatial entropy maps
SPATIAL_MAP_CITIES = {
    'chicago':  'Chicago, USA',
    'hanoi':    'Hanoi, Vietnam',
    'vienna':   'Vienna, Austria',
    'cairo':    'Cairo, Egypt',
}


def figure_1_overall_histogram(df: pd.DataFrame):
    """Overall histogram of Ho with bands marked."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # plot histogram with 50 bins
    ax.hist(df['entropy_normalised'], bins=50,
            color='steelblue', edgecolor='black',
            alpha=0.7, zorder=2)

    # mark entropy bands
    ax.axvline(0.80, color='green', linestyle='--',
               alpha=0.6, zorder=3,
               label='Ordered/Medium boundary (Ho=0.80)')
    ax.axvline(0.95, color='red', linestyle='--',
               alpha=0.6, zorder=3,
               label='Medium/Disordered boundary (Ho=0.95)')

    ax.set_xlabel("Ho (normalised street bearing entropy)",
                  fontsize=12)
    ax.set_ylabel("Number of patches", fontsize=12)
    ax.set_title(f"Patch-level entropy distribution "
                 f"({len(df):,} patches across "
                 f"{df['city_code'].nunique()} cities)",
                 fontsize=13)
    ax.legend(fontsize=10)

    # add band counts as text
    n_ordered = (df['entropy_normalised'] < 0.80).sum()
    n_medium = ((df['entropy_normalised'] >= 0.80) &
                (df['entropy_normalised'] < 0.95)).sum()
    n_disordered = (df['entropy_normalised'] >= 0.95).sum()
    total = len(df)

    info_text = (
        f"Ordered    (Ho<0.80):     {n_ordered:6,} ({100*n_ordered/total:.1f}%)\n"
        f"Medium     (0.80–0.95):   {n_medium:6,} ({100*n_medium/total:.1f}%)\n"
        f"Disordered (Ho≥0.95):     {n_disordered:6,} ({100*n_disordered/total:.1f}%)"
    )
    ax.text(0.02, 0.97, info_text,
            transform=ax.transAxes,
            fontsize=10, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.9, edgecolor='grey'))

    plt.tight_layout()
    path = FIGURES_DIR / "entropy_01_overall_histogram.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def figure_2_per_city_boxplots(df: pd.DataFrame):
    """Boxplot of Ho per city, sorted by median."""
    # sort cities by median Ho
    medians = df.groupby('city_code')['entropy_normalised'].median()
    city_order = medians.sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(16, 20))

    ax.set_facecolor('white')
    ax.grid(axis='x', color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # collect data per city
    data_per_city = [
        df[df['city_code'] == c]['entropy_normalised'].values
        for c in city_order
    ]
    counts_per_city = [len(d) for d in data_per_city]

    # color cities by median Ho using viridis
    norm = Normalize(vmin=medians.min(), vmax=medians.max())
    colors = [cm.viridis(norm(medians[c])) for c in city_order]

    bp = ax.boxplot(
        data_per_city,
        vert=False,
        patch_artist=True,
        widths=0.65,
        showfliers=False,
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    for whisker in bp['whiskers']:
        whisker.set_color('grey')
        whisker.set_linewidth(0.8)
    for cap in bp['caps']:
        cap.set_color('grey')
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    # labels with patch counts
    labels = [f"{c} (n={n:,})"
              for c, n in zip(city_order, counts_per_city)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Ho (normalised street bearing entropy)",
                  fontsize=12)
    ax.set_title(
        f"Per-city entropy distributions "
        f"(sorted by median Ho, lowest = most ordered)",
        fontsize=13,
    )

    ax.axvline(0.80, color='green', linestyle='--', alpha=0.4)
    ax.axvline(0.95, color='red', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = FIGURES_DIR / "entropy_02_per_city_boxplots.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def figure_3_three_metrics(df: pd.DataFrame):
    """Compare Ho, Hw, and phi distributions side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(color='lightgrey', linestyle='-',
                linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    # Ho histogram
    axes[0].hist(df['entropy_normalised'], bins=50,
                 color='steelblue', edgecolor='black',
                 alpha=0.7, zorder=2)
    axes[0].set_xlabel("Ho (unweighted)", fontsize=12)
    axes[0].set_ylabel("Number of patches", fontsize=12)
    axes[0].set_title(
        f"Ho — unweighted entropy\n"
        f"mean={df['entropy_normalised'].mean():.3f}, "
        f"std={df['entropy_normalised'].std():.3f}",
        fontsize=11,
    )

    # Hw histogram
    axes[1].hist(df['entropy_weighted_norm'], bins=50,
                 color='seagreen', edgecolor='black',
                 alpha=0.7, zorder=2)
    axes[1].set_xlabel("Hw (length-weighted)", fontsize=12)
    axes[1].set_title(
        f"Hw — length-weighted entropy\n"
        f"mean={df['entropy_weighted_norm'].mean():.3f}, "
        f"std={df['entropy_weighted_norm'].std():.3f}",
        fontsize=11,
    )

    # phi histogram
    axes[2].hist(df['phi'], bins=50,
                 color='indianred', edgecolor='black',
                 alpha=0.7, zorder=2)
    axes[2].set_xlabel("φ (orientation order)", fontsize=12)
    axes[2].set_title(
        f"φ — orientation order\n"
        f"mean={df['phi'].mean():.3f}, "
        f"std={df['phi'].std():.3f}",
        fontsize=11,
    )

    plt.tight_layout()
    path = FIGURES_DIR / "entropy_03_three_metrics.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def figure_4_spatial_maps(df: pd.DataFrame):
    """Spatial map of entropy for representative cities."""
    n_cities = len(SPATIAL_MAP_CITIES)
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    # use a colormap where low Ho = blue (ordered)
    # and high Ho = yellow/red (disordered)
    cmap = plt.colormaps['RdYlBu_r']
    norm = Normalize(vmin=0.3, vmax=1.0)

    for i, (code, name) in enumerate(SPATIAL_MAP_CITIES.items()):
        ax = axes[i]

        # subset to this city
        city_df = df[df['city_code'] == code]
        if len(city_df) == 0:
            ax.set_visible(False)
            continue

        ax.set_facecolor('white')
        ax.grid(color='lightgrey', linestyle='-',
                linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # try to load and draw the city's street network as background
        graph_path = RAW_DIR / f"{code}.graphml"
        if graph_path.exists():
            try:
                G = ox.load_graphml(graph_path)
                G_proj = ox.project_graph(G)
                _, edges_proj = ox.graph_to_gdfs(G_proj)
                edges_proj.plot(ax=ax, color='lightgrey',
                                linewidth=0.2, alpha=0.5, zorder=1)
            except Exception as e:
                print(f"  Warning: could not draw graph for {code}: {e}")

        # draw each patch as a coloured rectangle (no border)
        for _, row in city_df.iterrows():
            color = cmap(norm(row['entropy_normalised']))
            rect = Rectangle(
                (row['minx_utm'], row['miny_utm']),
                PATCH_SIZE_M, PATCH_SIZE_M,
                facecolor=color,
                edgecolor='none',
                alpha=0.7,
                zorder=2,
            )
            ax.add_patch(rect)

        ax.set_title(
            f"{name}\n{len(city_df)} patches, "
            f"mean Ho={city_df['entropy_normalised'].mean():.3f}, "
            f"range [{city_df['entropy_normalised'].min():.2f}, "
            f"{city_df['entropy_normalised'].max():.2f}]",
            fontsize=12,
        )
        ax.set_aspect('equal')
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")

    # add a shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.025, pad=0.05, aspect=40)
    cbar.set_label(
        "Ho (blue = ordered grid, red = disordered organic)",
        fontsize=12,
    )

    path = FIGURES_DIR / "entropy_04_spatial_maps.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    # load the entropy data
    entropy_path = PROCESSED_DIR / "patch_entropy.csv"
    if not entropy_path.exists():
        print(f"ERROR: {entropy_path} not found.")
        print(f"Run 06_compute_patch_entropy.py first.")
        return

    df = pd.read_csv(entropy_path)
    print(f"Loaded {len(df):,} patches with entropy from "
          f"{df['city_code'].nunique()} cities\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 1: overall histogram...")
    figure_1_overall_histogram(df)

    print("\nGenerating Figure 2: per-city boxplots...")
    figure_2_per_city_boxplots(df)

    print("\nGenerating Figure 3: three metrics comparison...")
    figure_3_three_metrics(df)

    print("\nGenerating Figure 4: spatial entropy maps...")
    figure_4_spatial_maps(df)

    print(f"\n{'=' * 60}")
    print(f"All entropy visualisations saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()