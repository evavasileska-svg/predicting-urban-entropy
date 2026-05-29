"""
Visualize patches for multiple cities to compare thresholds globally.

Tests cities from different regions to understand whether the
filtering threshold should be global or per-city.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    RAW_DIR, FIGURES_DIR,
    PATCH_SIZE_M, GRID_STEP_M, MIN_SEGMENTS,
)


# cities to visualize for threshold comparison
CITIES_TO_VISUALIZE = {
    'bruges':    'Bruges, Belgium',
    'vienna':    'Vienna, Austria',
    'chicago':   'Chicago, USA',
    'vancouver': 'Vancouver, Canada',
    'bengaluru': 'Bengaluru, India',
    'chennai':   'Chennai, India',
    'cairo':     'Cairo, Egypt',
    'tehran':    'Tehran, Iran',
}


def visualize_city_patches(code: str, name: str):
    """
    Generate patches for one city and visualise them with intersection
    counts so we can decide on a filtering threshold.
    """
    print(f"\nVisualizing patches for {name}...")

    # load and project graph
    graph_path = RAW_DIR / f"{code}.graphml"
    if not graph_path.exists():
        print(f"  Graph file not found: {graph_path}")
        return

    G = ox.load_graphml(graph_path)
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)

    # get bounding box
    minx, miny, maxx, maxy = edges_proj.total_bounds
    print(f"  Bounding box: "
          f"{(maxx - minx) / 1000:.1f} km wide x "
          f"{(maxy - miny) / 1000:.1f} km tall")
    print(f"  Total nodes: {len(nodes_proj):,}")
    print(f"  Total edges: {len(edges_proj):,}")

    # generate patches
    patches = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            patch_geom = box(x, y, x + PATCH_SIZE_M, y + PATCH_SIZE_M)

            edge_mask = edges_proj.geometry.intersects(patch_geom)
            n_segments = edge_mask.sum()

            node_mask = nodes_proj.geometry.within(patch_geom)
            n_intersections = node_mask.sum()

            if n_segments >= MIN_SEGMENTS:
                patches.append({
                    'minx': x,
                    'miny': y,
                    'n_segments':      int(n_segments),
                    'n_intersections': int(n_intersections),
                })

            y += GRID_STEP_M
        x += GRID_STEP_M

    df = pd.DataFrame(patches)
    print(f"  Total patches with >= {MIN_SEGMENTS} segments: {len(df)}")
    print(f"\n  Intersection count distribution:")
    print(df['n_intersections'].describe().round(1).to_string())

    # ── create figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    # left panel: patches coloured by intersection count
    ax = axes[0]

    # set up the background grid first
    ax.set_facecolor('white')
    ax.grid(which='both', color='lightgrey',
            linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # draw street network as background (light grey, on top of grid)
    edges_proj.plot(
        ax=ax, color='lightgrey',
        linewidth=0.3, alpha=0.6, zorder=1,
    )

    # colour patches by intersection count, no border
    cmap = plt.colormaps['viridis']
    max_int = df['n_intersections'].max()

    for _, row in df.iterrows():
        color = cmap(row['n_intersections'] / max_int)
        rect = Rectangle(
            (row['minx'], row['miny']),
            PATCH_SIZE_M, PATCH_SIZE_M,
            facecolor=color,
            edgecolor='none',           # no border on patches
            alpha=0.6,
            zorder=2,
        )
        ax.add_patch(rect)

    ax.set_title(
        f"{name}: {len(df)} patches "
        f"(coloured by intersection count)",
        fontsize=13,
    )
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.set_aspect('equal')

    # colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=max_int),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Intersections per patch",
                 fraction=0.04, pad=0.04)

    # right panel: histogram (no red threshold lines)
    ax = axes[1]
    ax.hist(df['n_intersections'], bins=50,
            color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Intersections per patch")
    ax.set_ylabel("Number of patches")
    ax.set_title(f"Distribution of intersection counts ({name})",
                 fontsize=13)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"patches_{code}_visualization.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved: {output_path}")

    # threshold table in terminal (kept for reference)
    print(f"\n  Effect of different intersection thresholds:")
    for threshold in [20, 30, 40, 50, 60, 75, 100, 125, 150, 200]:
        n_keep = (df['n_intersections'] >= threshold).sum()
        pct = 100 * n_keep / len(df) if len(df) > 0 else 0
        print(f"    >= {threshold:3d} intersections: "
              f"{n_keep:4d} patches kept ({pct:.0f}%)")


def main():
    print(f"Visualizing {len(CITIES_TO_VISUALIZE)} cities for "
          f"threshold comparison...\n")

    for code, name in CITIES_TO_VISUALIZE.items():
        visualize_city_patches(code, name)

    print(f"\n{'=' * 60}")
    print(f"All visualizations complete.")
    print(f"Check results/figures/ for PNGs.")


if __name__ == "__main__":
    main()