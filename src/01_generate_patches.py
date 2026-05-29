"""
Patch generation for global patch-level entropy analysis.

For each city, load the .graphml street graph and divide its bounding box
into 800m x 800m square patches with no overlap. For each patch we record:
  - patch_id (unique across all cities)
  - city code
  - centroid lat/lon (WGS84)
  - bounding box (UTM-projected)
  - number of street segments inside the patch
  - number of intersection nodes within the patch

Output: data/processed/patch_inventory.csv
"""

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box

# allow imports from src/
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    RAW_DIR, PROCESSED_DIR, CITIES,
    PATCH_SIZE_M, GRID_STEP_M, MIN_SEGMENTS,
)


def generate_patches_for_city(code: str, name: str) -> list[dict]:
    """
    Load a city's graph, project to UTM, and generate patches in a
    regular grid. Return a list of patch dictionaries.
    """
    graph_path = RAW_DIR / f"{code}.graphml"
    if not graph_path.exists():
        print(f"  {code:15s} — graph file not found, skipping")
        return []

    try:
        G = ox.load_graphml(graph_path)
    except Exception as e:
        print(f"  {code:15s} — failed to load graph: {e}")
        return []

    # project graph to local UTM zone (in meters)
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)
    crs_proj = edges_proj.crs

    # get the city bounding box in projected coordinates
    minx, miny, maxx, maxy = edges_proj.total_bounds

    # generate patch grid
    patches = []
    patch_idx = 0

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            # patch bounding box
            patch_geom = box(x, y, x + PATCH_SIZE_M, y + PATCH_SIZE_M)

            # find street segments intersecting this patch
            edge_mask = edges_proj.geometry.intersects(patch_geom)
            n_segments = edge_mask.sum()

            # count intersections (nodes) within this patch
            node_mask = nodes_proj.geometry.within(patch_geom)
            n_intersections = node_mask.sum()

            # only keep patches with enough segments
            if n_segments >= MIN_SEGMENTS:
                # get patch centroid in projected and WGS84 coordinates
                cent_proj = patch_geom.centroid
                cent_gdf = gpd.GeoDataFrame(
                    geometry=[cent_proj], crs=crs_proj
                ).to_crs(epsg=4326)
                cent_wgs = cent_gdf.geometry.iloc[0]

                patches.append({
                    'patch_id':        f"{code}_{patch_idx:04d}",
                    'city_code':       code,
                    'patch_idx':       patch_idx,
                    'centroid_lat':    round(cent_wgs.y, 6),
                    'centroid_lon':    round(cent_wgs.x, 6),
                    'minx_utm':        round(x, 2),
                    'miny_utm':        round(y, 2),
                    'maxx_utm':        round(x + PATCH_SIZE_M, 2),
                    'maxy_utm':        round(y + PATCH_SIZE_M, 2),
                    'crs_utm':         crs_proj.to_string(),
                    'n_segments':      int(n_segments),
                    'n_intersections': int(n_intersections),
                })
                patch_idx += 1

            y += GRID_STEP_M
        x += GRID_STEP_M

    return patches


def main():
    print(f"Generating patches for {len(CITIES)} cities...")
    print(f"Patch size: {PATCH_SIZE_M}m x {PATCH_SIZE_M}m, "
          f"step: {GRID_STEP_M}m, min segments: {MIN_SEGMENTS}\n")

    all_patches = []
    start_time = time.time()

    for code, name in CITIES.items():
        t0 = time.time()
        city_patches = generate_patches_for_city(code, name)
        elapsed = time.time() - t0

        if city_patches:
            print(f"  {code:15s} — {len(city_patches):4d} patches "
                  f"({elapsed:5.1f}s)")
            all_patches.extend(city_patches)

    # save inventory
    df = pd.DataFrame(all_patches)
    output_path = PROCESSED_DIR / "patch_inventory.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total patches:    {len(df):,}")
    print(f"Cities processed: {df['city_code'].nunique()}")
    print(f"Time elapsed:     {total_time / 60:.1f} minutes")
    print(f"Saved to:         {output_path}")
    print(f"\nPatches per city (top 10):")
    print(df['city_code'].value_counts().head(10).to_string())
    print(f"\nPatches per city (bottom 10):")
    print(df['city_code'].value_counts().tail(10).to_string())


if __name__ == "__main__":
    main()