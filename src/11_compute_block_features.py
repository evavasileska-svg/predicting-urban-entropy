"""
Compute block-level morphological features for each patch.

A "block" is an enclosed polygon formed by surrounding streets.
For each patch:
  1. Extract street geometries within the patch boundary
  2. Polygonize them to find block polygons
  3. Compute geometric features for each block
  4. Aggregate to patch-level statistics

Features computed (using momepy where available):
  - n_blocks           : count of blocks in patch
  - mean_block_area    : average block area (m²)
  - std_block_area     : standard deviation of block areas
  - cv_block_area      : coefficient of variation (std/mean)
  - mean_block_eri     : Equivalent Rectangular Index (rectangularity)
  - mean_block_compact : compactness (circular ↔ elongated)
  - mean_block_elong   : elongation (aspect ratio)
  - mean_block_convex  : convexity (convex ↔ concave)

Inputs:
  - data/processed/patch_stratified_sample.csv

Outputs:
  - data/processed/patch_block_features.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union

import momepy

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR, PROCESSED_DIR, PATCH_SIZE_M

warnings.filterwarnings('ignore')

# Minimum block area to consider (m²) — filters out tiny slivers
MIN_BLOCK_AREA = 100.0

# Maximum block area as fraction of patch — filters out the patch boundary
PATCH_AREA = PATCH_SIZE_M * PATCH_SIZE_M
MAX_BLOCK_AREA = 0.95 * PATCH_AREA   # exclude polygons >95% of patch area


def extract_blocks_for_patch(edges_in_patch, patch_geom):
    """
    Polygonize street edges within a patch to extract block polygons.

    Returns a GeoDataFrame of block polygons (or None if no blocks).
    """
    if len(edges_in_patch) == 0:
        return None

    # union all edge geometries to a single multilinestring,
    # then polygonize to find enclosed regions
    unioned = unary_union(edges_in_patch.geometry.values)

    try:
        polygons = list(polygonize(unioned))
    except Exception:
        return None

    if len(polygons) == 0:
        return None

    # filter by size: not too small (noise) and not too large (whole patch)
    valid_blocks = []
    for poly in polygons:
        if not poly.is_valid:
            continue
        area = poly.area
        if MIN_BLOCK_AREA <= area <= MAX_BLOCK_AREA:
            # also check the block centroid is inside the patch
            if patch_geom.contains(poly.centroid):
                valid_blocks.append(poly)

    if len(valid_blocks) == 0:
        return None

    blocks_gdf = gpd.GeoDataFrame(
        geometry=valid_blocks, crs=edges_in_patch.crs
    )
    return blocks_gdf


def compute_block_features(blocks_gdf):
    """
    Compute aggregated block features for a patch.
    Returns a dictionary of features.
    """
    n_blocks = len(blocks_gdf)
    if n_blocks == 0:
        return None

    # block areas
    areas = blocks_gdf.geometry.area.values
    mean_block_area = float(np.mean(areas))
    std_block_area  = float(np.std(areas))
    cv_block_area = std_block_area / mean_block_area if mean_block_area > 0 else 0.0

    # shape features via momepy
    # momepy takes a GeoDataFrame and returns a Series of values
    try:
        eri = momepy.equivalent_rectangular_index(blocks_gdf).values
        mean_block_eri = float(np.nanmean(eri))
    except Exception:
        mean_block_eri = np.nan

    try:
        compactness = momepy.circular_compactness(blocks_gdf).values
        mean_block_compact = float(np.nanmean(compactness))
    except Exception:
        mean_block_compact = np.nan

    try:
        elongation = momepy.elongation(blocks_gdf).values
        mean_block_elong = float(np.nanmean(elongation))
    except Exception:
        mean_block_elong = np.nan

    try:
        convexity = momepy.convexity(blocks_gdf).values
        mean_block_convex = float(np.nanmean(convexity))
    except Exception:
        mean_block_convex = np.nan

    return {
        'n_blocks':           int(n_blocks),
        'mean_block_area':    round(mean_block_area, 2),
        'std_block_area':     round(std_block_area, 2),
        'cv_block_area':      round(cv_block_area, 4),
        'mean_block_eri':     round(mean_block_eri, 4)
                              if not np.isnan(mean_block_eri) else np.nan,
        'mean_block_compact': round(mean_block_compact, 4)
                              if not np.isnan(mean_block_compact) else np.nan,
        'mean_block_elong':   round(mean_block_elong, 4)
                              if not np.isnan(mean_block_elong) else np.nan,
        'mean_block_convex':  round(mean_block_convex, 4)
                              if not np.isnan(mean_block_convex) else np.nan,
    }


def process_city_patches(code: str, patches_df: pd.DataFrame):
    """Compute block features for all patches in one city."""
    graph_path = RAW_DIR / f"{code}.graphml"
    if not graph_path.exists():
        print(f"  {code:15s} — graph file not found")
        return []

    print(f"  {code:15s} — loading graph...")
    G = ox.load_graphml(graph_path)
    G_proj = ox.project_graph(G)
    _, edges_proj = ox.graph_to_gdfs(G_proj)

    results = []
    t0 = time.time()

    for _, patch in patches_df.iterrows():
        # build patch geometry
        patch_geom = box(
            patch['minx_utm'],
            patch['miny_utm'],
            patch['maxx_utm'],
            patch['maxy_utm'],
        )

        # find edges that intersect the patch
        edge_mask = edges_proj.geometry.intersects(patch_geom)
        edges_in_patch = edges_proj[edge_mask].copy()

        # clip edges to patch boundary
        edges_in_patch['geometry'] = edges_in_patch.geometry.intersection(
            patch_geom
        )

        # remove empty or invalid geometries after clipping
        edges_in_patch = edges_in_patch[
            ~edges_in_patch.geometry.is_empty &
            edges_in_patch.geometry.is_valid
        ]

        if len(edges_in_patch) == 0:
            continue

        # extract blocks
        blocks_gdf = extract_blocks_for_patch(edges_in_patch, patch_geom)
        if blocks_gdf is None or len(blocks_gdf) == 0:
            # patch has no blocks - skip or record zeros
            results.append({
                'patch_id':           patch['patch_id'],
                'city_code':          code,
                'n_blocks':           0,
                'mean_block_area':    np.nan,
                'std_block_area':     np.nan,
                'cv_block_area':      np.nan,
                'mean_block_eri':     np.nan,
                'mean_block_compact': np.nan,
                'mean_block_elong':   np.nan,
                'mean_block_convex':  np.nan,
            })
            continue

        # compute features
        features = compute_block_features(blocks_gdf)
        if features is None:
            continue

        results.append({
            'patch_id':  patch['patch_id'],
            'city_code': code,
            **features,
        })

    elapsed = time.time() - t0
    n_with_blocks = sum(1 for r in results if r['n_blocks'] > 0)
    print(f"  {code:15s} — {len(results):4d} / {len(patches_df):4d} "
          f"patches processed, {n_with_blocks} with blocks "
          f"({elapsed:5.1f}s)")
    return results


def main():
    sample_path = PROCESSED_DIR / "patch_stratified_sample.csv"
    if not sample_path.exists():
        print(f"ERROR: {sample_path} not found.")
        print(f"Run 08_stratified_sample.py first.")
        return

    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df):,} stratified patches from "
          f"{df['city_code'].nunique()} cities\n")
    print(f"Computing block features (this may take 30-60 minutes)\n")

    all_results = []
    start_time = time.time()

    for code, group in df.groupby('city_code'):
        city_results = process_city_patches(code, group)
        all_results.extend(city_results)

        # save intermediate results after each city
        intermediate_df = pd.DataFrame(all_results)
        intermediate_path = PROCESSED_DIR / "patch_block_features.csv"
        intermediate_df.to_csv(intermediate_path, index=False)

    total_time = time.time() - start_time

    # final save
    results_df = pd.DataFrame(all_results)
    output_path = PROCESSED_DIR / "patch_block_features.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Block features computed for {len(results_df):,} patches")
    print(f"Time elapsed:         {total_time / 60:.1f} minutes")
    print(f"Saved: {output_path}")
    print()

    # diagnostics
    n_zero_blocks = (results_df['n_blocks'] == 0).sum()
    print(f"Patches with no blocks extracted: {n_zero_blocks}")
    print()

    # summary statistics
    valid_df = results_df[results_df['n_blocks'] > 0]
    print(f"Block feature statistics (excluding patches with 0 blocks):")
    print(valid_df.describe()[[
        'n_blocks', 'mean_block_area', 'cv_block_area',
        'mean_block_eri', 'mean_block_compact', 'mean_block_elong',
        'mean_block_convex',
    ]].round(3).to_string())
    print()

    # most rectangular patches (sanity check)
    print(f"5 most rectangular patches (high ERI):")
    top_eri = valid_df.nlargest(5, 'mean_block_eri')
    print(top_eri[['patch_id', 'city_code', 'mean_block_eri',
                   'n_blocks']].to_string(index=False))


if __name__ == "__main__":
    main()