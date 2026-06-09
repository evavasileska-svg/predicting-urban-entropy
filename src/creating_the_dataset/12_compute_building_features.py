"""
Compute building footprint features for each patch.

For each city:
  1. Download building footprints from OSM (cached to disk, with retries)
  2. For each patch, find buildings within the patch boundary
  3. Compute building features at patch level

Features computed:
  - n_buildings              : count of buildings in patch
  - building_density         : fraction of patch covered by buildings
  - mean_building_footprint  : average building area (m²)
  - std_building_footprint   : standard deviation of footprints
  - mean_nn_distance         : mean nearest-neighbour distance between buildings
  - std_nn_distance          : standard deviation of NN distances

Inputs:
  - data/processed/patch_stratified_sample.csv

Outputs:
  - data/processed/patch_building_features.csv
  - data/raw/buildings/{city_code}_buildings.gpkg (cached downloads)
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RAW_DIR, PROCESSED_DIR, PATCH_SIZE_M, CITIES

warnings.filterwarnings('ignore')

# directory to cache building downloads
BUILDINGS_DIR = RAW_DIR / "buildings"
BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)

# patch area in m² (constant since all patches are 800m × 800m)
PATCH_AREA_M2 = PATCH_SIZE_M * PATCH_SIZE_M

# retry configuration for downloads
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds between retry attempts


def download_buildings_for_city(code: str, name: str, patches_df: pd.DataFrame):
    """
    Download all building footprints for a city, save as GeoPackage.
    Uses the bounding box of the city's patches to define the area.
    Retries up to MAX_RETRIES times on failure.
    Does NOT save empty placeholder caches on failure.
    """
    cache_path = BUILDINGS_DIR / f"{code}_buildings.gpkg"

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"  {code:15s} — buildings cached ({size_mb:.1f} MB)")
        return cache_path

    # determine the bounding box of all patches for this city
    minx = patches_df['minx_utm'].min()
    miny = patches_df['miny_utm'].min()
    maxx = patches_df['maxx_utm'].max()
    maxy = patches_df['maxy_utm'].max()
    crs_utm = patches_df['crs_utm'].iloc[0]

    # convert UTM bbox to WGS84 for OSM query
    bbox_utm = gpd.GeoDataFrame(
        geometry=[box(minx, miny, maxx, maxy)], crs=crs_utm
    )
    bbox_wgs = bbox_utm.to_crs(epsg=4326)
    wgs_bounds = bbox_wgs.total_bounds   # [west, south, east, north]

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_label = (
            f"attempt {attempt}/{MAX_RETRIES}" if attempt > 1 else ""
        )
        print(f"  {code:15s} — downloading buildings from OSM {attempt_label}...")

        try:
            t0 = time.time()
            # use features_from_place instead of features_from_bbox
            # place-based queries use the city's administrative boundary
            # which is more reliable than raw bbox queries for large cities
            buildings = ox.features_from_place(
                name, tags={'building': True}
            )
            elapsed = time.time() - t0

            if buildings is None or len(buildings) == 0:
                # NO data returned - retry if attempts remaining
                if attempt < MAX_RETRIES:
                    print(f"  {code:15s} — no data returned, "
                          f"waiting {RETRY_DELAY}s and retrying...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    # final attempt failed - DO NOT save empty placeholder
                    print(f"  {code:15s} — FAILED after "
                          f"{MAX_RETRIES} attempts, no cache saved")
                    return None

            # keep only polygon geometries
            buildings = buildings[buildings.geometry.type.isin(
                ['Polygon', 'MultiPolygon']
            )].copy()

            # drop all attribute columns - we only need geometry
            # OSM building data has hundreds of tag columns with problematic
            # names (colons, mixed case) that GeoPackage cannot save
            buildings = buildings[['geometry']].copy()

            # project to local UTM
            buildings_proj = buildings.to_crs(crs_utm)

            # save
            buildings_proj.to_file(cache_path, driver='GPKG')

        except Exception as e:
            # actual error - retry if attempts remaining
            if attempt < MAX_RETRIES:
                print(f"  {code:15s} — error: {type(e).__name__}, "
                      f"waiting {RETRY_DELAY}s and retrying...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                print(f"  {code:15s} — FAILED after "
                      f"{MAX_RETRIES} attempts: {e}")
                return None

    return None


def compute_building_features(buildings_in_patch, patch_geom):
    """
    Compute aggregated building features for a patch.
    Returns a dictionary of features.
    """
    n_buildings = len(buildings_in_patch)

    if n_buildings == 0:
        return {
            'n_buildings':             0,
            'building_density':        0.0,
            'mean_building_footprint': np.nan,
            'std_building_footprint':  np.nan,
            'mean_nn_distance':        np.nan,
            'std_nn_distance':         np.nan,
        }

    # clip buildings to patch boundary so we count only the part
    # of each building actually inside the patch
    clipped = buildings_in_patch.geometry.intersection(patch_geom)
    areas = np.array([
        g.area for g in clipped if not g.is_empty
    ])
    if len(areas) == 0:
        return {
            'n_buildings':             0,
            'building_density':        0.0,
            'mean_building_footprint': np.nan,
            'std_building_footprint':  np.nan,
            'mean_nn_distance':        np.nan,
            'std_nn_distance':         np.nan,
        }

    total_building_area = float(np.sum(areas))
    building_density = total_building_area / PATCH_AREA_M2
    mean_footprint   = float(np.mean(areas))
    std_footprint    = float(np.std(areas))

    # nearest-neighbour distance between building centroids
    if n_buildings >= 2:
        centroids = np.array([
            [g.centroid.x, g.centroid.y]
            for g in buildings_in_patch.geometry
            if not g.is_empty and g.is_valid
        ])
        if len(centroids) >= 2:
            tree = cKDTree(centroids)
            # k=2: nearest neighbour besides self
            dists, _ = tree.query(centroids, k=2)
            nn_distances = dists[:, 1]   # exclude self (column 0)
            mean_nn = float(np.mean(nn_distances))
            std_nn  = float(np.std(nn_distances))
        else:
            mean_nn = np.nan
            std_nn  = np.nan
    else:
        mean_nn = np.nan
        std_nn  = np.nan

    return {
        'n_buildings':             int(n_buildings),
        'building_density':        round(building_density, 4),
        'mean_building_footprint': round(mean_footprint, 2),
        'std_building_footprint':  round(std_footprint, 2),
        'mean_nn_distance':        round(mean_nn, 2)
                                   if not np.isnan(mean_nn) else np.nan,
        'std_nn_distance':         round(std_nn, 2)
                                   if not np.isnan(std_nn) else np.nan,
    }


def process_city_patches(code: str, name: str, patches_df: pd.DataFrame):
    """Process all patches for one city."""
    # ensure buildings are downloaded
    buildings_path = download_buildings_for_city(code, name, patches_df)
    if buildings_path is None or not buildings_path.exists():
        print(f"  {code:15s} — no building data available, skipping")
        return []

    # load buildings from cache
    try:
        buildings = gpd.read_file(buildings_path)
    except Exception as e:
        print(f"  {code:15s} — failed to load buildings: {e}")
        return []

    if len(buildings) == 0:
        print(f"  {code:15s} — no buildings in this city")
        # return zero-building results for all patches
        return [
            {
                'patch_id':                p['patch_id'],
                'city_code':               code,
                'n_buildings':             0,
                'building_density':        0.0,
                'mean_building_footprint': np.nan,
                'std_building_footprint':  np.nan,
                'mean_nn_distance':        np.nan,
                'std_nn_distance':         np.nan,
            }
            for _, p in patches_df.iterrows()
        ]

    # build spatial index for fast patch-by-patch queries
    buildings_sindex = buildings.sindex

    results = []
    t0 = time.time()

    for _, patch in patches_df.iterrows():
        patch_geom = box(
            patch['minx_utm'],
            patch['miny_utm'],
            patch['maxx_utm'],
            patch['maxy_utm'],
        )

        # query candidate buildings via spatial index (fast)
        candidate_idx = list(buildings_sindex.intersection(patch_geom.bounds))
        candidates = buildings.iloc[candidate_idx]

        # refine: keep only those that actually intersect the patch
        intersects_mask = candidates.geometry.intersects(patch_geom)
        in_patch = candidates[intersects_mask]

        features = compute_building_features(in_patch, patch_geom)
        results.append({
            'patch_id':  patch['patch_id'],
            'city_code': code,
            **features,
        })

    elapsed = time.time() - t0
    n_with_buildings = sum(1 for r in results if r['n_buildings'] > 0)
    print(f"  {code:15s} — {len(results):4d} patches processed, "
          f"{n_with_buildings} with buildings "
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
    print(f"Computing building features\n")
    print(f"Note: buildings are downloaded once per city and cached.")
    print(f"Failed downloads are retried up to {MAX_RETRIES} times.\n")

    all_results = []
    start_time = time.time()

    for code, group in df.groupby('city_code'):
        name = CITIES.get(code, code)
        city_results = process_city_patches(code, name, group)
        all_results.extend(city_results)

        # save intermediate results after each city
        intermediate_df = pd.DataFrame(all_results)
        intermediate_path = PROCESSED_DIR / "patch_building_features.csv"
        intermediate_df.to_csv(intermediate_path, index=False)

    total_time = time.time() - start_time

    # final save
    results_df = pd.DataFrame(all_results)
    output_path = PROCESSED_DIR / "patch_building_features.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Building features computed for {len(results_df):,} patches")
    print(f"Time elapsed:           {total_time / 60:.1f} minutes")
    print(f"Saved: {output_path}")
    print()

    # diagnostics
    n_zero_buildings = (results_df['n_buildings'] == 0).sum()
    print(f"Patches with no buildings: {n_zero_buildings} "
          f"({100*n_zero_buildings/len(results_df):.1f}%)")
    print()

    # summary statistics
    valid_df = results_df[results_df['n_buildings'] > 0]
    print(f"Building feature statistics (excluding patches with 0 buildings):")
    print(valid_df.describe()[[
        'n_buildings', 'building_density', 'mean_building_footprint',
        'mean_nn_distance',
    ]].round(3).to_string())
    print()

    # cities with most/least buildings (sanity check)
    city_stats = results_df.groupby('city_code').agg({
        'n_buildings': 'mean',
        'building_density': 'mean',
    }).round(2).sort_values('n_buildings', ascending=False)

    print(f"Top 10 cities by mean buildings per patch:")
    print(city_stats.head(10).to_string())
    print()
    print(f"Bottom 10 cities by mean buildings per patch:")
    print(city_stats.tail(10).to_string())


if __name__ == "__main__":
    main()