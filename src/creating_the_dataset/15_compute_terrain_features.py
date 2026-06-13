"""
Compute terrain (elevation and slope) features for each patch.

Uses Open Topo Data public API (https://api.opentopodata.org)
which queries SRTM 30m elevation data globally without an API key.

For each patch:
  - Samples a 5x5 grid of 25 points within the 800x800m patch
  - Queries elevation via Open Topo Data (batched, 4 patches at a time)
  - Computes elev_mean, elev_std, elev_range, mean_slope

Outputs:
  - data/processed/patch_terrain_features.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, CITIES

warnings.filterwarnings('ignore')


# Open Topo Data public API
API_URL = "https://api.opentopodata.org/v1/srtm30m"

# how many sample points within each 800x800m patch (5x5 = 25 points)
SAMPLE_GRID_N = 5

# how many patches to batch per API request (4 patches x 25 points = 100)
PATCHES_PER_REQUEST = 4

# rate limit: pause between requests
RATE_LIMIT_SECONDS = 1.0


def generate_patch_sample_points(patch: pd.Series, grid_n: int):
    """
    Generate a grid of sample points within a patch.
    Returns lat/lon arrays and the grid shape.
    """
    minx = patch['minx_utm']
    miny = patch['miny_utm']
    maxx = patch['maxx_utm']
    maxy = patch['maxy_utm']
    crs_utm = patch['crs_utm']

    # generate sample points in UTM
    xs = np.linspace(minx, maxx, grid_n)
    ys = np.linspace(miny, maxy, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    sample_x_utm = xx.flatten()
    sample_y_utm = yy.flatten()

    # transform UTM to WGS84 (lat/lon)
    transformer = Transformer.from_crs(crs_utm, 'EPSG:4326', always_xy=True)
    sample_lon, sample_lat = transformer.transform(sample_x_utm, sample_y_utm)

    return sample_lat, sample_lon


def query_elevations(locations: list, max_retries: int = 3):
    """
    Query Open Topo Data for elevation at the given (lat, lon) points.
    Returns list of elevations in same order, with None for failures.
    """
    if not locations:
        return []

    # build the locations string
    locations_str = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in locations)
    params = {
        "locations":     locations_str,
        "interpolation": "bilinear",
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(API_URL, params=params, timeout=60)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK":
                    elevations = [
                        r.get("elevation")
                        for r in data.get("results", [])
                    ]
                    return elevations
                else:
                    print(f"    API status: {data.get('status')}")
                    return [None] * len(locations)

            elif response.status_code == 429:
                # rate limited
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s")
                time.sleep(wait)

            elif response.status_code in (502, 503, 504):
                # server overload
                wait = 5 * (attempt + 1)
                print(f"    Server busy ({response.status_code}), "
                      f"waiting {wait}s")
                time.sleep(wait)

            else:
                print(f"    HTTP {response.status_code}: "
                      f"{response.text[:200]}")
                return [None] * len(locations)

        except requests.exceptions.Timeout:
            print(f"    Timeout, attempt {attempt + 1}/{max_retries}")
            time.sleep(5)

        except Exception as e:
            print(f"    Error: {type(e).__name__}: {e}")
            time.sleep(5)

    return [None] * len(locations)


def compute_slope_from_grid(elev_grid: np.ndarray, pixel_size_m: float):
    """
    Compute mean slope (in degrees) from an elevation grid using
    finite differences (gradient method).
    """
    if elev_grid.size == 0 or np.isnan(elev_grid).all():
        return np.nan

    # mean over valid cells; if not enough, return NaN
    if np.isnan(elev_grid).sum() > elev_grid.size / 2:
        return np.nan

    # fill NaN with nearest neighbor for gradient calculation
    filled = elev_grid.copy()
    if np.isnan(filled).any():
        mean_val = np.nanmean(filled)
        filled[np.isnan(filled)] = mean_val

    # numpy gradient returns (dy, dx) for a 2D array
    dy, dx = np.gradient(filled)

    # convert pixel gradient to slope (rise/run)
    slope_x = dx / pixel_size_m
    slope_y = dy / pixel_size_m
    slope = np.sqrt(slope_x**2 + slope_y**2)

    # convert to degrees
    slope_deg = np.degrees(np.arctan(slope))

    return float(np.mean(slope_deg))


def compute_terrain_for_patch(elev_values: list, grid_n: int,
                              pixel_size_m: float):
    """
    Given a list of elevation values (in grid order) for one patch,
    compute the terrain features.
    """
    # convert None to NaN
    elevations = np.array(
        [e if e is not None else np.nan for e in elev_values],
        dtype=float,
    )

    # SRTM nodata typically -32768; mask values outside plausible range
    elevations[elevations < -500] = np.nan
    elevations[elevations > 9000] = np.nan

    elev_grid = elevations.reshape(grid_n, grid_n)
    valid = elevations[~np.isnan(elevations)]

    if len(valid) < 5:
        return {
            'elev_mean':  np.nan,
            'elev_std':   np.nan,
            'elev_range': np.nan,
            'mean_slope': np.nan,
        }

    return {
        'elev_mean':  round(float(np.mean(valid)), 2),
        'elev_std':   round(float(np.std(valid)), 2),
        'elev_range': round(float(np.max(valid) - np.min(valid)), 2),
        'mean_slope': round(compute_slope_from_grid(
            elev_grid, pixel_size_m), 3),
    }


def main():
    sample_path = PROCESSED_DIR / "patch_stratified_sample.csv"
    if not sample_path.exists():
        print(f"ERROR: {sample_path} not found.")
        return

    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df):,} stratified patches "
          f"from {df['city_code'].nunique()} cities\n")
    print(f"Computing terrain features (elevation + slope)")
    print(f"Source: Open Topo Data public API "
          f"(SRTM 30m, no key required)")
    print(f"Sampling {SAMPLE_GRID_N}x{SAMPLE_GRID_N} = "
          f"{SAMPLE_GRID_N**2} points per patch")
    print(f"Batching {PATCHES_PER_REQUEST} patches per API request")
    print(f"Estimated time: ~{len(df) / PATCHES_PER_REQUEST / 60:.0f} "
          f"minutes\n")

    # pre-compute sample points for all patches
    print(f"Generating sample points for all patches...")
    patch_sample_points = {}   # patch_id -> (lats, lons)
    for _, patch in df.iterrows():
        try:
            lats, lons = generate_patch_sample_points(patch, SAMPLE_GRID_N)
            patch_sample_points[patch['patch_id']] = (lats, lons)
        except Exception as e:
            print(f"  ERROR generating points for "
                  f"patch {patch['patch_id']}: {e}")
            patch_sample_points[patch['patch_id']] = None

    print(f"Sample points ready for {len(patch_sample_points)} patches\n")

    # process patches in batches
    all_results = []
    n_points_per_patch = SAMPLE_GRID_N ** 2
    patch_ids = list(patch_sample_points.keys())
    n_batches = int(np.ceil(len(patch_ids) / PATCHES_PER_REQUEST))

    print(f"Querying elevations in {n_batches} batched requests...\n")
    start_time = time.time()
    last_progress = 0

    for batch_idx in range(n_batches):
        batch_start = batch_idx * PATCHES_PER_REQUEST
        batch_end = min(batch_start + PATCHES_PER_REQUEST, len(patch_ids))
        batch_patch_ids = patch_ids[batch_start:batch_end]

        # collect all sample points for this batch
        batch_locations = []
        patch_offsets = []  # where each patch's points start in the batch

        for pid in batch_patch_ids:
            if patch_sample_points[pid] is None:
                patch_offsets.append(None)
                continue
            lats, lons = patch_sample_points[pid]
            patch_offsets.append(len(batch_locations))
            for lat, lon in zip(lats, lons):
                batch_locations.append((lat, lon))

        # query API
        elevations = query_elevations(batch_locations)

        # parse results per patch
        patch_meta = df.set_index('patch_id').loc[batch_patch_ids]
        for i, pid in enumerate(batch_patch_ids):
            offset = patch_offsets[i]
            if offset is None:
                all_results.append({
                    'patch_id':   pid,
                    'city_code':  patch_meta.loc[pid, 'city_code'],
                    'elev_mean':  np.nan,
                    'elev_std':   np.nan,
                    'elev_range': np.nan,
                    'mean_slope': np.nan,
                })
                continue

            patch_elev = elevations[offset:offset + n_points_per_patch]

            # estimate pixel size in meters at this latitude
            patch_lat = patch_meta.loc[pid, 'centroid_lat']
            # 800m / 5 grid steps gives ~160m spacing
            spacing_m = 800.0 / SAMPLE_GRID_N

            terrain = compute_terrain_for_patch(
                patch_elev, SAMPLE_GRID_N, spacing_m
            )

            all_results.append({
                'patch_id':   pid,
                'city_code':  patch_meta.loc[pid, 'city_code'],
                **terrain,
            })

        # progress report every 5%
        progress_pct = int(100 * (batch_idx + 1) / n_batches)
        if progress_pct >= last_progress + 5:
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            print(f"  Progress: {progress_pct}% "
                  f"({batch_idx + 1}/{n_batches} requests), "
                  f"elapsed {elapsed/60:.1f} min, "
                  f"ETA {eta/60:.1f} min")
            last_progress = progress_pct

        # rate limit
        time.sleep(RATE_LIMIT_SECONDS)

    total_time = time.time() - start_time

    # save results
    results_df = pd.DataFrame(all_results)
    output_path = PROCESSED_DIR / "patch_terrain_features.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Terrain features computed for {len(results_df):,} patches")
    print(f"Time elapsed: {total_time / 60:.1f} minutes")
    print(f"Saved: {output_path}\n")

    # coverage report
    coverage = results_df['elev_mean'].notna().sum()
    coverage_pct = 100 * coverage / len(df)
    print(f"Coverage: {coverage}/{len(df)} patches "
          f"({coverage_pct:.1f}%)\n")

    if coverage == 0:
        print(f"WARNING: no patches got terrain data. "
              f"Check API connectivity.")
        return

    valid_df = results_df.dropna(subset=['elev_mean'])

    print(f"Terrain feature statistics:")
    print(valid_df[['elev_mean', 'elev_std',
                    'elev_range', 'mean_slope']].describe().round(2)
          .to_string())
    print()

    # top 10 cities by mean slope (hilliest)
    print(f"Top 10 cities by mean slope (hilliest):")
    city_stats = valid_df.groupby('city_code').agg({
        'mean_slope': 'mean',
        'elev_range': 'mean',
        'elev_mean':  'mean',
    }).round(2)
    print(city_stats.nlargest(10, 'mean_slope').to_string())
    print()

    print(f"Bottom 10 cities by mean slope (flattest):")
    print(city_stats.nsmallest(10, 'mean_slope').to_string())


if __name__ == "__main__":
    main()