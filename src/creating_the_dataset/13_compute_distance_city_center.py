"""
Compute distance from each patch centroid to its city's center.

For each city:
  1. Geocode the city name to get its center coordinates
  2. For each patch in that city, compute geodesic distance
     from the patch centroid to the city center (in km)

Inputs:
  - data/processed/patch_stratified_sample.csv

Outputs:
  - data/processed/patch_distance_to_center.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
from geopy.distance import geodesic

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, CITIES

warnings.filterwarnings('ignore')


def get_city_center(city_code: str, city_name: str, patches_df: pd.DataFrame):
    """
    Get the city's center coordinates.
    First tries to geocode the city name with OSMnx.
    Falls back to the centroid of patches if geocoding fails.
    """
    # try OSMnx geocoding first
    try:
        center = ox.geocode(city_name)
        # geocode returns (lat, lon)
        return center, 'geocoded'
    except Exception as e:
        # fall back to centroid of all patches for this city
        print(f"  {city_code:15s} — geocoding failed ({type(e).__name__}), "
              f"using patch centroid fallback")
        city_patches = patches_df[patches_df['city_code'] == city_code]
        mean_lat = city_patches['centroid_lat'].mean()
        mean_lon = city_patches['centroid_lon'].mean()
        return (mean_lat, mean_lon), 'fallback'


def compute_distances_for_city(code: str, name: str, patches_df: pd.DataFrame):
    """Compute distance to city center for all patches in one city."""
    center, source = get_city_center(code, name, patches_df)
    center_lat, center_lon = center

    results = []
    for _, patch in patches_df.iterrows():
        patch_coords = (patch['centroid_lat'], patch['centroid_lon'])
        distance_km = geodesic(patch_coords, center).kilometers
        results.append({
            'patch_id':              patch['patch_id'],
            'city_code':             code,
            'distance_to_center_km': round(distance_km, 3),
        })

    return results, center, source


def main():
    sample_path = PROCESSED_DIR / "patch_stratified_sample.csv"
    if not sample_path.exists():
        print(f"ERROR: {sample_path} not found.")
        print(f"Run 08_stratified_sample.py first.")
        return

    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df):,} patches from {df['city_code'].nunique()} cities\n")
    print(f"Computing distance from each patch to its city center")
    print(f"(geocoded via OSMnx, fallback to patch centroid)\n")

    all_results = []
    city_centers_log = []
    start_time = time.time()

    for code, group in df.groupby('city_code'):
        name = CITIES.get(code, code)
        print(f"  {code:15s} — geocoding {name}...")

        try:
            city_results, center, source = compute_distances_for_city(
                code, name, group
            )
            all_results.extend(city_results)

            # compute stats
            distances = [r['distance_to_center_km'] for r in city_results]
            mean_dist = np.mean(distances)
            max_dist = np.max(distances)

            city_centers_log.append({
                'city_code':    code,
                'city_name':    name,
                'center_lat':   center[0],
                'center_lon':   center[1],
                'source':       source,
                'n_patches':    len(city_results),
                'mean_dist_km': round(mean_dist, 2),
                'max_dist_km':  round(max_dist, 2),
            })

            print(f"  {code:15s} — center: ({center[0]:.4f}, {center[1]:.4f}), "
                  f"{len(city_results)} patches, "
                  f"mean dist = {mean_dist:.1f} km, "
                  f"max = {max_dist:.1f} km")

        except Exception as e:
            print(f"  {code:15s} — FAILED: {type(e).__name__}: {e}")

    total_time = time.time() - start_time

    # save results
    results_df = pd.DataFrame(all_results)
    output_path = PROCESSED_DIR / "patch_distance_to_center.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Distances computed for {len(results_df):,} patches")
    print(f"Time elapsed: {total_time:.1f} seconds")
    print(f"Saved: {output_path}")
    print()

    # summary statistics
    print(f"Distance distribution (km):")
    print(results_df['distance_to_center_km'].describe().round(2).to_string())
    print()

    # top 5 furthest patches (sanity check)
    print(f"5 patches farthest from city center:")
    top5_far = results_df.nlargest(5, 'distance_to_center_km')
    print(top5_far.to_string(index=False))
    print()

    # top 5 closest patches (sanity check)
    print(f"5 patches closest to city center:")
    top5_close = results_df.nsmallest(5, 'distance_to_center_km')
    print(top5_close.to_string(index=False))
    print()

    # city centers used (for verification)
    print(f"City centers used:")
    centers_df = pd.DataFrame(city_centers_log).sort_values('mean_dist_km',
                                                             ascending=False)
    print(centers_df.head(10).to_string(index=False))
    print(f"  ... and {len(centers_df) - 10} more")
    print()

    # cities with fallback (any failed geocoding)
    fallback_cities = centers_df[centers_df['source'] == 'fallback']
    if len(fallback_cities) > 0:
        print(f"Cities where geocoding failed (used patch centroid fallback):")
        print(fallback_cities.to_string(index=False))


if __name__ == "__main__":
    main()