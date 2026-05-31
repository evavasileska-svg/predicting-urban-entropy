"""
Download street networks for all cities in the config as .graphml files.

Uses OSMnx to query each city. Cities can be queried by:
  - Place name (default) — works for most cities
  - Coordinates + radius — used as fallback for ambiguous cities

Skips cities that have already been downloaded.

Output: data/raw/{city_code}.graphml for each city
"""

import sys
import time
from pathlib import Path

import osmnx as ox

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR, CITIES


# ── download configuration ─────────────────────────────────────────
NETWORK_TYPE = 'walk'

# Cities that must be downloaded by coordinates instead of place name
# (because the place name is ambiguous or resolves to too large an area)
COORDINATE_OVERRIDES = {
    'minneapolis': {'lat':  44.9778, 'lon':  -93.2650, 'radius_m': 12000},
    'seattle':     {'lat':  47.6062, 'lon': -122.3321, 'radius_m': 12000},
    'detroit':     {'lat':  42.3314, 'lon':  -83.0458, 'radius_m': 14000},
    'lasvegas':    {'lat':  36.1716, 'lon': -115.1391, 'radius_m': 15000},
    'cleveland':   {'lat':  41.4993, 'lon':  -81.6944, 'radius_m': 12000},
    'orlando':     {'lat':  28.5384, 'lon':  -81.3789, 'radius_m': 14000},
    'toronto':     {'lat':  43.6532, 'lon':  -79.3832, 'radius_m': 14000},
    'houston':     {'lat':  29.7604, 'lon':  -95.3698, 'radius_m': 16000},
}


def download_by_place(code: str, name: str):
    """Download a city using place name lookup."""
    print(f"  {code:15s} — downloading by place: '{name}'...")
    return ox.graph_from_place(
        name, network_type=NETWORK_TYPE, simplify=True
    )


def download_by_point(code: str, override: dict):
    """Download a city using center coordinates + radius."""
    point = (override['lat'], override['lon'])
    radius = override['radius_m']
    print(f"  {code:15s} — downloading by point: "
          f"({override['lat']}, {override['lon']}) "
          f"radius {radius}m...")
    return ox.graph_from_point(
        point, dist=radius,
        network_type=NETWORK_TYPE, simplify=True,
    )


def download_city(code: str, name: str) -> str:
    """
    Download a single city's street network.
    Returns 'downloaded', 'skipped', or 'failed'.
    """
    output_path = RAW_DIR / f"{code}.graphml"

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  {code:15s} — already exists ({size_mb:.1f} MB), skipping")
        return 'skipped'

    try:
        t0 = time.time()

        # use coordinate override if specified, else use place name
        if code in COORDINATE_OVERRIDES:
            G = download_by_point(code, COORDINATE_OVERRIDES[code])
        else:
            G = download_by_place(code, name)

        elapsed = time.time() - t0

        # save as graphml
        ox.save_graphml(G, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"  {code:15s} — DONE: {n_nodes:,} nodes, "
              f"{n_edges:,} edges, {size_mb:.1f} MB "
              f"({elapsed:.0f}s)")
        return 'downloaded'

    except Exception as e:
        print(f"  {code:15s} — FAILED: {e}")
        return 'failed'


def main():
    print(f"Downloading street networks for {len(CITIES)} cities")
    print(f"Network type: {NETWORK_TYPE}")
    print(f"Destination: {RAW_DIR}\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    start_time = time.time()

    for code, name in CITIES.items():
        result = download_city(code, name)
        if result == 'downloaded':
            n_downloaded += 1
        elif result == 'skipped':
            n_skipped += 1
        else:
            n_failed += 1

    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"Downloaded:  {n_downloaded} cities")
    print(f"Skipped:     {n_skipped} cities (already existed)")
    print(f"Failed:      {n_failed} cities")
    print(f"Time:        {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()