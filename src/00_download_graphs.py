"""
Download street networks for all cities in the config as .graphml files.

Uses OSMnx to query each city by name and download the walk network.
Skips cities that have already been downloaded (file exists in data/raw/).

The walk network includes:
  - All driveable streets (residential, tertiary, secondary, primary)
  - Pedestrian-only paths (footways, plazas, steps)
  - Bike paths and shared infrastructure
  - Park paths

Output: data/raw/{city_code}.graphml for each city

This script is safe to re-run. It will only download cities that
do not already have a .graphml file.
"""

import sys
import time
from pathlib import Path

import osmnx as ox

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR, CITIES


# ── download configuration ─────────────────────────────────────────
NETWORK_TYPE = 'walk'    # walk includes drive + pedestrian paths


def download_city(code: str, name: str) -> bool:
    """
    Download a single city's street network.
    Returns True if downloaded, False if skipped or failed.
    """
    output_path = RAW_DIR / f"{code}.graphml"

    # skip if already exists
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  {code:15s} — already exists ({size_mb:.1f} MB), skipping")
        return False

    print(f"  {code:15s} — downloading '{name}'...")

    try:
        t0 = time.time()
        # query OSM by place name
        G = ox.graph_from_place(
            name,
            network_type=NETWORK_TYPE,
            simplify=True,
        )
        elapsed = time.time() - t0

        # save as graphml
        ox.save_graphml(G, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"  {code:15s} — DONE: {n_nodes:,} nodes, "
              f"{n_edges:,} edges, {size_mb:.1f} MB "
              f"({elapsed:.0f}s)")
        return True

    except Exception as e:
        print(f"  {code:15s} — FAILED: {e}")
        return False


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
        if result:
            n_downloaded += 1
        elif (RAW_DIR / f"{code}.graphml").exists():
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