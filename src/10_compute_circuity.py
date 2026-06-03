"""
Compute network circuity for each patch in the stratified sample.

Circuity measures how curved/winding a street network is, defined as:

    circuity = total network distance / total great-circle distance
             = Σ(edge lengths) / Σ(straight-line distances between nodes)

Interpretation:
  - circuity = 1.0 → all streets are perfectly straight
  - circuity > 1.0 → streets curve or zigzag
  - typical values: 1.01 (rigid grid) to 1.20 (organic curvilinear)

Boeing (2019) found circuity moderately negatively correlated with φ
(grid-like cities have lower circuity).

Inputs:
  - data/processed/patch_stratified_sample.csv

Outputs:
  - data/processed/patch_circuity.csv
"""

import sys
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import box

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR, PROCESSED_DIR


def compute_circuity_for_patch(G_patch):
    """
    Compute the average circuity of a patch's street network.

    Circuity = sum of edge lengths / sum of straight-line distances
    between the same pairs of nodes.

    Returns a dictionary of circuity stats.
    """
    G_undirected = G_patch.to_undirected()
    n_edges = G_undirected.number_of_edges()

    if n_edges == 0:
        return None

    total_network_length = 0.0
    total_straight_length = 0.0
    n_edges_used = 0

    for u, v, data in G_undirected.edges(data=True):
        # actual network length stored by OSMnx (in meters)
        length = data.get('length')
        if length is None or length <= 0:
            continue

        # straight-line distance between endpoints (projected coords, in meters)
        u_x = G_patch.nodes[u]['x']
        u_y = G_patch.nodes[u]['y']
        v_x = G_patch.nodes[v]['x']
        v_y = G_patch.nodes[v]['y']

        dx = v_x - u_x
        dy = v_y - u_y
        straight = math.sqrt(dx ** 2 + dy ** 2)

        if straight <= 0:
            continue   # degenerate (self-loop or zero-length edge)

        total_network_length  += length
        total_straight_length += straight
        n_edges_used += 1

    if total_straight_length <= 0 or n_edges_used == 0:
        return None

    circuity = total_network_length / total_straight_length

    return {
        'circuity':              round(circuity, 6),
        'total_network_length':  round(total_network_length, 2),
        'total_straight_length': round(total_straight_length, 2),
        'n_edges_in_circuity':   n_edges_used,
    }


def process_city_patches(code: str, patches_df: pd.DataFrame):
    """Compute circuity for all patches in one city."""
    graph_path = RAW_DIR / f"{code}.graphml"
    if not graph_path.exists():
        print(f"  {code:15s} — graph file not found")
        return []

    print(f"  {code:15s} — loading graph...")
    G = ox.load_graphml(graph_path)
    G_proj = ox.project_graph(G)
    nodes_proj, _ = ox.graph_to_gdfs(G_proj)

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

        # find nodes in patch
        node_mask = nodes_proj.geometry.within(patch_geom)
        patch_node_ids = nodes_proj[node_mask].index.tolist()

        # extract subgraph and keep largest component
        G_patch = G_proj.subgraph(patch_node_ids).copy()
        G_undirected = G_patch.to_undirected()

        if G_undirected.number_of_nodes() == 0:
            continue

        components = list(nx.connected_components(G_undirected))
        if len(components) == 0:
            continue

        largest = max(components, key=len)
        G_largest = G_patch.subgraph(largest).copy()

        # compute circuity
        circuity_data = compute_circuity_for_patch(G_largest)
        if circuity_data is None:
            continue

        results.append({
            'patch_id': patch['patch_id'],
            'city_code': code,
            **circuity_data,
        })

    elapsed = time.time() - t0
    print(f"  {code:15s} — {len(results):4d} / {len(patches_df):4d} "
          f"patches computed ({elapsed:5.1f}s)")
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
    print(f"Computing circuity\n")

    all_results = []
    start_time = time.time()

    for code, group in df.groupby('city_code'):
        city_results = process_city_patches(code, group)
        all_results.extend(city_results)

        # save intermediate results after each city
        intermediate_df = pd.DataFrame(all_results)
        intermediate_path = PROCESSED_DIR / "patch_circuity.csv"
        intermediate_df.to_csv(intermediate_path, index=False)

    total_time = time.time() - start_time

    # final save
    results_df = pd.DataFrame(all_results)
    output_path = PROCESSED_DIR / "patch_circuity.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Circuity computed for {len(results_df):,} patches")
    print(f"Time elapsed:         {total_time / 60:.1f} minutes")
    print(f"Saved: {output_path}")
    print()

    # summary statistics
    print(f"Circuity statistics:")
    print(results_df['circuity'].describe().round(4).to_string())
    print()

    # most and least circuitous patches (sanity check)
    print(f"5 least circuitous patches (most grid-like):")
    least_circ = results_df.nsmallest(5, 'circuity')
    print(least_circ[['patch_id', 'city_code', 'circuity']].to_string(index=False))
    print()

    print(f"5 most circuitous patches (most winding):")
    most_circ = results_df.nlargest(5, 'circuity')
    print(most_circ[['patch_id', 'city_code', 'circuity']].to_string(index=False))


if __name__ == "__main__":
    main()