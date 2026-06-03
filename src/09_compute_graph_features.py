"""
Compute graph topological features for each patch in the stratified sample.

For each patch, extracts the street subgraph (largest connected component)
and computes:
  - Counts and proportions of intersection types (4-way, 3-way, dead-ends)
  - Edge length statistics (mean, total)
  - Meshedness (graph connectivity measure)
  - Mean node degree
  - Densities normalised by patch area (intersection_density, street_density)

These features describe the topology and density of the street network
without requiring expensive computations like betweenness.

Inputs:
  - data/processed/patch_stratified_sample.csv

Outputs:
  - data/processed/patch_graph_features.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import box

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RAW_DIR, PROCESSED_DIR, PATCH_SIZE_M


# patch area in km² (constant since all patches are 800m × 800m)
PATCH_AREA_KM2 = (PATCH_SIZE_M / 1000) ** 2


def compute_features_for_patch(G_patch):
    """
    Compute graph features for a single patch subgraph.
    Assumes G_patch is already the largest connected component.

    Returns a dictionary of features.
    """
    # work on undirected version for node-degree counting
    G_undirected = G_patch.to_undirected()
    n_nodes = G_undirected.number_of_nodes()
    n_edges = G_undirected.number_of_edges()

    if n_nodes == 0 or n_edges == 0:
        return None

    # ── node-type counts ──────────────────────────────────────────
    # for each node, count its undirected degree (number of neighbours)
    degrees = dict(G_undirected.degree())
    degree_values = list(degrees.values())

    n_4way = sum(1 for d in degree_values if d == 4)
    n_3way = sum(1 for d in degree_values if d == 3)
    n_deadend = sum(1 for d in degree_values if d == 1)
    n_other = sum(1 for d in degree_values if d not in (1, 3, 4))

    prop_4way    = n_4way    / n_nodes
    prop_3way    = n_3way    / n_nodes
    prop_deadend = n_deadend / n_nodes
    mean_degree  = sum(degree_values) / n_nodes

    # ── edge length statistics ────────────────────────────────────
    # use the 'length' attribute stored by OSMnx (in meters)
    edge_lengths = []
    for u, v, data in G_undirected.edges(data=True):
        length = data.get('length')
        if length is not None and length > 0:
            edge_lengths.append(length)

    if len(edge_lengths) == 0:
        return None

    mean_edge_length  = float(np.mean(edge_lengths))
    total_edge_length = float(np.sum(edge_lengths))

    # ── meshedness ────────────────────────────────────────────────
    # meshedness = (E - V + 1) / (2V - 5)
    # measures how mesh-like (grid-like) the network is
    # 0 = tree, ~1 = complete planar mesh
    if n_nodes > 2:
        meshedness = (n_edges - n_nodes + 1) / (2 * n_nodes - 5)
    else:
        meshedness = 0.0

    # ── densities ─────────────────────────────────────────────────
    intersection_density = n_nodes / PATCH_AREA_KM2          # per km²
    street_density = (total_edge_length / 1000) / PATCH_AREA_KM2  # km per km²

    return {
        'n_nodes_used':         n_nodes,
        'n_edges_used':         n_edges,
        'n_4way':               n_4way,
        'n_3way':               n_3way,
        'n_deadend':            n_deadend,
        'proportion_4way':      round(prop_4way, 6),
        'proportion_3way':      round(prop_3way, 6),
        'proportion_deadend':   round(prop_deadend, 6),
        'mean_degree':          round(mean_degree, 4),
        'mean_edge_length':     round(mean_edge_length, 2),
        'total_edge_length':    round(total_edge_length, 2),
        'meshedness':           round(meshedness, 6),
        'intersection_density': round(intersection_density, 2),
        'street_density':       round(street_density, 4),
    }


def process_city_patches(code: str, patches_df: pd.DataFrame):
    """Compute graph features for all patches in one city."""
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

        # compute features
        features = compute_features_for_patch(G_largest)
        if features is None:
            continue

        results.append({
            'patch_id': patch['patch_id'],
            'city_code': code,
            **features,
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
    print(f"Computing graph features\n")

    all_results = []
    start_time = time.time()

    for code, group in df.groupby('city_code'):
        city_results = process_city_patches(code, group)
        all_results.extend(city_results)

        # save intermediate results after each city
        intermediate_df = pd.DataFrame(all_results)
        intermediate_path = PROCESSED_DIR / "patch_graph_features.csv"
        intermediate_df.to_csv(intermediate_path, index=False)

    total_time = time.time() - start_time

    # final save
    results_df = pd.DataFrame(all_results)
    output_path = PROCESSED_DIR / "patch_graph_features.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Graph features computed for {len(results_df):,} patches")
    print(f"Time elapsed:           {total_time / 60:.1f} minutes")
    print(f"Saved: {output_path}")
    print()

    # summary statistics
    print(f"Feature statistics:")
    print(results_df.describe()[[
        'proportion_4way', 'proportion_3way', 'proportion_deadend',
        'mean_degree', 'meshedness', 'intersection_density',
        'street_density', 'mean_edge_length',
    ]].round(3).to_string())


if __name__ == "__main__":
    main()