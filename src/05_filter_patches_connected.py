"""
Filter patches by connectivity of their street network.

For each patch, extracts the street subgraph clipped to the patch
boundary, then computes the fraction of nodes belonging to the
largest connected component.

Patches where less than 80% of nodes belong to one connected
component are dropped. These are typically patches dominated by:
  - Parks with disconnected paths
  - Rivers separating two street networks (no bridges)
  - Industrial zones with isolated road clusters
  - Mixed land use with disconnected fragments

This is the third filtering pass after percentile + intersection floor.

Inputs:
  - data/processed/patch_inventory_filtered_v2.csv

Outputs:
  - data/processed/patch_inventory_filtered_v3.csv (kept patches)
  - data/processed/patch_filter_v3_report.csv (per-city statistics)
  - data/processed/patch_connectivity_all.csv (full results
    including dropped patches, for inspection)
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

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR, PROCESSED_DIR


# ── filter configuration ───────────────────────────────────────────
MIN_CONNECTED_FRACTION = 0.80   # 80% of nodes must be in one component
MIN_NODES_FOR_CHECK    = 10     # if patch has fewer nodes, skip
                                 # check entirely (already too small)


def compute_largest_component_fraction(G_patch):
    """
    Given a subgraph, return:
      - largest_fraction: fraction of nodes in the largest component
      - n_components: total number of connected components
      - n_nodes_largest: size of the largest component
    """
    if G_patch.number_of_nodes() == 0:
        return 0.0, 0, 0

    # treat as undirected for connectivity (OSMnx graphs are directed)
    G_undirected = G_patch.to_undirected()
    components = list(nx.connected_components(G_undirected))
    n_components = len(components)

    if n_components == 0:
        return 0.0, 0, 0

    largest = max(components, key=len)
    n_nodes_largest = len(largest)
    largest_fraction = n_nodes_largest / G_patch.number_of_nodes()

    return largest_fraction, n_components, n_nodes_largest


def process_city_patches(code: str, patches_df: pd.DataFrame):
    """
    Process all patches for one city. Returns a list of result dicts.
    """
    graph_path = RAW_DIR / f"{code}.graphml"
    if not graph_path.exists():
        print(f"  {code:15s} — graph file not found, skipping "
              f"{len(patches_df)} patches")
        return []

    # load and project the city's graph
    print(f"  {code:15s} — loading graph...")
    G = ox.load_graphml(graph_path)
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)

    results = []
    n_patches = len(patches_df)

    for i, (_, patch) in enumerate(patches_df.iterrows()):
        # build the patch geometry
        patch_geom = box(
            patch['minx_utm'],
            patch['miny_utm'],
            patch['maxx_utm'],
            patch['maxy_utm'],
        )

        # find nodes inside the patch boundary
        node_mask = nodes_proj.geometry.within(patch_geom)
        patch_node_ids = nodes_proj[node_mask].index.tolist()

        # extract the subgraph for this patch
        G_patch = G_proj.subgraph(patch_node_ids).copy()

        # compute connectivity metrics
        if G_patch.number_of_nodes() < MIN_NODES_FOR_CHECK:
            # already too small - mark as failed
            largest_fraction = 0.0
            n_components = 0
            n_nodes_largest = G_patch.number_of_nodes()
        else:
            (largest_fraction,
             n_components,
             n_nodes_largest) = compute_largest_component_fraction(G_patch)

        # determine if patch passes the filter
        passes_filter = (
            G_patch.number_of_nodes() >= MIN_NODES_FOR_CHECK
            and largest_fraction >= MIN_CONNECTED_FRACTION
        )

        results.append({
            'patch_id':          patch['patch_id'],
            'city_code':         code,
            'n_nodes_in_patch':  G_patch.number_of_nodes(),
            'n_components':      n_components,
            'n_nodes_largest':   n_nodes_largest,
            'largest_fraction':  round(largest_fraction, 4),
            'passes_filter':     passes_filter,
        })

    n_passed = sum(r['passes_filter'] for r in results)
    print(f"  {code:15s} — {n_passed:4d} / {n_patches:4d} patches "
          f"passed ({100 * n_passed / n_patches:.1f}%)")

    return results


def main():
    # load the v2 filtered inventory
    inventory_path = PROCESSED_DIR / "patch_inventory_filtered_v2.csv"
    if not inventory_path.exists():
        print(f"ERROR: {inventory_path} not found.")
        print(f"Run 04_filter_patches_minintersection.py first.")
        return

    df = pd.read_csv(inventory_path)
    print(f"Loaded {len(df):,} patches from "
          f"{df['city_code'].nunique()} cities\n")
    print(f"Applying connectivity filter: "
          f"largest component >= {int(MIN_CONNECTED_FRACTION * 100)}% "
          f"of nodes\n")

    # process each city
    all_results = []
    start_time = time.time()

    for code, group in df.groupby('city_code'):
        t0 = time.time()
        city_results = process_city_patches(code, group)
        elapsed = time.time() - t0
        all_results.extend(city_results)

        # save intermediate results after each city
        # (in case the script crashes, we don't lose all progress)
        results_df = pd.DataFrame(all_results)
        intermediate_path = (
            PROCESSED_DIR / "patch_connectivity_all.csv"
        )
        results_df.to_csv(intermediate_path, index=False)

    total_time = time.time() - start_time

    # convert results to dataframe
    results_df = pd.DataFrame(all_results)

    # merge with original inventory to keep all original columns
    df_merged = df.merge(
        results_df[['patch_id', 'n_nodes_in_patch', 'n_components',
                     'n_nodes_largest', 'largest_fraction',
                     'passes_filter']],
        on='patch_id',
        how='left',
    )

    # split into passed and failed
    filtered_df = df_merged[df_merged['passes_filter']].copy()

    # build per-city report
    report_rows = []
    for city, group in df_merged.groupby('city_code'):
        n_before = len(group)
        n_passed = group['passes_filter'].sum()
        report_rows.append({
            'city_code':       city,
            'n_before_filter': n_before,
            'n_after_filter':  int(n_passed),
            'n_dropped':       n_before - int(n_passed),
            'kept_pct':        round(100 * n_passed / n_before, 1),
        })
    report_df = pd.DataFrame(report_rows).sort_values(
        'n_after_filter', ascending=False
    )

    # save outputs
    filtered_path = PROCESSED_DIR / "patch_inventory_filtered_v3.csv"
    report_path = PROCESSED_DIR / "patch_filter_v3_report.csv"

    filtered_df.drop(columns=['passes_filter']).to_csv(
        filtered_path, index=False
    )
    report_df.to_csv(report_path, index=False)

    # summary
    print(f"\n{'=' * 70}")
    print(f"Connectivity filter applied")
    print(f"Threshold: largest component >= "
          f"{int(MIN_CONNECTED_FRACTION * 100)}% of nodes")
    print()
    print(f"Patches before filter:  {len(df):,}")
    print(f"Patches after filter:   {len(filtered_df):,}")
    print(f"Dropped by filter:      {len(df) - len(filtered_df):,} "
          f"({100 * (1 - len(filtered_df) / len(df)):.1f}%)")
    print(f"Time elapsed:           {total_time / 60:.1f} minutes")
    print()
    print(f"Saved:")
    print(f"  {filtered_path}")
    print(f"  {report_path}")
    print(f"  {PROCESSED_DIR / 'patch_connectivity_all.csv'} "
          f"(includes dropped patches for inspection)")
    print()

    # show cities most affected by the connectivity filter
    most_affected = report_df.sort_values('n_dropped', ascending=False)
    print(f"Top 10 cities most affected by connectivity filter:")
    print(most_affected.head(10).to_string(index=False))
    print()

    # show cities least affected
    least_affected = report_df.sort_values('n_dropped', ascending=True)
    print(f"Top 10 cities least affected (most patches survive):")
    print(least_affected.head(10).to_string(index=False))
    print()

    # final per-city counts after all three filters
    print(f"Final patches per city (top 10 largest):")
    print(report_df.head(10)[['city_code', 'n_after_filter']]
          .to_string(index=False))
    print()
    print(f"Final patches per city (bottom 10 smallest):")
    print(report_df.tail(10)[['city_code', 'n_after_filter']]
          .to_string(index=False))


if __name__ == "__main__":
    main()