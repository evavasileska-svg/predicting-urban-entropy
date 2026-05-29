"""
Compute Boeing's street bearing entropy for each filtered patch.

For each patch in patch_inventory_filtered_v3.csv:
  - Load the city's projected street graph
  - Extract the subgraph clipped to the patch boundary
  - Keep only the largest connected component
  - Compute street bearings for each edge
  - Add reciprocal bearings (streets are undirected)
  - Apply -5° angular shift (Boeing's convention)
  - Bin bearings into 36 bins of 10°
  - Compute Shannon entropy (unweighted and length-weighted)
  - Normalise by log(36) to get Ho and Hw on a 0-1 scale
  - Compute phi (orientation order)

Outputs:
  - data/processed/patch_entropy.csv (one row per patch with entropy)

Progress is saved incrementally after each city.
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
from src.config import RAW_DIR, PROCESSED_DIR, N_BEARING_BINS


# ── entropy computation settings ───────────────────────────────────
N_BINS = N_BEARING_BINS    # 36 bins of 10° (Boeing's convention)
BIN_WIDTH = 360.0 / N_BINS  # 10° per bin
ANGULAR_SHIFT = -5.0        # Boeing's -5° shift

# log(N_BINS) used for normalisation
LOG_N_BINS = math.log(N_BINS)

# log(4) used in phi computation (perfect grid baseline)
LOG_4 = math.log(4)


def compute_bearings_from_subgraph(G_patch):
    """
    Extract bearings and edge lengths from a projected subgraph.

    Returns:
      bearings_deg: array of bearings in degrees (0-360)
      lengths: array of edge lengths in meters
    """
    bearings_deg = []
    lengths = []

    for u, v, key, data in G_patch.edges(keys=True, data=True):
        # get start and end coordinates from the projected graph
        u_x = G_patch.nodes[u]['x']
        u_y = G_patch.nodes[u]['y']
        v_x = G_patch.nodes[v]['x']
        v_y = G_patch.nodes[v]['y']

        # compute bearing in degrees (0 = north, 90 = east)
        dx = v_x - u_x
        dy = v_y - u_y
        if dx == 0 and dy == 0:
            continue   # degenerate zero-length edge
        bearing = math.degrees(math.atan2(dx, dy)) % 360

        # edge length (use stored 'length' if available)
        length = data.get('length', math.sqrt(dx ** 2 + dy ** 2))

        bearings_deg.append(bearing)
        lengths.append(length)

    return np.array(bearings_deg), np.array(lengths)


def compute_entropy(G_patch):
    """
    Compute Boeing's normalised entropy (Ho, Hw) and phi.

    Returns dictionary with:
      entropy_raw           : raw unweighted entropy in nats
      entropy_normalised    : Ho (0-1 scale)
      entropy_weighted_raw  : raw length-weighted entropy in nats
      entropy_weighted_norm : Hw (0-1 scale)
      phi                   : orientation order
      n_edges_used          : number of edges used
    """
    bearings, lengths = compute_bearings_from_subgraph(G_patch)
    n_edges = len(bearings)

    if n_edges == 0:
        return None

    # add reciprocal bearings (a street pointing N also points S)
    bearings_full = np.concatenate([bearings, (bearings + 180) % 360])
    lengths_full = np.concatenate([lengths, lengths])

    # apply Boeing's -5° angular shift so cardinal directions
    # sit at the centre of bins rather than on bin edges
    bearings_shifted = (bearings_full + ANGULAR_SHIFT) % 360

    # define bin edges
    bin_edges = np.linspace(0, 360, N_BINS + 1)

    # ── unweighted (count) histogram ──────────────────────────────
    counts, _ = np.histogram(bearings_shifted, bins=bin_edges)
    probs = counts / counts.sum()
    # Shannon entropy in nats (natural log)
    nonzero = probs > 0
    entropy_raw = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
    entropy_norm = entropy_raw / LOG_N_BINS

    # ── length-weighted histogram ─────────────────────────────────
    weights_per_bin, _ = np.histogram(
        bearings_shifted, bins=bin_edges, weights=lengths_full
    )
    probs_w = weights_per_bin / weights_per_bin.sum()
    nonzero_w = probs_w > 0
    entropy_weighted_raw = -np.sum(
        probs_w[nonzero_w] * np.log(probs_w[nonzero_w])
    )
    entropy_weighted_norm = entropy_weighted_raw / LOG_N_BINS

    # ── phi (orientation order) ───────────────────────────────────
    # phi = 1 - ((Ho - Hg) / (Hmax - Hg))²
    # where Hg = log(4) / log(36) is a perfect grid normalised
    Hg_norm = LOG_4 / LOG_N_BINS
    Hmax_norm = 1.0
    phi = 1.0 - ((entropy_norm - Hg_norm) / (Hmax_norm - Hg_norm)) ** 2

    return {
        'entropy_raw':           round(entropy_raw, 6),
        'entropy_normalised':    round(entropy_norm, 6),
        'entropy_weighted_raw':  round(entropy_weighted_raw, 6),
        'entropy_weighted_norm': round(entropy_weighted_norm, 6),
        'phi':                   round(phi, 6),
        'n_edges_used':          n_edges,
    }


def process_city_patches(code: str, patches_df: pd.DataFrame):
    """
    Compute entropy for all patches of one city.
    Returns a list of result dicts.
    """
    graph_path = RAW_DIR / f"{code}.graphml"
    if not graph_path.exists():
        print(f"  {code:15s} — graph file not found")
        return []

    print(f"  {code:15s} — loading graph...")
    G = ox.load_graphml(graph_path)
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)

    results = []
    n_patches = len(patches_df)
    t0 = time.time()

    for i, (_, patch) in enumerate(patches_df.iterrows()):
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

        # extract subgraph
        G_patch = G_proj.subgraph(patch_node_ids).copy()

        # keep only the largest connected component
        # (we already filtered for >=80% in one component, but
        #  we use the largest one for the actual computation)
        G_undirected = G_patch.to_undirected()
        if G_undirected.number_of_nodes() == 0:
            continue
        components = list(nx.connected_components(G_undirected))
        if len(components) == 0:
            continue
        largest = max(components, key=len)
        G_largest = G_patch.subgraph(largest).copy()

        # compute entropy
        entropy_data = compute_entropy(G_largest)
        if entropy_data is None:
            continue

        results.append({
            'patch_id':  patch['patch_id'],
            'city_code': code,
            **entropy_data,
        })

    elapsed = time.time() - t0
    print(f"  {code:15s} — {len(results):4d} / {n_patches:4d} patches "
          f"computed ({elapsed:5.1f}s)")

    return results


def main():
    # load the v3 filtered inventory
    inventory_path = PROCESSED_DIR / "patch_inventory_filtered_v3.csv"
    if not inventory_path.exists():
        print(f"ERROR: {inventory_path} not found.")
        print(f"Run 05_filter_patches_connected.py first.")
        return

    df = pd.read_csv(inventory_path)
    print(f"Loaded {len(df):,} filtered patches from "
          f"{df['city_code'].nunique()} cities\n")
    print(f"Computing Boeing entropy with {N_BINS} bins, "
          f"{ANGULAR_SHIFT}° shift\n")

    all_results = []
    start_time = time.time()

    for code, group in df.groupby('city_code'):
        city_results = process_city_patches(code, group)
        all_results.extend(city_results)

        # save intermediate results after each city
        results_df = pd.DataFrame(all_results)
        intermediate_path = PROCESSED_DIR / "patch_entropy.csv"
        results_df.to_csv(intermediate_path, index=False)

    total_time = time.time() - start_time

    # final save with merged metadata
    results_df = pd.DataFrame(all_results)

    # merge with the v3 inventory to keep all metadata
    df_merged = df.merge(
        results_df,
        on=['patch_id', 'city_code'],
        how='inner',
    )

    output_path = PROCESSED_DIR / "patch_entropy.csv"
    df_merged.to_csv(output_path, index=False)

    # summary
    print(f"\n{'=' * 70}")
    print(f"Entropy computation complete")
    print(f"Patches with entropy:  {len(df_merged):,}")
    print(f"Time elapsed:          {total_time / 60:.1f} minutes")
    print()
    print(f"Saved: {output_path}")
    print()

    # entropy distribution
    print(f"Entropy (Ho) statistics:")
    print(df_merged['entropy_normalised'].describe().round(4).to_string())
    print()

    print(f"Phi statistics:")
    print(df_merged['phi'].describe().round(4).to_string())
    print()

    # ordered vs disordered patches
    n_ordered     = (df_merged['entropy_normalised'] < 0.80).sum()
    n_medium      = ((df_merged['entropy_normalised'] >= 0.80) &
                     (df_merged['entropy_normalised'] < 0.95)).sum()
    n_disordered  = (df_merged['entropy_normalised'] >= 0.95).sum()

    print(f"Distribution across entropy bands:")
    print(f"  ordered    (Ho < 0.80):        "
          f"{n_ordered:6,} patches ({100 * n_ordered / len(df_merged):.1f}%)")
    print(f"  medium     (0.80 ≤ Ho < 0.95): "
          f"{n_medium:6,} patches ({100 * n_medium / len(df_merged):.1f}%)")
    print(f"  disordered (Ho ≥ 0.95):        "
          f"{n_disordered:6,} patches "
          f"({100 * n_disordered / len(df_merged):.1f}%)")
    print()

    # per-city entropy ranges
    print(f"Per-city entropy ranges (5 most ordered cities):")
    city_means = df_merged.groupby('city_code')['entropy_normalised'].agg(
        ['mean', 'std', 'min', 'max', 'count']
    ).round(4).sort_values('mean')
    print(city_means.head(5).to_string())
    print()
    print(f"Per-city entropy ranges (5 most disordered cities):")
    print(city_means.tail(5).to_string())


if __name__ == "__main__":
    main()