
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR


# ── target and feature column definitions ──────────────────────────
# columns to drop from feature CSVs to avoid duplicates after merge
DROP_FROM_GRAPH    = ['n_edges_used']
DROP_FROM_CIRCUITY = ['total_network_length', 'total_straight_length',
                      'n_edges_in_circuity']
DROP_FROM_DISTANCE = []

# target variable for ML
TARGET_COLUMN = 'entropy_normalised'

# features to exclude from the clean training dataset
# (metadata, diagnostics, alternative targets)
METADATA_COLUMNS = [
    'patch_id', 'city_code', 'patch_idx',
    'centroid_lat', 'centroid_lon',
    'minx_utm', 'miny_utm', 'maxx_utm', 'maxy_utm', 'crs_utm',
    'n_segments', 'n_intersections',
    'n_nodes_in_patch', 'n_components', 'n_nodes_largest',
    'largest_fraction',
    'entropy_raw', 'entropy_weighted_raw', 'n_edges_used',
]

# alternative targets (we are not predicting these but they live in the file)
ALTERNATIVE_TARGETS = [
    'entropy_weighted_norm', 'phi',
]


def merge_features():
    
    
    print(f"{'=' * 70}")
    print(f"Merging features into final training dataset")
    print(f"(graph + circuity + distance to center)")
    print(f"{'=' * 70}\n")

    sample_path    = PROCESSED_DIR / "patch_stratified_sample.csv"
    graph_path     = PROCESSED_DIR / "patch_graph_features.csv"
    circuity_path  = PROCESSED_DIR / "patch_circuity.csv"
    distance_path  = PROCESSED_DIR / "patch_distance_to_center.csv"

    for path in [sample_path, graph_path, circuity_path, distance_path]:
        if not path.exists():
            print(f"ERROR: {path.name} not found")
            return

    sample    = pd.read_csv(sample_path)
    graph     = pd.read_csv(graph_path)
    circuity  = pd.read_csv(circuity_path)
    distance  = pd.read_csv(distance_path)

    print(f"Loaded inputs:")
    print(f"  Sample:    {len(sample):,} patches "
          f"({len(sample.columns)} columns)")
    print(f"  Graph:     {len(graph):,} patches "
          f"({len(graph.columns)} columns)")
    print(f"  Circuity:  {len(circuity):,} patches "
          f"({len(circuity.columns)} columns)")
    print(f"  Distance:  {len(distance):,} patches "
          f"({len(distance.columns)} columns)")
    print()

    # ── prepare each features dataframe for merge ───────────────────
    # drop redundant columns and the city_code (we already have it
    # from the sample CSV)
    def prep(df, drop_cols):
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.drop(columns=['city_code'], errors='ignore')
        return df

    graph_clean    = prep(graph,    DROP_FROM_GRAPH)
    circuity_clean = prep(circuity, DROP_FROM_CIRCUITY)
    distance_clean = prep(distance, DROP_FROM_DISTANCE)

    # ── left-join everything onto the sample ────────────────────────
    # sample is the master because it has all 2,500 patches.
    # left-join preserves all sample rows; missing features become NaN
    merged = sample.merge(graph_clean,    on='patch_id', how='left')
    merged = merged.merge(circuity_clean, on='patch_id', how='left')
    merged = merged.merge(distance_clean, on='patch_id', how='left')

    n_rows = len(merged)
    n_cols = len(merged.columns)
    print(f"Merged dataset: {n_rows:,} rows × {n_cols} columns\n")

    # ── save the FULL version (with metadata) ───────────────────────
    full_path = PROCESSED_DIR / "patch_training_data_full.csv"
    merged.to_csv(full_path, index=False)
    print(f"Saved full dataset:  {full_path}")
    print(f"  Columns: {n_cols}\n")

    # ── build the CLEAN version (ML-ready) ──────────────────────────
    # the clean version keeps: patch_id, city_code, the target, and features
    # we want to DROP metadata columns and alternative targets
    # but KEEP patch_id and city_code (they are identifiers needed for tracking)

    # columns to exclude (everything that is NOT patch_id, city_code, target,
    # or a real feature)
    exclude_for_clean = set(METADATA_COLUMNS + ALTERNATIVE_TARGETS)
    # but make sure patch_id and city_code stay
    exclude_for_clean.discard('patch_id')
    exclude_for_clean.discard('city_code')

    keep_cols = [
        c for c in merged.columns
        if c not in exclude_for_clean
        and c not in ('n_nodes_used',)   # diagnostic from graph script
    ]
    clean = merged[keep_cols].copy()

    # reorder: patch_id, city_code, target, then features
    feature_cols = [
        c for c in clean.columns
        if c not in ('patch_id', 'city_code', TARGET_COLUMN)
    ]
    clean = clean[['patch_id', 'city_code', TARGET_COLUMN] + feature_cols]

    clean_path = PROCESSED_DIR / "patch_training_data_clean.csv"
    clean.to_csv(clean_path, index=False)
    print(f"Saved clean dataset: {clean_path}")
    print(f"  Columns: {len(clean.columns)}\n")

    # ── report on the clean dataset ─────────────────────────────────
    print(f"{'=' * 70}")
    print(f"CLEAN DATASET SUMMARY (for ML training)")
    print(f"{'=' * 70}\n")
    print(f"Rows:    {len(clean):,}")
    print(f"Target:  {TARGET_COLUMN}")
    print(f"Features ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  - {f}")
    print()

    # ── coverage report ─────────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"FEATURE COVERAGE (non-null counts)")
    print(f"{'=' * 70}\n")
    coverage = clean[feature_cols].notna().sum()
    coverage_pct = (coverage / len(clean) * 100).round(1)
    coverage_df = pd.DataFrame({
        'non_null': coverage,
        'pct': coverage_pct,
    }).sort_values('non_null', ascending=False)
    print(coverage_df.to_string())
    print()

    # ── target distribution ─────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"TARGET ({TARGET_COLUMN}) DISTRIBUTION")
    print(f"{'=' * 70}\n")
    print(clean[TARGET_COLUMN].describe().round(4).to_string())


if __name__ == "__main__":
    merge_features()