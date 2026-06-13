import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR


# columns to drop from feature CSVs to avoid duplicates after merge
DROP_FROM_GRAPH    = ['n_edges_used']
DROP_FROM_CIRCUITY = ['total_network_length', 'total_straight_length',
                      'n_edges_in_circuity']

# features to drop from the final dataset (redundant or weak predictors)
FEATURES_TO_DROP = [
    'meshedness',
    'mean_degree',
    'street_density',
    'n_4way',
    'intersection_density',
]

TARGET_COLUMN = 'entropy_normalised'

# columns to exclude from clean dataset (metadata + diagnostics + alt targets)
METADATA_COLUMNS = [
    'patch_id', 'city_code', 'patch_idx',
    'centroid_lat', 'centroid_lon',
    'minx_utm', 'miny_utm', 'maxx_utm', 'maxy_utm', 'crs_utm',
    'n_segments', 'n_intersections',
    'n_nodes_in_patch', 'n_components', 'n_nodes_largest',
    'largest_fraction',
    'entropy_raw', 'entropy_weighted_raw', 'n_edges_used',
]

ALTERNATIVE_TARGETS = [
    'entropy_weighted_norm', 'phi',
]


def merge_features():
    print(f"{'=' * 70}")
    print(f"Merging features into final training dataset")
    print(f"(graph + circuity, with redundant features dropped)")
    print(f"{'=' * 70}\n")

    sample_path   = PROCESSED_DIR / "patch_stratified_sample.csv"
    graph_path    = PROCESSED_DIR / "patch_graph_features.csv"
    circuity_path = PROCESSED_DIR / "patch_circuity.csv"

    for path in [sample_path, graph_path, circuity_path]:
        if not path.exists():
            print(f"ERROR: {path.name} not found")
            return

    sample   = pd.read_csv(sample_path)
    graph    = pd.read_csv(graph_path)
    circuity = pd.read_csv(circuity_path)

    print(f"Loaded inputs:")
    print(f"  Sample:    {len(sample):,} patches "
          f"({len(sample.columns)} columns)")
    print(f"  Graph:     {len(graph):,} patches "
          f"({len(graph.columns)} columns)")
    print(f"  Circuity:  {len(circuity):,} patches "
          f"({len(circuity.columns)} columns)")
    print()

    # prepare each features dataframe for merge
    def prep(df, drop_cols):
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.drop(columns=['city_code'], errors='ignore')
        return df

    graph_clean    = prep(graph,    DROP_FROM_GRAPH)
    circuity_clean = prep(circuity, DROP_FROM_CIRCUITY)

    # left-join everything onto the sample
    merged = sample.merge(graph_clean,    on='patch_id', how='left')
    merged = merged.merge(circuity_clean, on='patch_id', how='left')

    # drop the redundant features
    cols_dropped = []
    for col in FEATURES_TO_DROP:
        if col in merged.columns:
            merged = merged.drop(columns=[col])
            cols_dropped.append(col)

    print(f"Dropped {len(cols_dropped)} redundant features:")
    for col in cols_dropped:
        print(f"  - {col}")
    print()

    n_rows = len(merged)
    n_cols = len(merged.columns)
    print(f"Merged dataset: {n_rows:,} rows x {n_cols} columns\n")

    # save the FULL version (with metadata)
    full_path = PROCESSED_DIR / "dataset_dropped_features_full.csv"
    merged.to_csv(full_path, index=False)
    print(f"Saved full dataset:  {full_path}")
    print(f"  Columns: {n_cols}\n")

    # build the CLEAN version (ML-ready)
    exclude_for_clean = set(METADATA_COLUMNS + ALTERNATIVE_TARGETS)

    keep_cols = [
        c for c in merged.columns
        if c not in exclude_for_clean
        and c not in ('n_nodes_used',)
    ]
    clean = merged[keep_cols].copy()

    # reorder: target first, then features
    feature_cols = [c for c in clean.columns if c != TARGET_COLUMN]
    clean = clean[[TARGET_COLUMN] + feature_cols]

    clean_path = PROCESSED_DIR / "dataset_dropped_features_clean.csv"
    clean.to_csv(clean_path, index=False)
    print(f"Saved clean dataset: {clean_path}")
    print(f"  Columns: {len(clean.columns)}\n")

    print(f"{'=' * 70}")
    print(f"CLEAN DATASET SUMMARY (for ML training)")
    print(f"{'=' * 70}\n")
    print(f"Rows:    {len(clean):,}")
    print(f"Target:  {TARGET_COLUMN}")
    print(f"Features ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  - {f}")
    print()

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

    print(f"{'=' * 70}")
    print(f"TARGET ({TARGET_COLUMN}) DISTRIBUTION")
    print(f"{'=' * 70}\n")
    print(clean[TARGET_COLUMN].describe().round(4).to_string())


if __name__ == "__main__":
    merge_features()