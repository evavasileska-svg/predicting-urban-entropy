"""
Data inspection script for the merged training dataset.

Prints to terminal:
  1. Dataset shape (rows, columns, memory)
  2. Column overview (dtype, null counts)
  3. Numeric feature statistics (describe)
  4. Per-feature completeness (sorted)
  5. Per-patch null analysis (how many patches missing N features)
  6. Per-city completeness
  7. Per-entropy-bin completeness
  8. Anomaly checks (inf, negatives, out-of-range)
  9. Target variable check

No plots, no files saved — pure terminal output.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR


TARGET = 'entropy_normalised'

# columns that are identifiers/metadata, not features
ID_COLUMNS = {
    'patch_id', 'city_code', 'patch_idx',
    'centroid_lat', 'centroid_lon',
    'minx_utm', 'miny_utm', 'maxx_utm', 'maxy_utm', 'crs_utm',
}

# target-like columns (alternative entropy measures)
TARGET_COLUMNS = {
    'entropy_raw', 'entropy_normalised',
    'entropy_weighted_raw', 'entropy_weighted_norm', 'phi',
}

# features expected to be in [0, 1] range
PROPORTION_FEATURES = {
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'largest_fraction', 'building_density',
    'mean_block_eri', 'mean_block_compact',
    'mean_block_elong', 'mean_block_convex',
    'entropy_normalised', 'entropy_weighted_norm', 'phi',
}

# features expected to be strictly non-negative
NON_NEGATIVE_FEATURES = {
    'n_segments', 'n_intersections', 'n_nodes_in_patch',
    'n_components', 'n_nodes_largest',
    'n_edges_used', 'n_4way', 'n_3way', 'n_deadend',
    'mean_edge_length', 'total_edge_length',
    'intersection_density', 'street_density',
    'meshedness', 'circuity',
    'n_blocks', 'mean_block_area', 'std_block_area', 'cv_block_area',
    'n_buildings', 'mean_building_footprint',
    'std_building_footprint',
    'mean_nn_distance', 'std_nn_distance',
}


def section_header(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def inspect_dataset_shape(df):
    """Section 1: dataset shape and memory."""
    section_header("1. DATASET SHAPE")
    print(f"\n  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  Memory:  {mem_mb:.2f} MB")


def inspect_columns(df):
    """Section 2: column overview."""
    section_header("2. COLUMN OVERVIEW")
    print()
    overview = pd.DataFrame({
        'dtype':     df.dtypes.astype(str),
        'non_null':  df.notna().sum(),
        'null':      df.isna().sum(),
        'null_pct':  (df.isna().sum() / len(df) * 100).round(1),
    })
    print(overview.to_string())


def inspect_numeric_stats(df):
    """Section 3: numeric feature statistics."""
    section_header("3. NUMERIC FEATURE STATISTICS")
    numeric = df.select_dtypes(include=[np.number])
    print()
    print(numeric.describe().round(3).to_string())


def inspect_completeness_sorted(df):
    """Section 4: per-feature completeness, sorted."""
    section_header("4. PER-FEATURE COMPLETENESS (sorted)")

    completeness = pd.DataFrame({
        'non_null': df.notna().sum(),
        'null':     df.isna().sum(),
        'pct':      (df.notna().sum() / len(df) * 100).round(1),
    }).sort_values('pct', ascending=False)

    print()

    # break into bands for clarity
    full = completeness[completeness['pct'] == 100.0]
    partial = completeness[
        (completeness['pct'] < 100.0) & (completeness['pct'] > 0.0)
    ]
    empty = completeness[completeness['pct'] == 0.0]

    print(f"  FULL coverage (100%): {len(full)} columns")
    if len(full) > 0:
        for col in full.index:
            print(f"    - {col}")
    print()

    print(f"  PARTIAL coverage: {len(partial)} columns")
    if len(partial) > 0:
        print(partial.to_string())
    print()

    print(f"  EMPTY (0%): {len(empty)} columns")
    if len(empty) > 0:
        for col in empty.index:
            print(f"    - {col}")


def inspect_per_patch_nulls(df):
    """Section 5: how many features are missing per patch."""
    section_header("5. PER-PATCH NULL ANALYSIS")

    # consider only feature columns (exclude IDs and targets)
    feature_cols = [
        c for c in df.columns
        if c not in ID_COLUMNS and c not in TARGET_COLUMNS
    ]
    feature_df = df[feature_cols]

    null_per_row = feature_df.isna().sum(axis=1)
    counts = null_per_row.value_counts().sort_index()

    print(f"\n  Considering {len(feature_cols)} feature columns")
    print(f"  (excluding identifiers and target variables)\n")
    print(f"  Distribution of # missing features per patch:")
    print(f"  {'# missing':<12} {'# patches':<12} {'% patches':<12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12}")
    for n_missing, count in counts.items():
        pct = 100 * count / len(df)
        print(f"  {n_missing:<12} {count:<12,} {pct:<12.1f}")

    n_complete = (null_per_row == 0).sum()
    n_any_missing = (null_per_row > 0).sum()
    print()
    print(f"  Patches with ALL features: {n_complete:,} "
          f"({100*n_complete/len(df):.1f}%)")
    print(f"  Patches missing >=1 feat:  {n_any_missing:,} "
          f"({100*n_any_missing/len(df):.1f}%)")


def inspect_per_city(df):
    """Section 6: per-city completeness."""
    section_header("6. PER-CITY FEATURE COMPLETENESS")

    feature_cols = [
        c for c in df.columns
        if c not in ID_COLUMNS and c not in TARGET_COLUMNS
    ]

    city_stats = []
    for city, group in df.groupby('city_code'):
        n_patches = len(group)
        feature_df = group[feature_cols]
        total_cells = n_patches * len(feature_cols)
        null_cells = feature_df.isna().sum().sum()
        completeness_pct = round(
            100 * (1 - null_cells / total_cells), 1
        )
        n_full_patches = (feature_df.isna().sum(axis=1) == 0).sum()
        city_stats.append({
            'city_code':           city,
            'n_patches':           n_patches,
            'pct_complete':        completeness_pct,
            'patches_full':        n_full_patches,
            'patches_full_pct':    round(
                100 * n_full_patches / n_patches, 1
            ),
        })

    city_df = pd.DataFrame(city_stats).sort_values(
        'pct_complete', ascending=False
    )

    print(f"\n  Considering {len(feature_cols)} feature columns\n")
    print(f"  Cities sorted by overall completeness:\n")
    print(city_df.to_string(index=False))


def inspect_per_entropy_bin(df):
    """Section 7: per-entropy-bin completeness.

    Uses the same bin edges as the stratification script (08), which
    computed bins from the full patch_entropy.csv (25,462 patches).
    """
    section_header("7. PER-ENTROPY-BIN FEATURE COMPLETENESS")

    feature_cols = [
        c for c in df.columns
        if c not in ID_COLUMNS and c not in TARGET_COLUMNS
    ]

    # load the full entropy dataset to get the same bin edges
    # the stratification script used
    entropy_path = PROCESSED_DIR / "patch_entropy.csv"
    if not entropy_path.exists():
        print(f"\n  WARNING: {entropy_path.name} not found.")
        print(f"  Cannot reproduce stratification bins. Skipping section.")
        return

    full_entropy = pd.read_csv(entropy_path)
    ho_min = full_entropy[TARGET].min()
    ho_max = full_entropy[TARGET].max()

    # reproduce the exact bin edges from script 08
    n_bins = 10
    bin_edges = np.linspace(ho_min, ho_max + 1e-6, n_bins + 1)

    df = df.copy()
    df['bin'] = pd.cut(
        df[TARGET],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    ).astype(int)

    bin_stats = []
    for bin_idx, group in df.groupby('bin'):
        n_patches = len(group)
        feature_df = group[feature_cols]
        total_cells = n_patches * len(feature_cols)
        null_cells = feature_df.isna().sum().sum()
        completeness_pct = round(
            100 * (1 - null_cells / total_cells), 1
        )

        bin_low  = bin_edges[bin_idx]
        bin_high = bin_edges[bin_idx + 1]

        n_full_patches = (feature_df.isna().sum(axis=1) == 0).sum()
        bin_stats.append({
            'bin':              int(bin_idx),
            'ho_range':         f"[{bin_low:.3f}, {bin_high:.3f}]",
            'n_patches':        n_patches,
            'pct_complete':     completeness_pct,
            'patches_full':     n_full_patches,
            'patches_full_pct': round(
                100 * n_full_patches / n_patches, 1
            ),
        })

    bin_df = pd.DataFrame(bin_stats)

    print(f"\n  Considering {len(feature_cols)} feature columns")
    print(f"  Using stratification bin edges from script 08")
    print(f"  (computed from full patch_entropy.csv: "
          f"Ho range {ho_min:.4f}–{ho_max:.4f})\n")
    print(f"  Entropy bins (ordered low → high Ho):\n")
    print(bin_df.to_string(index=False))

def inspect_anomalies(df):
    """Section 8: anomaly checks."""
    section_header("8. ANOMALY CHECKS")

    numeric = df.select_dtypes(include=[np.number])

    # infinite values
    inf_per_col = np.isinf(numeric).sum()
    inf_cols = inf_per_col[inf_per_col > 0]
    print(f"\n  Columns with infinite values:")
    if len(inf_cols) == 0:
        print(f"    None ✓")
    else:
        for col, n in inf_cols.items():
            print(f"    {col}: {n}")

    # negative values in features that should be non-negative
    print(f"\n  Negative values in non-negative features:")
    found_any_neg = False
    for col in NON_NEGATIVE_FEATURES:
        if col in numeric.columns:
            n_neg = (numeric[col] < 0).sum()
            if n_neg > 0:
                print(f"    {col}: {n_neg} negative values")
                found_any_neg = True
    if not found_any_neg:
        print(f"    None ✓")

    # proportion features outside [0, 1]
    print(f"\n  Proportion features outside [0, 1]:")
    found_any_oor = False
    for col in PROPORTION_FEATURES:
        if col in numeric.columns:
            below = (numeric[col] < 0).sum()
            above = (numeric[col] > 1).sum()
            if below > 0 or above > 0:
                print(f"    {col}: {below} below 0, {above} above 1")
                found_any_oor = True
    if not found_any_oor:
        print(f"    None ✓")


def inspect_target(df):
    """Section 9: target variable check."""
    section_header("9. TARGET VARIABLE CHECK")

    if TARGET not in df.columns:
        print(f"\n  ERROR: target column '{TARGET}' not in dataset")
        return

    target = df[TARGET]
    print(f"\n  Target column: {TARGET}")
    print(f"  Non-null:      {target.notna().sum():,} "
          f"({100*target.notna().sum()/len(df):.1f}%)")
    print(f"  Null:          {target.isna().sum():,}")
    print()
    print(f"  Statistics:")
    print(target.describe().round(4).to_string())


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Run 13_merge_features.py first.")
        return

    df = pd.read_csv(csv_path)

    print(f"\nInspecting: {csv_path.name}")
    print(f"=" * 70)

    inspect_dataset_shape(df)
    inspect_columns(df)
    inspect_numeric_stats(df)
    inspect_completeness_sorted(df)
    inspect_per_patch_nulls(df)
    inspect_per_city(df)
    inspect_per_entropy_bin(df)
    inspect_anomalies(df)
    inspect_target(df)

    print(f"\n{'=' * 70}")
    print(f"  Inspection complete")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()