"""
Filter the patch inventory to remove low-quality perimetral patches.

Applies a per-city percentile-based filter:
  - Keep the top 70% of patches per city by intersection count
  - Removes the sparsest perimetral patches from each city
  - Adapts to each city's character (fair to small cities)

Outputs:
  - data/processed/patch_inventory_filtered.csv (kept patches)
  - data/processed/patch_filter_report.csv (per-city statistics)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR


# ── filter configuration ───────────────────────────────────────────
KEEP_TOP_PERCENT = 0.70   # keep top 70% by intersection count per city


def filter_patches():
    """Apply the per-city percentile filter."""
    # load the raw inventory
    inventory_path = PROCESSED_DIR / "patch_inventory.csv"
    if not inventory_path.exists():
        print(f"ERROR: {inventory_path} not found.")
        print(f"Run 01_generate_patches.py first.")
        return

    df = pd.read_csv(inventory_path)
    print(f"Loaded inventory with {len(df):,} patches "
          f"from {df['city_code'].nunique()} cities")
    print()

    # apply filter per city
    kept_rows = []
    report_rows = []

    for city, group in df.groupby('city_code'):
        n_original = len(group)

        # compute the intersection threshold as the (1 - KEEP_TOP_PERCENT)
        # quantile of intersection counts within this city
        threshold = group['n_intersections'].quantile(
            1.0 - KEEP_TOP_PERCENT
        )

        # keep patches at or above the threshold
        kept = group[group['n_intersections'] >= threshold]
        n_kept = len(kept)

        kept_rows.append(kept)

        report_rows.append({
            'city_code':           city,
            'n_original':          n_original,
            'n_kept':              n_kept,
            'n_dropped':           n_original - n_kept,
            'threshold_int_count': round(threshold, 1),
            'min_int_count':       int(group['n_intersections'].min()),
            'median_int_count':    int(group['n_intersections'].median()),
            'max_int_count':       int(group['n_intersections'].max()),
            'kept_pct':            round(100 * n_kept / n_original, 1),
        })

    # combine all kept patches
    filtered_df = pd.concat(kept_rows, ignore_index=True)
    report_df = pd.DataFrame(report_rows).sort_values(
        'n_kept', ascending=False
    )

    # save outputs
    filtered_path = PROCESSED_DIR / "patch_inventory_filtered.csv"
    report_path = PROCESSED_DIR / "patch_filter_report.csv"

    filtered_df.to_csv(filtered_path, index=False)
    report_df.to_csv(report_path, index=False)

    # summary
    print(f"{'=' * 70}")
    print(f"Filter applied: keeping top {int(KEEP_TOP_PERCENT * 100)}% "
          f"of patches per city by intersection count")
    print()
    print(f"Original patches:  {len(df):,}")
    print(f"Filtered patches:  {len(filtered_df):,}")
    print(f"Dropped:           {len(df) - len(filtered_df):,} "
          f"({100 * (1 - len(filtered_df) / len(df)):.1f}%)")
    print()
    print(f"Saved:")
    print(f"  {filtered_path}")
    print(f"  {report_path}")
    print()

    # show top 10 and bottom 10 cities by kept count
    print(f"Filter report (top 10 by kept patches):")
    print(report_df.head(10).to_string(index=False))
    print()
    print(f"Filter report (bottom 10 by kept patches):")
    print(report_df.tail(10).to_string(index=False))
    print()

    # show threshold distribution across cities
    print(f"Per-city intersection thresholds:")
    print(f"  Min threshold:    {report_df['threshold_int_count'].min():.1f}")
    print(f"  Median threshold: {report_df['threshold_int_count'].median():.1f}")
    print(f"  Max threshold:    {report_df['threshold_int_count'].max():.1f}")


if __name__ == "__main__":
    filter_patches()