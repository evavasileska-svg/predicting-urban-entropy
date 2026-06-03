"""
Apply an absolute minimum intersection floor to the patch inventory.

This is the second filtering pass. The first filter
(03_filter_patches_intersection.py) kept the top 70% of patches
per city by intersection count. That filter is relative — it adapts
to each city's distribution.

This second filter applies an absolute global floor: drop any patch
with fewer than 30 intersections. This catches sparse patches that
survived the per-city percentile but are still essentially rural —
particularly common in cities with large bounding boxes that pulled
in agricultural surroundings (Verona, Thessaloniki, Brasília etc.).

Together the two filters ensure:
  - Each city contributes its relative best patches
  - Every kept patch meets a universal "urban density" threshold

Outputs:
  - data/processed/patch_inventory_filtered_v2.csv (kept patches)
  - data/processed/patch_filter_v2_report.csv (per-city statistics)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR


# ── filter configuration ───────────────────────────────────────────
MIN_INTERSECTIONS = 30   # absolute floor: drop patches below this


def filter_patches_min_intersection():
    """Apply the absolute intersection floor filter."""
    # load the percentile-filtered inventory
    input_path = PROCESSED_DIR / "patch_inventory_filtered.csv"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        print(f"Run 03_filter_patches_intersection.py first.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded percentile-filtered inventory with "
          f"{len(df):,} patches from {df['city_code'].nunique()} cities")
    print()

    # build a per-city report before applying the filter
    report_rows = []
    for city, group in df.groupby('city_code'):
        n_before = len(group)
        kept = group[group['n_intersections'] >= MIN_INTERSECTIONS]
        n_after = len(kept)

        report_rows.append({
            'city_code':       city,
            'n_before_floor':  n_before,
            'n_after_floor':   n_after,
            'n_dropped':       n_before - n_after,
            'kept_pct':        round(100 * n_after / n_before, 1)
                               if n_before > 0 else 0,
        })

    # apply the floor filter to the full dataset
    filtered_df = df[df['n_intersections'] >= MIN_INTERSECTIONS].copy()

    # save outputs
    filtered_path = PROCESSED_DIR / "patch_inventory_filtered_v2.csv"
    report_path = PROCESSED_DIR / "patch_filter_v2_report.csv"

    filtered_df.to_csv(filtered_path, index=False)
    report_df = pd.DataFrame(report_rows).sort_values(
        'n_after_floor', ascending=False
    )
    report_df.to_csv(report_path, index=False)

    # summary
    print(f"{'=' * 70}")
    print(f"Floor filter applied: minimum {MIN_INTERSECTIONS} "
          f"intersections per patch (absolute threshold)")
    print()
    print(f"Patches before floor:  {len(df):,}")
    print(f"Patches after floor:   {len(filtered_df):,}")
    print(f"Dropped by floor:      {len(df) - len(filtered_df):,} "
          f"({100 * (1 - len(filtered_df) / len(df)):.1f}%)")
    print()
    print(f"Saved:")
    print(f"  {filtered_path}")
    print(f"  {report_path}")
    print()

    # show cities most affected by the floor
    most_affected = report_df.sort_values('n_dropped', ascending=False)
    print(f"Top 10 cities most affected by the floor filter:")
    print(most_affected.head(10).to_string(index=False))
    print()

    # show cities least affected
    least_affected = report_df.sort_values('n_dropped', ascending=True)
    print(f"Top 10 cities least affected (most patches survive):")
    print(least_affected.head(10).to_string(index=False))
    print()

    # final per-city counts after both filters
    print(f"Final patches per city (top 10 largest):")
    print(report_df.head(10)[['city_code', 'n_after_floor']]
          .to_string(index=False))
    print()
    print(f"Final patches per city (bottom 10 smallest):")
    print(report_df.tail(10)[['city_code', 'n_after_floor']]
          .to_string(index=False))


if __name__ == "__main__":
    filter_patches_min_intersection()