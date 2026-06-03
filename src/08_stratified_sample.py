"""
Stratified sampling of patches by entropy (Ho).

Reads the entropy-computed patch dataset and samples patches
in equal numbers from entropy bins to produce a balanced
training dataset.

Method:
  1. Define N_BINS bins evenly spanning the observed Ho range
  2. For each bin: sample min(TARGET_PER_BIN, available) patches
  3. If a bin has fewer than TARGET_PER_BIN patches, take all
  4. Result: roughly TARGET_TOTAL patches with balanced entropy

Why stratification:
  The raw entropy distribution is heavily skewed toward
  high Ho. Without stratification, the ML model would see
  far more disordered than ordered patches and become biased.
  Stratified sampling ensures the model learns the full range.

Inputs:
  - data/processed/patch_entropy.csv

Outputs:
  - data/processed/patch_stratified_sample.csv (the sample)
  - data/processed/patch_stratification_report.csv (bin sizes)
  - results/figures/stratification_distribution.png (visualisation)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


# ── stratification configuration ───────────────────────────────────
TARGET_TOTAL    = 2500          # target final sample size
N_BINS          = 10            # number of entropy bins
TARGET_PER_BIN  = TARGET_TOTAL // N_BINS   # 250 per bin
RANDOM_SEED     = 42            # for reproducibility

def stratified_sample():
    """Apply stratified sampling on Ho."""
    # load the full entropy dataset
    entropy_path = PROCESSED_DIR / "patch_entropy.csv"
    if not entropy_path.exists():
        print(f"ERROR: {entropy_path} not found.")
        print(f"Run 06_compute_patch_entropy.py first.")
        return

    df = pd.read_csv(entropy_path)
    n_total = len(df)
    print(f"Loaded {n_total:,} patches from "
          f"{df['city_code'].nunique()} cities\n")
    print(f"Stratification target: {TARGET_TOTAL} patches "
          f"({TARGET_PER_BIN} per bin × {N_BINS} bins)\n")

    # define bins across the observed Ho range
    ho_min = df['entropy_normalised'].min()
    ho_max = df['entropy_normalised'].max()
    # slight padding on max so the right edge is inclusive
    bin_edges = np.linspace(ho_min, ho_max + 1e-6, N_BINS + 1)

    print(f"Ho range: [{ho_min:.4f}, {ho_max:.4f}]")
    print(f"Bin width: {(ho_max - ho_min) / N_BINS:.4f}\n")

    # assign each patch to a bin
    df['bin'] = pd.cut(
        df['entropy_normalised'],
        bins=bin_edges,
        labels=range(N_BINS),
        include_lowest=True,
    ).astype(int)

    # report bin populations
    bin_counts = df['bin'].value_counts().sort_index()

    # apply stratified sampling
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_dfs = []
    report_rows = []

    print(f"{'Bin':>4} {'Range':>20} {'Available':>11} "
          f"{'Target':>8} {'Sampled':>9} {'Coverage':>10}")
    print("-" * 70)

    for bin_idx in range(N_BINS):
        bin_low  = bin_edges[bin_idx]
        bin_high = bin_edges[bin_idx + 1]
        n_available = bin_counts.get(bin_idx, 0)
        n_target = TARGET_PER_BIN
        n_sample = min(n_available, n_target)

        # pick patches from this bin
        bin_df = df[df['bin'] == bin_idx]
        if n_sample > 0:
            sampled_idx = rng.choice(
                bin_df.index.values, size=n_sample, replace=False
            )
            sampled_dfs.append(df.loc[sampled_idx])

        coverage = (
            f"{100 * n_sample / n_target:5.1f}%"
            if n_target > 0 else "n/a"
        )
        range_str = f"[{bin_low:.3f}, {bin_high:.3f}]"

        print(f"{bin_idx:>4} {range_str:>20} {n_available:>11,} "
              f"{n_target:>8} {n_sample:>9,} {coverage:>10}")

        report_rows.append({
            'bin':          bin_idx,
            'ho_low':       round(bin_low, 4),
            'ho_high':      round(bin_high, 4),
            'n_available':  n_available,
            'n_target':     n_target,
            'n_sampled':    n_sample,
            'coverage_pct': round(100 * n_sample / n_target, 1)
                            if n_target > 0 else 0,
        })

    # combine the sampled rows
    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    # drop the temporary bin column for cleanliness
    if 'bin' in sample_df.columns:
        sample_df = sample_df.drop(columns=['bin'])

    n_final = len(sample_df)

    # save outputs
    sample_path = PROCESSED_DIR / "patch_stratified_sample.csv"
    report_path = PROCESSED_DIR / "patch_stratification_report.csv"
    sample_df.to_csv(sample_path, index=False)
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)

    print("-" * 70)
    print(f"\n{'=' * 70}")
    print(f"Sample size:       {n_final:,} patches "
          f"({100 * n_final / TARGET_TOTAL:.1f}% of target)")
    print(f"Bins fully filled: "
          f"{(report_df['coverage_pct'] >= 100).sum()} / {N_BINS}")
    print(f"Bins undersized:   "
          f"{(report_df['coverage_pct'] < 100).sum()} / {N_BINS}")
    print()

    # diagnostics — cities most represented in the sample
    city_counts = sample_df['city_code'].value_counts()
    print(f"Cities most represented in sample (top 15):")
    print(city_counts.head(15).to_string())
    print()
    print(f"Cities least represented in sample (bottom 15):")
    print(city_counts.tail(15).to_string())
    print()

    print(f"Saved:")
    print(f"  {sample_path}")
    print(f"  {report_path}")

    # create visualisation
    create_visualisation(df, sample_df, bin_edges, report_df)


def create_visualisation(df_all, df_sample, bin_edges, report_df):
    """Create a figure comparing original and stratified distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── left panel: full vs sampled histogram ──────────────────
    ax = axes[0]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.hist(
        df_all['entropy_normalised'], bins=bin_edges,
        color='steelblue', alpha=0.5, edgecolor='black',
        label=f"All patches (n={len(df_all):,})",
        zorder=2,
    )
    ax.hist(
        df_sample['entropy_normalised'], bins=bin_edges,
        color='orange', alpha=0.8, edgecolor='black',
        label=f"Stratified sample (n={len(df_sample):,})",
        zorder=3,
    )
    ax.axvline(0.80, color='green', linestyle='--', alpha=0.5)
    ax.axvline(0.95, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Ho", fontsize=12)
    ax.set_ylabel("Number of patches", fontsize=12)
    ax.set_title("Original distribution vs stratified sample",
                 fontsize=13)
    ax.legend(fontsize=11)

    # ── right panel: bin coverage ──────────────────────────────
    ax = axes[1]
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    bin_centers = [
        (bin_edges[i] + bin_edges[i + 1]) / 2
        for i in range(len(bin_edges) - 1)
    ]
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.9
    colors = ['seagreen' if cov >= 100 else 'indianred'
              for cov in report_df['coverage_pct']]

    ax.bar(
        bin_centers, report_df['coverage_pct'],
        width=bar_width, color=colors, alpha=0.8,
        edgecolor='black', zorder=2,
    )
    ax.axhline(100, color='black', linestyle='--', alpha=0.5,
               label=f"Target ({TARGET_PER_BIN} patches per bin)")
    ax.set_xlabel("Ho (bin centre)", fontsize=12)
    ax.set_ylabel("Coverage (% of bin target)", fontsize=12)
    ax.set_title(
        "Per-bin coverage\n"
        f"green = fully filled, red = undersized",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "stratification_distribution.png"
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  {fig_path}")


if __name__ == "__main__":
    stratified_sample()