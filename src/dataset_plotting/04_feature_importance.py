"""
Plot feature importance ranking by correlation with entropy.

For each of the 13 graph + circuity features:
  - Compute Pearson correlation with entropy_normalised
  - Display as horizontal bar chart
  - Sorted by absolute correlation (strongest at top)
  - Colored by sign (positive = blue, negative = red)

Outputs:
  - results/figures/feature_importance_correlation.png
  - Prints the ranking to terminal
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR


TARGET = 'entropy_normalised'

# graph + circuity features (all 100% complete)
FEATURES_TO_RANK = [
    'n_4way', 'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_degree',
    'mean_edge_length', 'total_edge_length',
    'meshedness',
    'intersection_density', 'street_density',
    'circuity',
]


def compute_correlations(df):
    """Compute Pearson correlation of each feature with target."""
    correlations = []
    for feature in FEATURES_TO_RANK:
        if feature not in df.columns:
            print(f"  WARNING: {feature} not in dataset, skipping")
            continue
        valid = df.dropna(subset=[feature, TARGET])
        if len(valid) == 0:
            continue
        r = valid[feature].corr(valid[TARGET])
        correlations.append({
            'feature': feature,
            'r':       r,
            'abs_r':   abs(r),
            'n':       len(valid),
        })

    # sort by absolute correlation, strongest first
    corr_df = pd.DataFrame(correlations).sort_values(
        'abs_r', ascending=False
    ).reset_index(drop=True)

    return corr_df


def plot_correlation_ranking(corr_df):
    """Horizontal bar chart of correlations."""
    # set up figure
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor('white')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5, alpha=0.5, zorder=0, axis='x')
    ax.set_axisbelow(True)

    # reverse for plotting (strongest at top of chart)
    plot_df = corr_df.iloc[::-1].reset_index(drop=True)

    # colors: blue for positive, red for negative
    colors = ['steelblue' if r > 0 else 'indianred'
              for r in plot_df['r']]

    # horizontal bars
    bars = ax.barh(
        plot_df['feature'],
        plot_df['r'],
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        zorder=2,
    )

    # add text labels at end of each bar
    for bar, r_val in zip(bars, plot_df['r']):
        width = bar.get_width()
        label_x = width + (0.015 if width >= 0 else -0.015)
        ha = 'left' if width >= 0 else 'right'
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{r_val:+.3f}",
            va='center',
            ha=ha,
            fontsize=10,
            fontweight='bold',
        )

    # vertical line at zero
    ax.axvline(0, color='black', linewidth=0.8, zorder=3)

    # reference lines for "strong" correlation thresholds
    ax.axvline(0.5, color='grey', linestyle=':', alpha=0.5, zorder=1)
    ax.axvline(-0.5, color='grey', linestyle=':', alpha=0.5, zorder=1)
    ax.axvline(0.3, color='grey', linestyle=':', alpha=0.3, zorder=1)
    ax.axvline(-0.3, color='grey', linestyle=':', alpha=0.3, zorder=1)

    # labels
    ax.set_xlabel(f"Pearson correlation with {TARGET}", fontsize=12)
    ax.set_title(
        "Feature importance ranked by correlation with entropy\n"
        f"({len(plot_df)} features, sorted by |r|, "
        f"n = {plot_df.iloc[0]['n']:,} patches)",
        fontsize=13,
        pad=15,
    )

    # x axis limits
    max_abs = plot_df['abs_r'].max()
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)

    # legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor='steelblue',
                      edgecolor='black', linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1,
                      facecolor='indianred',
                      edgecolor='black', linewidth=0.5),
    ]
    legend_labels = [
        'Positive (feature ↑, entropy ↑)',
        'Negative (feature ↑, entropy ↓)',
    ]
    ax.legend(legend_handles, legend_labels,
              loc='lower right', fontsize=10,
              frameon=True, framealpha=0.9)

    plt.tight_layout()
    output_path = FIGURES_DIR / "feature_importance_correlation.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def print_ranking(corr_df):
    """Print the ranking to terminal."""
    print(f"\n{'=' * 60}")
    print(f"FEATURE IMPORTANCE RANKING")
    print(f"{'=' * 60}\n")
    print(f"  {'Rank':<6} {'Feature':<25} {'r':<10} {'|r|':<8} {'n':<8}")
    print(f"  {'-' * 6} {'-' * 25} {'-' * 10} {'-' * 8} {'-' * 8}")
    for rank, row in enumerate(corr_df.itertuples(), 1):
        direction = '+' if row.r > 0 else '−'
        print(f"  {rank:<6} {row.feature:<25} "
              f"{direction}{abs(row.r):.3f}   "
              f"{row.abs_r:<8.3f} {row.n:<8,}")


def interpret_correlations(corr_df):
    """Print interpretation of correlation strengths."""
    print(f"\n{'=' * 60}")
    print(f"INTERPRETATION")
    print(f"{'=' * 60}\n")

    strong = corr_df[corr_df['abs_r'] >= 0.5]
    moderate = corr_df[(corr_df['abs_r'] >= 0.3) & (corr_df['abs_r'] < 0.5)]
    weak = corr_df[(corr_df['abs_r'] >= 0.1) & (corr_df['abs_r'] < 0.3)]
    negligible = corr_df[corr_df['abs_r'] < 0.1]

    print(f"  Strong predictors (|r| ≥ 0.5):    {len(strong)} features")
    for _, row in strong.iterrows():
        sign = '+' if row['r'] > 0 else '−'
        print(f"    {row['feature']:<25} r = {sign}{abs(row['r']):.3f}")

    print(f"\n  Moderate predictors (0.3 ≤ |r| < 0.5): {len(moderate)} features")
    for _, row in moderate.iterrows():
        sign = '+' if row['r'] > 0 else '−'
        print(f"    {row['feature']:<25} r = {sign}{abs(row['r']):.3f}")

    print(f"\n  Weak predictors (0.1 ≤ |r| < 0.3):   {len(weak)} features")
    for _, row in weak.iterrows():
        sign = '+' if row['r'] > 0 else '−'
        print(f"    {row['feature']:<25} r = {sign}{abs(row['r']):.3f}")

    print(f"\n  Negligible (|r| < 0.1):              {len(negligible)} features")
    for _, row in negligible.iterrows():
        sign = '+' if row['r'] > 0 else '−'
        print(f"    {row['feature']:<25} r = {sign}{abs(row['r']):.3f}")


def main():
    csv_path = PROCESSED_DIR / "patch_training_data_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Run 13_merge_features.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patches from {csv_path.name}\n")

    # compute correlations
    corr_df = compute_correlations(df)

    # print ranking
    print_ranking(corr_df)

    # interpretation
    interpret_correlations(corr_df)

    # plot
    plot_correlation_ranking(corr_df)


if __name__ == "__main__":
    main()