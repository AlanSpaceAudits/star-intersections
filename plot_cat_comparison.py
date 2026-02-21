#!/usr/bin/env python3
"""
Central Angle Theorem — "What If" comparison plots.

For each peak, shows two scenarios side-by-side:
  Left:  "If Globe Correct" — hypothetical where ΔGE ≈ 0 and |ΔFE| ≈ drop
  Right: "Actual Measurement" — real data where |ΔFE| is small and |ΔGE| ≈ |ΔFE| + drop

This is a visual comparison only. The hypothetical is not real data.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CSV_PATH = Path(__file__).resolve().parent / "data" / "intersections.csv"
OUTPUT_DIR = Path(__file__).parent / "plots" / "cat_individuals_2"
DPI = 150

FE_COLOR = "#a0e632"
FE_COLOR_DARK = "#4a8a0e"
GE_COLOR = "#ff1493"
GE_COLOR_DARK = "#cc1e8a"
BG_COLOR = "#fafafa"
HYPO_ALPHA = 0.55
ACTUAL_ALPHA = 0.85

R_EARTH_M = 6_371_000


def deg2rad(d: float) -> float:
    return d * np.pi / 180.0


def distance_from_drop(drop_dd: float) -> float:
    return 2.0 * R_EARTH_M * deg2rad(drop_dd)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["dist_m"] = 2.0 * R_EARTH_M * deg2rad(df["GE_terrestrial_drop_dd"])
    return df


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").lower()


def plot_peak(peak_df: pd.DataFrame, peak_name: str):
    n = len(peak_df)
    stars = peak_df["Star"].values
    fe_abs_actual = peak_df["ΔFE_intersection_dd"].abs().values
    ge_abs_actual = peak_df["ΔGE_intersection__dd"].abs().values
    drops = peak_df["GE_terrestrial_drop_dd"].values

    # Hypothetical "globe correct": ΔGE = 0, |ΔFE| = drop
    fe_abs_hypo = drops.copy()
    ge_abs_hypo = np.zeros(n)

    # Distance for title
    avg_drop = drops.mean()
    avg_dist_m = distance_from_drop(avg_drop)
    title = f"{peak_name}  (~{avg_dist_m/1000:.0f} km)"

    fig, ax = plt.subplots(figsize=(max(8, 4 + n * 5.5), 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Layout: for each star, two groups of paired bars
    # Group spacing
    group_width = 1.8
    x_stars = np.arange(n) * group_width
    bar_w = 0.28
    gap = 0.08

    # Positions within each star group
    # Hypothetical pair: FE, GE
    # Actual pair: FE, GE
    # With a gap between the two pairs
    pair_gap = 0.22
    x_hypo_fe = x_stars - pair_gap / 2 - bar_w - gap / 2
    x_hypo_ge = x_stars - pair_gap / 2 + gap / 2
    x_act_fe = x_stars + pair_gap / 2 + gap / 2
    x_act_ge = x_stars + pair_gap / 2 + bar_w + gap + gap / 2

    max_val = max(fe_abs_hypo.max(), ge_abs_actual.max(),
                  fe_abs_actual.max(), drops.max())
    headroom = max_val * 0.55
    ax.set_ylim(0, max_val + headroom)
    text_gap = max_val * 0.02

    for i in range(n):
        drop_dd = drops[i]
        dist_m = distance_from_drop(drop_dd)
        drop_m = dist_m * deg2rad(drop_dd)

        # === HYPOTHETICAL (left group) ===
        # FE bar (tall — off by drop)
        ax.bar(x_hypo_fe[i], fe_abs_hypo[i], bar_w, color=FE_COLOR,
               alpha=HYPO_ALPHA, edgecolor="white", linewidth=0.5,
               hatch="//")
        # GE bar (zero)
        ax.bar(x_hypo_ge[i], ge_abs_hypo[i] + 0.001, bar_w, color=GE_COLOR,
               alpha=HYPO_ALPHA, edgecolor="white", linewidth=0.5,
               hatch="//")

        # Labels
        ax.text(x_hypo_fe[i], fe_abs_hypo[i] + text_gap,
                f"|ΔFE|\n{fe_abs_hypo[i]:.2f}°",
                ha="center", va="bottom", fontsize=7.5, color=FE_COLOR_DARK,
                fontweight="bold")
        ax.text(x_hypo_ge[i], text_gap * 2,
                f"|ΔGE|\n≈ 0°",
                ha="center", va="bottom", fontsize=7.5, color=GE_COLOR_DARK,
                fontweight="bold")

        # "If Globe Correct" label
        mid_hypo = (x_hypo_fe[i] + x_hypo_ge[i]) / 2
        label_y = max_val + headroom * 0.38
        ax.text(mid_hypo, label_y, "If Globe\nCorrect",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#555555", fontstyle="italic")

        # === ACTUAL (right group) ===
        # FE bar (small)
        ax.bar(x_act_fe[i], fe_abs_actual[i], bar_w, color=FE_COLOR,
               alpha=ACTUAL_ALPHA, edgecolor="white", linewidth=0.5)
        # GE bar (large)
        ax.bar(x_act_ge[i], ge_abs_actual[i], bar_w, color=GE_COLOR,
               alpha=ACTUAL_ALPHA, edgecolor="white", linewidth=0.5)

        # Labels
        fe_m = dist_m * deg2rad(fe_abs_actual[i])
        ge_m = dist_m * deg2rad(ge_abs_actual[i])
        ax.text(x_act_fe[i], fe_abs_actual[i] + text_gap,
                f"|ΔFE|\n{fe_abs_actual[i]:.3f}°\n({fe_m:.0f} m)",
                ha="center", va="bottom", fontsize=7.5, color=FE_COLOR_DARK,
                fontweight="bold")
        ax.text(x_act_ge[i], ge_abs_actual[i] + text_gap,
                f"|ΔGE|\n{ge_abs_actual[i]:.3f}°\n({ge_m:.0f} m)",
                ha="center", va="bottom", fontsize=7.5, color=GE_COLOR_DARK,
                fontweight="bold")

        # "Actual Measurement" label
        mid_act = (x_act_fe[i] + x_act_ge[i]) / 2
        ax.text(mid_act, label_y, "Actual\nMeasurement",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#333333")

        # Drop annotation between the two groups
        mid_x = x_stars[i]
        drop_y = max_val + headroom * 0.12
        ax.text(mid_x, drop_y,
                f"drop: {drop_dd:.2f}° ({drop_m:.0f} m)",
                ha="center", va="center", fontsize=7.5, color="#ff8c00",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3e0",
                          edgecolor="#ff8c00", alpha=0.9))

        # Vertical separator between stars
        if i < n - 1:
            sep_x = (x_stars[i] + x_stars[i + 1]) / 2
            ax.axvline(sep_x, color="#dddddd", lw=1, ls="--", zorder=0)

    ax.set_xticks(x_stars)
    ax.set_xticklabels(stars, fontsize=10, fontweight="bold")
    ax.set_ylabel("Absolute Residual |Δ| (°)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    legend_handles = [
        Patch(facecolor=FE_COLOR, edgecolor="white", alpha=ACTUAL_ALPHA,
              label="|ΔFE| (FE residual)"),
        Patch(facecolor=GE_COLOR, edgecolor="white", alpha=ACTUAL_ALPHA,
              label="|ΔGE| (GE residual)"),
        Patch(facecolor="#cccccc", edgecolor="white", hatch="//",
              label="Hypothetical (not real data)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    df = load_data(csv_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    peaks = df.groupby("Peak", sort=False)
    for peak_name, peak_df in peaks:
        fig = plot_peak(peak_df, peak_name)
        fname = f"{safe_filename(peak_name)}.png"
        fig.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        stars = ", ".join(peak_df["Star"].values)
        print(f"  {fname}  ({stars})")

    print(f"\nSaved {len(peaks)} plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
