#!/usr/bin/env python3
"""
Central Angle Theorem — Individual per-peak decomposition plots.
Hypothetical: intersection differences reversed between FE and GE.

Same exact format as plot_cat_individuals.py but ΔFE and ΔGE columns swapped.
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

FE_COLOR = "#a0e632"       # bright lime green
FE_COLOR_DARK = "#4a8a0e"
GE_COLOR = "#ff1493"       # hot pink
GE_COLOR_DARK = "#cc1e8a"
DROP_COLOR = "#ff8c00"     # dark orange
DROP_COLOR_LIGHT = "#ffb347"
BG_COLOR = "#fafafa"

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
    # SWAPPED: FE reads from GE column, GE reads from FE column
    fe_delta = peak_df["ΔGE_intersection__dd"].values
    ge_delta = peak_df["ΔFE_intersection_dd"].values
    drops = peak_df["GE_terrestrial_drop_dd"].values

    fe_abs = np.abs(fe_delta)
    ge_abs = np.abs(ge_delta)

    # ΔGE = ΔFE + drop (algebraic identity)
    # So |ΔGE| should ≈ |ΔFE + drop| = |ΔFE| + drop when both ΔFE and drop are positive
    # The stacked bar shows ΔFE + drop; the solid bar shows ΔGE
    stacked_total = fe_delta + drops  # this should ≈ ge_delta

    # Distance for title
    avg_drop = drops.mean()
    avg_dist_m = distance_from_drop(avg_drop)
    title = f"{peak_name}  (~{avg_dist_m/1000:.0f} km)"

    fig, ax = plt.subplots(figsize=(max(7, 4 + n * 3.0), 6.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(n)
    width = 0.32
    gap = 0.04  # small gap between stacked and GE bar

    max_val = max(ge_abs.max(), np.abs(stacked_total).max())
    headroom = max_val * 0.60
    ax.set_ylim(0, max_val + headroom)
    text_gap = max_val * 0.015

    for i in range(n):
        x_left = x[i] - width / 2 - gap / 2   # stacked bar center
        x_right = x[i] + width / 2 + gap / 2   # GE bar center
        drop_dd = drops[i]
        dist_m = distance_from_drop(drop_dd)
        drop_m = dist_m * deg2rad(drop_dd)

        # --- Left: stacked bar (|ΔFE| bottom + drop top) ---
        ax.bar(x_left, fe_abs[i], width, color=FE_COLOR, alpha=0.85,
               edgecolor="white", linewidth=0.5)
        ax.bar(x_left, drop_dd, width, bottom=fe_abs[i], color=DROP_COLOR, alpha=0.85,
               edgecolor="white", linewidth=0.5)

        # --- Right: solid |ΔGE| bar ---
        ax.bar(x_right, ge_abs[i], width, color=GE_COLOR, alpha=0.85,
               edgecolor="white", linewidth=0.5)

        # --- Annotations ---
        # |ΔFE| label on the FE portion
        if fe_abs[i] > max_val * 0.04:
            ax.text(x_left, fe_abs[i] / 2,
                    f"|ΔFE|\n{fe_abs[i]:.3f}°",
                    ha="center", va="center", fontsize=7, color=FE_COLOR_DARK,
                    fontweight="bold")

        # Drop label on the drop portion
        ax.text(x_left, fe_abs[i] + drop_dd / 2,
                f"drop\n{drop_dd:.2f}°\n({drop_m:.0f} m)",
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")

        # Stacked total above left bar
        stacked_h = fe_abs[i] + drop_dd
        ax.text(x_left, stacked_h + text_gap,
                f"ΔFE+drop\n= {stacked_h:.3f}°",
                ha="center", va="bottom", fontsize=7.5, color="black",
                fontweight="bold")

        # |ΔGE| label above right bar
        ge_err_m = dist_m * deg2rad(ge_abs[i])
        ax.text(x_right, ge_abs[i] + text_gap,
                f"|ΔGE|\n= {ge_abs[i]:.3f}°",
                ha="center", va="bottom", fontsize=7.5, color=GE_COLOR_DARK,
                fontweight="bold")

        # Difference annotation (how close stacked is to ΔGE)
        residual = ge_abs[i] - stacked_h
        taller = max(stacked_h, ge_abs[i])
        badge_y = taller + headroom * 0.45
        if abs(residual) < 0.0005:
            badge_text = "Exact match"
        else:
            badge_text = f"diff: {residual:+.3f}°"
        ax.text(x[i], badge_y, badge_text,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8",
                          edgecolor="#999999", alpha=0.9))

        # Connecting line showing the comparison
        line_y = taller + headroom * 0.32
        ax.plot([x_left, x_right], [line_y, line_y],
                color="#999999", lw=1, ls="--")
        ax.plot([x_left, x_left], [stacked_h, line_y],
                color="#999999", lw=0.8, ls=":")
        ax.plot([x_right, x_right], [ge_abs[i], line_y],
                color="#999999", lw=0.8, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels(stars, fontsize=10, fontweight="bold")
    ax.set_ylabel("Angle (°)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Legend
    legend_handles = [
        Patch(facecolor=FE_COLOR, edgecolor="white", label="|ΔFE| (FE residual)"),
        Patch(facecolor=DROP_COLOR, edgecolor="white", label="GE terrestrial drop"),
        Patch(facecolor=GE_COLOR, edgecolor="white", label="|ΔGE| (GE residual)"),
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
