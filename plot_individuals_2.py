#!/usr/bin/env python3
"""
Individual per-peak absolute error comparison — |ΔFE| vs |ΔGE| per star.

For each peak, a single chart shows the absolute residual of each model
side-by-side, so you can see the size of the errors and the difference
between models at the peak's own scale.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = Path(__file__).resolve().parent / "data" / "intersections.csv"
OUTPUT_DIR = Path(__file__).parent / "plots" / "individuals_2"
DPI = 150

FE_COLOR = "#a0e632"       # bright lime green
FE_COLOR_DARK = "#4a8a0e"  # darker lime
GE_COLOR = "#ff1493"       # hot pink
GE_COLOR_DARK = "#cc1e8a"  # softer deep pink
BG_COLOR = "#fafafa"

R_EARTH_M = 6_371_000
MAX_DIST_M = 100_000


def deg2rad(d: float) -> float:
    return d * np.pi / 180.0


def distance_from_drop(drop_dd: float) -> float:
    return 2.0 * R_EARTH_M * deg2rad(drop_dd)


def sigma_to_meters(sigma_dd: float, dist_m: float) -> float:
    return dist_m * deg2rad(sigma_dd)


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
    fe_abs = peak_df["ΔFE_intersection_dd"].abs().values
    ge_abs = peak_df["ΔGE_intersection__dd"].abs().values
    diff = fe_abs - ge_abs  # negative = FE closer

    # Compute distance from first star's drop for the title
    avg_drop = peak_df["GE_terrestrial_drop_dd"].values.mean()
    avg_dist_m = distance_from_drop(avg_drop)
    title = f"{peak_name}  (~{avg_dist_m/1000:.0f} km)"

    fig, ax = plt.subplots(figsize=(max(6, 3 + n * 2.5), 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(n)
    width = 0.32

    # Bars
    bars_fe = ax.bar(x - width / 2, fe_abs, width, color=FE_COLOR, alpha=0.85,
                     edgecolor="white", linewidth=0.5, label="|ΔFE|")
    bars_ge = ax.bar(x + width / 2, ge_abs, width, color=GE_COLOR, alpha=0.85,
                     edgecolor="white", linewidth=0.5, label="|ΔGE|")

    # Y-axis scaling
    max_val = max(fe_abs.max(), ge_abs.max())
    headroom = max_val * 0.55
    ax.set_ylim(0, max_val + headroom)
    text_gap = max_val * 0.02

    # Annotate each star
    for i in range(n):
        drop_dd = peak_df["GE_terrestrial_drop_dd"].values[i]
        dist_m = distance_from_drop(drop_dd)
        fe_err_m = dist_m * deg2rad(fe_abs[i])
        ge_err_m = dist_m * deg2rad(ge_abs[i])

        # FE value above its bar
        ax.text(x[i] - width / 2, fe_abs[i] + text_gap,
                f"{fe_abs[i]:.3f}°\n({fe_err_m:.0f} m)",
                ha="center", va="bottom", fontsize=8, color=FE_COLOR_DARK,
                fontweight="bold")

        # GE value above its bar
        ax.text(x[i] + width / 2, ge_abs[i] + text_gap,
                f"{ge_abs[i]:.3f}°\n({ge_err_m:.0f} m)",
                ha="center", va="bottom", fontsize=8, color=GE_COLOR_DARK,
                fontweight="bold")

        # Winner badge above both bars
        taller = max(fe_abs[i], ge_abs[i])
        badge_y = taller + headroom * 0.45

        if diff[i] < 0:
            winner = "FE closer"
            badge_color = "#27ae60"
            margin = abs(diff[i])
            margin_m = dist_m * deg2rad(margin)
        elif diff[i] > 0:
            winner = "GE closer"
            badge_color = "#27ae60"
            margin = abs(diff[i])
            margin_m = dist_m * deg2rad(margin)
        else:
            winner = "Tied"
            badge_color = "#7f8c8d"
            margin = 0
            margin_m = 0

        badge_text = f"{winner}\nby {margin:.3f}° ({margin_m:.0f} m)"
        ax.text(x[i], badge_y, badge_text,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color,
                          edgecolor="none", alpha=0.9))

        # Drop annotation centered between FE and GE at FE bar height
        drop_m = dist_m * deg2rad(drop_dd)
        ax.text(x[i], fe_abs[i],
                f"drop {drop_dd:.2f}° ({drop_m:.0f} m)",
                ha="center", va="center", fontsize=7.5, color="black",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(stars, fontsize=10, fontweight="bold")
    ax.set_ylabel("Absolute Residual |Δ| (°)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Make room for drop annotations below x-axis
    ax.tick_params(axis="x", pad=20)

    fig.tight_layout()
    return fig


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    df = load_data(csv_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot all peaks (including excluded ones)
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
