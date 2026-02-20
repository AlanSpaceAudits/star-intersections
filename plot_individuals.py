#!/usr/bin/env python3
"""
Individual per-peak scatter plots — FE vs GE predicted angles with error bars.
Each peak gets its own plot scaled to its own data range.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = Path.home() / "Documents" / "claude_dump" / \
    "Celestial_Theodolite_Calculator_V_v9_1_1 - intersections.csv"
OUTPUT_DIR = Path(__file__).parent / "plots" / "individuals"
DPI = 150

FE_COLOR = "#a0e632"       # bright lime green (predicted)
FE_COLOR_MEAS = "#4a8a0e"  # darker lime (measured)
GE_COLOR = "#ff1493"       # hot pink (predicted)
GE_COLOR_MEAS = "#cc1e8a"  # softer deep pink (measured)
R_EARTH_M = 6_371_000  # mean Earth radius in meters
MAX_DIST_M = 100_000   # exclude observations beyond 100 km


def deg2rad(d: float) -> float:
    return d * np.pi / 180.0


def distance_from_drop(drop_dd: float) -> float:
    """Estimate observation distance (m) from GE terrestrial drop angle.
    drop ≈ d / (2R) radians  →  d ≈ 2R × drop_rad"""
    return 2.0 * R_EARTH_M * deg2rad(drop_dd)


def sigma_to_meters(sigma_dd: float, dist_m: float) -> float:
    """Convert angular 1σ (degrees) to linear error (m) at a given distance."""
    return dist_m * deg2rad(sigma_dd)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    # Estimate observation distance from GE terrestrial drop
    df["dist_m"] = 2.0 * R_EARTH_M * deg2rad(df["GE_terrestrial_drop_dd"])
    # Exclude observations beyond 100 km
    excluded = df[df["dist_m"] > MAX_DIST_M]
    if len(excluded) > 0:
        for _, row in excluded.iterrows():
            print(f"  Excluded: {row['Peak']} / {row['Star']} "
                  f"(~{row['dist_m']/1000:.0f} km > 100 km)")
    df = df[df["dist_m"] <= MAX_DIST_M].reset_index(drop=True)
    return df


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").lower()


def plot_peak(peak_df: pd.DataFrame, peak_name: str, fe_sigma: float, ge_sigma: float):
    n = len(peak_df)

    stars = peak_df["Star"].values
    fe_pred = peak_df["FE_pred_angle_dd"].values
    ge_pred = peak_df["GE_pred_angle_dd"].values
    fe_delta = peak_df["ΔFE_intersection_dd"].values
    ge_delta = peak_df["ΔGE_intersection__dd"].values

    # Measured = predicted + delta
    fe_meas = fe_pred + fe_delta
    ge_meas = ge_pred + ge_delta

    # Two side-by-side subplots: FE panel (left), GE panel (right)
    fig, (ax_fe, ax_ge) = plt.subplots(1, 2, figsize=(12, 6), facecolor="#fafafa")
    fig.suptitle(peak_name, fontsize=14, fontweight="bold", y=0.98)

    for ax, pred, meas, delta, sigma, color, color_meas, model_label in [
        (ax_fe, fe_pred, fe_meas, fe_delta, fe_sigma, FE_COLOR, FE_COLOR_MEAS, "Flat Earth"),
        (ax_ge, ge_pred, ge_meas, ge_delta, ge_sigma, GE_COLOR, GE_COLOR_MEAS, "Globe Earth"),
    ]:
        ax.set_facecolor("#fafafa")
        x = np.arange(n)
        width = 0.3

        # Predicted bar (no error bar)
        ax.bar(x - width / 2, pred, width, color=color, alpha=0.8,
               label=f"{model_label} predicted")
        # Measured bar with error bars (darker variant)
        ax.bar(x + width / 2, meas, width, color=color_meas, alpha=0.8,
               label=f"{model_label} measured", yerr=sigma, capsize=5,
               error_kw={"lw": 1.2, "capthick": 1.2})

        # Compute y-axis limits (error bars are on measured)
        all_vals = np.concatenate([pred, meas + sigma, meas - sigma])
        data_top = all_vals.max()
        data_bottom = all_vals.min()
        data_span = max(data_top - data_bottom, 0.3)
        y_lo = data_bottom - data_span * 0.15
        y_hi = data_top + data_span * 0.70
        ax.set_ylim(max(0, y_lo), y_hi)

        text_offset = data_span * 0.04

        # Annotate delta, ± meters, and pass/fail above the measured bar
        for i in range(n):
            drop_dd = peak_df["GE_terrestrial_drop_dd"].values[i]
            dist_m = distance_from_drop(drop_dd)
            err_m = sigma_to_meters(sigma, dist_m)

            # How many sigma is the prediction from the measurement?
            n_sigma = abs(pred[i] - meas[i]) / sigma
            within_1s = n_sigma <= 1.0
            within_2s = n_sigma <= 2.0

            if within_1s:
                verdict = f"WITHIN 1\u03c3 ({n_sigma:.1f}\u03c3)"
                v_color = "#27ae60"  # green
            elif within_2s:
                verdict = f"WITHIN 2\u03c3 ({n_sigma:.1f}\u03c3)"
                v_color = "#f39c12"  # amber
            else:
                verdict = f"OUTSIDE 2\u03c3 ({n_sigma:.1f}\u03c3)"
                v_color = "#c0392b"  # red

            y_label = max(pred[i], meas[i] + sigma) + text_offset
            ax.text(x[i], y_label,
                    f"Δ={delta[i]:+.2f}° (±{err_m:.0f} m)",
                    ha="center", va="bottom", fontsize=8, color=color_meas, fontweight="bold")

            # Pass/fail badge above the delta label
            y_badge = y_label + data_span * 0.12
            ax.text(x[i], y_badge, verdict,
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=v_color, edgecolor="none",
                              alpha=0.9))

        # Drop annotation centered between bars at GE prediction height
        for i in range(n):
            drop = peak_df["GE_terrestrial_drop_dd"].values[i]
            ax.text(x[i], min(pred[i], meas[i]) - text_offset,
                    f"drop = {drop:.2f}°",
                    ha="center", va="top", fontsize=8, color="black", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(stars, fontsize=9)
        ax.set_ylabel("Angle (°)")
        ax.set_title(model_label, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    df = load_data(csv_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute global 1σ from all residuals for error bars
    fe_sigma = stats.norm.fit(df["ΔFE_intersection_dd"].values)[1]
    ge_sigma = stats.norm.fit(df["ΔGE_intersection__dd"].values)[1]

    peaks = df.groupby("Peak", sort=False)
    for peak_name, peak_df in peaks:
        fig = plot_peak(peak_df, peak_name, fe_sigma, ge_sigma)
        fname = f"{safe_filename(peak_name)}.png"
        fig.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        stars = ", ".join(peak_df["Star"].values)
        print(f"  {fname}  ({stars})")

    print(f"\nSaved {len(peaks)} plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
