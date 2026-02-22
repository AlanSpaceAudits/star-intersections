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

from gaussian import GaussianAnalysis

CSV_PATH = Path(__file__).resolve().parent / "data" / "intersections.csv"
OUTPUT_DIR = Path(__file__).parent / "plots" / "individuals"
DPI = 150

FE_COLOR = "#a0e632"       # bright lime green (predicted)
FE_COLOR_MEAS = "#4a8a0e"  # darker lime (measured)
GE_COLOR = "#ff1493"       # hot pink (predicted)
GE_COLOR_MEAS = "#cc1e8a"  # softer deep pink (measured)
R_EARTH_M = 6_371_000  # mean Earth radius in meters


def deg2rad(d: float) -> float:
    return d * np.pi / 180.0


def distance_from_drop(drop_dd: float) -> float:
    """Estimate observation distance (m) from GE terrestrial drop angle.
    drop ≈ d / (2R) radians  →  d ≈ 2R × drop_rad"""
    return 2.0 * R_EARTH_M * deg2rad(drop_dd)


def sigma_to_meters(sigma_dd: float, dist_m: float) -> float:
    """Convert angular σ (degrees) to linear error (m) at a given distance."""
    return dist_m * deg2rad(sigma_dd)


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").lower()


REF_COLOR = "#e74c3c"  # red reference line


def plot_peak(peak_df: pd.DataFrame, peak_name: str, fe_sigma: float, ge_sigma: float):
    n = len(peak_df)

    stars = peak_df["Star"].values
    fe_pred = peak_df["FE_pred_angle_dd"].values
    ge_pred = peak_df["GE_pred_angle_dd"].values
    fe_delta = peak_df["ΔFE_intersection_dd"].values
    ge_delta = peak_df["ΔGE_intersection__dd"].values

    # Actual star positions: Δ = predicted - measured, so measured = predicted - Δ
    # FE: Star→Peak ♈︎ = FE_pred - ΔFE
    # GE: Star→Obs ♈︎  = GE_pred - ΔGE
    fe_meas = fe_pred - fe_delta
    ge_meas = ge_pred - ge_delta

    # Two side-by-side subplots: FE panel (left), GE panel (right)
    fig_width = max(12, n * 2.5)
    fig, (ax_fe, ax_ge) = plt.subplots(1, 2, figsize=(fig_width, 6), facecolor="#fafafa")
    fig.suptitle(peak_name, fontsize=14, fontweight="bold", y=0.98)

    # Compute shared y-axis: always includes 0, same range for both panels
    fe_sigma_2 = 2.0 * fe_sigma
    ge_sigma_2 = 2.0 * ge_sigma
    all_highs = np.concatenate([
        fe_pred, fe_meas + fe_sigma_2,
        ge_pred, ge_meas + ge_sigma_2,
    ])
    all_lows = np.concatenate([
        fe_pred, fe_meas - fe_sigma_2,
        ge_pred, ge_meas - ge_sigma_2,
    ])
    data_top = all_highs.max()
    data_bottom = min(0, all_lows.min())
    data_span = max(data_top - data_bottom, 0.3)
    y_lo = data_bottom - data_span * 0.05 if data_bottom < 0 else 0
    y_hi = data_top + data_span * 0.70
    text_offset = data_span * 0.04

    for ax, pred, meas, delta, sigma, color, color_meas, model_label, pred_label, meas_label in [
        (ax_fe, fe_pred, fe_meas, fe_delta, fe_sigma, FE_COLOR, FE_COLOR_MEAS,
         "Flat Earth", "FE peak angle", "Star\u2192Peak \u2648\ufe0e"),
        (ax_ge, ge_pred, ge_meas, ge_delta, ge_sigma, GE_COLOR, GE_COLOR_MEAS,
         "Globe Earth", "GE peak angle", "Star\u2192Obs \u2648\ufe0e"),
    ]:
        ax.set_facecolor("#fafafa")
        x = np.arange(n)
        width = 0.3
        sigma_2 = 2.0 * sigma  # error bars show ±2σ

        # Predicted bar (no error bar)
        ax.bar(x - width / 2, pred, width, color=color, alpha=0.8,
               label=pred_label)
        # Star true position bar with 2σ error bars (darker variant)
        ax.bar(x + width / 2, meas, width, color=color_meas, alpha=0.8,
               label=meas_label, yerr=sigma_2, capsize=5,
               error_kw={"lw": 1.2, "capthick": 1.2})

        # Red reference line at predicted value extending to measured bar
        for i in range(n):
            ax.hlines(pred[i],
                      x[i] - width, x[i] + width,
                      colors=REF_COLOR, linewidths=1.8, linestyles="--",
                      zorder=6, label="predicted ref" if i == 0 else "")

        ax.set_ylim(y_lo, y_hi)

        # Annotate delta, ± meters, and pass/fail above the measured bar
        for i in range(n):
            drop_dd = peak_df["GE_terrestrial_drop_dd"].values[i]
            dist_m = distance_from_drop(drop_dd)
            err_m = sigma_to_meters(sigma_2, dist_m)

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

            y_label = max(pred[i], meas[i] + sigma_2) + text_offset
            ax.text(x[i], y_label,
                    f"\u0394={delta[i]:+.4f}\u00b0 (\u00b1{err_m:.0f} m)",
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
                    f"drop = {drop:.4f}\u00b0",
                    ha="center", va="top", fontsize=8, color="black", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(stars, fontsize=9)
        ax.set_ylabel("Angle (\u00b0)")
        ax.set_title(f"{model_label}  (error bars = \u00b12\u03c3)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH

    # Use gaussian.py module for all statistical computations
    ga = GaussianAnalysis.from_csv(csv_path)
    df = ga.df  # full dataframe with dist_m already computed

    fe_sigma = ga.fe.sigma
    ge_sigma = ga.ge.sigma

    print(ga.summary())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot each observation individually (one star per chart)
    count = 0
    for idx, row in df.iterrows():
        obs_df = df.loc[[idx]]
        peak_name = row["Peak"]
        star_name = row["Star"]
        title = f"{peak_name} — {star_name}"
        fig = plot_peak(obs_df, title, fe_sigma, ge_sigma)
        fname = f"{safe_filename(peak_name)}_{safe_filename(star_name)}.png"
        fig.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")
        count += 1

    print(f"\nSaved {count} plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
