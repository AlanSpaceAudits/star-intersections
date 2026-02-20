#!/usr/bin/env python3
"""
Star Intersection Analysis — Gaussian Error Distribution Plots

Reads celestial theodolite intersection data and produces four plots:
1. Residual comparison bar chart (ΔFE vs ΔGE per observation)
2. Histogram + Gaussian overlay for each model's residuals
3. Predicted angle vs observation scatter with error bars
4. Box/violin plot of residual distributions
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH = Path.home() / "Documents" / "claude_dump" / \
    "Celestial_Theodolite_Calculator_V_v9_1_1 - intersections.csv"
OUTPUT_DIR = Path(__file__).parent / "plots"
DPI = 150
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 7)

FE_COLOR = "#a0e632"   # bright lime green
GE_COLOR = "#ff1493"   # hot pink
BG_COLOR = "#fafafa"

R_EARTH_M = 6_371_000  # mean Earth radius in meters
MAX_DIST_M = 100_000   # exclude observations beyond 100 km


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()
    # Estimate observation distance from GE terrestrial drop
    df["dist_m"] = 2.0 * R_EARTH_M * np.deg2rad(df["GE_terrestrial_drop_dd"])
    # Exclude observations beyond 100 km
    excluded = df[df["dist_m"] > MAX_DIST_M]
    if len(excluded) > 0:
        for _, row in excluded.iterrows():
            print(f"  Excluded: {row['Peak']} / {row['Star']} "
                  f"(~{row['dist_m']/1000:.0f} km > 100 km)")
    df = df[df["dist_m"] <= MAX_DIST_M].reset_index(drop=True)
    # Short label for each observation
    df["label"] = df["Peak"] + "\n" + df["Star"]
    return df


def gaussian_params(values: np.ndarray) -> tuple[float, float]:
    """Fit a Gaussian (mean, std) to the data."""
    mu, sigma = stats.norm.fit(values)
    return mu, sigma


# ── Plot 1: Side-by-side residual bar chart with 1σ error bars ─────────
def plot_residual_bars(df: pd.DataFrame, ax: plt.Axes):
    n = len(df)
    x = np.arange(n)
    width = 0.38

    fe_vals = df["ΔFE_intersection_dd"].values
    ge_vals = df["ΔGE_intersection__dd"].values

    fe_mu, fe_sigma = gaussian_params(fe_vals)
    ge_mu, ge_sigma = gaussian_params(ge_vals)

    ax.bar(x - width / 2, fe_vals, width, label=f"ΔFE (μ={fe_mu:.3f}°, σ={fe_sigma:.3f}°)",
           color=FE_COLOR, alpha=0.85, yerr=fe_sigma, capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + width / 2, ge_vals, width, label=f"ΔGE (μ={ge_mu:.3f}°, σ={ge_sigma:.3f}°)",
           color=GE_COLOR, alpha=0.85, yerr=ge_sigma, capsize=3, error_kw={"lw": 0.8})

    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.axhline(fe_mu, color=FE_COLOR, lw=0.8, ls=":")
    ax.axhline(ge_mu, color=GE_COLOR, lw=0.8, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Δ Intersection (°)")
    ax.set_title("Residuals per Observation — FE vs GE Model")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ── Plot 2: Histogram + Gaussian overlay ───────────────────────────────
def plot_histograms(df: pd.DataFrame, axes: tuple[plt.Axes, plt.Axes]):
    for ax, col, color, model in [
        (axes[0], "ΔFE_intersection_dd", FE_COLOR, "Flat Earth"),
        (axes[1], "ΔGE_intersection__dd", GE_COLOR, "Globe Earth"),
    ]:
        vals = df[col].values
        mu, sigma = gaussian_params(vals)

        # Histogram
        n_bins = max(6, len(vals) // 2)
        counts, bins, patches = ax.hist(vals, bins=n_bins, density=True,
                                         alpha=0.6, color=color, edgecolor="white")

        # Gaussian curve
        x_range = np.linspace(vals.min() - 2 * sigma, vals.max() + 2 * sigma, 200)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                color=color, lw=2, label=f"N({mu:.3f}, {sigma:.3f}²)")

        # Vertical lines for mean ± σ
        ax.axvline(mu, color="black", lw=1, ls="--", label=f"μ = {mu:.3f}°")
        ax.axvspan(mu - sigma, mu + sigma, alpha=0.12, color=color, label="±1σ")

        ax.set_xlabel("Δ Intersection (°)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model} Residual Distribution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)


# ── Plot 3: Predicted angle scatter with error bars ────────────────────
def plot_predicted_scatter(df: pd.DataFrame, ax: plt.Axes):
    fe_pred = df["FE_pred_angle_dd"].values
    ge_pred = df["GE_pred_angle_dd"].values
    fe_delta = df["ΔFE_intersection_dd"].values
    ge_delta = df["ΔGE_intersection__dd"].values

    _, fe_sigma = gaussian_params(fe_delta)
    _, ge_sigma = gaussian_params(ge_delta)

    x = np.arange(len(df))

    ax.errorbar(x - 0.15, fe_pred, yerr=fe_sigma, fmt="o", color=FE_COLOR,
                markersize=5, capsize=3, label=f"FE predicted (±{fe_sigma:.3f}°)")
    ax.errorbar(x + 0.15, ge_pred, yerr=ge_sigma, fmt="s", color=GE_COLOR,
                markersize=5, capsize=3, label=f"GE predicted (±{ge_sigma:.3f}°)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Predicted Angle (°)")
    ax.set_title("FE vs GE Predicted Angles with 1σ Error Bars")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ── Plot 4: Violin + box plot ──────────────────────────────────────────
def plot_violin(df: pd.DataFrame, ax: plt.Axes):
    fe_vals = df["ΔFE_intersection_dd"].values
    ge_vals = df["ΔGE_intersection__dd"].values

    parts = ax.violinplot([fe_vals, ge_vals], positions=[1, 2], showmeans=True,
                          showmedians=True, showextrema=True)

    for i, (pc, color) in enumerate(zip(parts["bodies"], [FE_COLOR, GE_COLOR])):
        pc.set_facecolor(color)
        pc.set_alpha(0.4)

    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")

    # Overlay individual points (jittered)
    for pos, vals, color in [(1, fe_vals, FE_COLOR), (2, ge_vals, GE_COLOR)]:
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
        ax.scatter(np.full_like(vals, pos) + jitter, vals,
                   color=color, alpha=0.7, s=30, zorder=5, edgecolors="white", linewidth=0.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["ΔFE (Flat Earth)", "ΔGE (Globe Earth)"])
    ax.set_ylabel("Δ Intersection (°)")
    ax.set_title("Residual Distribution — Violin + Swarm")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.grid(axis="y", alpha=0.3)

    # Annotate stats
    for pos, vals, color in [(1, fe_vals, FE_COLOR), (2, ge_vals, GE_COLOR)]:
        mu, sigma = gaussian_params(vals)
        ax.text(pos, ax.get_ylim()[1] * 0.92,
                f"μ={mu:.3f}°\nσ={sigma:.3f}°",
                ha="center", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))


# ── Summary statistics ─────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for col, model in [("ΔFE_intersection_dd", "Flat Earth"),
                       ("ΔGE_intersection__dd", "Globe Earth")]:
        vals = df[col].values
        mu, sigma = gaussian_params(vals)
        print(f"\n  {model} residuals (Δ intersection):")
        print(f"    n     = {len(vals)}")
        print(f"    mean  = {mu:.4f}°")
        print(f"    std   = {sigma:.4f}°")
        print(f"    min   = {vals.min():.4f}°")
        print(f"    max   = {vals.max():.4f}°")
        print(f"    |μ|   = {abs(mu):.4f}°  (closer to 0 = less systematic bias)")

    # Shapiro-Wilk normality test
    print(f"\n  Shapiro-Wilk normality test:")
    for col, model in [("ΔFE_intersection_dd", "FE"),
                       ("ΔGE_intersection__dd", "GE")]:
        stat, p = stats.shapiro(df[col].values)
        print(f"    {model}: W={stat:.4f}, p={p:.4f}" +
              (" (normal)" if p > 0.05 else " (non-normal)"))
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    df = load_data(csv_path)
    print(f"Loaded {len(df)} observations from {csv_path.name}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print_summary(df)

    # ── Figure 1: Residual bars ─────────
    fig1, ax1 = plt.subplots(figsize=FIGSIZE_WIDE, facecolor=BG_COLOR)
    ax1.set_facecolor(BG_COLOR)
    plot_residual_bars(df, ax1)
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "1_residual_bars.png", dpi=DPI, bbox_inches="tight")
    print("Saved: plots/1_residual_bars.png")

    # ── Figure 2: Histograms ───────────
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, facecolor=BG_COLOR)
    ax2a.set_facecolor(BG_COLOR)
    ax2b.set_facecolor(BG_COLOR)
    plot_histograms(df, (ax2a, ax2b))
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "2_gaussian_histograms.png", dpi=DPI, bbox_inches="tight")
    print("Saved: plots/2_gaussian_histograms.png")

    # ── Figure 3: Predicted scatter ────
    fig3, ax3 = plt.subplots(figsize=FIGSIZE_WIDE, facecolor=BG_COLOR)
    ax3.set_facecolor(BG_COLOR)
    plot_predicted_scatter(df, ax3)
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "3_predicted_scatter.png", dpi=DPI, bbox_inches="tight")
    print("Saved: plots/3_predicted_scatter.png")

    # ── Figure 4: Violin ──────────────
    fig4, ax4 = plt.subplots(figsize=FIGSIZE_SQUARE, facecolor=BG_COLOR)
    ax4.set_facecolor(BG_COLOR)
    plot_violin(df, ax4)
    fig4.tight_layout()
    fig4.savefig(OUTPUT_DIR / "4_violin_plot.png", dpi=DPI, bbox_inches="tight")
    print("Saved: plots/4_violin_plot.png")

    plt.show()
    print("\nDone. All plots saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
