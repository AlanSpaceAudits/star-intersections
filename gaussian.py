#!/usr/bin/env python3
"""
Gaussian error distribution analysis module.

Fits a Gaussian (normal) distribution to FE and GE residuals using
scipy.stats.norm.fit (Maximum Likelihood Estimation). All statistical
computations live here — plotting scripts import from this module.

Usage:
    from gaussian import GaussianAnalysis
    ga = GaussianAnalysis.from_csv("data/intersections.csv")
    print(ga.summary())
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from scipy import stats


R_EARTH_M = 6_371_000
MAX_DIST_M = 100_000


@dataclass
class GaussianFit:
    """Result of fitting a Gaussian to a set of residuals."""
    mu: float          # mean (systematic bias)
    sigma: float       # standard deviation (measurement uncertainty)
    n: int             # number of observations used in fit
    residuals: np.ndarray  # the raw residual values

    @property
    def abs_mean(self) -> float:
        return abs(self.mu)

    def n_sigma(self, residual: float) -> float:
        """How many sigma a given residual is from the mean."""
        return abs(residual - self.mu) / self.sigma

    def n_sigma_from_zero(self, residual: float) -> float:
        """How many sigma a given residual is from zero."""
        return abs(residual) / self.sigma

    def verdict(self, residual: float) -> tuple[str, str]:
        """Return (label, color) for a given residual's n_sigma."""
        ns = self.n_sigma_from_zero(residual)
        if ns <= 1.0:
            return f"WITHIN 1\u03c3 ({ns:.1f}\u03c3)", "#27ae60"
        elif ns <= 2.0:
            return f"WITHIN 2\u03c3 ({ns:.1f}\u03c3)", "#f39c12"
        else:
            return f"OUTSIDE 2\u03c3 ({ns:.1f}\u03c3)", "#c0392b"

    def shapiro_wilk(self) -> tuple[float, float, str]:
        """Run Shapiro-Wilk normality test. Returns (W, p, verdict)."""
        w, p = stats.shapiro(self.residuals)
        verdict = "Normal (p > 0.05)" if p > 0.05 else "Non-normal (p < 0.05)"
        return w, p, verdict


class GaussianAnalysis:
    """Full Gaussian error analysis for FE and GE models."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip()

        # Estimate distance from GE drop for exclusion filter
        self.df["dist_m"] = (
            2.0 * R_EARTH_M
            * np.deg2rad(self.df["GE_terrestrial_drop_dd"])
        )

        # Split included / excluded
        self.excluded = self.df[self.df["dist_m"] > MAX_DIST_M]
        self.included = self.df[self.df["dist_m"] <= MAX_DIST_M]

        # Fit Gaussians to included observations only
        fe_vals = self.included["ΔFE_intersection_dd"].values
        ge_vals = self.included["ΔGE_intersection__dd"].values

        fe_mu, fe_sigma = stats.norm.fit(fe_vals)
        ge_mu, ge_sigma = stats.norm.fit(ge_vals)

        self.fe = GaussianFit(fe_mu, fe_sigma, len(fe_vals), fe_vals)
        self.ge = GaussianFit(ge_mu, ge_sigma, len(ge_vals), ge_vals)

    @classmethod
    def from_csv(cls, path: str | Path) -> "GaussianAnalysis":
        df = pd.read_csv(path, skipinitialspace=True)
        return cls(df)

    def wilcoxon(self) -> tuple[float, float]:
        """Wilcoxon signed-rank test on |ΔFE| vs |ΔGE|."""
        fe_abs = np.abs(self.fe.residuals)
        ge_abs = np.abs(self.ge.residuals)
        w, p = stats.wilcoxon(fe_abs, ge_abs, alternative="two-sided")
        return w, p

    def win_loss(self) -> tuple[int, int, int]:
        """Count FE wins, GE wins, ties."""
        fe_abs = np.abs(self.fe.residuals)
        ge_abs = np.abs(self.ge.residuals)
        diff = fe_abs - ge_abs
        return int((diff < 0).sum()), int((diff > 0).sum()), int((diff == 0).sum())

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("GAUSSIAN ERROR ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"  Total observations: {len(self.df)}")
        lines.append(f"  Included (<100 km): {len(self.included)}")
        lines.append(f"  Excluded (>100 km): {len(self.excluded)}")

        for label, fit in [("Flat Earth", self.fe), ("Globe Earth", self.ge)]:
            lines.append(f"\n  {label} residuals:")
            lines.append(f"    n     = {fit.n}")
            lines.append(f"    \u03bc     = {fit.mu:.6f}\u00b0")
            lines.append(f"    \u03c3     = {fit.sigma:.6f}\u00b0")
            lines.append(f"    min   = {fit.residuals.min():.6f}\u00b0")
            lines.append(f"    max   = {fit.residuals.max():.6f}\u00b0")
            lines.append(f"    |\u03bc|   = {fit.abs_mean:.6f}\u00b0")
            w, p, v = fit.shapiro_wilk()
            lines.append(f"    Shapiro-Wilk: W={w:.4f}, p={p:.4f} ({v})")

        fe_w, ge_w, ties = self.win_loss()
        lines.append(f"\n  Paired comparison:")
        lines.append(f"    FE closer: {fe_w}/{self.fe.n}")
        lines.append(f"    GE closer: {ge_w}/{self.fe.n}")
        if ties:
            lines.append(f"    Tied:      {ties}/{self.fe.n}")

        try:
            w, p = self.wilcoxon()
            lines.append(f"\n  Wilcoxon signed-rank:")
            lines.append(f"    W={w:.1f}, p={p:.6f}")
        except ValueError as e:
            lines.append(f"\n  Wilcoxon: could not compute ({e})")

        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent / "data" / "intersections.csv"
    ga = GaussianAnalysis.from_csv(csv_path)
    print(ga.summary())

    # Per-observation breakdown
    print("\nPer-observation n_sigma (FE):")
    for _, row in ga.included.iterrows():
        r = row["ΔFE_intersection_dd"]
        ns = ga.fe.n_sigma_from_zero(r)
        label, _ = ga.fe.verdict(r)
        print(f"  {row['Peak']:20s} {row['Star']:20s}  "
              f"\u0394FE={r:+.4f}\u00b0  {ns:.2f}\u03c3  {label}")
