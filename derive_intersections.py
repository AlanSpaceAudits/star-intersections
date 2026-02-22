#!/usr/bin/env python3
"""
Derive intersections.csv from the canonical data_table.csv.

Computes FE/GE predicted angles and residuals from raw observation data
exported from the Celestial Theodolite Calculator spreadsheet v9.1.1.

Formulas:
    FE_pred  = arctan((Peak_Alt - Obs_Alt) / Distance)   [degrees]
    GE_drop  = Distance / (2 * R_earth)                   [degrees]
    GE_pred  = FE_pred - GE_drop
    ΔFE      = FE_pred - Star->Peak ♈︎
    ΔGE      = GE_pred - Star->Obs ♈︎
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
R_EARTH_M = 6_371_000

# Normalize abbreviated peak names from data_table.csv to canonical names
PEAK_NAME_MAP = {
    "Cheyenne Mtn": "Cheyenne Mountain",
    "Blue Mtn": "Blue Mountain",
    "Green Mtn": "Green Mountain",
    "Getaway Mtn": "Getaway Peak",
    "Old-Blyn Mtn": "Old Blyn",
    "Ediz-Blyn Mtn": "Ediz Blyn",
    "Puhitempi Kuaikatete": "Puhitampi",
    "Varley SE2 Squamish": "Varley SE",
}


def main():
    dt = pd.read_csv(DATA_DIR / "data_table.csv")
    dt.columns = dt.columns.str.strip()

    # Core measurement columns
    peak_alt = dt["Peak Alt (m)"].astype(float)
    obs_alt = dt["Obs Alt (m)"].astype(float)
    distance = dt["Distance (m)"].astype(float)
    star_peak_aries = dt["Star->Peak ♈︎ (dd)"].astype(float)
    star_obs_aries = dt["Star->Obs ♈︎ (dd)"].astype(float)
    star_obs_epsilon = dt["Star->Obs ϵ (dd)"].astype(float)

    # FE predicted angle: arctan(Δalt / distance)
    fe_pred = np.degrees(np.arctan((peak_alt - obs_alt) / distance))

    # GE terrestrial drop: distance / (2R)
    ge_drop = np.degrees(distance / (2 * R_EARTH_M))

    # GE predicted angle
    ge_pred = fe_pred - ge_drop

    # Residuals
    delta_fe = fe_pred - star_peak_aries
    delta_ge = ge_pred - star_obs_aries

    # Format dates from M/D/YYYY to YYYY-MM-DD
    dates = pd.to_datetime(dt["Occlusion Date"].str.strip(), format="%m/%d/%Y")
    date_str = dates.dt.strftime("%Y-%m-%d")

    # Build output DataFrame
    out = pd.DataFrame({
        "Peak": dt["Peak Name"].str.strip(),
        "Star": dt["Star Name"].str.strip(),
        "Date": date_str,
        "Time": dt["Occlusion Time"].str.strip(),
        "FE_pred_angle_dd": fe_pred.round(6),
        "ΔFE_intersection_dd": delta_fe.round(6),
        "GE_terrestrial_drop_dd": ge_drop.round(6),
        "GE_pred_angle_dd": ge_pred.round(6),
        "ΔGE_intersection__dd": delta_ge.round(6),
        "θStar,Obs,Apparent": star_obs_epsilon,
        "Distance (m)": distance.round(4),
        "θStar,Obs,True": star_obs_aries,
    })

    # Normalize peak names
    out["Peak"] = out["Peak"].replace(PEAK_NAME_MAP)

    out_path = DATA_DIR / "intersections.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote {len(out)} observations to {out_path}")
    print(f"  Peaks: {out['Peak'].nunique()}")
    print(f"  Date range: {out['Date'].min()} to {out['Date'].max()}")

    # Quick summary
    max_dist = 100_000
    dist_est = 2.0 * R_EARTH_M * np.deg2rad(ge_drop)
    included = dist_est <= max_dist
    n_inc = included.sum()
    n_exc = (~included).sum()
    print(f"  Included (<100 km): {n_inc}")
    print(f"  Excluded (>100 km): {n_exc}")

    fe_sigma = delta_fe[included].std()
    ge_sigma = delta_ge[included].std()
    print(f"  FE σ = {fe_sigma:.6f}°")
    print(f"  GE σ = {ge_sigma:.6f}°")


if __name__ == "__main__":
    main()
