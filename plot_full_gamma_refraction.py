import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "data" / "intersections.csv"

OUT = Path(__file__).resolve().parent / "plots" / "full_gamma"
OUT.mkdir(parents=True, exist_ok=True)

R_EARTH = 6_371_000  # meters

df = pd.read_csv(CSV_PATH)

# Clean distance column (has commas)
df['dist_m'] = df['Distance (m)'].astype(str).str.replace(',', '').astype(float)

# Compute full central angle from distance
df['gamma_deg'] = np.degrees(df['dist_m'] / R_EARTH)
df['gamma_arcmin'] = df['gamma_deg'] * 60

# Compute actual refraction: Apparent - True
df['star_apparent'] = df['θStar,Obs,Apparent'].astype(float)
df['star_true'] = df['θStar,Obs,True'].astype(float)
df['refr_deg'] = df['star_apparent'] - df['star_true']
df['refr_arcmin'] = df['refr_deg'] * 60

# Convert γ and refraction to meters
df['gamma_meters'] = np.tan(np.radians(df['gamma_deg'])) * df['dist_m']
df['refr_meters'] = np.tan(np.radians(df['refr_deg'])) * df['dist_m']

plt.style.use('dark_background')

COLOR_GAMMA = '#d4a0d4'   # soft lavender for gamma
COLOR_REFR  = '#6fa8dc'   # soft blue for refraction

# Individual peak plots
for i, row in df.iterrows():
    fig, ax = plt.subplots(figsize=(6, 5))

    vals = [row['gamma_arcmin'], row['refr_arcmin']]
    labels = ['Central\nAngle (γ)', 'Astronomical\nRefraction']
    colors = [COLOR_GAMMA, COLOR_REFR]

    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.55)

    for bar, v in zip(bars, vals):
        ypos = max(v, 0) + 0.3
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:.1f}'", ha='center', va='bottom', fontsize=12, color='white')

    ax.set_ylabel('Arcminutes', fontsize=12)
    ax.set_title(f"{row['Peak']}  —  {row['Star']}\n"
                 f"Distance: {row['dist_m']/1000:.1f} km   |   "
                 f"γ Drop: {row['gamma_meters']:.0f} m  |  Astro Lift: {row['refr_meters']:.0f} m",
                 fontsize=10, pad=12)
    ax.grid(True, axis='y', color='#333333', alpha=0.5)

    max_val = max(abs(v) for v in vals)
    if min(vals) < 0:
        ax.set_ylim(min(vals) * 1.3, max_val * 1.25)
    else:
        ax.set_ylim(0, max_val * 1.25)

    fig.tight_layout()

    safe_name = f"{row['Peak']}_{row['Star']}".replace(' ', '_').replace('/', '_')
    fig.savefig(OUT / f"fg_{safe_name}.png", dpi=150)
    plt.close()
    print(f"  saved fg_{safe_name}.png")

# Print summary
print(f"\n{'Peak':<20} {'Star':<15} {'Refr (arcmin)':>14} {'γ (arcmin)':>14} {'Refr/γ':>10}")
print("-" * 78)
for _, row in df.iterrows():
    ratio = row['refr_arcmin'] / row['gamma_arcmin'] if row['gamma_arcmin'] != 0 else 0
    print(f"{row['Peak']:<20} {row['Star']:<15} {row['refr_arcmin']:>14.2f} {row['gamma_arcmin']:>14.2f} {ratio:>10.2f}")

print(f"\nAll graphs saved to {OUT}/")
