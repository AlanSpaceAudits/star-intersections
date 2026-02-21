import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "data" / "intersections.csv"

OUT = Path(__file__).resolve().parent / "plots" / "astro_refraction"
OUT.mkdir(parents=True, exist_ok=True)

R_EARTH = 6_371_000  # meters

df = pd.read_csv(CSV_PATH)

# Clean distance column (has commas)
df['dist_m'] = df['Distance (m)'].astype(str).str.replace(',', '').astype(float)

# Compute half central angle from distance
df['half_gamma_deg'] = np.degrees(df['dist_m'] / (2 * R_EARTH))
df['half_gamma_arcmin'] = df['half_gamma_deg'] * 60

# Compute actual refraction: Apparent - True
df['star_apparent'] = df['θStar,Obs,Apparent'].astype(float)
df['star_true'] = df['θStar,Obs,True'].astype(float)
df['refr_deg'] = df['star_apparent'] - df['star_true']
df['refr_arcmin'] = df['refr_deg'] * 60

# Convert γ/2 and refraction to meters over the arc length
# meters = tan(angle) * distance
df['half_gamma_meters'] = np.tan(np.radians(df['half_gamma_deg'])) * df['dist_m']
df['refr_meters'] = np.tan(np.radians(df['refr_deg'])) * df['dist_m']

plt.style.use('dark_background')

COLOR_GAMMA = '#d4a0d4'   # soft lavender for half-gamma
COLOR_REFR  = '#6fa8dc'   # soft blue for refraction

# ─────────────────────────────────────────────────
# Individual peak plots
# ─────────────────────────────────────────────────
for i, row in df.iterrows():
    fig, ax = plt.subplots(figsize=(6, 5))

    vals = [row['half_gamma_arcmin'], row['refr_arcmin']]
    labels = ['Half Central\nAngle (γ/2)', 'Astronomical\nRefraction']
    colors = [COLOR_GAMMA, COLOR_REFR]

    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.55)

    for bar, v in zip(bars, vals):
        ypos = max(v, 0) + 0.3
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:.1f}'", ha='center', va='bottom', fontsize=12, color='white')

    ax.set_ylabel('Arcminutes', fontsize=12)
    ax.set_title(f"{row['Peak']}  —  {row['Star']}\n"
                 f"Distance: {row['dist_m']/1000:.1f} km   |   "
                 f"γ/2 Drop: {row['half_gamma_meters']:.0f} m  |  Astro Lift: {row['refr_meters']:.0f} m",
                 fontsize=10, pad=12)
    ax.grid(True, axis='y', color='#333333', alpha=0.5)

    # set y limits with headroom
    max_val = max(abs(v) for v in vals)
    if min(vals) < 0:
        ax.set_ylim(min(vals) * 1.3, max_val * 1.25)
    else:
        ax.set_ylim(0, max_val * 1.25)

    fig.tight_layout()

    safe_name = f"{row['Peak']}_{row['Star']}".replace(' ', '_').replace('/', '_')
    fig.savefig(OUT / f"{safe_name}.png", dpi=150)
    plt.close()
    print(f"  saved {safe_name}.png")

# ─────────────────────────────────────────────────
# Scatter overview
# ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(df['half_gamma_arcmin'], df['refr_arcmin'],
           c=COLOR_REFR, s=120, zorder=5, edgecolors='white', linewidths=0.5)

for i, row in df.iterrows():
    ax.annotate(row['Peak'], (row['half_gamma_arcmin'], row['refr_arcmin']),
                textcoords="offset points", xytext=(8, 8), fontsize=9, color='#cccccc')

max_val = max(df['half_gamma_arcmin'].max(), df['refr_arcmin'].max())
min_val = min(0, df['refr_arcmin'].min())
lim = max_val + 2
ax.plot([0, lim], [0, lim], '--', color='white', alpha=0.4, label='1:1 line')

ax.set_xlabel('Half Central Angle γ/2 (arcmin)', fontsize=13)
ax.set_ylabel('Astronomical Refraction (arcmin)', fontsize=13)
ax.set_title('Half Central Angle vs Measured Astronomical Refraction\n(Apparent − True)', fontsize=15, pad=15)
ax.legend(fontsize=11)
ax.grid(True, color='#333333', alpha=0.5)
ax.set_xlim(0, lim)
if min_val < 0:
    ax.set_ylim(min_val - 2, lim)
else:
    ax.set_ylim(0, lim)

fig.tight_layout()
fig.savefig(OUT / "scatter_overview.png", dpi=150)
plt.close()
print("  saved scatter_overview.png")

# Print summary
print(f"\n{'Peak':<20} {'Star':<15} {'Apparent':>10} {'True':>10} {'Refr (arcmin)':>14} {'γ/2 (arcmin)':>14}")
print("-" * 88)
for _, row in df.iterrows():
    print(f"{row['Peak']:<20} {row['Star']:<15} {row['star_apparent']:>10.3f} {row['star_true']:>10.3f} {row['refr_arcmin']:>14.2f} {row['half_gamma_arcmin']:>14.2f}")

print(f"\nAll graphs saved to {OUT}/")
