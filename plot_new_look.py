import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "data" / "intersections.csv"

OUT = Path(__file__).resolve().parent / "plots" / "new_look"
OUT.mkdir(parents=True, exist_ok=True)

R_EARTH = 6_371_000

df = pd.read_csv(CSV_PATH)
df['dist_m'] = df['Distance (m)'].astype(str).str.replace(',', '').astype(float)
df['dist_km'] = df['dist_m'] / 1000

df['half_gamma_deg'] = np.degrees(df['dist_m'] / (2 * R_EARTH))
df['half_gamma_arcmin'] = df['half_gamma_deg'] * 60

df['star_apparent'] = df['θStar,Obs,Apparent'].astype(float)
df['star_true'] = df['θStar,Obs,True'].astype(float)
df['refr_deg'] = df['star_apparent'] - df['star_true']
df['refr_arcmin'] = df['refr_deg'] * 60

df['fe_resid'] = df[' ΔFE_intersection_dd'].astype(float)
df['ge_resid'] = df[' ΔGE_intersection__dd'].astype(float)
df['drop'] = df['GE_terrestrial_drop_dd'].astype(float)

df['label'] = df['Peak'] + '\n' + df['Star']
df['short_label'] = df['Peak']

plt.style.use('dark_background')

COLOR_FE = '#77dd77'      # lime green
COLOR_GE = '#ff6b6b'      # soft red/pink
COLOR_GAMMA = '#d4a0d4'   # soft lavender
COLOR_REFR = '#6fa8dc'    # soft blue
COLOR_DROP = '#ffb347'     # orange
COLOR_LINE = '#ffffff'

# ═══════════════════════════════════════════════════════
# 1. GEOMETRIC COLLINEARITY DIAGRAM
# ═══════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left panel: Flat baseline ---
obs_x, obs_y = 0, 0
peak_x, peak_y = 8, 2.5
star_x, star_y = 12, 3.75  # collinear extension

# Ground line
ax1.plot([obs_x - 0.5, peak_x + 0.5], [obs_y, obs_y], '-', color='#888888', linewidth=2)

# Peak triangle
ax1.fill([peak_x - 0.8, peak_x, peak_x + 0.8], [0, peak_y, 0],
         color='#555555', edgecolor='#aaaaaa', linewidth=1.5)

# Line of sight: star -> peak -> observer (collinear)
ax1.plot([obs_x, star_x], [obs_y, star_y], '--', color=COLOR_FE, linewidth=2, alpha=0.8)

# Star
ax1.plot(star_x, star_y, '*', color='#ffdd44', markersize=20, zorder=5)
ax1.annotate('Star (true)', (star_x, star_y), textcoords="offset points",
             xytext=(10, 5), fontsize=11, color='#ffdd44')

# Observer
ax1.plot(obs_x, obs_y, 'o', color=COLOR_FE, markersize=10, zorder=5)
ax1.annotate('Observer', (obs_x, obs_y), textcoords="offset points",
             xytext=(-15, -20), fontsize=11, color=COLOR_FE)

# Peak label
ax1.annotate('Peak', (peak_x, peak_y), textcoords="offset points",
             xytext=(10, 5), fontsize=11, color='#aaaaaa')

# Angle arc at observer
angle = np.degrees(np.arctan2(peak_y, peak_x))
theta_range = np.linspace(0, np.radians(angle), 30)
r_arc = 2.0
ax1.plot(r_arc * np.cos(theta_range), r_arc * np.sin(theta_range),
         '-', color=COLOR_FE, linewidth=1.5)
ax1.annotate('rise/run', (r_arc * 0.7, 0.5), fontsize=9, color=COLOR_FE)

ax1.set_title('Flat Baseline\nStar → Peak → Observer: Collinear', fontsize=14, pad=15)
ax1.set_xlim(-2, 14)
ax1.set_ylim(-1.5, 5.5)
ax1.set_aspect('equal')
ax1.axis('off')

# Big green checkmark
ax1.text(6, -1, '✓ Occlusion explained', fontsize=13, color=COLOR_FE,
         ha='center', fontweight='bold')

# --- Right panel: Curved baseline ---
# Draw curved ground
curve_x = np.linspace(-0.5, peak_x + 0.5, 100)
sag = 0.6  # visual sag to represent curvature
curve_y = -sag * 4 * (curve_x - (peak_x/2))**2 / peak_x**2 + 0  # parabolic sag
curve_y = -sag * np.sin(np.pi * (curve_x - (-0.5)) / (peak_x + 1))

ax2.plot(curve_x, curve_y, '-', color='#888888', linewidth=2)

# Peak sits lower due to curvature
peak_drop = 0.7  # visual γ/2
peak_y_globe = peak_y - peak_drop

# Peak triangle on curved surface
base_y = curve_y[int(len(curve_y) * 0.9)]
ax2.fill([peak_x - 0.8, peak_x, peak_x + 0.8],
         [base_y, peak_y_globe, base_y],
         color='#555555', edgecolor='#aaaaaa', linewidth=1.5)

# Observer on curved surface
obs_y_globe = curve_y[5]
ax2.plot(obs_x, obs_y_globe, 'o', color=COLOR_GE, markersize=10, zorder=5)
ax2.annotate('Observer', (obs_x, obs_y_globe), textcoords="offset points",
             xytext=(-15, -20), fontsize=11, color=COLOR_GE)

# Globe line of sight to lowered peak
slope_globe = (peak_y_globe - obs_y_globe) / (peak_x - obs_x)
star_y_globe = obs_y_globe + slope_globe * (star_x - obs_x)

ax2.plot([obs_x, star_x], [obs_y_globe, star_y_globe], '--',
         color=COLOR_GE, linewidth=2, alpha=0.8)

# Star at TRUE position (same as flat panel)
ax2.plot(star_x, star_y, '*', color='#ffdd44', markersize=20, zorder=5)
ax2.annotate('Star (true)', (star_x, star_y), textcoords="offset points",
             xytext=(10, 5), fontsize=11, color='#ffdd44')

# Show the gap (γ/2)
ax2.annotate('', xy=(star_x - 0.3, star_y_globe), xytext=(star_x - 0.3, star_y),
             arrowprops=dict(arrowstyle='<->', color=COLOR_DROP, lw=2))
ax2.annotate('γ/2', (star_x - 0.8, (star_y + star_y_globe) / 2),
             fontsize=12, color=COLOR_DROP, fontweight='bold')

# Peak label
ax2.annotate('Peak\n(lowered by γ/2)', (peak_x, peak_y_globe), textcoords="offset points",
             xytext=(10, 8), fontsize=10, color='#aaaaaa')

# Globe angle arc
angle_globe = np.degrees(np.arctan2(peak_y_globe - obs_y_globe, peak_x - obs_x))
theta_range2 = np.linspace(0, np.radians(angle_globe), 30)
ax2.plot(obs_x + r_arc * np.cos(theta_range2),
         obs_y_globe + r_arc * np.sin(theta_range2),
         '-', color=COLOR_GE, linewidth=1.5)

ax2.set_title('Curved Baseline\nStar → Peak → Observer: Offset by γ/2', fontsize=14, pad=15)
ax2.set_xlim(-2, 14)
ax2.set_ylim(-1.5, 5.5)
ax2.set_aspect('equal')
ax2.axis('off')

ax2.text(6, -1, '✗ Star still visible — needs refraction', fontsize=13,
         color=COLOR_GE, ha='center', fontweight='bold')

fig.suptitle('The Collinearity Test', fontsize=18, y=0.98, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "1_collinearity_diagram.png", dpi=150)
plt.close()
print("  saved 1_collinearity_diagram.png")


# ═══════════════════════════════════════════════════════
# 2. REFRACTION vs γ/2 SCATTER (all observations)
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 9))

regions = {
    'Colorado': ['Pikes Peak', 'Blodgett Peak', 'Cheyenne Mountain',
                 'Blue Mountain', 'Mount Rosa', 'Green Mountain', 'North Peak'],
    'Idaho': ['Getaway Peak', 'Hounds Tooth', 'Lucky Peak'],
    'PNW': ['Old Blyn', 'Ediz Blyn'],
    'BC': ['Varley SE'],
    'NZ': ['Puhitampi'],
}

region_colors = {
    'Colorado': '#ff6b6b',
    'Idaho': '#6fa8dc',
    'PNW': '#77dd77',
    'BC': '#ffb347',
    'NZ': '#d4a0d4',
}

def get_region(peak):
    for region, peaks in regions.items():
        if peak in peaks:
            return region
    return 'Other'

df['region'] = df['Peak'].apply(get_region)

for region, color in region_colors.items():
    mask = df['region'] == region
    ax.scatter(df.loc[mask, 'half_gamma_arcmin'], df.loc[mask, 'refr_arcmin'],
               c=color, s=140, zorder=5, edgecolors='white', linewidths=0.5,
               label=region)

# 1:1 line
max_val = max(df['half_gamma_arcmin'].max(), df['refr_arcmin'].max()) + 2
ax.plot([0, max_val], [0, max_val], '--', color='white', alpha=0.4, linewidth=1.5,
        label='1:1 (Refr = γ/2)')

# Annotations
for i, row in df.iterrows():
    ax.annotate(f"{row['Peak']}", (row['half_gamma_arcmin'], row['refr_arcmin']),
                textcoords="offset points", xytext=(8, 6), fontsize=8, color='#bbbbbb')

# Shade region above 1:1
ax.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                alpha=0.05, color=COLOR_REFR)
ax.text(5, 24, 'Refraction > γ/2', fontsize=11, color=COLOR_REFR, alpha=0.7,
        fontstyle='italic')
ax.text(25, 8, 'Refraction < γ/2', fontsize=11, color=COLOR_GAMMA, alpha=0.7,
        fontstyle='italic')

ax.set_xlabel('Half Central Angle γ/2 (arcmin)', fontsize=13)
ax.set_ylabel('Measured Astronomical Refraction (arcmin)', fontsize=13)
ax.set_title('All 18 Observations: Refraction vs Half Central Angle', fontsize=16, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, color='#333333', alpha=0.5)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)

fig.tight_layout()
fig.savefig(OUT / "2_refraction_vs_half_gamma_scatter.png", dpi=150)
plt.close()
print("  saved 2_refraction_vs_half_gamma_scatter.png")


# ═══════════════════════════════════════════════════════
# 3. TIMING RESIDUALS
# ═══════════════════════════════════════════════════════
# Convert angular residuals to time using sidereal rate ~15'/min = 0.25°/min
# So time_offset_seconds = (angular_residual_deg / 0.25) * 60
SIDEREAL_RATE = 15 / 60  # degrees per minute -> 0.25 deg/min

df['fe_time_sec'] = (df['fe_resid'].abs() / SIDEREAL_RATE) * 60
df['ge_time_sec'] = (df['ge_resid'].abs() / SIDEREAL_RATE) * 60

fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(df))
width = 0.35

bars_fe = ax.bar(x - width/2, df['fe_time_sec'], width, color=COLOR_FE,
                 edgecolor='white', linewidth=0.5, label='FE timing error')
bars_ge = ax.bar(x + width/2, df['ge_time_sec'], width, color=COLOR_GE,
                 edgecolor='white', linewidth=0.5, label='GE timing error')

# Value labels
for bar in bars_fe:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.0f}s", ha='center', va='bottom', fontsize=7, color=COLOR_FE)

for bar in bars_ge:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.0f}s", ha='center', va='bottom', fontsize=7, color=COLOR_GE)

ax.set_xticks(x)
ax.set_xticklabels([f"{row['Peak']}\n{row['Star']}" for _, row in df.iterrows()],
                    fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Timing Error (seconds)', fontsize=13)
ax.set_title('Occlusion Timing Error by Model\n(Angular residual converted at 15\'/min sidereal rate)',
             fontsize=15, pad=15)
ax.legend(fontsize=12)
ax.grid(True, axis='y', color='#333333', alpha=0.5)

fig.tight_layout()
fig.savefig(OUT / "3_timing_residuals.png", dpi=150)
plt.close()
print("  saved 3_timing_residuals.png")


# ═══════════════════════════════════════════════════════
# 4. |ΔFE| vs |ΔGE| PAIRED BAR CHART
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(df))
width = 0.35

bars_fe = ax.bar(x - width/2, df['fe_resid'].abs(), width, color=COLOR_FE,
                 edgecolor='white', linewidth=0.5, label='|ΔFE| (Flat Earth error)')
bars_ge = ax.bar(x + width/2, df['ge_resid'].abs(), width, color=COLOR_GE,
                 edgecolor='white', linewidth=0.5, label='|ΔGE| (Globe error)')

# Value labels
for bar in bars_fe:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f"{h:.2f}°", ha='center', va='bottom', fontsize=6.5, color=COLOR_FE)

for bar in bars_ge:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f"{h:.2f}°", ha='center', va='bottom', fontsize=6.5, color=COLOR_GE)

ax.set_xticks(x)
ax.set_xticklabels([f"{row['Peak']}\n{row['Star']}" for _, row in df.iterrows()],
                    fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Intersection Error (degrees)', fontsize=13)
ax.set_title('Intersection Residuals: Flat Earth vs Globe\nEvery observation, side by side',
             fontsize=15, pad=15)
ax.legend(fontsize=12)
ax.grid(True, axis='y', color='#333333', alpha=0.5)

# Annotate means
mean_fe = df['fe_resid'].abs().mean()
mean_ge = df['ge_resid'].abs().mean()
ax.axhline(mean_fe, color=COLOR_FE, linestyle=':', alpha=0.6, linewidth=1)
ax.axhline(mean_ge, color=COLOR_GE, linestyle=':', alpha=0.6, linewidth=1)
ax.text(len(df) - 0.5, mean_fe + 0.008, f'FE mean: {mean_fe:.3f}°',
        fontsize=9, color=COLOR_FE, ha='right')
ax.text(len(df) - 0.5, mean_ge + 0.008, f'GE mean: {mean_ge:.3f}°',
        fontsize=9, color=COLOR_GE, ha='right')

fig.tight_layout()
fig.savefig(OUT / "4_fe_vs_ge_residuals.png", dpi=150)
plt.close()
print("  saved 4_fe_vs_ge_residuals.png")


# ═══════════════════════════════════════════════════════
# 5. THE "UNDO" — PIKES PEAK STEP-BY-STEP
# ═══════════════════════════════════════════════════════
# Use Pikes Peak as the example
pp = df[df['Peak'] == 'Pikes Peak'].iloc[0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))

bar_width = 0.55

# Step 1: Flat geometry (works)
vals1 = [pp['FE_pred_angle_dd'], pp['star_apparent']]
labels1 = ['FE\nPrediction', 'Star\nApparent']
colors1 = [COLOR_FE, COLOR_REFR]
bars1 = ax1.bar(labels1, vals1, color=colors1, edgecolor='white', linewidth=0.5, width=bar_width)
for bar, v in zip(bars1, vals1):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02,
             f"{v:.2f}°", ha='center', va='bottom', fontsize=12, color='white')

diff1 = abs(vals1[0] - vals1[1])
ax1.set_title(f'Step 1: Flat Geometry\nΔFE = {pp["fe_resid"]:.2f}°',
              fontsize=13, pad=15)
ax1.set_ylabel('Degrees', fontsize=12)
ax1.grid(True, axis='y', color='#333333', alpha=0.5)
ax1.set_ylim(0, 3.0)
ax1.text(0.5, 0.02, '✓ Close match', transform=ax1.transAxes,
         ha='center', fontsize=12, color=COLOR_FE, fontweight='bold')

# Step 2: Apply curvature (breaks)
vals2 = [pp['GE_pred_angle_dd'], pp['star_apparent']]
labels2 = ['GE\nPrediction', 'Star\nApparent']
colors2 = [COLOR_GE, COLOR_REFR]
bars2 = ax2.bar(labels2, vals2, color=colors2, edgecolor='white', linewidth=0.5, width=bar_width)
for bar, v in zip(bars2, vals2):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
             f"{v:.2f}°", ha='center', va='bottom', fontsize=12, color='white')

# Show the drop
ax2.annotate('', xy=(0, pp['GE_pred_angle_dd']),
             xytext=(0, pp['FE_pred_angle_dd']),
             arrowprops=dict(arrowstyle='->', color=COLOR_DROP, lw=2.5))
ax2.text(-0.35, (pp['FE_pred_angle_dd'] + pp['GE_pred_angle_dd']) / 2,
         f"−{pp['drop']:.2f}°\n(drop)", fontsize=9, color=COLOR_DROP, ha='center')

ax2.set_title(f'Step 2: Apply Curvature\nΔGE = {pp["ge_resid"]:.2f}°',
              fontsize=13, pad=15)
ax2.set_ylabel('Degrees', fontsize=12)
ax2.grid(True, axis='y', color='#333333', alpha=0.5)
ax2.set_ylim(0, 3.0)
ax2.text(0.5, 0.02, '✗ Gap introduced', transform=ax2.transAxes,
         ha='center', fontsize=12, color=COLOR_GE, fontweight='bold')

# Step 3: Add refraction (fixes it back to step 1)
vals3 = [pp['GE_pred_angle_dd'] + pp['drop'], pp['star_apparent']]
labels3 = ['GE + Refr\n(= FE)', 'Star\nApparent']
colors3 = [COLOR_FE, COLOR_REFR]
bars3 = ax3.bar(labels3, vals3, color=colors3, edgecolor='white', linewidth=0.5, width=bar_width)
for bar, v in zip(bars3, vals3):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.02,
             f"{v:.2f}°", ha='center', va='bottom', fontsize=12, color='white')

# Show the refraction addition
ax3.annotate('', xy=(0, pp['GE_pred_angle_dd'] + pp['drop']),
             xytext=(0, pp['GE_pred_angle_dd']),
             arrowprops=dict(arrowstyle='->', color=COLOR_DROP, lw=2.5))
ax3.text(-0.35, pp['GE_pred_angle_dd'] + pp['drop'] / 2,
         f"+{pp['drop']:.2f}°\n(refr)", fontsize=9, color=COLOR_DROP, ha='center')

ax3.set_title(f'Step 3: Add Refraction Back\nReturns to FE prediction',
              fontsize=13, pad=15)
ax3.set_ylabel('Degrees', fontsize=12)
ax3.grid(True, axis='y', color='#333333', alpha=0.5)
ax3.set_ylim(0, 3.0)
ax3.text(0.5, 0.02, '✓ Back where we started', transform=ax3.transAxes,
         ha='center', fontsize=12, color=COLOR_FE, fontweight='bold')

fig.suptitle('Pikes Peak — 39 Aquarii (50.6 km)\nThe Round Trip: Flat → Curved → Refraction → Flat',
             fontsize=16, y=1.02, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / "5_undo_pikes_peak.png", dpi=150, bbox_inches='tight')
plt.close()
print("  saved 5_undo_pikes_peak.png")


# ═══════════════════════════════════════════════════════
# 6. RATIO vs DISTANCE
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

df['ratio_half'] = df['refr_arcmin'] / df['half_gamma_arcmin']

for region, color in region_colors.items():
    mask = df['region'] == region
    ax.scatter(df.loc[mask, 'dist_km'], df.loc[mask, 'ratio_half'],
               c=color, s=140, zorder=5, edgecolors='white', linewidths=0.5,
               label=region)

for i, row in df.iterrows():
    ax.annotate(f"{row['Peak']}", (row['dist_km'], row['ratio_half']),
                textcoords="offset points", xytext=(8, 6), fontsize=8, color='#bbbbbb')

ax.axhline(1.0, color='white', linestyle='--', alpha=0.4, linewidth=1.5,
           label='Refr = γ/2')
ax.fill_between([0, 140], [1, 1], [0, 0], alpha=0.03, color=COLOR_GE)
ax.fill_between([0, 140], [1, 1], [8, 8], alpha=0.03, color=COLOR_FE)

ax.text(135, 0.5, 'Refr < γ/2', fontsize=10, color=COLOR_GE, alpha=0.7,
        ha='right', fontstyle='italic')
ax.text(135, 1.5, 'Refr > γ/2', fontsize=10, color=COLOR_FE, alpha=0.7,
        ha='right', fontstyle='italic')

ax.set_xlabel('Distance (km)', fontsize=13)
ax.set_ylabel('Refr / (γ/2)', fontsize=13)
ax.set_title('Refraction-to-Curvature Ratio vs Distance\nHow the ratio behaves across all baselines',
             fontsize=15, pad=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, color='#333333', alpha=0.5)
ax.set_xlim(0, 140)
ax.set_ylim(0, max(df['ratio_half']) * 1.1)

fig.tight_layout()
fig.savefig(OUT / "6_ratio_vs_distance.png", dpi=150)
plt.close()
print("  saved 6_ratio_vs_distance.png")


# ═══════════════════════════════════════════════════════
# BONUS: DECOMPOSITION OVERVIEW (all obs, one chart)
# Shows ΔFE (lime) + drop (orange) stacked = ΔGE (pink)
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(df))
width = 0.35

# Left stacked: |ΔFE| + drop
bars_fe_stack = ax.bar(x - width/2, df['fe_resid'].abs(), width,
                       color=COLOR_FE, edgecolor='white', linewidth=0.5,
                       label='|ΔFE|')
bars_drop = ax.bar(x - width/2, df['drop'], width,
                   bottom=df['fe_resid'].abs(), color=COLOR_DROP,
                   edgecolor='white', linewidth=0.5, label='+ Drop (γ/2)')

# Right solid: |ΔGE|
bars_ge = ax.bar(x + width/2, df['ge_resid'].abs(), width,
                 color=COLOR_GE, edgecolor='white', linewidth=0.5,
                 label='|ΔGE|')

ax.set_xticks(x)
ax.set_xticklabels([f"{row['Peak']}\n{row['Star']}" for _, row in df.iterrows()],
                    fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Degrees', fontsize=13)
ax.set_title('Central Angle Decomposition: All Observations\n'
             'Left (stacked): |ΔFE| + Drop = Right (solid): |ΔGE|',
             fontsize=15, pad=15)
ax.legend(fontsize=11)
ax.grid(True, axis='y', color='#333333', alpha=0.5)

fig.tight_layout()
fig.savefig(OUT / "7_decomposition_overview.png", dpi=150)
plt.close()
print("  saved 7_decomposition_overview.png")


print(f"\nAll new_look graphs saved to {OUT}/")
