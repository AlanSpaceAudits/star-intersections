# Celestial Theodolite Intersection Analysis

Gaussian error distribution analysis of celestial theodolite observations comparing **Flat Earth (FE)** and **Globe Earth (GE)** model predictions against actual measured star positions.

## What This Does

This project takes theodolite intersection data from star observations at 13 mountain peaks and asks a simple question for each observation: **does the model's predicted angle match the measured angle within the measurement uncertainty?**

Two competing geometric models are tested:

- **Flat Earth (FE)**: Predicts the elevation angle to a star assuming a planar surface with no curvature correction
- **Globe Earth (GE)**: Predicts the elevation angle accounting for Earth's curvature via a "terrestrial drop" correction

For each observation, we have:
- The model's **predicted** elevation angle
- The **measured** elevation angle from the theodolite intersection
- The **residual** (delta) between prediction and measurement

We fit a Gaussian error distribution to the residuals across all observations to quantify the measurement uncertainty, then check whether each prediction falls within the statistical error bars of its measurement.

## Dataset

17 star observations across 13 peaks:

| Peak | Stars | Location |
|------|-------|----------|
| Pikes Peak | 39 Aquarii | Colorado |
| Blodgett Peak | LP Aquarii | Colorado |
| Cheyenne Mountain | Mu Fornacis | Colorado |
| Blue Mountain | HD 32515 | Colorado |
| Mount Rosa | HD 17320 | Colorado |
| Green Mountain | HD 28388 | Colorado |
| North Peak | HD55892 | Colorado |
| Getaway Peak | HD102928 | — |
| Hounds Tooth | HD199828 | — |
| Old Blyn | HD206088, HD207098 | Washington |
| Ediz Spit Blyn | HD187663, 3 Cap | Washington |
| Puhitampi | Baten Kaitos, 53 Cet | — |
| Lucky Peak | HD 76600, HIP 45592 | Idaho |

### CSV Columns

| Column | Description |
|--------|-------------|
| `Peak` | Observation location |
| `Star` | Target star |
| `Date`, `Time` | Observation timestamp |
| `FE_pred_angle_dd` | Flat Earth predicted elevation angle (decimal degrees) |
| `ΔFE_intersection_dd` | Residual: measured minus FE predicted (decimal degrees) |
| `GE_terrestrial_drop_dd` | Globe Earth curvature correction (decimal degrees) |
| `GE_pred_angle_dd` | Globe Earth predicted elevation angle (decimal degrees) |
| `ΔGE_intersection_dd` | Residual: measured minus GE predicted (decimal degrees) |

## The Math

### Step 1: Gaussian Error Model

We model measurement errors as normally distributed. The probability density function of the normal distribution is:

```
                    1              (x - μ)²
f(x) = ────────────────── · exp(- ─────────)
        σ · sqrt(2π)                2σ²
```

Where:
- **μ** (mu) = mean of the residuals — represents systematic bias
- **σ** (sigma) = standard deviation — represents measurement uncertainty
- **x** = an individual residual value

### Step 2: Maximum Likelihood Estimation (MLE)

We collect all residuals from the 17 observations and fit a Gaussian using Maximum Likelihood Estimation via `scipy.stats.norm.fit()`. This computes:

```
        1   n
μ̂  =  ─── Σ  Δᵢ
        n  i=1

            ┌─────────────────────┐
            │  1   n              │
σ̂  = sqrt │ ─── Σ  (Δᵢ - μ̂)²  │
            │  n  i=1             │
            └─────────────────────┘
```

**Important**: MLE uses `n` in the denominator (population standard deviation), not `n-1` (sample standard deviation). With n=17 the difference is small (~3%).

**Results from this dataset:**

| | FE Residuals | GE Residuals |
|---|---|---|
| μ (mean/bias) | 0.1053° | 0.2882° |
| σ (std dev) | 0.1172° | 0.0917° |
| \|μ\| (absolute bias) | 0.1053° | 0.2882° |

### Step 3: Normality Validation (Shapiro-Wilk Test)

Before relying on the Gaussian model, we verify the residuals actually follow a normal distribution using the **Shapiro-Wilk test**:

- **Null hypothesis (H₀)**: The data is drawn from a normal distribution
- **If p > 0.05**: Cannot reject H₀ — data is consistent with normal
- **If p ≤ 0.05**: Reject H₀ — data is not normally distributed

**Results:**

| Model | W statistic | p-value | Verdict |
|-------|-------------|---------|---------|
| FE | 0.8874 | 0.0419 | Non-normal (p < 0.05) |
| GE | 0.9631 | 0.6904 | Normal (p > 0.05) |

The FE residuals fail the normality test (driven by the North Peak outlier at -0.23°), while GE residuals pass comfortably.

### Step 4: Error Bars on Measurements

The error bars on the measured values represent **±1σ** from the fitted Gaussian. Under a normal distribution:

```
±1σ  →  68.3% of measurements fall within this range
±2σ  →  95.4% of measurements fall within this range
±3σ  →  99.7% of measurements fall within this range
```

### Step 5: Angular-to-Linear Conversion (Degrees to Meters)

To make the error bars physically intuitive, we convert from angular units (degrees) to linear distance (meters).

**Estimating observation distance from the terrestrial drop:**

The GE terrestrial drop angle is caused by Earth's curvature over the observation distance. The approximate relationship is:

```
θ_drop ≈ d / (2R)    (in radians)
```

Solving for distance:

```
d ≈ 2R · θ_drop_rad
```

Where R = 6,371,000 m (Earth's mean radius).

**Converting angular error to linear error:**

```
ε_meters = d · σ_rad
```

Where `σ_rad = σ_degrees · π / 180`.

**Example — Pikes Peak (drop = 0.23°):**

```
d = 2 × 6,371,000 × (0.23 × π/180) = 51,154 m ≈ 51 km

FE error: 51,154 × (0.1172 × π/180) = ±105 m
GE error: 51,154 × (0.0917 × π/180) = ±82 m
```

### Step 6: Pass/Fail Determination

For each observation, we compute how many standard deviations separate the predicted value from the measured value:

```
              |θ_predicted - θ_measured|
n_sigma  =  ─────────────────────────────
                        σ
```

The verdict:

| Badge | Condition | Meaning |
|-------|-----------|---------|
| **WITHIN 1σ** (green) | n_σ ≤ 1.0 | Prediction is statistically consistent with the measurement |
| **WITHIN 2σ** (amber) | 1.0 < n_σ ≤ 2.0 | Marginal — prediction is outside typical error but within extended range |
| **OUTSIDE 2σ** (red) | n_σ > 2.0 | Prediction is statistically inconsistent with the measurement |

In a well-calibrated model, you'd expect:
- ~68% of observations within 1σ
- ~95% within 2σ
- Only ~5% outside 2σ

## Output Plots

### Overview Plots (`plots/`)

1. **Residual bar chart** — All 17 observations side-by-side, ΔFE vs ΔGE with 1σ error bars
2. **Gaussian histograms** — Residual distribution for each model with the fitted normal curve overlaid
3. **Predicted angle scatter** — FE and GE predictions across all observations
4. **Violin + swarm plot** — Distribution shape comparison with individual data points

### Individual Peak Plots (`plots/individuals/`)

13 plots, one per peak. Each has two panels:

- **Left panel (lime green)**: FE predicted vs FE measured
- **Right panel (hot pink)**: GE predicted vs GE measured

Each panel shows:
- Light color bar = model's predicted angle
- Dark color bar = measured angle (with ±1σ error bars)
- **Δ** value and **±meters** annotation
- **Drop** value (GE terrestrial curvature correction)
- Color-coded **pass/fail badge** (green/amber/red)

## How to Use

### Requirements

- Python 3.10+
- `matplotlib`, `numpy`, `pandas`, `scipy`

On Arch Linux:
```bash
sudo pacman -S python-matplotlib python-numpy python-pandas python-scipy
```

On other systems:
```bash
pip install matplotlib numpy pandas scipy
```

### Running the Scripts

**Generate overview plots:**
```bash
python3 plot_intersections.py
```

**Generate individual peak plots:**
```bash
python3 plot_individuals.py
```

**Use a different CSV file:**
```bash
python3 plot_individuals.py /path/to/your/data.csv
```

Both scripts save PNGs to the `plots/` directory.

### Jupyter Notebook

```bash
jupyter-lab
```

Open `analysis.ipynb` and run cells with `Shift+Enter`. The notebook includes all overview plots, individual peak plots inline, summary statistics, and the full methodology documentation with LaTeX-rendered equations.

**Open in Google Colab (no install needed):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlanSpaceAudits/star-intersections/blob/main/analysis.ipynb)

## Project Structure

```
star-intersections/
├── README.md
├── analysis.ipynb              # Interactive Jupyter notebook
├── plot_intersections.py       # Overview plots (4 figures)
├── plot_individuals.py         # Individual per-peak plots (13 figures)
├── .gitignore
└── plots/
    ├── 1_residual_bars.png
    ├── 2_gaussian_histograms.png
    ├── 3_predicted_scatter.png
    ├── 4_violin_plot.png
    └── individuals/
        ├── pikes_peak.png
        ├── blodgett_peak.png
        ├── cheyenne_mountain.png
        ├── blue_mountain.png
        ├── mount_rosa.png
        ├── green_mountain.png
        ├── north_peak.png
        ├── getaway_peak.png
        ├── hounds_tooth.png
        ├── old_blyn.png
        ├── ediz_spit_blyn.png
        ├── puhitampi.png
        └── lucky_peak.png
```

## License

This project is provided for educational and research purposes.
