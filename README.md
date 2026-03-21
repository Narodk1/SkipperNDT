# SkipperNDT — Pipeline Detection System

> Magnetic anomaly detection and width estimation for buried pipelines using drone survey data.

---

## Project Overview

This project was developed as part of an NDT (Non-Destructive Testing) challenge. A drone flies over a field and records magnetic signals. The goal is to detect buried pipelines and measure the width of their magnetic influence zone.

The dataset contains **2833 simulated NPZ files** + **102 real terrain files**, each representing a 4-channel magnetic image captured by the drone.

---

## Tasks

| Task | Description | Owner |
|------|-------------|-------|
| TASK1 | Binary classification — pipe presence detection | Teammate |
| **TASK2** | Regression — magnetic influence zone width estimation | **Narodk1** |
| TASK3 | Advanced classification with optimized threshold | Teammate |

---

## TASK 2 — Magnetic Width Estimation

**Owner: NarodK1**

### Objective

Predict the width (in metres) of the magnetic influence zone of a buried pipeline from a 4-channel magnetic image.

**Target: MAE < 1.0m**

---

### Approach 1 — CNN Regressor (abandoned)

The first approach was a custom CNN trained to predict the width directly from the image.

**Architecture:**
- 4 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
- Adaptive average pooling
- Fully connected head → single output (width in metres)
- Loss: MSELoss → HuberLoss
- Optimizer: AdamW with weight decay

**Why it failed:**

After several iterations and improvements (ResNet with skip connections, log-transform on labels, CosineAnnealingLR, data augmentation), the best result was:

```
MAE  : 14.25m   ✗ objective not met
RMSE : 22.06m
R²   : 0.60
```

The fundamental problem was **not enough data** — only ~1190 valid training samples for a regression task spanning 2m to 154m. The model was memorizing instead of learning generalizable features.

> Key insight: A CNN needs to "learn" the relationship between pixel patterns and width. With so few samples and such a wide range of values, it simply could not generalize.

---

### Approach 2 — Geometric Method (final solution)

Instead of learning the width statistically, we measure it **directly from the physics of the magnetic signal**.

**Key observation:**

By visualizing the 4 channels of the NPZ files, channel 2 showed the clearest signal — a thin red line marking the exact pipe axis, with the magnetic influence spreading outward on both sides.

```
Small pipe profile:        Large pipe profile:
       ▲                       ▲           ▲
      /|\                     / \         / \
_____/ | \_____           ___/   \_______/   \___

→ 1 Gaussian peak          → 2 peaks with a valley
→ measure FWHM             → measure outer edges
```

**Pipeline:**

```
1. Load channel 2 (clearest signal)
         ↓
2. PCA on brightest pixels (top 15%)
   → finds pipe orientation angle automatically
         ↓
3. Extract perpendicular profile
   → cut across the pipe at 90°
         ↓
4. Apply Gaussian smoothing (sigma=5)
   → reduce noise
         ↓
5. Detect oscillating profiles
   → increase sigma to 15 if needed
         ↓
6. Measure width at 10% threshold
   → captures outer edges of both peaks
         ↓
7. Repeat on 5 positions along the pipe
   → take median for robustness
         ↓
8. × 0.20 m/pixel = width in metres
   (resolution confirmed by supervisor)
         ↓
9. Apply -2m bias correction
   → systematic overestimation observed
```

**Why PCA?**

The pipe can be oriented at any angle in the image. Measuring horizontally would give wrong results for a diagonal pipe. PCA automatically finds the dominant direction of the bright pixels = the pipe axis. We then cut perpendicularly to get the true width.

**Why 10% threshold instead of 50% (FWHM)?**

For large pipes (> 50m), the signal forms two distinct peaks. The classic FWHM at 50% only captures one peak and misses half the width. The 10% threshold captures the outer edges of both peaks correctly.

**Resolution (0.20 m/pixel):**

Empirically derived by comparing pixel measurements to known widths across 15 samples. Confirmed by project supervisor. All simulated data uses the same drone altitude and sensor configuration.

---

### Results

**Simulated data (1700 samples):**

| Version | Method | MAE |
|---------|--------|-----|
| V1 | CNN baseline | 14.25m |
| V2 | Geometric PCA + FWHM 50% | 5.49m |
| V3 | Geometric PCA + 10% threshold + oscillation detection | **4.59m** |

**Real terrain data (51 samples with pipe):**

```
MAE          : 2.01m
Median error : 0.80m   ← more than half of predictions under 1m
< 2m error   : 74.5% of cases
< 5m error   : 84.3% of cases
```

> The median error of 0.80m on real terrain data shows the geometric approach generalizes well to real-world conditions, unlike the CNN which overfit on simulated data.

---

### MAE by category (simulated data)

| Category | MAE |
|----------|-----|
| missed | 1.57m |
| offset | 1.06m |
| perfect | 6.95m |
| straight | 4.42m |
| curved | 5.25m |

The higher MAE on `perfect` samples is due to very large pipes (up to 154m) where the double-peak profile is harder to measure precisely.

---

## Project Structure

```
SkipperNDT/
main/
├── TASK1/
├── TASK2/
├── TASK3/
├── requirements.txt
└── test_results_task3.json
├── TASK1/
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── TASK2/
│   └── map_width_geometric_v3.py
└── TASK3/
    ├── train.py
    ├── inference.py
    └── evaluate.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- torch
- torchvision

---

## Usage — TASK 2

```bash
python TASK2/map_width_geometric_v3.py
```

Edit the paths at the top of the file:

```python
DATA_DIR  = '/path/to/Training_database_float16'
CSV_PATH  = '/path/to/pipe_detection_label.csv'
```

---

## Key Technical Decisions

| Decision | Reason |
|----------|--------|
| Drop CNN for geometric method | Only 1190 training samples — not enough for regression over 2-154m range |
| Use channel 2 | Clearest signal — thin line marks exact pipe center |
| PCA for orientation | Pipe can be at any angle — PCA finds it automatically |
| 10% threshold | Works for both single-peak (small pipes) and double-peak (large pipes) |
| 5 profiles + median | Robust against noisy or NaN zones along the pipe |
| 0.20 m/pixel | Confirmed by supervisor — constant across all simulated files |
