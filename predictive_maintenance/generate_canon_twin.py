"""
predictive_maintenance/generate_canon_twin.py
=============================================
Phase 1 — Canon Aquilion LB Digital Twin (Time-Series Degradation)

Generates a 1500-sample time-series dataset that simulates realistic
hardware degradation over a 1000-day operational lifecycle.

Unlike the static Phase 1 generator, this produces a TRUE Digital Twin:
  - Each sample represents a specific day in the scanner's lifespan
  - Metrics degrade monotonically with operational wear
  - RUL targets are causally tied to both elapsed time AND metric drift

Physics Model:
  - Noise increases with anode roughening / focal spot growth (~0.005 HU/day)
  - Uniformity worsens with detector gain drift / gantry bearing play (~0.002 HU/day)
  - Scaling drifts with mechanical wear in table positioning (~0.001 mm/day)
  - Precision drifts with HV generator capacitor aging (~0.001 HU/day)

All noise values are referenced to a 5.0 mm standard slice thickness.

Usage:
    python generate_canon_twin.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)
N_SAMPLES = 1500
MAX_OPERATIONAL_DAYS = 1000

# Canon Aquilion LB constants
CANON_NOMINAL_DIAMETER = 330.0  # mm
REFERENCE_SLICE_THICKNESS = 5.0  # mm — noise training baseline

# ══════════════════════════════════════════════════════════════════
# Step 1: Simulate Operational Timeline
# ══════════════════════════════════════════════════════════════════

# Each sample = a random day in the scanner's 1000-day lifecycle
days_passed = np.random.randint(0, MAX_OPERATIONAL_DAYS, N_SAMPLES)

# ══════════════════════════════════════════════════════════════════
# Step 2: Time-Series Degradation Models (Digital Twin Physics)
# ══════════════════════════════════════════════════════════════════

# --- Noise (SD) at 5.0 mm reference thickness ---
# Day 0: ~4.5 HU (healthy Canon Aquilion)
# Degrades at ~0.005 HU/day (anode roughening, focal spot growth)
# Measurement jitter: sigma = 0.5 HU
noise_base = 4.5 + (days_passed * 0.005) + np.random.normal(0, 0.5, N_SAMPLES)
noise_base = np.clip(noise_base, 2.0, 15.0)

# --- Uniformity (NUI) ---
# Day 0: ~1.0 HU (well-calibrated)
# Degrades at ~0.002 HU/day (detector gain drift, gantry bearing play)
# Measurement jitter: sigma = 0.2 HU
unif_base = 1.0 + (days_passed * 0.002) + np.random.normal(0, 0.2, N_SAMPLES)
unif_base = np.clip(unif_base, 0.0, 6.0)

# --- Scaling Vertical (mm) ---
# Day 0: ~330.0 mm (nominal Canon phantom)
# Drifts at ~0.001 mm/day in random direction (table rail wear)
# Measurement jitter: sigma = 0.5 mm
drift_direction = np.random.choice([-1, 1], N_SAMPLES)
scaling_base = (CANON_NOMINAL_DIAMETER
                + days_passed * 0.001 * drift_direction
                + np.random.normal(0, 0.5, N_SAMPLES))
scaling_base = np.clip(scaling_base, 325.0, 335.0)

# --- HU Precision (|center - 0|) ---
# Day 0: ~0.0 HU (perfect calibration)
# Drifts at ~0.001 HU/day (HV generator capacitor aging)
# Measurement jitter: sigma = 0.2 HU
precision_base = (0.0
                  + days_passed * 0.001
                  + np.random.normal(0, 0.2, N_SAMPLES))
precision_base = np.clip(precision_base, 0.0, 4.0)

# ══════════════════════════════════════════════════════════════════
# Step 3: RUL Targets (Causally Linked to Time + Degradation)
# ══════════════════════════════════════════════════════════════════

# --- Tube RUL ---
# Lifespan: ~1095 days. Drops with time + accelerated by noise increase.
# Noise increase = anode pitting / focal spot blooming.
rul_tube = 1095 - days_passed - (noise_base - 4.5) * 20
rul_tube += np.random.normal(0, 8, N_SAMPLES)  # Prediction uncertainty
rul_tube = np.clip(rul_tube, 0, 1095).astype(int)

# --- Gantry RUL ---
# Lifespan: ~1825 days (bearings are robust). Drops with uniformity drift.
# Uniformity drift = gantry rotation instability / detector misalignment.
rul_gantry = 1825 - days_passed - (unif_base - 1.0) * 50
rul_gantry += np.random.normal(0, 10, N_SAMPLES)
rul_gantry = np.clip(rul_gantry, 0, 1825).astype(int)

# --- Table RUL ---
# Lifespan: ~1825 days (mechanical, slow wear). Drops with scaling error.
# Scaling error = table rail friction / belt stretch.
rul_table = 1825 - days_passed - np.abs(scaling_base - CANON_NOMINAL_DIAMETER) * 30
rul_table += np.random.normal(0, 12, N_SAMPLES)
rul_table = np.clip(rul_table, 0, 1825).astype(int)

# --- Generator RUL ---
# Lifespan: ~2555 days. Drops with noise (kVp ripple) + precision (cap aging).
rul_gen = 2555 - days_passed - (noise_base - 4.5) * 10 - (precision_base * 20)
rul_gen += np.random.normal(0, 8, N_SAMPLES)
rul_gen = np.clip(rul_gen, 0, 2555).astype(int)

# ══════════════════════════════════════════════════════════════════
# Step 4: Assemble and Save
# ══════════════════════════════════════════════════════════════════

df = pd.DataFrame({
    "Noise_HU": np.round(noise_base, 4),
    "Uniformity_HU": np.round(unif_base, 4),
    "Scaling_V_mm": np.round(scaling_base, 4),
    "HU_Precision": np.round(precision_base, 4),
    "RUL_Tube": rul_tube,
    "RUL_Gantry": rul_gantry,
    "RUL_Table": rul_table,
    "RUL_Generator": rul_gen,
})

output_path = DATA_DIR / "canon_digital_twin.csv"
df.to_csv(output_path, index=False)

# ══════════════════════════════════════════════════════════════════
# Step 5: Summary
# ══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  PHASE 1 COMPLETE -- Canon Digital Twin Dataset")
print("=" * 65)
print(f"  Samples          : {N_SAMPLES}")
print(f"  Operational span : 0-{MAX_OPERATIONAL_DAYS} days")
print(f"  Noise baseline   : {REFERENCE_SLICE_THICKNESS:.1f} mm slice thickness")
print(f"  Output           : {output_path}")
print("-" * 65)
print("  INPUT STATISTICS (5.0mm reference):")
print(f"    Noise_HU      : mean={df['Noise_HU'].mean():.2f}  std={df['Noise_HU'].std():.2f}  [{df['Noise_HU'].min():.2f}, {df['Noise_HU'].max():.2f}]")
print(f"    Uniformity_HU : mean={df['Uniformity_HU'].mean():.2f}  std={df['Uniformity_HU'].std():.2f}  [{df['Uniformity_HU'].min():.2f}, {df['Uniformity_HU'].max():.2f}]")
print(f"    Scaling_V_mm  : mean={df['Scaling_V_mm'].mean():.2f}  std={df['Scaling_V_mm'].std():.2f}  [{df['Scaling_V_mm'].min():.2f}, {df['Scaling_V_mm'].max():.2f}]")
print(f"    HU_Precision  : mean={df['HU_Precision'].mean():.2f}  std={df['HU_Precision'].std():.2f}  [{df['HU_Precision'].min():.2f}, {df['HU_Precision'].max():.2f}]")
print("-" * 65)
print("  TARGET STATISTICS (RUL in days):")
print(f"    RUL_Tube      : mean={df['RUL_Tube'].mean():.0f}  std={df['RUL_Tube'].std():.0f}  [{df['RUL_Tube'].min()}, {df['RUL_Tube'].max()}]")
print(f"    RUL_Gantry    : mean={df['RUL_Gantry'].mean():.0f}  std={df['RUL_Gantry'].std():.0f}  [{df['RUL_Gantry'].min()}, {df['RUL_Gantry'].max()}]")
print(f"    RUL_Table     : mean={df['RUL_Table'].mean():.0f}  std={df['RUL_Table'].std():.0f}  [{df['RUL_Table'].min()}, {df['RUL_Table'].max()}]")
print(f"    RUL_Generator : mean={df['RUL_Generator'].mean():.0f}  std={df['RUL_Generator'].std():.0f}  [{df['RUL_Generator'].min()}, {df['RUL_Generator'].max()}]")
print("=" * 65)
