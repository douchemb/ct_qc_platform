"""
predictive_maintenance/generate_canon_data.py
=============================================
Phase 1 — Canon Aquilion LB Synthetic Data Generation

Generates a 1500-sample physics-based synthetic dataset for the
Canon Aquilion LB Digital Twin. Each RUL target is causally linked
to specific input metrics via degradation rules derived from
Canon service literature and CT physics principles.

Inputs (4 features):
  - Noise_HU      : Image noise (SD of center ROI in HU)
  - Uniformity_HU : Non-Uniformity Index (max |edge - center| in HU)
  - Scaling_V_mm  : Vertical phantom diameter measurement (mm)
  - HU_Precision  : |measured_center - 0| drift (HU)

Targets (4 RUL components):
  - RUL_Tube      : X-Ray tube remaining useful life (days)
  - RUL_Gantry    : Gantry/bearing remaining useful life (days)
  - RUL_Table     : Patient table remaining useful life (days)
  - RUL_Generator : HV generator remaining useful life (days)

Physics-Based Rules:
  Tube      ← Noise (photon starvation) + Precision (focal spot drift)
  Gantry    ← Uniformity (detector/rotation misalignment)
  Table     ← Scaling error (mechanical positioning drift)
  Generator ← Noise + Uniformity (overall system stress)

Usage:
    python generate_canon_data.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
N_SAMPLES = 1500

# Canon Aquilion LB nominal phantom diameter
CANON_NOMINAL_DIAMETER = 330.0  # mm

# ══════════════════════════════════════════════════════════════════
# Step 1: Generate Input Features (4 Features)
# ══════════════════════════════════════════════════════════════════

# Noise (SD): Canon Aquilion LB typical range 12-18 HU at standard protocol
noise = np.random.normal(15.0, 3.0, N_SAMPLES)

# Uniformity (NUI): max |edge - center|, healthy < 2 HU
unif = np.random.normal(1.5, 0.6, N_SAMPLES)

# Scaling Vertical: measured phantom diameter in mm (nominal = 330.0)
scaling = np.random.normal(CANON_NOMINAL_DIAMETER, 2.0, N_SAMPLES)

# HU Precision: |center_mean - 0|, healthy < 2 HU
precision = np.random.normal(1.0, 0.4, N_SAMPLES)

# Clip inputs to physically realistic ranges
noise = np.clip(noise, 5.0, 30.0)
unif = np.clip(unif, 0.0, 8.0)
scaling = np.clip(scaling, 320.0, 340.0)
precision = np.clip(precision, 0.0, 6.0)

# ══════════════════════════════════════════════════════════════════
# Step 2: Physics-Based Target Rules (Causal Degradation Models)
# ══════════════════════════════════════════════════════════════════

# --- Tube RUL ---
# Physics: Noise increases with anode pitting / focal spot blooming.
#          Precision degrades as kVp output drifts from tube aging.
# Rule: Base 300 days, penalized by excess noise and HU drift.
y_tube = 300 - ((noise - 15) * 10) - (precision * 15)

# --- Gantry RUL ---
# Physics: Uniformity degrades when gantry bearings wear or
#          detector ring alignment drifts during rotation.
# Rule: Base 365 days, penalized heavily by uniformity degradation.
y_gantry = 365 - ((unif - 1.5) * 40)

# --- Table RUL ---
# Physics: Scaling error is a direct measure of mechanical positioning
#          accuracy. Table rail friction and belt stretch cause drift.
# Rule: Base 400 days, penalized by absolute scaling error from nominal.
y_table = 400 - (np.abs(scaling - CANON_NOMINAL_DIAMETER) * 25)

# --- Generator RUL ---
# Physics: Overall system stress indicator. A degrading generator
#          causes both increased noise (unstable kVp ripple) and
#          uniformity drift (inconsistent mA regulation).
# Rule: Base 350 days, penalized by both noise and uniformity excess.
y_gen = 350 - ((noise - 15) * 5) - ((unif - 1.5) * 20)

# Add realistic measurement noise to targets
y_tube += np.random.normal(0, 5, N_SAMPLES)
y_gantry += np.random.normal(0, 5, N_SAMPLES)
y_table += np.random.normal(0, 5, N_SAMPLES)
y_gen += np.random.normal(0, 5, N_SAMPLES)

# Clip to realistic operational ranges (0 to max lifespan)
y_tube = np.clip(y_tube, 0, 300)
y_gantry = np.clip(y_gantry, 0, 365)
y_table = np.clip(y_table, 0, 400)
y_gen = np.clip(y_gen, 0, 350)

# Round to integer days
y_tube = np.round(y_tube).astype(int)
y_gantry = np.round(y_gantry).astype(int)
y_table = np.round(y_table).astype(int)
y_gen = np.round(y_gen).astype(int)

# ══════════════════════════════════════════════════════════════════
# Step 3: Assemble and Save Dataset
# ══════════════════════════════════════════════════════════════════

df = pd.DataFrame({
    "Noise_HU": np.round(noise, 4),
    "Uniformity_HU": np.round(unif, 4),
    "Scaling_V_mm": np.round(scaling, 4),
    "HU_Precision": np.round(precision, 4),
    "RUL_Tube": y_tube,
    "RUL_Gantry": y_gantry,
    "RUL_Table": y_table,
    "RUL_Generator": y_gen,
})

file_path = os.path.join(DATA_DIR, "canon_synthetic_dataset.csv")
df.to_csv(file_path, index=False)

# ══════════════════════════════════════════════════════════════════
# Step 4: Summary Statistics
# ══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  PHASE 1 COMPLETE — Canon Aquilion LB Synthetic Dataset")
print("=" * 65)
print(f"  Samples     : {N_SAMPLES}")
print(f"  Features    : {list(df.columns[:4])}")
print(f"  Targets     : {list(df.columns[4:])}")
print(f"  Output      : {file_path}")
print("-" * 65)
print("  INPUT STATISTICS:")
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
