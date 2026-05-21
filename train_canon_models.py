"""
predictive_maintenance/train_canon_models.py
=============================================
Phase 2 — Canon Aquilion LB Random Forest Model Training (Digital Twin)

Reads the Digital Twin time-series dataset and trains 4 independent
RandomForest Regressors (one per hardware subsystem). Each model
accepts a 4-feature input vector:

    X = [Noise_HU, Uniformity_HU, Scaling_V_mm, HU_Precision]

IMPORTANT: Noise_HU must be normalized to 5.0mm slice equivalent
before inference. The models are trained on 5.0mm reference noise.

Models are saved as .pkl files directly in the predictive_maintenance/
directory so that inference.py can load them without path changes.

File naming matches CANON_MODELS in inference.py:
    rf_canon_tube.pkl
    rf_canon_gantry.pkl
    rf_canon_table.pkl
    rf_canon_generator.pkl

Usage:
    python train_canon_models.py
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

# Use Digital Twin dataset (time-series degradation)
DATA_FILE = SCRIPT_DIR / "data" / "canon_digital_twin.csv"

# Fallback to static dataset if twin not yet generated
DATA_FILE_FALLBACK = SCRIPT_DIR / "data" / "canon_synthetic_dataset.csv"

# Output directory = same as inference.py MODEL_DIR
MODEL_DIR = SCRIPT_DIR

# Feature columns (must match inference.py CANON_FEATURES exactly)
FEATURE_COLS = ["Noise_HU", "Uniformity_HU", "Scaling_V_mm", "HU_Precision"]

# Target columns and their corresponding .pkl filenames
TARGETS = {
    "RUL_Tube":      "rf_canon_tube.pkl",
    "RUL_Gantry":    "rf_canon_gantry.pkl",
    "RUL_Table":     "rf_canon_table.pkl",
    "RUL_Generator": "rf_canon_generator.pkl",
}

# Hyperparameters — tuned for Digital Twin time-series data
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}

# ══════════════════════════════════════════════════════════════════
# Step 1: Load Dataset
# ══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  PHASE 2 -- Canon Aquilion LB Model Training (Digital Twin)")
print("=" * 65)

# Prefer Digital Twin, fallback to static
if DATA_FILE.exists():
    data_source = DATA_FILE
    data_label = "Digital Twin (time-series)"
elif DATA_FILE_FALLBACK.exists():
    data_source = DATA_FILE_FALLBACK
    data_label = "Static synthetic (fallback)"
    print("  WARNING: Digital Twin dataset not found, using static fallback.")
    print(f"           Run generate_canon_twin.py first for best results.")
else:
    raise FileNotFoundError(
        f"No Canon dataset found.\n"
        f"  Tried: {DATA_FILE}\n"
        f"  Tried: {DATA_FILE_FALLBACK}\n"
        f"Run generate_canon_twin.py first (Phase 1)."
    )

df = pd.read_csv(data_source)
print(f"  Dataset     : {data_label}")
print(f"  Source      : {data_source.name}")
print(f"  Samples     : {len(df)}")
print(f"  Features    : {FEATURE_COLS}")
print(f"  Targets     : {list(TARGETS.keys())}")

X = df[FEATURE_COLS].values
print(f"  X shape     : {X.shape}")
print(f"  Noise ref   : 5.0 mm slice thickness")
print("-" * 65)

# ══════════════════════════════════════════════════════════════════
# Step 2: Train 4 Independent Random Forest Regressors
# ══════════════════════════════════════════════════════════════════

results = []

for target_col, pkl_filename in TARGETS.items():
    y = df[target_col].values

    print(f"\n  Training: {target_col} -> {pkl_filename}")
    print(f"    Target range: [{y.min()}, {y.max()}]  mean={y.mean():.1f}")

    # Train the model
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X, y)

    # Cross-validation score (5-fold R^2)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    mean_r2 = cv_scores.mean()
    std_r2 = cv_scores.std()

    # Feature importances
    importances = model.feature_importances_
    imp_str = ", ".join(
        f"{feat}={imp:.3f}" for feat, imp in zip(FEATURE_COLS, importances)
    )

    print(f"    CV R^2: {mean_r2:.4f} (+/- {std_r2:.4f})")
    print(f"    Feature importances: {imp_str}")
    print(f"    n_features_in_: {model.n_features_in_}")

    # Save .pkl
    output_path = MODEL_DIR / pkl_filename
    joblib.dump(model, output_path)
    file_size_kb = output_path.stat().st_size / 1024
    print(f"    Saved: {output_path.name} ({file_size_kb:.0f} KB)")

    results.append({
        "Target": target_col,
        "Model": pkl_filename,
        "R2_mean": mean_r2,
        "R2_std": std_r2,
        "n_features": model.n_features_in_,
    })

# ══════════════════════════════════════════════════════════════════
# Step 3: Summary
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE -- 4 Canon RF Models (Digital Twin)")
print("=" * 65)
print(f"  Data source   : {data_label}")
print(f"  Output dir    : {MODEL_DIR}")
print(f"  Feature shape : (1, {len(FEATURE_COLS)})")
print(f"  Noise ref     : 5.0 mm (normalize before inference!)")
print()

for r in results:
    status = "PASS" if r["R2_mean"] > 0.7 else "WARN"
    print(
        f"  [{status}] {r['Model']:30s}  "
        f"R2={r['R2_mean']:.4f}  "
        f"features={r['n_features']}"
    )

print()
print("  Next: Restart Streamlit or clear cache to load new models.")
print("=" * 65)
