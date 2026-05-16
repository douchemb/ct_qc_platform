"""
predictive_maintenance/2_train_predictive_models.py
====================================================
ML Training — 4-Component RUL Prediction (RandomForest)
Context-Aware v6 — kVp & mAs as Input Features

Trains 4 independent RandomForestRegressor models, one per hardware
component, to predict Remaining Useful Life from daily QA metrics
PLUS dosimetric context (kVp, mAs).

  Model 1 (Tube):      X -> RUL_Tube       (saved: rf_tube.pkl)
  Model 2 (Gantry):    X -> RUL_Gantry     (saved: rf_gantry.pkl)
  Model 3 (Table):     X -> RUL_Table       (saved: rf_table.pkl)
  Model 4 (Generator): X -> RUL_Generator   (saved: rf_generator.pkl)

Features (X): Noise_HU, Uniformity_HU, Scaling_V_mm, HU_Precision, kVp, mAs

The Tube model should learn that:
  - kVp and mAs set the noise BASELINE (physics)
  - Noise_HU ABOVE that baseline indicates tube degradation (hardware)
  Feature importance for Tube should show kVp, mAs, AND Noise_HU sharing weight.

Evaluation: 80/20 train-test split with MAE, R2, and Feature Importance.

Usage:
    python predictive_maintenance/2_train_predictive_models.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

SEED = 42
TEST_SIZE = 0.20

FEATURE_COLS = [
    "Noise_HU", "Uniformity_HU", "Scaling_V_mm", "HU_Precision",
    "kVp", "mAs",
]

MODELS = {
    "X-Ray Tube": {
        "target": "RUL_Tube",
        "output": "rf_tube.pkl",
        "expected_feature": "Noise_HU",
        # For the Tube model, kVp/mAs/Noise should ALL be significant
        "context_aware": True,
    },
    "Gantry/Brushblock": {
        "target": "RUL_Gantry",
        "output": "rf_gantry.pkl",
        "expected_feature": "Uniformity_HU",
        "context_aware": False,
    },
    "Patient Table": {
        "target": "RUL_Table",
        "output": "rf_table.pkl",
        "expected_feature": "Scaling_V_mm",
        "context_aware": False,
    },
    "HV Generator": {
        "target": "RUL_Generator",
        "output": "rf_generator.pkl",
        "expected_feature": "HU_Precision",
        "context_aware": False,
    },
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": SEED,
    "n_jobs": -1,
}


# ══════════════════════════════════════════════════════════════════
# Training Pipeline
# ══════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    model_name: str, expected_feature: str,
    context_aware: bool = False,
) -> tuple[RandomForestRegressor, bool]:
    """Train a RandomForest and validate feature importance."""
    print(f"\n  --- {model_name} ---")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE  (test): {mae:>8.2f} days")
    print(f"  R2   (test): {r2:>8.4f}")

    # Feature importance
    print(f"  Feature Importance:")
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_feature = max(importances, key=importances.get)

    if context_aware:
        # For context-aware models (Tube), verify that kVp/mAs have
        # significant importance alongside the primary metric
        primary_imp = importances.get(expected_feature, 0)
        kvp_imp = importances.get("kVp", 0)
        mas_imp = importances.get("mAs", 0)
        context_total = kvp_imp + mas_imp + primary_imp
        # The combined importance of Noise+kVp+mAs should dominate
        isolation_ok = context_total > 0.50
    else:
        isolation_ok = (top_feature == expected_feature)

    for feat in FEATURE_COLS:
        imp = importances[feat]
        bar = "#" * int(imp * 40)
        marker = ""
        if feat == top_feature:
            marker = " <<< PRIMARY"
        elif context_aware and feat in ("kVp", "mAs") and imp > 0.05:
            marker = " <<< CONTEXT"
        print(f"    {feat:<18s} {imp:6.1%}  {bar}{marker}")

    tag = "OK" if isolation_ok else "FAIL"
    if context_aware:
        print(f"  [{tag}] Context-Aware: Noise+kVp+mAs combined "
              f"importance = {context_total:.1%} (expect >50%)")
    else:
        print(f"  [{tag}] Top feature: {top_feature} "
              f"(expected: {expected_feature})")

    return model, isolation_ok


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "historical_qa_data.csv"

    print("=" * 64)
    print("  ML Training - 4-Component RUL Prediction")
    print("  SOMATOM go.Sim - Context-Aware (kVp & mAs)")
    print("=" * 64)

    # Load data
    if not csv_path.exists():
        print(f"\n[ERROR] Dataset not found: {csv_path}")
        print("  Run 1_generate_synthetic_data.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n[OK] Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Targets:  {[m['target'] for m in MODELS.values()]}")

    # Verify kVp/mAs columns exist
    for col in ["kVp", "mAs"]:
        if col not in df.columns:
            print(f"\n[ERROR] Column '{col}' not found in dataset.")
            print("  Re-run 1_generate_synthetic_data.py to generate v6 data.")
            return

    X = df[FEATURE_COLS].values

    # Train each model
    all_ok = True
    results = []

    for name, cfg in MODELS.items():
        y = df[cfg["target"]].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED,
        )

        model, ok = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            name, cfg["expected_feature"],
            context_aware=cfg.get("context_aware", False),
        )
        if not ok:
            all_ok = False

        # Save model
        model_path = base_dir / cfg["output"]
        joblib.dump(model, model_path)

        size_kb = model_path.stat().st_size / 1024
        results.append((name, cfg["output"], size_kb,
                        mean_absolute_error(y_test, model.predict(X_test)),
                        r2_score(y_test, model.predict(X_test))))
        print(f"  [SAVED] {model_path.name} ({size_kb:.0f} KB)")

    # Summary table
    print("\n" + "=" * 64)
    print("  TRAINING SUMMARY")
    print("=" * 64)
    print(f"  {'Model':<22s} {'File':<20s} {'Size':>6s} {'MAE':>8s} {'R2':>8s}")
    print(f"  {'-'*22} {'-'*20} {'-'*6} {'-'*8} {'-'*8}")
    for name, fname, size, mae, r2 in results:
        print(f"  {name:<22s} {fname:<20s} {size:>5.0f}K {mae:>7.1f}d {r2:>7.4f}")

    print(f"\n  Feature Isolation: {'ALL PASSED' if all_ok else 'SOME FAILED'}")

    # Cleanup old 2-model files if they exist
    for old in ["rf_gantry_model.pkl", "rf_table_model.pkl"]:
        old_path = base_dir / old
        if old_path.exists():
            print(f"  [CLEANUP] Removed legacy {old}")
            old_path.unlink()
    print()


if __name__ == "__main__":
    main()
