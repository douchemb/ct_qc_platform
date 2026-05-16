"""
predictive_maintenance/2_train_predictive_models_ge.py
======================================================
ML Training — GE Discovery RT 4-Component RUL Prediction

Trains 4 independent RandomForestRegressor models using GE-specific
QA metrics to predict Remaining Useful Life per hardware component.

  Model 1 (Tube):       X -> RUL_Tube       (saved: rf_ge_tube.pkl)
  Model 2 (Detectors):  X -> RUL_Detectors  (saved: rf_ge_detectors.pkl)
  Model 3 (Table):      X -> RUL_Table      (saved: rf_ge_table.pkl)
  Model 4 (Generator):  X -> RUL_Generator  (saved: rf_ge_generator.pkl)

Features (X): MTF_50_lp_cm, Uniformity_HU, Slice_Thickness_mm, HU_Precision

Usage:
    python predictive_maintenance/2_train_predictive_models_ge.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

SEED = 42
TEST_SIZE = 0.20

FEATURE_COLS = ["MTF_50_lp_cm", "Uniformity_HU", "Slice_Thickness_mm", "HU_Precision"]

MODELS = {
    "X-Ray Tube": {
        "target": "RUL_Tube",
        "output": "rf_ge_tube.pkl",
        "expected_feature": "MTF_50_lp_cm",
    },
    "Detectors/DAS": {
        "target": "RUL_Detectors",
        "output": "rf_ge_detectors.pkl",
        "expected_feature": "Uniformity_HU",
    },
    "Patient Table": {
        "target": "RUL_Table",
        "output": "rf_ge_table.pkl",
        "expected_feature": "Slice_Thickness_mm",
    },
    "HV Generator": {
        "target": "RUL_Generator",
        "output": "rf_ge_generator.pkl",
        "expected_feature": "HU_Precision",
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


def train_and_evaluate(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    model_name: str, expected_feature: str,
) -> tuple[RandomForestRegressor, bool]:
    """Train a RandomForest and validate feature importance."""
    print(f"\n  --- {model_name} ---")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE  (test): {mae:>8.2f} days")
    print(f"  R2   (test): {r2:>8.4f}")

    print(f"  Feature Importance:")
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_feature = max(importances, key=importances.get)
    isolation_ok = (top_feature == expected_feature)

    for feat in FEATURE_COLS:
        imp = importances[feat]
        bar = "#" * int(imp * 40)
        marker = " <<< PRIMARY" if feat == top_feature else ""
        print(f"    {feat:<22s} {imp:6.1%}  {bar}{marker}")

    tag = "OK" if isolation_ok else "FAIL"
    print(f"  [{tag}] Top feature: {top_feature} "
          f"(expected: {expected_feature})")

    return model, isolation_ok


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "historical_qa_data_ge.csv"

    print("=" * 64)
    print("  ML Training - GE Discovery RT 4-Component RUL")
    print("  Independent RandomForest Models")
    print("=" * 64)

    if not csv_path.exists():
        print(f"\n[ERROR] Dataset not found: {csv_path}")
        print("  Run 1_generate_synthetic_data_ge.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n[OK] Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Targets:  {[m['target'] for m in MODELS.values()]}")

    X = df[FEATURE_COLS].values

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
        )
        if not ok:
            all_ok = False

        model_path = base_dir / cfg["output"]
        joblib.dump(model, model_path)

        size_kb = model_path.stat().st_size / 1024
        results.append((name, cfg["output"], size_kb,
                        mean_absolute_error(y_test, model.predict(X_test)),
                        r2_score(y_test, model.predict(X_test))))
        print(f"  [SAVED] {model_path.name} ({size_kb:.0f} KB)")

    print("\n" + "=" * 64)
    print("  TRAINING SUMMARY (GE)")
    print("=" * 64)
    print(f"  {'Model':<22s} {'File':<24s} {'Size':>6s} {'MAE':>8s} {'R2':>8s}")
    print(f"  {'-'*22} {'-'*24} {'-'*6} {'-'*8} {'-'*8}")
    for name, fname, size, mae, r2 in results:
        print(f"  {name:<22s} {fname:<24s} {size:>5.0f}K {mae:>7.1f}d {r2:>7.4f}")

    print(f"\n  Feature Isolation: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print()


if __name__ == "__main__":
    main()
