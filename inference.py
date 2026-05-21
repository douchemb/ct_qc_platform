"""
predictive_maintenance/inference.py
====================================
Multi-Vendor ML Inference Engine — 4-Component RUL Prediction

Loads all trained RandomForest models and provides
a unified API for the Streamlit dashboard.

Models loaded via @st.cache_resource (singleton, loaded once per session).

⚠️  IMPORTANT: After retraining models (.pkl files), you MUST either:
     1. Restart the Streamlit server, OR
     2. Clear the cache via the Streamlit UI (☰ → Clear cache)
    Otherwise @st.cache_resource will serve stale models with the
    wrong feature shape and cause ValueError on .predict().

Siemens features (6): [Noise_HU, Uniformity_HU, Scaling_V_mm, HU_Precision, kVp, mAs]
GE features      (4): [MTF_50_lp_cm, Uniformity_HU, Slice_Thickness_mm, HU_Precision]
Canon features   (4): [Noise_HU, Uniformity_HU, Scaling_V_mm, HU_Precision]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent

# ── Cache version — bump this after every retrain to bust stale cache ──
_MODEL_VERSION = "v8_canon_digital_twin_20260516"

# ── Model registry ────────────────────────────────────────────────
SIEMENS_MODELS = {
    "tube":      "rf_tube.pkl",
    "gantry":    "rf_gantry.pkl",
    "table":     "rf_table.pkl",
    "generator": "rf_generator.pkl",
}
SIEMENS_FEATURES = ["Noise_HU", "Uniformity_HU", "Scaling_V_mm", "HU_Precision", "kVp", "mAs"]

# Safe defaults for Siemens context features (standard QA protocol)
SIEMENS_DEFAULTS = {
    "kVp": 120.0,
    "mAs": 200.0,
}

GE_MODELS = {
    "tube":       "rf_ge_tube.pkl",
    "detectors":  "rf_ge_detectors.pkl",
    "table":      "rf_ge_table.pkl",
    "generator":  "rf_ge_generator.pkl",
}
GE_FEATURES = ["MTF_50_lp_cm", "Uniformity_HU", "Slice_Thickness_mm", "HU_Precision"]

CANON_MODELS = {
    "tube":      "rf_canon_tube.pkl",
    "gantry":    "rf_canon_gantry.pkl",
    "table":     "rf_canon_table.pkl",
    "generator": "rf_canon_generator.pkl",
}
CANON_FEATURES = ["Noise_HU", "Uniformity_HU", "Scaling_V_mm", "HU_Precision"]


@st.cache_resource
def _load_all_models(_version: str = _MODEL_VERSION) -> dict:
    """Load all models (4 Siemens + 4 GE + 4 Canon), cached as singleton.

    The _version parameter is a cache-busting key. When models are
    retrained and _MODEL_VERSION is bumped, the cache is invalidated
    automatically — no manual restart required.
    """
    import joblib
    models = {"siemens": {}, "ge": {}, "canon": {}}

    for key, filename in SIEMENS_MODELS.items():
        path = MODEL_DIR / filename
        if path.exists():
            model = joblib.load(path)
            n_feat = model.n_features_in_
            logger.info("Loaded %s: %d features", filename, n_feat)
            models["siemens"][key] = model
        else:
            models["siemens"][key] = None

    for key, filename in GE_MODELS.items():
        path = MODEL_DIR / filename
        if path.exists():
            model = joblib.load(path)
            n_feat = model.n_features_in_
            logger.info("Loaded %s: %d features", filename, n_feat)
            models["ge"][key] = model
        else:
            models["ge"][key] = None

    for key, filename in CANON_MODELS.items():
        path = MODEL_DIR / filename
        if path.exists():
            model = joblib.load(path)
            n_feat = model.n_features_in_
            logger.info("Loaded %s: %d features", filename, n_feat)
            models["canon"][key] = model
        else:
            models["canon"][key] = None

    return models


def _canon_mock_rul(metrics: dict[str, float]) -> dict[str, Optional[int]]:
    """Physics-based mock RUL predictions for Canon (no trained models yet).

    Uses empirical degradation heuristics derived from Canon Aquilion LB
    service literature to produce realistic RUL estimates from the
    4 image-quality metrics.

    This function is a TEMPORARY fallback until real Canon .pkl models
    are trained and deployed.
    """
    noise = float(metrics.get("Noise_HU", 3.0))
    unif = float(metrics.get("Uniformity_HU", 1.0))
    scaling_v = float(metrics.get("Scaling_V_mm", 330.0))
    hu_prec = float(metrics.get("HU_Precision", 1.0))

    # Canon nominal phantom = 330 mm
    scaling_err = abs(scaling_v - 330.0)

    # Tube: noise is primary indicator
    # Healthy noise ~ 2-4 HU → RUL ~ 150 days
    # Degraded noise ~ 6+ HU → RUL drops
    rul_tube = max(0, int(round(150 - (noise - 2.5) * 20)))

    # Gantry: uniformity drift indicates brushblock / bearing wear
    # Healthy NUI ~ 0-2 HU → RUL ~ 140 days
    rul_gantry = max(0, int(round(140 - unif * 15)))

    # Table: scaling error indicates mechanical drift
    # Healthy error ~ 0-1 mm → RUL ~ 160 days
    rul_table = max(0, int(round(160 - scaling_err * 25)))

    # Generator: HU precision drift indicates kVp calibration
    # Healthy delta ~ 0-2 HU → RUL ~ 145 days
    rul_generator = max(0, int(round(145 - hu_prec * 18)))

    return {
        "tube": rul_tube,
        "gantry": rul_gantry,
        "table": rul_table,
        "generator": rul_generator,
    }


def predict_rul(
    manufacturer: str,
    metrics: dict[str, float],
) -> dict[str, Optional[int]]:
    """Predict RUL for all 4 components from today's QA metrics.

    Strict routing:
      - SIEMENS: 6-feature vector [Noise, Uni, Scaling, HU, kVp, mAs]
      - GE:      4-feature vector [MTF, Uni, SliceThk, HU]
      - CANON:   4-feature vector [Noise, Uni, Scaling_V, HU]

    kVp and mAs use safe defaults (120 kV / 200 mAs) if not provided,
    so inference never crashes even if acquisition params are missing.

    Args:
        manufacturer: "GE", "SIEMENS", or "CANON"
        metrics: dict with the appropriate feature keys and float values.

    Returns:
        dict with keys 'tube', 'gantry'/'detectors', 'table', 'generator'
        mapped to integer RUL in days, or None if model missing.
    """
    all_models = _load_all_models(_MODEL_VERSION)
    mfr = manufacturer.upper()

    # ── Build feature vector with strict manufacturer routing ─────
    if mfr == "GE":
        model_set = all_models["ge"]
        feature_cols = GE_FEATURES
        # GE: 4 features, no context columns
        try:
            X = np.array([[float(metrics[f]) for f in feature_cols]])
        except KeyError as exc:
            raise ValueError(
                f"Missing GE feature for inference: {exc}. "
                f"Expected keys: {feature_cols}"
            ) from exc

    elif mfr == "CANON":
        model_set = all_models["canon"]
        feature_cols = CANON_FEATURES

        # Check if any Canon models are actually available
        has_trained_models = any(m is not None for m in model_set.values())

        if not has_trained_models:
            # No trained Canon models → use physics-based mock fallback
            logger.info(
                "No trained Canon models found — using physics-based "
                "mock RUL predictions."
            )
            return _canon_mock_rul(metrics)

        # Canon: 4 features, no kVp/mAs
        try:
            X = np.array([[float(metrics[f]) for f in feature_cols]])
        except KeyError as exc:
            raise ValueError(
                f"Missing Canon feature for inference: {exc}. "
                f"Expected keys: {feature_cols}"
            ) from exc

    else:
        # Siemens (default)
        model_set = all_models["siemens"]
        feature_cols = SIEMENS_FEATURES
        # Siemens: 6 features — use .get() with safe defaults for kVp/mAs
        feature_values = []
        for f in feature_cols:
            val = metrics.get(f)
            if val is None:
                # Use safe default for context features, raise for QA metrics
                if f in SIEMENS_DEFAULTS:
                    val = SIEMENS_DEFAULTS[f]
                    logger.warning(
                        "Missing '%s' in metrics, using default %.1f", f, val
                    )
                else:
                    raise ValueError(
                        f"Missing Siemens QA metric for inference: '{f}'. "
                        f"Expected keys: {feature_cols}"
                    )
            feature_values.append(float(val))
        X = np.array([feature_values])

    # ── Predict per component ─────────────────────────────────────
    results = {}
    for key, model in model_set.items():
        if model is None:
            results[key] = None
            continue

        # Shape guard: verify the loaded model expects the right feature count
        expected_n = getattr(model, "n_features_in_", X.shape[1])
        if expected_n != X.shape[1]:
            logger.error(
                "SHAPE MISMATCH: model '%s' expects %d features, "
                "but inference built %d. Clear Streamlit cache or restart!",
                key, expected_n, X.shape[1],
            )
            results[key] = None
            continue

        pred = model.predict(X)[0]
        results[key] = max(0, int(round(pred)))

    return results
