"""
predictive_maintenance/inference.py
====================================
Multi-Vendor ML Inference Engine — 4-Component RUL Prediction

Loads all 8 trained RandomForest models (4 Siemens + 4 GE) and provides
a unified API for the Streamlit dashboard.

Models loaded via @st.cache_resource (singleton, loaded once per session).

⚠️  IMPORTANT: After retraining models (.pkl files), you MUST either:
     1. Restart the Streamlit server, OR
     2. Clear the cache via the Streamlit UI (☰ → Clear cache)
    Otherwise @st.cache_resource will serve stale models with the
    wrong feature shape and cause ValueError on .predict().

Siemens features (6): [Noise_HU, Uniformity_HU, Scaling_V_mm, HU_Precision, kVp, mAs]
GE features      (4): [MTF_50_lp_cm, Uniformity_HU, Slice_Thickness_mm, HU_Precision]
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
_MODEL_VERSION = "v6_context_aware_20260515"

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


@st.cache_resource
def _load_all_models(_version: str = _MODEL_VERSION) -> dict:
    """Load all 8 models (4 Siemens + 4 GE), cached as singleton.

    The _version parameter is a cache-busting key. When models are
    retrained and _MODEL_VERSION is bumped, the cache is invalidated
    automatically — no manual restart required.
    """
    import joblib
    models = {"siemens": {}, "ge": {}}

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

    return models


def predict_rul(
    manufacturer: str,
    metrics: dict[str, float],
) -> dict[str, Optional[int]]:
    """Predict RUL for all 4 components from today's QA metrics.

    Strict routing:
      - SIEMENS: 6-feature vector [Noise, Uni, Scaling, HU, kVp, mAs]
      - GE:      4-feature vector [MTF, Uni, SliceThk, HU]

    kVp and mAs use safe defaults (120 kV / 200 mAs) if not provided,
    so inference never crashes even if acquisition params are missing.

    Args:
        manufacturer: "GE" or "SIEMENS"
        metrics: dict with the appropriate feature keys and float values.

    Returns:
        dict with keys 'tube', 'gantry'/'detectors', 'table', 'generator'
        mapped to integer RUL in days, or None if model missing.
    """
    all_models = _load_all_models(_MODEL_VERSION)
    is_ge = manufacturer.upper() == "GE"

    # ── Build feature vector with strict manufacturer routing ─────
    if is_ge:
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
    else:
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
