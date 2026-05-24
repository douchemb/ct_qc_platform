"""
predictive_maintenance/1_generate_synthetic_data_ge.py
======================================================
Digital Twin - 4-Component Synthetic QA Data Generator (GE Discovery RT)

Simulates 1000 days of daily QA data with 4 INDEPENDENT hardware
degradation cycles using GE-specific metrics.

Component Map (strict causal isolation):
  X-Ray Tube      -> MTF_50_lp_cm       DECREASES with wear (8.0 -> 4.0)
  Detectors/DAS   -> Uniformity_HU      INCREASES with wear (0.5 -> 5.0+)
  Patient Table   -> Slice_Thickness_mm  DRIFTS from 5.0 mm nominal
  HV Generator    -> HU_Precision        INCREASES with wear (0.0 -> 4.0+)

Anti-leakage design:
  - Independent variable-length maintenance cycles per component
  - Coprime phase offsets to prevent sawtooth alignment
  - Metrics RESET on maintenance (RUL resets to max)
  - Post-failure plateau (no runaway)

Output: historical_qa_data_ge.csv (1000 rows x 9 columns)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 137  # Different seed from Siemens to avoid identical noise
N_DAYS = 1000


# ══════════════════════════════════════════════════════════════════
# Component Definitions (GE-Specific)
# ══════════════════════════════════════════════════════════════════

COMPONENTS = {
    "tube": {
        "metric": "MTF_50_lp_cm",
        "rul_col": "RUL_Tube",
        "cycles": [1035, 1065, 1095, 1050, 1080],  # ~1065d avg (max 1095d = 3 yrs)
        "phase_offset": 0,
        "baseline": 8.0,                       # healthy MTF50 (lp/cm)
        "drift_amplitude": 4.0,                # degrades by 4 -> reaches ~4.0
        "noise_sigma": 0.25,                   # daily measurement noise
        "steepness": 2.5,
        "direction": -1,                       # MTF DROPS with wear
    },
    "detectors": {
        "metric": "Uniformity_HU",
        "rul_col": "RUL_Detectors",
        "cycles": [3225, 3255, 3285, 3240, 3270],  # ~3255d avg (max 3285d = 9 yrs)
        "phase_offset": 67,                    # coprime offset
        "baseline": 0.5,
        "drift_amplitude": 4.5,
        "noise_sigma": 0.20,
        "steepness": 2.5,
        "direction": +1,                       # uniformity INCREASES with wear
    },
    "table": {
        "metric": "Slice_Thickness_mm",
        "rul_col": "RUL_Table",
        "cycles": [1765, 1795, 1825, 1780, 1810],  # ~1795d avg (max 1825d = 5 yrs)
        "phase_offset": 41,                    # coprime offset
        "baseline": 5.0,                       # nominal slice thickness
        "drift_amplitude": 1.5,                # drifts up to 6.5 mm
        "noise_sigma": 0.08,
        "steepness": 3.0,
        "direction": +1,                       # thickness INCREASES (table sag)
    },
    "generator": {
        "metric": "HU_Precision",
        "rul_col": "RUL_Generator",
        "cycles": [2495, 2525, 2555, 2510, 2540],  # ~2525d avg (max 2555d = 7 yrs)
        "phase_offset": 211,                   # coprime offset
        "baseline": 0.0,                       # perfect HU accuracy
        "drift_amplitude": 3.5,
        "noise_sigma": 0.25,
        "steepness": 2.0,
        "direction": +1,                       # HU drift INCREASES
    },
}


# ══════════════════════════════════════════════════════════════════
# Cyclic RUL Generator
# ══════════════════════════════════════════════════════════════════

def _generate_cyclic_rul(
    n_days: int, cycle_lengths: list[int], phase_offset: int = 0,
) -> np.ndarray:
    """Generate sawtooth RUL from variable-length maintenance cycles.

    phase_offset shifts the start point to decorrelate from other
    components' sawtooth patterns.
    """
    total = n_days + phase_offset
    rul = np.zeros(total, dtype=np.float64)
    day = 0
    cycle_idx = 0
    while day < total:
        clen = cycle_lengths[cycle_idx % len(cycle_lengths)]
        for t in range(clen):
            if day >= total:
                break
            rul[day] = clen - t
            day += 1
        cycle_idx += 1
    return rul[phase_offset:phase_offset + n_days]


def _degradation_from_rul(
    rul: np.ndarray,
    max_rul: float,
    amplitude: float,
    steepness: float,
    direction: int,
    baseline: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute metric values from RUL using power-law degradation.

    wear = (1 - RUL/max_RUL)^steepness
    metric = baseline + direction * amplitude * wear + noise
    """
    wear = np.clip(1.0 - rul / max_rul, 0.0, 1.0) ** steepness
    drift = direction * amplitude * wear
    noise = rng.normal(0, noise_sigma, len(rul))
    return baseline + drift + noise


# ══════════════════════════════════════════════════════════════════
# Main Generator
# ══════════════════════════════════════════════════════════════════

def generate_synthetic_data() -> pd.DataFrame:
    """Generate 1000-day GE QA data with 4 independent degradation cycles."""
    rng = np.random.default_rng(SEED)
    data = {"Days_Passed": np.arange(N_DAYS)}

    for comp_name, cfg in COMPONENTS.items():
        rul = _generate_cyclic_rul(
            N_DAYS, cfg["cycles"], cfg.get("phase_offset", 0))
        max_rul = max(cfg["cycles"])

        metric = _degradation_from_rul(
            rul, max_rul, cfg["drift_amplitude"], cfg["steepness"],
            cfg["direction"], cfg["baseline"], cfg["noise_sigma"], rng,
        )

        # Clamp physical bounds
        if cfg["metric"] == "MTF_50_lp_cm":
            metric = np.clip(metric, 0.5, 12.0)
        elif cfg["metric"] == "Slice_Thickness_mm":
            metric = np.clip(metric, 1.0, 10.0)
        elif cfg["direction"] > 0 and cfg["baseline"] >= 0:
            metric = np.clip(metric, 0.0, None)

        data[cfg["metric"]] = np.round(metric, 3)
        data[cfg["rul_col"]] = rul.astype(int)

    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("  Digital Twin - GE Discovery RT Synthetic QA Generator")
    print("  4-Component Independent Cyclic Degradation")
    print("=" * 64)

    df = generate_synthetic_data()

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "historical_qa_data_ge.csv"
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Generated {len(df)} rows x {len(df.columns)} columns")
    print(f"[->] Saved to: {output_path}\n")

    print("-- Column Summary " + "-" * 46)
    print(df.describe().round(3).to_string())

    print("\n-- Sample Checkpoints " + "-" * 43)
    for day in [0, 50, 150, 250, 400, 600, 800, 999]:
        r = df.iloc[day]
        print(
            f"  Day {day:>4d} | "
            f"MTF={r['MTF_50_lp_cm']:5.2f} | "
            f"Uni={r['Uniformity_HU']:5.2f} | "
            f"SlT={r['Slice_Thickness_mm']:5.2f} | "
            f"HUP={r['HU_Precision']:5.2f} | "
            f"T={int(r['RUL_Tube']):>3d} "
            f"D={int(r['RUL_Detectors']):>3d} "
            f"Tb={int(r['RUL_Table']):>3d} "
            f"Gn={int(r['RUL_Generator']):>3d}"
        )

    # Correlation matrix
    print("\n-- Pearson Correlations " + "-" * 41)
    features = ["MTF_50_lp_cm", "Uniformity_HU", "Slice_Thickness_mm", "HU_Precision"]
    targets = ["RUL_Tube", "RUL_Detectors", "RUL_Table", "RUL_Generator"]
    corr = df[features + targets].corr()
    cross = corr.loc[features, targets]
    print(cross.round(3).to_string())

    # Isolation verification
    print("\n-- ISOLATION VERIFICATION " + "-" * 39)
    expected_pairs = [
        ("MTF_50_lp_cm", "RUL_Tube", "strong"),
        ("Uniformity_HU", "RUL_Detectors", "strong"),
        ("Slice_Thickness_mm", "RUL_Table", "strong"),
        ("HU_Precision", "RUL_Generator", "strong"),
    ]
    all_ok = True
    for feat, tgt, _ in expected_pairs:
        r = cross.loc[feat, tgt]
        ok = abs(r) > 0.5
        tag = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{tag}] {feat:22s} <-> {tgt:16s}: r={r:+.3f}  (expect strong)")

    print("\n  Cross-contamination (should be |r| < 0.3):")
    for feat, tgt_own, _ in expected_pairs:
        for tgt in targets:
            if tgt == tgt_own:
                continue
            r = cross.loc[feat, tgt]
            ok = abs(r) < 0.3
            tag = "OK" if ok else "WARN"
            if not ok:
                all_ok = False
            print(f"  [{tag}] {feat:22s} <-> {tgt:16s}: r={r:+.3f}")

    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print()


if __name__ == "__main__":
    main()
