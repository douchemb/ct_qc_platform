"""
predictive_maintenance/1_generate_synthetic_data.py
====================================================
Digital Twin - 4-Component Synthetic QA Data Generator (v6)
SOMATOM go.Sim — Context-Aware (kVp & mAs)

Simulates 1000 days of daily QA data with 4 INDEPENDENT hardware
degradation cycles. Each component has its own maintenance schedule
and drives ONLY its physically-associated QA metric.

v6 UPGRADE: kVp and mAs are now included as acquisition context.
  - Noise_HU baseline is physics-dependent on kVp/mAs
  - The ML model can learn that 6 HU noise at 80 kV is healthy,
    but 6 HU at 120 kV indicates tube degradation.

Component Map (strict causal isolation):
  X-Ray Tube      -> Noise_HU         (cycle ~350-450 days)
  Gantry/Brushblock -> Uniformity_HU  (cycle ~250-350 days)
  Patient Table   -> Scaling_V_mm     (cycle ~200-280 days)
  HV Generator    -> HU_Precision     (cycle ~280-380 days)

Context Features (NOT degradation-coupled):
  kVp  — randomly sampled from [80, 100, 120, 140]
  mAs  — randomly sampled from [100, 150, 200, 250]

Anti-leakage design:
  - Each component follows independent, variable-length cycles
  - Metrics RESET to baseline when their component is maintained
  - No cross-subsystem coupling whatsoever
  - RUL is non-monotonic (sawtooth), preventing time-proxy leakage
  - kVp/mAs are independent random draws (no correlation with RUL)

Output: historical_qa_data.csv (1000 rows x 11 columns)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 1000

# Acquisition protocol space (Siemens SOMATOM go.Sim)
KVP_OPTIONS = [80, 100, 120, 140]
MAS_OPTIONS = [100, 150, 200, 250]


# ══════════════════════════════════════════════════════════════════
# Physics-Based Noise Baseline Model
# ══════════════════════════════════════════════════════════════════

def _noise_baseline(kvp: float, mas: float) -> float:
    """Compute physics-based noise baseline from kVp and mAs.

    CT image noise follows the inverse square-root law:
        σ ∝ 1 / √(dose) ∝ 1 / √(kVp^n × mAs)
    where n ≈ 2.5 for the kVp-to-dose relationship.

    Reference baselines (healthy tube):
        120 kV / 200 mAs → σ ≈ 3.0 HU  (standard QA protocol)
         80 kV / 100 mAs → σ ≈ 6.5 HU  (low-dose pediatric)
        140 kV / 250 mAs → σ ≈ 2.0 HU  (high-dose body)

    We anchor the model at 120 kV / 200 mAs = 3.0 HU and scale
    using the relative dose factor.
    """
    ref_kvp, ref_mas, ref_noise = 120.0, 200.0, 3.0
    n_kvp = 2.5  # kVp-to-dose power law exponent

    # Relative dose factor vs reference protocol
    dose_ratio = (kvp / ref_kvp) ** n_kvp * (mas / ref_mas)

    # Noise scales as 1/√dose
    return ref_noise / np.sqrt(dose_ratio)


# ══════════════════════════════════════════════════════════════════
# Component Definitions
# ══════════════════════════════════════════════════════════════════

COMPONENTS = {
    "tube": {
        "metric": "Noise_HU",
        "rul_col": "RUL_Tube",
        "cycles": [270, 310, 290, 330],       # ~300d avg (tube filament life)
        "phase_offset": 0,                    # tube starts at day 0
        # baseline is NOW dynamic (computed per-day from kVp/mAs)
        "baseline": None,                     # placeholder — overridden
        "drift_amplitude_factor": 0.8,        # wear multiplies baseline by up to 1.8x
        "noise_sigma_factor": 0.10,           # measurement noise = 10% of baseline
        "steepness": 2.5,                     # power-law exponent
        "direction": +1,                      # metric increases with wear
    },
    "gantry": {
        "metric": "Uniformity_HU",
        "rul_col": "RUL_Gantry",
        "cycles": [160, 180, 170, 190, 175],   # ~175d avg (frequent brushblock)
        "phase_offset": 73,                    # coprime offset from tube
        "baseline": 0.5,
        "drift_amplitude": 3.0,
        "noise_sigma": 0.20,
        "steepness": 2.5,
        "direction": +1,
    },
    "table": {
        "metric": "Scaling_V_mm",
        "rul_col": "RUL_Table",
        "cycles": [230, 270, 250, 210, 260],   # ~245d avg (table greasing)
        "phase_offset": 137,                   # coprime offset
        "baseline": 200.0,
        "drift_amplitude": 3.0,
        "noise_sigma": 0.10,
        "steepness": 3.0,
        "direction": -1,                       # scaling DROPS with wear
    },
    "generator": {
        "metric": "HU_Precision",
        "rul_col": "RUL_Generator",
        "cycles": [450, 490, 470, 510],        # ~480d avg (generator calibration)
        "phase_offset": 311,                   # large coprime offset from tube
        "baseline": 0.0,                       # perfect HU accuracy
        "drift_amplitude": 3.5,
        "noise_sigma": 0.25,
        "steepness": 2.0,
        "direction": +1,
    },
}


# ══════════════════════════════════════════════════════════════════
# Cyclic RUL Generator
# ══════════════════════════════════════════════════════════════════

def _generate_cyclic_rul(
    n_days: int, cycle_lengths: list[int], phase_offset: int = 0,
) -> np.ndarray:
    """Generate sawtooth RUL from variable-length maintenance cycles.

    Each cycle counts down from cycle_length to 1, then resets.
    phase_offset shifts the start point to decorrelate from other components.
    """
    # Generate extra days to account for offset, then slice
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
    # Slice off the offset to get the phase-shifted view
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
    """Generate 1000-day QA data with 4 independent degradation cycles.

    v6: kVp and mAs are randomly sampled per day as acquisition context.
    Tube noise baseline is physics-dependent on the daily kVp/mAs.
    """
    rng = np.random.default_rng(SEED)
    data = {"Days_Passed": np.arange(N_DAYS)}

    # ── Context features: random daily acquisition parameters ─────
    kvp_values = rng.choice(KVP_OPTIONS, size=N_DAYS)
    mas_values = rng.choice(MAS_OPTIONS, size=N_DAYS)
    data["kVp"] = kvp_values
    data["mAs"] = mas_values

    for comp_name, cfg in COMPONENTS.items():
        # Generate independent RUL with phase offset
        rul = _generate_cyclic_rul(
            N_DAYS, cfg["cycles"], cfg.get("phase_offset", 0))
        max_rul = max(cfg["cycles"])

        if comp_name == "tube":
            # ── Context-aware Tube noise model ────────────────────
            # Baseline noise depends on daily kVp/mAs (physics)
            # Degradation multiplies baseline by up to 1 + drift_factor
            wear = np.clip(1.0 - rul / max_rul, 0.0, 1.0) ** cfg["steepness"]

            metric = np.zeros(N_DAYS)
            for i in range(N_DAYS):
                base_noise = _noise_baseline(kvp_values[i], mas_values[i])
                degradation_multiplier = 1.0 + cfg["drift_amplitude_factor"] * wear[i]
                measurement_noise = rng.normal(0, base_noise * cfg["noise_sigma_factor"])
                metric[i] = base_noise * degradation_multiplier + measurement_noise

            metric = np.clip(metric, 0.5, 25.0)
        else:
            # Standard degradation model for non-tube components
            metric = _degradation_from_rul(
                rul, max_rul, cfg["drift_amplitude"], cfg["steepness"],
                cfg["direction"], cfg["baseline"], cfg["noise_sigma"], rng,
            )

            # Clamp non-negative for physical metrics
            if cfg["baseline"] >= 0 and cfg["direction"] > 0:
                metric = np.clip(metric, 0.0, None)

        data[cfg["metric"]] = np.round(metric, 3)
        data[cfg["rul_col"]] = rul.astype(int)

    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("  Digital Twin - 4-Component Synthetic QA Generator (v6)")
    print("  SOMATOM go.Sim - Context-Aware (kVp & mAs)")
    print("=" * 64)

    df = generate_synthetic_data()

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "historical_qa_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Generated {len(df)} rows x {len(df.columns)} columns")
    print(f"[->] Saved to: {output_path}\n")

    # Column summary
    print("-- Column Summary " + "-" * 46)
    print(df.describe().round(3).to_string())

    # kVp/mAs distribution
    print("\n-- kVp/mAs Distribution " + "-" * 40)
    print(f"  kVp values: {sorted(df['kVp'].unique())}")
    print(f"  mAs values: {sorted(df['mAs'].unique())}")
    print(f"  kVp counts: {dict(df['kVp'].value_counts().sort_index())}")
    print(f"  mAs counts: {dict(df['mAs'].value_counts().sort_index())}")

    # Physics validation: noise vs protocol
    print("\n-- Noise vs Protocol (Physics Validation) " + "-" * 22)
    for kvp in KVP_OPTIONS:
        for mas in MAS_OPTIONS:
            subset = df[(df["kVp"] == kvp) & (df["mAs"] == mas)]
            if len(subset) > 0:
                mean_noise = subset["Noise_HU"].mean()
                expected = _noise_baseline(kvp, mas)
                print(f"  {kvp:>3d} kV / {mas:>3d} mAs: "
                      f"mean={mean_noise:5.2f} HU  "
                      f"(physics baseline={expected:5.2f} HU, "
                      f"n={len(subset)})")

    # Sample checkpoints
    print("\n-- Sample Checkpoints " + "-" * 43)
    for day in [0, 50, 150, 250, 400, 600, 800, 999]:
        r = df.iloc[day]
        print(
            f"  Day {day:>4d} | "
            f"kVp={int(r['kVp']):>3d} mAs={int(r['mAs']):>3d} | "
            f"Noise={r['Noise_HU']:5.2f} | "
            f"Uni={r['Uniformity_HU']:5.2f} | "
            f"ScV={r['Scaling_V_mm']:7.2f} | "
            f"HUP={r['HU_Precision']:5.2f} | "
            f"T={int(r['RUL_Tube']):>3d} "
            f"G={int(r['RUL_Gantry']):>3d} "
            f"Tb={int(r['RUL_Table']):>3d} "
            f"Gn={int(r['RUL_Generator']):>3d}"
        )

    # Correlation matrix — THE PROOF
    print("\n-- Pearson Correlations " + "-" * 41)
    features = ["Noise_HU", "Uniformity_HU", "Scaling_V_mm", "HU_Precision",
                "kVp", "mAs"]
    targets = ["RUL_Tube", "RUL_Gantry", "RUL_Table", "RUL_Generator"]
    corr = df[features + targets].corr()
    # Show only the feature-target cross-correlation block
    cross = corr.loc[features, targets]
    print(cross.round(3).to_string())

    # Isolation verification
    print("\n-- ISOLATION VERIFICATION " + "-" * 39)
    expected_pairs = [
        ("Noise_HU", "RUL_Tube", "strong"),
        ("Uniformity_HU", "RUL_Gantry", "strong"),
        ("Scaling_V_mm", "RUL_Table", "strong"),
        ("HU_Precision", "RUL_Generator", "strong"),
    ]
    all_ok = True
    for feat, tgt, _ in expected_pairs:
        r = cross.loc[feat, tgt]
        ok = abs(r) > 0.3  # lower threshold since kVp/mAs add variance
        tag = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{tag}] {feat:16s} <-> {tgt:16s}: r={r:+.3f}  (expect strong)")

    # kVp/mAs should NOT correlate with any RUL (they're random draws)
    print("\n  kVp/mAs independence (should be |r| < 0.1):")
    for ctx in ["kVp", "mAs"]:
        for tgt in targets:
            r = cross.loc[ctx, tgt]
            ok = abs(r) < 0.1
            tag = "OK" if ok else "WARN"
            print(f"  [{tag}] {ctx:16s} <-> {tgt:16s}: r={r:+.3f}")

    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print()


if __name__ == "__main__":
    main()
