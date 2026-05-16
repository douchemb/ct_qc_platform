# -*- coding: utf-8 -*-
"""
modules/image_qc/ed_calibration.py — HU to Electron Density Calibration.

Computes the HU→RED calibration curve from phantom measurements using
the stoichiometric calibration method (Schneider et al. 1996).

Hardware failure mapping:
    curve_slope drift — kVp generator failure indicator.
    Soft-tissue slope → low-kVp instability.
    Bone slope → high-kVp instability.

References:
    - Schneider et al. Phys. Med. Biol. 41(1) 1996 — stoichiometric method
    - IAEA TRS-430 (2004) Section 4.2 — CT calibration for TPS
    - AAPM TG-66 Section 7 — CT simulator calibration
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config import CONFIG

if TYPE_CHECKING:
    from config import AppConfig
    from modules.image_qc.roi_stats import VolumetricQCResult, ROIDescriptor

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["EDCalibrationAnalyzer", "EDCalibrationResult", "EDMaterialMeasurement"]


@dataclass
class EDMaterialMeasurement:
    """Measured HU for one phantom insert, compared to its known RED."""
    material_name: str
    nominal_hu: float
    measured_mean_hu: float
    measured_std_hu: float
    reference_red: float
    computed_red: float
    red_deviation: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EDCalibrationResult:
    """
    Complete HU->ED calibration curve result for one CT session.

    Hardware failure mapping:
      soft_tissue_slope, bone_slope — both are indicators of
        kVp generator failure. A drifting kVp changes beam quality, which
        changes the HU-to-RED mapping non-uniformly across the density range.
        Soft-tissue slope drift indicates low-kVp instability.
        Bone slope drift indicates high-kVp instability.
    Reference: Schneider et al. 1996; IAEA TRS-430 Section 4.2.
    """
    acquisition_date: str
    series_description: str
    scanner_id: str
    measurements: list[EDMaterialMeasurement]
    soft_tissue_slope: float
    soft_tissue_intercept: float
    soft_tissue_r_squared: float
    soft_tissue_hu_range: tuple[float, float]
    bone_slope: float
    bone_intercept: float
    bone_r_squared: float
    bone_hu_range: tuple[float, float]
    segment_join_hu: float
    hu_curve: np.ndarray
    red_curve: np.ndarray
    max_red_deviation: float
    mean_red_deviation: float
    all_passed: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        # Convert EDMaterialMeasurement list
        d["measurements"] = [m.to_dict() if hasattr(m, 'to_dict') else m
                             for m in self.measurements]
        return d

    def passes_clinical_acceptance(self) -> bool:
        """Returns True if all materials pass and both segment R^2 > 0.999."""
        return (self.all_passed
                and self.soft_tissue_r_squared > 0.999
                and self.bone_r_squared > 0.999)

    def get_red_for_hu(self, hu_value: float) -> float:
        """Interpolates the piecewise linear calibration curve at hu_value.

        Uses soft-tissue segment for HU < segment_join_hu,
        bone segment for HU >= segment_join_hu.
        """
        if hu_value < self.segment_join_hu:
            return self.soft_tissue_slope * hu_value + self.soft_tissue_intercept
        else:
            return self.bone_slope * hu_value + self.bone_intercept

    def export_for_tps(self, output_path: Path, format: str = "generic_csv") -> Path:
        """Export calibration curve for TPS import.

        For 'generic_csv': two-column CSV (HU, RED), sampled every 10 HU.
        For 'varian_eclipse': tab-separated with header.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        hu_samples = np.arange(-1000, 1510, 10, dtype=np.float64)
        red_samples = np.array([self.get_red_for_hu(h) for h in hu_samples])

        if format == "generic_csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["HU", "RED"])
                for h, r in zip(hu_samples, red_samples):
                    writer.writerow([f"{h:.1f}", f"{r:.6f}"])
        elif format == "varian_eclipse":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# Varian Eclipse CT Calibration\n")
                f.write("# Scanner: %s\n" % self.scanner_id)
                f.write("# Date: %s\n" % self.acquisition_date)
                for h, r in zip(hu_samples, red_samples):
                    f.write("%.1f\t%.6f\n" % (h, r))
        else:
            raise ValueError("Unknown TPS format: %s" % format)

        logger.info("TPS calibration exported to: %s", output_path)
        return output_path


class EDCalibrationAnalyzer:
    """Computes the HU->Electron Density calibration curve from phantom measurements.

    Implements the stoichiometric calibration method (Schneider et al. 1996).
    """

    def __init__(self, config: "AppConfig") -> None:
        self._config = config
        self._ed_config = config.ed_calibration

    def analyze(
        self,
        volumetric_result: "VolumetricQCResult",
        material_rois: dict[str, "ROIDescriptor"],
        scanner_id: str = "SCANNER_001",
    ) -> EDCalibrationResult:
        """Primary entry point for ED calibration analysis."""
        # Build lookup of reference data from config
        ref_lookup = {}
        for name, nominal_hu, ref_red in self._ed_config.phantom_materials:
            ref_lookup[name] = (nominal_hu, ref_red)

        # Extract measured HU for each material
        hu_values = []
        red_values = []
        measurements = []

        for material_name, roi in material_rois.items():
            if material_name not in ref_lookup:
                logger.warning("Material '%s' not in config reference table", material_name)
                continue

            nominal_hu, ref_red = ref_lookup[material_name]
            mean_hu, std_hu = self._extract_material_hu(
                volumetric_result, material_name, roi
            )

            hu_values.append(mean_hu)
            red_values.append(ref_red)
            measurements.append({
                "material_name": material_name,
                "nominal_hu": nominal_hu,
                "measured_mean_hu": mean_hu,
                "measured_std_hu": std_hu,
                "reference_red": ref_red,
            })

        # Fit piecewise linear calibration curve
        segment_join_hu = 100.0
        soft_params, bone_params = self._fit_calibration_curve(
            hu_values, red_values, segment_join_hu
        )

        # Compute RED for each material and evaluate deviation
        final_measurements = []
        deviations = []
        for m_dict in measurements:
            computed_red = self._compute_red_from_curve(
                m_dict["measured_mean_hu"], soft_params, bone_params, segment_join_hu
            )
            deviation = abs(computed_red - m_dict["reference_red"])
            passed = deviation <= self._ed_config.max_red_deviation
            deviations.append(deviation)

            final_measurements.append(EDMaterialMeasurement(
                material_name=m_dict["material_name"],
                nominal_hu=m_dict["nominal_hu"],
                measured_mean_hu=m_dict["measured_mean_hu"],
                measured_std_hu=m_dict["measured_std_hu"],
                reference_red=m_dict["reference_red"],
                computed_red=computed_red,
                red_deviation=deviation,
                passed=passed,
            ))

        # Generate dense curve samples
        hu_curve, red_curve = self._generate_curve_samples(
            soft_params, bone_params, segment_join_hu
        )

        max_dev = max(deviations) if deviations else 0.0
        mean_dev = float(np.mean(deviations)) if deviations else 0.0
        all_passed = all(m.passed for m in final_measurements)

        return EDCalibrationResult(
            acquisition_date=volumetric_result.acquisition_date,
            series_description=volumetric_result.series_description,
            scanner_id=scanner_id,
            measurements=final_measurements,
            soft_tissue_slope=soft_params["slope"],
            soft_tissue_intercept=soft_params["intercept"],
            soft_tissue_r_squared=soft_params["r_squared"],
            soft_tissue_hu_range=soft_params["hu_range"],
            bone_slope=bone_params["slope"],
            bone_intercept=bone_params["intercept"],
            bone_r_squared=bone_params["r_squared"],
            bone_hu_range=bone_params["hu_range"],
            segment_join_hu=segment_join_hu,
            hu_curve=hu_curve,
            red_curve=red_curve,
            max_red_deviation=max_dev,
            mean_red_deviation=mean_dev,
            all_passed=all_passed,
        )

    def _extract_material_hu(
        self,
        volumetric_result: "VolumetricQCResult",
        material_name: str,
        roi: "ROIDescriptor",
    ) -> tuple[float, float]:
        """Extracts mean and std HU for a material ROI from all slices."""
        means = []
        stds = []
        for hu_array in volumetric_result.hu_arrays:
            sub = hu_array[
                roi.row_start:roi.row_end,
                roi.col_start:roi.col_end,
            ].copy().astype(np.float64).ravel()
            means.append(float(np.mean(sub)))
            stds.append(float(np.std(sub, ddof=1)))  # ddof=1: Bessel's correction

        mean_hu = float(np.mean(means))
        std_hu = float(np.mean(stds))
        return mean_hu, std_hu

    def _fit_calibration_curve(
        self,
        hu_values: list[float],
        red_values: list[float],
        segment_join_hu: float = 100.0,
    ) -> tuple[dict, dict]:
        """Fits piecewise linear calibration curve (2 segments)."""
        hu_arr = np.array(hu_values, dtype=np.float64)
        red_arr = np.array(red_values, dtype=np.float64)

        # Split into soft tissue (HU < join) and bone (HU >= join)
        soft_mask = hu_arr < segment_join_hu
        bone_mask = hu_arr >= segment_join_hu

        soft_params = self._fit_segment(hu_arr[soft_mask], red_arr[soft_mask], "soft_tissue")
        bone_params = self._fit_segment(hu_arr[bone_mask], red_arr[bone_mask], "bone")

        return soft_params, bone_params

    def _fit_segment(
        self, hu_vals: np.ndarray, red_vals: np.ndarray, segment_name: str
    ) -> dict:
        """Fit one linear segment and compute R^2."""
        if len(hu_vals) < 2:
            logger.warning("Segment '%s' has fewer than 2 points, using defaults", segment_name)
            return {
                "slope": 0.001, "intercept": 1.0, "r_squared": 0.0,
                "hu_range": (0.0, 0.0),
            }

        coeffs = np.polyfit(hu_vals, red_vals, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        predicted = np.polyval(coeffs, hu_vals)
        ss_res = np.sum((red_vals - predicted) ** 2)
        ss_tot = np.sum((red_vals - np.mean(red_vals)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        return {
            "slope": slope, "intercept": intercept, "r_squared": r_squared,
            "hu_range": (float(np.min(hu_vals)), float(np.max(hu_vals))),
        }

    def _compute_red_from_curve(
        self, hu_value: float, soft_tissue_params: dict,
        bone_params: dict, segment_join_hu: float,
    ) -> float:
        """Evaluate piecewise linear curve at a given HU value."""
        if hu_value < segment_join_hu:
            return soft_tissue_params["slope"] * hu_value + soft_tissue_params["intercept"]
        else:
            return bone_params["slope"] * hu_value + bone_params["intercept"]

    def _generate_curve_samples(
        self, soft_tissue_params: dict, bone_params: dict,
        segment_join_hu: float,
        hu_min: float = -1000.0, hu_max: float = 1500.0, n_points: int = 250,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate dense (HU, RED) sampling for plotting and TPS export."""
        hu_array = np.linspace(hu_min, hu_max, n_points)
        red_array = np.array([
            self._compute_red_from_curve(h, soft_tissue_params, bone_params, segment_join_hu)
            for h in hu_array
        ])
        return hu_array, red_array

    def compare_to_reference(
        self, result: EDCalibrationResult, reference_result: EDCalibrationResult,
    ) -> dict:
        """Compare a new calibration against a reference baseline."""
        slope_soft_drift = result.soft_tissue_slope - reference_result.soft_tissue_slope
        slope_bone_drift = result.bone_slope - reference_result.bone_slope

        # Find materials whose deviation worsened by > 0.01
        ref_devs = {m.material_name: m.red_deviation for m in reference_result.measurements}
        materials_changed = []
        for m in result.measurements:
            ref_dev = ref_devs.get(m.material_name, 0.0)
            if m.red_deviation - ref_dev > 0.01:
                materials_changed.append(m.material_name)

        hardware_warning = abs(slope_soft_drift) > 0.005 or abs(slope_bone_drift) > 0.005

        return {
            "slope_soft_tissue_drift": slope_soft_drift,
            "slope_bone_drift": slope_bone_drift,
            "max_red_deviation_change": result.max_red_deviation - reference_result.max_red_deviation,
            "materials_changed": materials_changed,
            "hardware_warning": hardware_warning,
        }
