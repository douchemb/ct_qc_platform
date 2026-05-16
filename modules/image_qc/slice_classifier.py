# -*- coding: utf-8 -*-
"""
modules/image_qc/slice_classifier.py — Helios Slice Classifier.

Routes DICOM slices using a two-tier strategy:

  1. REGEX INTERVAL (primary) — parse "Image X" from filename, group by
     GROUND-TRUTH intervals verified against Image Owl TotalQA:
       30–50  → Uniformity / Noise (Pure Water)
       55–65  → Contrast (Plastic Block with Horizontal Holes)
       66–75  → Resolution / Bar Patterns (Angled Bars)

  2. Z-POSITION (secondary) — SliceLocation / ImagePositionPatient[2]

  NOTE: HU heuristic fallback has been REMOVED — it caused misrouting.

Reference: Image Owl TotalQA GE Set 1; AAPM TG-233.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "HeliosSliceClassifier",
    "SliceClassificationResult",
]

# ── Regex for GE Helios filenames ─────────────────────────────────
# Matches: "CT.TPSQA2017.Image 55.dcm", "Image 55.dcm", "Image55.dcm"
_IMAGE_NUM_RE = re.compile(r"Image\s*(\d+)", re.IGNORECASE)

# ── Ground-Truth Intervals (aligned to Image Owl TotalQA) ────────
_UNIFORMITY_RANGE = (30, 50)    # Pure Water / Noise / Uniformity
_CONTRAST_RANGE   = (55, 65)    # Plastic Block w/ Horizontal Holes
_RESOLUTION_RANGE = (66, 75)    # Angled Bar Patterns


# ═══════════════════════════════════════════════════════════════════
# Classification Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SliceClassificationResult:
    """Result of volume-level slice classification."""

    water_slice_index: Optional[int] = None
    resolution_slice_index: Optional[int] = None
    sensitometry_slice_index: Optional[int] = None

    water_hu_array: Optional[np.ndarray] = None
    resolution_hu_array: Optional[np.ndarray] = None
    sensitometry_hu_array: Optional[np.ndarray] = None

    water_candidate_indices: list[int] = field(default_factory=list)
    routing_method: str = "none"
    classification_log: list[str] = field(default_factory=list)

    @property
    def has_water(self) -> bool:
        return self.water_slice_index is not None

    @property
    def has_resolution(self) -> bool:
        return self.resolution_slice_index is not None

    @property
    def has_sensitometry(self) -> bool:
        return self.sensitometry_slice_index is not None

    def is_water_candidate(self, idx: int) -> bool:
        return idx in self.water_candidate_indices

    def summary(self) -> str:
        parts = [f"Routing={self.routing_method}"]
        if self.has_water:
            parts.append(f"Water=#{self.water_slice_index + 1}")
        else:
            parts.append("Water=N/A")
        if self.has_resolution:
            parts.append(f"Resolution=#{self.resolution_slice_index + 1}")
        else:
            parts.append("Resolution=N/A")
        if self.has_sensitometry:
            parts.append(f"Contrast=#{self.sensitometry_slice_index + 1}")
        else:
            parts.append("Contrast=N/A")
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════

class HeliosSliceClassifier:
    """Two-tier slice classifier: Regex → Z-position. NO HU fallback."""

    # Z-position defaults (secondary strategy)
    DEFAULT_WATER_Z    = 60.0
    DEFAULT_MTF_Z      = 50.0
    DEFAULT_CONTRAST_Z = 40.0
    DEFAULT_Z_TOL      =  5.0

    def __init__(
        self,
        loader,
        *,
        water_z_mm: float = DEFAULT_WATER_Z,
        mtf_z_mm: float = DEFAULT_MTF_Z,
        contrast_z_mm: float = DEFAULT_CONTRAST_Z,
        z_tolerance_mm: float = DEFAULT_Z_TOL,
        mtf_max_hu_threshold: float = 1500.0,
        water_mean_hu_threshold: float = 20.0,
        water_std_hu_threshold: float = 30.0,
        sensitometry_std_threshold: float = 100.0,
    ) -> None:
        self._loader = loader
        self._water_z = water_z_mm
        self._mtf_z = mtf_z_mm
        self._contrast_z = contrast_z_mm
        self._z_tol = z_tolerance_mm
        self._mtf_thr = mtf_max_hu_threshold
        self._water_mean_thr = water_mean_hu_threshold
        self._water_std_thr = water_std_hu_threshold
        self._sensi_std_thr = sensitometry_std_threshold

    # ── Public API ────────────────────────────────────────────────

    def classify(self, datasets: list) -> SliceClassificationResult:
        """Classify slices: regex intervals → Z-position. NO HU fallback."""
        if not datasets:
            r = SliceClassificationResult()
            r.classification_log.append("No datasets — skipped.")
            return r

        # Strategy 1: Regex interval routing (ground-truth)
        result = self._classify_by_regex(datasets)
        if result is not None:
            return result

        # Strategy 2: Z-position routing
        result = self._classify_by_z(datasets)
        if result is not None:
            return result

        # NO Strategy 3 — HU heuristic fallback REMOVED (caused misrouting)
        r = SliceClassificationResult(routing_method="no_match")
        r.classification_log.append(
            "❌ No regex match and no Z-position match. "
            "HU fallback disabled to prevent misrouting.")
        return r

    # ── Strategy 1: Regex Interval ────────────────────────────────

    @staticmethod
    def _extract_image_number(ds) -> Optional[int]:
        """Extract integer X from 'CT.TPSQA2017.Image X.dcm'."""
        filename = str(getattr(ds, "filename", ""))
        basename = Path(filename).name if filename else ""
        m = _IMAGE_NUM_RE.search(basename)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _in_range(x: int, rng: tuple[int, int]) -> bool:
        return rng[0] <= x <= rng[1]

    def _classify_by_regex(self, datasets: list) -> Optional[SliceClassificationResult]:
        """Parse 'Image X' from filenames, group by known intervals."""
        result = SliceClassificationResult(routing_method="regex_interval")

        uniformity: list[tuple[int, int]] = []  # (dataset_idx, image_num)
        contrast:   list[tuple[int, int]] = []
        resolution: list[tuple[int, int]] = []
        unmatched:  list[tuple[int, int]] = []

        n_parsed = 0
        for idx, ds in enumerate(datasets):
            img_num = self._extract_image_number(ds)
            if img_num is None:
                continue
            n_parsed += 1

            if self._in_range(img_num, _UNIFORMITY_RANGE):
                uniformity.append((idx, img_num))
            elif self._in_range(img_num, _CONTRAST_RANGE):
                contrast.append((idx, img_num))
            elif self._in_range(img_num, _RESOLUTION_RANGE):
                resolution.append((idx, img_num))
            else:
                unmatched.append((idx, img_num))

        if n_parsed == 0:
            result.classification_log.append(
                "⚠️ Regex: no 'Image X' pattern found in filenames — skipping.")
            return None

        result.classification_log.append(
            f"📂 Regex interval routing: parsed {n_parsed}/{len(datasets)} filenames.")
        result.classification_log.append(
            f"   Uniformity[{_UNIFORMITY_RANGE[0]}-{_UNIFORMITY_RANGE[1]}]: "
            f"{len(uniformity)} slice(s)")
        result.classification_log.append(
            f"   Contrast[{_CONTRAST_RANGE[0]}-{_CONTRAST_RANGE[1]}]:    "
            f"{len(contrast)} slice(s)")
        result.classification_log.append(
            f"   Resolution[{_RESOLUTION_RANGE[0]}-{_RESOLUTION_RANGE[1]}]:  "
            f"{len(resolution)} slice(s)")
        if unmatched:
            nums = ", ".join(str(u[1]) for u in unmatched[:10])
            result.classification_log.append(
                f"   Unmatched: {len(unmatched)} slice(s) [{nums}...]")

        # Select middle slice from each group
        def _select_middle(group, label, attr_idx, attr_hu):
            if not group:
                result.classification_log.append(
                    f"⚠️ {label}: no slices in range — N/A.")
                return
            group.sort(key=lambda t: t[1])
            mid = group[len(group) // 2]
            ds_idx, img_num = mid
            try:
                hu = self._loader.to_hu_array(datasets[ds_idx])
                setattr(result, attr_idx, ds_idx)
                setattr(result, attr_hu, hu)
                result.classification_log.append(
                    f"✅ {label}: Image {img_num} "
                    f"(dataset #{ds_idx + 1}, mid of {len(group)})")
            except Exception as exc:
                result.classification_log.append(
                    f"❌ {label}: Image {img_num} HU conversion failed: {exc}")

        _select_middle(uniformity, "Water/Uniformity",
                       "water_slice_index", "water_hu_array")
        _select_middle(contrast, "Contrast (Plastic Block)",
                       "sensitometry_slice_index", "sensitometry_hu_array")
        _select_middle(resolution, "Resolution (Bar Patterns)",
                       "resolution_slice_index", "resolution_hu_array")

        # All uniformity slices are water candidates
        result.water_candidate_indices = [g[0] for g in uniformity]

        n_matched = sum([result.has_water, result.has_resolution,
                         result.has_sensitometry])
        if n_matched == 0:
            return None

        result.classification_log.append(
            f"📂 Regex routing complete: {n_matched}/3 groups populated.")
        logger.info("HeliosSliceClassifier (regex): %s", result.summary())
        return result

    # ── Strategy 2: Z-Position ────────────────────────────────────

    def _classify_by_z(self, datasets: list) -> Optional[SliceClassificationResult]:
        """Match SliceLocation / ImagePositionPatient[2] to known Z targets."""
        result = SliceClassificationResult(routing_method="z_position")

        z_pos: list[tuple[int, float]] = []
        for idx, ds in enumerate(datasets):
            z = self._get_z(ds)
            if z is not None:
                z_pos.append((idx, z))

        if not z_pos:
            return None

        result.classification_log.append(
            f"📐 Z-position routing: {len(z_pos)}/{len(datasets)} slices have Z tags.")

        targets = [
            ("Water", self._water_z, "water_slice_index", "water_hu_array"),
            ("MTF", self._mtf_z, "mtf_slice_index", "mtf_hu_array"),
            ("Contrast", self._contrast_z, "sensitometry_slice_index", "sensitometry_hu_array"),
        ]

        for label, target_z, attr_idx, attr_hu in targets:
            match = self._closest_z(z_pos, target_z, self._z_tol)
            if match:
                idx, z = match
                try:
                    hu = self._loader.to_hu_array(datasets[idx])
                    setattr(result, attr_idx, idx)
                    setattr(result, attr_hu, hu)
                    result.classification_log.append(
                        f"✅ {label}: #{idx + 1} (Z={z:.2f}, ΔZ={abs(z - target_z):.2f})")
                except Exception as exc:
                    result.classification_log.append(f"❌ {label}: HU error: {exc}")
            else:
                result.classification_log.append(f"⚠️ {label}: no Z match.")

        if result.has_water:
            result.water_candidate_indices = [
                i for i, z in z_pos if abs(z - self._water_z) <= self._z_tol]

        n = sum([result.has_water, result.has_resolution, result.has_sensitometry])
        return result if n > 0 else None

    @staticmethod
    def _get_z(ds) -> Optional[float]:
        for attr in ("SliceLocation", "ImagePositionPatient"):
            val = getattr(ds, attr, None)
            if val is None:
                continue
            try:
                return float(val[2]) if attr == "ImagePositionPatient" else float(val)
            except (TypeError, ValueError, IndexError):
                continue
        return None

    @staticmethod
    def _closest_z(z_pos, target, tol):
        hits = [(i, z) for i, z in z_pos if abs(z - target) <= tol]
        return min(hits, key=lambda t: abs(t[1] - target)) if hits else None

    # ── Strategy 3: HU Fallback — REMOVED ─────────────────────────
    # HU-based heuristic classification has been permanently removed.
    # It caused critical misrouting (uniformity ROIs on resolution block).
    # Ground-truth intervals from DICOM video analysis are the sole source
    # of truth. Z-position is kept only as a secondary fallback.
