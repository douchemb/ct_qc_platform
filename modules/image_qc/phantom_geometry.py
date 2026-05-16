# -*- coding: utf-8 -*-
"""
modules/image_qc/phantom_geometry.py
=====================================
Automatic geometric detection and ROI placement for the GE Helios QA Phantom.

The GE Helios phantom contains 7 material inserts arranged at fixed angular
positions and a fixed radius from the phantom center, within a cylindrical
water-equivalent body. Their positions in pixel space depend on:
  1. The phantom center in the reconstructed image (not always 256,256)
  2. The phantom outer radius in pixels (depends on pixel spacing)
  3. The angular reference (top of phantom = insert 0 = Air)

This module auto-detects (1) and (2) from the HU image and computes all
ROI coordinates algebraically.

Reference: GE Helios QA Phantom User Manual, P/N 2165993-100 Rev 1,
           Section 3: Phantom Description and Insert Layout.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from config import AppConfig
from modules.image_qc.roi_stats import ROIDescriptor

logger = logging.getLogger(__name__)

__all__ = ["HeliosGeometry", "HeliosPhantomDetector"]


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HeliosGeometry:
    """
    Detected geometric parameters of the GE Helios QA phantom.

    center_row, center_col: sub-pixel center of the phantom in pixel coordinates.
    phantom_radius_px: outer radius of the phantom body in pixels.
    insert_radius_px: radius at which inserts are positioned (fixed ratio of outer radius).
    pixel_spacing_mm: (row, col) pixel spacing from DICOM, used for physical scaling.
    roi_descriptors: computed ROIDescriptor for each of the 7 named inserts.
    detection_quality: "good" | "degraded" | "failed"
      "good"     — center within 10 px of image center, radius plausible
      "degraded" — center detected but confidence low (phantom off-center > 10 px)
      "failed"   — detection failed, roi_descriptors are computed from nominal geometry
    """
    center_row: float
    center_col: float
    phantom_radius_px: float
    insert_radius_px: float
    pixel_spacing_mm: tuple[float, float]
    roi_descriptors: dict[str, ROIDescriptor]   # keyed by insert material name
    detection_quality: str
    detection_notes: str

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        d = {
            "center_row": self.center_row,
            "center_col": self.center_col,
            "phantom_radius_px": self.phantom_radius_px,
            "insert_radius_px": self.insert_radius_px,
            "pixel_spacing_mm": list(self.pixel_spacing_mm),
            "detection_quality": self.detection_quality,
            "detection_notes": self.detection_notes,
            "roi_descriptors": {
                k: {"label": v.label, "row_start": v.row_start,
                     "col_start": v.col_start, "height_px": v.height_px,
                     "width_px": v.width_px}
                for k, v in self.roi_descriptors.items()
            },
        }
        return d


# ═══════════════════════════════════════════════════════════════════
# Detector Class
# ═══════════════════════════════════════════════════════════════════

class HeliosPhantomDetector:
    """
    Detects the GE Helios QA phantom geometry from a CT HU image and
    computes the exact pixel coordinates of all 7 material insert ROIs.

    The Helios phantom geometry (GE P/N 2165993-100 Rev 1, Section 3):
      Outer diameter: 200 mm (radius = 100 mm)
      Insert diameter: 28 mm each (ROI diameter used: 20 mm for safety margin)
      Insert radial position: 58 mm from phantom center
      Insert angular positions (0° = top = 12 o'clock, clockwise):
        Air          :   0°
        Acrylic      :  60°
        LDPE         : 120°
        Polystyrene  : 180°
        Delrin       : 240°
        Teflon       : 300°
        Water (center):  — (at center, radius = 0)
    """

    # Angular positions of peripheral inserts in degrees from top (clockwise)
    # Reference: GE Helios QA Phantom Manual P/N 2165993-100 Rev 1, Figure 3-1
    INSERT_ANGLES_DEG: dict[str, float] = {
        "air":         0.0,
        "acrylic":    60.0,
        "ldpe":      120.0,
        "polystyrene": 180.0,
        "delrin":    240.0,
        "teflon":    300.0,
    }

    # Physical insert radius from phantom center — GE Helios manual §3.2
    INSERT_RADIUS_MM: float = 58.0

    # Phantom outer radius — GE Helios manual §3.1
    PHANTOM_OUTER_RADIUS_MM: float = 100.0

    # ROI size for statistical measurement (smaller than insert to avoid PVE)
    # Partial volume effect: stay 4 mm inside the insert boundary
    ROI_DIAMETER_MM: float = 20.0

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def detect(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> HeliosGeometry:
        """
        Primary entry point. Detects phantom geometry from a single HU image.
        Returns HeliosGeometry with all 7 ROIDescriptor objects populated.
        Falls back to nominal geometry (image center, nominal radius) if
        detection fails, rather than raising — clinical workflow must continue.
        """
        try:
            center_row, center_col = self._detect_phantom_center(hu_array)
            phantom_radius_px = self._detect_phantom_radius(
                hu_array, center_row, center_col, pixel_spacing_mm
            )
        except Exception as exc:
            logger.error(
                "Phantom center/radius detection failed (%s) — using nominal geometry", exc
            )
            center_row, center_col, phantom_radius_px = self._nominal_geometry(
                hu_array, pixel_spacing_mm
            )

        # Compute insert radius in pixels
        # insert_radius_px = INSERT_RADIUS_MM / pixel_spacing_mm
        insert_radius_px = self.INSERT_RADIUS_MM / pixel_spacing_mm[0]

        # ROI radius in pixels
        roi_radius_px = (self.ROI_DIAMETER_MM / 2.0) / pixel_spacing_mm[0]

        # Validate geometry
        quality, notes = self._validate_geometry(
            center_row, center_col, phantom_radius_px,
            hu_array.shape, pixel_spacing_mm
        )

        # If failed, substitute nominal geometry
        if quality == "failed":
            logger.error(
                "Geometry validation failed — substituting nominal geometry. Notes: %s", notes
            )
            center_row, center_col, phantom_radius_px = self._nominal_geometry(
                hu_array, pixel_spacing_mm
            )
            insert_radius_px = self.INSERT_RADIUS_MM / pixel_spacing_mm[0]

        # Compute ROI descriptors for all 7 inserts
        roi_descriptors = self._compute_roi_descriptors(
            center_row, center_col, insert_radius_px, roi_radius_px,
            pixel_spacing_mm, hu_array.shape
        )

        logger.info(
            "Helios geometry: center=(%.1f, %.1f), radius=%.1f px, quality=%s",
            center_row, center_col, phantom_radius_px, quality
        )

        return HeliosGeometry(
            center_row=center_row,
            center_col=center_col,
            phantom_radius_px=phantom_radius_px,
            insert_radius_px=insert_radius_px,
            pixel_spacing_mm=pixel_spacing_mm,
            roi_descriptors=roi_descriptors,
            detection_quality=quality,
            detection_notes=notes,
        )

    def detect_from_volume(
        self,
        volumetric_result: "VolumetricQCResult",
    ) -> HeliosGeometry:
        """
        Detects geometry from the middle slice of a VolumetricQCResult.
        The middle slice minimizes cone-beam artifacts at the phantom edges.
        """
        middle_idx = len(volumetric_result.hu_arrays) // 2
        return self.detect(
            volumetric_result.hu_arrays[middle_idx],
            volumetric_result.pixel_spacing_mm,
        )

    def _detect_phantom_center(
        self,
        hu_array: np.ndarray,
    ) -> tuple[float, float]:
        """
        Detects the sub-pixel center of the circular phantom body,
        IGNORING the patient couch/table.

        Algorithm:
        1. Threshold at HU = -500 to isolate phantom + couch from air
        2. cv2.findContours → filter by circularity > 0.7 to reject couch
        3. Select the largest circular contour (phantom)
        4. Compute centroid via cv2.moments

        Falls back to scipy connected-component analysis with bounding-box
        circularity proxy if OpenCV is unavailable.

        Reference: Gonzalez & Woods, Digital Image Processing, §11.1;
                   AAPM TG-66 §4 phantom positioning.
        """
        import math as _math
        from scipy import ndimage

        rows, cols = hu_array.shape
        mask = (hu_array > -500.0).astype(np.uint8) * 255

        # ── Try OpenCV contour approach ────────────────────────────────
        try:
            import cv2

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour, best_area = None, 0.0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1.0 or area < 1000:
                    continue
                circ = 4.0 * _math.pi * area / (perimeter * perimeter)
                if circ > 0.7 and area > best_area:
                    best_area = area
                    best_contour = cnt

            if best_contour is None and contours:
                best_contour = max(contours, key=cv2.contourArea)

            if best_contour is not None:
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    logger.debug(
                        "Phantom center (cv2): (%.2f, %.2f)", cy, cx)
                    return float(cy), float(cx)

        except ImportError:
            pass

        # ── Scipy fallback ─────────────────────────────────────────────
        binary = mask > 0
        filled = ndimage.binary_fill_holes(binary)
        labeled, n_labels = ndimage.label(filled)

        best_label, best_area = None, 0
        for lbl in range(1, n_labels + 1):
            comp = labeled == lbl
            area = int(comp.sum())
            if area < 1000:
                continue
            rr, cc = np.where(comp)
            bbox_h = rr.max() - rr.min() + 1
            bbox_w = cc.max() - cc.min() + 1
            aspect = min(bbox_h, bbox_w) / max(bbox_h, bbox_w)
            fill_ratio = area / (bbox_h * bbox_w)
            if aspect > 0.7 and fill_ratio > 0.5 and area > best_area:
                best_area = area
                best_label = lbl

        if best_label is None:
            if n_labels > 0:
                sizes = ndimage.sum(filled, labeled, range(1, n_labels + 1))
                best_label = int(np.argmax(sizes)) + 1
            else:
                logger.warning("No phantom detected — using image center.")
                return float(rows // 2), float(cols // 2)

        center = ndimage.center_of_mass((labeled == best_label).astype(float))
        logger.debug(
            "Phantom center detected: (%.2f, %.2f)", center[0], center[1]
        )
        return float(center[0]), float(center[1])

    def _detect_phantom_radius(
        self,
        hu_array: np.ndarray,
        center_row: float,
        center_col: float,
        pixel_spacing_mm: tuple[float, float],
    ) -> float:
        """
        Detects the phantom outer radius in pixels using a radial HU profile.

        Algorithm:
        1. Sample the HU profile along 36 radial directions (every 10°)
        2. For each direction, find the radius where HU drops below -200
        3. Take the median across all 36 directions
        4. Return median radius in pixels
        """
        n_directions = 36
        max_radius_px = int(min(hu_array.shape) // 2)
        radii_found = []

        for k in range(n_directions):
            angle_rad = math.radians(k * 360.0 / n_directions)
            # Direction vector
            dr = -math.cos(angle_rad)  # row decreases upward
            dc = math.sin(angle_rad)

            # Sample along this direction using vectorized indexing
            radii = np.arange(1, max_radius_px)
            rows = (center_row + radii * dr).astype(int)
            cols = (center_col + radii * dc).astype(int)

            # Bounds check
            valid = (
                (rows >= 0) & (rows < hu_array.shape[0])
                & (cols >= 0) & (cols < hu_array.shape[1])
            )
            if not np.any(valid):
                continue

            rows_valid = rows[valid]
            cols_valid = cols[valid]
            radii_valid = radii[valid]
            hu_profile = hu_array[rows_valid, cols_valid]

            # Find where HU drops below -200 (phantom body to air)
            below_threshold = np.where(hu_profile < -200.0)[0]
            if len(below_threshold) > 0:
                radii_found.append(float(radii_valid[below_threshold[0]]))

        if not radii_found:
            # Fallback to nominal radius
            logger.warning("Radial edge detection failed — using nominal radius")
            return self.PHANTOM_OUTER_RADIUS_MM / pixel_spacing_mm[0]

        median_radius = float(np.median(radii_found))
        logger.debug("Phantom radius detected: %.1f px (median of %d directions)",
                     median_radius, len(radii_found))
        return median_radius

    def _compute_roi_descriptors(
        self,
        center_row: float,
        center_col: float,
        insert_radius_px: float,
        roi_radius_px: float,
        pixel_spacing_mm: tuple[float, float],
        image_shape: tuple[int, int],
    ) -> dict[str, ROIDescriptor]:
        """
        Computes ROIDescriptor for all 7 Helios inserts using the detected geometry.

        For peripheral inserts (6 inserts at INSERT_RADIUS_MM from center):
          row = center_row - insert_radius_px × cos(angle_rad)
          col = center_col + insert_radius_px × sin(angle_rad)

        For the water insert (center):
          row = center_row, col = center_col

        All ROIs are bounds-checked against image_shape.
        """
        roi_size_px = int(2 * roi_radius_px)
        descriptors: dict[str, ROIDescriptor] = {}

        # Water insert at center
        water_row_start = int(center_row - roi_radius_px)
        water_col_start = int(center_col - roi_radius_px)

        if (water_row_start >= 0 and water_col_start >= 0
                and water_row_start + roi_size_px <= image_shape[0]
                and water_col_start + roi_size_px <= image_shape[1]):
            descriptors["water"] = ROIDescriptor(
                label="water",
                row_start=water_row_start,
                col_start=water_col_start,
                height_px=roi_size_px,
                width_px=roi_size_px,
            )
        else:
            logger.warning("Water ROI falls outside image bounds — excluded")

        # Peripheral inserts
        for material, angle_deg in self.INSERT_ANGLES_DEG.items():
            angle_rad = math.radians(angle_deg)
            # Note: angle 0° = top = 12 o'clock direction
            # cos is subtracted from row (row increases downward in image space)
            # sin is added to col (col increases rightward)
            insert_row = center_row - insert_radius_px * math.cos(angle_rad)
            insert_col = center_col + insert_radius_px * math.sin(angle_rad)

            row_start = int(insert_row - roi_radius_px)
            col_start = int(insert_col - roi_radius_px)

            if (row_start >= 0 and col_start >= 0
                    and row_start + roi_size_px <= image_shape[0]
                    and col_start + roi_size_px <= image_shape[1]):
                descriptors[material] = ROIDescriptor(
                    label=material,
                    row_start=row_start,
                    col_start=col_start,
                    height_px=roi_size_px,
                    width_px=roi_size_px,
                )
            else:
                logger.warning(
                    "ROI '%s' at (%.1f, %.1f) falls outside image bounds — excluded",
                    material, insert_row, insert_col
                )

        return descriptors

    def _validate_geometry(
        self,
        center_row: float,
        center_col: float,
        phantom_radius_px: float,
        image_shape: tuple[int, int],
        pixel_spacing_mm: tuple[float, float],
    ) -> tuple[str, str]:
        """
        Validates detected geometry and returns (quality, notes).

        quality = "good" if radius within 10% and center within 30 px.
        quality = "degraded" if radius within 20% or center offset 30-60 px.
        quality = "failed" if radius outside 20% or center offset > 60 px.
        """
        notes_parts = []

        # Expected radius in pixels
        expected_radius_px = self.PHANTOM_OUTER_RADIUS_MM / pixel_spacing_mm[0]
        radius_deviation = abs(phantom_radius_px - expected_radius_px) / expected_radius_px

        # Center offset from image center
        image_center_row = image_shape[0] / 2.0
        image_center_col = image_shape[1] / 2.0
        center_offset = math.sqrt(
            (center_row - image_center_row) ** 2
            + (center_col - image_center_col) ** 2
        )

        notes_parts.append(
            "radius_deviation=%.1f%%, center_offset=%.1f px" % (
                radius_deviation * 100, center_offset
            )
        )

        # Determine quality
        if radius_deviation > 0.20 or center_offset > 60.0:
            quality = "failed"
            notes_parts.append("FAILED: geometry outside acceptable tolerance")
        elif radius_deviation > 0.10 or center_offset > 30.0:
            quality = "degraded"
            notes_parts.append("DEGRADED: geometry marginally acceptable")
        else:
            quality = "good"
            notes_parts.append("GOOD: geometry within tolerance")

        notes = "; ".join(notes_parts)
        return quality, notes

    def _nominal_geometry(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> tuple[float, float, float]:
        """
        Returns (center_row, center_col, phantom_radius_px) using nominal values.
        Used as fallback when detection fails.
        center = image center (rows//2, cols//2)
        radius = PHANTOM_OUTER_RADIUS_MM / pixel_spacing_mm[0]
        """
        rows, cols = hu_array.shape
        center_row = rows / 2.0
        center_col = cols / 2.0
        radius_px  = self.PHANTOM_OUTER_RADIUS_MM / pixel_spacing_mm[0]
        return center_row, center_col, radius_px
