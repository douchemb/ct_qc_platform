"""
modules/image_qc/phantom_adapters.py
======================================
Phantom Adapter Pattern — Vendor-Agnostic ROI Placement.

The Adapter pattern isolates all phantom-specific geometry and material
reference data behind a common interface (PhantomAdapter ABC). All
downstream modules (BasicMetricsEngine, AdvancedMetricsEngine, app.py)
call get_roi_descriptors() and receive correctly placed ROIDescriptor
objects without knowing which phantom they are analyzing.

Supported phantoms:
  - Siemens Water Phantom (standard cylindrical, 5-ROI layout)
  - GE Helios QA Phantom (7 material inserts at fixed angles)
  - Generic (universal fallback, 5-ROI layout)

Phantom detection uses SeriesDescription and ProtocolName DICOM tags,
configured via phantom_profiles.yaml (no hardcoding).

Reference: GE Helios QA Phantom Manual P/N 2165993-100 Rev 1 §3;
           Siemens CT QA Protocol Reference Guide;
           AAPM TG-66 Section 5 — phantom measurement geometry.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage
import yaml

from modules.image_qc.roi_stats import ROIDescriptor

logger = logging.getLogger(__name__)

_PHANTOM_PROFILES_YAML = Path(__file__).parent.parent.parent / "phantom_profiles.yaml"


# ── Custom exceptions ──────────────────────────────────────────────────────

class PhantomDetectionError(RuntimeError):
    """Raised when phantom center or radius detection fails critically."""


class PhantomProfileError(RuntimeError):
    """Raised when phantom_profiles.yaml cannot be loaded."""


class ROIOutOfBoundsError(ValueError):
    """
    Raised when a computed ROI position falls outside the image boundaries.
    Contains the ROI label, computed center, and image shape for diagnosis.
    """


# ── Detection result ───────────────────────────────────────────────────────

@dataclass
class PhantomDetectionResult:
    """
    Geometric parameters detected from the CT image for one phantom type.
    All coordinates are in pixel space (not mm).
    """
    center_row: float
    center_col: float
    phantom_radius_px: float
    pixel_spacing_mm: tuple[float, float]
    detection_quality: str      # "good" | "degraded" | "failed"
    detection_notes: str        # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "center_row":        self.center_row,
            "center_col":        self.center_col,
            "phantom_radius_px": self.phantom_radius_px,
            "pixel_spacing_mm":  list(self.pixel_spacing_mm),
            "detection_quality": self.detection_quality,
            "detection_notes":   self.detection_notes,
        }


# ── Material reference ─────────────────────────────────────────────────────

@dataclass
class MaterialReference:
    """
    Physical reference values for one phantom insert material.
    Source: GE Helios manual, Siemens QA guide, IAEA TRS-430 Table 4.1,
            Schneider et al. Phys Med Biol 41(1) 1996.
    """
    name: str
    nominal_hu: float       # expected CT number in HU
    red: float              # relative electron density (water = 1.000)
    hu_tolerance: float     # TG-66 Section 5.2: typically ±4 HU


# ── Abstract base class ────────────────────────────────────────────────────

class PhantomAdapter(ABC):
    """
    Abstract interface for all phantom geometry adapters.

    Concrete subclasses implement get_roi_descriptors() to return
    correctly positioned ROIs for their specific phantom geometry.
    All downstream modules depend only on this interface.
    """

    @abstractmethod
    def get_roi_descriptors(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """
        Returns a dict mapping material/ROI name to ROIDescriptor.
        Positions are computed from the detected phantom center in the image.
        Falls back to nominal center if detection fails — never raises.
        """
        ...

    @abstractmethod
    def get_material_references(self) -> dict[str, MaterialReference]:
        """Returns nominal HU and RED values for all phantom materials."""
        ...

    @abstractmethod
    def has_edge_insert(self) -> bool:
        """True if the phantom has an edge insert suitable for MTF measurement."""
        ...

    @abstractmethod
    def has_density_inserts(self) -> bool:
        """True if the phantom has multiple density inserts for ED calibration."""
        ...

    # ── Slice-Aware ROI Generation ─────────────────────────────────────
    # These methods return ROI subsets appropriate for each classified
    # slice type.  Concrete adapters override as needed.

    def get_water_rois(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Returns the 5 clock-face uniformity ROIs (center + 12/3/6/9).

        Positions are relative to the dynamically detected phantom center.
        Default implementation: standard 5-ROI water layout at 60 mm radius.
        """
        image_shape = hu_array.shape
        c_row, c_col = self._detect_center_safe(hu_array, image_shape)
        layout = {
            "center_water": (0.0,   0.0, 30.0),
            "peripheral_12": (60.0,   0.0, 20.0),
            "peripheral_3":  (60.0,  90.0, 20.0),
            "peripheral_6":  (60.0, 180.0, 20.0),
            "peripheral_9":  (60.0, 270.0, 20.0),
        }
        descriptors: dict[str, ROIDescriptor] = {}
        for label, (r_mm, angle, diam) in layout.items():
            try:
                descriptors[label] = self._build_roi_from_polar(
                    label, c_row, c_col, r_mm, angle, diam,
                    pixel_spacing_mm, image_shape,
                )
            except ROIOutOfBoundsError as exc:
                logger.warning("Water ROI '%s' out of bounds — skipped: %s", label, exc)
        return descriptors

    def get_mtf_roi(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Returns a tight ROI bounding the MTF wire/edge insert.

        Default implementation: locates the pixel with the highest HU value
        in the image (tungsten wire) and places a 5×5 pixel ROI around it.
        Returns an empty dict if max HU < 1500 (no wire detected).
        """
        max_hu = float(np.max(hu_array))
        if max_hu < 1500.0:
            logger.info("get_mtf_roi: max HU=%.0f < 1500 — no wire detected.", max_hu)
            return {}

        # Locate max-HU pixel (tungsten wire)
        max_pos = np.unravel_index(np.argmax(hu_array), hu_array.shape)
        wire_row, wire_col = int(max_pos[0]), int(max_pos[1])
        roi_half = 2  # 5×5 pixel ROI
        row_start = max(0, wire_row - roi_half)
        col_start = max(0, wire_col - roi_half)
        row_end = min(hu_array.shape[0], wire_row + roi_half + 1)
        col_end = min(hu_array.shape[1], wire_col + roi_half + 1)

        return {
            "mtf_wire": ROIDescriptor(
                label="mtf_wire",
                row_start=row_start,
                col_start=col_start,
                height_px=row_end - row_start,
                width_px=col_end - col_start,
            )
        }

    def get_sensitometry_rois(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Returns ROIs for sensitometry inserts (density materials).

        Default implementation returns empty dict — only phantoms with
        density inserts (e.g. GE Helios) override this.
        """
        return {}

    @property
    @abstractmethod
    def phantom_id(self) -> str:
        """Unique identifier matching phantom_profiles.yaml profile key."""
        ...

    def _detect_center_safe(
        self,
        hu_array: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[float, float]:
        """Returns the true geometric center of the phantom body.

        Algorithm (couch-safe):
          1. Threshold at HU > -300 (phantom body + couch > -300, air < -300)
          2. Fill internal holes (air inserts, wire channels)
          3. Label connected components
          4. Filter by circularity: aspect_ratio > 0.7 AND fill_ratio > 0.5
             → phantom is circular (aspect ~1.0, fill ~0.78)
             → couch is elongated (aspect ~0.1-0.3)
          5. Select the largest circular component
          6. Return center_of_mass of that component only

        Falls back to hardcoded (254, 254) if detection fails.
        """
        try:
            mask = hu_array > -300
            if not np.any(mask):
                logger.warning("Phantom mask empty — using (254, 254).")
                return 254.0, 254.0

            filled = scipy.ndimage.binary_fill_holes(mask)
            labeled, n_labels = scipy.ndimage.label(filled)

            if n_labels == 0:
                return 254.0, 254.0

            best_label = None
            best_area = 0

            for lbl in range(1, n_labels + 1):
                component = labeled == lbl
                area = int(component.sum())
                if area < 1000:
                    continue

                rr, cc = np.where(component)
                bbox_h = rr.max() - rr.min() + 1
                bbox_w = cc.max() - cc.min() + 1

                aspect = min(bbox_h, bbox_w) / max(bbox_h, bbox_w)
                fill_ratio = area / (bbox_h * bbox_w)

                # Phantom: aspect ~1.0, fill ~0.78 (circle in bbox)
                # Couch: aspect ~0.1-0.3, elongated
                if aspect > 0.7 and fill_ratio > 0.5 and area > best_area:
                    best_area = area
                    best_label = lbl

            if best_label is None:
                # No circular component — use largest as fallback
                sizes = scipy.ndimage.sum(
                    filled, labeled, range(1, n_labels + 1))
                best_label = int(np.argmax(sizes)) + 1

            component_mask = (labeled == best_label).astype(float)
            com = scipy.ndimage.center_of_mass(component_mask)
            cy, cx = float(com[0]), float(com[1])
            logger.debug(
                "Phantom center (circular component %d): (%.1f, %.1f), "
                "area=%d px",
                best_label, cy, cx, best_area,
            )
            return cy, cx

        except Exception as exc:
            logger.warning(
                "Phantom center detection failed (%s). "
                "Using hardcoded center (254, 254).", exc,
            )
            return 254.0, 254.0

    def detect_phantom_center(
        self,
        hu_array: np.ndarray,
        hu_threshold: float = -500.0,
    ) -> tuple[float, float]:
        """
        Detects the sub-pixel center of the phantom body using contour-based
        circularity filtering to isolate the circular phantom and IGNORE
        the patient couch/table.

        Algorithm:
          1. Threshold at hu_threshold to isolate phantom + couch from air
          2. Convert to uint8 binary mask for OpenCV
          3. cv2.findContours to extract all object boundaries
          4. Filter contours by circularity (4π·area/perimeter² > 0.7)
             → the phantom is circular, the couch is elongated
          5. Select the largest circular contour (by area)
          6. Compute centroid via cv2.moments on the winning contour

        Falls back to scipy.ndimage.label if cv2 is unavailable, or to
        image center if all detection fails.

        Reference: Gonzalez & Woods, Digital Image Processing §11.1;
                   AAPM TG-66 Section 4 — phantom positioning verification.
        """
        rows, cols = hu_array.shape
        mask = (hu_array > hu_threshold).astype(np.uint8) * 255

        try:
            import cv2
            return self._detect_center_cv2(mask, rows, cols)
        except ImportError:
            logger.debug("OpenCV not available — using scipy fallback.")
            return self._detect_center_scipy(mask, rows, cols)

    def _detect_center_cv2(
        self,
        mask: np.ndarray,
        rows: int,
        cols: int,
    ) -> tuple[float, float]:
        """Contour-based center detection using OpenCV."""
        import cv2

        # Fill holes so internal air inserts don't fragment the phantom
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning("No contours found — falling back to image center.")
            return rows / 2.0, cols / 2.0

        # Filter by circularity and pick the largest circular contour
        best_contour = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1.0 or area < 1000:
                continue  # skip tiny noise contours

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            # Phantom is circular (circularity ~0.85-1.0)
            # Couch is elongated (circularity ~0.1-0.3)
            if circularity > 0.7 and area > best_area:
                best_area = area
                best_contour = cnt

        if best_contour is None:
            # No circular contour found — try largest contour as fallback
            logger.warning(
                "No circular contour (circularity>0.7) found among %d contours. "
                "Using largest contour.", len(contours)
            )
            best_contour = max(contours, key=cv2.contourArea)

        # Compute centroid from moments
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]  # col
            cy = M["m01"] / M["m00"]  # row
            logger.debug(
                "Phantom center (cv2 contour): (%.1f, %.1f), "
                "circularity=%.3f, area=%.0f px",
                cy, cx,
                4.0 * math.pi * cv2.contourArea(best_contour)
                / (cv2.arcLength(best_contour, True) ** 2),
                cv2.contourArea(best_contour),
            )
            return float(cy), float(cx)  # (row, col)

        logger.warning("Contour moments zero — falling back to image center.")
        return rows / 2.0, cols / 2.0

    def _detect_center_scipy(
        self,
        mask: np.ndarray,
        rows: int,
        cols: int,
    ) -> tuple[float, float]:
        """Connected-component fallback when OpenCV is unavailable.

        Labels connected components, filters by circularity using
        bounding-box aspect ratio and fill ratio, picks the largest
        circular component.
        """
        binary = mask > 0
        filled = scipy.ndimage.binary_fill_holes(binary)
        labeled, n_labels = scipy.ndimage.label(filled)

        if n_labels == 0:
            logger.warning("No components found — using image center.")
            return rows / 2.0, cols / 2.0

        best_label = None
        best_area = 0

        for lbl in range(1, n_labels + 1):
            component = labeled == lbl
            area = int(component.sum())
            if area < 1000:
                continue

            # Bounding box
            rr, cc = np.where(component)
            bbox_h = rr.max() - rr.min() + 1
            bbox_w = cc.max() - cc.min() + 1

            # Circularity proxies:
            # 1. Aspect ratio close to 1.0 (square bounding box)
            aspect = min(bbox_h, bbox_w) / max(bbox_h, bbox_w)
            # 2. Fill ratio close to π/4 ≈ 0.785 (circle fills ~78.5% of bbox)
            fill_ratio = area / (bbox_h * bbox_w)

            # Phantom: aspect ~1.0, fill ~0.78
            # Couch: aspect ~0.1-0.3, fill varies
            if aspect > 0.7 and fill_ratio > 0.5 and area > best_area:
                best_area = area
                best_label = lbl

        if best_label is None:
            # No circular component — use the largest
            logger.warning("No circular component found — using largest.")
            sizes = scipy.ndimage.sum(filled, labeled, range(1, n_labels + 1))
            best_label = int(np.argmax(sizes)) + 1

        # Center of mass on the selected component only
        component_mask = (labeled == best_label).astype(float)
        center = scipy.ndimage.center_of_mass(component_mask)
        logger.debug(
            "Phantom center (scipy label %d): (%.1f, %.1f), area=%d px",
            best_label, center[0], center[1], best_area,
        )
        return float(center[0]), float(center[1])

    def detect_phantom_radius(
        self,
        hu_array: np.ndarray,
        center_row: float,
        center_col: float,
        pixel_spacing_mm: tuple[float, float],
        n_directions: int = 36,
    ) -> float:
        """
        Detects phantom outer radius in pixels using radial HU profiles.

        Samples 36 radial directions (every 10°). For each direction,
        finds the radius where HU drops below -200 HU (phantom → air).
        Returns the median radius across all directions (robust to inserts).

        Returns radius in pixels.
        """
        rows, cols = hu_array.shape
        max_r = min(center_row, center_col,
                    rows - center_row, cols - center_col) * 0.95
        radii = np.arange(1, int(max_r), 1)
        angles = np.linspace(0, 2 * math.pi, n_directions, endpoint=False)
        boundary_radii = []

        for angle in angles:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # Sample HU values along this radial direction
            r_vals = center_row + radii * sin_a
            c_vals = center_col + radii * cos_a
            # Clip to valid image indices
            valid = (
                (r_vals >= 0) & (r_vals < rows - 1) &
                (c_vals >= 0) & (c_vals < cols - 1)
            )
            r_idx = np.clip(r_vals[valid].astype(int), 0, rows - 1)
            c_idx = np.clip(c_vals[valid].astype(int), 0, cols - 1)
            hu_profile = hu_array[r_idx, c_idx]

            # Find first index where HU drops below -200 (air boundary)
            below = np.where(hu_profile < -200.0)[0]
            if len(below) > 0:
                boundary_radii.append(float(radii[valid][below[0]]))

        if not boundary_radii:
            logger.warning(
                "Phantom radius detection: no clear boundary found in %d directions. "
                "Using nominal radius.", n_directions
            )
            return 100.0  # nominal fallback in pixels

        return float(np.median(boundary_radii))

    def _build_roi_from_polar(
        self,
        label: str,
        center_row: float,
        center_col: float,
        radius_mm: float,
        angle_deg: float,
        roi_diameter_mm: float,
        pixel_spacing_mm: tuple[float, float],
        image_shape: tuple[int, int],
    ) -> ROIDescriptor:
        """
        Constructs a ROIDescriptor from polar coordinates relative to
        the phantom center. Angles measured clockwise from top (12 o'clock).

        Coordinate transform (image space: row increases downward):
          row = center_row - radius_px × cos(angle_rad)   [top = angle 0°]
          col = center_col + radius_px × sin(angle_rad)

        Raises ROIOutOfBoundsError if the ROI falls outside the image.
        """
        # Convert physical to pixel coordinates
        # Use average pixel spacing for isotropic approximation
        px_spacing = (pixel_spacing_mm[0] + pixel_spacing_mm[1]) / 2.0
        radius_px   = radius_mm / px_spacing
        roi_rad_px  = (roi_diameter_mm / 2.0) / px_spacing

        angle_rad = math.radians(angle_deg)
        insert_row = center_row - radius_px * math.cos(angle_rad)
        insert_col = center_col + radius_px * math.sin(angle_rad)

        row_start = int(round(insert_row - roi_rad_px))
        col_start = int(round(insert_col - roi_rad_px))
        size_px   = max(int(round(2 * roi_rad_px)), 10)

        # Bounds check
        rows, cols = image_shape
        if (row_start < 0 or col_start < 0 or
                row_start + size_px > rows or col_start + size_px > cols):
            raise ROIOutOfBoundsError(
                "ROI '%s' at (row=%d, col=%d, size=%dpx) "
                "falls outside image shape %s. "
                "Phantom center=(%.1f, %.1f), "
                "radius=%smm=%.1fpx, angle=%s°. "
                "Check phantom is centered within the image."
                % (label, row_start, col_start, size_px,
                   image_shape, center_row, center_col,
                   radius_mm, radius_px, angle_deg)
            )

        return ROIDescriptor(
            label=label,
            row_start=row_start,
            col_start=col_start,
            height_px=size_px,
            width_px=size_px,
        )


# ── Concrete adapter: Siemens Water Phantom ────────────────────────────────

class SiemensWaterPhantomAdapter(PhantomAdapter):
    """
    Adapter for the Siemens standard cylindrical water phantom.

    Geometry: 5 ROIs — one central + four peripheral at 90° intervals
    (12, 3, 6, 9 o'clock positions at 60 mm radius from center).
    Used for daily QA noise and uniformity measurements.

    Reference: Siemens CT QA Protocol Reference Guide, Section 3.
    """

    PHANTOM_ID       = "siemens_water_phantom"
    OUTER_RADIUS_MM  = 100.0   # 200 mm diameter phantom
    PERIPHERAL_R_MM  = 60.0    # radial distance of peripheral ROIs from center
    CENTER_ROI_DM_MM = 30.0    # center ROI diameter
    PERIPH_ROI_DM_MM = 20.0    # peripheral ROI diameter
    HU_TOLERANCE     = 4.0     # TG-66 Section 5.2

    @property
    def phantom_id(self) -> str:
        return self.PHANTOM_ID

    def has_edge_insert(self) -> bool:
        return False

    def has_density_inserts(self) -> bool:
        return False

    def get_material_references(self) -> dict[str, MaterialReference]:
        return {
            "water": MaterialReference(
                name="water", nominal_hu=0.0, red=1.000,
                hu_tolerance=self.HU_TOLERANCE
            ),
        }

    def get_roi_descriptors(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """
        Returns 5 ROIDescriptors positioned on the Siemens water phantom.
        Center is auto-detected; falls back to image center if detection fails.
        """
        image_shape = hu_array.shape
        c_row, c_col = self._detect_center_safe(hu_array, image_shape)

        roi_layout = {
            "center":        (0.0,   0.0,   self.CENTER_ROI_DM_MM),
            "peripheral_12": (self.PERIPHERAL_R_MM,   0.0,   self.PERIPH_ROI_DM_MM),
            "peripheral_3":  (self.PERIPHERAL_R_MM,  90.0,   self.PERIPH_ROI_DM_MM),
            "peripheral_6":  (self.PERIPHERAL_R_MM, 180.0,   self.PERIPH_ROI_DM_MM),
            "peripheral_9":  (self.PERIPHERAL_R_MM, 270.0,   self.PERIPH_ROI_DM_MM),
        }

        descriptors = {}
        for label, (r_mm, angle, diam) in roi_layout.items():
            try:
                descriptors[label] = self._build_roi_from_polar(
                    label, c_row, c_col, r_mm, angle, diam,
                    pixel_spacing_mm, image_shape
                )
            except ROIOutOfBoundsError as exc:
                logger.error(
                    "Siemens water phantom: ROI '%s' out of bounds — skipped. %s",
                    label, exc
                )

        logger.debug(
            "SiemensWaterPhantomAdapter: %d ROIs computed for image shape %s",
            len(descriptors), image_shape
        )
        return descriptors

    def get_water_rois(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Siemens water phantom: water ROIs ARE the full descriptor set."""
        return self.get_roi_descriptors(hu_array, pixel_spacing_mm)


# ── Concrete adapter: GE Helios QA Phantom ────────────────────────────────

class HeliosQAPhantomAdapter(PhantomAdapter):
    """
    Adapter for the GE Helios QA Phantom.

    Geometry: 7 inserts — water (center) + 6 peripheral at 60° intervals.
    Uses HeliosPhantomDetector (existing Phase 5 GE module) for center/radius
    detection with automatic fallback to nominal geometry.

    Reference: GE Helios QA Phantom Manual P/N 2165993-100 Rev 1, Figure 3-1.
    """

    PHANTOM_ID      = "ge_helios_qa"
    INSERT_R_MM     = 58.0    # radial distance from center (GE manual §3.2)
    OUTER_RADIUS_MM = 100.0   # outer radius (GE manual §3.1)
    ROI_DIAMETER_MM = 20.0    # 4 mm safety margin from 28 mm insert
    HU_TOLERANCE    = 4.0     # TG-66 Section 5.2

    # Angular positions — GE Helios manual Figure 3-1
    # 0° = top (12 o'clock), clockwise
    INSERT_ANGLES: dict[str, float] = {
        "air":         0.0,
        "acrylic":    60.0,
        "ldpe":      120.0,
        "polystyrene": 180.0,
        "delrin":    240.0,
        "teflon":    300.0,
    }

    # Material references — Schneider et al. 1996; IAEA TRS-430 Table 4.1
    MATERIALS: dict[str, tuple[float, float]] = {
        # (nominal_hu, RED)
        "water":       (   0.0, 1.000),
        "air":         (-1000.0, 0.001),
        "acrylic":     ( 120.0, 1.173),
        "ldpe":        (-100.0, 0.944),
        "polystyrene": ( -35.0, 0.976),
        "delrin":      ( 340.0, 1.359),
        "teflon":      ( 990.0, 1.869),
    }

    @property
    def phantom_id(self) -> str:
        return self.PHANTOM_ID

    def has_edge_insert(self) -> bool:
        # Teflon insert provides a high-contrast edge suitable for MTF
        return True

    def has_density_inserts(self) -> bool:
        return True

    def get_material_references(self) -> dict[str, MaterialReference]:
        return {
            name: MaterialReference(
                name=name,
                nominal_hu=vals[0],
                red=vals[1],
                hu_tolerance=self.HU_TOLERANCE,
            )
            for name, vals in self.MATERIALS.items()
        }

    def get_roi_descriptors(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """
        Returns 7 ROIDescriptors positioned on the GE Helios phantom.
        Attempts auto-detection of center/radius; falls back to nominal.
        """
        image_shape = hu_array.shape
        c_row, c_col = self._detect_center_safe(hu_array, image_shape)

        descriptors = {}

        # Center insert (water)
        try:
            descriptors["water"] = self._build_roi_from_polar(
                "water", c_row, c_col,
                0.0, 0.0, self.ROI_DIAMETER_MM,
                pixel_spacing_mm, image_shape
            )
        except ROIOutOfBoundsError as exc:
            logger.error("Helios: center ROI out of bounds — %s", exc)

        # Peripheral inserts
        for label, angle in self.INSERT_ANGLES.items():
            try:
                descriptors[label] = self._build_roi_from_polar(
                    label, c_row, c_col,
                    self.INSERT_R_MM, angle, self.ROI_DIAMETER_MM,
                    pixel_spacing_mm, image_shape
                )
            except ROIOutOfBoundsError as exc:
                logger.error("Helios: ROI '%s' out of bounds — skipped. %s", label, exc)

        logger.debug(
            "HeliosQAPhantomAdapter: %d/7 ROIs computed for image shape %s",
            len(descriptors), image_shape
        )
        return descriptors

    def get_water_rois(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Helios water/uniformity ROIs: center + 4 peripheral clock positions.

        Uses a 60 mm radius for peripheral ROIs (uniformity measurement),
        NOT the insert radius — we want to sample the water body itself
        at the standard 5-ROI clock-face positions.
        """
        image_shape = hu_array.shape
        c_row, c_col = self._detect_center_safe(hu_array, image_shape)
        # Uniformity ROIs at 60 mm — samples the water body between inserts
        water_periph_r_mm = 60.0
        layout = {
            "center_water":  (0.0,                0.0, self.ROI_DIAMETER_MM),
            "peripheral_12": (water_periph_r_mm,   0.0, self.ROI_DIAMETER_MM),
            "peripheral_3":  (water_periph_r_mm,  90.0, self.ROI_DIAMETER_MM),
            "peripheral_6":  (water_periph_r_mm, 180.0, self.ROI_DIAMETER_MM),
            "peripheral_9":  (water_periph_r_mm, 270.0, self.ROI_DIAMETER_MM),
        }
        descriptors: dict[str, ROIDescriptor] = {}
        for label, (r_mm, angle, diam) in layout.items():
            try:
                descriptors[label] = self._build_roi_from_polar(
                    label, c_row, c_col, r_mm, angle, diam,
                    pixel_spacing_mm, image_shape,
                )
            except ROIOutOfBoundsError as exc:
                logger.warning("Helios water ROI '%s' out of bounds: %s", label, exc)
        return descriptors

    def get_sensitometry_rois(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Helios sensitometry ROIs: 6 material inserts at their angular offsets.

        Returns Air, Acrylic, LDPE, Polystyrene, Delrin, Teflon positioned
        at their correct angular offsets relative to the dynamically
        detected phantom center.

        Reference: GE Helios QA Phantom Manual P/N 2165993-100, Figure 3-1.
        """
        image_shape = hu_array.shape
        c_row, c_col = self._detect_center_safe(hu_array, image_shape)
        descriptors: dict[str, ROIDescriptor] = {}
        for label, angle in self.INSERT_ANGLES.items():
            try:
                descriptors[label] = self._build_roi_from_polar(
                    label, c_row, c_col,
                    self.INSERT_R_MM, angle, self.ROI_DIAMETER_MM,
                    pixel_spacing_mm, image_shape,
                )
            except ROIOutOfBoundsError as exc:
                logger.warning(
                    "Helios sensitometry ROI '%s' out of bounds: %s", label, exc
                )
        return descriptors

    def get_mtf_roi(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Helios MTF ROI: STATIC geometry — tungsten bead 120px ABOVE center.

        Bypasses the base-class HU threshold (max > 1500) which fails on
        clinical noise. The GE Helios MTF bead is physically located at
        the TOP of the phantom, approximately 120px above the detected center.

        Reference: GE Helios QA Phantom Manual P/N 2165993-100, §3.
        """
        image_shape = hu_array.shape
        c_row, c_col = self._detect_center_safe(hu_array, image_shape)
        cr, cc = int(round(c_row)), int(round(c_col))
        # Bead is at TOP of phantom, 120px above center (row decreases = upward)
        bead_row = max(2, cr - 120)
        roi_half = 2  # 5×5 pixel ROI

        row_start = max(0, bead_row - roi_half)
        col_start = max(0, cc - roi_half)
        row_end = min(image_shape[0], bead_row + roi_half + 1)
        col_end = min(image_shape[1], cc + roi_half + 1)

        logger.info(
            "Helios MTF ROI (static): center=(%d, %d), bead=(%d, %d), roi=%dx%d",
            cr, cc, bead_row, cc, row_end - row_start, col_end - col_start,
        )

        return {
            "mtf_wire": ROIDescriptor(
                label="mtf_wire",
                row_start=row_start,
                col_start=col_start,
                height_px=row_end - row_start,
                width_px=col_end - col_start,
            )
        }


# ── Concrete adapter: Generic ──────────────────────────────────────────────

class GenericPhantomAdapter(PhantomAdapter):
    """
    Universal fallback adapter for unrecognized phantoms.
    Uses the standard 5-ROI layout (center + 4 peripheral at 90°).
    """

    PHANTOM_ID       = "generic"
    PERIPHERAL_R_MM  = 60.0
    CENTER_ROI_DM_MM = 30.0
    PERIPH_ROI_DM_MM = 20.0
    HU_TOLERANCE     = 5.0   # slightly looser tolerance for unknown phantoms

    @property
    def phantom_id(self) -> str:
        return self.PHANTOM_ID

    def has_edge_insert(self) -> bool:
        return False

    def has_density_inserts(self) -> bool:
        return False

    def get_material_references(self) -> dict[str, MaterialReference]:
        return {
            "water": MaterialReference(
                name="water", nominal_hu=0.0, red=1.000,
                hu_tolerance=self.HU_TOLERANCE
            ),
        }

    def get_roi_descriptors(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        image_shape = hu_array.shape
        c_row, c_col = self._detect_center_safe(hu_array, image_shape)

        layout = {
            "center":        (0.0,                   0.0),
            "peripheral_12": (self.PERIPHERAL_R_MM,   0.0),
            "peripheral_3":  (self.PERIPHERAL_R_MM,  90.0),
            "peripheral_6":  (self.PERIPHERAL_R_MM, 180.0),
            "peripheral_9":  (self.PERIPHERAL_R_MM, 270.0),
        }
        roi_diam = {
            "center": self.CENTER_ROI_DM_MM,
        }
        descriptors = {}
        for label, (r_mm, angle) in layout.items():
            diam = roi_diam.get(label, self.PERIPH_ROI_DM_MM)
            try:
                descriptors[label] = self._build_roi_from_polar(
                    label, c_row, c_col, r_mm, angle, diam,
                    pixel_spacing_mm, image_shape
                )
            except ROIOutOfBoundsError as exc:
                logger.error("Generic adapter: ROI '%s' out of bounds — %s", label, exc)

        return descriptors

    def get_water_rois(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> dict[str, ROIDescriptor]:
        """Generic phantom: water ROIs ARE the full descriptor set."""
        return self.get_roi_descriptors(hu_array, pixel_spacing_mm)


# ── Factory ────────────────────────────────────────────────────────────────

class PhantomAdapterFactory:
    """
    Selects and instantiates the correct PhantomAdapter based on
    DICOM SeriesDescription and ProtocolName tags, configured via
    phantom_profiles.yaml. Returns GenericPhantomAdapter if no match.
    """

    _ADAPTER_MAP: dict[str, type[PhantomAdapter]] = {
        "SiemensWaterPhantomAdapter": SiemensWaterPhantomAdapter,
        "HeliosQAPhantomAdapter":     HeliosQAPhantomAdapter,
        "GenericPhantomAdapter":      GenericPhantomAdapter,
    }

    def __init__(
        self,
        profiles_yaml: Path = _PHANTOM_PROFILES_YAML,
    ) -> None:
        self._profiles = self._load_profiles(profiles_yaml)
        logger.info(
            "PhantomAdapterFactory loaded %d phantom profiles",
            len(self._profiles)
        )

    def _load_profiles(self, yaml_path: Path) -> dict:
        if not yaml_path.exists():
            logger.warning(
                "phantom_profiles.yaml not found at %s — using generic adapter only.",
                yaml_path
            )
            return {}
        try:
            return yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("profiles", {})
        except yaml.YAMLError as exc:
            raise PhantomProfileError(
                "Failed to parse phantom_profiles.yaml: %s" % exc
            ) from exc

    def create(
        self,
        ds: "pydicom.Dataset",
        override_phantom_id: Optional[str] = None,
    ) -> PhantomAdapter:
        """
        Creates the appropriate PhantomAdapter for the given DICOM dataset.

        Detection uses:
          1. override_phantom_id if explicitly provided (CLI --phantom flag)
          2. SeriesDescription tag — matched against series_description_match
          3. ProtocolName tag — matched against protocol_name_match
          4. GenericPhantomAdapter fallback

        Matching is case-insensitive substring matching.
        """
        if override_phantom_id:
            return self._instantiate(override_phantom_id)

        series_desc  = str(getattr(ds, "SeriesDescription",  "")).upper()
        protocol     = str(getattr(ds, "ProtocolName",       "")).upper()

        for phantom_id, cfg in self._profiles.items():
            if phantom_id == "generic":
                continue
            desc_matches = cfg.get("series_description_match", [])
            prot_matches = cfg.get("protocol_name_match", [])
            if any(m.upper() in series_desc for m in desc_matches):
                logger.info(
                    "Phantom detected by SeriesDescription: '%s' → %s",
                    series_desc, phantom_id
                )
                return self._instantiate(phantom_id)
            if any(m.upper() in protocol for m in prot_matches):
                logger.info(
                    "Phantom detected by ProtocolName: '%s' → %s",
                    protocol, phantom_id
                )
                return self._instantiate(phantom_id)

        logger.warning(
            "No phantom profile matched SeriesDescription='%s' ProtocolName='%s'. "
            "Using GenericPhantomAdapter fallback. "
            "Add a profile to phantom_profiles.yaml if this is a known phantom.",
            series_desc, protocol
        )
        return GenericPhantomAdapter()

    def _instantiate(self, phantom_id: str) -> PhantomAdapter:
        """Instantiates an adapter by phantom profile ID."""
        profile = self._profiles.get(phantom_id, {})
        adapter_class_name = profile.get("adapter_class", "GenericPhantomAdapter")
        adapter_class = self._ADAPTER_MAP.get(
            adapter_class_name, GenericPhantomAdapter
        )
        logger.debug(
            "Instantiating PhantomAdapter: %s (profile: %s)",
            adapter_class.__name__, phantom_id
        )
        return adapter_class()
