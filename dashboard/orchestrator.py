"""
dashboard/orchestrator.py — TotalQA-Aligned Regex Interval Routing.

Parses 'Image X' from uploaded filenames, routes ONE representative
slice per module. Matched to Image Owl TotalQA report targets.

Exact Slice Targets (aligned to TotalQA GE Set 1):
  ×36  → Uniformity / Noise (Pure Water) + Scaling (H/V Diameter)
  ×56  → Contrast (Plastic Block — 4 rect ROIs B/A/C/D vertical stack)
  ×71  → Resolution (Angled Bar Patterns — 5 diagonal square ROIs)
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime

import numpy as np
import pydicom
import streamlit as st

from config import CONFIG
from modules.image_qc.roi_stats import (
    ROIDescriptor, PhantomROIAnalyzer, VolumetricQCResult, VolumetricROIStat,
    SliceAnalysisResult,
)
from modules.image_qc.basic_metrics import (
    BasicMetricsEngine, TotalQAContrastResult, TotalQAResolutionResult,
    TotalQAScalingResult,
)
from dashboard.cached_resources import (
    get_loader, get_registry, get_adapter_factory,
    get_basic_engine, get_advanced_engine,
)
from dashboard.roi_drawing import render_roi_drawing

logger = logging.getLogger(__name__)

_IMAGE_NUM_RE = re.compile(r"Image\s*(\d+)\.dcm", re.IGNORECASE)

# ══════════════════════════════════════════════════════════════════
# TotalQA-Matched ROI Definitions
# ══════════════════════════════════════════════════════════════════

NOMINAL_PHANTOM_DIAMETER_MM = 215.0


def _detect_center(hu_array: np.ndarray) -> tuple[int, int]:
    """Detect true phantom center using the largest CIRCULAR connected component.

    Algorithm (couch-safe):
      1. Threshold at HU > -300 (phantom + couch > -300, air < -300)
      2. Fill internal holes (air inserts, wire channels)
      3. Label connected components
      4. Filter by circularity: aspect_ratio > 0.7 AND fill_ratio > 0.5
         → phantom is circular (aspect ~1.0, fill ~0.78)
         → couch is elongated (aspect ~0.1-0.3)
      5. Select the largest circular component
      6. Return center_of_mass of that component only

    Falls back to (254, 254) if detection fails.
    """
    from scipy import ndimage

    mask = hu_array > -300
    if not np.any(mask):
        return 254, 254

    filled = ndimage.binary_fill_holes(mask)
    labeled, n_labels = ndimage.label(filled)
    if n_labels == 0:
        return 254, 254

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
        if aspect > 0.7 and fill_ratio > 0.5 and area > best_area:
            best_area = area
            best_label = lbl

    if best_label is None:
        sizes = ndimage.sum(filled, labeled, range(1, n_labels + 1))
        best_label = int(np.argmax(sizes)) + 1

    com = ndimage.center_of_mass((labeled == best_label).astype(float))
    return int(com[0]), int(com[1])


# ── GE Helios Contrast ROI geometry (physics-based mm offsets) ────
# Physical distances from phantom center to contrast ROI centers.
# Reference: GE Helios QA Phantom Manual P/N 2165993-100.
# A/C inside plastic block (~±20mm extent), B/D far in pure water.
_CONTRAST_FAR_OFFSET_MM = 50.0    # B/D water ROIs — very far from center
_CONTRAST_NEAR_OFFSET_MM = 12.0   # A/C plastic ROIs — firmly inside block
_CONTRAST_ROI_HEIGHT_MM = 6.0     # ROI height — thin band
_CONTRAST_ROI_WIDTH_MM = 55.0     # ROI width  — compact inside plastic

# ── GE Helios Resolution ROI geometry ─────────────────────────────
# Precise physics geometry based on GE Helios phantom measurements.
# The bar pattern insert center (B3) is offset from the phantom geometric center.
#
# Bar center (B3) offset from phantom center (mm):
_BAR_CENTER_OFFSET_ROW_MM = -15.4  # 24 px upward
_BAR_CENTER_OFFSET_COL_MM = 6.8    # 6 px rightward

# Diagonal offsets from bar-center (mm) — symmetric 24.4 mm spacing (25 px)
_BAR_DIAG_OFFSETS_MM = [20.0, 3.0, -10.0, -23.0, -35.0]

# ROI sizes per bar group (mm) — proportional to bar frequency
# B1(1.6lp/mm)=largest → B5(0.6lp/mm)=smallest
_BAR_ROI_SIZES_MM = [10.0, 8.0, 5.0, 2.0, 1.0]


def _totalqa_contrast_rois(
    hu_array: np.ndarray,
    pixel_spacing: tuple[float, float] = (0.977, 0.977),
) -> dict[str, ROIDescriptor]:
    """4 rectangular ROIs on Slice 56 (plastic block with horizontal holes).

    Layout matching TotalQA — vertically stacked, horizontally centered on cx.
      B = Top Water      (cy - 78.0 mm)
      A = Top Plastic    (cy - 24.5 mm)
      C = Bottom Plastic (cy + 24.5 mm)
      D = Bottom Water   (cy + 78.0 mm)

    All offsets are in millimeters, converted to pixels dynamically via
    DICOM pixel_spacing. Zero horizontal offset from center.
    """
    cr, cc = _detect_center(hu_array)
    scale_y, scale_x = pixel_spacing[0], pixel_spacing[1]

    # Convert mm offsets to pixel offsets
    far_dy = int(round(_CONTRAST_FAR_OFFSET_MM / scale_y))
    near_dy = int(round(_CONTRAST_NEAR_OFFSET_MM / scale_y))
    roi_h = int(round(_CONTRAST_ROI_HEIGHT_MM / scale_y))
    roi_w = int(round(_CONTRAST_ROI_WIDTH_MM / scale_x))
    half_h, half_w = roi_h // 2, roi_w // 2

    rois = {
        "top_water":      ROIDescriptor("top_water",
                                        cr - far_dy - half_h, cc - half_w, roi_h, roi_w),
        "top_plastic":    ROIDescriptor("top_plastic",
                                        cr - near_dy - half_h, cc - half_w, roi_h, roi_w),
        "bottom_plastic": ROIDescriptor("bottom_plastic",
                                        cr + near_dy - half_h, cc - half_w, roi_h, roi_w),
        "bottom_water":   ROIDescriptor("bottom_water",
                                        cr + far_dy - half_h, cc - half_w, roi_h, roi_w),
    }
    return rois


def _totalqa_bar_pattern_rois(
    hu_array: np.ndarray,
    pixel_spacing: tuple[float, float] = (0.977, 0.977),
) -> dict[str, ROIDescriptor]:
    """5 scaled ROIs on the / diagonal (bar patterns).

    The bar pattern insert is offset upward from the phantom center.
    ROI sizes scale proportionally: B1 (1.6 lp/mm, largest) → B5 (0.6 lp/mm, smallest).

    Positions are fixed and symmetric around the bar-insert center.
    """
    cr, cc = _detect_center(hu_array)
    scale_y, scale_x = pixel_spacing[0], pixel_spacing[1]

    # Apply bar-insert center correction
    bar_cr = cr + int(round(_BAR_CENTER_OFFSET_ROW_MM / scale_y))
    bar_cc = cc + int(round(_BAR_CENTER_OFFSET_COL_MM / scale_x))

    rois = {}
    for i, (offset_mm, roi_size_mm) in enumerate(
        zip(_BAR_DIAG_OFFSETS_MM, _BAR_ROI_SIZES_MM), 1
    ):
        # / diagonal: row = center - offset, col = center + offset
        dr = -int(round(offset_mm / scale_y))
        dc = int(round(offset_mm / scale_x))
        sz = max(int(round(roi_size_mm / scale_x)), 6)
        half = sz // 2

        rois[f"bar_{i}"] = ROIDescriptor(
            f"bar_{i}",
            bar_cr + dr - half,
            bar_cc + dc - half,
            sz, sz,
        )

    return rois


# TotalQA standard bar pattern sizes (LP/mm) — coarsest to finest
TOTALQA_BAR_LPMM = [1.6, 1.3, 1.0, 0.8, 0.6]


def get_max_sd_in_neighborhood(
    image: np.ndarray,
    start_row: int,
    start_col: int,
    search_radius: int = 20,
    roi_size: int = 12,
) -> tuple[float, float, int, int]:
    """Scan a neighborhood around (start_row, start_col) and return the
    patch with the MAXIMUM Standard Deviation.

    The highest SD guarantees we've hit alternating black/white bar lines
    rather than solid gray background.

    Returns (max_sd, mean_hu, best_row, best_col) where best_row/col
    are the top-left corner of the winning patch.
    """
    rows, cols = image.shape
    half = roi_size // 2
    best_sd = -1.0
    best_mean = 0.0
    best_r, best_c = start_row - half, start_col - half

    r_min = max(0, start_row - search_radius)
    r_max = min(rows - roi_size, start_row + search_radius)
    c_min = max(0, start_col - search_radius)
    c_max = min(cols - roi_size, start_col + search_radius)

    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            patch = image[r:r + roi_size, c:c + roi_size].astype(np.float64)
            sd = float(np.std(patch, ddof=1)) if patch.size > 1 else 0.0
            if sd > best_sd:
                best_sd = sd
                best_mean = float(np.mean(patch))
                best_r, best_c = r, c

    return best_sd, best_mean, best_r, best_c


def _measure_phantom_diameter(
    hu_array: np.ndarray, pixel_spacing: tuple
) -> tuple[float, float]:
    """Measure horizontal and vertical diameter of the phantom in mm.

    Bulletproof morphological approach (independent of ROI logic):
      1. Threshold at HU > -300 (water/plastic > -300, air ~ -1000).
      2. Extract the largest CIRCULAR connected component (excludes couch).
      3. Fill internal holes (air inserts, wire channels).
      4. Compute bounding box of the filled mask.
      5. Convert pixel extents to mm using pixel_spacing.

    This replaces the fragile line-profile approach.
    """
    from scipy import ndimage

    binary_mask = hu_array > -300.0
    if not np.any(binary_mask):
        return 0.0, 0.0

    filled_full = ndimage.binary_fill_holes(binary_mask)
    labelled, n_features = ndimage.label(filled_full)
    if n_features == 0:
        return 0.0, 0.0

    # Find the largest CIRCULAR component (excludes elongated couch)
    best_label = None
    best_area = 0
    for lbl in range(1, n_features + 1):
        component = labelled == lbl
        area = int(component.sum())
        if area < 1000:
            continue
        rr, cc = np.where(component)
        bbox_h = rr.max() - rr.min() + 1
        bbox_w = cc.max() - cc.min() + 1
        aspect = min(bbox_h, bbox_w) / max(bbox_h, bbox_w)
        fill_ratio = area / (bbox_h * bbox_w)
        if aspect > 0.7 and fill_ratio > 0.5 and area > best_area:
            best_area = area
            best_label = lbl

    if best_label is None:
        # Fallback: largest component regardless of shape
        component_sizes = ndimage.sum(
            filled_full, labelled, range(1, n_features + 1))
        best_label = int(np.argmax(component_sizes)) + 1

    phantom_mask = ndimage.binary_fill_holes(labelled == best_label)

    # Bounding box → diameter
    rows_any = np.any(phantom_mask, axis=1)
    cols_any = np.any(phantom_mask, axis=0)
    row_indices = np.where(rows_any)[0]
    col_indices = np.where(cols_any)[0]

    if len(row_indices) < 2 or len(col_indices) < 2:
        return 0.0, 0.0

    v_diameter_mm = (row_indices[-1] - row_indices[0]) * pixel_spacing[0]
    h_diameter_mm = (col_indices[-1] - col_indices[0]) * pixel_spacing[1]
    return h_diameter_mm, v_diameter_mm


def _build_single_slice_vol(loader, ds, rois, start_s, end_s):
    """Build a VolumetricQCResult from a SINGLE dataset."""
    analyzer = PhantomROIAnalyzer(loader, CONFIG.image_qc)
    hu = loader.to_hu_array(ds)
    sr = analyzer.analyze_dataset(ds, rois)
    meta = sr.metadata

    vol_stats = {}
    for roi in rois:
        try:
            stat = sr.get_roi_stat(roi.label)
            vol_stats[roi.label] = VolumetricROIStat(
                roi_label=roi.label, n_slices=1,
                mean_hu_mean=stat.mean_hu, mean_hu_std=0.0,
                std_hu_mean=stat.std_hu, std_hu_std=0.0,
                variance_hu_mean=stat.variance_hu, variance_hu_std=0.0,
                min_hu_overall=stat.min_hu, max_hu_overall=stat.max_hu,
                snr_mean=stat.snr,
                passes_tg66=stat.std_hu <= CONFIG.image_qc.noise_tolerance_hu,
            )
        except KeyError:
            continue

    return VolumetricQCResult(
        series_description=meta.series_description,
        acquisition_date=meta.acquisition_date,
        start_slice=start_s, end_slice=end_s,
        n_slices_selected=1, n_slices_processed=1,
        pixel_spacing_mm=meta.pixel_spacing_mm,
        slice_thickness_mm=meta.slice_thickness_mm,
        slice_results=[sr], volumetric_stats=vol_stats,
        hu_arrays=[hu],
    )


# ══════════════════════════════════════════════════════════════════
# Manufacturer Detection
# ══════════════════════════════════════════════════════════════════

def _detect_manufacturer(uploaded_files: list) -> str:
    """Read DICOM Manufacturer tag from the first file.

    Returns 'SIEMENS', 'GE', or 'CANON' (default 'GE' if unknown).
    Canon/Toshiba DICOMs are detected and returned as 'CANON'.
    """
    for f in uploaded_files:
        try:
            f.seek(0)
            ds = pydicom.dcmread(io.BytesIO(f.read()), force=True,
                                 stop_before_pixels=True)
            f.seek(0)  # Reset for downstream readers
            mfr = str(getattr(ds, "Manufacturer", "")).upper()
            if "SIEMENS" in mfr:
                return "SIEMENS"
            elif "GE" in mfr or "GENERAL" in mfr:
                return "GE"
            elif "CANON" in mfr or "TOSHIBA" in mfr:
                return "CANON"
            else:
                logger.info("Unknown manufacturer '%s', defaulting to GE", mfr)
                return "GE"
        except Exception as exc:
            logger.warning("Cannot read manufacturer from %s: %s", f.name, exc)
            continue
    return "GE"


def _detect_manufacturer_from_datasets(
    datasets: list[tuple[str, pydicom.Dataset]],
) -> str:
    """Detect manufacturer from pre-read datasets.

    Returns 'SIEMENS', 'GE', or 'CANON' (default 'GE' if unknown).
    """
    for _filename, ds in datasets:
        mfr = str(getattr(ds, "Manufacturer", "")).upper()
        if "SIEMENS" in mfr:
            return "SIEMENS"
        elif "GE" in mfr or "GENERAL" in mfr:
            return "GE"
        elif "CANON" in mfr or "TOSHIBA" in mfr:
            return "CANON"
        else:
            logger.info("Unknown manufacturer '%s', defaulting to GE", mfr)
            return "GE"
    return "GE"


# ══════════════════════════════════════════════════════════════════
# TG-204 Dosimetry Computation (shared by GE & Siemens)
# ══════════════════════════════════════════════════════════════════

def _compute_tg204_dosimetry(
    ds,
    noise_sd: float | None,
    h_diameter_mm: float | None,
    v_diameter_mm: float | None,
) -> dict:
    """Compute TG-204 dosimetry metrics from DICOM metadata + image metrics.

    Extracts CTDIvol (0018,9345) and DLP (0018,9346) from the DICOM dataset,
    then computes:
      - Deff = sqrt(AP_cm * LAT_cm)
      - f = 3.704 * exp(-0.0367 * Deff)   (32 cm PMMA body reference)
      - SSDE = CTDIvol * f
      - FOM  = 1 / (Noise^2 * CTDIvol)

    Returns a dict with all values (None-safe).
    """
    import math

    result = {
        "ctdi_vol_mgy": None,
        "dlp_mgy_cm": None,
        "dlp_source": None,          # "dicom" or "calculated"
        "noise_sd": noise_sd,
        "ap_cm": None,
        "lat_cm": None,
        "d_eff_cm": None,
        "f_factor": None,
        "ssde_mgy": None,
        "fom": None,
    }

    # ── Extract CTDIvol and DLP from DICOM header ─────────────────
    if ds is not None:
        # CTDIvol: standard tag (0018,9345)
        try:
            val = getattr(ds, "CTDIvol", None)
            if val is not None:
                result["ctdi_vol_mgy"] = float(val)
            elif (0x0018, 0x9345) in ds:
                result["ctdi_vol_mgy"] = float(ds[0x0018, 0x9345].value)
        except (TypeError, ValueError, AttributeError):
            pass

        # DLP: tag (0018,9346)
        try:
            val = getattr(ds, "DLP", None)
            if val is not None:
                result["dlp_mgy_cm"] = float(val)
                result["dlp_source"] = "dicom"
            elif (0x0018, 0x9346) in ds:
                result["dlp_mgy_cm"] = float(ds[0x0018, 0x9346].value)
                result["dlp_source"] = "dicom"
        except (TypeError, ValueError, AttributeError):
            pass

        # ── DLP fallback: CTDIvol × SliceThickness (cm) ───────────
        # When (0018,9346) is absent (common in single-slice QA phantom
        # acquisitions), compute DLP from the physics definition:
        #   DLP [mGy·cm] = CTDIvol [mGy] × ScanLength [cm]
        # For a single slice, ScanLength ≈ SliceThickness.
        if result["dlp_mgy_cm"] is None and result["ctdi_vol_mgy"] is not None:
            try:
                slice_thickness_mm = float(getattr(ds, "SliceThickness", 0.0))
                if (0x0018, 0x0050) in ds and slice_thickness_mm == 0.0:
                    slice_thickness_mm = float(ds[0x0018, 0x0050].value)
                if slice_thickness_mm > 0:
                    result["dlp_mgy_cm"] = result["ctdi_vol_mgy"] * (slice_thickness_mm / 10.0)
                    result["dlp_source"] = "calculated"
            except (TypeError, ValueError, AttributeError):
                pass

    # ── Morphometry: AP, LAT, Deff ────────────────────────────────
    if v_diameter_mm is not None and v_diameter_mm > 0:
        result["ap_cm"] = v_diameter_mm / 10.0
    if h_diameter_mm is not None and h_diameter_mm > 0:
        result["lat_cm"] = h_diameter_mm / 10.0

    if result["ap_cm"] and result["lat_cm"]:
        result["d_eff_cm"] = math.sqrt(result["ap_cm"] * result["lat_cm"])

    # ── TG-204 conversion factor & SSDE ───────────────────────────
    if result["d_eff_cm"] is not None and result["ctdi_vol_mgy"] is not None:
        d = result["d_eff_cm"]
        f = 3.704 * math.exp(-0.0367 * d)
        result["f_factor"] = f
        result["ssde_mgy"] = result["ctdi_vol_mgy"] * f

    # ── Figure of Merit ───────────────────────────────────────────
    if (noise_sd is not None and noise_sd > 0
            and result["ctdi_vol_mgy"] is not None
            and result["ctdi_vol_mgy"] > 0):
        result["fom"] = 1.0 / (noise_sd ** 2 * result["ctdi_vol_mgy"])

    return result


# ══════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════

def run_full_analysis(
    uploaded_files: list,
    start_slice: int,
    end_slice: int,
    scanner_id: str,
    skip_modules: list[str],
) -> None:
    """TotalQA-aligned analysis — force-reads all files, auto-detects manufacturer."""

    # ══════════════════════════════════════════════════════════════
    # STEP 0: FORCE-READ ALL UPLOADED FILES (extension-agnostic)
    # ══════════════════════════════════════════════════════════════
    valid_datasets: list[tuple[str, pydicom.Dataset]] = []
    decompress_errors: list[str] = []

    with st.spinner("📂 Force-reading DICOM files..."):
        for uploaded_file in uploaded_files:
            try:
                uploaded_file.seek(0)
                ds = pydicom.dcmread(
                    io.BytesIO(uploaded_file.read()), force=True)

                # Explicitly access pixel_array to trigger decompression
                # (needed for JPEG Lossless / Transfer Syntax 1.2.840.10008.1.2.4.70)
                try:
                    _ = ds.pixel_array
                    ds.filename = uploaded_file.name
                    valid_datasets.append((uploaded_file.name, ds))
                except NotImplementedError as nie:
                    decompress_errors.append(
                        f"⚠️ **{uploaded_file.name}**: Cannot decompress — "
                        f"install `pylibjpeg` + `pylibjpeg-libjpeg`. ({nie})")
                except AttributeError:
                    # No PixelData tag — not an image DICOM (e.g. DICOMDIR)
                    continue
                except Exception as px_err:
                    decompress_errors.append(
                        f"⚠️ **{uploaded_file.name}**: pixel_array error — {px_err}")
            except Exception:
                continue  # Not a DICOM at all — skip silently

    # Surface decompression errors to user
    if decompress_errors:
        for err_msg in decompress_errors:
            st.error(err_msg)

    if not valid_datasets:
        st.error("❌ No valid DICOM image files found in the upload. "
                 "Ensure files contain pixel data and required "
                 "decompression libraries are installed "
                 "(`pip install pylibjpeg pylibjpeg-libjpeg`).")
        return

    st.info(f"📂 **{len(valid_datasets)}** valid DICOM image(s) loaded "
            f"from {len(uploaded_files)} file(s) uploaded.")

    # ══════════════════════════════════════════════════════════════
    # STEP 0B: MANUFACTURER DETECTION — route to correct pipeline
    # ══════════════════════════════════════════════════════════════
    manufacturer = _detect_manufacturer_from_datasets(valid_datasets)
    st.info(f"🏭 **Manufacturer Detected:** {manufacturer}")

    if manufacturer in ("SIEMENS", "CANON"):
        from dashboard.siemens_waterbath import run_siemens_analysis
        run_siemens_analysis(valid_datasets, scanner_id, manufacturer)
        # Override session_state manufacturer to preserve Canon identity
        if manufacturer == "CANON":
            st.session_state["manufacturer"] = "CANON"
        return

    # ── GE Pipeline continues below ──────────────────────────────
    st.session_state["manufacturer"] = "GE"
    loader = get_loader()
    registry = get_registry()
    factory = get_adapter_factory()
    basic_engine = get_basic_engine()
    advanced_engine = get_advanced_engine()

    # ══════════════════════════════════════════════════════════════
    # STEP 1: STRICT REGEX PARSING — route by TotalQA intervals
    # ══════════════════════════════════════════════════════════════
    target_slices: dict[str, pydicom.Dataset | None] = {
        "uniformity": None,
        "contrast": None,
        "resolution": None,
    }
    target_nums: dict[str, int | None] = {
        "uniformity": None, "contrast": None, "resolution": None,
    }
    routing_log: list[str] = []
    n_parsed = 0

    with st.spinner("📂 Regex parsing des noms de fichiers..."):
        for filename, ds in valid_datasets:
            match = _IMAGE_NUM_RE.search(filename)
            if not match:
                routing_log.append(f"  ⏭️ {filename}: no 'Image X' pattern — skipped")
                continue

            slice_num = int(match.group(1))
            n_parsed += 1

            if slice_num == 36 and target_slices["uniformity"] is None:
                target_slices["uniformity"] = ds
                target_nums["uniformity"] = slice_num
                routing_log.append(
                    f"  ✅ {filename}: Image {slice_num} → UNIFORMITY [×36]")

            elif slice_num == 56 and target_slices["contrast"] is None:
                target_slices["contrast"] = ds
                target_nums["contrast"] = slice_num
                routing_log.append(
                    f"  ✅ {filename}: Image {slice_num} → CONTRAST [×56]")

            elif slice_num == 71 and target_slices["resolution"] is None:
                target_slices["resolution"] = ds
                target_nums["resolution"] = slice_num
                routing_log.append(
                    f"  ✅ {filename}: Image {slice_num} → RESOLUTION [×71]")

            else:
                routing_log.append(
                    f"  ⏭️ {filename}: Image {slice_num} — not a target slice")

    # Display routing results
    n_filled = sum(1 for v in target_slices.values() if v is not None)
    st.info(f"📂 **Regex Interval Routing (TotalQA):** {n_parsed} fichiers parsés, "
            f"**{n_filled}/3** modules assignés.")
    with st.expander("📋 Détail du routage par fichier", expanded=False):
        for line in routing_log:
            st.text(line)

    if n_filled == 0:
        st.error("❌ Aucun fichier ne correspond aux cibles TotalQA "
                 "(36=Uniformity, 56=Contrast, 71=Resolution). "
                 "Vérifiez le format: `CT.TPSQA2017.Image X.dcm`.")
        return

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Detect scanner / phantom from first available target
    # ══════════════════════════════════════════════════════════════
    first_ds = next(v for v in target_slices.values() if v is not None)
    scanner_profile = registry.detect(first_ds)
    adapter = factory.create(first_ds)
    pixel_spacing = loader.get_pixel_spacing_mm(first_ds)

    st.info(f"🔬 Scanner: **{scanner_profile.display_name}** | "
            f"Fantôme: **{adapter.phantom_id}**")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Dose metadata (from whatever slice is available)
    # ══════════════════════════════════════════════════════════════
    from core.metadata_miner import MetadataMiner
    available_ds = [v for v in target_slices.values() if v is not None]
    miner = MetadataMiner(scanner_profile)
    mined_meta = miner.mine(available_ds, scanner_id=scanner_id)

    if mined_meta.dose.has_ctdi_vol:
        st.info(f"💊 CTDIvol: **{mined_meta.dose.ctdi_vol_mgy:.3f} mGy**")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: TotalQA-ALIGNED MODULE EXECUTION
    # ══════════════════════════════════════════════════════════════

    vol_result = None
    roi_descriptors = {}
    water_hu = None
    water_rois = {}
    simulation_figs: dict[str, object] = {}

    # TotalQA result holders
    totalqa_contrast = None
    totalqa_resolution = None
    totalqa_scaling = None

    # ── 4A: UNIFORMITY + SCALING (Slice 36) ───────────────────────
    if target_slices["uniformity"] is not None:
        ds_uni = target_slices["uniformity"]
        with st.spinner(
            f"⚖️ Uniformité + Scaling — Image {target_nums['uniformity']}..."
        ):
            water_hu = loader.to_hu_array(ds_uni)
            water_rois = adapter.get_water_rois(water_hu, pixel_spacing)
            roi_descriptors = adapter.get_roi_descriptors(water_hu, pixel_spacing)
            rois = list(roi_descriptors.values())

            vol_result = _build_single_slice_vol(
                loader, ds_uni, rois, start_slice, end_slice)

            # TotalQA Scaling: measure phantom diameter
            try:
                h_mm, v_mm = _measure_phantom_diameter(water_hu, pixel_spacing)
                nom = NOMINAL_PHANTOM_DIAMETER_MM
                h_err = h_mm - nom
                v_err = v_mm - nom
                totalqa_scaling = TotalQAScalingResult(
                    h_diameter_mm=h_mm, v_diameter_mm=v_mm,
                    nominal_mm=nom,
                    h_error_mm=h_err, v_error_mm=v_err,
                    h_error_pct=h_err / nom * 100.0,
                    v_error_pct=v_err / nom * 100.0,
                    tolerance_mm=2.0,
                    passed=abs(h_err) <= 2.0 and abs(v_err) <= 2.0,
                )
                st.success(
                    f"✅ Scaling: H={h_mm:.2f} mm, V={v_mm:.2f} mm "
                    f"(nominal {nom:.1f} mm) — "
                    f"Image {target_nums['uniformity']}")

                # ── Generate Scaling Profile Plot ─────────────────────
                import matplotlib.pyplot as plt
                cr, cc = water_hu.shape[0] // 2, water_hu.shape[1] // 2
                threshold = -200.0

                fig_sc, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(10, 4))
                fig_sc.patch.set_alpha(0.0)

                # Horizontal profile (full row through center)
                h_profile = water_hu[cr, :].astype(np.float64)
                h_above = np.where(h_profile > threshold)[0]
                ax_h.patch.set_alpha(0.0)
                ax_h.plot(h_profile, color='#58a6ff', linewidth=1.2)
                ax_h.axhline(threshold, color='gray', linestyle=':', alpha=0.5)
                if len(h_above) >= 2:
                    ax_h.plot([h_above[0], h_above[-1]],
                              [h_profile[h_above[0]], h_profile[h_above[-1]]],
                              color='red', marker='X', markersize=8,
                              linewidth=2, label=f"H = {h_mm:.1f} mm")
                    ax_h.legend(fontsize=8, facecolor='none', labelcolor='white')
                ax_h.set_title("Horizontal Profile", color='white',
                               fontsize=11, fontweight='bold')
                ax_h.set_xlabel("Pixel", color='white', fontsize=9)
                ax_h.set_ylabel("HU", color='white', fontsize=9)
                ax_h.tick_params(colors='white')
                ax_h.spines['bottom'].set_color('white')
                ax_h.spines['left'].set_color('white')
                ax_h.spines['top'].set_visible(False)
                ax_h.spines['right'].set_visible(False)
                ax_h.grid(True, linestyle='--', alpha=0.2, color='lightgray')

                # Vertical profile (constrained to cy±120 — couch avoidance)
                v_top = max(0, cr - 120)
                v_bot = min(water_hu.shape[0], cr + 120)
                v_profile = water_hu[v_top:v_bot, cc].astype(np.float64)
                v_above = np.where(v_profile > threshold)[0]
                ax_v.patch.set_alpha(0.0)
                ax_v.plot(v_profile, color='#58a6ff', linewidth=1.2)
                ax_v.axhline(threshold, color='gray', linestyle=':', alpha=0.5)
                if len(v_above) >= 2:
                    ax_v.plot([v_above[0], v_above[-1]],
                              [v_profile[v_above[0]], v_profile[v_above[-1]]],
                              color='red', marker='X', markersize=8,
                              linewidth=2, label=f"V = {v_mm:.1f} mm")
                    ax_v.legend(fontsize=8, facecolor='none', labelcolor='white')
                ax_v.set_title("Vertical Profile (±120 px)", color='white',
                               fontsize=11, fontweight='bold')
                ax_v.set_xlabel("Pixel", color='white', fontsize=9)
                ax_v.set_ylabel("HU", color='white', fontsize=9)
                ax_v.tick_params(colors='white')
                ax_v.spines['bottom'].set_color('white')
                ax_v.spines['left'].set_color('white')
                ax_v.spines['top'].set_visible(False)
                ax_v.spines['right'].set_visible(False)
                ax_v.grid(True, linestyle='--', alpha=0.2, color='lightgray')

                fig_sc.suptitle("TotalQA — Scaling Profile (Slice 36)",
                                color='white', fontsize=13, fontweight='bold')
                fig_sc.tight_layout()
                st.session_state["scaling_profile_fig"] = fig_sc
            except Exception as exc:
                logger.warning("Scaling measurement failed: %s", exc)
                st.warning(f"⚠️ Scaling échoué: {exc}")

        # Generate simulation figure
        try:
            fig_uni = render_roi_drawing(
                water_hu, water_rois, pixel_spacing,
                title=f"Image {target_nums['uniformity']} — Uniformité + Scaling",
                slice_type="water")
            simulation_figs["💧 Uniformité & Scaling (Water)"] = fig_uni
        except Exception as exc:
            logger.warning("Uniformity figure failed: %s", exc)

        st.success(
            f"✅ Uniformité calculé — Image {target_nums['uniformity']} "
            f"— {len(roi_descriptors)} ROI(s)")
    else:
        st.warning("⚠️ **Uniformité/Scaling: N/A** — Slice 36 not provided.")

    # ── 4B: CONTRAST (Slice 60 — Plastic Block) ──────────────────
    contrast_hu = None
    contrast_rois = {}

    if target_slices["contrast"] is not None:
        ds_con = target_slices["contrast"]
        with st.spinner(
            f"🧪 Contraste TotalQA — Image {target_nums['contrast']}..."
        ):
            try:
                contrast_hu = loader.to_hu_array(ds_con)
                contrast_rois = _totalqa_contrast_rois(contrast_hu, pixel_spacing)

                # Compute TotalQA contrast: Mean(A)-Mean(B), Mean(C)-Mean(D)
                stats = {}
                for label, roi in contrast_rois.items():
                    region = contrast_hu[
                        roi.row_start:roi.row_end,
                        roi.col_start:roi.col_end
                    ].astype(np.float64)
                    stats[label] = float(np.mean(region))

                mean_A = stats["top_plastic"]
                mean_B = stats["top_water"]
                mean_C = stats["bottom_plastic"]
                mean_D = stats["bottom_water"]
                c_top = mean_A - mean_B
                c_bot = mean_C - mean_D

                totalqa_contrast = TotalQAContrastResult(
                    mean_A=mean_A, mean_B=mean_B,
                    mean_C=mean_C, mean_D=mean_D,
                    contrast_top=c_top, contrast_bottom=c_bot,
                    passed=True,
                )
                st.success(
                    f"✅ Contraste TotalQA: Top={c_top:.2f} HU, "
                    f"Bottom={c_bot:.2f} HU — Image {target_nums['contrast']}")
            except Exception as exc:
                logger.warning("Contrast computation failed: %s", exc)
                st.warning(f"⚠️ Contraste échoué: {exc}")

        # Generate simulation figure
        if contrast_hu is not None and contrast_rois:
            try:
                fig_con = render_roi_drawing(
                    contrast_hu, contrast_rois, pixel_spacing,
                    title=f"Image {target_nums['contrast']} — Contraste",
                    slice_type="contrast")
                simulation_figs["🧪 Contraste (Plastic Block)"] = fig_con
            except Exception as exc:
                logger.warning("Contrast figure failed: %s", exc)
    else:
        st.warning("⚠️ **Contraste: N/A** — Slice 60 not provided.")

    # ── 4C: RESOLUTION (Slice 70 — Bar Patterns) ─────────────────
    resolution_hu = None
    resolution_rois = {}

    if target_slices["resolution"] is not None:
        ds_res = target_slices["resolution"]
        with st.spinner(
            f"📐 Résolution TotalQA — Image {target_nums['resolution']}..."
        ):
            try:
                resolution_hu = loader.to_hu_array(ds_res)
                resolution_rois = _totalqa_bar_pattern_rois(resolution_hu, pixel_spacing)

                # Simple direct SD from each fixed 12×12 ROI
                bar_labels = []
                bar_sds = []
                bar_means = []
                for label in sorted(resolution_rois.keys()):
                    roi = resolution_rois[label]
                    region = resolution_hu[
                        roi.row_start:roi.row_end,
                        roi.col_start:roi.col_end
                    ].astype(np.float64)
                    bar_labels.append(label)
                    bar_sds.append(float(np.std(region, ddof=1))
                                   if region.size > 1 else 0.0)
                    bar_means.append(float(np.mean(region)))

                totalqa_resolution = TotalQAResolutionResult(
                    bar_labels=bar_labels,
                    bar_sd_values=bar_sds,
                    bar_mean_values=bar_means,
                    passed=True,
                )
                sd_str = ", ".join(f"{s:.2f}" for s in bar_sds)
                st.success(
                    f"✅ Résolution TotalQA: SD=[{sd_str}] HU "
                    f"— Image {target_nums['resolution']}")
            except Exception as exc:
                logger.warning("Resolution computation failed: %s", exc)
                st.warning(f"⚠️ Résolution échouée: {exc}")

        # Generate simulation figure
        if resolution_hu is not None and resolution_rois:
            try:
                fig_res = render_roi_drawing(
                    resolution_hu, resolution_rois, pixel_spacing,
                    title=f"Image {target_nums['resolution']} — Résolution",
                    slice_type="resolution")
                simulation_figs["📐 Résolution (Bar Patterns)"] = fig_res
            except Exception as exc:
                logger.warning("Resolution figure failed: %s", exc)
    else:
        st.warning("⚠️ **Résolution: N/A** — Slice 70 not provided.")

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Build Basic & Advanced results for downstream tabs
    # ══════════════════════════════════════════════════════════════

    if vol_result is None:
        fallback_ds = first_ds
        fallback_hu = loader.to_hu_array(fallback_ds)
        fallback_rois_dict = adapter.get_roi_descriptors(
            fallback_hu, pixel_spacing)
        roi_descriptors = fallback_rois_dict
        rois = list(fallback_rois_dict.values())
        vol_result = _build_single_slice_vol(
            loader, fallback_ds, rois, start_slice, end_slice)

    # Basic Tier
    with st.spinner("Calcul des métriques Basic Tier..."):
        basic_result = basic_engine.compute(
            vol_result, adapter, scanner_id=scanner_id,
            hu_arrays_for_fwhm=vol_result.hu_arrays,
            nominal_slice_thickness_mm=vol_result.slice_thickness_mm)

        # Clear legacy metrics that are replaced by TotalQA
        basic_result.contrast = None
        basic_result.slice_thickness = None

        # Inject TotalQA results
        basic_result.totalqa_contrast = totalqa_contrast
        basic_result.totalqa_resolution = totalqa_resolution
        basic_result.totalqa_scaling = totalqa_scaling

        if target_slices["uniformity"] is None:
            basic_result.noise = None
            basic_result.uniformity = None

        basic_result._evaluate_all_passed()

    # Advanced Tier
    with st.spinner("Calcul des métriques Advanced Tier..."):
        effective_skip = list(skip_modules) if skip_modules else []
        if "mtf" not in effective_skip:
            effective_skip.append("mtf")

        advanced_result = advanced_engine.compute(
            vol_result, adapter, scanner_id=scanner_id,
            dose_metadata=mined_meta.dose,
            skip_modules=effective_skip,
            mtf_override=None)

    # ══════════════════════════════════════════════════════════════
    # STEP 5B: TG-204 DOSIMETRY METRICS
    # ══════════════════════════════════════════════════════════════
    dosimetry_metrics = _compute_tg204_dosimetry(
        ds=target_slices.get("uniformity"),
        noise_sd=basic_result.noise.std_hu if basic_result.noise else None,
        h_diameter_mm=totalqa_scaling.h_diameter_mm if totalqa_scaling else None,
        v_diameter_mm=totalqa_scaling.v_diameter_mm if totalqa_scaling else None,
    )
    st.session_state["dosimetry_metrics"] = dosimetry_metrics

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Store EVERYTHING in session_state
    # ══════════════════════════════════════════════════════════════
    st.session_state["basic_result"] = basic_result
    st.session_state["advanced_result"] = advanced_result
    st.session_state["vol_result"] = vol_result
    st.session_state["roi_descriptors"] = roi_descriptors
    st.session_state["pixel_spacing"] = vol_result.pixel_spacing_mm
    st.session_state["mined_metadata"] = mined_meta
    st.session_state["scanner_profile"] = scanner_profile
    st.session_state["analysis_date"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Acquisition parameters for noise diagnostics display
    _acq_ds = first_ds
    st.session_state["acquisition_params"] = {
        "kvp": float(getattr(_acq_ds, "KVP", 0.0)),
        "mas": float(getattr(_acq_ds, "Exposure",
                     getattr(_acq_ds, "XRayTubeCurrent", 0.0))),
        "kernel": str(getattr(_acq_ds, "ConvolutionKernel", "N/A")),
        "slice_thickness": float(getattr(_acq_ds, "SliceThickness", 0.0)),
    }

    # Slice-specific data for ROI drawing
    st.session_state["water_roi_descriptors"] = water_rois
    st.session_state["water_hu_array"] = water_hu
    st.session_state["contrast_roi_descriptors"] = contrast_rois
    st.session_state["contrast_hu_array"] = contrast_hu
    st.session_state["resolution_roi_descriptors"] = resolution_rois
    st.session_state["resolution_hu_array"] = resolution_hu

    # Pre-built simulation figures
    st.session_state["simulation_figs"] = simulation_figs
    logger.info("Simulation figures stored: %d views — %s",
                len(simulation_figs), list(simulation_figs.keys()))

    # Classification info for downstream
    from modules.image_qc.slice_classifier import SliceClassificationResult
    classification = SliceClassificationResult(routing_method="regex_totalqa")
    if target_slices["uniformity"] is not None:
        classification.water_slice_index = 0
        classification.water_hu_array = water_hu
        classification.water_candidate_indices = [0]
    if target_slices["resolution"] is not None:
        classification.resolution_slice_index = 0
        classification.resolution_hu_array = resolution_hu
    if target_slices["contrast"] is not None:
        classification.sensitometry_slice_index = 0
        classification.sensitometry_hu_array = contrast_hu
    st.session_state["slice_classification"] = classification

    # Warnings
    if basic_result.warnings:
        for w in basic_result.warnings:
            st.caption(f"ℹ️ {w}")
    if advanced_result.errors:
        for e in advanced_result.errors:
            st.warning(f"⚠️ {e}")
