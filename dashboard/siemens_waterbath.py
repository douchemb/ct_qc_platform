"""
dashboard/siemens_waterbath.py — Siemens Waterbath TotalQA Pipeline.

Self-contained analysis module for Siemens Water-bath phantoms.
ROI geometry follows TotalQA standard:
  - Center ROI: 40% of phantom diameter
  - Edge ROIs (Upper/Lower/Left/Right): 10% of phantom diameter

All slices are analyzed and results are stored per-slice for the
downstream Summary (3 tables) and Advanced (3 plots) tabs.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st

from dashboard.roi_drawing import render_roi_drawing
from modules.image_qc.roi_stats import ROIDescriptor

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════

@dataclass
class SiemensSliceResult:
    """Per-slice analysis result for Siemens Waterbath phantom."""
    image_number: int
    kernel: str
    kvp: float
    mas: float
    slice_thickness: float

    center_mean: float
    center_sd: float  # Noise

    edge_means: dict  # {upper, lower, left, right}
    edge_diffs: dict  # abs(edge_mean - center_mean)

    # All 5 ROI stats for plotting
    roi_means: dict  # {center, upper, lower, left, right}
    roi_sds: dict    # {center, upper, lower, left, right}


@dataclass
class SiemensWaterbathResult:
    """Aggregate result across all analyzed slices."""
    slices: list[SiemensSliceResult] = field(default_factory=list)
    simulation_fig: Optional[plt.Figure] = None
    noise_mean_fig: Optional[plt.Figure] = None
    h_profile_fig: Optional[plt.Figure] = None
    v_profile_fig: Optional[plt.Figure] = None
    nps_fig: Optional[plt.Figure] = None
    nps_peak_freq: float = 0.0
    nps_integral: float = 0.0


# ══════════════════════════════════════════════════════════════════
# Geometry & Metadata Extraction
# ══════════════════════════════════════════════════════════════════

def _detect_phantom_center_and_radius(hu_array: np.ndarray) -> tuple[int, int, float]:
    """Detect phantom center and radius using threshold at HU > -200."""
    mask = hu_array > -200.0
    ys, xs = np.where(mask)
    if len(ys) < 100:
        # Fallback to image center
        cr, cc = hu_array.shape[0] // 2, hu_array.shape[1] // 2
        return cr, cc, 100.0
    cr = int(np.mean(ys))
    cc = int(np.mean(xs))
    # Estimate radius from horizontal span
    row_mask = mask[cr, :]
    cols = np.where(row_mask)[0]
    radius = (cols[-1] - cols[0]) / 2.0 if len(cols) >= 2 else 100.0
    return cr, cc, radius


def _build_siemens_rois(
    hu_array: np.ndarray,
) -> tuple[dict[str, ROIDescriptor], int, int, float]:
    """Build TotalQA Siemens ROIs: 40% center, 10% edges at 80% radius.

    Returns (rois_dict, center_row, center_col, phantom_radius_px).
    """
    cr, cc, radius = _detect_phantom_center_and_radius(hu_array)

    # Center ROI: 40% of diameter = 40% of 2*radius
    center_size = int(radius * 2 * 0.40)
    center_half = center_size // 2

    # Edge ROI: 10% of diameter
    edge_size = int(radius * 2 * 0.10)
    edge_half = edge_size // 2

    # Edge placement: at 80% of radius from center
    edge_offset = int(radius * 0.80)

    rois = {
        "center": ROIDescriptor("center",
                                cr - center_half, cc - center_half,
                                center_size, center_size),
        "upper":  ROIDescriptor("upper",
                                cr - edge_offset - edge_half, cc - edge_half,
                                edge_size, edge_size),
        "lower":  ROIDescriptor("lower",
                                cr + edge_offset - edge_half, cc - edge_half,
                                edge_size, edge_size),
        "left":   ROIDescriptor("left",
                                cr - edge_half, cc - edge_offset - edge_half,
                                edge_size, edge_size),
        "right":  ROIDescriptor("right",
                                cr - edge_half, cc + edge_offset - edge_half,
                                edge_size, edge_size),
    }
    return rois, cr, cc, radius


def _extract_metadata(ds: pydicom.Dataset) -> dict:
    """Extract TotalQA-required DICOM metadata tags."""
    instance_num = int(getattr(ds, "InstanceNumber", 0))
    kernel = str(getattr(ds, "ConvolutionKernel", "N/A"))
    kvp = float(getattr(ds, "KVP", 0.0))
    current = float(getattr(ds, "XRayTubeCurrent", 0.0))
    exposure_time = float(getattr(ds, "ExposureTime", 0.0))
    # mAs = current (mA) * time (ms) / 1000 → but often stored as mAs directly
    mas_direct = getattr(ds, "Exposure", None)
    if mas_direct is not None:
        mas = float(mas_direct)
    else:
        mas = current * exposure_time / 1000.0 if exposure_time > 0 else current
    slice_thickness = float(getattr(ds, "SliceThickness", 0.0))
    return {
        "image_number": instance_num,
        "kernel": kernel,
        "kvp": kvp,
        "mas": mas,
        "slice_thickness": slice_thickness,
    }


def _compute_roi_stats(
    hu_array: np.ndarray, rois: dict[str, ROIDescriptor]
) -> tuple[dict, dict]:
    """Compute mean and SD for each ROI. Returns (means_dict, sds_dict)."""
    means = {}
    sds = {}
    rows, cols = hu_array.shape
    for label, roi in rois.items():
        r0 = max(0, roi.row_start)
        r1 = min(rows, roi.row_start + roi.height_px)
        c0 = max(0, roi.col_start)
        c1 = min(cols, roi.col_start + roi.width_px)
        region = hu_array[r0:r1, c0:c1].astype(np.float64)
        means[label] = float(np.mean(region)) if region.size > 0 else 0.0
        sds[label] = float(np.std(region, ddof=1)) if region.size > 1 else 0.0
    return means, sds


def _measure_siemens_diameter(
    hu_array: np.ndarray, pixel_spacing: tuple[float, float],
) -> tuple[float, float]:
    """Measure horizontal and vertical phantom diameter in mm.

    Bulletproof morphological approach:
      1. Threshold at HU > -300 (water/plastic are > -300, air ~ -1000).
      2. Extract the largest connected component (ignores CT table, artefacts).
      3. Fill internal holes (air bubbles, cradle gaps) with binary_fill_holes.
      4. Compute bounding box of the filled mask.
      5. Convert pixel extents to mm using pixel_spacing.

    This replaces the fragile line-profile approach that failed when
    internal air bubbles or cradle interference split the vertical mask.
    """
    from scipy import ndimage

    # Step 1: Binary threshold — isolate phantom from surrounding air
    binary_mask = hu_array > -300.0

    # Step 2: Connected component analysis — keep ONLY the largest blob
    labelled, n_features = ndimage.label(binary_mask)
    if n_features == 0:
        logger.warning("No phantom detected at HU > -300 threshold")
        return 0.0, 0.0

    # Find the largest component by voxel count
    component_sizes = ndimage.sum(binary_mask, labelled, range(1, n_features + 1))
    largest_label = int(np.argmax(component_sizes)) + 1
    largest_mask = labelled == largest_label

    # Step 3: Fill internal holes (air bubbles, cradle gaps)
    filled_mask = ndimage.binary_fill_holes(largest_mask)

    # Step 4: Bounding box of the filled mask → [min_row, max_row, min_col, max_col]
    rows_any = np.any(filled_mask, axis=1)
    cols_any = np.any(filled_mask, axis=0)
    row_indices = np.where(rows_any)[0]
    col_indices = np.where(cols_any)[0]

    if len(row_indices) < 2 or len(col_indices) < 2:
        logger.warning("Bounding box degenerate — phantom mask too small")
        return 0.0, 0.0

    min_row, max_row = int(row_indices[0]), int(row_indices[-1])
    min_col, max_col = int(col_indices[0]), int(col_indices[-1])

    # Step 5: Convert pixel extents to mm
    v_diameter_mm = (max_row - min_row) * pixel_spacing[0]
    h_diameter_mm = (max_col - min_col) * pixel_spacing[1]

    return h_diameter_mm, v_diameter_mm


# ══════════════════════════════════════════════════════════════════
# Plot Generation
# ══════════════════════════════════════════════════════════════════

def _apply_dark_transparent(fig, *axes):
    """Apply dark-theme transparent styling to figure and axes."""
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')


def _generate_noise_mean_plot(result: SiemensSliceResult) -> plt.Figure:
    """Bar chart: Mean HU (blue) and Noise/SD (black) for all 5 ROIs.
    Green lines at center_mean ± 4 HU tolerance bounds.
    """
    labels = ["Center", "Upper", "Lower", "Left", "Right"]
    keys = ["center", "upper", "lower", "left", "right"]
    means = [result.roi_means[k] for k in keys]
    sds = [result.roi_sds[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _apply_dark_transparent(fig, ax)

    bars_mean = ax.bar(x - width / 2, means, width, color='#4193EF',
                       label='Mean HU', zorder=3)
    bars_sd = ax.bar(x + width / 2, sds, width, color='#333333',
                     edgecolor='white', linewidth=0.5,
                     label='Noise (SD)', zorder=3)

    # Tolerance bounds: center mean ± 4 HU
    center_mean = result.center_mean
    ax.axhline(center_mean + 4, color='#3fb950', linestyle='--', linewidth=1.5,
               label=f'Upper Limit ({center_mean + 4:.1f} HU)', zorder=4)
    ax.axhline(center_mean - 4, color='#3fb950', linestyle='--', linewidth=1.5,
               label=f'Lower Limit ({center_mean - 4:.1f} HU)', zorder=4)
    ax.axhline(center_mean, color='#8b949e', linestyle=':', linewidth=1.0,
               alpha=0.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("HU", fontsize=10)
    ax.set_title("TotalQA — Noise and Mean Values", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, facecolor='none', labelcolor='white', framealpha=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='lightgray')
    fig.tight_layout()
    return fig


def _generate_profile_plot(
    hu_array: np.ndarray, cr: int, cc: int, direction: str
) -> plt.Figure:
    """Generate H or V profile plot with raw + smoothed curves."""
    if direction == "horizontal":
        # Average 5 rows centered on cr
        r0 = max(0, cr - 2)
        r1 = min(hu_array.shape[0], cr + 3)
        raw_profile = np.mean(hu_array[r0:r1, :].astype(np.float64), axis=0)
        title = "Horizontal Profile (Uniformity)"
        xlabel = "Column (px)"
    else:
        # Average 5 columns centered on cc
        c0 = max(0, cc - 2)
        c1 = min(hu_array.shape[1], cc + 3)
        raw_profile = np.mean(hu_array[:, c0:c1].astype(np.float64), axis=1)
        title = "Vertical Profile (Uniformity)"
        xlabel = "Row (px)"

    # Smoothed curve using convolution (fallback if scipy unavailable)
    try:
        from scipy.signal import savgol_filter
        smooth_profile = savgol_filter(raw_profile, window_length=31, polyorder=3)
    except ImportError:
        # Fallback: simple moving average
        kernel = np.ones(15) / 15
        smooth_profile = np.convolve(raw_profile, kernel, mode='same')

    fig, ax = plt.subplots(figsize=(8, 3.5))
    _apply_dark_transparent(fig, ax)

    ax.plot(raw_profile, color='#58a6ff', linewidth=0.8, alpha=0.6,
            label='Raw Profile')
    ax.plot(smooth_profile, color='#D32F2F', linewidth=2.0,
            label='Fitted Curve')

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("HU", fontsize=10)
    ax.set_title(f"TotalQA — {title}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, facecolor='none', labelcolor='white', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.2, color='lightgray')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# NPS — Noise Power Spectrum (Fourier)
# ══════════════════════════════════════════════════════════════════

def _calculate_nps(
    hu_array: np.ndarray, cr: int, cc: int, pixel_spacing: tuple[float, float],
    roi_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute 1D radial Noise Power Spectrum from a center ROI.

    Returns (freq_axis_lpmm, nps_1d, peak_freq, nps_integral).
    """
    half = roi_size // 2
    rows, cols = hu_array.shape

    # Clamp ROI to image bounds
    r0 = max(0, cr - half)
    r1 = min(rows, cr + half)
    c0 = max(0, cc - half)
    c1 = min(cols, cc + half)
    roi = hu_array[r0:r1, c0:c1].astype(np.float64)

    # ── Step 1: 2D Polynomial Detrending (order 2) ────────────────
    # Removes the beam-hardening cupping artifact that causes a
    # massive spike at low spatial frequencies in the NPS.
    # Coordinates normalized to [-1, 1] for numerical stability.
    nr, nc = roi.shape
    coords_r = np.linspace(-1.0, 1.0, nr)
    coords_c = np.linspace(-1.0, 1.0, nc)
    ii, jj = np.meshgrid(coords_r, coords_c, indexing='ij')
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()
    # Basis: [1, j, i, j^2, ij, i^2]
    basis = np.column_stack([
        np.ones_like(ii_flat),
        jj_flat, ii_flat,
        jj_flat ** 2, ii_flat * jj_flat, ii_flat ** 2,
    ])
    coeffs, _, _, _ = np.linalg.lstsq(basis, roi.ravel(), rcond=None)
    surface = (basis @ coeffs).reshape(nr, nc)
    roi = roi - surface

    # ── Step 2: Explicit DC zeroing ───────────────────────────────
    # Guarantees exactly zero mean after polynomial subtraction,
    # eliminating any residual DC leakage into the FFT.
    roi -= np.mean(roi)

    # ── Step 3: 2D Hanning window ─────────────────────────────────
    # Suppresses spectral leakage from the hard ROI edges that
    # causes high-frequency ripples and artificial jaggedness.
    window_1d_y = np.hanning(roi.shape[0])
    window_1d_x = np.hanning(roi.shape[1])
    window_2d = np.outer(window_1d_y, window_1d_x)
    roi = roi * window_2d

    # ── Step 4: 2D FFT → shift → squared magnitude ───────────────
    fft_2d = np.fft.fft2(roi)
    fft_shifted = np.fft.fftshift(fft_2d)
    power_2d = np.abs(fft_shifted) ** 2

    # Normalize by ROI area and pixel area
    dx = pixel_spacing[0]  # mm/pixel
    dy = pixel_spacing[1]
    pixel_area = dx * dy
    ny, nx = roi.shape
    norm = pixel_area / (nx * ny)
    power_2d *= norm

    # Radial average → 1D NPS
    cy_fft, cx_fft = ny // 2, nx // 2
    y_idx, x_idx = np.ogrid[:ny, :nx]
    r_map = np.sqrt((x_idx - cx_fft) ** 2 + (y_idx - cy_fft) ** 2).astype(int)
    max_r = min(cx_fft, cy_fft)

    nps_1d = np.zeros(max_r)
    for r in range(max_r):
        mask = r_map == r
        if np.any(mask):
            nps_1d[r] = np.mean(power_2d[mask])

    # Frequency axis (cycles/mm = lp/mm)
    freq_step = 1.0 / (nx * dx)  # cycles/mm per bin
    freq_axis = np.arange(max_r) * freq_step

    # Skip DC bin (index 0)
    freq_axis = freq_axis[1:]
    nps_1d = nps_1d[1:]

    # ── Low-frequency cutoff (AAPM TG-233 standard practice) ──────
    # Frequencies < 0.04 lp/mm represent macroscopic background
    # trends (cupping, beam hardening) — NOT quantum noise texture.
    # Zero them so they don't dominate the peak search or Y-axis.
    cutoff_freq = 0.04  # lp/mm
    low_freq_mask = freq_axis < cutoff_freq
    nps_1d[low_freq_mask] = 0.0

    # Peak frequency & integral (computed on filtered data)
    peak_idx = int(np.argmax(nps_1d)) if len(nps_1d) > 0 else 0
    peak_freq = float(freq_axis[peak_idx]) if len(freq_axis) > 0 else 0.0
    nps_integral = float(np.trapezoid(nps_1d, freq_axis)) if len(freq_axis) > 1 else 0.0

    return freq_axis, nps_1d, peak_freq, nps_integral


def _generate_nps_plot(
    freq_axis: np.ndarray, nps_1d: np.ndarray,
    peak_freq: float, nps_integral: float,
) -> plt.Figure:
    """Dark-themed 1D radial NPS plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_transparent(fig, ax)

    ax.plot(freq_axis, nps_1d, color='#1f6feb', linewidth=2.0,
            label='NPS 1D radial')
    ax.fill_between(freq_axis, nps_1d, alpha=0.10, color='#1f6feb')
    ax.axvline(peak_freq, color='#a371f7', linestyle='--', linewidth=1.5,
               label=f'f_peak = {peak_freq:.3f} lp/mm')

    ax.set_xlabel('Spatial Frequency (lp/mm)', fontsize=10)
    ax.set_ylabel('NPS (HU²·mm²)', fontsize=10)
    ax.set_title('TotalQA — Noise Power Spectrum (Water Phantom)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, facecolor='none', labelcolor='white', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.2, color='lightgray')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# Main Pipeline Entry Point
# ══════════════════════════════════════════════════════════════════

def run_siemens_analysis(
    valid_datasets: list[tuple[str, pydicom.Dataset]],
    scanner_id: str,
) -> None:
    """Full Siemens Waterbath TotalQA pipeline.

    Accepts pre-read (filename, dataset) tuples from the orchestrator.
    Analyzes ALL slices, generates per-slice metrics,
    simulation figures, and TotalQA plots. Stores everything in
    st.session_state for downstream tab rendering.
    """
    from datetime import datetime

    result = SiemensWaterbathResult()
    simulation_figs = {}

    if not valid_datasets:
        st.error("❌ No valid DICOM files found.")
        return

    st.info(f"🔬 **Siemens Waterbath Pipeline** — {len(valid_datasets)} slices detected")

    # Sort by InstanceNumber
    all_datasets = sorted(
        valid_datasets,
        key=lambda x: int(getattr(x[1], "InstanceNumber", 0)),
    )

    # Get pixel spacing from first dataset
    first_ds = all_datasets[0][1]
    pixel_spacing = (
        float(getattr(first_ds, "PixelSpacing", [1.0, 1.0])[0]),
        float(getattr(first_ds, "PixelSpacing", [1.0, 1.0])[1]),
    )

    first_hu = None
    first_rois = None
    first_cr, first_cc = 0, 0

    with st.spinner("⚖️ Analyzing all slices (Siemens Waterbath)..."):
        for filename, ds in all_datasets:
            try:
                # Convert to HU
                intercept = float(getattr(ds, "RescaleIntercept", 0))
                slope = float(getattr(ds, "RescaleSlope", 1))
                pixel_data = ds.pixel_array.astype(np.float64)
                hu_array = pixel_data * slope + intercept

                # Build ROIs
                rois, cr, cc, radius = _build_siemens_rois(hu_array)

                # Compute stats
                means, sds = _compute_roi_stats(hu_array, rois)

                # Extract metadata
                meta = _extract_metadata(ds)

                # Compute edge diffs
                center_mean = means["center"]
                edge_diffs = {
                    k: abs(means[k] - center_mean)
                    for k in ["upper", "lower", "left", "right"]
                }

                slice_result = SiemensSliceResult(
                    image_number=meta["image_number"],
                    kernel=meta["kernel"],
                    kvp=meta["kvp"],
                    mas=meta["mas"],
                    slice_thickness=meta["slice_thickness"],
                    center_mean=center_mean,
                    center_sd=sds["center"],
                    edge_means={k: means[k] for k in
                                ["upper", "lower", "left", "right"]},
                    edge_diffs=edge_diffs,
                    roi_means=means,
                    roi_sds=sds,
                )
                result.slices.append(slice_result)

                # Save first slice data for figures
                if first_hu is None:
                    first_hu = hu_array
                    first_rois = rois
                    first_cr, first_cc = cr, cc

            except Exception as exc:
                logger.warning("Failed to analyze %s: %s", filename, exc)

    if not result.slices:
        st.error("❌ No slices could be analyzed.")
        return

    n = len(result.slices)
    st.success(f"✅ Siemens Waterbath: {n} slice(s) analyzed")

    # ── Generate simulation figure (first slice) ──────────────────
    if first_hu is not None and first_rois:
        try:
            fig_sim = render_roi_drawing(
                first_hu, first_rois, pixel_spacing,
                title=f"Siemens Waterbath — ROIs (Slice {result.slices[0].image_number})",
                slice_type="water")
            simulation_figs["💧 Siemens Waterbath ROIs"] = fig_sim
        except Exception as exc:
            logger.warning("Simulation figure failed: %s", exc)

    # ── Generate TotalQA plots (using first slice) ────────────────
    first_slice = result.slices[0]
    result.noise_mean_fig = _generate_noise_mean_plot(first_slice)
    result.h_profile_fig = _generate_profile_plot(
        first_hu, first_cr, first_cc, "horizontal")
    result.v_profile_fig = _generate_profile_plot(
        first_hu, first_cr, first_cc, "vertical")

    # NPS — Fourier noise texture analysis
    try:
        freq_axis, nps_1d, peak_freq, nps_int = _calculate_nps(
            first_hu, first_cr, first_cc, pixel_spacing)
        result.nps_peak_freq = peak_freq
        result.nps_integral = nps_int
        result.nps_fig = _generate_nps_plot(
            freq_axis, nps_1d, peak_freq, nps_int)
        st.success(f"✅ NPS: peak={peak_freq:.3f} lp/mm, "
                   f"integral={nps_int:.4f} HU²·mm²")
    except Exception as exc:
        logger.warning("NPS computation failed: %s", exc)
        st.warning(f"⚠️ NPS computation failed: {exc}")

    # ══════════════════════════════════════════════════════════════
    # Compute 4 KPI Metrics (Noise, Uniformity, HU Precision, Scaling)
    # ══════════════════════════════════════════════════════════════
    siemens_kpi = {}
    if result.slices:
        s0 = result.slices[0]
        # KPI 1: Noise — SD of center ROI
        siemens_kpi["noise_sd"] = s0.center_sd
        siemens_kpi["noise_passed"] = s0.center_sd <= 5.0

        # KPI 2: Uniformity — max |edge - center| (already computed as edge_diffs)
        max_diff = max(s0.edge_diffs.values()) if s0.edge_diffs else 0.0
        worst_edge = max(s0.edge_diffs, key=s0.edge_diffs.get) if s0.edge_diffs else "N/A"
        siemens_kpi["uniformity_nui"] = max_diff
        siemens_kpi["uniformity_passed"] = max_diff <= 5.0
        siemens_kpi["uniformity_worst"] = worst_edge

        # KPI 3: HU Precision — |center_mean - 0| (water should be 0 HU)
        hu_delta = abs(s0.center_mean)
        siemens_kpi["hu_precision_delta"] = hu_delta
        siemens_kpi["hu_precision_passed"] = hu_delta <= 4.0

        # KPI 4: Scaling — phantom diameter H/V via threshold edge detection
        if first_hu is not None:
            try:
                h_mm, v_mm = _measure_siemens_diameter(first_hu, pixel_spacing)
                siemens_kpi["scaling_h_mm"] = h_mm
                siemens_kpi["scaling_v_mm"] = v_mm
                siemens_kpi["scaling_nominal_mm"] = 200.0
                siemens_kpi["scaling_passed"] = (
                    abs(h_mm - 200.0) <= 2.0 and abs(v_mm - 200.0) <= 2.0
                )
            except Exception as exc:
                logger.warning("Siemens scaling measurement failed: %s", exc)
                siemens_kpi["scaling_h_mm"] = 0.0
                siemens_kpi["scaling_v_mm"] = 0.0
                siemens_kpi["scaling_nominal_mm"] = 200.0
                siemens_kpi["scaling_passed"] = False

    # ══════════════════════════════════════════════════════════════
    # TG-204 Dosimetry Metrics
    # ══════════════════════════════════════════════════════════════
    from dashboard.orchestrator import _compute_tg204_dosimetry
    dosimetry_metrics = _compute_tg204_dosimetry(
        ds=first_ds,
        noise_sd=siemens_kpi.get("noise_sd"),
        h_diameter_mm=siemens_kpi.get("scaling_h_mm"),
        v_diameter_mm=siemens_kpi.get("scaling_v_mm"),
    )

    # ══════════════════════════════════════════════════════════════
    # Store in session_state
    # ══════════════════════════════════════════════════════════════
    st.session_state["manufacturer"] = "SIEMENS"
    st.session_state["siemens_result"] = result
    st.session_state["siemens_kpi_metrics"] = siemens_kpi
    st.session_state["dosimetry_metrics"] = dosimetry_metrics
    st.session_state["simulation_figs"] = simulation_figs
    st.session_state["analysis_date"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Acquisition parameters for noise diagnostics display
    if result.slices:
        s0 = result.slices[0]
        st.session_state["acquisition_params"] = {
            "kvp": s0.kvp,
            "mas": s0.mas,
            "kernel": s0.kernel,
            "slice_thickness": s0.slice_thickness,
        }

    # Create minimal basic_result stub for downstream compatibility
    st.session_state["basic_result"] = _build_stub_basic_result(
        result, scanner_id, first_ds)
    st.session_state["advanced_result"] = _build_stub_advanced_result()
    st.session_state["roi_descriptors"] = first_rois or {}
    st.session_state["pixel_spacing"] = pixel_spacing
    st.session_state["vol_result"] = None


def _build_stub_basic_result(
    siemens_result: SiemensWaterbathResult,
    scanner_id: str,
    ds: pydicom.Dataset,
):
    """Build a minimal BasicQAResult stub for Siemens compatibility."""
    from modules.image_qc.basic_metrics import BasicQAResult

    acq_date = str(getattr(ds, "AcquisitionDate", "unknown"))
    series_desc = str(getattr(ds, "SeriesDescription", "Siemens Waterbath"))

    return BasicQAResult(
        acquisition_date=acq_date,
        series_description=series_desc,
        scanner_id=scanner_id,
        phantom_id="siemens_waterbath",
        n_slices_analyzed=len(siemens_result.slices),
        all_passed=True,
        noise=None,
        uniformity=None,
        ct_number_accuracy=None,
        contrast=None,
        slice_thickness=None,
        totalqa_contrast=None,
        totalqa_resolution=None,
        totalqa_scaling=None,
        warnings=[],
    )


def _build_stub_advanced_result():
    """Build a minimal AdvancedQAResult stub for Siemens compatibility."""
    from modules.image_qc.advanced_metrics_engine import AdvancedQAResult
    return AdvancedQAResult(
        acquisition_date="",
        series_description="Siemens Waterbath",
        scanner_id="",
        phantom_id="siemens_waterbath",
        nps=None, mtf=None, hu_linearity=None,
        ed_calibration=None, ssde_series=None, dw_series=None,
        skipped=["All — Siemens Waterbath mode"],
        errors=[],
    )

