"""
dashboard/roi_drawing.py — Live ROI Drawing Component.
Shows user exactly where each ROI is placed on the phantom image.

Slice-aware rendering (TotalQA-aligned):
  - Water slice:        5 clock-face uniformity ROIs, WC=0 / WW=400
  - Contrast slice:     4 rectangular A/B/C/D ROIs,  WC=0  / WW=2500
  - Resolution slice:   5 bar-pattern square ROIs,   WC=0  / WW=2000
  - Default/unknown:    all provided ROIs,           WC=0  / WW=400

Reference: Image Owl TotalQA GE Set 1 report.
"""
from __future__ import annotations

import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

logger = logging.getLogger(__name__)

ROI_COLORS = {
    "center": "#00d4ff", "center_water": "#00d4ff", "water": "#00d4ff",
    "peripheral_12": "#ffd700", "peripheral_3": "#ffd700",
    "peripheral_6": "#ffd700", "peripheral_9": "#ffd700",
    # Contrast zones (TotalQA)
    "top_plastic": "#ff6b6b", "top_water": "#51cf66",
    "bottom_plastic": "#cc5de8", "bottom_water": "#ff922b",
    # Resolution bar patterns (TotalQA)
    "bar_1": "#ff0000", "bar_2": "#ff6b6b", "bar_3": "#ffd700",
    "bar_4": "#51cf66", "bar_5": "#74c0fc",
}

# Window/level presets per slice type
_WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    "water":      (0,   400),
    "contrast":   (0,   2500),
    "resolution": (0,   2000),
}

# Human-readable labels for slice type display
_SLICE_TYPE_LABELS: dict[str, str] = {
    "water":      "Coupe Uniformité (Water)",
    "contrast":   "Coupe Contraste (Plastic Block)",
    "resolution": "Coupe Résolution (Bar Patterns)",
}


def _detect_phantom_center(hu_array: np.ndarray) -> tuple[float, float]:
    """Returns the HARDCODED phantom center (254, 254).

    The physical phantom does not move between slices. Dynamic contour-based
    detection was getting fooled by the patient couch on geometry slices
    (returned 439, 252 instead of 254, 254). All ROI geometry is calibrated
    against this fixed center.

    Returns (row_center, col_center).
    """
    return 254.0, 254.0


def render_roi_drawing(
    hu_array: np.ndarray,
    roi_descriptors: dict,
    pixel_spacing_mm: tuple,
    title: str = "ROI Placement",
    slice_type: str | None = None,
) -> plt.Figure:
    """
    Renders the phantom CT image with all ROI positions drawn as colored
    rectangles. Background: HU image with slice-type-appropriate window.

    Parameters
    ----------
    hu_array : 2D HU image array
    roi_descriptors : dict[str, ROIDescriptor] — ROIs to draw
    pixel_spacing_mm : (row_spacing, col_spacing) in mm
    title : plot title prefix
    slice_type : "water" | "contrast" | "resolution" | None
        Controls window/level preset and title badge.
    """
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#161b22")
    ax.set_facecolor("#0d1117")

    # ── Window/level from slice type ──────────────────────────────────
    wc, ww = _WINDOW_PRESETS.get(slice_type or "", (0, 400))
    ax.imshow(hu_array, cmap="gray", vmin=wc - ww / 2, vmax=wc + ww / 2,
              origin="upper", aspect="equal")

    # ── Dynamic phantom boundary circle ───────────────────────────────
    rows, cols = hu_array.shape
    center_row, center_col = _detect_phantom_center(hu_array)
    px_avg = (pixel_spacing_mm[0] + pixel_spacing_mm[1]) / 2.0
    phantom_r_px = 100.0 / px_avg
    # NOTE: matplotlib uses (x, y) = (col, row) for patches
    circle = plt.Circle((center_col, center_row), phantom_r_px, fill=False,
                         edgecolor="#ffffff", linewidth=1.0,
                         linestyle="--", alpha=0.4)
    ax.add_patch(circle)

    # ── Center crosshair (small + at detected center) ─────────────────
    ch_len = 6  # pixels
    ax.plot([center_col - ch_len, center_col + ch_len],
            [center_row, center_row], color="#ffffff", lw=0.8, alpha=0.35)
    ax.plot([center_col, center_col],
            [center_row - ch_len, center_row + ch_len],
            color="#ffffff", lw=0.8, alpha=0.35)

    # ── Draw each ROI ─────────────────────────────────────────────────
    legend_handles = []
    drawn_labels = set()

    for label, roi in roi_descriptors.items():
        color = ROI_COLORS.get(label.lower(), "#ffd700")
        rect = mpatches.Rectangle(
            (roi.col_start, roi.row_start), roi.width_px, roi.height_px,
            linewidth=1.8, edgecolor=color, facecolor=color, alpha=0.18,
        )
        ax.add_patch(rect)
        rect_border = mpatches.Rectangle(
            (roi.col_start, roi.row_start), roi.width_px, roi.height_px,
            linewidth=1.8, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect_border)

        rc = roi.row_start + roi.height_px / 2
        cc = roi.col_start + roi.width_px / 2

        # Label rendering
        display_label = label.replace("_", "\n").upper()
        if label.lower().startswith("bar_"):
            display_label = label.replace("bar_", "B")
        ax.text(cc, rc, display_label,
                ha="center", va="center", fontsize=7, color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1117",
                          edgecolor="none", alpha=0.7))

        if label not in drawn_labels:
            legend_handles.append(
                mpatches.Patch(facecolor=color, alpha=0.7, label=label))
            drawn_labels.add(label)

    # ── Scale bar: 50 mm ──────────────────────────────────────────────
    scale_px = 50.0 / px_avg
    bar_x, bar_y = cols * 0.08, rows * 0.93
    ax.plot([bar_x, bar_x + scale_px], [bar_y, bar_y], "w-", linewidth=2.5)
    ax.text(bar_x + scale_px / 2, bar_y - rows * 0.02, "50 mm",
            ha="center", va="bottom", fontsize=8, color="white")

    # ── Title with slice-type badge ───────────────────────────────────
    slice_badge = ""
    if slice_type and slice_type in _SLICE_TYPE_LABELS:
        slice_badge = f"  │  {_SLICE_TYPE_LABELS[slice_type]}"

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_title(
        f"{title}{slice_badge}\n"
        f"WC={wc} HU  WW={ww} HU  |  {len(roi_descriptors)} ROIs  |  "
        f"Centre=({center_row:.0f}, {center_col:.0f})",
        color="#f0f6fc", fontsize=10, pad=8)
    ax.set_xlabel(f"Pixel (spacing {pixel_spacing_mm[1]:.3f} mm)",
                  color="#8b949e", fontsize=9)
    ax.set_ylabel(f"Pixel (spacing {pixel_spacing_mm[0]:.3f} mm)",
                  color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_color("#21262d")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
                  facecolor="#161b22", labelcolor="#c9d1d9",
                  framealpha=0.85, ncol=2)

    fig.tight_layout()
    return fig
