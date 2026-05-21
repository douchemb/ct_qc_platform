"""
dashboard/helpers.py — Helper Utilities and Shared Components.
KPI cards, dark matplotlib style, figure export, upload handling,
and DICOM windowing utilities for GE Helios QA phantom rendering.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ═══════════════════════════════════════════════════════════════════
# DICOM Windowing Utilities — GE Helios QA Phantom Rendering
# ═══════════════════════════════════════════════════════════════════
# These functions apply radiological windowing (Window Level / Window
# Width) to produce high-contrast images matching commercial viewers
# like TotalQA, instead of the washed-out grayscale from raw pixels.
#
# Pipeline:  raw pixels → HU → windowed clip → uint8 (0–255)
#
# Default WL=50 / WW=400 is optimized for GE Helios water+plastic
# phantoms. Adjust per slice type:
#   Water/Uniformity:  WL=0,   WW=400
#   Contrast (plastic): WL=0,  WW=2500
#   Resolution (bars):  WL=0,  WW=2000
# ═══════════════════════════════════════════════════════════════════

def dicom_to_hu(ds) -> np.ndarray:
    """Convert raw DICOM pixel data to Hounsfield Units (HU).

    Extracts RescaleSlope (0028,1053) and RescaleIntercept (0028,1052)
    from the dataset and applies the standard DICOM affine transform:
        HU = RescaleSlope × StoredPixelValue + RescaleIntercept

    For GE Discovery RT datasets missing standard rescale tags, falls
    back to safe defaults (slope=1.0, intercept=-1024.0) matching the
    GE DICOM Conformance Statement.

    Parameters
    ----------
    ds : pydicom.Dataset
        Loaded DICOM dataset with pixel data.

    Returns
    -------
    np.ndarray
        Float32 2D array of HU values.
    """
    # Extract rescale parameters — GE-safe defaults
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))

    # Extract pixel array and handle multi-frame
    pixel_array = ds.pixel_array
    if pixel_array.ndim > 2:
        pixel_array = pixel_array[0]

    # Apply HU transform: DICOM PS3.3 C.11.1.1.2
    hu_array = slope * pixel_array.astype(np.float32) + intercept

    return hu_array


def apply_windowing(
    hu_array: np.ndarray,
    window_level: float = 50.0,
    window_width: float = 400.0,
) -> np.ndarray:
    """Apply radiological windowing to a HU array → uint8 for display.

    Implements the standard DICOM VOI LUT windowing transform:
        1. Compute bounds: lower = WL - WW/2,  upper = WL + WW/2
        2. Clip all HU values to [lower, upper] via numpy.clip
        3. Normalize linearly to [0, 255] and cast to uint8

    This produces the high-contrast images matching commercial QA
    viewers (TotalQA, SunCheck) instead of pale, washed-out renders.

    Parameters
    ----------
    hu_array : np.ndarray
        2D float array of Hounsfield Unit values.
    window_level : float
        Center of the display window in HU (default: 50 for QA phantoms).
    window_width : float
        Width of the display window in HU (default: 400 for water+plastic).

    Returns
    -------
    np.ndarray
        2D uint8 array (0–255), ready for st.image() or PIL rendering.

    Examples
    --------
    >>> img = apply_windowing(hu_array, window_level=0, window_width=400)
    >>> st.image(img, caption="Uniformité — WL=0 / WW=400")
    """
    # Compute window bounds
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0

    # Clip HU values to the display window
    clipped = np.clip(hu_array, lower, upper)

    # Normalize to [0, 255] — linear mapping
    if upper > lower:
        normalized = (clipped - lower) / (upper - lower) * 255.0
    else:
        # Degenerate case: zero-width window → flat gray
        normalized = np.full_like(clipped, 128.0)

    return normalized.astype(np.uint8)


def render_ge_dicom_image(
    ds,
    window_level: float = 50.0,
    window_width: float = 400.0,
    caption: str = "",
    use_column_width: bool = True,
) -> np.ndarray:
    """Full pipeline: DICOM dataset → windowed uint8 → st.image() display.

    Convenience function that chains dicom_to_hu() → apply_windowing()
    and renders directly in Streamlit. Returns the uint8 image for
    optional downstream use (overlay drawing, saving, etc.).

    Parameters
    ----------
    ds : pydicom.Dataset
        Loaded GE DICOM dataset with pixel data.
    window_level : float
        Window center in HU (default: 50).
    window_width : float
        Window width in HU (default: 400).
    caption : str
        Optional caption displayed below the image.
    use_column_width : bool
        If True, stretch image to column width in Streamlit.

    Returns
    -------
    np.ndarray
        The uint8 windowed image (for further processing if needed).

    Example — Streamlit integration
    --------------------------------
    >>> import pydicom
    >>> import streamlit as st
    >>> from dashboard.helpers import render_ge_dicom_image
    >>>
    >>> ds = pydicom.dcmread("CT.TPSQA2017.Image 36.dcm", force=True)
    >>> img = render_ge_dicom_image(
    ...     ds,
    ...     window_level=0,
    ...     window_width=400,
    ...     caption="GE Helios — Uniformité (WL=0 / WW=400)"
    ... )
    """
    # Step 1: Raw pixels → HU
    hu_array = dicom_to_hu(ds)

    # Step 2: HU → Windowed uint8
    windowed = apply_windowing(hu_array, window_level, window_width)

    # Step 3: Render in Streamlit
    display_caption = caption or (
        f"WL={window_level:.0f} HU | WW={window_width:.0f} HU"
    )
    st.image(windowed, caption=display_caption,
             use_column_width=use_column_width)

    return windowed


# ── Urgency color mapping — matches FailurePredictor.URGENCY_ORDER ────────
URGENCY_COLORS = {
    "breached": "#da3633", "critical": "#f85149",
    "warning": "#d29922", "monitor": "#c9a227",
    "stable": "#3fb950", "improving": "#56d364",
}
URGENCY_CSS = {
    "breached": "hw-critical", "critical": "hw-critical",
    "warning": "hw-warning", "monitor": "hw-monitor",
    "stable": "hw-stable", "improving": "hw-stable",
}
HARDWARE_ICONS = {
    "X-ray tube filament": "🔌",
    "Anode focal spot": "🎯",
    "kVp high-voltage generator": "⚡",
}


def save_uploads_to_tmp(uploaded_files: list) -> Path:
    """Saves UploadedFile objects to a temp directory. Returns the Path."""
    tmp = Path(tempfile.mkdtemp(prefix="ct_qc_"))
    for uf in uploaded_files:
        (tmp / uf.name).write_bytes(uf.read())
    return tmp


def fig_to_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    """Converts matplotlib Figure to PNG bytes for download buttons."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def fig_to_pdf_bytes(figures: list[tuple[str, plt.Figure]]) -> bytes:
    """Concatenates multiple figures into a single multi-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for title, fig in figures:
            fig.suptitle(title, fontsize=10, y=0.99)
            pdf.savefig(fig, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def render_fig(fig: plt.Figure, key: str, filename: str, caption: str = "") -> None:
    """Renders a matplotlib figure with an optional download button."""
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    st.download_button(
        "⬇️ PNG (300 DPI)", data=fig_to_bytes(fig),
        file_name=filename, mime="image/png", key=f"dl_{key}",
    )
    plt.close(fig)


def apply_dark_style(fig: plt.Figure, *axes) -> None:
    """Applies consistent dark theme to matplotlib figures."""
    bg = "#161b22"
    fig.patch.set_facecolor(bg)
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors="#8b949e", labelsize=9)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#21262d")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#f0f6fc")
        if ax.get_legend():
            ax.get_legend().get_frame().set_facecolor("#161b22")
            for t in ax.get_legend().get_texts():
                t.set_color("#8b949e")


def hdr(text: str) -> None:
    """Renders a section header."""
    st.markdown(f'<p class="section-hdr">{text}</p>', unsafe_allow_html=True)


def kpi_card(icon: str, label: str, value: str, sub: str,
             passed: Optional[bool]) -> str:
    """Returns HTML for a KPI Pass/Fail card."""
    if passed is None:
        css, icon = "kpi-na", "⚪"
    elif passed:
        css = "kpi-pass"
    else:
        css = "kpi-fail"
    val_color = "#3fb950" if passed else (
        "#da3633" if passed is not None else "#8b949e")
    return (
        f'<div class="{css}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{val_color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )
