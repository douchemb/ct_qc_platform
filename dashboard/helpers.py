"""
dashboard/helpers.py — Helper Utilities and Shared Components.
KPI cards, dark matplotlib style, figure export, upload handling.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import streamlit as st


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
