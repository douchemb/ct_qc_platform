"""
app.py — CT QC Platform — Unified Clinical Dashboard.
Phase 7: Streamlit entry point.

Usage:
    streamlit run app.py

Architecture:
    This file is the SOLE entry point. All UI logic lives in dashboard/*.
    All clinical logic lives in core/ and modules/ — ZERO modifications.

Standards compliance:
    AAPM TG-66   — CT QC tolerances
    AAPM TG-220  — Water-equivalent diameter
    AAPM TG-233  — Noise power spectrum
    AAPM Report 204 — Size-specific dose estimate
    IAEA TRS-430 — ED calibration
"""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard.config_ui import apply_page_config
from dashboard.sidebar import render_sidebar_and_main


def main() -> None:
    """Streamlit application entry point."""
    apply_page_config()
    render_sidebar_and_main()


if __name__ == "__main__":
    main()
