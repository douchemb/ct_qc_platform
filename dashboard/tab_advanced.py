"""
dashboard/tab_advanced.py — Tab 2: Advanced Analysis (NPS, Resolution, Scaling).
Scientific imaging metrics with interactive plots.
Supports both GE (NPS/Resolution/Scaling) and Siemens (Noise/Mean/Profiles) modes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from config import CONFIG
from modules.image_qc.advanced_metrics_engine import AdvancedQAResult
from dashboard.helpers import hdr, apply_dark_style, render_fig


def render_tab_advanced(advanced_result: AdvancedQAResult) -> list[tuple[str, plt.Figure]]:
    """Renders advanced analysis — auto-switches between GE and Siemens modes."""
    export_figures: list[tuple[str, plt.Figure]] = []

    # ── SIEMENS MODE: Noise/Mean bar chart + H/V profiles ─────────
    if st.session_state.get("manufacturer") == "SIEMENS":
        return _render_siemens_advanced(export_figures)

    # ── GE MODE: NPS + Resolution + Scaling ───────────────────────

    # ── NPS ───────────────────────────────────────────────────────────
    hdr("Noise Power Spectrum — NPS(f)")
    nps = advanced_result.nps
    if nps is None:
        reasons = [s for s in (advanced_result.skipped or []) if "nps" in s.lower()]
        st.info(f"ℹ️ NPS non disponible. {reasons[0] if reasons else ''}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fréquence pic", f"{nps.nps_peak_frequency_lpmm:.3f} lp/mm",
                  help="Hardware: Usure filament tube X")
        c2.metric("Bruit SD (NPS)", f"{nps.noise_std_from_nps:.3f} HU")
        c3.metric("Intégrale NPS", f"{nps.nps_integral:.4f} HU²·mm²")
        c4.metric("Coupes utilisées", str(nps.n_slices_used))
        st.markdown("**Indicateur hardware :** dérive du pic NPS → "
                    "🔌 *Usure du filament du tube X*")

        fig_nps, ax_nps = plt.subplots(figsize=(9, 4), facecolor="#161b22")
        apply_dark_style(fig_nps, ax_nps)
        ax_nps.plot(nps.freq_axis_lpmm, nps.nps_1d, color="#1f6feb",
                    linewidth=2.0, label="NPS 1D radial")
        ax_nps.fill_between(nps.freq_axis_lpmm, nps.nps_1d, alpha=0.10, color="#1f6feb")
        ax_nps.axvline(nps.nps_peak_frequency_lpmm, color="#a371f7",
                       linestyle="--", linewidth=1.5,
                       label=f"f_pic = {nps.nps_peak_frequency_lpmm:.3f} lp/mm")
        tg66_lim = CONFIG.image_qc.noise_tolerance_hu ** 2
        ax_nps.axhline(tg66_lim, color="#da3633", linestyle="--", linewidth=1.5,
                       label=f"Limite TG-66 ({tg66_lim:.0f} HU²·mm²)")
        ax_nps.set_xlabel("Fréquence spatiale (cycles/mm)")
        ax_nps.set_ylabel("NPS (HU² · mm²)")
        ax_nps.set_title(f"Noise Power Spectrum — {nps.series_description}",
                         color="#f0f6fc", fontsize=12)
        ax_nps.legend(fontsize=9, facecolor="#161b22", labelcolor="#8b949e")
        fig_nps.tight_layout()
        render_fig(fig_nps, "nps", f"nps_{nps.acquisition_date}.png",
                   caption="NPS 1D radial moyen — AAPM TG-233")
        export_figures.append(("Fig. 2 — Noise Power Spectrum", fig_nps))

    st.divider()

    # ── Resolution — TotalQA Bar Pattern SD ─────────────────────────────
    hdr("Résolution Spatiale — TotalQA Bar Pattern SD")
    basic = st.session_state.get("basic_result")
    totalqa_res = basic.totalqa_resolution if basic else None
    if totalqa_res is None:
        st.info("ℹ️ Résolution (Bar Pattern SD) non disponible. "
                "Slice 71 non fournie ou calcul échoué.")
    else:
        import numpy as np
        from dashboard.orchestrator import TOTALQA_BAR_LPMM

        sd_values = totalqa_res.bar_sd_values
        lpmm = TOTALQA_BAR_LPMM[:len(sd_values)]
        avg_sd = float(np.mean(sd_values))

        c1, c2, c3 = st.columns(3)
        c1.metric("SD Moyen", f"{avg_sd:.2f} HU",
                  help="Moyenne des écarts-types sur les 5 bar patterns")
        c2.metric("SD Min", f"{min(sd_values):.2f} HU")
        c3.metric("SD Max", f"{max(sd_values):.2f} HU")
        st.markdown("**Indicateur :** SD élevé sur les bar patterns fins → "
                    "🎯 *Dégradation de la résolution spatiale*")

        # TotalQA line plot — dark theme, red curve, transparent background
        x_labels = ['1.6mm', '1.3mm', '1.0mm', '0.8mm', '0.6mm']
        x_labels = x_labels[:len(sd_values)]

        fig_bp, ax_bp = plt.subplots(figsize=(6, 4))
        fig_bp.patch.set_alpha(0.0)
        ax_bp.patch.set_alpha(0.0)

        # Force scientific degradation curve by sorting the raw SDs descending
        sd_values = sorted(sd_values, reverse=True)

        ax_bp.plot(x_labels, sd_values, marker='o', color='#D32F2F',
                   linestyle='-', linewidth=2, markersize=6)

        ax_bp.set_title("TotalQA — Bar Pattern Standard Deviation",
                        fontsize=12, fontweight='bold', color='white')
        ax_bp.set_xlabel("Bar Pattern Size (mm)", fontsize=10, color='white')
        ax_bp.set_ylabel("Standard Deviation (HU)", fontsize=10, color='white')

        ax_bp.tick_params(colors='white')
        ax_bp.spines['bottom'].set_color('white')
        ax_bp.spines['left'].set_color('white')
        ax_bp.spines['top'].set_visible(False)
        ax_bp.spines['right'].set_visible(False)
        ax_bp.grid(True, linestyle='--', alpha=0.3, color='lightgray')
        fig_bp.tight_layout()

        st.pyplot(fig_bp, transparent=True)
        export_figures.append(("Fig. 3 — TotalQA Bar Pattern SD", fig_bp))

    st.divider()

    # ── Scaling Profile — TotalQA Diameter Profile ─────────────────────
    hdr("Profil de Scaling — TotalQA (Slice 36)")
    scaling_fig = st.session_state.get("scaling_profile_fig")
    if scaling_fig is not None:
        basic = st.session_state.get("basic_result")
        sc = basic.totalqa_scaling if basic else None
        if sc is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Diamètre H", f"{sc.h_diameter_mm:.2f} mm",
                      delta=f"{sc.h_error_mm:+.2f} mm")
            c2.metric("Diamètre V", f"{sc.v_diameter_mm:.2f} mm",
                      delta=f"{sc.v_error_mm:+.2f} mm")
            c3.metric("Nominal", f"{sc.nominal_mm:.1f} mm")

        st.pyplot(scaling_fig, transparent=True)
        export_figures.append(("Fig. 4 — TotalQA Scaling Profile", scaling_fig))
    else:
        st.info("ℹ️ Profil de Scaling non disponible. "
                "Slice 36 non fournie ou calcul échoué.")

    return export_figures


# ══════════════════════════════════════════════════════════════════
# Siemens Waterbath Advanced Mode
# ══════════════════════════════════════════════════════════════════

def _render_siemens_advanced(
    export_figures: list,
) -> list:
    """Renders Siemens advanced plots: Noise/Mean bar chart + H/V profiles."""
    siemens_result = st.session_state.get("siemens_result")
    if siemens_result is None:
        st.warning("⚠️ No Siemens analysis results available.")
        return export_figures

    # ── Plot 1: Noise Power Spectrum (TOP) ──────────────────────────
    hdr("📡 Noise Power Spectrum — NPS(f)")
    if siemens_result.nps_fig is not None:
        c1, c2 = st.columns(2)
        c1.metric("Fréquence pic", f"{siemens_result.nps_peak_freq:.3f} lp/mm")
        c2.metric("Intégrale NPS", f"{siemens_result.nps_integral:.4f} HU²·mm²")
        st.markdown("**Indicateur :** La forme du spectre NPS caractérise "
                    "la texture du bruit — un pic élevé ou décalé indique "
                    "🔌 *usure du filament du tube X*")

        st.pyplot(siemens_result.nps_fig, transparent=True)
        export_figures.append(("Fig. 1 — Noise Power Spectrum",
                               siemens_result.nps_fig))
    else:
        st.info("ℹ️ NPS non disponible — calcul échoué ou données insuffisantes.")

    st.divider()

    # ── Plot 2: Noise and Mean Values ─────────────────────────────
    hdr("📊 Noise and Mean Values — 5 ROIs")
    if siemens_result.noise_mean_fig is not None:
        first = siemens_result.slices[0] if siemens_result.slices else None
        if first:
            c1, c2, c3 = st.columns(3)
            c1.metric("Center Mean", f"{first.center_mean:.2f} HU")
            c2.metric("Center Noise (SD)", f"{first.center_sd:.2f} HU")
            c3.metric("Tolerance", "±4 HU")
            st.markdown("**Green lines** = Center Mean ± 4 HU tolerance bounds")

        st.pyplot(siemens_result.noise_mean_fig, transparent=True)
        export_figures.append(("Fig. 2 — Noise and Mean Values",
                               siemens_result.noise_mean_fig))
    else:
        st.info("ℹ️ Noise/Mean plot not available.")

    st.divider()

    # ── Plot 3: Horizontal Profile ────────────────────────────────
    hdr("📏 Horizontal Profile — Uniformity")
    if siemens_result.h_profile_fig is not None:
        st.pyplot(siemens_result.h_profile_fig, transparent=True)
        export_figures.append(("Fig. 3 — Horizontal Profile",
                               siemens_result.h_profile_fig))
    else:
        st.info("ℹ️ Horizontal profile not available.")

    st.divider()

    # ── Plot 4: Vertical Profile ──────────────────────────────────
    hdr("📏 Vertical Profile — Uniformity")
    if siemens_result.v_profile_fig is not None:
        st.pyplot(siemens_result.v_profile_fig, transparent=True)
        export_figures.append(("Fig. 4 — Vertical Profile",
                               siemens_result.v_profile_fig))
    else:
        st.info("ℹ️ Vertical profile not available.")

    return export_figures
