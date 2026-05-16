"""
dashboard/sidebar.py — Sidebar Navigation, Upload, and Main Router.
Wires all tabs together and handles archive persistence.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import streamlit as st

from modules.predictive.metrics_archive import QCSessionRecord
from dashboard.cached_resources import get_archive
from dashboard.orchestrator import run_full_analysis
from dashboard.tab_summary import render_tab_summary
from dashboard.tab_advanced import render_tab_advanced
from dashboard.tab_dosimetry import render_tab_dosimetry
from dashboard.tab_predictive import render_tab_predictive
from dashboard.helpers import hdr, fig_to_pdf_bytes


def render_sidebar_and_main() -> None:
    """Main entry point: renders sidebar + content area."""
    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 🏥 CT QC Platform")
        st.caption("Phase 7 — Dashboard Clinique Unifié")
        st.divider()

        uploaded = st.file_uploader(
            "📂 Séries DICOM (tous formats)",
            accept_multiple_files=True,
            help="Importer les fichiers DICOM (.dcm ou sans extension).")

        st.divider()
        st.markdown("### ⚙️ Paramètres")
        scanner_id = st.selectbox(
            "🏥 Choisir le Scanner",
            options=["Scanner SIEMENS (SOMATOM)", "Scanner GE (Discovery RT)"],
            index=0,
            help="Sélectionnez le scanner pour le rapport. "
                 "La détection réelle se fait automatiquement via les en-têtes DICOM.",
        )
        operator_id = st.selectbox(
            "👨‍⚕️ Profil Opérateur",
            options=["Technicien Biomédical", "Physicien Médical",
                     "Ingénieur d'Application"],
            index=0,
        )
        phantom_type = st.selectbox(
            "🧪 Fantôme utilisé",
            options=["Siemens Waterbath", "GE Helios QA", "Générique / Autre"],
            index=0,
            help="Information visuelle. Le pipeline s'adapte "
                 "automatiquement au fabricant détecté.",
        )
        c1, c2 = st.columns(2)
        start_slice = c1.number_input("Coupe début", 1, 999, 1)
        end_slice = c2.number_input("Coupe fin", 1, 999, 999)

        with st.expander("🔧 Modules à afficher", expanded=False):
            show_qa = st.checkbox("📊 Résumé QA", value=True)
            show_adv = st.checkbox("🔬 Analyse Avancée", value=True)
            show_dosi = st.checkbox("💊 Dosimétrie", value=True)
            show_health = st.checkbox("🩺 État de l'appareil", value=True)
            show_pred = st.checkbox("🧠 Prédictif", value=True)

        st.divider()
        run = st.button("🚀 Lancer l'analyse", type="primary",
                        disabled=not uploaded, use_container_width=True)

        if run and uploaded:
            run_full_analysis(uploaded, start_slice, end_slice,
                              scanner_id, [])

        # Archive session option
        if "basic_result" in st.session_state:
            st.divider()
            if st.button("💾 Archiver cette session", use_container_width=True):
                _archive_current_session(scanner_id, operator_id)

    # ── Main Content ───────────────────────────────────────────────────
    if "basic_result" not in st.session_state:
        _render_welcome()
        return

    basic = st.session_state["basic_result"]
    advanced = st.session_state["advanced_result"]
    vol = st.session_state["vol_result"]
    rois = st.session_state["roi_descriptors"]
    pxsp = st.session_state["pixel_spacing"]
    mined = st.session_state.get("mined_metadata")
    scanner_id_val = basic.scanner_id

    all_export_figs: list[tuple[str, "plt.Figure"]] = []

    # ── Build dynamic tab list based on sidebar visibility toggles ───
    tab_registry = []
    if show_qa:
        tab_registry.append(("📊 Résumé QA", "qa"))
    if show_adv:
        tab_registry.append(("🔬 Analyse Avancée", "adv"))
    if show_dosi:
        tab_registry.append(("💊 Dosimétrie", "dosi"))
    if show_health:
        tab_registry.append(("🩺 État de l'appareil", "health"))
    if show_pred:
        tab_registry.append(("🧠 Prédictif", "pred"))

    if not tab_registry:
        st.warning(
            "⚠️ Tous les modules sont masqués. "
            "Veuillez cocher au moins un module dans le menu latéral."
        )
        return

    tab_labels = [label for label, _ in tab_registry]
    tab_keys = [key for _, key in tab_registry]
    created_tabs = st.tabs(tab_labels)

    for tab_widget, tab_key in zip(created_tabs, tab_keys):
        with tab_widget:
            if tab_key == "qa":
                figs = render_tab_summary(basic, vol, rois, pxsp)
                all_export_figs.extend(figs)
            elif tab_key == "adv":
                figs = render_tab_advanced(advanced)
                all_export_figs.extend(figs)
            elif tab_key == "dosi":
                figs = render_tab_dosimetry(advanced, mined)
                all_export_figs.extend(figs)
            elif tab_key == "health":
                from dashboard.tab_hardware_health import render_tab_hardware_health
                figs = render_tab_hardware_health()
                all_export_figs.extend(figs)
            elif tab_key == "pred":
                figs = render_tab_predictive(scanner_id_val)
                all_export_figs.extend(figs)

    # ── Multi-page PDF export ─────────────────────────────────────────
    if all_export_figs:
        st.divider()
        hdr("📄 Rapport multi-page PDF")
        pdf_bytes = fig_to_pdf_bytes(all_export_figs)
        ts = st.session_state.get("analysis_date", "report")
        st.download_button(
            "⬇️ Télécharger le rapport PDF complet",
            data=pdf_bytes,
            file_name=f"ct_qc_report_{ts}.pdf",
            mime="application/pdf", key="dl_full_pdf",
            use_container_width=True)


def _render_welcome() -> None:
    """Welcome screen when no analysis has been run."""
    st.markdown("# 🏥 CT Quality Control Platform")
    st.markdown("### Dashboard Clinique Unifié — Phase 7")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📊 Image QC")
        st.markdown("GE Helios + Siemens Waterbath\n\n"
                     "Auto-detect manufacturer via DICOM")
    with c2:
        st.markdown("#### 💊 Dosimétrie")
        st.markdown("D_w, SSDE, DRL\n\n"
                     "AAPM TG-220, Report 204")
    with c3:
        st.markdown("#### 🧠 Prédictif")
        st.markdown("Tendances, alertes matérielles\n\n"
                     "Régression linéaire, intervalles prédictifs")
    st.divider()
    st.info("👈 **Importer les fichiers DICOM dans la sidebar** pour démarrer.")


def _archive_current_session(scanner_id: str, operator_id: str) -> None:
    """Archives current session results to MetricsArchive."""
    archive = get_archive()
    basic = st.session_state.get("basic_result")
    advanced = st.session_state.get("advanced_result")
    if not basic:
        st.warning("Aucun résultat à archiver.")
        return

    now = datetime.now()
    record = QCSessionRecord(
        session_id=str(uuid.uuid4()),
        session_date=now.strftime("%Y-%m-%d"),
        session_timestamp=now.isoformat(),
        scanner_id=scanner_id,
        operator_id=operator_id,
    )

    # Basic metrics
    if basic.noise:
        record.center_water_mean_hu = basic.noise.mean_hu
        record.center_water_std_hu = basic.noise.std_hu
        record.center_water_variance_hu = basic.noise.variance_hu
    record.all_image_qc_passed = basic.all_passed

    # Advanced metrics
    if advanced:
        if advanced.nps:
            record.nps_peak_frequency_lpmm = advanced.nps.nps_peak_frequency_lpmm
            record.nps_peak_value_hu2mm2 = advanced.nps.nps_peak_value
            record.nps_integral_hu2mm2 = advanced.nps.nps_integral
        if advanced.mtf:
            record.mtf_50_lpmm = advanced.mtf.mtf_50_lpmm
            record.mtf_10_lpmm = advanced.mtf.mtf_10_lpmm
        if advanced.hu_linearity:
            record.hu_linearity_max_deviation_hu = advanced.hu_linearity.max_deviation_hu
            record.hu_linearity_r_squared = advanced.hu_linearity.r_squared
        if advanced.ed_calibration:
            record.ed_soft_tissue_slope = advanced.ed_calibration.soft_tissue_slope
            record.ed_bone_slope = advanced.ed_calibration.bone_slope
            record.ed_max_red_deviation = advanced.ed_calibration.max_red_deviation
            record.ed_calibration_passed = advanced.ed_calibration.all_passed
        if advanced.ssde_series:
            record.ctdi_vol_mgy = advanced.ssde_series.ctdi_vol_mgy
            record.ssde_mean_mgy = advanced.ssde_series.ssde_mean_mgy
            record.effective_dose_msv = advanced.ssde_series.effective_dose_msv
            record.all_dosimetry_computed = True
        if advanced.dw_series:
            record.dw_mean_cm = advanced.dw_series.dw_mean_cm

    try:
        archive.append_session(record)
        st.success(f"✅ Session archivée : {record.session_id[:8]}...")
    except Exception as exc:
        st.error(f"❌ Erreur archivage : {exc}")
