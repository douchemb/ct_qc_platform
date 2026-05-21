"""
dashboard/tab_summary.py — Tab 1: Summary KPI (TotalQA-Aligned).
Uniformity/Noise KPIs + TotalQA Contrast/Resolution/Scaling + Live ROI Drawing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import CONFIG
from modules.image_qc.basic_metrics import BasicQAResult
from modules.image_qc.roi_stats import VolumetricQCResult
from dashboard.helpers import hdr, kpi_card, render_fig
from dashboard.roi_drawing import render_roi_drawing


# ══════════════════════════════════════════════════════════════════
# Acquisition Parameters Display
# ══════════════════════════════════════════════════════════════════

def _render_acquisition_params_bar() -> None:
    """Render DICOM acquisition parameters bar for noise diagnostics.

    Displays kVp, mA, Convolution Kernel, and slice thickness
    from session_state['acquisition_params']. These parameters
    are critical for understanding noise fluctuations — a sharper
    kernel (e.g., Br64) will always produce higher noise than a
    smooth kernel (e.g., Br40).
    """
    params = st.session_state.get("acquisition_params")
    if not params:
        return

    kvp = params.get("kvp", 0)
    mas = params.get("mas", 0)
    kernel = params.get("kernel", "N/A")
    thickness = params.get("slice_thickness", 0)

    # Build parameter chips
    parts = []
    if kvp and kvp > 0:
        parts.append(f"**{kvp:.0f} kV**")
    if mas and mas > 0:
        parts.append(f"**{mas:.0f} mA**")
    if kernel and kernel != "N/A":
        parts.append(f"Filtre (Kernel) : **{kernel}**")
    if thickness and thickness > 0:
        parts.append(f"Épaisseur : **{thickness:.1f} mm**")

    if not parts:
        return

    param_text = " | ".join(parts)

    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, rgba(88,166,255,0.10), rgba(163,113,247,0.10));
            border: 1px solid rgba(88,166,255,0.25);
            border-radius: 8px;
            padding: 10px 18px;
            margin-bottom: 12px;
            font-size: 14px;
            color: #c9d1d9;
        ">
            ⚙️ <span style="color:#8b949e;">Paramètres d'acquisition :</span> {param_text}
        </div>""",
        unsafe_allow_html=True,
    )


def render_tab_summary(
    basic_result: BasicQAResult,
    volumetric_result: VolumetricQCResult,
    roi_descriptors: dict,
    pixel_spacing_mm: tuple,
) -> list[tuple[str, "plt.Figure"]]:
    """Renders Summary tab — auto-switches between GE and Siemens modes."""
    import matplotlib.pyplot as plt
    export_figures: list[tuple[str, plt.Figure]] = []

    # ── SIEMENS MODE: 3 TotalQA standard tables ──────────────────────
    if st.session_state.get("manufacturer") in ("SIEMENS", "CANON"):
        return _render_siemens_summary(export_figures)

    # ── 1. Interactive Simulation Viewer ─────────────────────────────────
    hdr("🖼️ Visualisation des ROIs (Simulation)")

    # Read pre-built simulation figures from the orchestrator
    simulation_figs: dict = st.session_state.get("simulation_figs", {})

    if simulation_figs:
        available_simulations = list(simulation_figs.keys())

        # Navigation selector
        if len(available_simulations) > 1:
            st.markdown(
                f"**{len(available_simulations)} coupes analysées** — "
                f"naviguez entre les vues ci-dessous."
            )
            selected_sim = st.selectbox(
                "🔍 Naviguer entre les coupes analysées :",
                available_simulations,
                key="sim_viewer_select",
                help="Chaque vue montre les ROIs dessinées sur la coupe "
                     "physiquement correcte pour ce module.")
        else:
            selected_sim = available_simulations[0]
            st.markdown(f"**1 coupe analysée** — {selected_sim}")

        # Display the selected pre-built figure
        fig_selected = simulation_figs[selected_sim]
        col_fig, col_meta = st.columns([2, 1])
        with col_fig:
            render_fig(
                fig_selected,
                f"sim_{selected_sim.replace(' ', '_')[:30]}",
                f"simulation_{selected_sim.replace(' ', '_')[:30]}_{basic_result.acquisition_date}.png",
                caption=f"{selected_sim}")
            export_figures.append((f"Fig. — {selected_sim}", fig_selected))
        with col_meta:
            st.markdown("**Coupes disponibles :**")
            for i, label in enumerate(available_simulations, 1):
                marker = "▶" if label == selected_sim else "○"
                st.markdown(f"{marker} **{i}.** {label}")

            # Show ROI detail table for the selected view
            _roi_source_map = {
                "💧": "water_roi_descriptors",
                "🧪": "contrast_roi_descriptors",
                "📐": "resolution_roi_descriptors",
            }
            source_key = None
            for emoji, key in _roi_source_map.items():
                if selected_sim.startswith(emoji):
                    source_key = key
                    break

            view_rois = st.session_state.get(source_key, {}) if source_key else {}
            if not view_rois and roi_descriptors:
                view_rois = roi_descriptors

            if view_rois:
                st.markdown("**Détails des ROIs :**")
                roi_rows = []
                for label, roi in view_rois.items():
                    roi_rows.append({
                        "ROI": label,
                        "Centre (px)": f"({roi.row_start + roi.height_px // 2}, "
                                       f"{roi.col_start + roi.width_px // 2})",
                        "Taille": f"{roi.height_px}×{roi.width_px} px",
                        "Surface": f"{roi.area_px} px²",
                    })
                st.dataframe(pd.DataFrame(roi_rows),
                             use_container_width=True, hide_index=True)

    else:
        # No pre-built figures — fallback to legacy on-the-fly rendering
        water_rois = st.session_state.get("water_roi_descriptors")
        water_hu = st.session_state.get("water_hu_array")
        if water_rois and water_hu is not None:
            with st.spinner("Génération de la carte ROI..."):
                fig_roi = render_roi_drawing(
                    water_hu, water_rois, pixel_spacing_mm,
                    title=f"Placement ROI — {basic_result.phantom_id.replace('_', ' ').title()}",
                    slice_type="water")
            render_fig(fig_roi, "roi_drawing_fallback",
                       f"roi_placement_fallback_{basic_result.acquisition_date}.png",
                       caption="Vue Uniformité (fallback)")
            export_figures.append(("Fig. 1 — ROIs (Fallback)", fig_roi))
        else:
            st.warning(
                "⚠️ Aucune coupe n'a pu être analysée. "
                "Vérifiez les fichiers DICOM et les intervalles de routage.")

    st.divider()

    # ── 2. Global verdict banner ───────────────────────────────────────
    if basic_result.all_passed:
        st.success(f"✅ **VERDICT GLOBAL : CONFORME** — "
                   f"Toutes les métriques TotalQA sont dans les tolérances. "
                   f"Scanner : {basic_result.scanner_id} | "
                   f"Fantôme : {basic_result.phantom_id}")
    else:
        # Count how many metrics are out of tolerance
        _metric_checks = [
            (basic_result.noise, "passed"),
            (basic_result.uniformity, "passed"),
            (basic_result.ct_number_accuracy, "all_passed"),
            (basic_result.contrast, "passed"),
            (basic_result.totalqa_contrast, "passed"),
            (basic_result.totalqa_resolution, "passed"),
            (basic_result.totalqa_scaling, "passed"),
        ]
        n_fail = sum(
            1 for m, attr in _metric_checks
            if m is not None and not getattr(m, attr, True)
        )
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(88,166,255,0.10), rgba(210,153,34,0.12));
                border: 1px solid rgba(210,153,34,0.35);
                border-radius: 10px;
                padding: 14px 20px;
                margin-bottom: 12px;
                font-size: 15px;
                color: #c9d1d9;
            ">
                📊 <b>Statut de l'Analyse QA :</b> <b style="color:#d29922;">{n_fail}</b>
                métrique(s) hors tolérances standards.<br/>
                <span style="font-size:13px;color:#8b949e;">
                    Scanner : {basic_result.scanner_id} | Fantôme : {basic_result.phantom_id}<br/>
                    <em>Note : Vérifiez si le protocole d'acquisition justifie ces écarts
                    (ex : Low-Dose, filtre sharp).</em>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 3. Acquisition Parameters Bar ─────────────────────────────────
    _render_acquisition_params_bar()

    # ── 4. TotalQA KPI cards ──────────────────────────────────────────
    hdr("Métriques QA — TotalQA Benchmarking")
    c1, c2, c3 = st.columns(3)

    with c1:
        if basic_result.noise:
            n = basic_result.noise
            st.markdown(kpi_card("🔊", "Bruit Image (SD)",
                                 f"{n.std_hu:.3f} HU",
                                 f"Tolérance ≤ {n.tolerance_hu:.1f} HU | TG-66 §5.1",
                                 n.passed), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("🔊", "Bruit", "N/A", "Non calculé", None),
                        unsafe_allow_html=True)
    with c2:
        if basic_result.uniformity:
            u = basic_result.uniformity
            st.markdown(kpi_card("⚖️", "Uniformité (NUI)",
                                 f"{u.non_uniformity_index:.3f} HU",
                                 f"Tolérance ≤ {u.tolerance_hu:.1f} HU | "
                                 f"Pire ROI: {u.worst_roi}",
                                 u.passed), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("⚖️", "Uniformité", "N/A", "Non calculé", None),
                        unsafe_allow_html=True)
    with c3:
        if basic_result.ct_number_accuracy:
            a = basic_result.ct_number_accuracy
            hu_passed = abs(a.max_delta_hu) <= a.tolerance_hu
            st.markdown(kpi_card("🎯", "Précision HU",
                                 f"ΔHU_max = {a.max_delta_hu:.2f}",
                                 f"Tolérance ≤ {a.tolerance_hu:.1f} HU | TG-66 §5.2",
                                 hu_passed), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("🎯", "Précision HU", "N/A", "Non calculé", None),
                        unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)

    with c4:
        if basic_result.totalqa_contrast:
            tc = basic_result.totalqa_contrast
            st.markdown(kpi_card("🧪", "Contraste (TotalQA)",
                                 f"Top={tc.contrast_top:.2f} | Bot={tc.contrast_bottom:.2f}",
                                 f"A={tc.mean_A:.1f} B={tc.mean_B:.1f} "
                                 f"C={tc.mean_C:.1f} D={tc.mean_D:.1f} HU",
                                 tc.passed), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("🧪", "Contraste", "N/A",
                                 "Slice 60 non fourni", None),
                        unsafe_allow_html=True)
    with c5:
        if basic_result.totalqa_resolution:
            tr = basic_result.totalqa_resolution
            avg_sd = np.mean(tr.bar_sd_values) if tr.bar_sd_values else 0.0
            st.markdown(kpi_card("📐", "Résolution (Bar SD)",
                                 f"Moy SD = {avg_sd:.2f} HU",
                                 f"{len(tr.bar_sd_values)} bar patterns analysés",
                                 tr.passed), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("📐", "Résolution", "N/A",
                                 "Slice 70 non fourni", None),
                        unsafe_allow_html=True)
    with c6:
        if basic_result.totalqa_scaling:
            sc = basic_result.totalqa_scaling
            st.markdown(kpi_card("📏", "Scaling (Diamètre)",
                                 f"H={sc.h_diameter_mm:.2f} V={sc.v_diameter_mm:.2f} mm",
                                 f"Nominal={sc.nominal_mm:.1f} mm | "
                                 f"ΔH={sc.h_error_mm:+.2f} ΔV={sc.v_error_mm:+.2f} mm",
                                 sc.passed), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("📏", "Scaling", "N/A",
                                 "Slice 36 non fourni", None),
                        unsafe_allow_html=True)

    st.divider()

    # ── 4. Uniformity detail table ─────────────────────────────────────
    if basic_result.uniformity:
        hdr("Détail Uniformité — Centre vs Périphérie")
        u = basic_result.uniformity
        rows = [{"ROI": "centre", "HU moyen": round(u.centre_mean_hu, 3),
                 "Δ HU vs centre": 0.0, "Conforme": "✓"}]
        for label, mean in u.peripheral_means.items():
            dev = u.deviations[label]
            rows.append({"ROI": label, "HU moyen": round(mean, 3),
                         "Δ HU vs centre": round(dev, 3),
                         "Conforme": "✓" if dev <= u.tolerance_hu else "✗"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 5. TotalQA Contrast detail ────────────────────────────────────
    if basic_result.totalqa_contrast:
        hdr("Détail Contraste TotalQA — Plastic Block (Slice 60)")
        tc = basic_result.totalqa_contrast

        # Tableau des valeurs brutes — inchangées, pour transparence physique
        contrast_rows = [
            {"Zone": "A — Top Plastic",    "Mean HU (Raw)": round(tc.mean_A, 3)},
            {"Zone": "B — Top Water",      "Mean HU (Raw)": round(tc.mean_B, 3)},
            {"Zone": "C — Bottom Plastic", "Mean HU (Raw)": round(tc.mean_C, 3)},
            {"Zone": "D — Bottom Water",   "Mean HU (Raw)": round(tc.mean_D, 3)},
        ]
        st.dataframe(pd.DataFrame(contrast_rows),
                     use_container_width=True, hide_index=True)

        # Affichage des contrastes calibrés (tc.contrast_top/bottom = brut × 0.409)
        st.markdown(
            f"**Contrast Top (Calibrated)** = (Mean(A) − Mean(B)) × *f* = "
            f"**{tc.contrast_top:.2f} HU**  \n"
            f"**Contrast Bottom (Calibrated)** = (Mean(C) − Mean(D)) × *f* = "
            f"**{tc.contrast_bottom:.2f} HU**  \n"
            f"<small>*f* = Contrast Scale Factor GE = 0.409 "
            f"— valeurs brutes : Top={tc.mean_A - tc.mean_B:.2f} HU, "
            f"Bottom={tc.mean_C - tc.mean_D:.2f} HU</small>",
            unsafe_allow_html=True)



    # ── 6. TotalQA Resolution detail ──────────────────────────────────
    if basic_result.totalqa_resolution:
        hdr("Détail Résolution TotalQA — Bar Patterns (Slice 70)")
        tr = basic_result.totalqa_resolution

        # Listes de référence TotalQA — ordre strict coarse→fine
        tailles_mm = [1.6, 1.3, 1.0, 0.8, 0.6]
        frequences = [0.312, 0.385, 0.500, 0.625, 0.833]

        # Correction 2 : tri décroissant — SD max sur grosse barre, SD min sur petite barre
        # Associé strictement à tailles_mm[i] et frequences[i]
        sd_sorted = sorted(tr.bar_sd_values, reverse=True)

        n = len(sd_sorted)
        tailles_mm = tailles_mm[:n]
        frequences = frequences[:n]

        res_rows = []
        for i, sd in enumerate(sd_sorted):
            res_rows.append({
                # Colonnes exactes : ['Bar Pattern', 'Taille (mm)', 'Fréquence (LP/mm)', 'SD HU']
                "Bar Pattern":       f"Bar {i + 1}",
                "Taille (mm)":       tailles_mm[i],
                "Fréquence (LP/mm)": frequences[i],
                "SD HU":             round(sd, 3),  # ← np.std trié décroissant
            })

        st.dataframe(
            pd.DataFrame(res_rows),
            use_container_width=True,
            hide_index=True,
        )





    # ── 7. TotalQA Scaling detail ─────────────────────────────────────
    if basic_result.totalqa_scaling:
        hdr("Détail Scaling TotalQA — Diamètre Fantôme (Slice 36)")
        sc = basic_result.totalqa_scaling
        scaling_rows = [
            {"Direction": "Horizontal", "Mesuré (mm)": round(sc.h_diameter_mm, 3),
             "Nominal (mm)": sc.nominal_mm, "Erreur (mm)": round(sc.h_error_mm, 3),
             "Erreur (%)": round(sc.h_error_pct, 3),
             "Conforme": "✓" if abs(sc.h_error_mm) <= sc.tolerance_mm else "✗"},
            {"Direction": "Vertical", "Mesuré (mm)": round(sc.v_diameter_mm, 3),
             "Nominal (mm)": sc.nominal_mm, "Erreur (mm)": round(sc.v_error_mm, 3),
             "Erreur (%)": round(sc.v_error_pct, 3),
             "Conforme": "✓" if abs(sc.v_error_mm) <= sc.tolerance_mm else "✗"},
        ]
        st.dataframe(pd.DataFrame(scaling_rows),
                     use_container_width=True, hide_index=True)

    # ── 8. CT number accuracy detail ──────────────────────────────────
    if basic_result.ct_number_accuracy and basic_result.ct_number_accuracy.measurements:
        hdr("Précision CT Number — Par Matériau")
        acc_rows = []
        for m in basic_result.ct_number_accuracy.measurements:
            acc_rows.append({
                "Matériau": m["material"], "HU nominal": m["nominal_hu"],
                "HU mesuré": round(m["measured_hu"], 2),
                "ΔHU": round(m["delta_hu"], 3),
                "|ΔHU|": round(m["abs_delta"], 3),
                "Conforme": "✓" if m["passed"] else "✗",
            })
        st.dataframe(pd.DataFrame(acc_rows),
                     use_container_width=True, hide_index=True)

    return export_figures


# ══════════════════════════════════════════════════════════════════
# Siemens Waterbath Summary Mode
# ══════════════════════════════════════════════════════════════════

def _render_siemens_summary(
    export_figures: list,
) -> list:
    """Renders Siemens Waterbath summary: simulation viewer + 3 TotalQA tables."""
    import matplotlib.pyplot as plt
    from dashboard.helpers import render_fig

    siemens_result = st.session_state.get("siemens_result")
    if siemens_result is None or not siemens_result.slices:
        st.warning("⚠️ No Siemens Waterbath results available.")
        return export_figures

    slices = siemens_result.slices

    # ── Simulation Viewer ─────────────────────────────────────────
    simulation_figs = st.session_state.get("simulation_figs", {})
    manufacturer = st.session_state.get("manufacturer", "SIEMENS")
    phantom_label = "Canon Aquilion Water Phantom" if manufacturer == "CANON" else "Siemens Waterbath"
    if simulation_figs:
        hdr(f"🖼️ Visualisation des ROIs — {phantom_label}")
        for label, fig in simulation_figs.items():
            render_fig(fig, f"sim_{label.replace(' ', '_')[:30]}",
                       f"siemens_sim.png", caption=label)
            export_figures.append((f"Fig. — {label}", fig))

    st.divider()

    # ── Acquisition Parameters Bar ─────────────────────────────────
    _render_acquisition_params_bar()

    # ── KPI Cards — 4 Visual Metrics ──────────────────────────────
    kpi = st.session_state.get("siemens_kpi_metrics", {})
    if kpi:
        hdr(f"Métriques QA — {phantom_label}")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            noise_sd = kpi.get("noise_sd", 0.0)
            noise_ok = kpi.get("noise_passed")
            noise_limit = kpi.get("noise_limit", 5.0)
            noise_thick = kpi.get("noise_thickness", 5.0)
            limit_note = (
                f"Ajusté pour {noise_thick:.1f} mm"
                if noise_thick != 5.0
                else "TG-66 §5.1"
            )
            st.markdown(kpi_card(
                "🔊", "Bruit Image (SD)",
                f"{noise_sd:.3f} HU",
                f"Tolérance ≤ {noise_limit:.1f} HU | {limit_note}",
                noise_ok,
            ), unsafe_allow_html=True)

        with c2:
            nui = kpi.get("uniformity_nui", 0.0)
            uni_ok = kpi.get("uniformity_passed")
            worst = kpi.get("uniformity_worst", "N/A")
            st.markdown(kpi_card(
                "⚖️", "Uniformité (NUI)",
                f"{nui:.3f} HU",
                f"Tolérance ≤ 5.0 HU | Pire ROI: {worst}",
                uni_ok,
            ), unsafe_allow_html=True)

        with c3:
            hu_delta = kpi.get("hu_precision_delta", 0.0)
            hu_ok = kpi.get("hu_precision_passed")
            st.markdown(kpi_card(
                "🎯", "Précision HU",
                f"ΔHU = {hu_delta:.2f}",
                f"Tolérance ≤ 4.0 HU | TG-66 §5.2",
                hu_ok,
            ), unsafe_allow_html=True)

        with c4:
            h_mm = kpi.get("scaling_h_mm", 0.0)
            v_mm = kpi.get("scaling_v_mm", 0.0)
            default_nom = 330.0 if manufacturer == "CANON" else 200.0
            nom = kpi.get("scaling_nominal_mm", default_nom)
            sc_ok = kpi.get("scaling_passed")
            st.markdown(kpi_card(
                "📏", "Scaling (Diamètre)",
                f"H={h_mm:.2f} V={v_mm:.2f} mm",
                f"Nominal={nom:.1f} mm | "
                f"ΔH={h_mm - nom:+.2f} ΔV={v_mm - nom:+.2f} mm",
                sc_ok,
            ), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

    st.divider()

    # ── Verdict Banner ────────────────────────────────────────────
    all_ok = all(
        all(d <= 4.0 for d in s.edge_diffs.values())
        for s in slices
    )
    if all_ok:
        st.success(f"✅ **VERDICT: CONFORME** — "
                   f"All edge ROIs within ±4 HU of center "
                   f"({len(slices)} slices)")
    else:
        n_fail_edges = sum(
            1 for s in slices
            for d in s.edge_diffs.values() if d > 4.0
        )
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(88,166,255,0.10), rgba(210,153,34,0.12));
                border: 1px solid rgba(210,153,34,0.35);
                border-radius: 10px;
                padding: 14px 20px;
                margin-bottom: 12px;
                font-size: 15px;
                color: #c9d1d9;
            ">
                📊 <b>Statut de l'Analyse QA :</b> <b style="color:#d29922;">{n_fail_edges}</b>
                ROI(s) hors tolérance ±4 HU sur {len(slices)} coupe(s).<br/>
                <span style="font-size:13px;color:#8b949e;">
                    <em>Note : Vérifiez si le protocole d'acquisition justifie ces écarts
                    (ex : Low-Dose, filtre sharp).</em>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Table 1: Mean CT Value — Center Region ────────────────────
    hdr("📊 Table 1 — Mean CT Value (Center Region)")
    t1_rows = []
    for s in slices:
        t1_rows.append({
            "Image #": s.image_number,
            "Filter": s.kernel,
            "KVP": s.kvp,
            "mAs": s.mas,
            "Slice Thickness (mm)": s.slice_thickness,
            "Center Mean (HU)": round(s.center_mean, 2),
        })
    st.dataframe(pd.DataFrame(t1_rows),
                 use_container_width=True, hide_index=True)

    # ── Table 2: Absolute Differences ─────────────────────────────
    hdr("📊 Table 2 — Absolute Differences (|Edge − Center|)")
    t2_rows = []
    for s in slices:
        row = {
            "Image #": s.image_number,
            "Filter": s.kernel,
            "mAs": s.mas,
            "Slice Thickness (mm)": s.slice_thickness,
        }
        for edge in ["upper", "right", "lower", "left"]:
            val = s.edge_diffs.get(edge, 0.0)
            status = "✓" if val <= 4.0 else "✗"
            row[f"{edge.title()} (HU)"] = f"{val:.2f} {status}"
        t2_rows.append(row)
    st.dataframe(pd.DataFrame(t2_rows),
                 use_container_width=True, hide_index=True)

    # ── Table 3: Noise ────────────────────────────────────────────
    hdr("📊 Table 3 — Noise (Center ROI Standard Deviation)")
    t3_rows = []
    for s in slices:
        t3_rows.append({
            "Image #": s.image_number,
            "Filter": s.kernel,
            "mAs": s.mas,
            "Slice Thickness (mm)": s.slice_thickness,
            "Noise (HU)": round(s.center_sd, 3),
        })
    st.dataframe(pd.DataFrame(t3_rows),
                 use_container_width=True, hide_index=True)

    return export_figures
