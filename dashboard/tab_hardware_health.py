"""
dashboard/tab_hardware_health.py — Tab 4: État de l'Appareil.

Hardware Health & Condition-Based Monitoring.
Translates QA physics metrics into predictive hardware health scores
for the Biomedical Engineer.

Subsystem mapping:
  X-Ray Tube       → Noise + MTF (if available)
  Detectors (DAS)  → Uniformity + HU Precision
  Mechanics/Table  → Scaling H + Scaling V + Slice Thickness (if available)
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from dashboard.helpers import hdr, kpi_card


# ══════════════════════════════════════════════════════════════════
# Health Scoring Engine — Safe Zone (Plateau) Model
# ══════════════════════════════════════════════════════════════════

def _health_score_plateau(
    value: float, safe_limit: float, critical_limit: float,
) -> float:
    """Compute health % using a plateau (safe zone) approach.

    Physics rationale:
      - CT image noise, uniformity, HU drift are NEVER zero in practice.
      - Any value within normal operating limits is 100% healthy.
      - Health only degrades when the value crosses the safe_limit
        towards the critical_limit.

    Score logic:
      value ≤ safe_limit     → 100% (within normal operating envelope)
      value ≥ critical_limit → 0%   (hardware failure imminent)
      in between             → linear interpolation

    Args:
        value: Absolute magnitude of the measured metric.
        safe_limit: Upper bound of normal operation (100% health).
        critical_limit: Threshold for hardware failure (0% health).
    """
    value = abs(float(value))
    if value <= safe_limit:
        return 100.0
    if value >= critical_limit:
        return 0.0
    # Linear degradation between safe and critical
    return 100.0 * (1.0 - (value - safe_limit) / (critical_limit - safe_limit))


# Legacy wrapper for GE slice thickness (still uses ideal ± tolerance)
def _health_score(measured: float, ideal: float, tolerance: float) -> float:
    """Compute a 0–100% health score based on distance to tolerance.

    Score = 100 × max(0, 1 − |Measured − Ideal| / Tolerance)
    Used only for metrics with a true ideal value (e.g., slice thickness = 5.0 mm).
    """
    if tolerance <= 0:
        return 100.0
    return 100.0 * max(0.0, 1.0 - abs(measured - ideal) / tolerance)


def _avg_scores(scores: list[float]) -> Optional[float]:
    """Average of non-empty score list, or None if empty."""
    return sum(scores) / len(scores) if scores else None


def _collect_health_metrics() -> dict:
    """Collect all metric scores from session_state (GE or Siemens).

    Uses the plateau (safe-zone) model:
      - Values within normal operating limits → 100%
      - Only penalize when crossing the safe→critical boundary

    mA-Aware Tube Scoring (Physics):
      Quantum noise scales as sqrt(Reference_mA / Current_mA).
      At low mA (e.g., 20 mA), expected noise is inherently higher.
      The safe/critical limits are dynamically adjusted so that
      a noise of 10.3 HU at 20 mA is NOT scored as hardware failure.
      Reference mA = 200 mA (standard clinical protocol).

    GE MTF Linkage:
      When true MTF is unavailable, the GE bar pattern SD from
      totalqa_resolution is used as a resolution proxy.
      Higher SD → better bar contrast → better spatial resolution.

    Returns a dict with individual metric scores and subsystem averages.
    """
    manufacturer = st.session_state.get("manufacturer", "GE")
    basic = st.session_state.get("basic_result")
    kpi = st.session_state.get("siemens_kpi_metrics", {})
    acq_params = st.session_state.get("acquisition_params", {})

    metrics = {}  # metric_label -> score (0-100)

    # ── mA-aware noise correction ──────────────────────────────────
    # Physics: noise ∝ 1/sqrt(mA). At low mA, expected noise is higher.
    # We scale the safe/critical limits so the scoring is protocol-aware.
    REFERENCE_MA = 200.0
    current_ma = acq_params.get("mas", REFERENCE_MA)
    if not current_ma or current_ma <= 0:
        current_ma = REFERENCE_MA
    correction_factor = (REFERENCE_MA / current_ma) ** 0.5

    noise_safe = 5.0 * correction_factor
    noise_critical = 7.0 * correction_factor

    # ── Extract raw values ────────────────────────────────────────
    if manufacturer == "SIEMENS" and kpi:
        # Noise (SD): dynamically adjusted for mA
        noise_sd = kpi.get("noise_sd")
        if noise_sd is not None:
            metrics["noise"] = _health_score_plateau(noise_sd, noise_safe, noise_critical)

        # Uniformity (NUI): healthy < 3.0, critical > 5.5
        nui = kpi.get("uniformity_nui")
        if nui is not None:
            metrics["uniformity"] = _health_score_plateau(nui, 3.0, 5.5)

        # HU Precision: healthy < 3.0, critical > 5.0
        hu_delta = kpi.get("hu_precision_delta")
        if hu_delta is not None:
            metrics["hu_precision"] = _health_score_plateau(hu_delta, 3.0, 5.0)

        # Scaling Error: max deviation from nominal (200.0 mm)
        h_mm = kpi.get("scaling_h_mm")
        v_mm = kpi.get("scaling_v_mm")
        if h_mm is not None and h_mm > 0 and v_mm is not None and v_mm > 0:
            scaling_error = max(abs(h_mm - 200.0), abs(v_mm - 200.0))
            metrics["scaling_h"] = _health_score_plateau(scaling_error, 1.0, 2.5)
            metrics["scaling_v"] = metrics["scaling_h"]  # same score, single error

    elif basic is not None:
        # GE pipeline — mA-aware noise scoring
        if basic.noise is not None:
            metrics["noise"] = _health_score_plateau(
                basic.noise.std_hu, noise_safe, noise_critical)

        if basic.uniformity is not None:
            metrics["uniformity"] = _health_score_plateau(
                basic.uniformity.non_uniformity_index, 3.0, 5.5)

        if basic.ct_number_accuracy is not None:
            metrics["hu_precision"] = _health_score_plateau(
                abs(basic.ct_number_accuracy.max_delta_hu), 3.0, 5.0)

        if basic.totalqa_scaling is not None:
            sc = basic.totalqa_scaling
            scaling_error = max(
                abs(sc.h_diameter_mm - 215.0),  # GE Helios nominal
                abs(sc.v_diameter_mm - 215.0),
            )
            metrics["scaling_h"] = _health_score_plateau(scaling_error, 1.0, 2.5)
            metrics["scaling_v"] = metrics["scaling_h"]

        if basic.slice_thickness is not None:
            st_result = basic.slice_thickness
            metrics["slice_thickness"] = _health_score(
                st_result.measured_fwhm_mm,
                st_result.nominal_thickness_mm,
                st_result.tolerance_mm,
            )
        elif acq_params.get("slice_thickness") and acq_params["slice_thickness"] > 0:
            # Fallback: no FWHM measurement available (no Z-ramp slice)
            # Use DICOM nominal SliceThickness and score at 100%
            metrics["slice_thickness"] = 100.0

        # ── GE MTF Linkage: bar pattern SD as resolution proxy ────
        if basic.totalqa_resolution is not None:
            import numpy as np
            tr = basic.totalqa_resolution
            if tr.bar_sd_values:
                avg_sd = float(np.mean(tr.bar_sd_values))
                # Higher bar SD = better contrast in bar patterns = better MTF
                # Scale: SD > 40 HU → excellent (100%), SD < 5 HU → poor (0%)
                mtf_proxy = min(100.0, max(0.0, (avg_sd - 5.0) / 35.0 * 100.0))
                metrics["mtf"] = mtf_proxy

    # Siemens: also apply nominal fallback if no measured thickness
    if "slice_thickness" not in metrics:
        nominal_st = acq_params.get("slice_thickness", 0)
        if nominal_st and nominal_st > 0:
            metrics["slice_thickness"] = 100.0

    # MTF — from advanced result (both pipelines, overrides proxy if available)
    advanced = st.session_state.get("advanced_result")
    if advanced and hasattr(advanced, "mtf") and advanced.mtf is not None:
        # MTF50 > 0 is good; score based on a reasonable baseline
        # Use a simple pass: if mtf_50 > 0.5 lp/mm → healthy
        metrics["mtf"] = min(100.0, advanced.mtf.mtf_50_lpmm / 0.5 * 100.0)

    # ── Build nominal display labels ──────────────────────────────
    # When a metric uses DICOM nominal instead of a measurement,
    # store a human-readable label for the UI detail breakdown.
    nominal_labels = {}
    nominal_st = acq_params.get("slice_thickness", 0)
    if nominal_st and nominal_st > 0:
        if basic is None or basic.slice_thickness is None:
            nominal_labels["slice_thickness"] = f"{nominal_st:.1f} mm (Nominal DICOM)"

    # GE MTF proxy label
    if basic is not None and basic.totalqa_resolution is not None:
        import numpy as np
        tr = basic.totalqa_resolution
        if tr.bar_sd_values:
            avg_sd = float(np.mean(tr.bar_sd_values))
            nominal_labels["mtf"] = f"{avg_sd:.1f} HU (Bar SD)"

    # ── Subsystem grouping ────────────────────────────────────────
    tube_scores = [v for k, v in metrics.items() if k in ("noise", "mtf")]
    detector_scores = [v for k, v in metrics.items()
                       if k in ("uniformity", "hu_precision")]
    mechanics_scores = [v for k, v in metrics.items()
                        if k in ("scaling_h", "scaling_v", "slice_thickness")]

    subsystems = {
        "tube": _avg_scores(tube_scores),
        "detectors": _avg_scores(detector_scores),
        "mechanics": _avg_scores(mechanics_scores),
    }

    # Global = average of available subsystems
    available = [v for v in subsystems.values() if v is not None]
    global_score = sum(available) / len(available) if available else None

    return {
        "metrics": metrics,
        "subsystems": subsystems,
        "global_score": global_score,
        "nominal_labels": nominal_labels,
    }


def _urgency_color(score: Optional[float]) -> str:
    """Return color hex based on score threshold."""
    if score is None:
        return "#8b949e"
    if score >= 90:
        return "#3fb950"
    if score >= 70:
        return "#d29922"
    return "#da3633"


def _urgency_label(score: Optional[float]) -> str:
    """Return label for score range."""
    if score is None:
        return "N/A"
    if score >= 90:
        return "Optimal"
    if score >= 70:
        return "Attention"
    return "Critique"


# ══════════════════════════════════════════════════════════════════
# Tab Renderer
# ══════════════════════════════════════════════════════════════════

def render_tab_hardware_health() -> list:
    """Renders the Hardware Health / État de l'Appareil tab."""
    export_figures = []

    health = _collect_health_metrics()
    metrics = health["metrics"]
    subs = health["subsystems"]
    global_score = health["global_score"]

    if not metrics:
        st.info(
            "ℹ️ Aucune métrique disponible pour l'analyse de santé. "
            "Lancez l'analyse QA pour alimenter ce module."
        )
        return export_figures

    # ── 1. Global Health Gauge ────────────────────────────────────
    hdr("🏥 Santé Globale du Système")

    if global_score is not None:
        color = _urgency_color(global_score)
        label = _urgency_label(global_score)

        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=global_score,
                number={"suffix": "%", "font": {"size": 48, "color": "white"}},
                title={"text": "Santé Globale",
                       "font": {"size": 18, "color": "#8b949e"}},
                delta={"reference": 90, "increasing": {"color": "#3fb950"},
                       "decreasing": {"color": "#da3633"},
                       "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickcolor": "#8b949e",
                             "dtick": 10},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#21262d",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 70], "color": "rgba(218,54,51,0.15)"},
                        {"range": [70, 90], "color": "rgba(210,153,34,0.15)"},
                        {"range": [90, 100], "color": "rgba(63,185,80,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f0f6fc", "width": 3},
                        "thickness": 0.8,
                        "value": global_score,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f0f6fc"},
                height=300,
                margin={"t": 60, "b": 20, "l": 40, "r": 40},
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback if plotly not installed
            st.metric("Santé Globale", f"{global_score:.1f}%",
                       delta=f"{global_score - 90:.1f}% vs seuil 90%")

        if global_score >= 90:
            st.success(f"✅ **{label}** — Score global : {global_score:.1f}%")
        elif global_score >= 70:
            st.warning(f"⚠️ **{label}** — Score global : {global_score:.1f}%")
        else:
            st.error(f"❌ **{label}** — Score global : {global_score:.1f}%")
    else:
        st.info("ℹ️ Score global indisponible — métriques insuffisantes.")

    st.divider()

    # ── 2. Subsystem Diagnostics ──────────────────────────────────
    hdr("🔧 Diagnostic par Sous-Système")

    c1, c2, c3 = st.columns(3)

    subsystem_info = [
        (c1, "tube", "🔌 Tube à Rayons X",
         "Noise + MTF (résolution spatiale)",
         ["noise", "mtf"]),
        (c2, "detectors", "📡 Détecteurs DAS",
         "Uniformité + Précision HU",
         ["uniformity", "hu_precision"]),
        (c3, "mechanics", "⚙️ Mécanique & Table",
         "Scaling H/V + Épaisseur de coupe",
         ["scaling_h", "scaling_v", "slice_thickness"]),
    ]

    for col, key, title, subtitle, metric_keys in subsystem_info:
        with col:
            score = subs.get(key)
            color = _urgency_color(score)
            label = _urgency_label(score)

            st.markdown(
                f"<h4 style='margin-bottom:4px'>{title}</h4>"
                f"<p style='color:#8b949e;font-size:12px;margin-top:0'>"
                f"{subtitle}</p>",
                unsafe_allow_html=True,
            )

            if score is not None:
                # Progress bar (clamped 0-100)
                st.progress(min(100, max(0, int(score))) / 100)
                st.markdown(
                    f"<p style='text-align:center;font-size:24px;"
                    f"font-weight:bold;color:{color};margin:0'>"
                    f"{score:.1f}%</p>"
                    f"<p style='text-align:center;color:#8b949e;"
                    f"font-size:13px;margin-top:2px'>{label}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.progress(0)
                st.markdown(
                    "<p style='text-align:center;color:#8b949e;"
                    "font-size:14px'>N/A — Données manquantes</p>",
                    unsafe_allow_html=True,
                )

            # Detail breakdown
            nominal_labels = health.get("nominal_labels", {})
            with st.expander("Détail des métriques"):
                for mk in metric_keys:
                    val = metrics.get(mk)
                    nom_label = nominal_labels.get(mk)
                    if val is not None:
                        mk_color = _urgency_color(val)
                        suffix = f" — {nom_label}" if nom_label else ""
                        st.markdown(
                            f"- **{mk.replace('_', ' ').title()}** : "
                            f"<span style='color:{mk_color}'>"
                            f"**{val:.1f}%**</span>"
                            f"<span style='color:#8b949e;font-size:12px'>"
                            f"{suffix}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"- **{mk.replace('_', ' ').title()}** : "
                            f"<span style='color:#8b949e'>N/A</span>",
                            unsafe_allow_html=True,
                        )

    st.divider()

    # ── 3. Actionable Recommendations ─────────────────────────────
    hdr("📋 Recommandations de Maintenance")

    has_warning = False
    tube_score = subs.get("tube")
    det_score = subs.get("detectors")
    mech_score = subs.get("mechanics")

    if det_score is not None and det_score < 85:
        has_warning = True
        st.warning(
            "🔧 **Détecteurs DAS** — Score : "
            f"{det_score:.1f}%\n\n"
            "**Recommandation :** Prévoir une calibration à l'air "
            "(Air Calibration) ou vérifier les logs de bruit "
            "électronique du système DAS."
        )

    if tube_score is not None and tube_score < 85:
        has_warning = True
        st.warning(
            "🔌 **Tube à Rayons X** — Score : "
            f"{tube_score:.1f}%\n\n"
            "**Recommandation :** Le bruit augmente et/ou la résolution "
            "spatiale baisse. Vérifier les logs de refroidissement "
            "(Cooling logs) et l'usure de l'anode."
        )

    if mech_score is not None and mech_score < 85:
        has_warning = True
        st.warning(
            "⚙️ **Mécanique & Table** — Score : "
            f"{mech_score:.1f}%\n\n"
            "**Recommandation :** Erreur géométrique détectée. "
            "Vérifier l'alignement des lasers, la planéité de "
            "la table, ou la courroie d'entraînement."
        )

    if not has_warning:
        st.success(
            "✅ **Système optimal.** Tous les sous-systèmes sont "
            "au-dessus du seuil de 85%. Aucune action de maintenance "
            "préventive n'est requise."
        )

    # ── 4. Raw Scores Table ───────────────────────────────────────
    if metrics:
        with st.expander("📊 Tableau des scores détaillés"):
            import pandas as pd
            rows = []
            for k, v in metrics.items():
                rows.append({
                    "Métrique": k.replace("_", " ").title(),
                    "Score (%)": round(v, 1),
                    "État": _urgency_label(v),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    return export_figures
