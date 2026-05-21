"""
dashboard/tab_predictive.py — Tab 5: Predictif (AI Prognostic)

Multi-Vendor 4-Component Predictive Maintenance Dashboard.
Dynamically selects Siemens, GE, or Canon RandomForest models based on the
detected manufacturer and renders:

  1. Input Metrics Strip  — What the AI is analyzing today
  2. 4 RUL Forecast Cards — Color-coded RAG predictions
  3. Survival Probability — Plotly multi-line chart (90-day horizon)
  4. Actionable Advisory  — Smart recommendation for weakest component
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.helpers import hdr

# Inference engine import
_PM_DIR = Path(__file__).resolve().parent.parent / "predictive_maintenance"
if str(_PM_DIR) not in sys.path:
    sys.path.insert(0, str(_PM_DIR))

from inference import predict_rul  # noqa: E402


# ══════════════════════════════════════════════════════════════════
# Component Metadata (per manufacturer)
# ══════════════════════════════════════════════════════════════════

SIEMENS_COMPONENTS = {
    "tube": {
        "label": "Tube a Rayons X",
        "icon": "🔌",
        "metric_label": "Noise (SD)",
        "failure_desc": "Filament / Anode usee",
        "action": (
            "Verifier les logs de refroidissement et l'usure de l'anode. "
            "Commander un tube de remplacement (Siemens P/N Straton MX)."
        ),
    },
    "gantry": {
        "label": "Gantry / Brushblock",
        "icon": "⚡",
        "metric_label": "Uniformite (NUI)",
        "failure_desc": "Usure courroie / charbons",
        "action": (
            "Inspecter les charbons du collecteur rotatif (Brushblock) "
            "et la tension de la courroie. Commander kit P/N 8377990."
        ),
    },
    "table": {
        "label": "Table Patient",
        "icon": "⚙️",
        "metric_label": "Scaling V (mm)",
        "failure_desc": "Friction / Graissage requis",
        "action": (
            "Appliquer la graisse Isoflex Topas NCA 52 sur les rails "
            "et recalibrer le positionnement geometrique."
        ),
    },
    "generator": {
        "label": "Generateur HT",
        "icon": "🔋",
        "metric_label": "Precision HU",
        "failure_desc": "Derive kVp / Calibration requise",
        "action": (
            "Executer la calibration HV automatique (Service > Calibration > kVp). "
            "Verifier les condensateurs du generateur haute tension."
        ),
    },
}

GE_COMPONENTS = {
    "tube": {
        "label": "Tube a Rayons X",
        "icon": "🔌",
        "metric_label": "MTF50 (lp/cm)",
        "failure_desc": "Blooming spot focal / Anode usee",
        "action": (
            "Resolution spatiale en baisse. Verifier l'usure de l'anode "
            "et les logs de refroidissement GE Performix. "
            "Commander un tube de remplacement si MTF50 < 5.0 lp/cm."
        ),
    },
    "detectors": {
        "label": "Detecteurs / DAS",
        "icon": "📡",
        "metric_label": "Uniformite (HU)",
        "failure_desc": "Derive detecteurs / Calibration DAS",
        "action": (
            "Non-uniformite en augmentation. Executer une calibration "
            "a l'air (Air Calibration) et verifier les logs de bruit "
            "electronique du systeme DAS."
        ),
    },
    "table": {
        "label": "Table Patient",
        "icon": "⚙️",
        "metric_label": "Epaisseur coupe (mm)",
        "failure_desc": "Derive mecanique / Alignement",
        "action": (
            "Epaisseur de coupe hors tolerance. Verifier l'alignement "
            "du plan de coupe, la planeite de la table et les rails "
            "de positionnement. Planifier un recalibrage geometrique."
        ),
    },
    "generator": {
        "label": "Generateur HT",
        "icon": "🔋",
        "metric_label": "Precision HU",
        "failure_desc": "Derive kVp / Calibration requise",
        "action": (
            "Precision HU en degradation. Executer la procedure de "
            "calibration kVp GE (Service Mode > X-Ray > Calibration). "
            "Verifier les relais du generateur haute tension."
        ),
    },
}

CANON_COMPONENTS = {
    "tube": {
        "label": "Tube a Rayons X",
        "icon": "🔌",
        "metric_label": "Noise (SD)",
        "failure_desc": "Filament / Anode usee",
        "action": (
            "Bruit image en augmentation. Verifier les logs de "
            "refroidissement et l'usure de l'anode Canon Aquilion. "
            "Planifier un remplacement preventif si SD > 6.0 HU."
        ),
    },
    "gantry": {
        "label": "Gantry / Roulement",
        "icon": "⚡",
        "metric_label": "Uniformite (NUI)",
        "failure_desc": "Usure roulements / Derive detecteurs",
        "action": (
            "Uniformite en degradation. Inspecter les roulements du "
            "gantry et executer une calibration a l'air. "
            "Commander kit de roulements si NUI > 4.0 HU."
        ),
    },
    "table": {
        "label": "Table Patient",
        "icon": "⚙️",
        "metric_label": "Scaling V (mm)",
        "failure_desc": "Friction / Desalignement mecanique",
        "action": (
            "Erreur geometrique detectee. Verifier l'alignement des "
            "lasers, la planeite de la table et les rails de "
            "positionnement. Recalibrer le systeme de positionnement."
        ),
    },
    "generator": {
        "label": "Generateur HT",
        "icon": "🔋",
        "metric_label": "Precision HU",
        "failure_desc": "Derive kVp / Calibration requise",
        "action": (
            "Precision HU en degradation. Executer la calibration HV "
            "automatique (Service > Calibration > kVp). "
            "Verifier les condensateurs du generateur haute tension."
        ),
    },
}

HORIZON_DAYS = 90


# ══════════════════════════════════════════════════════════════════
# Metric Extraction (GE + Siemens)
# ══════════════════════════════════════════════════════════════════

def _extract_metrics() -> tuple[str, dict[str, Optional[float]]]:
    """Extract manufacturer and QA metrics from session_state.

    Returns:
        (manufacturer, metrics_dict)
        manufacturer: "GE" or "SIEMENS"
        metrics_dict: keys depend on manufacturer
    """
    manufacturer = st.session_state.get("manufacturer", "GE")
    kpi = st.session_state.get("siemens_kpi_metrics", {})
    basic = st.session_state.get("basic_result")
    advanced = st.session_state.get("advanced_result")
    acq_params = st.session_state.get("acquisition_params", {})

    if manufacturer in ("SIEMENS", "CANON") and kpi:
        hu_delta = kpi.get("hu_precision_delta")
        raw_noise = kpi.get("noise_sd")
        metrics = {
            "Noise_HU": raw_noise,
            "Uniformity_HU": kpi.get("uniformity_nui"),
            "Scaling_V_mm": kpi.get("scaling_v_mm"),
            "HU_Precision": abs(hu_delta) if hu_delta is not None else None,
        }

        # Siemens also includes acquisition context features
        if manufacturer == "SIEMENS":
            metrics["kVp"] = acq_params.get("kvp")
            metrics["mAs"] = acq_params.get("mas")

        # Imputation for missing QA metrics (both Siemens & Canon)
        imputed = {}
        if metrics["Noise_HU"] is None:
            metrics["Noise_HU"] = 3.0
            imputed["Noise_HU"] = "Baseline (3.0 HU)"
        if metrics["Uniformity_HU"] is None:
            metrics["Uniformity_HU"] = 0.0
            imputed["Uniformity_HU"] = "Baseline (0.0 HU)"
        if metrics["Scaling_V_mm"] is None:
            nominal = 330.0 if manufacturer == "CANON" else 200.0
            metrics["Scaling_V_mm"] = nominal
            imputed["Scaling_V_mm"] = f"Nominal ({nominal:.0f} mm)"
        if metrics["HU_Precision"] is None:
            metrics["HU_Precision"] = 0.0
            imputed["HU_Precision"] = "Baseline (0.0 HU)"

        # ── Canon Noise Normalization ──────────────────────────────
        # The Canon RF models were trained on 5.0 mm slice noise.
        # Physics: noise ~ 1/sqrt(thickness). A 0.5 mm slice has
        # ~3.16x more noise than a 5.0 mm slice.
        # We normalize to the 5.0 mm equivalent so the model
        # doesn't panic on thin-slice protocols.
        if manufacturer == "CANON":
            actual_thickness = acq_params.get("slice_thickness", 5.0)
            if not actual_thickness or actual_thickness <= 0:
                actual_thickness = 5.0
            actual_thickness = float(actual_thickness)

            REFERENCE_THICKNESS = 5.0  # mm — training baseline
            raw_noise_val = metrics["Noise_HU"]

            if actual_thickness != REFERENCE_THICKNESS:
                # Normalize: noise_5mm = noise_raw * sqrt(thickness / 5.0)
                normalized_noise = raw_noise_val * math.sqrt(
                    actual_thickness / REFERENCE_THICKNESS
                )
                metrics["Noise_HU_Raw"] = raw_noise_val
                metrics["Noise_HU"] = normalized_noise
                metrics["Noise_Thickness_mm"] = actual_thickness
            else:
                metrics["Noise_HU_Raw"] = raw_noise_val
                metrics["Noise_Thickness_mm"] = actual_thickness

        return manufacturer, metrics, imputed

    # GE pipeline
    metrics: dict[str, Optional[float]] = {
        "MTF_50_lp_cm": None,
        "Uniformity_HU": None,
        "Slice_Thickness_mm": None,
        "HU_Precision": None,
    }
    imputed: dict[str, str] = {}  # key -> reason

    if advanced is not None and hasattr(advanced, "mtf") and advanced.mtf is not None:
        # Convert lp/mm to lp/cm: 1 lp/mm = 10 lp/cm
        metrics["MTF_50_lp_cm"] = advanced.mtf.mtf_50_lpmm * 10.0

    if basic is not None:
        if basic.uniformity is not None:
            metrics["Uniformity_HU"] = basic.uniformity.non_uniformity_index
        if basic.slice_thickness is not None:
            metrics["Slice_Thickness_mm"] = basic.slice_thickness.measured_fwhm_mm
        if basic.ct_number_accuracy is not None:
            metrics["HU_Precision"] = abs(basic.ct_number_accuracy.max_delta_hu)

    # ── Data Imputation for missing GE metrics ────────────────────
    # Physics-based fallbacks so the RF model can still run.

    # 1. MTF: use bar pattern SD proxy or healthy baseline
    if metrics["MTF_50_lp_cm"] is None:
        if (basic is not None and basic.totalqa_resolution is not None
                and basic.totalqa_resolution.bar_sd_values):
            avg_sd = float(np.mean(basic.totalqa_resolution.bar_sd_values))
            # Heuristic: avg_sd ~ 20-60 HU maps to ~3-6 lp/cm
            mtf_proxy = max(1.0, min(8.0, avg_sd / 10.0))
            metrics["MTF_50_lp_cm"] = mtf_proxy
            imputed["MTF_50_lp_cm"] = f"Bar SD proxy ({avg_sd:.1f} HU)"
        else:
            metrics["MTF_50_lp_cm"] = 4.0  # Healthy baseline
            imputed["MTF_50_lp_cm"] = "Baseline (4.0 lp/cm)"

    # 2. Slice Thickness: use DICOM nominal or fallback
    if metrics["Slice_Thickness_mm"] is None:
        nominal_st = acq_params.get("slice_thickness", 0)
        if nominal_st and nominal_st > 0:
            metrics["Slice_Thickness_mm"] = float(nominal_st)
            imputed["Slice_Thickness_mm"] = f"DICOM nominal ({nominal_st:.1f} mm)"
        else:
            metrics["Slice_Thickness_mm"] = 2.5
            imputed["Slice_Thickness_mm"] = "Baseline (2.5 mm)"

    # 3. Uniformity: fallback to 0.0 (perfect)
    if metrics["Uniformity_HU"] is None:
        metrics["Uniformity_HU"] = 0.0
        imputed["Uniformity_HU"] = "Baseline (0.0 HU)"

    # 4. HU Precision: fallback to 0.0 (perfect)
    if metrics["HU_Precision"] is None:
        metrics["HU_Precision"] = 0.0
        imputed["HU_Precision"] = "Baseline (0.0 HU)"

    return "GE", metrics, imputed


# ══════════════════════════════════════════════════════════════════
# RAG Helpers
# ══════════════════════════════════════════════════════════════════

def _rul_color(rul: Optional[int]) -> str:
    if rul is None:
        return "#8b949e"
    if rul > 60:
        return "#3fb950"
    if rul >= 30:
        return "#d29922"
    return "#f85149"


def _rul_bg(rul: Optional[int]) -> str:
    if rul is None:
        return "rgba(139,148,158,0.08)"
    if rul > 60:
        return "rgba(63,185,80,0.08)"
    if rul >= 30:
        return "rgba(210,153,34,0.08)"
    return "rgba(248,81,73,0.08)"


def _rul_status(rul: Optional[int]) -> tuple[str, str]:
    if rul is None:
        return "N/A", "⚪"
    if rul > 60:
        return "Optimal", "🟢"
    if rul >= 30:
        return "Alerte", "🟡"
    return "Critique", "🔴"


# ══════════════════════════════════════════════════════════════════
# Tab Renderer
# ══════════════════════════════════════════════════════════════════

def render_tab_predictive(scanner_id: str) -> list:
    """Renders the AI Predictive Maintenance dashboard (GE + Siemens)."""
    export_figures: list = []

    hdr("🤖 Intelligence Artificielle — Maintenance Predictive")
    st.caption(
        "Modeles Random Forest entraines sur les Jumeaux Numeriques "
        "SOMATOM go.Sim (Siemens), Discovery RT (GE) et Aquilion LB (Canon). "
        "Prediction de la Duree de Vie Residuelle (RUL) des 4 composants "
        "critiques a partir des metriques QA du jour."
    )

    # ── 1. Extract metrics ────────────────────────────────────────
    manufacturer, metrics, imputed = _extract_metrics()

    # ── Select component map based on manufacturer ────────────────
    if manufacturer == "GE":
        components = GE_COMPONENTS
    elif manufacturer == "CANON":
        components = CANON_COMPONENTS
    else:
        components = SIEMENS_COMPONENTS
    missing = [k for k, v in metrics.items() if v is None]
    all_available = len(missing) == 0

    # ── 2. Input Metrics — Dynamic Cards ─────────────────────────
    vendor_labels = {
        "GE": "GE Discovery RT",
        "SIEMENS": "Siemens SOMATOM",
        "CANON": "Canon Aquilion LB",
    }
    vendor_icons = {
        "GE": "🔵",
        "SIEMENS": "🟠",
        "CANON": "🟣",
    }
    vendor_label = vendor_labels.get(manufacturer, manufacturer)
    vendor_icon = vendor_icons.get(manufacturer, "🟠")

    st.markdown(
        f"### {vendor_icon} Fabricant detecte : {vendor_label}"
    )
    st.caption("Metriques d'entree IA extraites de la session QA du jour.")

    if all_available:
        if manufacturer == "GE":
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                label="🎯 MTF 50%",
                value=f"{metrics['MTF_50_lp_cm']:.2f} lp/cm",
                help=imputed.get("MTF_50_lp_cm"),
            )
            mc2.metric(
                label="📊 Uniformite (NUI)",
                value=f"{metrics['Uniformity_HU']:.3f} HU",
                help=imputed.get("Uniformity_HU"),
            )
            mc3.metric(
                label="📏 Epaisseur de coupe",
                value=f"{metrics['Slice_Thickness_mm']:.2f} mm",
                help=imputed.get("Slice_Thickness_mm"),
            )
            mc4.metric(
                label="⚖️ Precision HU",
                value=f"{metrics['HU_Precision']:.3f} HU",
                help=imputed.get("HU_Precision"),
            )
        elif manufacturer == "CANON":
            # Canon: 4 columns ONLY (No kVp, No mAs)
            # Show RAW noise in UI, but normalized noise goes to AI
            raw_noise = metrics.get("Noise_HU_Raw", metrics["Noise_HU"])
            norm_noise = metrics["Noise_HU"]
            thickness = metrics.get("Noise_Thickness_mm", 5.0)
            is_normalized = abs(raw_noise - norm_noise) > 0.001

            mc1, mc2, mc3, mc4 = st.columns(4)
            noise_help = (
                f"Bruit brut: {raw_noise:.3f} HU @ {thickness:.1f} mm | "
                f"Normalise 5.0mm: {norm_noise:.3f} HU"
                if is_normalized
                else imputed.get("Noise_HU")
            )
            mc1.metric(
                label="🔊 Bruit / Noise",
                value=f"{raw_noise:.3f} HU",
                help=noise_help,
            )
            mc2.metric(
                label="📊 Uniformite (NUI)",
                value=f"{metrics['Uniformity_HU']:.3f} HU",
                help=imputed.get("Uniformity_HU"),
            )
            mc3.metric(
                label="📏 Scaling Vertical",
                value=f"{metrics['Scaling_V_mm']:.3f} mm",
                help=imputed.get("Scaling_V_mm"),
            )
            mc4.metric(
                label="🎯 Precision HU",
                value=f"{metrics['HU_Precision']:.3f} HU",
                help=imputed.get("HU_Precision"),
            )

            # Show normalization notice if noise was adjusted
            if is_normalized:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(163,113,247,0.08), rgba(88,166,255,0.08));
                        border: 1px solid rgba(163,113,247,0.20);
                        border-radius: 8px;
                        padding: 10px 16px;
                        margin-top: 8px;
                        font-size: 13px;
                        color: #8b949e;
                    ">
                        🔬 <b>Normalisation Bruit :</b>
                        Epaisseur de coupe = <b>{thickness:.1f} mm</b> |
                        Bruit brut = <b>{raw_noise:.3f} HU</b> →
                        Normalise 5.0mm = <b>{norm_noise:.3f} HU</b><br/>
                        <em>Formule : Noise_5mm = Noise_raw × sqrt(thickness / 5.0)</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Siemens: 6 metrics including kVp/mAs context
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric(
                label="🔊 Bruit / Noise",
                value=f"{metrics['Noise_HU']:.3f} HU",
            )
            mc2.metric(
                label="📊 Uniformite (NUI)",
                value=f"{metrics['Uniformity_HU']:.3f} HU",
            )
            mc3.metric(
                label="📏 Scaling Vertical",
                value=f"{metrics['Scaling_V_mm']:.3f} mm",
            )
            mc4.metric(
                label="⚖️ Precision HU",
                value=f"{metrics['HU_Precision']:.3f} HU",
            )
            mc5.metric(
                label="⚡ kVp",
                value=f"{metrics['kVp']:.0f} kV",
            )
            mc6.metric(
                label="💡 mAs",
                value=f"{metrics['mAs']:.0f} mAs",
            )

        # Show imputation notice if any values were estimated
        if imputed:
            imputed_list = ", ".join(
                f"**{k}** ({v})" for k, v in imputed.items()
            )
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(88,166,255,0.08), rgba(163,113,247,0.08));
                    border: 1px solid rgba(88,166,255,0.20);
                    border-radius: 8px;
                    padding: 10px 16px;
                    margin-top: 8px;
                    font-size: 13px;
                    color: #8b949e;
                ">
                    ℹ️ <b>Valeurs estimees :</b> {imputed_list}.<br/>
                    <em>Ces metriques manquantes ont ete imputees a partir de
                    donnees DICOM ou de baselines cliniques pour permettre
                    l'inference IA.</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Truly missing values that could not be imputed (should be rare)
        keys = list(metrics.keys())
        n_cols = len(keys)
        cols = st.columns(min(n_cols, 6))
        labels_ge = {
            "MTF_50_lp_cm": "🎯 MTF 50%",
            "Uniformity_HU": "📊 Uniformite",
            "Slice_Thickness_mm": "📏 Epaisseur coupe",
            "HU_Precision": "⚖️ Precision HU",
        }
        labels_si = {
            "Noise_HU": "🔊 Bruit / Noise",
            "Uniformity_HU": "📊 Uniformite",
            "Scaling_V_mm": "📏 Scaling V",
            "HU_Precision": "⚖️ Precision HU",
            "kVp": "⚡ kVp",
            "mAs": "💡 mAs",
        }
        labels_canon = {
            "Noise_HU": "🔊 Bruit / Noise",
            "Uniformity_HU": "📊 Uniformite",
            "Scaling_V_mm": "📏 Scaling V",
            "HU_Precision": "🎯 Precision HU",
        }
        if manufacturer == "GE":
            labels = labels_ge
        elif manufacturer == "CANON":
            labels = labels_canon
        else:
            labels = labels_si
        for col, k in zip(cols, keys):
            v = metrics[k]
            col.metric(
                label=labels.get(k, k),
                value=f"{v:.3f}" if v is not None else "N/A",
            )
        st.warning(
            f"⚠️ **Metriques manquantes** : {', '.join(missing)}.  \n"
            f"L'IA necessite toutes les metriques pour generer des predictions. "
            f"Verifiez que l'analyse QA a ete executee avec succes."
        )
        return export_figures

    st.divider()

    # ── 3. Run AI Inference ───────────────────────────────────────
    preds = predict_rul(manufacturer, metrics)

    # ── 4. RUL Forecast Cards ────────────────────────────────────
    st.markdown("### 🔮 Predictions RUL — 4 Composants")
    cols = st.columns(4)

    for col, (key, cfg) in zip(cols, components.items()):
        rul = preds.get(key)
        color = _rul_color(rul)
        bg = _rul_bg(rul)
        status, icon = _rul_status(rul)

        with col:
            if rul is not None:
                st.markdown(
                    f"<div style='"
                    f"background:{bg};"
                    f"border:1px solid {color};"
                    f"border-radius:12px;"
                    f"padding:16px 12px;"
                    f"text-align:center;"
                    f"min-height:230px;"
                    f"display:flex;flex-direction:column;"
                    f"justify-content:center;align-items:center'>"
                    f"<div style='font-size:32px;margin-bottom:4px'>"
                    f"{cfg['icon']}</div>"
                    f"<div style='font-size:13px;color:#c9d1d9;"
                    f"font-weight:600;margin-bottom:12px'>"
                    f"{cfg['label']}</div>"
                    f"<div style='font-size:42px;font-weight:bold;"
                    f"color:{color};line-height:1'>{rul}</div>"
                    f"<div style='font-size:12px;color:#8b949e;"
                    f"margin-top:4px'>jours restants</div>"
                    f"<div style='margin-top:8px;font-size:11px;"
                    f"color:{color}'>{icon} {status}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='"
                    f"background:rgba(33,38,45,0.6);"
                    f"border:1px solid #30363d;"
                    f"border-radius:12px;"
                    f"padding:20px;text-align:center;"
                    f"min-height:230px;"
                    f"display:flex;flex-direction:column;"
                    f"justify-content:center;align-items:center'>"
                    f"<div style='font-size:32px'>{cfg['icon']}</div>"
                    f"<div style='font-size:13px;color:#c9d1d9;"
                    f"margin:8px 0'>{cfg['label']}</div>"
                    f"<div style='color:#8b949e;font-size:14px'>"
                    f"Modele non disponible</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── 5. Survival Probability Chart ─────────────────────────────
    st.markdown("### 📉 Courbes de Survie — Horizon 90 Jours")
    st.caption(
        "Probabilite estimee qu'un composant reste fonctionnel au jour J.  \n"
        "Modele : P(survie) = max(0, 1 − J / RUL) × 100%."
    )

    fig = _build_survival_chart(preds, components)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 6. Actionable Advisory ────────────────────────────────────
    st.markdown("### 📋 Recommandation de Maintenance Prioritaire")
    _render_advisory(preds, components, vendor_label)

    # ── 7. Detail Table ───────────────────────────────────────────
    with st.expander("📊 Tableau detaille des predictions"):
        import pandas as pd
        rows = []
        for key, cfg in components.items():
            rul = preds.get(key)
            status, _ = _rul_status(rul)
            rows.append({
                "Composant": f"{cfg['icon']} {cfg['label']}",
                "RUL (jours)": rul if rul is not None else "—",
                "Statut": status,
                "Indicateur": cfg["metric_label"],
                "Defaillance": cfg["failure_desc"],
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    return export_figures


# ══════════════════════════════════════════════════════════════════
# Survival Chart Builder
# ══════════════════════════════════════════════════════════════════

_LINE_COLORS = {
    "tube":      "#58a6ff",
    "gantry":    "#f0883e",
    "detectors": "#f0883e",
    "table":     "#3fb950",
    "generator": "#bc8cff",
}


def _build_survival_chart(
    preds: dict[str, Optional[int]],
    components: dict,
) -> go.Figure:
    """Build multi-line Plotly survival probability chart."""
    days = np.arange(0, HORIZON_DAYS + 1)
    fig = go.Figure()

    for key, cfg in components.items():
        rul = preds.get(key)
        if rul is None:
            continue

        if rul > 0:
            survival = np.clip(100.0 * (1.0 - days / rul), 0, 100)
        else:
            survival = np.zeros_like(days, dtype=float)

        fig.add_trace(go.Scatter(
            x=days, y=survival, mode="lines",
            name=f"{cfg['icon']} {cfg['label']} (RUL={rul}j)",
            line=dict(color=_LINE_COLORS.get(key, "#8b949e"), width=2.5),
            hovertemplate=(
                f"<b>{cfg['label']}</b><br>"
                "Jour: %{x}<br>"
                "Survie: %{y:.0f}%<br>"
                "<extra></extra>"
            ),
        ))

    # Danger zone
    fig.add_hrect(
        y0=0, y1=20,
        fillcolor="rgba(248,81,73,0.08)", line_width=0,
        annotation_text="Zone critique",
        annotation_position="bottom right",
        annotation_font_color="#8b949e",
        annotation_font_size=10,
    )

    # 30-day warning line
    fig.add_vline(
        x=30, line_dash="dot", line_color="#d29922",
        annotation_text="30j", annotation_position="top",
        annotation_font_color="#d29922", annotation_font_size=10,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font=dict(color="#c9d1d9", family="Inter, sans-serif"),
        height=420,
        margin=dict(t=40, b=50, l=60, r=30),
        xaxis=dict(
            title="Jours a partir d'aujourd'hui",
            gridcolor="rgba(139,148,158,0.12)",
            range=[0, HORIZON_DAYS], dtick=10,
        ),
        yaxis=dict(
            title="Probabilite de Survie (%)",
            gridcolor="rgba(139,148,158,0.12)",
            range=[0, 105], dtick=10,
        ),
        legend=dict(
            bgcolor="rgba(33,38,45,0.8)",
            bordercolor="#30363d", borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
        ),
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# Advisory
# ══════════════════════════════════════════════════════════════════

def _render_advisory(
    preds: dict[str, Optional[int]],
    components: dict,
    vendor_label: str,
) -> None:
    """Priority maintenance advisory for the weakest component."""
    valid = {k: v for k, v in preds.items() if v is not None}
    if not valid:
        st.info("Aucune prediction disponible pour generer un avis.")
        return

    weakest_key = min(valid, key=valid.get)
    weakest_rul = valid[weakest_key]
    cfg = components[weakest_key]

    if weakest_rul < 30:
        st.error(
            f"🚨 **ACTION URGENTE — {cfg['icon']} {cfg['label']} "
            f"({vendor_label})** "
            f"(RUL = **{weakest_rul} jours**)\n\n"
            f"**Defaillance anticipee :** {cfg['failure_desc']}\n\n"
            f"**Action requise :** {cfg['action']}"
        )
    elif weakest_rul < 60:
        st.warning(
            f"⚠️ **ALERTE PRECOCE — {cfg['icon']} {cfg['label']} "
            f"({vendor_label})** "
            f"(RUL = **{weakest_rul} jours**)\n\n"
            f"**Defaillance anticipee :** {cfg['failure_desc']}\n\n"
            f"**Action recommandee :** {cfg['action']}"
        )
    else:
        st.success(
            f"✅ **Systeme optimal ({vendor_label}).** "
            f"Tous les composants ont un RUL > 60 jours. "
            f"Composant le plus proche : "
            f"**{cfg['icon']} {cfg['label']}** ({weakest_rul} jours).\n\n"
            f"Aucune action de maintenance preventive n'est requise."
        )

    # Priority list of other components
    sorted_preds = sorted(valid.items(), key=lambda x: x[1])
    remaining = [(k, v) for k, v in sorted_preds if k != weakest_key]
    if remaining:
        with st.expander("📋 Priorite des autres composants"):
            for key, rul in remaining:
                c = components[key]
                color = _rul_color(rul)
                st.markdown(
                    f"- {c['icon']} **{c['label']}** : "
                    f"<span style='color:{color};font-weight:bold'>"
                    f"{rul} jours</span> — {c['failure_desc']}",
                    unsafe_allow_html=True,
                )
