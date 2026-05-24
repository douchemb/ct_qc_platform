"""
dashboard/tab_hardware_health.py — Tab 4: Conformité Clinique (QA).

Affiche le statut clinique et l'impact matériel basé sur les prédictions RUL de l'IA.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

_PM_DIR = Path(__file__).resolve().parent.parent / "predictive_maintenance"
if str(_PM_DIR) not in sys.path:
    sys.path.insert(0, str(_PM_DIR))

try:
    from inference import predict_rul
    from dashboard.tab_predictive import _extract_metrics
except ImportError:
    pass

def render_tab_hardware_health() -> list:
    """Renders the Intégrité Matérielle & Impact Clinique tab."""
    export_figures = []

    st.markdown("## 📊 STATUT OPÉRATIONNEL & CONFORMITÉ CLINIQUE (IA)")

    # Extraction des métriques RUL (Priorité à st.session_state)
    preds = st.session_state.get("ai_rul_predictions")
    if not preds:
        try:
            manufacturer, metrics, _ = _extract_metrics()
            preds = predict_rul(manufacturer, metrics)
        except Exception:
            preds = {}
            st.info("ℹ️ Les prédictions RUL de l'IA ne sont pas disponibles. Veuillez analyser une série.")
            return export_figures

    rul_values = [v for v in preds.values() if v is not None]
    if not rul_values:
        st.info("ℹ️ Aucune donnée QA suffisante pour l'évaluation matérielle.")
        return export_figures

    min_rul = min(rul_values)

    # Section Supérieure : Jauge et Verdict
    col_gauge, col_verdict = st.columns([2, 1])

    with col_gauge:
        import plotly.graph_objects as go
        
        # Plafonner le score à 100 pour l'affichage en pourcentage
        score = min(min_rul, 100)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            number = {'suffix': "%"},
            title = {'text': "Score de Conformité Globale"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "rgba(255, 255, 255, 0.4)", 'thickness': 0.2},
                'steps': [
                    {'range': [0, 60], 'color': "#f85149"},      # Rouge
                    {'range': [60, 85], 'color': "#d29922"},     # Orange/Jaune
                    {'range': [85, 100], 'color': "#3fb950"}     # Vert
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_verdict:
        st.markdown("### 📋 VERDICT CLINIQUE")
        if min_rul >= 85:
            st.success("✅ **Autorisation Totale.** Le système est parfaitement calibré. Apte pour les protocoles exigeants (Oncologie, Cardio, HRCT).")
        elif min_rul >= 60:
            st.warning("⚠️ **Restrictions Recommandées.** Le système présente une légère dégradation. Évitez les protocoles à très basse dose ou haute résolution spatiale. Apte pour le standard.")
        else:
            st.error("🛑 **Intervention Requise.** Qualité d'image ou géométrie compromise. Maintenance immédiate conseillée.")

    st.markdown("<br/>", unsafe_allow_html=True)

    def get_status_tuple(rul: Optional[int]) -> tuple[str, str]:
        if rul is None:
            return "⚪ INCONNU", "unknown"
        if rul > 60:
            return "🟢 OPTIMAL", "success"
        if rul >= 30:
            return "🟡 DÉGRADÉ", "warning"
        return "🔴 CRITIQUE", "error"

    def render_impact_alert(state: str, risk_text: str):
        if state == "success":
            st.success("✅ **Impact Clinique :** Matériel conforme. Qualité d'image optimale.")
        elif state == "warning":
            st.warning(f"⚠️ **Risque Modéré :** {risk_text}")
        elif state == "error":
            st.error(f"🚨 **Risque Critique :** {risk_text}")
        else:
            st.info("ℹ️ Données insuffisantes pour évaluer l'impact.")

    # Extraction des RUL par composant
    rul_tube = preds.get("tube")
    rul_das = preds.get("detectors") if "detectors" in preds else preds.get("gantry")
    rul_meca = preds.get("table")
    rul_gen = preds.get("generator")

    # Matrice d'Impact (2x2)
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("⚛️ Tube à Rayons X")
            val, state = get_status_tuple(rul_tube)
            st.metric(label="Diagnostic IA", value=val)
            st.divider()
            st.info("👁️ **Métrique cible :** Bruit (SD) & MTF")
            render_impact_alert(state, "Augmentation du bruit quantique, perte de résolution spatiale (MTF).")

        st.markdown("<br/>", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("⚙️ Mécanique (Gantry & Table)")
            val, state = get_status_tuple(rul_meca)
            st.metric(label="Diagnostic IA", value=val)
            st.divider()
            st.info("👁️ **Métrique cible :** Épaisseur de coupe, Alignement, Scaling")
            render_impact_alert(state, "Erreur de scaling, imprécision géométrique.")

    with col2:
        with st.container(border=True):
            st.subheader("📡 Détecteurs / DAS")
            val, state = get_status_tuple(rul_das)
            st.metric(label="Diagnostic IA", value=val)
            st.divider()
            st.info("👁️ **Métrique cible :** Uniformité (NUI), Artefacts")
            render_impact_alert(state, "Artefacts en anneau, dérive de l'uniformité.")

        st.markdown("<br/>", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("🔋 Générateur HT")
            val, state = get_status_tuple(rul_gen)
            st.metric(label="Diagnostic IA", value=val)
            st.divider()
            st.info("👁️ **Métrique cible :** Linéarité HU, Précision CT Number")
            render_impact_alert(state, "Instabilité du contraste et de la précision HU.")

    return export_figures

