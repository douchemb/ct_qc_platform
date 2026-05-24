import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from dashboard.db_manager import get_all_sessions
from predictive_maintenance.inference import predict_rul

def render_tab_history() -> list:
    """Affiche l'onglet Historique & Tendances avec Progressive Disclosure."""
    st.markdown("## 📈 Historique & Tendances Réactives")
    st.caption("Filtrez par scanner pour visualiser son historique. Cliquez sur le bouton pour générer les analyses de tendances.")
    
    df = get_all_sessions()
    
    if df.empty:
        st.info("ℹ️ La base de données est vide.")
        return []
    
    # ── Étape 1 : Filtre & Tableau (Toujours visible) ──────────────────────────
    options_scanners = ["Scanner SIEMENS (SOMATOM)", "Scanner GE (Discovery RT)", "Scanner CANON (Aquilion LB)"]
    scanner_selection = st.selectbox("Sélectionnez l'équipement à analyser :", options=options_scanners)
    
    df_filtered = df[df["scanner_model"] == scanner_selection].copy()
    
    if df_filtered.empty:
        st.warning(f"Aucune donnée d'historique pour {scanner_selection}.")
        return []
        
    st.markdown("### 📋 Base de Données des Sessions (Filtrée)")
    st.dataframe(
        df_filtered, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID Session", format="%d"),
            "date_analyse": st.column_config.DatetimeColumn("Date d'Analyse", format="DD/MM/YYYY HH:mm"),
            "scanner_model": "Modèle Scanner",
            "noise_hu": st.column_config.NumberColumn("Bruit (HU)", format="%.3f"),
            "uniformity_hu": st.column_config.NumberColumn("Uniformité (HU)", format="%.3f"),
            "mtf_50": st.column_config.NumberColumn("MTF 50% (lp/cm)", format="%.2f"),
            "global_score_percent": st.column_config.NumberColumn("Score Global (%)", format="%.1f"),
            "tube_rul_days": st.column_config.NumberColumn("RUL Tube (Jours)", format="%d")
        }
    )
    
    st.divider()

    # ── Étape 2 : Le Bouton Déclencheur (Progressive Disclosure) ───────────────
    if st.button("🚀 Lancer l'Analyse Historique", use_container_width=True):
        
        # --- (Logique backend de recalcul historique) ---
        mfr_map = {
            "Scanner SIEMENS (SOMATOM)": "SIEMENS",
            "Scanner CANON (Aquilion LB)": "CANON",
            "Scanner GE (Discovery RT)": "GE",
        }
        mfr_code = mfr_map[scanner_selection]
        
        history_ruls = {"date": [], "tube": [], "gantry": [], "table": [], "generator": []}
        
        for _, row in df_filtered.sort_values("date_analyse").iterrows():
            metrics = {
                "Noise_HU": float(row["noise_hu"]) if pd.notna(row["noise_hu"]) else 0.0,
                "Uniformity_HU": float(row["uniformity_hu"]) if pd.notna(row["uniformity_hu"]) else 0.0,
                "MTF_50_lp_cm": float(row["mtf_50"]) if pd.notna(row["mtf_50"]) else 0.0,
                "Scaling_V_mm": 330.0, "Slice_Thickness_mm": 5.0, "HU_Precision": 0.5,
                "kVp": 120.0, "mAs": 200.0,
            }
            preds = predict_rul(mfr_code, metrics)
            history_ruls["date"].append(row["date_analyse"])
            history_ruls["tube"].append(preds.get("tube", 0))
            gantry_key = "detectors" if mfr_code == "GE" else "gantry"
            history_ruls["gantry"].append(preds.get(gantry_key, 0))
            history_ruls["table"].append(preds.get("table", 0))
            history_ruls["generator"].append(preds.get("generator", 0))
            
        df_rul_history = pd.DataFrame(history_ruls)
        
        # ── Étape 3 : Génération des Graphiques ────────────────────────────────────
        
        # Section A : Évolution des Métriques QA (Cause)
        st.markdown("### 🔬 Section A : Évolution de la Qualité d'Image")
        
        fig_qa = px.line(
            df_filtered.sort_values("date_analyse"), 
            x="date_analyse", 
            y=["noise_hu", "uniformity_hu", "mtf_50"],
            labels={"value": "Valeur", "date_analyse": "Date d'Analyse", "variable": "Métrique"},
            markers=True
        )
        fig_qa.update_traces(mode="lines+markers")
        fig_qa.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", 
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_qa, use_container_width=True)

        st.divider()

        # Section B : Évolution de l'Usure Matérielle (Conséquence)
        st.markdown("### ⚙️ Section B : Dégradation Historique du Matériel (RUL)")
        
        fig_rul = px.line(
            df_rul_history, 
            x="date", 
            y=["tube", "gantry", "table", "generator"],
            labels={"value": "Jours Restants (RUL)", "date": "Date d'Analyse", "variable": "Composant"},
            markers=True
        )
        fig_rul.update_traces(mode="lines+markers")
        fig_rul.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", 
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_title="Jours Restants"
        )
        # Ligne rouge critique à 90 jours
        fig_rul.add_hline(y=90, line_dash="dot", line_color="#f85149", annotation_text="Seuil Critique (90j)")
        
        st.plotly_chart(fig_rul, use_container_width=True)

    return []
