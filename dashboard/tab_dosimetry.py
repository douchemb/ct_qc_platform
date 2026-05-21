# -*- coding: utf-8 -*-
"""
dashboard/tab_dosimetry.py -- Tab 3: Dosimetrie (AAPM TG-204 / SSDE / FOM).

Scientific dosimetry module compatible with both GE and Siemens pipelines.
Reads pre-computed dosimetry metrics from session_state["dosimetry_metrics"].

Reference standards:
  AAPM Report 204 (2011) -- Size-Specific Dose Estimates
  AAPM TG-220 (2014) -- Water-equivalent diameter
  IEC 60601-2-44 -- CTDIvol definition
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from dashboard.helpers import hdr, kpi_card, apply_dark_style, render_fig


def render_tab_dosimetry(
    advanced_result=None,
    mined_metadata=None,
) -> list[tuple[str, plt.Figure]]:
    """Renders the TG-204 Dosimetry dashboard.

    Reads dosimetry_metrics dict from session_state (computed by
    the GE or Siemens orchestrator). Falls back gracefully when
    CTDIvol is missing from DICOM headers.
    """
    export_figures: list[tuple[str, plt.Figure]] = []

    dm = st.session_state.get("dosimetry_metrics")
    if dm is None:
        st.info(
            "Dosimetrie non disponible. "
            "Lancez l'analyse avec des fichiers DICOM contenant les "
            "metadonnees de dose (CTDIvol, DLP)."
        )
        return export_figures

    ctdi = dm.get("ctdi_vol_mgy")
    dlp = dm.get("dlp_mgy_cm")
    ssde = dm.get("ssde_mgy")
    fom = dm.get("fom")
    noise = dm.get("noise_sd")
    ap = dm.get("ap_cm")
    lat = dm.get("lat_cm")
    d_eff = dm.get("d_eff_cm")
    f_factor = dm.get("f_factor")

    # ── EARLY EXIT — CTDIvol absent : Total Wipeout + Premium UI ────────
    # return export_figures sort uniquement de cette fonction — les autres
    # onglets (Predictif, Sante) continuent de s'executer normalement.
    # Ne PAS utiliser st.stop() ici : il tuerait toute l'application.
    if ctdi is None:
        _html = (
            "<style>"
            ".premium-error-wrapper {"
            "  display:flex; justify-content:center; align-items:center;"
            "  min-height:62vh; width:100%; padding:2rem 1rem; box-sizing:border-box;"
            "}"
            ".premium-error-container {"
            "  background:linear-gradient(145deg,#161b22 0%,#1a1f2e 60%,#0d1117 100%);"
            "  border:1px solid rgba(210,153,34,0.45); border-radius:16px;"
            "  box-shadow:0 0 0 1px rgba(210,153,34,0.08),"
            "             0 8px 32px rgba(0,0,0,0.55),"
            "             0 2px 8px rgba(210,153,34,0.12),"
            "             inset 0 1px 0 rgba(255,255,255,0.04);"
            "  padding:3.5rem 4rem; max-width:640px; width:100%;"
            "  text-align:center; position:relative; overflow:hidden;"
            "}"
            ".premium-error-container::before {"
            "  content:''; position:absolute; top:0; left:0; right:0; height:3px;"
            "  background:linear-gradient(90deg,transparent 0%,"
            "    rgba(210,153,34,0.8) 30%,rgba(248,81,73,0.6) 70%,transparent 100%);"
            "  border-radius:16px 16px 0 0;"
            "}"
            ".pec-icon {"
            "  margin:0 auto 1.6rem; width:72px; height:72px;"
            "  display:flex; align-items:center; justify-content:center;"
            "  background:rgba(210,153,34,0.08); border:1px solid rgba(210,153,34,0.25);"
            "  border-radius:50%;"
            "}"
            ".pec-title {"
            "  font-family:'Inter','Segoe UI',system-ui,sans-serif;"
            "  font-size:1.55rem; font-weight:700; color:#f0f6fc;"
            "  margin:0 0 0.85rem; letter-spacing:-0.02em; line-height:1.25;"
            "}"
            ".pec-tag {"
            "  display:inline-block;"
            "  font-family:'SF Mono','Fira Code',monospace;"
            "  font-size:0.78rem; color:#d29922;"
            "  background:rgba(210,153,34,0.1); border:1px solid rgba(210,153,34,0.3);"
            "  border-radius:6px; padding:0.2rem 0.6rem; margin-bottom:1.2rem;"
            "  letter-spacing:0.04em;"
            "}"
            ".pec-subtitle {"
            "  font-size:0.92rem; color:#8b949e; line-height:1.7;"
            "  margin:0 0 2rem; max-width:480px; margin-left:auto; margin-right:auto;"
            "}"
            ".pec-subtitle strong { color:#c9d1d9; }"
            ".pec-blocked { display:flex; flex-direction:column; gap:0.5rem; margin-bottom:2rem; }"
            ".pec-blocked-item {"
            "  display:flex; align-items:center; gap:0.6rem;"
            "  font-size:0.85rem; color:#6e7681;"
            "  background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04);"
            "  border-radius:8px; padding:0.5rem 0.9rem;"
            "}"
            ".pec-dot { width:6px; height:6px; border-radius:50%;"
            "  background:rgba(248,81,73,0.5); flex-shrink:0; }"
            ".pec-action {"
            "  font-size:0.80rem; color:#484f58;"
            "  border-top:1px solid rgba(255,255,255,0.05);"
            "  padding-top:1.2rem; line-height:1.6;"
            "}"
            "</style>"
            '<div class="premium-error-wrapper">'
            '  <div class="premium-error-container">'
            '    <div class="pec-icon">'
            '      <svg width="34" height="34" viewBox="0 0 24 24" fill="none"'
            '           xmlns="http://www.w3.org/2000/svg">'
            '        <path d="M12 2L2 7v5c0 5.25 4.3 10.15 10 11.36'
            '                 C17.7 22.15 22 17.25 22 12V7L12 2z"'
            '              stroke="#d29922" stroke-width="1.6"'
            '              stroke-linejoin="round" fill="rgba(210,153,34,0.08)"/>'
            '        <line x1="8" y1="8" x2="16" y2="16"'
            '              stroke="#f85149" stroke-width="1.8" stroke-linecap="round"/>'
            '        <line x1="16" y1="8" x2="8" y2="16"'
            '              stroke="#f85149" stroke-width="1.8" stroke-linecap="round"/>'
            '      </svg>'
            '    </div>'
            '    <h2 class="pec-title">Donnees de Dosimetrie Insuffisantes</h2>'
            '    <div class="pec-tag">DICOM TAG (0018,9345) &mdash; CTDIvol &middot; ABSENT</div>'
            '    <p class="pec-subtitle">'
            '      Le parametre <strong>CTDIvol</strong> est absent des en-tetes DICOM.'
            '      Tous les calculs de dosimetrie personnalisee et les profils'
            '      morphometriques ont ete <strong>desactives</strong> pour cette session.'
            '    </p>'
            '    <div class="pec-blocked">'
            '      <div class="pec-blocked-item"><span class="pec-dot"></span>'
            '        SSDE &nbsp;&mdash;&nbsp; Size-Specific Dose Estimate (AAPM TG-204)</div>'
            '      <div class="pec-blocked-item"><span class="pec-dot"></span>'
            '        FOM &nbsp;&mdash;&nbsp; Figure of Merit (efficacite dose / bruit)</div>'
            '      <div class="pec-blocked-item"><span class="pec-dot"></span>'
            '        Courbe f(D_eff) &nbsp;&mdash;&nbsp; Facteur de conversion PMMA 32 cm</div>'
            '      <div class="pec-blocked-item"><span class="pec-dot"></span>'
            '        Morphometrie &nbsp;&mdash;&nbsp; Diametre effectif AP / LAT / D_eff</div>'
            '    </div>'
            '    <p class="pec-action">'
            '      Verifiez que le scanner transfere les tags de dose dans les'
            '      en-tetes DICOM, ou activez la <strong>dose structuree (RDSR)</strong>'
            '      dans le menu de configuration du systeme.'
            '    </p>'
            '  </div>'
            '</div>'
        )
        st.markdown(_html, unsafe_allow_html=True)
        return export_figures   # sort de la fonction uniquement -- app intacte

    # ── Dynamic DRL limits based on scanner type ──────────────────
    manufacturer = st.session_state.get("manufacturer", "GE")
    if manufacturer == "CANON":
        drl_limit = 100.0   # RT Simulation / large QA phantom
        drl_label = "DRL RT Sim / QA"
    else:
        drl_limit = 25.0    # Standard diagnostic abdomen
        drl_label = "DRL Abdomen"

    # ── 1. Top Row: 4 KPI Cards ───────────────────────────────────
    hdr("Metriques Dosimetriques -- AAPM TG-204")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if ctdi is not None:
            drl_ok = ctdi < drl_limit
            st.markdown(kpi_card(
                "[D]", "CTDIvol",
                f"{ctdi:.2f} mGy",
                f"NRD < {drl_limit:.0f} mGy ({drl_label}) | Source: DICOM",
                drl_ok,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "[D]", "CTDIvol",
                "N/A",
                "Tag DICOM absent -- verifiez les en-tetes",
                None,
            ), unsafe_allow_html=True)

    with c2:
        if dlp is not None:
            dlp_source = dm.get("dlp_source", "dicom")
            if dlp_source == "calculated":
                dlp_label = "Source: Calcule (CTDIvol x Epaisseur)"
            else:
                dlp_label = "Dose-Length Product | DICOM (0018,9346)"
            st.markdown(kpi_card(
                "[G]", "DLP",
                f"{dlp:.2f} mGy.cm",
                dlp_label,
                True,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "[G]", "DLP",
                "N/A",
                "Tag DICOM absent",
                None,
            ), unsafe_allow_html=True)

    with c3:
        if ssde is not None:
            ssde_ok = ssde < drl_limit
            st.markdown(kpi_card(
                "[S]", "SSDE",
                f"{ssde:.2f} mGy",
                f"f={f_factor:.4f} x CTDIvol | NRD < {drl_limit:.0f} mGy",
                ssde_ok,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "[S]", "SSDE",
                "N/A",
                "Necessite CTDIvol + diametres",
                None,
            ), unsafe_allow_html=True)

    with c4:
        if fom is not None:
            fom_ok = fom > 0.0
            st.markdown(kpi_card(
                "[F]", "FOM (Efficacite)",
                f"{fom:.4e}",
                f"1/(sigma2 x CTDIvol) | sigma={noise:.2f} HU",
                fom_ok,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "[F]", "FOM",
                "N/A",
                "Necessite Noise + CTDIvol",
                None,
            ), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Verdict Banner ────────────────────────────────────────────
    if ctdi is not None:
        if ctdi < drl_limit:
            st.success(
                f"CONFORME -- CTDIvol ({ctdi:.2f} mGy) < {drl_limit:.0f} mGy -- "
                f"Conforme aux NRD ({drl_label})"
            )
        else:
            st.error(
                f"HORS LIMITE -- CTDIvol ({ctdi:.2f} mGy) >= {drl_limit:.0f} mGy -- "
                f"Depasse les NRD ({drl_label} = {drl_limit:.0f} mGy)"
            )
    else:
        st.warning(
            "CTDIvol non disponible -- Le tag DICOM (0018,9345) "
            "est absent des en-tetes. Les metriques SSDE et FOM ne "
            "peuvent pas etre calculees."
        )

    st.divider()

    # ── 2. Morphometrie & SSDE Breakdown ──────────────────────────
    hdr("Morphometrie & Calcul SSDE -- AAPM Report 204")

    col_morph, col_ssde = st.columns(2)

    with col_morph:
        st.markdown("#### Dimensions Mesurees")
        morph_data = {
            "Parametre": [
                "AP (Vertical)", "LAT (Horizontal)",
                "D_eff (Effectif)",
            ],
            "Valeur": [
                f"{ap:.2f} cm" if ap else "N/A",
                f"{lat:.2f} cm" if lat else "N/A",
                f"{d_eff:.2f} cm" if d_eff else "N/A",
            ],
            "Formule": [
                "V_diameter / 10",
                "H_diameter / 10",
                "sqrt(AP x LAT)",
            ],
        }
        import pandas as pd
        st.dataframe(
            pd.DataFrame(morph_data),
            use_container_width=True, hide_index=True,
        )

        if d_eff is not None:
            st.latex(
                r"D_{eff} = \sqrt{AP \times LAT} = "
                r"\sqrt{%.2f \times %.2f} = %.2f \text{ cm}"
                % (ap or 0, lat or 0, d_eff)
            )

    with col_ssde:
        st.markdown("#### Calcul SSDE (TG-204)")

        if f_factor is not None and ctdi is not None:
            ssde_data = {
                "Etape": [
                    "1. CTDIvol",
                    "2. D_eff",
                    "3. f(D_eff)",
                    "4. SSDE",
                ],
                "Valeur": [
                    f"{ctdi:.3f} mGy",
                    f"{d_eff:.2f} cm",
                    f"{f_factor:.4f}",
                    f"{ssde:.3f} mGy",
                ],
                "Source": [
                    "DICOM (0018,9345)",
                    "sqrt(AP x LAT)",
                    "3.704 x exp(-0.0367 x D_eff)",
                    "CTDIvol x f",
                ],
            }
            st.dataframe(
                pd.DataFrame(ssde_data),
                use_container_width=True, hide_index=True,
            )

            st.latex(
                r"f = 3.704 \times e^{-0.0367 \times %.2f} = %.4f"
                % (d_eff, f_factor)
            )
            st.latex(
                r"\text{SSDE} = %.3f \times %.4f = %.3f \text{ mGy}"
                % (ctdi, f_factor, ssde)
            )
        else:
            st.info(
                "Calcul SSDE impossible -- CTDIvol ou diametres manquants."
            )

    st.divider()

    # ── 3. FOM Breakdown ──────────────────────────────────────────
    if fom is not None:
        hdr("Figure of Merit -- Efficacite Dose/Bruit")
        st.latex(
            r"\text{FOM} = \frac{1}{\sigma^2 \times \text{CTDIvol}} = "
            r"\frac{1}{%.3f^2 \times %.3f} = %.4e"
            % (noise, ctdi, fom)
        )
        st.caption(
            "Le FOM quantifie l'efficacite du protocole : "
            "un FOM plus eleve indique un meilleur rapport qualite "
            "image / dose. Comparer entre protocoles pour optimiser."
        )
        st.divider()

    # ── 4. TG-204 Conversion Factor Curve ─────────────────────────
    if d_eff is not None:
        hdr("Courbe f(D_eff) -- AAPM Report 204 (32 cm PMMA)")
        fig_f, ax = plt.subplots(figsize=(9, 4))
        fig_f.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        d_range = np.linspace(5, 45, 200)
        f_curve = 3.704 * np.exp(-0.0367 * d_range)

        ax.plot(d_range, f_curve, color="#58a6ff", linewidth=2.5,
                label="f(D_eff) = 3.704 x exp(-0.0367 x D_eff)")
        ax.fill_between(d_range, f_curve, alpha=0.08, color="#58a6ff")

        # Mark the measured point
        if f_factor is not None:
            ax.plot(d_eff, f_factor, "o", color="#f85149", markersize=12,
                    zorder=5, label=f"Mesure: D_eff={d_eff:.1f} cm, f={f_factor:.3f}")
            ax.axvline(d_eff, color="#f85149", linestyle="--",
                       linewidth=1.0, alpha=0.5)
            ax.axhline(f_factor, color="#f85149", linestyle="--",
                       linewidth=1.0, alpha=0.5)

        # Reference line f=1 (SSDE = CTDIvol)
        ax.axhline(1.0, color="#8b949e", linestyle=":", linewidth=1.0,
                   alpha=0.6, label="f = 1.0 (SSDE = CTDIvol)")

        ax.set_xlabel("Diametre Effectif D_eff (cm)",
                      color="white", fontsize=10)
        ax.set_ylabel("Facteur de conversion f",
                      color="white", fontsize=10)
        ax.set_title("AAPM Report 204 -- Conversion Factor (32 cm PMMA)",
                     color="white", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="none", labelcolor="white",
                  framealpha=0.7)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.2, color="lightgray")
        fig_f.tight_layout()

        render_fig(fig_f, "tg204_f_curve",
                   "tg204_f_curve_report204.png",
                   caption="f(D_eff) -- Ref. AAPM Report 204, Table 1")
        export_figures.append(("Fig. -- TG-204 f(D_eff) Curve", fig_f))

    return export_figures
