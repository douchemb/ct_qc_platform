"""
dashboard/tab_dosimetry.py — Tab 3: Dosimétrie (AAPM TG-204 / SSDE / FOM).

Scientific dosimetry module compatible with both GE and Siemens pipelines.
Reads pre-computed dosimetry metrics from session_state["dosimetry_metrics"].

Reference standards:
  AAPM Report 204 (2011) — Size-Specific Dose Estimates
  AAPM TG-220 (2014) — Water-equivalent diameter
  IEC 60601-2-44 — CTDIvol definition
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
            "ℹ️ Dosimétrie non disponible. "
            "Lancez l'analyse avec des fichiers DICOM contenant les "
            "métadonnées de dose (CTDIvol, DLP)."
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

    # ── Dynamic DRL limits based on scanner type ──────────────────
    manufacturer = st.session_state.get("manufacturer", "GE")
    if manufacturer == "CANON":
        drl_limit = 100.0   # RT Simulation / large QA phantom
        drl_label = "DRL RT Sim / QA"
    else:
        drl_limit = 25.0    # Standard diagnostic abdomen
        drl_label = "DRL Abdomen"

    # ── 1. Top Row: 4 KPI Cards ───────────────────────────────────
    hdr("Métriques Dosimétriques — AAPM TG-204")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if ctdi is not None:
            drl_ok = ctdi < drl_limit
            st.markdown(kpi_card(
                "💊", "CTDIvol",
                f"{ctdi:.2f} mGy",
                f"NRD < {drl_limit:.0f} mGy ({drl_label}) | Source: DICOM",
                drl_ok,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "💊", "CTDIvol",
                "N/A",
                "Tag DICOM absent — vérifiez les en-têtes",
                None,
            ), unsafe_allow_html=True)

    with c2:
        if dlp is not None:
            dlp_source = dm.get("dlp_source", "dicom")
            if dlp_source == "calculated":
                dlp_label = "Source: Calculé (CTDIvol × Épaisseur)"
            else:
                dlp_label = "Dose-Length Product | DICOM (0018,9346)"
            st.markdown(kpi_card(
                "📊", "DLP",
                f"{dlp:.2f} mGy·cm",
                dlp_label,
                True,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "📊", "DLP",
                "N/A",
                "Tag DICOM absent",
                None,
            ), unsafe_allow_html=True)

    with c3:
        if ssde is not None:
            ssde_ok = ssde < drl_limit
            st.markdown(kpi_card(
                "🎯", "SSDE",
                f"{ssde:.2f} mGy",
                f"f={f_factor:.4f} × CTDIvol | NRD < {drl_limit:.0f} mGy",
                ssde_ok,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "🎯", "SSDE",
                "N/A",
                "Nécessite CTDIvol + diamètres",
                None,
            ), unsafe_allow_html=True)

    with c4:
        if fom is not None:
            fom_ok = fom > 0.0
            st.markdown(kpi_card(
                "⚡", "FOM (Efficacité)",
                f"{fom:.4e}",
                f"1/(σ²×CTDIvol) | σ={noise:.2f} HU",
                fom_ok,
            ), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card(
                "⚡", "FOM",
                "N/A",
                "Nécessite Noise + CTDIvol",
                None,
            ), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Verdict Banner ────────────────────────────────────────────
    if ctdi is not None:
        if ctdi < drl_limit:
            st.success(
                f"✅ **CTDIvol ({ctdi:.2f} mGy) < {drl_limit:.0f} mGy** — "
                f"Conforme aux NRD ({drl_label})"
            )
        else:
            st.error(
                f"❌ **CTDIvol ({ctdi:.2f} mGy) ≥ {drl_limit:.0f} mGy** — "
                f"Dépasse les NRD ({drl_label} = {drl_limit:.0f} mGy)"
            )
    else:
        st.warning(
            "⚠️ **CTDIvol non disponible** — Le tag DICOM (0018,9345) "
            "est absent des en-têtes. Les métriques SSDE et FOM ne "
            "peuvent pas être calculées."
        )

    st.divider()

    # ── 2. Morphométrie & SSDE Breakdown ──────────────────────────
    hdr("Morphométrie & Calcul SSDE — AAPM Report 204")

    col_morph, col_ssde = st.columns(2)

    with col_morph:
        st.markdown("#### 📐 Dimensions Mesurées")
        morph_data = {
            "Paramètre": [
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
                "√(AP × LAT)",
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
        st.markdown("#### 🧮 Calcul SSDE (TG-204)")

        if f_factor is not None and ctdi is not None:
            ssde_data = {
                "Étape": [
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
                    "√(AP × LAT)",
                    "3.704 × exp(-0.0367 × D_eff)",
                    "CTDIvol × f",
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
                "ℹ️ Calcul SSDE impossible — CTDIvol ou diamètres manquants."
            )

    st.divider()

    # ── 3. FOM Breakdown ──────────────────────────────────────────
    if fom is not None:
        hdr("Figure of Merit — Efficacité Dose/Bruit")
        st.latex(
            r"\text{FOM} = \frac{1}{\sigma^2 \times \text{CTDIvol}} = "
            r"\frac{1}{%.3f^2 \times %.3f} = %.4e"
            % (noise, ctdi, fom)
        )
        st.caption(
            "Le FOM quantifie l'efficacité du protocole : "
            "un FOM plus élevé indique un meilleur rapport qualité "
            "image / dose. Comparer entre protocoles pour optimiser."
        )
        st.divider()

    # ── 4. TG-204 Conversion Factor Curve ─────────────────────────
    if d_eff is not None:
        hdr("Courbe f(D_eff) — AAPM Report 204 (32 cm PMMA)")
        fig_f, ax = plt.subplots(figsize=(9, 4))
        fig_f.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        d_range = np.linspace(5, 45, 200)
        f_curve = 3.704 * np.exp(-0.0367 * d_range)

        ax.plot(d_range, f_curve, color="#58a6ff", linewidth=2.5,
                label="f(D_eff) = 3.704·exp(−0.0367·D_eff)")
        ax.fill_between(d_range, f_curve, alpha=0.08, color="#58a6ff")

        # Mark the measured point
        if f_factor is not None:
            ax.plot(d_eff, f_factor, "o", color="#f85149", markersize=12,
                    zorder=5, label=f"Mesuré: D_eff={d_eff:.1f} cm, f={f_factor:.3f}")
            ax.axvline(d_eff, color="#f85149", linestyle="--",
                       linewidth=1.0, alpha=0.5)
            ax.axhline(f_factor, color="#f85149", linestyle="--",
                       linewidth=1.0, alpha=0.5)

        # Reference line f=1 (SSDE = CTDIvol)
        ax.axhline(1.0, color="#8b949e", linestyle=":", linewidth=1.0,
                   alpha=0.6, label="f = 1.0 (SSDE = CTDIvol)")

        ax.set_xlabel("Diamètre Effectif D_eff (cm)",
                      color="white", fontsize=10)
        ax.set_ylabel("Facteur de conversion f",
                      color="white", fontsize=10)
        ax.set_title("AAPM Report 204 — Conversion Factor (32 cm PMMA)",
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
                   caption="f(D_eff) — Réf. AAPM Report 204, Table 1")
        export_figures.append(("Fig. — TG-204 f(D_eff) Curve", fig_f))

    return export_figures
