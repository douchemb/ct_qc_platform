import io
import math
import datetime
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def format_value(val, unit=""):
    """Vérifie si une valeur est manquante et la formate avec 3 décimales le cas échéant."""
    if val is None or val == "" or str(val).lower() == "nan" or str(val) == "N/A":
        return "N/A"
    try:
        fval = float(val)
        if math.isnan(fval):
            return "N/A"
        if fval.is_integer() and unit == "Jours":
            return f"{int(fval)} {unit}".strip()
        return f"{fval:.3f} {unit}".strip()
    except (ValueError, TypeError):
        return str(val)

def create_global_pdf_report(session_state):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='MainTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor('#1a3a5f'), alignment=TA_CENTER, spaceAfter=20)
    section_style = ParagraphStyle(name='SectionTitle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor('#1a3a5f'), spaceBefore=25, spaceAfter=15)
    center_style = ParagraphStyle(name='CenterNormal', parent=styles['Normal'], alignment=TA_CENTER)
    italic_style = ParagraphStyle(name='ItalicNote', parent=styles['Italic'], textColor=colors.HexColor('#475569'), spaceBefore=15)

    basic = session_state.get("basic_result")
    adv = session_state.get("advanced_result")
    preds = session_state.get("ai_rul_predictions", {})
    
    scanner_model = getattr(basic, 'scanner_id', "N/A") if basic else "N/A"
    operator_name = session_state.get("operator_id", "N/A")
    date_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    
    is_ge = "GE" in scanner_model.upper()
    is_canon = "CANON" in scanner_model.upper() or "TOSHIBA" in scanner_model.upper()
    is_siemens = "SIEMENS" in scanner_model.upper()

    noise = getattr(basic.noise, 'std_hu', None) if basic and getattr(basic, 'noise', None) else None
    unif = getattr(basic.uniformity, 'non_uniformity_index', None) if basic and getattr(basic, 'uniformity', None) else None
    mtf = getattr(adv.mtf, 'mtf_50_lpmm', None) * 10 if adv and getattr(adv, 'mtf', None) else None
    precision = getattr(basic, 'hu_precision', None)
    scaling = getattr(basic, 'scaling_hv', None)

    ctdi = getattr(adv.ssde_series, 'ctdi_vol_mgy', None) if adv and getattr(adv, 'ssde_series', None) else None
    ssde = getattr(adv.ssde_series, 'ssde_mean_mgy', None) if adv and getattr(adv, 'ssde_series', None) else None
    dlp = getattr(adv, 'dlp', None)
    fom = getattr(adv, 'fom', None)
    dim_ap = getattr(adv.ssde_series, 'ap_dim_cm', None) if adv and getattr(adv, 'ssde_series', None) else None
    dim_lat = getattr(adv.ssde_series, 'lat_dim_cm', None) if adv and getattr(adv, 'ssde_series', None) else None
    d_eff = getattr(adv.ssde_series, 'd_eff_cm', None) if adv and getattr(adv, 'ssde_series', None) else None

    tube_rul = preds.get("tube", "N/A")
    das_rul = preds.get("detectors", preds.get("gantry", "N/A"))
    meca_rul = preds.get("table", "N/A")
    gen_rul = preds.get("generator", "N/A")

    elements = []
    elements.append(Paragraph("RAPPORT CLINIQUE & MAINTENANCE PRÉDICTIVE", title_style))

    def get_base_table_style():
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 1), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ])

    def apply_zebra_striping(style, num_rows):
        for i in range(1, num_rows):
            if i % 2 == 0:
                style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f8fafc'))
        return style

    # Page 1: Meta & QA
    data_meta = [
        ["Paramètre", "Valeur"],
        ["Date d'Analyse", date_str],
        ["Modèle du Scanner", scanner_model],
        ["Opérateur", operator_name],
        ["Fantôme", "Automatique (Header DICOM)"]
    ]
    t_meta = Table(data_meta, colWidths=[200, 250])
    t_meta.setStyle(apply_zebra_striping(get_base_table_style(), len(data_meta)))
    elements.append(t_meta)
    
    elements.append(Paragraph("1. Résumé QA", section_style))
    data_qa = [
        ["Métrique", "Valeur"],
        ["Bruit (Noise)", format_value(noise, "HU")],
        ["Uniformité", format_value(unif, "HU")],
        ["Résolution (MTF 50%)", format_value(mtf, "lp/cm")],
        ["Précision HU", format_value(precision, "HU")],
        ["Scaling H/V", format_value(scaling, "mm")]
    ]
    if is_ge:
        data_qa.extend([
            ["Contraste Top", format_value(getattr(basic, 'contrast_top', None), "HU")],
            ["Contraste Bottom", format_value(getattr(basic, 'contrast_bottom', None), "HU")],
            ["Résolution (Moy SD Bar)", format_value(getattr(basic, 'res_sd_bar', None), "HU")]
        ])
    t_qa = Table(data_qa, colWidths=[250, 200])
    t_qa.setStyle(apply_zebra_striping(get_base_table_style(), len(data_qa)))
    elements.append(t_qa)
    
    elements.append(PageBreak())

    # Page 2: Dosi
    elements.append(Paragraph("2. Métriques Dosimétriques", section_style))
    data_dosi = [
        ["Métrique", "Valeur"],
        ["CTDIvol", format_value(ctdi, "mGy")],
        ["DLP", format_value(dlp, "mGy.cm")],
        ["SSDE", format_value(ssde, "mGy")],
        ["FOM", format_value(fom)],
        ["Dimension AP", format_value(dim_ap, "cm")],
        ["Dimension LAT", format_value(dim_lat, "cm")],
        ["Diamètre Efficace (D_eff)", format_value(d_eff, "cm")]
    ]
    t_dosi = Table(data_dosi, colWidths=[250, 200])
    t_dosi.setStyle(apply_zebra_striping(get_base_table_style(), len(data_dosi)))
    elements.append(t_dosi)
    
    elements.append(PageBreak())

    # Page 3: RUL
    elements.append(Paragraph("3. Maintenance Prédictive (RUL)", section_style))
    
    comp_das = "Détecteurs / DAS"
    comp_meca = "Mécanique (Gantry & Table)"
    if is_canon:
        comp_das = "Gantry / Roulement"
        comp_meca = "Table Patient"
    elif is_siemens:
        comp_das = "Gantry / Brushblock"
        comp_meca = "Table Patient"

    data_rul = [
        ["Composant", "Jours Restants"],
        ["Tube à Rayons X", format_value(tube_rul, "Jours")],
        [comp_das, format_value(das_rul, "Jours")],
        [comp_meca, format_value(meca_rul, "Jours")],
        ["Générateur HT", format_value(gen_rul, "Jours")]
    ]
    t_rul = Table(data_rul, colWidths=[250, 200])
    t_rul.setStyle(apply_zebra_striping(get_base_table_style(), len(data_rul)))
    elements.append(t_rul)
    
    elements.append(Paragraph("Note : Les graphiques interactifs (NPS, Profils) sont à consulter directement sur la plateforme logicielle.", italic_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer
