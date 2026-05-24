# PROJECT_CONTEXT.md
# CT QC Platform — Technical Reference Documentation
> **Role:** Central "System Prompt" for all future development interactions.
> **Version:** 2.0 — Post-Fixes (May 2026)
> **Stack:** Python 3.12 · Streamlit >=1.35 · Pydicom >=2.4 · NumPy · Pandas · scikit-learn · Matplotlib

---

# PHASE 1 — SOFTWARE ARCHITECTURE AND FILE STRUCTURE

## 1.1 Architectural Principles

- **Single DICOM import point:** pydicom is imported ONLY in `core/dicom_loader.py`
- **Strict layer separation:** core/ (I/O) → modules/ (physics) → dashboard/ (UI)
- **Zero data leakage rule:** Raw HU arrays for computation, windowed uint8 for display only
- **Multi-manufacturer routing:** GE Helios / Siemens Waterbath / Canon Aquilion pipelines

## 1.2 Complete File Tree

```
ct_qc_platform/
│
├── app.py                          # Streamlit entry point (streamlit run app.py)
│   ├── → dashboard/config_ui.py   # Page config: title, favicon, global CSS theme
│   └── → dashboard/sidebar.py     # Sidebar: upload, scanner ID, analysis trigger
│
├── config.py                       # Global config (dataclasses: DicomConfig,
│                                   #   ImageQCConfig, PredictiveConfig). SOURCE OF TRUTH.
│
├── core/                           # Layer 1 — Low-level DICOM I/O (pure)
│   ├── dicom_loader.py             # DicomLoader: ONLY file that imports pydicom
│   │                               #   → to_hu_array(): pixel_array * slope + intercept
│   │                               #   → _resolve_ge_rescale(): GE 4-step fallback chain
│   │                               #   → get_pixel_spacing_mm(), extract_metadata()
│   ├── scanner_profiles.py         # ScannerRegistry: manufacturer detection via DICOM tags
│   ├── metadata_miner.py           # MetadataMiner: aggregates CTDIvol, DLP, kVp, mAs
│   ├── dose_metadata_extractor.py  # Dosimetry metadata extraction (TG-204)
│   ├── pipeline_orchestrator.py    # CLI orchestrator (non-Streamlit pipeline)
│   ├── spatial_sort.py             # SpatialSortEngine: anatomical slice ordering
│   └── result_aggregator.py        # Multi-slice result aggregation
│
├── modules/                        # Layer 2 — Physics computation (receives np.ndarray HU)
│   ├── image_qc/
│   │   ├── basic_metrics.py        # BasicMetricsEngine: Noise, Uniformity, Contrast
│   │   │                           #   Dataclasses: TotalQAContrastResult,
│   │   │                           #   TotalQAResolutionResult, TotalQAScalingResult,
│   │   │                           #   BasicQAResult, NoiseResult, UniformityResult
│   │   ├── roi_stats.py            # PhantomROIAnalyzer, ROIDescriptor, VolumetricQCResult
│   │   │                           #   → analyze_dataset(): mean_hu, std_hu, snr per ROI
│   │   ├── advanced_metrics_engine.py  # AdvancedQAResult: NPS + MTF aggregated
│   │   ├── nps_calculator.py       # NPSCalculator: 1D radial NPS (AAPM TG-233)
│   │   ├── mtf_calculator.py       # MTFCalculator: ESF/LSF/MTF curves (AAPM TG-233)
│   │   ├── hu_linearity.py         # CT Number Accuracy per material
│   │   ├── ed_calibration.py       # Electron density calibration (IAEA TRS-430)
│   │   ├── phantom_geometry.py     # Phantom geometry (center detection, diameter)
│   │   ├── phantom_adapters.py     # Multi-phantom adapters (GE Helios, Siemens, Canon)
│   │   └── slice_classifier.py     # SliceClassifier: routing by slice position range
│   ├── dosimetry/                  # TG-204 calculations (CTDIvol, SSDE, FOM)
│   └── predictive/                 # ML interface (model loading, inference)
│
├── dashboard/                      # Layer 3 — Streamlit presentation
│   ├── orchestrator.py             # ★ MAIN ORCHESTRATOR: run_full_analysis()
│   │                               #   → Regex routing by "Image X" filename
│   │                               #   → ROI computation: Contrast, Resolution, Scaling
│   │                               #   → Injects all results into st.session_state
│   ├── sidebar.py                  # File upload + parameters + analysis trigger
│   ├── helpers.py                  # dicom_to_hu(), apply_windowing()
│   │                               #   render_ge_dicom_image() → (img_hu_raw, img_windowed)
│   │                               #   apply_dark_style(), kpi_card(), render_fig()
│   ├── roi_drawing.py              # render_roi_drawing(): Matplotlib figure with ROI overlays
│   ├── config_ui.py                # apply_page_config(): CSS, favicon, dark theme
│   ├── cached_resources.py         # @st.cache_resource singletons: DicomLoader, etc.
│   ├── tab_summary.py              # Tab 1: TotalQA KPIs + Contrast/Resolution/Scaling tables
│   ├── tab_advanced.py             # Tab 2: NPS(f) + Bar Pattern SD vs LP/mm + Scaling profile
│   ├── tab_dosimetry.py            # Tab 3: CTDIvol, SSDE, FOM, DLP
│   ├── tab_predictive.py           # Tab 4: RUL per component, predictive charts
│   ├── tab_hardware_health.py      # Tab 5: Hardware health dashboard
│   └── siemens_waterbath.py        # Complete Siemens/Canon pipeline
│
├── predictive_maintenance/         # ML models + training data
│   ├── historical_qa_data.csv      # Synthetic Siemens data (1002 QA sessions)
│   ├── historical_qa_data_ge.csv   # Synthetic GE data (1002 QA sessions)
│   ├── rf_tube.pkl / rf_ge_tube.pkl
│   ├── rf_gantry.pkl / rf_ge_detectors.pkl
│   ├── rf_generator.pkl / rf_ge_generator.pkl
│   ├── rf_table.pkl / rf_ge_table.pkl
│   ├── rf_canon_tube/gantry/generator/table.pkl
│   ├── inference.py                # FailurePredictor: predict_rul() multi-manufacturer
│   ├── 1_generate_synthetic_data*.py
│   └── 2_train_predictive_models*.py
│
├── phantom_profiles.yaml           # Phantom profiles: GE Helios, Siemens, Canon
├── scanner_profiles.yaml           # Scanner profiles: TG-66 tolerances per model
├── phantom_config.json             # GE Helios ROI coordinates
└── requirements_streamlit.txt      # Runtime dependencies
```

## 1.3 Key Dataclasses (basic_metrics.py)

| Dataclass | Fields | Created by |
|-----------|--------|------------|
| `TotalQAContrastResult` | mean_A, mean_B, mean_C, mean_D, contrast_top, contrast_bottom, passed | orchestrator.py |
| `TotalQAResolutionResult` | bar_labels[], bar_sd_values[], bar_mean_values[], passed | orchestrator.py |
| `TotalQAScalingResult` | h_diameter_mm, v_diameter_mm, nominal_mm, h/v_error_mm, passed | orchestrator.py |
| `BasicQAResult` | noise, uniformity, ct_number_accuracy, totalqa_contrast, totalqa_resolution, totalqa_scaling | basic_engine.compute() |

---

# PHASE 2 — END-TO-END DATA FLOW AND THE GOLDEN RULE

## 2.1 GE Helios Pipeline (run_full_analysis)

```
UPLOAD (Streamlit)
    | sidebar.py: st.file_uploader() -> list[UploadedFile]
    v
FORCE-READ ALL DICOM  [orchestrator.py -> run_full_analysis()]
    | pydicom.dcmread(BytesIO(f.read()), force=True)
    | Access ds.pixel_array to trigger JPEG Lossless decompression
    v
MANUFACTURER DETECTION  [orchestrator._detect_manufacturer_from_datasets()]
    | Tag (0008,0070) Manufacturer -> "GE" / "SIEMENS" / "CANON"
    | If SIEMENS or CANON -> run_siemens_analysis() (separate pipeline)
    v
TOTALQA REGEX ROUTING  [orchestrator.py]
    | Regex: r"Image\s*(\d+)\.dcm"  (case-insensitive)
    | Image 36  -> UNIFORMITY + SCALING  (pure water zone)
    | Image 56  -> CONTRAST             (plastic block, 4 ROIs)
    | Image 71  -> RESOLUTION           (angled bar patterns, 5 ROIs)
    v
HU CONVERSION  [core/dicom_loader.py -> DicomLoader.to_hu_array()]
    | slope, intercept = _resolve_ge_rescale(ds)
    |   Priority 1: standard tags (0028,1053) / (0028,1052)
    |   Priority 2: GE private tag (0009,100D) -> slope only, intercept=-1024
    |   Priority 3: RealWorldValueMappingSequence (0040,9096)
    |   Priority 4: default slope=1.0, intercept=-1024.0 (WARNING logged)
    | img_hu_raw = slope * pixel_array.astype(float32) + intercept
    v
PHYSICS COMPUTATIONS  [orchestrator.py]  <- ALWAYS on img_hu_raw
    | Contrast:    np.mean(img_hu_raw[roi]) per zone A/B/C/D
    | Resolution:  np.std(img_hu_raw[roi], ddof=1) per bar_1..bar_5
    | Scaling:     morphological bounding box (HU > -300 threshold)
    | Uniformity:  std + NUI on peripheral water ROIs
    v
WINDOWING (DISPLAY ONLY)  [dashboard/helpers.py -> apply_windowing()]
    | img_windowed = clip(img_hu_raw, WL-WW/2, WL+WW/2)
    | img_windowed = normalize to 0-255 -> cast to uint8
    | img_windowed -> ONLY for st.image() — NEVER for statistics
    v
ML INFERENCE  [predictive_maintenance/inference.py -> FailurePredictor]
    | Features: [noise_sd, uniformity_nui, contrast_top, bar_sd_mean,
    |            h_diameter_mm, v_diameter_mm, ctdi_vol, ...]
    | Models:   rf_ge_tube, rf_ge_detectors, rf_ge_generator, rf_ge_table
    | Output:   RUL (days) + urgency level per hardware component
    v
SESSION STATE STORAGE  [orchestrator.py -> st.session_state]
    | basic_result, advanced_result, dosimetry_metrics
    | simulation_figs, water/contrast/resolution_roi_descriptors
    | scaling_profile_fig, acquisition_params
    v
TAB RENDERING  [dashboard/tab_*.py]
    Tab 1 (tab_summary.py)         : KPI cards + detail tables
    Tab 2 (tab_advanced.py)        : NPS(f) + Bar Pattern SD vs LP/mm
    Tab 3 (tab_dosimetry.py)       : CTDIvol, SSDE, DLP, FOM
    Tab 4 (tab_predictive.py)      : RUL predictions + maintenance charts
    Tab 5 (tab_hardware_health.py) : Hardware health dashboard
```

## 2.2 THE GOLDEN RULE — Strict Separation: Computation vs Display

> **This rule must NEVER be violated. Any breach produces physically wrong values.**

```python
# ═══════════════════════════════════════════════════════════════
# SOURCE: core/dicom_loader.py -> DicomLoader.to_hu_array()
# ═══════════════════════════════════════════════════════════════
img_hu_raw = slope * pixel_array.astype(np.float32) + intercept
# dtype : float32
# range : -1024 HU (air) to +3000 HU (metal)
# USE   : EXCLUSIVELY for np.mean(), np.std(), ROI stats, contrast
# NEVER : pass to st.image() or any display function

# ═══════════════════════════════════════════════════════════════
# SOURCE: dashboard/helpers.py -> apply_windowing()
# ═══════════════════════════════════════════════════════════════
lower = window_level - window_width / 2
upper = window_level + window_width / 2
clipped = np.clip(img_hu_raw, lower, upper)
img_windowed = ((clipped - lower) / (upper - lower) * 255).astype(np.uint8)
# dtype : uint8
# range : 0-255
# USE   : EXCLUSIVELY for st.image() visual rendering
# NEVER : pass to np.mean(), np.std(), or any computation

# ═══════════════════════════════════════════════════════════════
# USAGE PATTERN (dashboard/helpers.py -> render_ge_dicom_image)
# ═══════════════════════════════════════════════════════════════
img_hu_raw, img_windowed = render_ge_dicom_image(ds, WL=0, WW=400)
# img_hu_raw   -> physics calculations
# img_windowed -> already rendered inside function via st.image()

# ═══════════════════════════════════════════════════════════════
# ANTI-LEAKAGE GUARD (orchestrator.py — contrast computation)
# ═══════════════════════════════════════════════════════════════
_hu_range = float(contrast_hu.max()) - float(contrast_hu.min())
if _hu_range < 100.0:
    raise ValueError(
        f"Data leakage detected: range={_hu_range:.1f} HU. "
        f"Valid HU matrix must have range >> 100 HU. "
        f"Check loader.to_hu_array() was not replaced by windowed image."
    )
```

**Recommended windowing presets per slice type:**

| Slice | WL (HU) | WW (HU) | Rationale |
|-------|---------|---------|-----------|
| Uniformity (water) | 0 | 400 | Centered on water, narrow window |
| Contrast (plastic) | 0 | 2500 | Wide to capture all materials |
| Resolution (bars) | 0 | 2000 | Wide for bar pattern contrast |

## 2.3 GE Rescale Fallback Chain (_resolve_ge_rescale)

Some GE Discovery RT units omit standard DICOM rescale tags. The loader handles this via a 4-step priority chain:

```python
Priority 1: RescaleSlope (0028,1053) + RescaleIntercept (0028,1052)  # Standard — fast path
Priority 2: GE private tag (0009,100D)  # Raw calibration slope; intercept defaults to -1024
Priority 3: RealWorldValueMappingSequence (0040,9096)  # GE firmware >= 27.x
Priority 4: slope=1.0, intercept=-1024.0  # Safe default — WARNING always logged
```

## 2.4 Siemens / Canon Waterbath Pipeline

Activated automatically when `Manufacturer` tag contains "SIEMENS", "CANON", or "TOSHIBA".

```
valid_datasets -> _detect_manufacturer_from_datasets()
    -> "SIEMENS" or "CANON"
    -> dashboard/siemens_waterbath.py: run_siemens_analysis()
        -> 5 ROIs (center + 4 edges: upper/right/lower/left)
        -> center_mean (target: 0 HU)
        -> center_sd   (target: <= 5 HU)
        -> edge_diffs  (tolerance: |edge - center| <= 4 HU)
        -> NPS computation per slice
        -> H/V profiles for uniformity visualization
```

---

# PHASE 3 — THE THREE MODULES (QA, DOSIMETRY, AI)

## 3.1 Module 1 — Quality Control (TotalQA Benchmarking)

### 3.1.1 Target Slices (GE Helios)

| Image # | Module | Content | Key metric |
|---------|--------|---------|------------|
| 36 | Uniformity + Noise + Scaling | Pure water zone | SD_HU, NUI, diameter |
| 56 | Contrast | Plastic block (4 zones A/B/C/D) | Mean(A)-Mean(B) |
| 71 | Spatial Resolution | Angled bar patterns (5 groups) | np.std per bar group |

### 3.1.2 Contrast ROI Geometry (Slice 56) — orchestrator.py lines 100-118

```python
# CORRECTED OFFSETS (May 2026 fix — was 12.0mm/50.0mm causing ~31 HU instead of ~12-15 HU)
_CONTRAST_FAR_OFFSET_MM  = 78.0   # ROIs B/D (pure water)   — cy +/- 78 mm
_CONTRAST_NEAR_OFFSET_MM = 24.5   # ROIs A/C (plastic block) — cy +/- 24.5 mm
_CONTRAST_ROI_HEIGHT_MM  = 6.0    # ROI height (thin band)
_CONTRAST_ROI_WIDTH_MM   = 40.0   # ROI width  (inside plastic, was 55.0)

# TotalQA vertical layout (all aligned on phantom center cx):
#   B (top_water)      : cy - 78 mm  -> ~0 HU   (pure water)
#   A (top_plastic)    : cy - 24.5mm -> ~12-15 HU (plastic center)
#   C (bottom_plastic) : cy + 24.5mm -> ~12-15 HU (plastic center)
#   D (bottom_water)   : cy + 78 mm  -> ~0 HU   (pure water)

# Contrast formula:
#   Contrast_Top    = Mean(A) - Mean(B)   # ~12-15 HU expected
#   Contrast_Bottom = Mean(C) - Mean(D)   # ~12-15 HU expected
```

### 3.1.3 Spatial Resolution ROI Geometry (Slice 71) — orchestrator.py lines 109-122

```python
# Bar insert center offset from phantom center
_BAR_CENTER_OFFSET_ROW_MM = -15.4   # 24 px upward
_BAR_CENTER_OFFSET_COL_MM =  6.8    # 6 px rightward

# Diagonal offsets from bar-insert center (/) — bar_1 to bar_5
_BAR_DIAG_OFFSETS_MM = [20.0, 3.0, -10.0, -23.0, -35.0]

# ROI sizes proportional to spatial frequency
_BAR_ROI_SIZES_MM = [10.0, 8.0, 5.0, 2.0, 1.0]
```

**TotalQA bar pattern correspondence table:**

| ROI | Size (mm) | Frequency (LP/mm) | Formula | ROI size |
|-----|-----------|-------------------|---------|----------|
| bar_1 | 1.6 | 0.313 | 1/(2*1.6) | 10x10 mm |
| bar_2 | 1.3 | 0.385 | 1/(2*1.3) |  8x8 mm  |
| bar_3 | 1.0 | 0.500 | 1/(2*1.0) |  5x5 mm  |
| bar_4 | 0.8 | 0.625 | 1/(2*0.8) |  2x2 mm  |
| bar_5 | 0.6 | 0.833 | 1/(2*0.6) |  1x1 mm  |

**Resolution metric:** `SD_HU = np.std(img_hu_raw[roi], ddof=1)`
High SD on fine bars indicates good spatial resolution (strong bar/gap contrast).

> **CRITICAL:** bar_sd_values[] is stored in bar_1->bar_5 order (coarse->fine).
> DO NOT sort sd_values independently — this decouples SDs from their physical positions.
> The list `sorted(resolution_rois.keys())` already produces correct bar_1..bar_5 order.

### 3.1.4 All QA Metrics and Tolerances

| Metric | Computation | TG-66 Tolerance | Dataclass |
|--------|-------------|-----------------|-----------|
| Noise (SD) | np.std(center_roi) | <= 5.0 HU | NoiseResult |
| Uniformity (NUI) | max(|mean_edge - mean_center|) | <= 5.0 HU | UniformityResult |
| HU Precision | |mean_water - 0 HU| | <= 4.0 HU | CTNumberAccuracyResult |
| Contrast Top | Mean(A) - Mean(B) | > 0 HU | TotalQAContrastResult |
| Contrast Bottom | Mean(C) - Mean(D) | > 0 HU | TotalQAContrastResult |
| Resolution (mean SD) | mean(bar_sd_values) | configurable | TotalQAResolutionResult |
| Scaling H | Morphological bounding box | +/- 2.0 mm / 215 mm nominal | TotalQAScalingResult |
| Scaling V | Morphological bounding box | +/- 2.0 mm / 215 mm nominal | TotalQAScalingResult |

## 3.2 Module 2 — Personalized Dosimetry (AAPM TG-204)

Implemented in `orchestrator.py -> _compute_tg204_dosimetry()`.

### 3.2.1 DICOM Tag Extraction

```python
CTDIvol <- tag (0018,9345)  # Dose per rotation (mGy)
DLP     <- tag (0018,9346)  # Dose * Length (mGy.cm)
         # Fallback: CTDIvol * SliceThickness(cm) when tag absent
```

### 3.2.2 TG-204 Calculation Chain

```python
# 1. Patient morphometry (from phantom geometry measurement)
AP_cm  = v_diameter_mm / 10.0          # Antero-posterior diameter
LAT_cm = h_diameter_mm / 10.0          # Lateral diameter

# 2. Effective diameter (TG-204 Eq. 1)
D_eff = sqrt(AP_cm * LAT_cm)

# 3. Size-Specific Dose Estimate conversion factor (32 cm PMMA body phantom ref.)
f = 3.704 * exp(-0.0367 * D_eff)       # TG-204 Table A.1 regression

# 4. SSDE (TG-204 Eq. 2)
SSDE = CTDIvol * f                      # Patient-specific dose (mGy)

# 5. Figure of Merit (image quality per unit dose)
FOM = 1 / (noise_SD^2 * CTDIvol)       # Higher = better quality/dose ratio
```

### 3.2.3 Expected Ranges

| Parameter | Typical Range | Unit |
|-----------|--------------|------|
| CTDIvol   | 5 - 30       | mGy  |
| D_eff     | 15 - 35      | cm   |
| f factor  | 0.8 - 2.0    | —    |
| SSDE      | CTDIvol * f  | mGy  |
| FOM       | 0.001 - 0.05 | —    |

## 3.3 Module 3 — Artificial Intelligence (Predictive Maintenance)

### 3.3.1 ML Architecture

| Manufacturer | Component | Model file | Key features |
|-------------|-----------|------------|--------------|
| GE | X-ray tube | rf_ge_tube.pkl | noise_sd, nps_peak_freq, ctdi_vol |
| GE | Detectors | rf_ge_detectors.pkl | bar_sd_mean, uniformity_nui |
| GE | HV Generator | rf_ge_generator.pkl | hu_precision_delta, kvp |
| GE | Patient table | rf_ge_table.pkl | scaling_h_mm, scaling_v_mm |
| Siemens | X-ray tube | rf_tube.pkl | noise_sd, nps_peak_freq |
| Siemens | Gantry | rf_gantry.pkl | uniformity_nui, edge_diffs |
| Siemens | Generator | rf_generator.pkl | hu_precision_delta |
| Siemens | Table | rf_table.pkl | scaling_error_mm |
| Canon | All 4 components | rf_canon_*.pkl | (same as Siemens) |

### 3.3.2 Strict Physical Correlation (Anti-Data-Leakage in ML)

```
Noise SD increase / MTF degradation  ->  X-ray tube  (filament wear, focal spot broadening)
Uniformity degradation (NUI)         ->  Gantry      (bearing misalignment, wobble)
Scaling drift / slice thickness err  ->  Patient table (mechanical drift)
HU precision degradation             ->  HV Generator  (kVp instability)
NPS peak frequency shift             ->  X-ray tube  (filtration change, aging)
```

> **ML Anti-Leakage principle:** Each Random Forest model is trained ONLY on features
> physically correlated with that specific hardware component. Cross-contamination of
> features between component models is forbidden by design.

### 3.3.3 Digital Twin Architecture

```
Historical QA Data (synthetic, physics-realistic):
  - historical_qa_data.csv      : 1002 Siemens QA sessions
  - historical_qa_data_ge.csv   : 1002 GE QA sessions

Generation scripts:
  1_generate_synthetic_data.py    : Siemens degradation simulation
  1_generate_synthetic_data_ge.py : GE degradation simulation
  -> Gaussian noise overlaid on component-specific degradation curves
  -> RUL ground truth computed from simulated lifetime trajectory

Training:
  2_train_predictive_models.py    : Trains rf_tube/gantry/generator/table.pkl
  2_train_predictive_models_ge.py : Trains rf_ge_*.pkl
```

### 3.3.4 FailurePredictor Output Schema

```python
FailurePredictor.predict_rul(metrics_dict) -> {
    "tube": {
        "rul_days": 180,
        "urgency": "monitor",   # stable|monitor|warning|critical|breached
        "confidence": 0.87
    },
    "detectors": {"rul_days": 450, "urgency": "stable"},
    "generator": {"rul_days": 60,  "urgency": "warning"},
    "table":     {"rul_days": 320, "urgency": "stable"},
}
```

**Urgency color mapping (dashboard/helpers.py):**

| Level | Color | Action |
|-------|-------|--------|
| stable | #3fb950 (green) | Routine monitoring |
| monitor | #c9a227 (amber) | Increased monitoring frequency |
| warning | #d29922 (orange) | Schedule maintenance |
| critical | #f85149 (red) | Urgent maintenance required |
| breached | #da3633 (dark red) | Immediate shutdown recommended |

---

# PHASE 4 — KNOWN BUGS, EXTENSIONS, AND REFERENCE COMMANDS

## 4.1 Bug Registry — Corrected Issues

| # | File | Bug description | Root cause | Fix applied | Status |
|---|------|----------------|------------|-------------|--------|
| 1 | orchestrator.py | Contrast ~31 HU instead of ~12-15 HU | `_CONTRAST_NEAR_OFFSET_MM=12.0mm` placed ROIs A/C at water/plastic interface instead of plastic center | Changed to 24.5mm; `_CONTRAST_FAR_OFFSET_MM: 50->78mm`; `_CONTRAST_ROI_WIDTH_MM: 55->40mm` | FIXED |
| 2 | tab_advanced.py | SD values shuffled in resolution chart | `sorted(sd_values, reverse=True)` decoupled SDs from their physical bar positions | Removed the sort — `sorted(keys())` already preserves bar_1..bar_5 physical order | FIXED |
| 3 | helpers.py | `render_ge_dicom_image()` returned only `img_windowed` | Callers could accidentally use uint8 image for physics calculations | Returns tuple `(img_hu_raw, img_windowed)`; docstring forbids using windowed for computation | FIXED |
| 4 | tab_summary.py | Resolution DataFrame showed `Mean HU` column instead of `SD HU` | Wrong variable name in dict | Removed `Mean HU` column; added `Taille (mm)` and `Frequence (LP/mm)` columns | FIXED |
| 5 | tab_advanced.py | X-axis of resolution chart in "mm" instead of "LP/mm" | Used string labels like '1.6mm' instead of computed frequencies | Conversion `lpmm = 1/(2*size_mm)`; double-line xtick labels show both LP/mm and mm | FIXED |
| 6 | orchestrator.py | Silent wrong contrast with no error message | No range validation on HU matrix | Added guard: raises `ValueError` if `contrast_hu.max()-min() < 100 HU` | FIXED |

## 4.2 Critical Rules for Future Development

### Rule 1 — The HU Separation Contract

```python
# Before ANY ROI computation, verify the matrix type:
assert img_hu_raw.dtype in (np.float32, np.float64), "Must be float HU"
assert img_hu_raw.min() < -100, "Air should be ~-1000 HU"
assert img_hu_raw.max() > 100, "Bone/plastic should be > 100 HU"

# Anti-leakage guard (already in orchestrator.py):
if (img_hu_raw.max() - img_hu_raw.min()) < 100.0:
    raise ValueError("Likely windowed image passed to computation")
```

### Rule 2 — Bar Pattern SD Order

```python
# ALWAYS iterate in this exact order to keep SD synchronized with LP/mm:
BAR_SIZES_MM = [1.6, 1.3, 1.0, 0.8, 0.6]   # coarse -> fine
lpmm_values  = [1/(2*s) for s in BAR_SIZES_MM]   # [0.313, 0.385, 0.500, 0.625, 0.833]

# bar_sd_values[i] MUST correspond to BAR_SIZES_MM[i]
# NEVER sort sd_values independently from lpmm_values
```

### Rule 3 — Manufacturer Routing

```python
# Detection order in _detect_manufacturer_from_datasets():
if "SIEMENS" in mfr: return "SIEMENS"
elif "GE" in mfr or "GENERAL" in mfr: return "GE"
elif "CANON" in mfr or "TOSHIBA" in mfr: return "CANON"
else: return "GE"  # safe default — WARNING logged

# Pipeline routing:
if manufacturer in ("SIEMENS", "CANON"):
    run_siemens_analysis()   # siemens_waterbath.py
    return
# GE pipeline continues in run_full_analysis()
```

### Rule 4 — Session State Keys Contract

```python
# Keys written by orchestrator.py (DO NOT rename without updating all tabs):
st.session_state["basic_result"]              # BasicQAResult
st.session_state["advanced_result"]           # AdvancedQAResult
st.session_state["dosimetry_metrics"]         # dict (TG-204)
st.session_state["simulation_figs"]           # dict[str, Figure]
st.session_state["water_roi_descriptors"]     # dict[str, ROIDescriptor]
st.session_state["contrast_roi_descriptors"]  # dict[str, ROIDescriptor]
st.session_state["resolution_roi_descriptors"]# dict[str, ROIDescriptor]
st.session_state["scaling_profile_fig"]       # matplotlib.Figure
st.session_state["acquisition_params"]        # dict {kvp, mas, kernel, slice_thickness}
st.session_state["manufacturer"]              # "GE" | "SIEMENS" | "CANON"
st.session_state["siemens_result"]            # SiemensWaterbathResult (Siemens/Canon only)
st.session_state["siemens_kpi_metrics"]       # dict (Siemens/Canon only)
```

## 4.3 Priority Roadmap

| Priority | Feature | Target file | Notes |
|----------|---------|-------------|-------|
| P1 | Validate corrected contrast ROI offsets on real GE DICOM | orchestrator.py | Verify `_CONTRAST_NEAR_OFFSET_MM=24.5mm` gives ~12-15 HU on actual phantom |
| P1 | PDF export of full QA report (figures + tables) | helpers.py `fig_to_pdf_bytes()` | Already implemented, needs UI trigger |
| P2 | Interactive ROI offset sliders in sidebar | sidebar.py | Allow physicist to fine-tune offsets per scanner |
| P2 | DICOM SR output (Structured Report) | core/ new module | Export results as DICOM SR for PACS integration |
| P2 | Historical QA trend charts | tab_predictive.py | Plot metrics over time using stored session data |
| P3 | PostgreSQL/SQLite backend for QA history | core/db/ new module | Replace in-memory session state with persistent storage |
| P3 | Multi-user authentication | app.py + new auth module | For SaaS deployment |
| P3 | Real-time monitoring WebSocket | app.py | Continuous QA during acquisition |

## 4.4 Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| Only Image 36/56/71 processed (GE) | Other slices ignored | Add more regex targets if needed |
| Bar pattern ROI sizes use fixed mm offsets | May drift if pixel_spacing != 0.977mm | All offsets converted dynamically via pixel_spacing |
| ML models trained on synthetic data only | RUL predictions are indicative, not clinical | Retrain with real historical QA data when available |
| Single-frame extraction (multi-frame DICOM) | Only first frame used | Logged as WARNING; acceptable for single-slice QA phantoms |
| Canon modeled as Siemens-equivalent | Minor differences possible | Canon-specific models in rf_canon_*.pkl |

## 4.5 Reference Commands

```bash
# Launch the application
python -m streamlit run app.py

# Install all dependencies
python -m pip install -r requirements_streamlit.txt

# Regenerate GE synthetic training data
python predictive_maintenance/1_generate_synthetic_data_ge.py

# Regenerate Siemens synthetic training data
python predictive_maintenance/1_generate_synthetic_data.py

# Retrain GE predictive models
python predictive_maintenance/2_train_predictive_models_ge.py

# Retrain Siemens predictive models
python predictive_maintenance/2_train_predictive_models.py

# Run unit tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html
```

## 4.6 DICOM Tag Quick Reference

| Tag | Name | Usage in this project |
|-----|------|-----------------------|
| (0008,0070) | Manufacturer | Pipeline routing (GE/Siemens/Canon) |
| (0028,1053) | RescaleSlope | HU conversion (Priority 1) |
| (0028,1052) | RescaleIntercept | HU conversion (Priority 1) |
| (0028,0030) | PixelSpacing | mm/px conversion for all ROI geometry |
| (0018,9345) | CTDIvol | TG-204 dosimetry |
| (0018,9346) | DLP | TG-204 dosimetry |
| (0018,0050) | SliceThickness | DLP fallback calculation |
| (0018,0060) | KVP | Acquisition parameters display |
| (0018,1151) | XRayTubeCurrent | mAs computation |
| (0008,0060) | Modality | CT validation (must be "CT") |
| (0020,0013) | InstanceNumber | Slice ordering |
| (0009,100D) | GE private slope | GE rescale fallback (Priority 2) |
| (0040,9096) | RealWorldValueMapping | GE firmware >= 27.x (Priority 3) |

---

*Document generated from source code analysis — ct_qc_platform v2.0 — May 2026*
*Maintainer: update this file whenever new bugs are fixed or new features are added.*
