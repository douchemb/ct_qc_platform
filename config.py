# -*- coding: utf-8 -*-
"""
config.py — Single Source of Truth for the CT-QC Platform.

All numeric constants, file paths, and protocol thresholds are defined here
as frozen dataclasses. Every module in the project imports values exclusively
from the global CONFIG object instantiated at the bottom of this file.

Importing CONFIG has zero side effects — no file I/O, no network calls, no print.

Standards References:
    - AAPM TG-66  (2003): Quality Assurance for CT Simulators
    - AAPM TG-233 (2019): NPS/MTF Measurement Protocol
    - AAPM Report 204 (2011): Size-Specific Dose Estimates (SSDE)
    - AAPM TG-220 (2014): Water Equivalent Diameter for SSDE
    - Schneider et al. Phys. Med. Biol. 41(1) 1996: HU→RED Conversion
    - IAEA TRS-430 (2004): Electron Density Calibration
    - IEC 61223-3-5: Acceptance and constancy tests — CT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# Path Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PathConfig:
    """All filesystem paths, resolved relative to project root."""
    project_root: Path
    data_dir: Path
    outputs_dir: Path
    plots_dir: Path
    reports_dir: Path
    logs_dir: Path
    archive_file: Path      # outputs/reports/qc_archive.json


# ═══════════════════════════════════════════════════════════════════
# DICOM Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DicomConfig:
    """DICOM protocol constants and SOP Class UIDs."""
    # DICOM PS3.4, Table B.5-1 — CT Image Storage
    ct_sop_class_uid: str
    # DICOM PS3.4 — CT Localizer (Enhanced CT Image Storage)
    ct_localizer_sop_class_uid: str
    # Expected modality tag value
    expected_modality: str
    # Pixel value sanity bounds (HU range)
    min_pixel_value: float
    max_pixel_value: float


# ═══════════════════════════════════════════════════════════════════
# Image QC Configuration (AAPM TG-66 / IEC 61223-3-5)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImageQCConfig:
    """Image quality control thresholds and analysis parameters."""

    # --- TG-66 Section 5.1 — Noise ---
    noise_tolerance_hu: float       # Maximum acceptable noise std (HU)
    noise_warning_hu: float         # Warning threshold before failure

    # --- TG-66 Section 5.2 — HU Linearity ---
    hu_linearity_tolerance: float   # Maximum HU deviation from nominal

    # --- Nominal HU reference values — CatPhan phantom inserts ---
    # Source: CatPhan 504/600 manual; AAPM TG-66 Table 1
    hu_water_nominal: float
    hu_air_nominal: float
    hu_acrylic_nominal: float
    hu_ldpe_nominal: float
    hu_polystyrene_nominal: float
    hu_bone_nominal: float
    hu_pmp_nominal: float
    hu_delrin_nominal: float

    # --- NPS computation parameters — AAPM TG-233 ---
    nps_patch_size_px: int
    nps_n_patches_min: int
    nps_detrend_polynomial_order: int

    # --- MTF computation parameters — AAPM TG-66 Section 5.3 ---
    mtf_oversampling_factor: int

    # --- ROI minimum for statistically valid variance — AAPM TG-66 Section 5.1 ---
    roi_min_area_px: int

    # --- Volumetric slice selection defaults — Step 5 ---
    default_start_slice: int
    default_end_slice: int
    min_slices_for_volumetric: int


# ═══════════════════════════════════════════════════════════════════
# Dosimetry Configuration (AAPM TG-220 / Report 204)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DosimetryConfig:
    """SSDE calculation parameters and conversion factor table.

    Source: AAPM Report 204 (2011), Table A.1
    Format: tuple of (d_w_cm, f_factor) pairs, sorted ascending by d_w
    """
    ssde_conversion_table: tuple
    dw_min_cm: float            # Minimum valid water-equivalent diameter
    dw_max_cm: float            # Maximum valid water-equivalent diameter
    lpv_air_value: float        # Localizer pixel value for air
    lpv_scale_factor: float     # Localizer calibration scale factor


# ═══════════════════════════════════════════════════════════════════
# ED Calibration Configuration (Schneider 1996 / IAEA TRS-430)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EDCalibrationConfig:
    """
    HU → Electron Density calibration for radiotherapy TPS integration.

    Source: Schneider et al. Phys. Med. Biol. 41(1) 1996, Table 1;
            IAEA TRS-430 (2004) Table 4.1.
    Each phantom_materials entry: (material_name, nominal_hu, relative_electron_density)
    RED = 1.0 for water by definition. RED = ρ_e / ρ_e_water.
    """
    phantom_materials: tuple    # 13 entries — see instantiation below
    fit_method: str             # "piecewise_linear"
    n_fit_segments: int         # 2
    max_red_deviation: float    # 0.02 (2% RED — IAEA TRS-430 Section 4.2.3)
    tps_format: str             # "generic_csv"


# ═══════════════════════════════════════════════════════════════════
# Predictive Maintenance Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PredictiveConfig:
    """Parameters for trend analysis and failure prediction."""
    min_history_points: int         # Minimum QC sessions for regression
    forecast_horizon_days: int      # How far ahead to predict
    r2_minimum_acceptable: float    # Minimum R² for model validity
    # Metric key names used in JSON archive — must match archive schema
    tracked_metric_noise_std: str
    tracked_metric_nps_peak: str
    tracked_metric_mtf50: str
    # Hardware failure metric keys — supervisor requirement
    tracked_metric_ed_soft_slope: str
    tracked_metric_ed_bone_slope: str


# ═══════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LoggingConfig:
    """Rotating file + console logging parameters."""
    log_level: str
    log_format: str
    max_bytes: int          # Max log file size before rotation
    backup_count: int       # Number of rotated log files to keep
    console_output: bool


# ═══════════════════════════════════════════════════════════════════
# GE Helios Phantom Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HeliosPhantomConfig:
    """
    GE Helios QA Phantom physical reference values.
    Source: GE Helios QA Phantom User Manual P/N 2165993-100 Rev 1;
            IAEA TRS-430 Table 4.1; Schneider et al. Phys Med Biol 41(1) 1996.

    phantom_materials: tuple of (material_name, nominal_hu, RED, radius_mm, angle_deg)
      radius_mm = 0 means center insert (water)
      angle_deg = 0 = top (12 o'clock), clockwise
    """
    phantom_materials: tuple    # (name, nominal_hu, RED, radius_mm, angle_deg)
    phantom_outer_radius_mm: float      # 100.0
    insert_radius_mm: float             # 58.0
    insert_diameter_mm: float           # 28.0
    roi_diameter_mm: float              # 20.0
    # HU tolerance for each insert — AAPM TG-66 Section 5.2
    hu_tolerance: float                 # 4.0 HU
    # RED tolerance — IAEA TRS-430 Section 4.2.3
    red_tolerance: float                # 0.02


# ═══════════════════════════════════════════════════════════════════
# GE Scanner Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GEScannerConfig:
    """
    GE Discovery RT scanner-specific DICOM parsing parameters.
    Source: GE DICOM Conformance Statement, Discovery RT, Rev 3.
    """
    # GE private tag for rescale slope — (0009,100D)
    ge_private_rescale_slope_tag: tuple     # (0x0009, 0x100D)
    # GE default CT intercept when only private slope is available
    ge_default_intercept: float             # -1024.0
    # GE localizer ImageType string — alternative to SOP Class UID detection
    ge_localizer_image_type_string: str     # "LOCALIZER"
    # GE known firmware versions with InstanceNumber=0 bug
    ge_instance_number_bug_firmware: tuple  # ("27.0", "28.0")
    # Fallback sort key when InstanceNumber is identical across all slices
    ge_sort_fallback_key: str               # "SliceLocation"


# ═══════════════════════════════════════════════════════════════════
# Top-Level Application Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AppConfig:
    """Root configuration container — the single CONFIG object."""
    paths: PathConfig
    dicom: DicomConfig
    image_qc: ImageQCConfig
    dosimetry: DosimetryConfig
    ed_calibration: EDCalibrationConfig
    predictive: PredictiveConfig
    logging: LoggingConfig
    helios_phantom: HeliosPhantomConfig
    ge_scanner: GEScannerConfig


# ═══════════════════════════════════════════════════════════════════
# CONFIG Instantiation — Global Singleton
# ═══════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).parent.resolve()

CONFIG = AppConfig(
    # ── Paths ──
    paths=PathConfig(
        project_root=_PROJECT_ROOT,
        data_dir=_PROJECT_ROOT / "data",
        outputs_dir=_PROJECT_ROOT / "outputs",
        plots_dir=_PROJECT_ROOT / "outputs" / "plots",
        reports_dir=_PROJECT_ROOT / "outputs" / "reports",
        logs_dir=_PROJECT_ROOT / "outputs" / "logs",
        archive_file=_PROJECT_ROOT / "outputs" / "reports" / "qc_archive.json",
    ),

    # ── DICOM ──
    dicom=DicomConfig(
        ct_sop_class_uid="1.2.840.10008.5.1.4.1.1.2",             # DICOM PS3.4 B.5-1
        ct_localizer_sop_class_uid="1.2.840.10008.5.1.4.1.1.2.1", # DICOM PS3.4
        expected_modality="CT",
        min_pixel_value=-10000.0,   # Extended HU range for safety
        max_pixel_value=10000.0,
    ),

    # ── Image QC ──
    image_qc=ImageQCConfig(
        # AAPM TG-66 Section 5.1 — water phantom noise tolerance
        noise_tolerance_hu=5.0,
        # AAPM TG-66 Section 5.1 — warning level (80% of tolerance)
        noise_warning_hu=4.0,
        # AAPM TG-66 Section 5.2 — HU linearity tolerance
        hu_linearity_tolerance=4.0,

        # CatPhan 504/600 nominal HU values — AAPM TG-66 Table 1
        hu_water_nominal=0.0,           # Water: 0 HU by definition
        hu_air_nominal=-1000.0,         # Air: -1000 HU by definition
        hu_acrylic_nominal=120.0,       # Acrylic (PMMA): ~120 HU
        hu_ldpe_nominal=-100.0,         # Low-density polyethylene
        hu_polystyrene_nominal=-35.0,   # Polystyrene
        hu_bone_nominal=955.0,          # Bone-equivalent (SB3/Teflon)
        hu_pmp_nominal=-200.0,          # Polymethylpentene (PMP)
        hu_delrin_nominal=340.0,        # Delrin (polyoxymethylene)

        # AAPM TG-233 — NPS computation parameters
        nps_patch_size_px=64,
        nps_n_patches_min=8,
        nps_detrend_polynomial_order=2,

        # AAPM TG-66 Section 5.3 — MTF oversampling
        mtf_oversampling_factor=4,

        # Minimum ROI area for statistically valid variance — AAPM TG-66 Section 5.1
        roi_min_area_px=400,

        # Volumetric slice selection defaults — Step 5
        default_start_slice=1,
        default_end_slice=999,
        min_slices_for_volumetric=3,
    ),

    # ── Dosimetry ──
    dosimetry=DosimetryConfig(
        # AAPM Report 204 (2011), Table A.1 — f-factors for body/trunk at 120 kVp
        # Format: (effective_diameter_cm, f_size_factor) sorted ascending by d_w
        ssde_conversion_table=(
            (10.0, 1.528),  # AAPM Report 204, Table A — 120 kVp body
            (12.0, 1.456),  # AAPM Report 204, Table A
            (14.0, 1.388),  # AAPM Report 204, Table A
            (16.0, 1.323),  # AAPM Report 204, Table A
            (18.0, 1.261),  # AAPM Report 204, Table A
            (20.0, 1.202),  # AAPM Report 204, Table A
            (22.0, 1.145),  # AAPM Report 204, Table A
            (24.0, 1.091),  # AAPM Report 204, Table A
            (26.0, 1.040),  # AAPM Report 204, Table A
            (28.0, 0.991),  # AAPM Report 204, Table A
            (30.0, 0.945),  # AAPM Report 204, Table A
            (32.0, 0.900),  # AAPM Report 204, Table A
            (34.0, 0.858),  # AAPM Report 204, Table A
            (36.0, 0.818),  # AAPM Report 204, Table A
            (38.0, 0.779),  # AAPM Report 204, Table A
            (40.0, 0.742),  # AAPM Report 204, Table A
        ),
        dw_min_cm=10.0,   # AAPM Report 204 — minimum tabulated Dw
        dw_max_cm=40.0,   # AAPM Report 204 — maximum tabulated Dw
        lpv_air_value=0.0,
        lpv_scale_factor=1.0,
    ),

    # ── ED Calibration ──
    ed_calibration=EDCalibrationConfig(
        # Schneider et al. Phys. Med. Biol. 41(1) 1996, Table 1;
        # IAEA TRS-430 (2004) Table 4.1
        # Format: (material_name, nominal_hu, relative_electron_density)
        phantom_materials=(
            ("air",             -1000,  0.001),   # Schneider 1996 Table 1
            ("lung_inhale",      -700,  0.190),   # Schneider 1996 Table 1
            ("lung_exhale",      -400,  0.489),   # Schneider 1996 Table 1
            ("adipose",           -98,  0.950),   # Schneider 1996 Table 1
            ("breast",            -50,  0.971),   # Schneider 1996 Table 1
            ("water",               0,  1.000),   # RED = 1.0 by definition
            ("csf",                15,  1.007),   # Schneider 1996 Table 1
            ("grey_matter",        37,  1.045),   # Schneider 1996 Table 1
            ("muscle",             40,  1.054),   # Schneider 1996 Table 1
            ("liver",              57,  1.064),   # Schneider 1996 Table 1
            ("trabecular_bone",   400,  1.159),   # IAEA TRS-430 Table 4.1
            ("dense_bone",        700,  1.420),   # IAEA TRS-430 Table 4.1
            ("cortical_bone",     950,  1.695),   # IAEA TRS-430 Table 4.1
        ),
        fit_method="piecewise_linear",
        n_fit_segments=2,
        max_red_deviation=0.02,   # 2% RED — IAEA TRS-430 Section 4.2.3
        tps_format="generic_csv",
    ),

    # ── Predictive ──
    predictive=PredictiveConfig(
        min_history_points=5,
        forecast_horizon_days=180,
        r2_minimum_acceptable=0.70,
        tracked_metric_noise_std="center_water_std_hu",
        tracked_metric_nps_peak="nps_peak_frequency_lpmm",
        tracked_metric_mtf50="mtf_50_lpmm",
        tracked_metric_ed_soft_slope="ed_soft_tissue_slope",
        tracked_metric_ed_bone_slope="ed_bone_slope",
    ),

    # ── Logging ──
    logging=LoggingConfig(
        log_level="INFO",
        log_format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        max_bytes=10 * 1024 * 1024,  # 10 MB
        backup_count=5,
        console_output=True,
    ),

    # ── Helios Phantom ──
    helios_phantom=HeliosPhantomConfig(
        phantom_materials=(
            # (name,         nominal_hu, RED,   radius_mm, angle_deg)
            ("water",              0,  1.000,   0.0,    0.0),   # center insert
            ("air",            -1000,  0.001,  58.0,    0.0),   # 12 o'clock
            ("acrylic",          120,  1.173,  58.0,   60.0),   # 2 o'clock
            ("ldpe",            -100,  0.944,  58.0,  120.0),   # 4 o'clock
            ("polystyrene",      -35,  0.976,  58.0,  180.0),   # 6 o'clock
            ("delrin",           340,  1.359,  58.0,  240.0),   # 8 o'clock
            ("teflon",           990,  1.869,  58.0,  300.0),   # 10 o'clock
        ),
        phantom_outer_radius_mm=100.0,
        insert_radius_mm=58.0,
        insert_diameter_mm=28.0,
        roi_diameter_mm=20.0,
        hu_tolerance=4.0,       # AAPM TG-66 Section 5.2
        red_tolerance=0.02,     # IAEA TRS-430 Section 4.2.3
    ),

    # ── GE Scanner ──
    ge_scanner=GEScannerConfig(
        ge_private_rescale_slope_tag=(0x0009, 0x100D),
        ge_default_intercept=-1024.0,
        ge_localizer_image_type_string="LOCALIZER",
        ge_instance_number_bug_firmware=("27.0", "28.0"),
        ge_sort_fallback_key="SliceLocation",
    ),
)
