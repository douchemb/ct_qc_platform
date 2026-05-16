# -*- coding: utf-8 -*-
"""modules.image_qc — Image quality control analysis modules."""
__version__ = "0.1.0"
__all__ = [
    # ROI Statistics (Phase 1)
    "PhantomROIAnalyzer",
    "ROIDescriptor",
    "ROIStatistics",
    "SliceAnalysisResult",
    "VolumetricQCResult",
    "VolumetricROIStat",
    "ROIBoundsError",
    "compute_batch_statistics",
    # NPS (Phase 2 — Step 6)
    "NPSCalculator",
    "NPSResult",
    "NPSPatch",
    "InsufficientPatchesError",
    # MTF (Phase 2 — Step 8)
    "MTFCalculator",
    "MTFResult",
    # HU Linearity (Phase 2 — Step 9)
    "HULinearityAnalyzer",
    "HULinearityResult",
    # ED Calibration (Phase 2 — Step 10)
    "EDCalibrationAnalyzer",
    "EDCalibrationResult",
    "EDMaterialMeasurement",
    # Helios Phantom Geometry (Phase 5 — Step 20)
    "HeliosPhantomDetector",
    "HeliosGeometry",
    # Phantom Adapters (Phase 5 — Step 5.2)
    "PhantomAdapterFactory",
    "PhantomAdapter",
    "SiemensWaterPhantomAdapter",
    "HeliosQAPhantomAdapter",
    "GenericPhantomAdapter",
    "PhantomDetectionResult",
    "MaterialReference",
    # Basic Metrics Engine (Phase 6 — Step 6.1+6.2)
    "BasicMetricsEngine",
    "BasicQAResult",
    "NoiseResult",
    "UniformityResult",
    "CTNumberAccuracyResult",
    "ContrastResult",
    "SliceThicknessResult",
    "InsufficientROIError",
    "SliceThicknessROIError",
    # Advanced Metrics Engine (Phase 6 — Step 6.3)
    "AdvancedMetricsEngine",
    "AdvancedQAResult",
]
