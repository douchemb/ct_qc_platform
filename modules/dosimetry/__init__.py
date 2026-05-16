# -*- coding: utf-8 -*-
"""modules.dosimetry — Patient dosimetry modules (SSDE per AAPM TG-220)."""
__version__ = "0.1.0"
__all__ = [
    "LocalizerParser",
    "LocalizerData",
    "LocalizerMetadata",
    "LocalizerNotFoundError",
    "LocalizerCalibrationError",
    "LocalizerOrientationError",
    "DwCalculator",
    "DwSliceResult",
    "DwSeriesResult",
    "DwLocalizerResult",
    "BodySegmentationError",
    "InsufficientSlicesError",
    "SSDECalculator",
    "SSDESliceResult",
    "SSDESeriesResult",
    "MissingCTDIvolError",
    "SSDEComputationError",
    "DosimetryReport",
]
