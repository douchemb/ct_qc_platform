# -*- coding: utf-8 -*-
"""core — Core infrastructure for the CT QC Platform."""

__version__ = "0.1.0"
__all__ = [
    "DicomLoader",
    "DicomMetadata",
    "DicomLoadError",
    "DicomModalityError",
    "MissingRescaleTagsError",
    "PixelDataError",
    "MissingPixelSpacingError",
    "SliceRangeError",
    "InsufficientSlicesError",
    "ResultAggregator",
    # Phase 5 — Scanner Profiles (Step 5.1)
    "ScannerProfileRegistry",
    "ScannerProfile",
    "ScannerProfileError",
    "UnknownScannerError",
    # Phase 5 — Spatial Sort (Step 5.3)
    "SpatialSortEngine",
    "SpatialSortResult",
    "SpatialSortError",
    # Phase 5 — Dose Metadata Extractor (Step 5.4)
    "DoseMetadataExtractor",
    "DoseMetadata",
    "DoseCrossValidation",
    # Phase 6 — Metadata Miner (Step 6.4)
    "MetadataMiner",
    "MinedMetadata",
    "ScanProtocolMetadata",
]
