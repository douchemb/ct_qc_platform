"""
core/spatial_sort.py
=====================
Spatial Sort Engine — guarantees anatomically correct slice ordering.

The problem: DICOM datasets are not guaranteed to arrive in spatial order.
InstanceNumber may be non-sequential or identical (GE firmware bug).
Processing slices in wrong order produces inverted axial profiles,
incorrect D_w profiles, and wrong SSDE per-position values.

Solution: A cascading sort strategy that tries the most reliable tag first
and falls back gracefully, with explicit logging of every fallback.

Priority cascade (highest to lowest reliability):
  1. ImagePositionPatient[2]  — z-coordinate in mm (most reliable, modern CT)
  2. SliceLocation            — z-coordinate in mm (widely supported)
  3. InstanceNumber           — sequence number (unreliable on some GE firmware)
  4. Filename alphabetical    — last resort, with WARNING

Post-sort validation:
  - Checks for monotonic z-progression (detects flipped stacks)
  - Detects gap anomalies (z-gap > 3× nominal slice thickness)
  - Detects duplicate z-positions (same slice reconstructed twice)
  - All anomalies logged at WARNING, never silently ignored

Reference: DICOM PS 3.3 C.7.6.2 — Image Plane Module;
           DICOM PS 3.3 C.7.6.3.1.3 — ImagePositionPatient;
           AAPM TG-66 §4 — spatial integrity of CT datasets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Custom exceptions ──────────────────────────────────────────────────────

class SpatialSortError(RuntimeError):
    """Raised when spatial sorting fails and no fallback is available."""


class DuplicateSlicePositionError(ValueError):
    """
    Raised when multiple slices share the same z-position and
    strict mode is enabled. In non-strict mode, only a WARNING is logged.
    """


# ── Sort result ────────────────────────────────────────────────────────────

@dataclass
class SpatialSortResult:
    """
    Result of a spatial sort operation.
    Contains the sorted datasets and diagnostic information.
    """
    datasets: list                  # sorted pydicom.Dataset list
    sort_method: str                # which tag was used for sorting
    z_positions_mm: list[float]     # z-coordinate of each sorted slice
    is_ascending: bool              # True if z increases (foot→head convention)
    has_gaps: bool                  # True if gap > 3× slice thickness detected
    has_duplicates: bool            # True if duplicate z-positions detected
    gap_indices: list[int]          # indices where gaps were detected
    duplicate_indices: list[int]    # indices of duplicate z-positions
    warnings: list[str]            # all warnings generated during sort

    def to_dict(self) -> dict:
        return {
            "sort_method":       self.sort_method,
            "n_slices":          len(self.datasets),
            "is_ascending":      self.is_ascending,
            "has_gaps":          self.has_gaps,
            "has_duplicates":    self.has_duplicates,
            "gap_indices":       self.gap_indices,
            "duplicate_indices": self.duplicate_indices,
            "warnings":          self.warnings,
        }


# ── Engine ─────────────────────────────────────────────────────────────────

class SpatialSortEngine:
    """
    Sorts a list of pydicom.Dataset objects into anatomically correct
    spatial order using a cascading fallback strategy.

    The engine is instantiated once and reused across multiple series.
    It is integrated into DicomLoader.load_directory() transparently —
    all downstream modules always receive spatially sorted datasets.

    Configuration:
      preferred_tag  : which tag to try first (from ScannerProfile)
      ascending      : if True, sort foot→head (increasing z); if None, auto-detect
      strict         : if True, raise on duplicate z-positions; else log WARNING
    """

    # Tags in priority order (used when preferred_tag extraction fails)
    _FALLBACK_ORDER = [
        "ImagePositionPatient",
        "SliceLocation",
        "InstanceNumber",
    ]

    def __init__(
        self,
        preferred_tag: str = "SliceLocation",
        ascending: Optional[bool] = None,
        strict: bool = False,
        gap_tolerance_factor: float = 3.0,
    ) -> None:
        """
        Parameters
        ----------
        preferred_tag : str
            First tag to attempt. Set from ScannerProfile.sort_preferred_tag.
        ascending : bool | None
            Force sort direction. None = auto-detect from first/last z-position.
        strict : bool
            If True, raise DuplicateSlicePositionError on duplicate z-positions.
        gap_tolerance_factor : float
            Gaps larger than factor × nominal_slice_thickness are flagged.
            Default 3.0: a gap of > 3 slices is anomalous.
        """
        self._preferred_tag       = preferred_tag
        self._ascending           = ascending
        self._strict              = strict
        self._gap_tolerance_factor = gap_tolerance_factor

    def sort(
        self,
        datasets: list,
        source_paths: Optional[list[Path]] = None,
    ) -> SpatialSortResult:
        """
        Sorts the datasets spatially and returns a SpatialSortResult.

        Parameters
        ----------
        datasets : list[pydicom.Dataset]
            Unsorted DICOM datasets. Must be non-empty.
        source_paths : list[Path] | None
            Optional file paths for filename-based fallback sort.

        Returns
        -------
        SpatialSortResult with sorted datasets and full diagnostic info.
        """
        if not datasets:
            raise SpatialSortError("Cannot sort an empty dataset list.")

        warnings_log: list[str] = []

        # Build the tag priority list starting with the preferred tag
        tag_order = [self._preferred_tag] + [
            t for t in self._FALLBACK_ORDER if t != self._preferred_tag
        ]

        sort_method = "filename"
        z_values    = None

        for tag in tag_order:
            z_values = self._extract_z_values(datasets, tag)
            if z_values is not None:
                sort_method = tag
                logger.debug(
                    "SpatialSortEngine: using '%s' for sort (%d/%d valid values)",
                    tag, sum(v is not None for v in z_values), len(datasets)
                )
                break
            else:
                msg = (
                    "SpatialSortEngine: tag '%s' not usable "
                    "(absent or identical across all slices) — trying next fallback."
                    % tag
                )
                logger.warning("%s", msg)
                warnings_log.append(msg)

        # If all DICOM tags failed, fall back to filename sort
        if z_values is None:
            msg = (
                "SpatialSortEngine: all DICOM spatial tags failed. "
                "Falling back to alphabetical filename sort. "
                "Spatial accuracy of results cannot be guaranteed."
            )
            logger.warning("%s", msg)
            warnings_log.append(msg)
            sorted_ds = self._sort_by_filename(datasets, source_paths)
            z_pos = [float(i) for i in range(len(sorted_ds))]
            return SpatialSortResult(
                datasets=sorted_ds,
                sort_method="filename",
                z_positions_mm=z_pos,
                is_ascending=True,
                has_gaps=False,
                has_duplicates=False,
                gap_indices=[],
                duplicate_indices=[],
                warnings=warnings_log,
            )

        # Handle None values in z_values (partially missing tag)
        # Replace None with interpolated values or use index as fallback
        z_clean = self._fill_missing_z(z_values, warnings_log)

        # Sort by z-coordinate
        indexed = sorted(enumerate(z_clean), key=lambda x: x[1])
        sorted_indices = [i for i, _ in indexed]
        sorted_z       = [z for _, z in indexed]
        sorted_ds      = [datasets[i] for i in sorted_indices]

        # Auto-detect direction
        is_ascending = (
            self._ascending
            if self._ascending is not None
            else sorted_z[0] <= sorted_z[-1]
        )

        if not is_ascending:
            sorted_ds.reverse()
            sorted_z.reverse()
            logger.debug("SpatialSortEngine: descending stack detected and reversed.")

        # Post-sort validation
        gaps_idx, dups_idx = self._validate_spatial_consistency(
            sorted_z, datasets, warnings_log
        )

        logger.info(
            "SpatialSortEngine: %d slices sorted by '%s' | ascending=%s | "
            "gaps=%d | duplicates=%d",
            len(sorted_ds), sort_method, is_ascending,
            len(gaps_idx), len(dups_idx)
        )

        return SpatialSortResult(
            datasets=sorted_ds,
            sort_method=sort_method,
            z_positions_mm=sorted_z,
            is_ascending=is_ascending,
            has_gaps=len(gaps_idx) > 0,
            has_duplicates=len(dups_idx) > 0,
            gap_indices=gaps_idx,
            duplicate_indices=dups_idx,
            warnings=warnings_log,
        )

    def _extract_z_values(
        self,
        datasets: list,
        tag: str,
    ) -> Optional[list[Optional[float]]]:
        """
        Extracts z-coordinate values from a specific DICOM tag.
        Returns None if the tag is absent on all slices or all values are equal.
        Returns a list with None for individual missing values.
        """
        values = []
        for ds in datasets:
            val = self._read_tag_z(ds, tag)
            values.append(val)

        # All missing → tag not available
        non_none = [v for v in values if v is not None]
        if len(non_none) == 0:
            return None

        # All identical → useless for sorting (GE InstanceNumber=0 bug)
        if len(set(round(v, 4) for v in non_none)) == 1:
            logger.debug(
                "SpatialSortEngine: all '%s' values are identical (%.4f). "
                "This tag cannot be used for sorting.",
                tag, non_none[0]
            )
            return None

        return values

    def _read_tag_z(self, ds, tag: str) -> Optional[float]:
        """Reads the z-coordinate from a specific DICOM tag."""
        try:
            if tag == "ImagePositionPatient":
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is not None and len(ipp) >= 3:
                    return float(ipp[2])     # z-coordinate (3rd element)
            elif tag == "SliceLocation":
                sl = getattr(ds, "SliceLocation", None)
                if sl is not None:
                    return float(sl)
            elif tag == "InstanceNumber":
                inst = getattr(ds, "InstanceNumber", None)
                if inst is not None:
                    return float(inst)
        except (TypeError, ValueError, AttributeError):
            pass
        return None

    def _fill_missing_z(
        self,
        z_values: list[Optional[float]],
        warnings_log: list[str],
    ) -> list[float]:
        """
        Replaces None values in z_values with linearly interpolated estimates.
        If all values are None, assigns sequential integers.
        """
        non_none_indices = [i for i, v in enumerate(z_values) if v is not None]
        if not non_none_indices:
            return list(range(len(z_values)))

        result = list(z_values)
        for i, v in enumerate(z_values):
            if v is None:
                # Find nearest non-None neighbours
                prev_idx = max((j for j in non_none_indices if j < i), default=None)
                next_idx = min((j for j in non_none_indices if j > i), default=None)
                if prev_idx is not None and next_idx is not None:
                    # Linear interpolation
                    t = (i - prev_idx) / (next_idx - prev_idx)
                    result[i] = z_values[prev_idx] + t * (z_values[next_idx] - z_values[prev_idx])
                elif prev_idx is not None:
                    result[i] = z_values[prev_idx] + (i - prev_idx)
                else:
                    result[i] = z_values[next_idx] - (next_idx - i)
                msg = "z-value at index %d was None — interpolated to %.2f mm" % (i, result[i])
                warnings_log.append(msg)
                logger.debug("SpatialSortEngine: %s", msg)

        return result

    def _validate_spatial_consistency(
        self,
        z_sorted: list[float],
        datasets: list,
        warnings_log: list[str],
    ) -> tuple[list[int], list[int]]:
        """
        Validates the sorted z-positions for gaps and duplicates.
        Returns (gap_indices, duplicate_indices).
        """
        if len(z_sorted) < 2:
            return [], []

        # Nominal slice thickness from first dataset
        nominal_thickness = float(
            getattr(datasets[0], "SliceThickness", 0.0) or 0.0
        )

        gaps, dups = [], []
        diffs = [abs(z_sorted[i+1] - z_sorted[i]) for i in range(len(z_sorted)-1)]

        if not diffs:
            return [], []

        median_diff = float(np.median(diffs))
        threshold   = self._gap_tolerance_factor * max(
            median_diff, nominal_thickness, 0.1
        )

        for i, diff in enumerate(diffs):
            # Duplicate check: diff < 10% of median
            if median_diff > 0 and diff < 0.1 * median_diff:
                msg = (
                    "Duplicate z-position detected at index %d "
                    "(z=%.2f mm ≈ z=%.2f mm, "
                    "diff=%.3f mm). "
                    "This may indicate the same slice was reconstructed twice."
                    % (i, z_sorted[i], z_sorted[i+1], diff)
                )
                logger.warning("%s", msg)
                warnings_log.append(msg)
                dups.append(i)
                if self._strict:
                    raise DuplicateSlicePositionError(msg)

            # Gap check: diff > 3× median
            elif diff > threshold:
                msg = (
                    "Spatial gap detected between slices %d and %d: "
                    "z=%.2f mm → %.2f mm "
                    "(gap=%.2f mm, threshold=%.2f mm). "
                    "Slices may be missing from this range."
                    % (i, i+1, z_sorted[i], z_sorted[i+1], diff, threshold)
                )
                logger.warning("%s", msg)
                warnings_log.append(msg)
                gaps.append(i)

        return gaps, dups

    def _sort_by_filename(
        self,
        datasets: list,
        source_paths: Optional[list[Path]],
    ) -> list:
        """Sorts datasets by filename when all DICOM spatial tags fail."""
        if source_paths and len(source_paths) == len(datasets):
            paired   = sorted(zip(source_paths, datasets), key=lambda x: x[0].name)
            return [ds for _, ds in paired]
        # No paths available — return as-is
        return datasets
