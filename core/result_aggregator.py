# -*- coding: utf-8 -*-
"""
core/result_aggregator.py — QC Result Aggregation & Persistence.

Collects per-slice analysis results, computes summary statistics,
and provides atomic JSON serialization for audit trail and archival.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config import CONFIG

if TYPE_CHECKING:
    from modules.image_qc.roi_stats import SliceAnalysisResult

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["ResultAggregator"]


class ResultAggregator:
    """Collects and summarizes SliceAnalysisResult objects.

    Provides JSON serialization/deserialization for audit trail
    and downstream consumption by the predictive maintenance module.
    """

    def __init__(self) -> None:
        self._results: list[SliceAnalysisResult] = []

    def add_slice_result(self, result: "SliceAnalysisResult") -> None:
        """Append a single slice result to the aggregator.

        Parameters
        ----------
        result : SliceAnalysisResult
            Per-slice analysis result to add.
        """
        self._results.append(result)
        logger.debug("Added result for slice: %s", result.source_file)

    def get_summary(self) -> dict:
        """Compute summary statistics across all collected slices.

        Returns
        -------
        dict
            Contains: total_slices, per_roi_mean, per_roi_std,
            tg66_pass_count, tg66_fail_count, date_range.
        """
        if not self._results:
            return {
                "total_slices": 0,
                "per_roi_mean": {},
                "per_roi_std": {},
                "tg66_pass_count": 0,
                "tg66_fail_count": 0,
                "date_range": {"earliest": None, "latest": None},
            }

        total = len(self._results)

        # Collect per-ROI statistics across slices
        roi_means: dict[str, list[float]] = {}
        roi_stds: dict[str, list[float]] = {}
        tg66_pass = 0
        tg66_fail = 0
        dates: list[str] = []

        for result in self._results:
            dates.append(result.metadata.acquisition_date)
            for stat in result.roi_results:
                label = stat.roi_label
                if label not in roi_means:
                    roi_means[label] = []
                    roi_stds[label] = []
                roi_means[label].append(stat.mean_hu)
                roi_stds[label].append(stat.std_hu)

                # TG-66 noise tolerance — AAPM TG-66 Section 5.1
                if stat.passes_tg66_noise_tolerance():
                    tg66_pass += 1
                else:
                    tg66_fail += 1

        # Compute per-ROI mean/std across slices
        per_roi_mean = {}
        per_roi_std = {}
        for label in roi_means:
            arr = np.array(roi_means[label])
            per_roi_mean[label] = float(np.mean(arr))
            std_arr = np.array(roi_stds[label])
            per_roi_std[label] = float(np.mean(std_arr))

        sorted_dates = sorted(dates)

        return {
            "total_slices": total,
            "per_roi_mean": per_roi_mean,
            "per_roi_std": per_roi_std,
            "tg66_pass_count": tg66_pass,
            "tg66_fail_count": tg66_fail,
            "date_range": {
                "earliest": sorted_dates[0] if sorted_dates else None,
                "latest": sorted_dates[-1] if sorted_dates else None,
            },
        }

    def to_json(self, output_path: Path) -> None:
        """Serialize all results to JSON with atomic file write.

        Writes to a temporary file first, then renames atomically
        using Path.replace() to prevent data corruption.

        Parameters
        ----------
        output_path : Path
            Destination path for the JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self._results],
        }

        # Atomic write: write to .tmp, then rename
        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        tmp_path.replace(output_path)  # Atomic rename
        logger.info("Results saved to: %s", output_path)

    @classmethod
    def from_json(cls, input_path: Path) -> "ResultAggregator":
        """Load and reconstruct a ResultAggregator from a JSON file.

        Parameters
        ----------
        input_path : Path
            Path to a previously saved JSON results file.

        Returns
        -------
        ResultAggregator
            Reconstructed aggregator with loaded results.
        """
        from modules.image_qc.roi_stats import SliceAnalysisResult, ROIStatistics
        from core.dicom_loader import DicomMetadata

        input_path = Path(input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        aggregator = cls()

        for result_dict in data.get("results", []):
            # Reconstruct metadata
            meta_dict = result_dict.get("metadata", {})
            metadata = DicomMetadata(**meta_dict)

            # Reconstruct ROI results
            roi_results = []
            for roi_dict in result_dict.get("roi_results", []):
                roi_results.append(ROIStatistics(**roi_dict))

            slice_result = SliceAnalysisResult(
                metadata=metadata,
                roi_results=roi_results,
                analysis_timestamp=result_dict.get("analysis_timestamp", ""),
                source_file=result_dict.get("source_file", ""),
            )
            aggregator.add_slice_result(slice_result)

        logger.info("Loaded %d results from: %s", len(aggregator._results), input_path)
        return aggregator
