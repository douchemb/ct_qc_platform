# -*- coding: utf-8 -*-
"""
modules/dosimetry/dosimetry_report.py — Dosimetry Report Generator.

Assembles and serializes the complete dosimetry results for one CT session.
Combines SSDE, D_w, and optional localizer results into a unified report.

Standards References:
    - AAPM Report 204 (2011): SSDE
    - AAPM TG-220 (2014): Water Equivalent Diameter
    - ICRP Publication 103: Effective Dose
    - EUR 16262: Diagnostic Reference Levels
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from modules.dosimetry.ssde_calculator import SSDESeriesResult
from modules.dosimetry.dw_calculator import DwSeriesResult, DwLocalizerResult

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["DosimetryReport"]


class DosimetryReport:
    """
    Assembles and serializes the complete dosimetry results for one CT session.
    Combines SSDE, D_w, and optional localizer results into a unified report.
    """

    def __init__(
        self,
        ssde_result: SSDESeriesResult,
        dw_result: DwSeriesResult,
        localizer_result: Optional[DwLocalizerResult] = None,
    ) -> None:
        self.ssde_result = ssde_result
        self.dw_result = dw_result
        self.localizer_result = localizer_result

    def to_dict(self) -> dict:
        """Serialize the complete dosimetry report to a dictionary."""
        result = {
            "ssde_result": self.ssde_result.to_dict(),
            "dw_result": self.dw_result.to_dict(),
            "compliance_flags": self.get_compliance_flags(),
        }
        if self.localizer_result is not None:
            result["localizer_result"] = self.localizer_result.to_dict()
        return result

    def to_json(self, output_path: Path) -> None:
        """Atomic write: tmp file first, then Path.replace().

        Creates parent directories if needed.

        Parameters
        ----------
        output_path : Path
            Destination path for the JSON report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = output_path.with_suffix(".tmp")

        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)

        # Write to temporary file first (atomic write pattern)
        tmp_path.write_text(json_str, encoding="utf-8")

        # Atomic rename
        tmp_path.replace(output_path)

        logger.info("Dosimetry report saved to: %s", output_path)

    def print_summary(self) -> None:
        """
        Logs a structured summary using logger.info (not print).
        """
        ssde = self.ssde_result
        dw = self.dw_result
        flags = self.get_compliance_flags()

        drl_status = "PASS" if flags["ctdi_below_drl"] else "FAIL"
        dw_range_status = "PASS" if flags["dw_within_table_range"] else "FAIL"
        loc_status = "YES" if flags["localizer_available"] else "NO"

        summary = (
            "\n"
            "══════════════════════════════════════\n"
            " DOSIMETRY REPORT — %s\n"
            " Date: %s\n"
            "──────────────────────────────────────\n"
            " CTDIvol          : %.2f mGy\n"
            " D_w (mean)       : %.2f ± %.2f cm\n"
            " D_w (isocenter)  : %.2f cm\n"
            " SSDE (mean)      : %.2f mGy\n"
            " SSDE (isocenter) : %.2f mGy\n"
            " DLP              : %.1f mGy·cm\n"
            " Effective dose   : %.2f mSv\n"
            "──────────────────────────────────────\n"
            " DRL check (25mGy): %s\n"
            " D_w in range     : %s\n"
            " Localizer avail  : %s\n"
            "══════════════════════════════════════"
            % (
                ssde.series_description,
                ssde.acquisition_date,
                ssde.ctdi_vol_mgy,
                dw.dw_mean_cm,
                dw.dw_std_cm,
                dw.dw_at_isocenter_cm,
                ssde.ssde_mean_mgy,
                ssde.ssde_at_isocenter_mgy,
                ssde.dlp_mgy_cm,
                ssde.effective_dose_msv,
                drl_status,
                dw_range_status,
                loc_status,
            )
        )
        logger.info(summary)

    def get_compliance_flags(self) -> dict[str, bool]:
        """
        Returns compliance checks.

        Returns
        -------
        dict[str, bool]
            Dictionary with exactly 5 boolean compliance flags.
        """
        # Check if all D_w values are within table range
        dw_in_range = all(
            sr.within_table_range
            for sr in self.ssde_result.slice_results
        )

        return {
            "ctdi_below_drl": self.ssde_result.passes_diagnostic_reference_level(25.0),
            "ssde_computed": self.ssde_result.ssde_mean_mgy > 0.0,
            "dw_within_table_range": dw_in_range,
            "effective_dose_estimated": self.ssde_result.effective_dose_msv > 0.0,
            "localizer_available": self.localizer_result is not None,
        }
