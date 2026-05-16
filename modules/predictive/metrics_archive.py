# -*- coding: utf-8 -*-
"""
modules/predictive/metrics_archive.py — QC Metrics Archive.

Persistent JSON time-series store for QC session records.
Every QC session appends its key metrics to a persistent JSON file.
The predictive model reads this file to detect degradation trends.

Thread safety: file-based lock using companion .lock file.
Atomic writes: tmp file + Path.replace() pattern.
Schema migration: _migrate_schema() handles version upgrades.

Reference: AAPM TG-66 Section 7 — QC Record Keeping
"""

from __future__ import annotations

import csv
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "MetricsArchive",
    "QCSessionRecord",
    "ArchiveLockError",
    "ArchiveCorruptionError",
    "ArchiveSchemaError",
]


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class ArchiveLockError(RuntimeError):
    """
    Raised when the archive file lock cannot be acquired after maximum retries.
    Indicates another process is writing to the archive simultaneously.
    """


class ArchiveCorruptionError(RuntimeError):
    """
    Raised when the archive JSON cannot be parsed.
    Indicates manual intervention is required.
    """


class ArchiveSchemaError(ValueError):
    """Raised when a session record has an incompatible schema version."""


# ═══════════════════════════════════════════════════════════════════
# QC Session Record Dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class QCSessionRecord:
    """
    One complete QC session entry in the archive.
    All metric fields are Optional — older sessions may lack newer metrics.
    Schema versioning ensures forward compatibility as new metrics are added.

    Hardware failure metric mapping (supervisor requirement):
      center_water_std_hu        → X-ray tube filament wear
      nps_peak_frequency_lpmm    → X-ray tube filament wear
      mtf_50_lpmm                → Anode focal spot blooming
      hu_linearity_max_deviation_hu → kVp generator instability
      ed_soft_tissue_slope       → kVp generator instability (low density)
      ed_bone_slope              → kVp generator instability (high density)
    """
    # Session identity
    session_id: str = ""                        # UUID4 string
    session_date: str = ""                      # "YYYY-MM-DD"
    session_timestamp: str = ""                 # ISO 8601 full timestamp
    scanner_id: str = ""
    operator_id: str = ""

    # Image QC metrics
    center_water_mean_hu: Optional[float] = None
    center_water_std_hu: Optional[float] = None         # hardware: tube filament
    center_water_variance_hu: Optional[float] = None
    nps_peak_frequency_lpmm: Optional[float] = None     # hardware: tube filament
    nps_peak_value_hu2mm2: Optional[float] = None
    nps_integral_hu2mm2: Optional[float] = None
    mtf_50_lpmm: Optional[float] = None                 # hardware: focal spot
    mtf_10_lpmm: Optional[float] = None
    hu_linearity_max_deviation_hu: Optional[float] = None  # hardware: kVp generator
    hu_linearity_r_squared: Optional[float] = None

    # ED Calibration metrics — supervisor addition
    ed_soft_tissue_slope: Optional[float] = None        # hardware: kVp generator
    ed_bone_slope: Optional[float] = None               # hardware: kVp generator
    ed_max_red_deviation: Optional[float] = None

    # Dosimetry metrics
    ctdi_vol_mgy: Optional[float] = None
    ssde_mean_mgy: Optional[float] = None
    dw_mean_cm: Optional[float] = None
    effective_dose_msv: Optional[float] = None

    # Session status flags
    all_image_qc_passed: Optional[bool] = None
    all_dosimetry_computed: Optional[bool] = None
    ed_calibration_passed: Optional[bool] = None
    notes: str = ""

    # Schema version for forward compatibility
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QCSessionRecord":
        """
        Reconstruct from dict, tolerating missing keys from older schema versions.
        Any missing field defaults to None. Never raises KeyError.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered = {}
        for key in valid_fields:
            filtered[key] = data.get(key, None)
        # Ensure non-optional fields have defaults
        if filtered.get("session_id") is None:
            filtered["session_id"] = ""
        if filtered.get("session_date") is None:
            filtered["session_date"] = ""
        if filtered.get("session_timestamp") is None:
            filtered["session_timestamp"] = ""
        if filtered.get("scanner_id") is None:
            filtered["scanner_id"] = ""
        if filtered.get("operator_id") is None:
            filtered["operator_id"] = ""
        if filtered.get("notes") is None:
            filtered["notes"] = ""
        if filtered.get("schema_version") is None:
            filtered["schema_version"] = "1.0"
        return cls(**filtered)

    def get_metric(self, metric_name: str) -> Optional[float]:
        """
        Returns any metric field value by name string.
        Returns None if field does not exist or is None.
        Used by QCTrendModel to retrieve values generically.
        """
        return getattr(self, metric_name, None)


# ═══════════════════════════════════════════════════════════════════
# Metrics Archive Class
# ═══════════════════════════════════════════════════════════════════

class MetricsArchive:
    """
    Persistent JSON time-series store for QC session records.

    File format:
    {
        "schema_version": "1.0",
        "sessions": [ {session_dict}, ... ]
    }

    Thread safety: file-based lock using companion .lock file.
    Atomic writes: tmp file + Path.replace() pattern.
    Schema migration: _migrate_schema() handles version upgrades.
    """

    CURRENT_SCHEMA_VERSION = "1.0"
    LOCK_MAX_RETRIES = 10
    LOCK_RETRY_SLEEP_S = 0.5

    def __init__(self, archive_path: Path) -> None:
        self._archive_path = Path(archive_path)

    def append_session(self, record: QCSessionRecord) -> None:
        """Append a QC session record to the archive.

        Thread-safe via file lock. Atomic write.

        Parameters
        ----------
        record : QCSessionRecord
            The session record to append.
        """
        self._acquire_lock()
        try:
            data = self._load_raw()
            data = self._migrate_schema(data)
            data["sessions"].append(record.to_dict())
            self._save_raw(data)
            logger.info(
                "Session %s appended. Total sessions: %d",
                record.session_id[:8] if record.session_id else "N/A",
                len(data["sessions"]),
            )
        finally:
            self._release_lock()

    def load_all_sessions(self) -> list[QCSessionRecord]:
        """Load all sessions from archive, sorted by session_date ascending.

        Returns
        -------
        list[QCSessionRecord]
            All sessions, sorted by date.
        """
        data = self._load_raw()
        data = self._migrate_schema(data)
        sessions = [
            QCSessionRecord.from_dict(s) for s in data.get("sessions", [])
        ]
        sessions.sort(key=lambda r: r.session_date)
        return sessions

    def get_metric_series(
        self,
        metric_name: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> tuple[list[str], list[float]]:
        """Extract a time series for one metric from the archive.

        Parameters
        ----------
        metric_name : str
            Field name on QCSessionRecord.
        min_date : str, optional
            Minimum date filter "YYYY-MM-DD" (inclusive).
        max_date : str, optional
            Maximum date filter "YYYY-MM-DD" (inclusive).

        Returns
        -------
        tuple[list[str], list[float]]
            (dates, values) — two parallel lists sorted by date.
            Returns ([], []) if nothing found.
        """
        sessions = self.load_all_sessions()
        dates: list[str] = []
        values: list[float] = []

        for record in sessions:
            # Date filtering (string comparison works for YYYY-MM-DD)
            if min_date is not None and record.session_date < min_date:
                continue
            if max_date is not None and record.session_date > max_date:
                continue

            value = record.get_metric(metric_name)
            if value is not None:
                dates.append(record.session_date)
                values.append(float(value))

        return dates, values

    def get_latest_session(self) -> Optional[QCSessionRecord]:
        """Return the most recent session by date, or None if empty.

        Returns
        -------
        Optional[QCSessionRecord]
        """
        sessions = self.load_all_sessions()
        if not sessions:
            return None
        return sessions[-1]  # Already sorted ascending

    def get_sessions_in_range(
        self, start_date: str, end_date: str
    ) -> list[QCSessionRecord]:
        """Return sessions within a date range (inclusive).

        Parameters
        ----------
        start_date : str
            "YYYY-MM-DD" start date.
        end_date : str
            "YYYY-MM-DD" end date.

        Returns
        -------
        list[QCSessionRecord]
        """
        sessions = self.load_all_sessions()
        return [
            s for s in sessions
            if start_date <= s.session_date <= end_date
        ]

    def export_to_csv(self, output_path: Path) -> None:
        """Export all sessions to CSV.

        Parameters
        ----------
        output_path : Path
            Output CSV file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sessions = self.load_all_sessions()
        if not sessions:
            logger.warning("No sessions to export")
            return

        field_names = [f.name for f in fields(QCSessionRecord)]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for session in sessions:
                writer.writerow(session.to_dict())

        logger.info("Exported %d sessions to CSV: %s", len(sessions), output_path)

    # ─────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────

    def _load_raw(self) -> dict:
        """Load raw JSON from archive file.

        If file does not exist, initializes empty archive.

        Returns
        -------
        dict
            Raw archive data.

        Raises
        ------
        ArchiveCorruptionError
            If JSON is malformed.
        """
        if not self._archive_path.is_file():
            self._initialize_archive()
            return {"schema_version": self.CURRENT_SCHEMA_VERSION, "sessions": []}

        try:
            text = self._archive_path.read_text(encoding="utf-8")
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArchiveCorruptionError(
                "Archive JSON is malformed at %s: %s" % (self._archive_path, exc)
            ) from exc

    def _save_raw(self, data: dict) -> None:
        """Atomic write: write to .tmp then rename.

        Parameters
        ----------
        data : dict
            Raw archive data to write.
        """
        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._archive_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(self._archive_path)

    def _acquire_lock(self) -> None:
        """Acquire file-based lock using exclusive create.

        Raises
        ------
        ArchiveLockError
            After exhausting retries.
        """
        lock_path = self._archive_path.with_suffix(".lock")
        for attempt in range(self.LOCK_MAX_RETRIES):
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = open(lock_path, "x")
                fd.close()
                return
            except FileExistsError:
                logger.debug(
                    "Lock file exists, retry %d/%d",
                    attempt + 1, self.LOCK_MAX_RETRIES,
                )
                time.sleep(self.LOCK_RETRY_SLEEP_S)

        raise ArchiveLockError(
            "Cannot acquire archive lock after %d retries: %s"
            % (self.LOCK_MAX_RETRIES, lock_path)
        )

    def _release_lock(self) -> None:
        """Release file-based lock by deleting lock file."""
        lock_path = self._archive_path.with_suffix(".lock")
        lock_path.unlink(missing_ok=True)

    def _migrate_schema(self, raw_data: dict) -> dict:
        """Handle schema version upgrades.

        Parameters
        ----------
        raw_data : dict
            Raw archive data.

        Returns
        -------
        dict
            Migrated data.
        """
        version = raw_data.get("schema_version")
        if version == self.CURRENT_SCHEMA_VERSION:
            return raw_data

        # Missing or None version — add it
        if version is None:
            logger.warning(
                "Archive has no schema_version — migrating to %s",
                self.CURRENT_SCHEMA_VERSION,
            )
            raw_data["schema_version"] = self.CURRENT_SCHEMA_VERSION
            for session in raw_data.get("sessions", []):
                if "schema_version" not in session:
                    session["schema_version"] = self.CURRENT_SCHEMA_VERSION

        return raw_data

    def _initialize_archive(self) -> None:
        """Create empty archive file with correct structure."""
        data = {
            "schema_version": self.CURRENT_SCHEMA_VERSION,
            "sessions": [],
        }
        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._archive_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._archive_path)
        logger.info("Initialized empty archive at: %s", self._archive_path)
