# -*- coding: utf-8 -*-
"""
tests/test_metrics_archive.py — Tests for QC Metrics Archive.

Validates atomic writes, file locking, schema migration,
session append/load, metric series extraction, and CSV export.
"""

from __future__ import annotations

import json
import uuid

import pytest

from modules.predictive.metrics_archive import (
    MetricsArchive,
    QCSessionRecord,
    ArchiveCorruptionError,
)


def _make_record(date_str: str = "2024-01-15", std_hu: float = 3.5) -> QCSessionRecord:
    """Helper to create a minimal test QCSessionRecord."""
    return QCSessionRecord(
        session_id=str(uuid.uuid4()),
        session_date=date_str,
        session_timestamp="%sT10:00:00Z" % date_str,
        scanner_id="TEST_SCANNER",
        operator_id="TEST_OP",
        center_water_std_hu=std_hu,
        center_water_mean_hu=0.1,
        notes="test session",
        schema_version="1.0",
    )


class TestMetricsArchiveCreation:
    """Test archive file creation."""

    def test_creates_archive_on_append(self, tmp_path):
        """MetricsArchive creates the archive file on append_session if missing."""
        archive_path = tmp_path / "new_archive.json"
        archive = MetricsArchive(archive_path)
        record = _make_record()
        archive.append_session(record)
        assert archive_path.is_file()

    def test_append_then_load_returns_length_1(self, tmp_path):
        """append_session then load_all_sessions returns a list of length 1."""
        archive = MetricsArchive(tmp_path / "archive.json")
        archive.append_session(_make_record())
        sessions = archive.load_all_sessions()
        assert len(sessions) == 1


class TestMetricsArchiveMultipleSessions:
    """Test multi-session operations."""

    def test_five_sessions_all_loaded(self, tmp_path):
        """Appending 5 sessions then loading returns all 5, sorted by date."""
        archive = MetricsArchive(tmp_path / "archive.json")
        dates = ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"]
        for d in dates:
            archive.append_session(_make_record(date_str=d))
        sessions = archive.load_all_sessions()
        assert len(sessions) == 5
        # Check sorted order
        for i in range(len(sessions) - 1):
            assert sessions[i].session_date <= sessions[i + 1].session_date

    def test_get_metric_series_returns_equal_length_lists(self, populated_archive):
        """get_metric_series returns two lists of equal length."""
        dates, values = populated_archive.get_metric_series("center_water_std_hu")
        assert len(dates) == len(values)
        assert len(dates) == 8

    def test_date_filtering(self, populated_archive):
        """Date filtering returns only sessions within the range."""
        dates, values = populated_archive.get_metric_series(
            "center_water_std_hu", min_date="2024-03-01", max_date="2024-06-30",
        )
        assert len(dates) == 4  # March, April, May, June

    def test_all_none_metric_returns_empty(self, tmp_path):
        """get_metric_series for an all-None metric returns ([], [])."""
        archive = MetricsArchive(tmp_path / "archive.json")
        record = _make_record()
        record.mtf_50_lpmm = None  # Ensure it's None
        archive.append_session(record)
        dates, values = archive.get_metric_series("mtf_50_lpmm")
        assert dates == []
        assert values == []


class TestLatestSession:
    """Test get_latest_session."""

    def test_latest_session_returns_most_recent(self, populated_archive):
        """get_latest_session returns the most recent session by date."""
        latest = populated_archive.get_latest_session()
        assert latest is not None
        assert latest.session_date == "2024-08-15"

    def test_latest_session_empty_archive(self, tmp_path):
        """get_latest_session returns None on an empty archive."""
        archive = MetricsArchive(tmp_path / "empty.json")
        assert archive.get_latest_session() is None


class TestCSVExport:
    """Test CSV export."""

    def test_export_creates_csv(self, populated_archive, tmp_path):
        """export_to_csv creates a CSV file with header + 8 data rows."""
        csv_path = tmp_path / "export.csv"
        populated_archive.export_to_csv(csv_path)
        assert csv_path.is_file()
        lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 9  # 1 header + 8 data rows


class TestQCSessionRecord:
    """Test QCSessionRecord methods."""

    def test_from_dict_missing_keys_no_error(self):
        """from_dict on a dict with missing optional keys returns None for them."""
        data = {"session_id": "abc", "session_date": "2024-01-15"}
        record = QCSessionRecord.from_dict(data)
        assert record.session_id == "abc"
        assert record.center_water_std_hu is None
        assert record.mtf_50_lpmm is None

    def test_get_metric_returns_float(self):
        """get_metric for existing field returns the correct float."""
        record = _make_record(std_hu=4.2)
        assert record.get_metric("center_water_std_hu") == 4.2

    def test_get_metric_nonexistent_returns_none(self):
        """get_metric for nonexistent field returns None."""
        record = _make_record()
        assert record.get_metric("nonexistent_field") is None


class TestAtomicWrite:
    """Test atomic write guarantees."""

    def test_no_tmp_file_after_append(self, tmp_path):
        """The .tmp file should not exist after a successful append_session."""
        archive_path = tmp_path / "archive.json"
        archive = MetricsArchive(archive_path)
        archive.append_session(_make_record())
        tmp_path_check = archive_path.with_suffix(".tmp")
        assert not tmp_path_check.is_file()

    def test_valid_json_after_10_appends(self, tmp_path):
        """Archive file is valid JSON after 10 consecutive append_session calls."""
        archive_path = tmp_path / "archive.json"
        archive = MetricsArchive(archive_path)
        for i in range(10):
            archive.append_session(_make_record(date_str="2024-%02d-15" % (i % 12 + 1)))

        with open(archive_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data["sessions"]) == 10
