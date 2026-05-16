# -*- coding: utf-8 -*-
"""
tests/test_pipeline_orchestrator.py — Tests for Pipeline Orchestrator.

Integration tests using temp_dicom_dir and related fixtures.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from config import CONFIG
from core.dicom_loader import DicomLoader
from core.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineResult,
    ImageQCBundle,
    DosimetryBundle,
    PredictiveBundle,
)
from modules.image_qc.roi_stats import ROIDescriptor
from modules.predictive.metrics_archive import MetricsArchive


@pytest.fixture
def standard_rois():
    """Standard 5-ROI layout for a 512×512 phantom."""
    return [
        ROIDescriptor("center_water", 226, 226, 60, 60),
        ROIDescriptor("peripheral_12", 80, 236, 40, 40),
        ROIDescriptor("peripheral_3", 236, 390, 40, 40),
        ROIDescriptor("peripheral_6", 390, 236, 40, 40),
        ROIDescriptor("peripheral_9", 236, 80, 40, 40),
    ]


@pytest.fixture
def orchestrator(tmp_path):
    """Returns a PipelineOrchestrator with a fresh archive."""
    loader = DicomLoader(CONFIG.dicom)
    archive = MetricsArchive(tmp_path / "test_archive.json")
    return PipelineOrchestrator(CONFIG, loader, archive)


@pytest.fixture
def orchestrator_with_archive(tmp_path, populated_archive):
    """Returns a PipelineOrchestrator with a pre-populated archive."""
    loader = DicomLoader(CONFIG.dicom)
    return PipelineOrchestrator(CONFIG, loader, populated_archive)


class TestOrchestratorInstantiation:
    """Test basic instantiation."""

    def test_instantiates_without_error(self, orchestrator):
        """PipelineOrchestrator instantiates without error."""
        assert orchestrator is not None


class TestRunImageQC:
    """Test run_image_qc."""

    def test_returns_image_qc_bundle(self, orchestrator, temp_dicom_dir, standard_rois):
        """run_image_qc returns an ImageQCBundle without raising."""
        result = orchestrator.run_image_qc(temp_dicom_dir, standard_rois, 1, 5)
        assert isinstance(result, ImageQCBundle)

    def test_volumetric_result_not_none(self, orchestrator, temp_dicom_dir, standard_rois):
        """image_qc_bundle.volumetric_result is not None."""
        result = orchestrator.run_image_qc(temp_dicom_dir, standard_rois, 1, 5)
        assert result.volumetric_result is not None

    def test_n_slices_processed(self, orchestrator, temp_dicom_dir, standard_rois):
        """volumetric_result.n_slices_processed == 5."""
        result = orchestrator.run_image_qc(temp_dicom_dir, standard_rois, 1, 5)
        assert result.volumetric_result.n_slices_processed == 5

    def test_all_passed_is_bool(self, orchestrator, temp_dicom_dir, standard_rois):
        """all_passed is a boolean."""
        result = orchestrator.run_image_qc(temp_dicom_dir, standard_rois, 1, 5)
        assert isinstance(result.all_passed, bool)


class TestRunPredictive:
    """Test run_predictive."""

    def test_returns_predictive_bundle(self, orchestrator):
        """run_predictive returns a PredictiveBundle (empty archive)."""
        result = orchestrator.run_predictive("TEST_SCANNER")
        assert isinstance(result, PredictiveBundle)

    def test_empty_archive_insufficient_data(self, orchestrator):
        """Empty archive → has_sufficient_data=False."""
        result = orchestrator.run_predictive("TEST_SCANNER")
        assert result.has_sufficient_data is False

    def test_populated_archive_sufficient_data(self, orchestrator_with_archive):
        """Populated archive → has_sufficient_data=True."""
        result = orchestrator_with_archive.run_predictive("TEST_SCANNER")
        assert result.has_sufficient_data is True


class TestRunFullPipeline:
    """Test run_full_pipeline."""

    def test_skipped_pipeline_completes(self, orchestrator, temp_dicom_dir, standard_rois):
        """run_full_pipeline with all skips completes without error."""
        result = orchestrator.run_full_pipeline(
            dicom_dir=temp_dicom_dir,
            rois=standard_rois,
            start_slice=1,
            end_slice=5,
            scanner_id="TEST_SCANNER",
            skip_modules=["nps", "mtf", "ed", "dosimetry", "predictive"],
        )
        assert isinstance(result, PipelineResult)

    def test_session_record_schema(self, orchestrator, temp_dicom_dir, standard_rois):
        """result.session_record.schema_version == '1.0'."""
        result = orchestrator.run_full_pipeline(
            dicom_dir=temp_dicom_dir,
            rois=standard_rois,
            start_slice=1,
            end_slice=5,
            scanner_id="TEST_SCANNER",
            skip_modules=["nps", "mtf", "ed", "dosimetry", "predictive"],
        )
        assert result.session_record.schema_version == "1.0"

    def test_session_id_is_uuid(self, orchestrator, temp_dicom_dir, standard_rois):
        """result.session_record.session_id is a valid UUID string."""
        result = orchestrator.run_full_pipeline(
            dicom_dir=temp_dicom_dir,
            rois=standard_rois,
            start_slice=1,
            end_slice=5,
            scanner_id="TEST_SCANNER",
            skip_modules=["nps", "mtf", "ed", "dosimetry", "predictive"],
        )
        sid = result.session_record.session_id
        assert len(sid) == 36  # UUID format
        # Verify it's a valid UUID
        parsed = uuid.UUID(sid)
        assert str(parsed) == sid

    def test_center_water_std_is_float(self, orchestrator, temp_dicom_dir, standard_rois):
        """center_water_std_hu is a float when image QC ran."""
        result = orchestrator.run_full_pipeline(
            dicom_dir=temp_dicom_dir,
            rois=standard_rois,
            start_slice=1,
            end_slice=5,
            scanner_id="TEST_SCANNER",
            skip_modules=["nps", "mtf", "ed", "dosimetry", "predictive"],
        )
        assert isinstance(result.session_record.center_water_std_hu, float)

    def test_run_duration_positive(self, orchestrator, temp_dicom_dir, standard_rois):
        """result.run_duration_seconds > 0.0."""
        result = orchestrator.run_full_pipeline(
            dicom_dir=temp_dicom_dir,
            rois=standard_rois,
            start_slice=1,
            end_slice=5,
            scanner_id="TEST_SCANNER",
            skip_modules=["nps", "mtf", "ed", "dosimetry", "predictive"],
        )
        assert result.run_duration_seconds > 0.0
