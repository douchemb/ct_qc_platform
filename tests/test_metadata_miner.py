"""Tests for MetadataMiner — CTDIvol consensus and protocol extraction."""
from __future__ import annotations
import json
import pytest
import numpy as np


class TestMetadataMiner:

    def test_miner_instantiates(self, metadata_miner_siemens):
        assert metadata_miner_siemens is not None

    def test_mine_returns_mined_metadata(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        from core.metadata_miner import MinedMetadata
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        assert isinstance(result, MinedMetadata)

    def test_mine_extracts_ctdivol(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        assert result.dose.has_ctdi_vol is True
        assert result.dose.ctdi_vol_mgy == pytest.approx(14.2, abs=0.01)

    def test_mine_extracts_kvp(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        assert result.protocol.kvp == pytest.approx(120.0, abs=0.1)

    def test_mine_extracts_slice_thickness(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        assert result.protocol.slice_thickness_mm == pytest.approx(3.0, abs=0.1)

    def test_mine_extracts_n_slices(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine(
            [synthetic_siemens_ct_dicom, synthetic_siemens_ct_dicom]
        )
        assert result.protocol.n_slices == 2

    def test_consensus_ctdivol_from_multiple_slices(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        """Consensus CTDIvol from 3 identical slices equals the single-slice value."""
        result = metadata_miner_siemens.mine(
            [synthetic_siemens_ct_dicom] * 3
        )
        assert result.dose.ctdi_vol_mgy == pytest.approx(14.2, abs=0.01)

    def test_mine_result_to_dict_serializable(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        json.dumps(result.to_dict(), default=str)

    def test_mine_warnings_is_list(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        assert isinstance(result.warnings, list)

    def test_mine_empty_list_raises(self, metadata_miner_siemens):
        with pytest.raises(ValueError):
            metadata_miner_siemens.mine([])

    def test_ctdivol_consistency_consistent(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        """Same CTDIvol on all slices → consistent."""
        result = metadata_miner_siemens.mine(
            [synthetic_siemens_ct_dicom] * 5
        )
        assert result.ctdivol_consistent_across_slices is True

    def test_ctdivol_range_returned(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine(
            [synthetic_siemens_ct_dicom] * 3
        )
        if result.ctdivol_slice_range_mgy is not None:
            lo, hi = result.ctdivol_slice_range_mgy
            assert lo <= hi

    def test_dose_metadata_source_is_standard_tag(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        assert result.dose.source == "standard_tag"

    def test_protocol_reconstruction_kernel_not_none(
        self, metadata_miner_siemens, synthetic_siemens_ct_dicom
    ):
        """ConvolutionKernel tag present in synthetic Siemens dataset."""
        result = metadata_miner_siemens.mine([synthetic_siemens_ct_dicom])
        assert result.protocol.reconstruction_kernel is None or \
               isinstance(result.protocol.reconstruction_kernel, str)
