# -*- coding: utf-8 -*-
"""
CT QC Platform v0.1.0 — Main Entry Point
Radiotherapy CT Simulator Quality Control & Dosimetry Platform

Modes:
  --mode qc          Image QC analysis only
  --mode dosimetry   Dosimetry calculation only
  --mode predictive  Predictive maintenance only (no DICOM required)
  --mode all         Full pipeline: QC → Dosimetry → Predictive
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import CONFIG
from core.dicom_loader import DicomLoader
from core.pipeline_orchestrator import PipelineOrchestrator
from modules.image_qc.roi_stats import ROIDescriptor
from modules.predictive.metrics_archive import MetricsArchive

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# CLI Argument Parser
# ═══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ct-qc-platform",
        description="CT Simulator QC, Dosimetry & Predictive Maintenance Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dicom-dir", type=Path, default=None,
                        help="Path to DICOM directory.")
    parser.add_argument("--mode", required=True,
                        choices=["qc", "dosimetry", "predictive", "all"],
                        help="Execution mode.")
    parser.add_argument("--scanner-id", default="SCANNER_001",
                        help="Scanner identifier.")
    parser.add_argument("--operator-id", default="",
                        help="Operator identifier.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory.")
    parser.add_argument("--start-slice", type=int, default=None,
                        help="Start slice (1-indexed).")
    parser.add_argument("--end-slice", type=int, default=None,
                        help="End slice (1-indexed).")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["nps", "mtf", "ed", "dosimetry", "predictive"],
                        help="Modules to skip.")
    parser.add_argument("--notes", default="",
                        help="Session notes.")
    parser.add_argument("--archive-path", type=Path, default=None,
                        help="Path to QC archive JSON.")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export archive to CSV after run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse inputs but write no outputs.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    parser.add_argument(
        "--phantom",
        type=str,
        default=None,
        choices=["ge_helios", "catphan_504"],
        help=(
            "Phantom type for automatic geometry detection and reference values. "
            "Default: reads 'active_phantom' from phantom_config.json. "
            "ge_helios: GE Helios QA Phantom (P/N 2165993-100). "
            "catphan_504: The Phantom Laboratory CatPhan 504."
        ),
    )
    parser.add_argument(
        "--scanner",
        type=str,
        default=None,
        choices=["ge_discovery_rt", "generic"],
        help=(
            "Scanner profile for DICOM parsing compatibility. "
            "Default: reads 'active_scanner' from phantom_config.json. "
            "ge_discovery_rt: enables GE private tag fallback chain. "
            "generic: standard DICOM tags only."
        ),
    )
    return parser


# ═══════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════

def _initialize_logging(verbose: bool = False, output_dir: Path = None) -> None:
    """Configure rotating file handler and console handler."""
    log_level = logging.DEBUG if verbose else getattr(
        logging, CONFIG.logging.log_level, logging.INFO
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(CONFIG.logging.log_format)

    # File handler
    log_file = CONFIG.paths.logs_dir / "ct_qc_platform.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=CONFIG.logging.max_bytes,
        backupCount=CONFIG.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    if CONFIG.logging.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def _log_startup_banner(args, start_slice, end_slice, output_dir) -> None:
    """Log platform identification and resolved configuration."""
    logger.info("=" * 51)
    logger.info(" CT QC Platform v%s", __version__)
    logger.info(" Mode        : %s", args.mode)
    logger.info(" DICOM Dir   : %s", args.dicom_dir)
    logger.info(" Scanner ID  : %s", args.scanner_id)
    logger.info(" Slice Range : %d -> %d", start_slice, end_slice)
    logger.info(" Output Dir  : %s", output_dir)
    logger.info(" Dry Run     : %s", args.dry_run)
    logger.info(" Timestamp   : %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 51)


def _build_default_rois() -> list[ROIDescriptor]:
    """Return the standard 5-ROI layout for a 512×512 CT phantom."""
    return [
        ROIDescriptor("center_water", 226, 226, 60, 60),
        ROIDescriptor("peripheral_12", 80, 236, 40, 40),
        ROIDescriptor("peripheral_3", 236, 390, 40, 40),
        ROIDescriptor("peripheral_6", 390, 236, 40, 40),
        ROIDescriptor("peripheral_9", 236, 80, 40, 40),
    ]


def _load_phantom_config() -> dict:
    """Load phantom_config.json from the project root.

    Returns the parsed dict. If the file is absent, logs WARNING and
    returns a default dict with active_phantom='catphan_504' and
    active_scanner='generic'.
    """
    config_path = Path(__file__).parent / "phantom_config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        logger.debug("Loaded phantom configuration from %s", config_path)
        return cfg
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning(
            "phantom_config.json not found or invalid (%s) — using defaults", exc
        )
        return {
            "active_phantom": "catphan_504",
            "active_scanner": "generic",
        }


def _build_helios_rois(dicom_dir: Path, start_slice: int, end_slice: int) -> list[ROIDescriptor]:
    """
    Builds ROI descriptors for the GE Helios phantom using automatic geometry
    detection. Called when --phantom ge_helios is active.

    Loads the middle slice of the selected range, runs HeliosPhantomDetector,
    and returns the detected ROIDescriptor list.
    Falls back to default 5-ROI layout if detection quality is 'failed'.
    """
    from modules.image_qc.phantom_geometry import HeliosPhantomDetector
    try:
        loader   = DicomLoader(CONFIG.dicom)
        mid      = (start_slice + end_slice) // 2
        datasets = loader.load_slice_range(dicom_dir, mid, mid)
        if not datasets:
            raise ValueError("No slices loaded for geometry detection")
        hu_array = loader.to_hu_array(datasets[0])
        spacing  = loader.get_pixel_spacing_mm(datasets[0])
        detector = HeliosPhantomDetector(CONFIG)
        geometry = detector.detect(hu_array, spacing)
        logger.info(
            "Helios geometry detected: center=(%.1f, %.1f) radius=%.1f px quality=%s",
            geometry.center_row, geometry.center_col,
            geometry.phantom_radius_px, geometry.detection_quality
        )
        return list(geometry.roi_descriptors.values())
    except Exception as exc:
        logger.warning(
            "Helios geometry detection failed (%s) — using default 5-ROI layout", exc
        )
        return _build_default_rois()


def _log_image_qc_summary(result) -> None:
    """Log Image QC summary."""
    logger.info("-- Image QC Summary --")
    if result.volumetric_result is not None:
        vol = result.volumetric_result
        for label, stat in vol.volumetric_stats.items():
            status = "PASS" if stat.passes_tg66 else "FAIL"
            logger.info(
                "  ROI %-20s: mean=%+7.2f HU, SD=%.2f HU [%s]",
                label, stat.mean_hu_mean, stat.std_hu_mean, status,
            )
    if result.nps_result is not None:
        logger.info(
            "  NPS peak: %.4f lp/mm, noise SD: %.2f HU²mm²",
            result.nps_result.peak_frequency_lpmm,
            result.nps_result.integral_hu2mm2,
        )
    if result.mtf_result is not None:
        logger.info("  MTF50: %.3f lp/mm", result.mtf_result.mtf_50_lpmm)
    logger.info("  All passed: %s", result.all_passed)


def _log_predictive_summary(result) -> None:
    """Log Predictive Maintenance summary."""
    logger.info("-- Predictive Maintenance Summary --")
    if result.maintenance_alert is not None:
        alert = result.maintenance_alert
        logger.info("  Overall urgency: %s", alert.overall_urgency)
        logger.info("  Recommended action: %s", alert.recommended_action[:100])
        if alert.recommended_maintenance_date:
            logger.info("  Recommended date: %s", alert.recommended_maintenance_date)
        for pred in alert.predictions:
            logger.info(
                "  %-35s: slope=%+.6f/day, breach=%s, urgency=%s",
                pred.metric_name, pred.slope_per_day,
                pred.predicted_breach_date or "N/A",
                pred.get_urgency_level(),
            )


def _log_full_pipeline_summary(result) -> None:
    """Log full pipeline summary."""
    logger.info("=" * 51)
    logger.info(" FULL PIPELINE SUMMARY")
    logger.info("=" * 51)
    if result.image_qc is not None:
        _log_image_qc_summary(result.image_qc)
    if result.predictive is not None:
        _log_predictive_summary(result.predictive)
    if result.errors:
        logger.warning("  Errors: %d", len(result.errors))
        for err in result.errors:
            logger.warning("    - %s", err)
    logger.info("  Duration: %.1f seconds", result.run_duration_seconds)
    logger.info("=" * 51)


# ═══════════════════════════════════════════════════════════════════
# Main Function
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    """Parse arguments, initialize platform, dispatch to selected mode."""
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve defaults
    start_slice = args.start_slice or CONFIG.image_qc.default_start_slice
    end_slice = args.end_slice or CONFIG.image_qc.default_end_slice
    output_dir = args.output_dir or CONFIG.paths.outputs_dir
    archive_path = args.archive_path or CONFIG.paths.archive_file

    # Initialize logging
    _initialize_logging(verbose=args.verbose, output_dir=output_dir)

    # Create output directories
    for d in [CONFIG.paths.plots_dir, CONFIG.paths.reports_dir, CONFIG.paths.logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Startup banner
    _log_startup_banner(args, start_slice, end_slice, output_dir)

    # Load phantom/scanner profile
    phantom_cfg = _load_phantom_config()
    active_phantom  = args.phantom  or phantom_cfg.get("active_phantom",  "catphan_504")
    active_scanner  = args.scanner  or phantom_cfg.get("active_scanner",  "generic")
    logger.info("Phantom profile : %s", active_phantom)
    logger.info("Scanner profile : %s", active_scanner)

    # Validate dicom-dir requirement
    if args.mode != "predictive" and args.dicom_dir is None:
        logger.error("--dicom-dir is required for mode '%s'", args.mode)
        return 1
    if args.dicom_dir is not None and not args.dicom_dir.is_dir():
        logger.error("DICOM directory does not exist: %s", args.dicom_dir)
        return 1

    # ROI layout: automatic for Helios, manual default otherwise
    if active_phantom == "ge_helios" and args.dicom_dir is not None:
        rois = _build_helios_rois(args.dicom_dir, start_slice, end_slice)
    else:
        rois = _build_default_rois()

    # Instantiate infrastructure
    dicom_loader = DicomLoader(CONFIG.dicom)
    archive = MetricsArchive(archive_path)
    orchestrator = PipelineOrchestrator(CONFIG, dicom_loader, archive)

    start_time = time.monotonic()
    exit_code = 0

    try:
        if args.mode == "qc":
            result = orchestrator.run_image_qc(
                args.dicom_dir, rois, start_slice, end_slice,
            )
            _log_image_qc_summary(result)
            exit_code = 2 if not result.all_passed else 0

        elif args.mode == "dosimetry":
            result = orchestrator.run_dosimetry(args.dicom_dir)
            if result.dosimetry_report:
                result.dosimetry_report.print_summary()
            exit_code = 0

        elif args.mode == "predictive":
            result = orchestrator.run_predictive(args.scanner_id)
            _log_predictive_summary(result)
            exit_code = 0

        elif args.mode == "all":
            result = orchestrator.run_full_pipeline(
                dicom_dir=args.dicom_dir,
                rois=rois,
                start_slice=start_slice,
                end_slice=end_slice,
                scanner_id=args.scanner_id,
                operator_id=args.operator_id,
                notes=args.notes,
                skip_modules=args.skip,
                dry_run=args.dry_run,
            )
            if not args.dry_run and result.session_record is not None:
                archive.append_session(result.session_record)
                if args.export_csv:
                    csv_path = output_dir / "qc_archive_export.csv"
                    archive.export_to_csv(csv_path)
            _log_full_pipeline_summary(result)
            # Exit codes: 0=success, 1=error, 2=QC breach
            if result.errors:
                exit_code = 1
            elif result.image_qc and not result.image_qc.all_passed:
                exit_code = 2

    except KeyboardInterrupt:
        logger.info("Run cancelled by user (KeyboardInterrupt)")
        return 130
    except Exception as exc:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return 1

    duration = time.monotonic() - start_time
    logger.info("=== Run complete in %.1f seconds ===", duration)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
