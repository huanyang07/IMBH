#!/usr/bin/env python3
"""Freeze the first bounded seven-field radial crossing experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_semidiscrete_relaxation_audit_wp10c9d6c7c3b5c4f25fizei as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizej_"
    "entropy_complete_bounded_radial_crossing_manifest"
)
CLASSIFICATION = "entropy_complete_bounded_radial_crossing_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizek_"
    "entropy_complete_bounded_radial_crossing_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_bounded_radial_crossing_manifest_"
    "wp10c9d6c7c3b5c4f25fizej"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_BOUNDED_"
    "RADIAL_CROSSING_MANIFEST_WP10C9D6C7C3B5C4F25FIZEJ_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizej.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizej.py"
RADIAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_generalized_maxwell_cattaneo_radial.py"
RADIAL_TEST = "tests/test_causal_inner_generalized_maxwell_cattaneo_radial.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "0c7e0c3740750d0ec7a6b1174ae4b8bc3694e0a243c81d26f7d397c36def08e2"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
TIMESTEP_SECONDS = 6.25e-5
ACCEPTED_STEPS = 4
HORIZON_SECONDS = TIMESTEP_SECONDS * ACCEPTED_STEPS


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("semidiscrete relaxation certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "audit_metrics.json")
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["local_sources_certified"]
        or not summary["periodic_semidiscrete_operator_certified"]
        or not summary["hydrostatic_relaxation_limit_certified"]
        or not summary["bounded_crossing_manifest_authorized"]
        or summary["bounded_crossing_trajectory_authorized"]
        or summary["authorized_next"]
        != "definitions_only_WP10c9d6c7c3b5c4f25fizej_entropy_complete_bounded_radial_crossing_manifest"
        or metrics["first_failure"] is not None
    ):
        raise RuntimeError("semidiscrete relaxation authorization changed")
    for relative, expected in utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"semidiscrete audit source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("bounded crossing manifest requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "seed": {
            "profile": "accepted_terminal_base_charts5",
            "legacy_accepted_endpoints": 72,
            "legacy_elapsed_seconds": 0.18587500000000012,
            "seven_field_lift": "hydrostatic_height_and_zero_vertical_velocity",
            "rejected_legacy_candidate_not_used": True,
        },
        "radial_operator": {
            "spatial_order": 1,
            "reconstruction": "piecewise_constant",
            "interior_faces": "complete_DLM_signed_fluctuations_at_face_geometry",
            "face_measure_multiplies_every_fluctuation": True,
            "exact_rows": "one_shared_flux_difference",
            "shear_row": "Dplus_left_plus_Dminus_right",
            "cell_sources": "center_source_times_cell_measure_plus_stream_once",
            "inner_boundary": "outgoing_excision_trace_from_first_cell",
            "outer_boundary": "frozen_exterior_five_field_chart_lifted_hydrostatically",
            "inner_incoming_characteristics_required": 0,
            "accepted_history_only": True,
        },
        "time_integrator": {
            "method": "explicit_SSPRK2_in_seven_primitive_chart",
            "timestep_seconds": TIMESTEP_SECONDS,
            "accepted_steps": ACCEPTED_STEPS,
            "horizon_seconds": HORIZON_SECONDS,
            "maximum_CFL": 0.4,
            "stagewise_fail_closed_audits": True,
            "matched_control": "one_1.25e-4_s_step_vs_two_6.25e-5_s_steps",
            "checkpoint_each_endpoint": True,
            "full_suffix_bitwise_replay": True,
        },
        "binding_gates": {
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "eigenvector_condition_number_max": 1.0e8,
            "maximum_CFL": 0.4,
            "maximum_scaled_chart_change_per_step": 0.05,
            "minimum_height_over_radius": 1.0e-4,
            "maximum_height_over_radius": 0.5,
            "minimum_optical_depth": 1.0,
            "maximum_temporal_solve_relative_residual": 1.0e-10,
            "maximum_exact_flux_balance_relative_defect": 5.0e-5,
            "maximum_matched_endpoint_scaled_state_defect": 2.0e-3,
            "checkpoint_roundtrip_bitwise": True,
            "suffix_replay_bitwise": True,
            "all_stages_all_cells_all_faces_required": True,
            "fail_closed": True,
        },
        "claim_boundary": {
            "radial_operator_implementation_authorized": True,
            "bounded_crossing_execution_authorized": True,
            "maximum_new_trajectory_steps": ACCEPTED_STEPS,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decision": {
            "pass": "authorize_definitions_only_fixed_Q_invariant_object_manifest",
            "numerical_or_physical_failure": "stop_without_propagating_failed_candidate",
            "cost_is_diagnostic_not_binding_for_this_four_step_crossing": True,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("bounded crossing manifest already exists")
    utils = _utils(); validated = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "crossing_contract.json", _contract())
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "parent_metrics": validated["metrics"]})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "semidiscrete_relaxation_certificate_preserved": True,
        "radial_operator_implementation_authorized": True,
        "bounded_crossing_execution_authorized": True,
        "maximum_new_trajectory_steps": ACCEPTED_STEPS,
        "new_trajectory_steps": 0,
        "fixed_Q_invariant_object_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete bounded radial crossing manifest", "", f"Classification: `{CLASSIFICATION}`.", "", f"The sole authorized trajectory is four SSPRK2 steps of `{TIMESTEP_SECONDS}` s from the hash-locked accepted terminal legacy profile after hydrostatic seven-field lifting. The rejected legacy candidate is not used.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, parent.SOURCE, parent.SOURCE_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
