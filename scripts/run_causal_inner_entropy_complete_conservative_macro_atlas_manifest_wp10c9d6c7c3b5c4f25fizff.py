#!/usr/bin/env python3
"""Freeze the conservative 16-cell entropy-complete macro-atlas audit."""

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

import run_causal_inner_entropy_complete_hydrostatic_inverse_order_recovery_execution_wp10c9d6c7c3b5c4f25fizfe as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizff_"
    "entropy_complete_local_slow_flux_atlas_manifest"
)
CLASSIFICATION = "entropy_complete_conservative_16_cell_macro_atlas_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfg_"
    "entropy_complete_conservative_macro_atlas_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_conservative_macro_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25fizff"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_CONSERVATIVE_"
    "MACRO_ATLAS_MANIFEST_WP10C9D6C7C3B5C4F25FIZFF_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_conservative_macro_atlas_"
    "manifest_wp10c9d6c7c3b5c4f25fizff.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_conservative_macro_atlas_"
    "manifest_wp10c9d6c7c3b5c4f25fizff.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "7defd337da78a241b3e831f793a43339665853ddffb264d19171ac6035f9b987"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("implicit inverse certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "order_recovery_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["hydrostatic_implicit_inverse_tangent_certified"]
        or not summary["four_saved_truth_samples_certified_for_atlas_use"]
        or not summary["local_slow_flux_atlas_manifest_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["new_seven_field_operator_calls"] != 0
        or metrics["minimum_global_worst_raw_defect_order"] < 1.8
    ):
        raise RuntimeError("implicit inverse atlas authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"implicit inverse certificate source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("macro-atlas manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "selected_architecture": {
            "truth_radial_cells": 112,
            "online_radial_cells": 16,
            "fine_cells_per_macro_cell": 7,
            "macro_fields_per_cell": ("M", "J", "E", "beta_r", "chi"),
            "online_state_dimension": 80,
            "maximum_online_state_dimension": 86,
            "MJE_restriction": "exact_sum_of_seven_cell_integrals",
            "auxiliary_restriction": "mass_weighted_mean",
            "prolongation": "anchor_subcell_invariant_fractions_plus_uniform_mass_weighted_auxiliary_shift",
            "prolonged_truth_state": "cellwise_hydrostatic_invariant_reconstruction",
            "MJE_online_update": "single_valued_face_flux_difference_plus_cell_source",
            "auxiliary_online_update": "mass_weighted_exact_derivative_including_mass_weight_transport",
            "five_field_characteristic_pencil_used": False,
        },
        "atlas_construction": {
            "primary_training_anchor": "primary_20ms_base",
            "saved_primary_near_witness": "primary_20ms_perturbed",
            "strict_blind_profiles": (
                "heldout_16ms_base",
                "heldout_16ms_perturbed",
            ),
            "coordinate_scales": "cellwise_absolute_primary_macro_coordinates",
            "coordinate_scale_floor_fraction": 1.0e-12,
            "radius_one_block_stencil": True,
            "input_fields": 5,
            "cell_colors_per_field": 3,
            "central_colored_coordinate_step": 1.0e-2,
            "colored_truth_calls": 30,
            "independent_JVP_directions": 4,
            "independent_JVP_coordinate_step": 5.0e-3,
            "independent_JVP_truth_calls": 8,
            "maximum_new_truth_operator_calls": 38,
            "maximum_truth_reconstruction_relative_defect": 1.0e-10,
            "maximum_colored_JVP_relative_defect": 5.0e-2,
            "maximum_saved_near_witness_output_relative_defect": 1.0e-2,
            "maximum_blind_output_relative_defect_per_block": 5.0e-2,
            "maximum_blind_macro_rate_relative_defect_per_field": 5.0e-2,
            "maximum_blind_coordinate_infinity": 5.0e-2,
            "fail_on_any_truth_physical_or_hyperbolicity_gate": True,
        },
        "conservative_outputs": {
            "coarse_boundary_faces": 17,
            "MJE_face_flux_components": 3,
            "cell_source_components": ("J", "E"),
            "mass_cell_source_is_identically_zero": True,
            "auxiliary_rate_components": ("beta_r", "chi"),
            "conservative_ledger_relative_defect_max": 1.0e-12,
            "restriction_roundtrip_relative_defect_max": 1.0e-12,
        },
        "online_cost": {
            "truth_calls_per_macrostep": 0,
            "nonlinear_roots_per_macrostep": 0,
            "maximum_macrosteps_per_cycle": 100000,
            "benchmark_evaluations": 100000,
            "maximum_benchmark_wall_seconds": 10.0,
            "maximum_average_wall_seconds_per_macrostep": 1.0,
        },
        "decision": {
            "pass": "authorize_definitions_only_structure_preserving_macro_integrator_manifest",
            "fail": "reject_single_anchor_affine_atlas_and_require_additional_entropy_complete_truth_anchors",
        },
        "claim_boundary": {
            "offline_truth_sampling_authorized": True,
            "atlas_execution_authorized": True,
            "state_propagation_authorized": False,
            "macro_integrator_authorized": False,
            "complete_cycle_execution_authorized": False,
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
        raise RuntimeError("macro-atlas manifest already exists")
    validated = _validate_parent(require_clean=True); utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "macro_atlas_contract.json", _contract())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "hydrostatic_implicit_inverse_tangent_preserved": True,
        "conservative_16_cell_macro_atlas_selected": True,
        "offline_truth_sampling_authorized": True,
        "state_propagation_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"]})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete conservative macro-atlas manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The selected online state has 16 radial cells and five fields per cell. Fine-to-coarse M/J/E restriction is exact, the atlas predicts single-valued coarse-face fluxes plus sources, and beta_r/chi remain dynamic.", "", "Thirty-eight bounded offline seven-field calls recover and audit a radius-one colored tangent. The 16 ms base and perturbed profiles are strict blind validations. No state propagation is authorized.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
