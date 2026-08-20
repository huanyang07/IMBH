#!/usr/bin/env python3
"""Freeze the short departure-28 reduced-vector-field validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_departure28_rate_validation_wp10c9d6c7c3b5c4f25bx as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25by"
PARENT_COMMIT = "05727ac26ad9b95515990aaa9ea52f4ebaa0438c"
PARENT_PARENT = "ba59ab16136dd890ebd569fd1ca3868055e05635"
PARENT_TREE = "04486b8d04a46c607535af64544d2e73b390b4f2"
CLASSIFICATION = "departure28_short_vector_field_validation_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25bz"

ARTIFACT = (
    "causal_inner_departure28_short_vector_field_manifest_"
    "wp10c9d6c7c3b5c4f25by"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_departure28_short_vector_field_manifest_"
    "wp10c9d6c7c3b5c4f25by.py"
)
THIS_TEST = (
    "tests/test_causal_inner_departure28_short_vector_field_manifest_"
    "wp10c9d6c7c3b5c4f25by.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_departure28_short_vector_field_validation_"
    "wp10c9d6c7c3b5c4f25bz.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_departure28_short_vector_field_validation_"
    "wp10c9d6c7c3b5c4f25bz.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DEPARTURE28_SHORT_VECTOR_FIELD_"
    "MANIFEST_WP10C9D6C7C3B5C4F25BY_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

RETRY_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
DESCRIPTOR_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c"
)
ONLINE_GEOMETRY_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_explicit_nonlinear_470_architecture_audit_"
    "wp10c9d6c7c3b5c4f25aw"
)

TIMESTEP_SECONDS = 1.0e-7
PHYSICAL_DIMENSION = 162
MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28
ONLINE_DIMENSION = PHYSICAL_DIMENSION + MEMORY_DIMENSION + DEPARTURE_DIMENSION
FULL_DIMENSION = 560


_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_sha = parent._sha
_checksums = parent._checksums


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("departure-28 validation commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("departure-28 validation lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("departure-28 validation tree changed")
    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.PASS_CLASSIFICATION
        or summary["authorized_next"]
        != "definitions_only_departure28_short_reduced_vector_field_validation_manifest"
        or summary["completed_nonbase_rate_evaluations"] != 48
        or summary["failed_rate_evaluations"] != 0
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or not all(metrics["model_checks"].values())
        or not all(metrics["truth_checks"].values())
    ):
        raise RuntimeError("departure-28 rate-validation authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"departure-28 source changed: {relative}")
    retry_hashes = _checksums(RETRY_DIRECTORY)
    retry_summary = _read(RETRY_DIRECTORY / "summary.json")
    warm_2 = _read(RETRY_DIRECTORY / "metrics_warm_2.json")
    warm_3 = _read(RETRY_DIRECTORY / "metrics_warm_3.json")
    if (
        retry_summary["accepted_main_BDF2_roots"] != 4
        or retry_summary["rejected_main_BDF2_roots"] != 0
        or not warm_2["accepted"]
        or not warm_3["accepted"]
        or not warm_2["checkpoint"]["bitwise_roundtrip"]
        or not warm_3["checkpoint"]["bitwise_roundtrip"]
        or warm_2["timestep_seconds"] != TIMESTEP_SECONDS
        or warm_3["timestep_seconds"] != TIMESTEP_SECONDS
    ):
        raise RuntimeError("accepted warm_2 to warm_3 reference interval changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("short-vector-field manifest requires a clean tracked tree")
    return {
        "parent_summary": summary,
        "parent_hashes": parent_hashes,
        "retry_summary": retry_summary,
        "retry_hashes": retry_hashes,
    }


def _decisive_inputs() -> dict[str, Path]:
    return {
        "departure28_closure": parent.CANONICAL_DIRECTORY
        / "departure28_closure.npz",
        "frozen_coefficients": parent.CANONICAL_DIRECTORY
        / "frozen_coefficients.npz",
        "coefficient_lock": parent.CANONICAL_DIRECTORY / "coefficient_lock.json",
        "online_470_geometry": ONLINE_GEOMETRY_DIRECTORY
        / "online_470_geometry.npz",
        "complete_generator": DESCRIPTOR_DIRECTORY / "descriptor_A.npz",
        "accepted_warm_2_checkpoint": RETRY_DIRECTORY / "checkpoint_warm_2.npz",
        "accepted_warm_3_checkpoint": RETRY_DIRECTORY / "checkpoint_warm_3.npz",
        "accepted_warm_2_metrics": RETRY_DIRECTORY / "metrics_warm_2.json",
        "accepted_warm_3_metrics": RETRY_DIRECTORY / "metrics_warm_3.json",
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "validate_one_short_zero_online_truth_forecast_of_the_frozen_"
            "departure28_reduced_vector_field_before_fast_slow_elimination"
        ),
        "architecture_role": {
            "model_470": "offline_fast_transient_and_closure_model",
            "final_cycle_integrator": "slow_physical_state_after_fast_elimination_or_averaging",
            "model_470_is_not_the_final_cycle_integrator": True,
            "direct_microsecond_marching_for_one_cycle": False,
        },
        "state": {
            "partition": "y_equals_q162_plus_z280_plus_a28",
            "dimension": ONLINE_DIMENSION,
            "q_definition": (
                "exact_nonlinear_160_integrated_mapped_storage_values_plus_"
                "2_explicit_stable_coordinates_relative_to_warm_3"
            ),
            "z_definition": "280_frozen_stable_memory_dual_coordinates",
            "a_definition": "28_frozen_departure_dual_coordinates",
            "anchor": "accepted_primary_warm_3_checkpoint",
            "anchor_elapsed_seconds": 0.0200006,
        },
        "algebraic_decoder": {
            "formula": "delta_hat_equals_L470_y_plus_B4_kappa8_of_E8_transpose_a",
            "rank4_curvature_decoder_frozen": True,
            "online_Newton_retractions_per_field_evaluation": 0,
            "decoder_coordinate_mismatch_is_measured_not_silently_corrected": True,
            "full_truth_state_dimension": FULL_DIMENSION,
        },
        "reduced_vector_field": {
            "decoded_full_rate": (
                "r_hat_equals_r0_plus_G_delta_hat_plus_D28_N28_of_a"
            ),
            "physical_rate": "q_dot_equals_D_Cphys_at_u_hat_times_r_hat",
            "memory_rate": "z_dot_equals_Z280_transpose_times_r_hat",
            "departure_rate": "a_dot_equals_D28_transpose_times_r_hat",
            "departure_nonlinearity": (
                "N28_equals_frozen_dual_quadratic_plus_cubic_closure_with_N28_of_0_equal_0"
            ),
            "online_truth_calls_per_field_evaluation": 0,
            "online_full_generator_assemblies": 0,
            "online_nonlinear_fixed_Q_roots": 0,
            "state_local_coordinate_Jacobian_is_geometry_not_PDE_truth": True,
        },
        "integration": {
            "method": "classical_RK4",
            "coarse_substeps": 1,
            "refined_substeps": 2,
            "forecast_timestep_seconds": TIMESTEP_SECONDS,
            "binding_forecast": "refined_RK4",
            "integration_error_measure": "coarse_to_refined_endpoint_difference",
        },
        "reference_sequence": {
            "readiness_interval": "accepted_warm_2_to_accepted_warm_3",
            "readiness_is_retrospective_and_cannot_by_itself_certify_forecasting": True,
            "prospective_forecast": "accepted_warm_3_to_new_warm_4",
            "forecast_must_be_serialized_and_hashed_before_truth_root": True,
            "truth_root": "one_authentic_equal_step_fixed_Q_BDF2_root_at_1e-7_seconds",
            "truth_predictor": "accepted_history_predictor_not_reduced_forecast",
            "truth_may_not_change_any_model_coefficient_threshold_or_forecast": True,
            "rejected_truth_candidate_must_not_enter_history": True,
        },
        "binding_structural_gates": {
            "anchor_state_array_equal": True,
            "anchor_base_rate_array_equal": True,
            "restriction_lifting_identity_defect_max": 1.0e-10,
            "base_vector_field_relative_identity_defect_max": 1.0e-10,
            "nonlinear_departure_at_zero_norm_max": 0.0,
            "coefficient_hash_frozen_before_forecast": True,
            "forecast_hash_frozen_before_truth": True,
        },
        "binding_retrospective_readiness_gates": {
            "decoded_start_full_scaled_state_relative_error_max": 0.03,
            "refined_endpoint_full_coordinate_relative_error_max": 0.02,
            "refined_endpoint_q162_relative_error_max": 0.05,
            "refined_endpoint_z280_relative_error_max": 0.02,
            "refined_endpoint_a28_relative_error_max": 0.02,
            "coarse_refined_endpoint_relative_difference_max": 1.0e-3,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
        },
        "binding_prospective_forecast_gates": {
            "truth_root_accepted": True,
            "truth_root_maximum_scaled_residual": 1.0e-10,
            "truth_root_maximum_Q3_relative_defect": 1.0e-12,
            "truth_root_minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "truth_root_maximum_H_over_R": 0.12,
            "truth_root_minimum_scattering_optical_depth": 1.0,
            "truth_root_maximum_exact_Jacobian_assemblies": 1,
            "forecast_full_coordinate_relative_error_max": 0.05,
            "forecast_q162_relative_error_max": 0.05,
            "forecast_z280_relative_error_max": 0.05,
            "forecast_a28_relative_error_max": 0.05,
            "forecast_full_scaled_state_relative_error_max": 0.05,
            "endpoint_vector_field_full_relative_error_max": 0.15,
            "endpoint_vector_field_q162_relative_error_max": 0.15,
            "endpoint_vector_field_z280_relative_error_max": 0.15,
            "endpoint_vector_field_a28_relative_error_max": 0.08,
            "coarse_refined_endpoint_relative_difference_max": 1.0e-3,
        },
        "decision": {
            "pass_classification": "departure28_short_reduced_vector_field_validated",
            "failure_classification": "departure28_short_reduced_vector_field_validation_failed",
            "pass_authorizes_only": (
                "definitions_only_fixed_Q_fast_attractor_and_normal_hyperbolicity_manifest"
            ),
            "heldout_anchor_forecast_authorized": False,
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("short-vector-field manifest already canonicalized")
    inputs = _decisive_inputs()
    for path in inputs.values():
        if not path.is_file():
            raise RuntimeError(f"decisive input is unavailable: {path}")
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "decisive_input_hashes": {
                name: _sha(path) for name, path in inputs.items()
            },
            "departure28_parent_hashes": frozen["parent_hashes"],
            "primary_retry_hashes": frozen["retry_hashes"],
            "primary_retry_classification_preserved": frozen["retry_summary"][
                "classification"
            ],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "anchor": "accepted_primary_warm_3",
        "retrospective_reference_interval": "accepted_warm_2_to_warm_3",
        "prospective_truth_roots_authorized": 1,
        "model_470_is_final_cycle_integrator": False,
        "fast_attractor_manifest_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                parent.THIS_RUNNER: _sha(ROOT / parent.THIS_RUNNER),
                parent.THIS_TEST: _sha(ROOT / parent.THIS_TEST),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Departure-28 short-vector-field manifest WP10c9d6c7c3b5c4f25by",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The geometry, affine generator, exact base rate, and departure closure are locked to the same accepted `warm_3` anchor.",
                "",
                "The 470-state model is an offline fast/transient closure model. It is explicitly not the final cycle integrator.",
                "",
                "Validation first uses the accepted `warm_2 -> warm_3` interval as a retrospective readiness gate. A refined RK4 forecast from `warm_3` is then frozen before one new authentic BDF2 `warm_4` truth root is evaluated.",
                "",
                "A pass authorizes only a definitions-only fixed-Q fast-attractor and normal-hyperbolicity manifest. It does not authorize a microburst, cycle prediction, or reduced slow evolution.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
