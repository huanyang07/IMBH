#!/usr/bin/env python3
"""Freeze the equation-form primary fixed-Q root preflight."""

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

import run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizer_"
    "entropy_complete_fixed_Q_primary_root_execution_manifest"
)
CLASSIFICATION = "entropy_complete_fixed_Q_primary_root_preflight_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizes_"
    "entropy_complete_fixed_Q_equation_form_root_preflight"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_primary_root_execution_manifest_"
    "wp10c9d6c7c3b5c4f25fizer"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "PRIMARY_ROOT_EXECUTION_MANIFEST_WP10C9D6C7C3B5C4F25FIZER_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_fixed_q_primary_root_"
    "execution_manifest_wp10c9d6c7c3b5c4f25fizer.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_fixed_q_primary_root_"
    "execution_manifest_wp10c9d6c7c3b5c4f25fizer.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "be306c444fdde78fd414f23401613219c0841fabe19ee5984b0f75a7d8e87db4"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("fixed-Q projected-field certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "implementation_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["projected_fast_field_certified"]
        or not summary["colored_fast_Jacobian_certified"]
        or not summary["primary_root_execution_manifest_authorized"]
        or summary["primary_root_execution_authorized"]
        or summary["authorized_next"] != f"definitions_only_{WORK_PACKAGE}"
        or metrics["maximum_colored_JVP_relative_defect"] > 2.0e-5
        or metrics["new_nonlinear_roots"] != 0
        or metrics["propagated_states"] != 0
    ):
        raise RuntimeError("fixed-Q primary-root manifest authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"fixed-Q projected-field source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("primary-root manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "objective": (
            "certify_a_conditioned_equation_form_linearization_before_any_"
            "448_variable_primary_fixed_Q_root_is_executed"
        ),
        "preserved_science": {
            "cellwise_slow_exact_rows": [0, 2, 3],
            "fast_equation_rows": [1, 4, 5, 6],
            "slow_chart_indices": [0, 2, 3],
            "fast_chart_indices": [1, 4, 5, 6],
            "fast_chart_scales": [0.1, 1.0e-4, 1.0, 0.03],
            "slow_chart_names": [
                "log_surface_density",
                "azimuthal_velocity_over_c",
                "log_temperature",
            ],
            "parent_logPi_label_is_metadata_only_and_is_corrected_here": True,
            "legacy_three_global_ledger_reaction_used": False,
            "same_entropy_complete_path_conservative_radial_operator": True,
            "same_primary_20ms_hydrostatic_lift": True,
        },
        "root_equivalence": {
            "binding_stationary_equations": (
                "radial_operator_equation_RHS_rows_1_4_5_6_equal_zero"
            ),
            "projected_fast_rate_zero_is_mathematically_equivalent": True,
            "projected_rate_is_independent_post_evaluation_parity_audit": True,
            "maximum_equation_rate_parity_relative_defect": 1.0e-10,
        },
        "equation_row_scaling": {
            "coordinate": "x_equals_fast_chart_minus_base_divided_by_fast_chart_scale",
            "projected_chart_tangent": (
                "P_fast_equals_I_and_P_slow_equals_minus_Tss_inverse_Tsf"
            ),
            "projected_fast_temporal_block": "A_equals_T_fast_rows_times_P",
            "one_second_equation_scale_per_cm": (
                "max_column_abs_A_times_fast_chart_scale_divided_by_c"
            ),
            "floor_relative_to_global_maximum": 1.0e-12,
            "scales_frozen_at_hash_locked_primary_base": True,
            "base_projected_rate_magnitude_is_not_a_row_scale": True,
        },
        "linearization_preflight": {
            "radial_stencil_radius": 1,
            "cell_colors": 3,
            "fast_fields": 4,
            "forward_colored_equation_evaluations": 12,
            "forward_relative_step": 1.0e-6,
            "independent_central_JVP_directions": 4,
            "central_relative_step": 2.0e-6,
            "maximum_colored_JVP_relative_defect": 2.0e-5,
            "all_singular_values_and_numerical_ranks_recorded": True,
            "condition_number_is_diagnostic_not_a_pass_by_itself": True,
            "no_nonlinear_candidate_or_state_propagation": True,
        },
        "prospective_linear_step": {
            "problem": "minimize_norm2_Js_plus_F_subject_to_abs_s_le_0p25",
            "solver": "scipy_optimize_lsq_linear_trf_exact_dense",
            "maximum_scaled_fast_chart_correction": 0.25,
            "maximum_predicted_infinity_merit_ratio": 0.95,
            "maximum_predicted_two_norm_merit_ratio": 0.90,
            "step_and_active_bounds_recorded": True,
        },
        "future_nonlinear_policy_if_preflight_passes": {
            "maximum_iterations": 12,
            "maximum_colored_Jacobian_assemblies": 2,
            "initial_exact_colored_assembly": True,
            "one_refresh_trigger": [
                "complete_scaled_infinity_merit_line_search_failure",
                "beginning_of_iteration_maximum_iterations_minus_2",
            ],
            "refresh_trigger_supersedes_parent_generic_failure_only_trigger": True,
            "reason": (
                "prospectively_preserve_two_post_refresh_corrections_and_avoid_"
                "the_already_certified_stale_matrix_iteration_exhaustion_mode"
            ),
            "line_search_reductions": 12,
            "maximum_scaled_equation_residual_infinity": 1.0e-8,
            "maximum_physical_fast_coordinate_rate_per_second": 1.0e-8,
            "failed_candidate_must_not_define_an_invariant_object": True,
        },
        "future_root_attraction_audit": {
            "physical_coordinate_rate": (
                "projected_fast_rate_per_second_divided_by_fast_chart_scale"
            ),
            "physical_tangent": (
                "derivative_of_physical_coordinate_rate_with_respect_to_x"
            ),
            "solver_row_normalized_Jacobian_eigenvalues_are_not_physical": True,
            "maximum_spectral_abscissa_per_second": -1.0,
            "slow_relative_rate": (
                "max_abs_cell_integrated_slow_drift_divided_by_abs_slow_target"
            ),
            "minimum_attraction_to_slow_relative_rate_ratio": 10.0,
            "fresh_colored_tangent_and_independent_JVP_audit_required": True,
        },
        "binding_physical_gates": {
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "eigenvector_condition_number_max": 1.0e8,
            "minimum_height_over_radius": 1.0e-4,
            "maximum_height_over_radius": 0.5,
            "minimum_optical_depth": 1.0,
            "maximum_fixed_slow_reconstruction_relative_defect": 1.0e-11,
            "maximum_temporal_projection_solve_relative_defect": 1.0e-10,
            "inner_incoming_characteristics_equal": 0,
            "fail_closed": True,
        },
        "decision": {
            "preflight_pass": (
                "authorize_definitions_only_primary_nonlinear_root_execution_manifest"
            ),
            "derivative_or_trust_step_failure": (
                "stop_and_redesign_equation_scaling_or_structured_solver"
            ),
            "no_threshold_may_be_relaxed_after_observing_results": True,
        },
        "claim_boundary": {
            "equation_form_preflight_authorized": True,
            "primary_nonlinear_root_execution_authorized": False,
            "heldout_root_execution_authorized": False,
            "slow_flux_atlas_authorized": False,
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
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utils._sha256(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("primary-root execution manifest already exists")
    utils = _utils()
    validated = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    contract = _contract()
    utils._write_json(CANONICAL_DIRECTORY / "primary_root_contract.json", contract)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "parent_maximum_colored_JVP_relative_defect": validated["metrics"][
                "maximum_colored_JVP_relative_defect"
            ],
            "parent_projected_field_calls": validated["metrics"][
                "projected_field_calls"
            ],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "projected_field_certificate_preserved": True,
        "equation_form_preflight_authorized": True,
        "primary_nonlinear_root_execution_authorized": False,
        "heldout_root_execution_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete fixed-Q primary-root execution manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The binding root is written in the fast equation rows, with row scales derived from the projected temporal block for one-second chart-scale motion. The base projected-rate magnitude is not used as a row scale. The attraction spectrum will be computed separately from the similarity-scaled physical fast-rate tangent.",
                "",
                "This package authorizes only an equation-form derivative and bounded-linear-step preflight. It executes no nonlinear root.",
                "",
                f"Authorized next: `{AUTHORIZED_NEXT}` only.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
