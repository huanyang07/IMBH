#!/usr/bin/env python3
"""Freeze the seven-field cellwise fixed-slow invariant-object search."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

import run_causal_inner_entropy_complete_corrected_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizeo as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizep_"
    "entropy_complete_fixed_Q_invariant_object_manifest"
)
CLASSIFICATION = "entropy_complete_cellwise_fixed_Q_invariant_object_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizeq_"
    "entropy_complete_fixed_Q_invariant_object_implementation"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_invariant_object_manifest_"
    "wp10c9d6c7c3b5c4f25fizep"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "INVARIANT_OBJECT_MANIFEST_WP10C9D6C7C3B5C4F25FIZEP_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_fixed_q_invariant_object_manifest_wp10c9d6c7c3b5c4f25fizep.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_fixed_q_invariant_object_manifest_wp10c9d6c7c3b5c4f25fizep.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "5b70cd489b0f729aaa88ee7ce9dee27e2bda9054219f6802a1edfa4e5d14c4b5"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
SLOW_EXACT_ROWS = (0, 2, 3)
FAST_EQUATION_ROWS = (1, 4, 5, 6)
SLOW_CHART_INDICES = (0, 2, 3)
FAST_CHART_INDICES = (1, 4, 5, 6)
FAST_CHART_SCALES = (0.1, 1.0e-4, 1.0, 0.03)


def _utils():
    return parent._utils()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("corrected radial crossing checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "execution_metrics.json")
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["bounded_radial_crossing_certified"]
        or not summary["crossed_old_rejected_time"]
        or not summary["first_endpoint_matches_diagnostic_bitwise"]
        or not summary["suffix_replay_bitwise"]
        or not summary["fixed_Q_invariant_object_manifest_authorized"]
        or summary["fixed_Q_invariant_object_execution_authorized"]
        or summary["authorized_next"] != (
            "definitions_only_WP10c9d6c7c3b5c4f25fizep_"
            "entropy_complete_fixed_Q_invariant_object_manifest"
        )
        or metrics["first_failure"] is not None
    ):
        raise RuntimeError("corrected radial crossing authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"corrected radial source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("fixed-Q invariant-object manifest requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "objective": "find_a_normally_attracting_cellwise_fixed_Q_fast_graph_for_the_seven_field_balance_law",
        "mathematical_split": {
            "primitive_chart_dimension_per_cell": 7,
            "slow_exact_rows": list(SLOW_EXACT_ROWS),
            "slow_exact_fields": ["column_mass", "azimuthal_angular_momentum", "total_energy"],
            "fast_equation_rows": list(FAST_EQUATION_ROWS),
            "fast_fields": ["radial_momentum", "causal_shear", "column_height", "vertical_momentum"],
            "slow_chart_indices_reconstructed": list(SLOW_CHART_INDICES),
            "fast_chart_indices_independent": list(FAST_CHART_INDICES),
            "fast_chart_scales": list(FAST_CHART_SCALES),
            "slow_dimension": 336,
            "fast_dimension": 448,
            "identity": "784_equals_336_plus_448",
            "fixed_Q_means_cellwise_exact_M_J_E_not_only_three_global_ledgers": True,
            "legacy_global_fixed_Q_reaction_used": False,
        },
        "fixed_slow_reconstruction": {
            "constraints": "cell_integrated_exact_rows_0_2_3_equal_frozen_targets",
            "unknown_local_charts": ["logSigma", "azimuthal_velocity", "logPi"],
            "solver": "damped_local_three_by_three_Newton",
            "maximum_constraint_relative_defect": 1.0e-11,
            "maximum_Newton_corrections": 8,
            "accepted_history_only": True,
        },
        "fast_vector_field": {
            "constraints": "D_Q_times_primitive_rate_equals_zero",
            "dynamic_rows": list(FAST_EQUATION_ROWS),
            "local_projected_temporal_solve": "stack_DQ_rows_with_fast_temporal_rows",
            "stationary_condition": "projected_fast_chart_rate_equals_zero_equivalently_fast_RHS_equals_zero",
            "slow_drift_output": "C_times_cell_integrated_RHS_rows_0_2_3",
            "same_path_conservative_radial_operator": True,
            "same_boundaries_and_lower_sources": True,
        },
        "states": {
            "primary": "hash_locked_primary_20ms_base_profile_hydrostatic_lift",
            "heldout": "hash_locked_heldout_16ms_base_profile_hydrostatic_lift",
            "corrected_terminal_is_dynamic_validation_not_a_root_training_target": True,
            "execution_order": ["primary", "heldout"],
            "stop_on_primary_failure": True,
        },
        "sparse_derivative": {
            "radial_stencil_radius": 1,
            "cell_coloring_count": 3,
            "fast_field_count": 4,
            "forward_colored_residual_evaluations_per_assembly": 12,
            "independent_central_directional_audits": 4,
            "maximum_colored_JVP_relative_defect": 2.0e-5,
            "maximum_off_stencil_absolute_entry": 1.0e-10,
        },
        "nonlinear_solver": {
            "method": "one_colored_Jacobian_then_dense_Broyden_with_bound_aware_line_search",
            "maximum_iterations": 10,
            "maximum_colored_Jacobian_assemblies": 2,
            "one_refresh_only_after_complete_line_search_failure": True,
            "maximum_scaled_fast_chart_correction_per_iteration": 0.25,
            "maximum_normalized_fast_rate_infinity": 1.0e-8,
            "failed_candidate_must_not_define_an_invariant_object": True,
        },
        "normal_hyperbolicity": {
            "operator": "fixed_Q_projected_fast_rate_Jacobian_per_second",
            "maximum_spectral_abscissa_per_second": -1.0,
            "minimum_attraction_to_slow_relative_rate_ratio": 10.0,
            "all_eigenvalues_finite": True,
            "normally_attracting_fixed_point_required": True,
            "periodic_or_statistical_invariant_measure_not_authorized_in_this_package": True,
        },
        "binding_physical_gates": {
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "eigenvector_condition_number_max": 1.0e8,
            "minimum_height_over_radius": 1.0e-4,
            "maximum_height_over_radius": 0.5,
            "minimum_optical_depth": 1.0,
            "maximum_temporal_projection_solve_relative_defect": 1.0e-10,
            "inner_incoming_characteristics_equal": 0,
            "fail_closed": True,
        },
        "decision": {
            "both_states_normally_attracting": "authorize_definitions_only_slow_flux_atlas_manifest",
            "root_exists_but_not_attracting": "authorize_only_invariant_measure_diagnosis_manifest",
            "root_or_physical_gate_failure": "stop_without_slow_atlas",
        },
        "claim_boundary": {
            "implementation_authorized": True,
            "nonlinear_root_execution_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("fixed-Q invariant-object manifest already exists")
    utils = _utils(); validated = _validate_parent(require_clean=True); CANONICAL_DIRECTORY.mkdir(parents=True)
    with np.load(parent.CANONICAL_DIRECTORY / "execution_arrays.npz", allow_pickle=False) as archive:
        terminal_hash = _array_sha256(archive["trajectory_charts7"][-1])
    utils._write_json(CANONICAL_DIRECTORY / "invariant_object_contract.json", _contract()); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "parent_terminal_charts_sha256": terminal_hash})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "corrected_bounded_crossing_preserved": True, "cellwise_fixed_Q_split_frozen": True, "implementation_authorized": True, "nonlinear_root_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete fixed-Q invariant-object manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The slow variables are the cellwise exact mass, angular-momentum, and total-energy states. Radial momentum, causal shear, height, and vertical momentum form a constrained 448-dimensional fast subsystem. The legacy three-global-ledger reaction is not the mathematical fixed-Q object used here.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
