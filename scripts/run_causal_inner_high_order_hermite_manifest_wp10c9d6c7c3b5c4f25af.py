#!/usr/bin/env python3
"""Freeze the high-order square-root Hermite confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_relative_hermite_resolvent_manifest_wp10c9d6c7c3b5c4f25ad as prior_manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25af"
CLASSIFICATION = (
    "higher_order_square_root_Hermite_confirmation_manifest_frozen_"
    "two_anchor_unseen_eighth_grid_audit_authorized"
)
PARENT_COMMIT = "6bb21867ae0ffe283c348e63f7e3a128614ccc98"
PARENT_PARENT = "e2f973dbb17fb52dcafb7d6c18e4a827b7874c23"
PARENT_TREE = "e9547e4f5ff5a67100ba3202176f5eb06ad4ac5a"

PARENT_ARTIFACT = (
    "causal_inner_relative_hermite_resolvent_audit_"
    "wp10c9d6c7c3b5c4f25ae"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
FIBER_DIRECTORY = prior_manifest.FIBER_DIRECTORY
R32_DIRECTORY = prior_manifest.R32_DIRECTORY
PRIMARY_GENERATOR_DIRECTORY = prior_manifest.PRIMARY_GENERATOR_DIRECTORY
CROSS_ANCHOR_DIRECTORY = prior_manifest.CROSS_ANCHOR_DIRECTORY

ARTIFACT = (
    "causal_inner_high_order_hermite_manifest_"
    "wp10c9d6c7c3b5c4f25af"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_high_order_hermite_manifest_"
    "wp10c9d6c7c3b5c4f25af.py"
)
THIS_TEST = (
    "tests/test_causal_inner_high_order_hermite_manifest_"
    "wp10c9d6c7c3b5c4f25af.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_high_order_hermite_audit_"
    "wp10c9d6c7c3b5c4f25ag.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_high_order_hermite_audit_"
    "wp10c9d6c7c3b5c4f25ag.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HIGH_ORDER_HERMITE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AF_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PHYSICAL_DIMENSION = 162
EXACT_NONSTABLE_DIMENSION = 28
HIDDEN_DIMENSION = 370
HIDDEN_ORDERS = (280, 288, 296, 300, 304, 320)
MAXIMUM_ONLINE_DIMENSION = 510
NONSTABLE_THRESHOLD_PER_SECOND = -1.0e-8


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent() -> tuple[dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("high-order Hermite parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("high-order Hermite parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("high-order Hermite parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "relative_Hermite_resolvent_reduction_failed_within_R320_tangential_residual_greedy_reassessment_required"
        or summary["best_hidden_order"] != 120
        or summary["authorized_next"] is not None
        or summary["physical_failure_detected"]
    ):
        raise RuntimeError("relative-Hermite rejection changed")
    for directory in (
        FIBER_DIRECTORY,
        R32_DIRECTORY,
        PRIMARY_GENERATOR_DIRECTORY,
        CROSS_ANCHOR_DIRECTORY,
    ):
        _checksums(directory)
    return summary, hashes


def _transfer_gates() -> dict:
    return {
        "maximum_normalized_dynamic_transfer_relative_error_max": 0.25,
        "RMS_normalized_dynamic_transfer_relative_error_max": 0.10,
        "DC_normalized_dynamic_transfer_relative_error_max": 0.10,
        "maximum_normalized_total_transfer_relative_error_max": 0.25,
        "RMS_normalized_total_transfer_relative_error_max": 0.10,
        "DC_normalized_total_transfer_relative_error_max": 0.10,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "exploratory_results_are_nonbinding": True,
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in (
                "summary.json",
                "metrics.json",
                "candidate_errors.npz",
                "decisive_model.npz",
            )
        },
        "saved_input_hashes": {
            "primary_generator": _sha(
                PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz"
            ),
            "primary_output": _sha(PRIMARY_GENERATOR_DIRECTORY / "projection.npz"),
            "heldout_generator_and_output": _sha(
                CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz"
            ),
            "R32_projection": _sha(
                R32_DIRECTORY / "R32_projection_promotion.npz"
            ),
            "frequency_ladder": _sha(R32_DIRECTORY / "R32_transfer.npz"),
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_new_truth_anchors": 0,
            "maximum_wall_hours": 1.0,
            "candidate_hidden_orders": list(HIDDEN_ORDERS),
            "fail_fast_after_first_joint_pass": True,
        },
        "frequency_split": {
            "parent_intervals": 32,
            "training_interval_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
            "training_frequency_count": 129,
            "validation_interval_fractions": [0.125, 0.375, 0.625, 0.875],
            "validation_frequency_count_including_shared_DC": 129,
            "zero_to_first_frequency_interpolation": "linear",
            "positive_frequency_interpolation": "geometric",
            "shared_DC_training_control": True,
            "previously_inspected_original_midpoint_and_quarter_sets_promoted_to_training": True,
            "non_DC_validation_frequencies_evaluated_before_freeze": False,
            "validation_information_may_influence_basis": False,
        },
        "exact_architecture": {
            "nonstable_fiber": "all_28_effective_nonstable_modes_retained_exactly",
            "stable_coordinates": "upper_Cholesky_Lyapunov_whitened",
            "conservative_trial": "minimum_Euclidean_norm_right_inverse_of_Chat",
            "hidden_constraint": "Z_transpose_Z_equals_identity_and_Chat_Z_equals_zero",
            "trial": "Vhat_equals_horizontal_concatenation_Lhat_Z",
            "test": "What_equals_horizontal_concatenation_Chat_transpose_Z",
            "stability_identity": "G_Ar_plus_Ar_transpose_G_equals_Vhat_transpose_Ahat_plus_Ahat_transpose_Vhat_strictly_negative",
        },
        "high_order_relative_Hermite_POD": {
            "basis_uses_training_only": True,
            "primal_state": "Xw_equals_iwI_minus_Ah_hidden_inverse_Bh",
            "frequency_derivative": "dXw_equals_minus_i_times_iwI_minus_Ah_hidden_inverse_Xw",
            "local_interval_scale": "half_nearest_linear_frequency_spacing_with_one_sided_endpoints",
            "snapshot_groups": ["Xw", "local_interval_scale_times_dXw"],
            "relative_block_weight": "inverse_squared_Frobenius_norm_of_each_output_block_applied_to_each_snapshot_group",
            "nested_basis": "descending_eigenvectors_of_real_symmetric_covariance",
            "candidate_hidden_orders": list(HIDDEN_ORDERS),
        },
        "binding_gates": {
            "stable_Lyapunov_relative_residual_max": 1.0e-8,
            "square_root_reconstruction_relative_defect_max": 5.0e-12,
            "whitened_Lyapunov_relative_defect_max": 1.0e-8,
            "conservative_lift_identity_defect_max": 5.0e-9,
            "trial_test_biorthogonality_defect_max": 5.0e-9,
            "hidden_conservative_annihilation_defect_max": 5.0e-9,
            "snapshot_covariance_effective_rank_min": max(HIDDEN_ORDERS),
            "maximum_snapshot_solve_relative_residual_max": 1.0e-10,
            "reduced_Lyapunov_identity_relative_defect_max": 1.0e-8,
            "reduced_stable_spectral_abscissa_per_second_max": NONSTABLE_THRESHOLD_PER_SECOND,
            "complete_nonstable_eigenvalue_count_equal": EXACT_NONSTABLE_DIMENSION,
            "extra_nonstable_eigenvalue_count_max": 0,
            "exact_nonstable_pole_relative_defect_max": 5.0e-9,
            "maximum_frequency_solve_relative_residual_max": 1.0e-10,
            "cross_anchor_hidden_principal_cosine_min": 0.5,
            "resolved_self_energy": _transfer_gates(),
            "conservative_face_flux": _transfer_gates(),
        },
        "dimension_budget": {
            "physical_conservative": PHYSICAL_DIMENSION,
            "exact_nonstable": EXACT_NONSTABLE_DIMENSION,
            "full_hidden": HIDDEN_DIMENSION,
            "hidden_orders": list(HIDDEN_ORDERS),
            "online_dimensions": [
                PHYSICAL_DIMENSION + EXACT_NONSTABLE_DIMENSION + order
                for order in HIDDEN_ORDERS
            ],
            "maximum_online_dimension": MAXIMUM_ONLINE_DIMENSION,
        },
        "decisions": {
            "first_joint_pass": (
                "two_anchor_higher_order_square_root_Hermite_reduction_passed_"
                "parametric_online_architecture_manifest_authorized"
            ),
            "no_order_through_320_passes": (
                "higher_order_Hermite_failed_through_R510_"
                "direct_stable_rational_realization_reassessment_required"
            ),
            "numerical_integrity_fails": "higher_order_Hermite_numerical_failure_stop",
        },
        "claim_boundary": {
            "online_integrator_implementation_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
            "physical_failure_can_be_declared": False,
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
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
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
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
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    _, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("high-order Hermite manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("high-order Hermite manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "candidate_hidden_orders": list(HIDDEN_ORDERS),
        "candidate_online_dimensions": [
            PHYSICAL_DIMENSION + EXACT_NONSTABLE_DIMENSION + order
            for order in HIDDEN_ORDERS
        ],
        "training_frequency_count": 129,
        "validation_frequency_count_including_shared_DC": 129,
        "non_DC_validation_information_used_in_basis": False,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25ag",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_package_hashes": parent_hashes,
            "fiber_package_hashes": _checksums(FIBER_DIRECTORY),
            "R32_package_hashes": _checksums(R32_DIRECTORY),
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_next_runner": NEXT_RUNNER,
            "authorized_next_test": NEXT_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
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
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# High-order square-root Hermite manifest WP10c9d6c7c3b5c4f25af",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This definitions-only confirmation tests whether the prior failure was caused by the R320 ceiling rather than the square-root conservative architecture.",
                "",
                "All previously inspected original, midpoint, and quarter-point frequencies are promoted to the 129-point training grid. A disjoint 128-frequency eighth-point set plus shared DC is frozen as validation before it is evaluated.",
                "",
                "Candidate hidden orders 280/288/296/300/304/320 correspond to total online dimensions 470/478/486/490/494/510. All transfer, exact-pole, stability, and cross-anchor gates remain unchanged.",
                "",
                "No nonlinear root, propagation, truth assembly, online integrator, or predictive cycle is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
