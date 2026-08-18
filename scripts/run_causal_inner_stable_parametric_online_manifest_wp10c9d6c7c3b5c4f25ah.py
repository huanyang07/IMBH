#!/usr/bin/env python3
"""Freeze the stable parametric-kernel and online-cost preflight."""

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

import run_causal_inner_high_order_hermite_manifest_wp10c9d6c7c3b5c4f25af as prior_manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ah"
CLASSIFICATION = (
    "stable_parametric_online_manifest_frozen_"
    "alignment_dissipation_cost_and_unstable_bundle_audit_authorized"
)
PARENT_COMMIT = "253796ac8da70a6715b96d9575425d04d3af7bc3"
PARENT_PARENT = "b9ba72d8bf438f61b8923b6528ca4cc7b94f4782"
PARENT_TREE = "fb445ff47294341cbba912c10d769701efe4ffa5"

PARENT_ARTIFACT = "causal_inner_high_order_hermite_audit_wp10c9d6c7c3b5c4f25ag"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
FIBER_DIRECTORY = prior_manifest.FIBER_DIRECTORY

ARTIFACT = (
    "causal_inner_stable_parametric_online_manifest_"
    "wp10c9d6c7c3b5c4f25ah"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_stable_parametric_online_manifest_"
    "wp10c9d6c7c3b5c4f25ah.py"
)
THIS_TEST = (
    "tests/test_causal_inner_stable_parametric_online_manifest_"
    "wp10c9d6c7c3b5c4f25ah.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_stable_parametric_online_audit_"
    "wp10c9d6c7c3b5c4f25ai.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_stable_parametric_online_audit_"
    "wp10c9d6c7c3b5c4f25ai.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_STABLE_PARAMETRIC_ONLINE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AH_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CONSERVATIVE_DIMENSION = 162
HIDDEN_DIMENSION = 280
STABLE_REDUCED_DIMENSION = 442
UNSTABLE_DIMENSION = 28
TOTAL_ARCHITECTURE_DIMENSION = 470
PARAMETER_GRID = tuple(index / 100.0 for index in range(101))
FIDUCIAL_CYCLE_SECONDS = 6.7 * 86_400.0
WALL_BUDGET_SECONDS = 3.0 * 86_400.0
MAXIMUM_MACROSTEPS = 100_000
MINIMUM_MACROSTEP_SECONDS = FIDUCIAL_CYCLE_SECONDS / MAXIMUM_MACROSTEPS


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
        raise RuntimeError("stable-parametric parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("stable-parametric parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("stable-parametric parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["selected_hidden_order"] != HIDDEN_DIMENSION
        or summary["selected_online_dimension"] != TOTAL_ARCHITECTURE_DIMENSION
        or summary["authorized_next"]
        != "definitions_only_stable_parametric_online_architecture_manifest"
        or summary["physical_failure_detected"]
    ):
        raise RuntimeError("high-order Hermite certificate changed")
    _checksums(FIBER_DIRECTORY)
    return summary, hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in (
                "summary.json",
                "metrics.json",
                "decisive_model.npz",
            )
        },
        "fiber_decisive_hashes": {
            name: _sha(FIBER_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "decisive_fibers.npz")
        },
        "state_partition": {
            "conservative_coarse_state": CONSERVATIVE_DIMENSION,
            "strictly_stable_memory": HIDDEN_DIMENSION,
            "stable_descriptor_total": STABLE_REDUCED_DIMENSION,
            "exact_center_unstable_bundle": UNSTABLE_DIMENSION,
            "total_local_architecture_dimension": TOTAL_ARCHITECTURE_DIMENSION,
            "unstable_bundle_may_be_treated_as_stable_memory": False,
        },
        "hidden_alignment": {
            "method": "orthogonal_Procrustes_on_560_coordinate_hidden_truth_trials",
            "heldout_transform": "S_equals_block_diag_I162_Q280",
            "conservative_coordinates_are_not_rotated": True,
        },
        "stable_descriptor_interpolation": {
            "endpoint_form": "G_a_xdot_equals_K_a_x_with_K_a_equals_G_a_A_a",
            "aligned_heldout_pair": "G1_tilde_equals_S_transpose_G1_S_and_K1_tilde_equals_S_transpose_K1_S",
            "parameter_grid": list(PARAMETER_GRID),
            "metric": "G_theta_equals_one_minus_theta_G0_plus_theta_G1_tilde",
            "generator_form": "K_theta_equals_one_minus_theta_K0_plus_theta_K1_tilde",
            "operator": "A_theta_equals_solve_G_theta_K_theta",
            "proof": "convex_SPD_metric_and_convex_strictly_negative_symmetric_K_preserve_stability",
            "intermediate_transfer_is_structural_only_without_intermediate_truth": True,
        },
        "unstable_bundle_interpolation": {
            "alignment": "existing_hash_locked_28_by_28_orthogonal_rotation",
            "operator": "U_theta_equals_one_minus_theta_U0_plus_theta_Q_transpose_U1_Q",
            "role": "fast_nonlinear_departure_bundle_not_linear_macro_memory",
            "required_online_replacement": "offline_identified_nonlinear_saturation_manifold_or_conservative_hybrid_event_map",
            "linear_macro_propagation_forbidden": True,
        },
        "runtime_contract": {
            "fiducial_cycle_seconds": FIDUCIAL_CYCLE_SECONDS,
            "wall_budget_seconds": WALL_BUDGET_SECONDS,
            "maximum_macrosteps": MAXIMUM_MACROSTEPS,
            "minimum_average_macrostep_seconds": MINIMUM_MACROSTEP_SECONDS,
            "online_truth_calls_per_macrostep": 0,
            "stable_kernel_maximum_wall_seconds_per_step": 1.0,
            "stable_kernel_maximum_projected_cycle_wall_seconds": 0.10
            * WALL_BUDGET_SECONDS,
            "benchmark_thread_count": 1,
            "benchmark_exponential_repetitions": 7,
            "benchmark_LU_factor_repetitions": 21,
            "benchmark_LU_solve_repetitions": 501,
            "benchmark_matvec_repetitions": 1001,
            "macro_update": "frozen_coefficient_exponential_or_equivalent_L_stable_descriptor_update",
        },
        "binding_gates": {
            "hidden_alignment_orthogonality_defect_max": 5.0e-12,
            "stable_metric_minimum_eigenvalue_min": 1.0e-6,
            "stable_metric_condition_number_max": 2.0e9,
            "stable_symmetric_dissipation_largest_eigenvalue_max": -1.0e-6,
            "stable_spectral_abscissa_per_second_max": -1.0e-4,
            "descriptor_identity_relative_defect_max": 1.0e-9,
            "endpoint_operator_relative_defect_max": 1.0e-8,
            "unstable_alignment_orthogonality_defect_max": 5.0e-12,
            "unstable_dimension_equal": UNSTABLE_DIMENSION,
            "unstable_positive_real_part_count_equal": UNSTABLE_DIMENSION,
            "unstable_minimum_real_part_per_second_min": 1.0,
            "stable_exponential_median_wall_seconds_max": 1.0,
            "stable_LU_factor_median_wall_seconds_max": 0.10,
            "stable_LU_solve_median_wall_seconds_max": 0.01,
            "stable_matvec_median_wall_seconds_max": 0.001,
            "stable_kernel_projected_cycle_wall_seconds_max": 0.10
            * WALL_BUDGET_SECONDS,
        },
        "decision": {
            "pass": (
                "stable_parametric_kernel_and_cost_passed_"
                "nonlinear_unstable_bundle_database_manifest_authorized"
            ),
            "stability_or_alignment_failure": (
                "stable_parametric_kernel_failed_coordinate_or_descriptor_reassessment_required"
            ),
            "cost_failure": (
                "stable_parametric_kernel_structural_pass_cost_failed_"
                "online_linear_algebra_optimization_required"
            ),
        },
        "claim_boundary": {
            "intermediate_anchor_physics_certified": False,
            "nonlinear_unstable_saturation_identified": False,
            "online_integrator_implementation_authorized": False,
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
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("stable-parametric manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("stable-parametric manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "stable_descriptor_dimension": STABLE_REDUCED_DIMENSION,
        "unstable_bundle_dimension": UNSTABLE_DIMENSION,
        "total_architecture_dimension": TOTAL_ARCHITECTURE_DIMENSION,
        "parameter_grid_count": len(PARAMETER_GRID),
        "fiducial_cycle_seconds": FIDUCIAL_CYCLE_SECONDS,
        "maximum_macrosteps": MAXIMUM_MACROSTEPS,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25ai",
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
                "# Stable parametric online manifest WP10c9d6c7c3b5c4f25ah",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This definitions-only package freezes orthogonal hidden-space alignment, convex dissipative descriptor interpolation, exact unstable-bundle alignment, and a single-thread online-kernel cost audit for the certified 470-state architecture.",
                "",
                "The 442-state conservative-plus-stable-memory block must remain strictly dissipative. The 28 positive-growth modes are explicitly forbidden from being treated as linear macro-memory; they require an offline-identified nonlinear saturation manifold or conservative hybrid event map.",
                "",
                "The fiducial runtime target remains one 6.7-day cycle in at most three wall-days with at most 100,000 macrosteps and zero online truth calls.",
                "",
                "No predictive interpolation, online integrator, or reduced cycle is authorized.",
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
