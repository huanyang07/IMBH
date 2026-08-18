#!/usr/bin/env python3
"""Freeze the conservative constrained-Lyapunov reduction preflight."""

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
for path in (ROOT / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_unstable_exact_conservative_fiber_manifest_wp10c9d6c7c3b5c4f25t as fiber_manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25x"
CLASSIFICATION = (
    "conservative_constrained_lyapunov_reduction_manifest_frozen_"
    "saved_generator_preflight_authorized"
)
PARENT_COMMIT = "c7b1660ccd564c522cc644c3a85a75dad3f2447e"
PARENT_PARENT = "c28f6b273afb585b2cf3c7ac0b85499428b5ea31"
PARENT_TREE = "6c37b1c3cb3877b30c241c088f83d0caac4286d7"

PARENT_ARTIFACT = "causal_inner_effective_real_rank_audit_wp10c9d6c7c3b5c4f25w"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
FIBER_ARTIFACT = (
    "causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u"
)
FIBER_DIRECTORY = ROOT / "results/canonical" / FIBER_ARTIFACT
R32_DIRECTORY = fiber_manifest.R32_DIRECTORY
PRIMARY_GENERATOR_DIRECTORY = fiber_manifest.PRIMARY_GENERATOR_DIRECTORY
CROSS_ANCHOR_DIRECTORY = fiber_manifest.CROSS_ANCHOR_DIRECTORY

ARTIFACT = (
    "causal_inner_constrained_lyapunov_reduction_manifest_"
    "wp10c9d6c7c3b5c4f25x"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_constrained_lyapunov_reduction_manifest_"
    "wp10c9d6c7c3b5c4f25x.py"
)
THIS_TEST = (
    "tests/test_causal_inner_constrained_lyapunov_reduction_manifest_"
    "wp10c9d6c7c3b5c4f25x.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_constrained_lyapunov_reduction_audit_"
    "wp10c9d6c7c3b5c4f25y.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_constrained_lyapunov_reduction_audit_"
    "wp10c9d6c7c3b5c4f25y.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSTRAINED_LYAPUNOV_REDUCTION_"
    "MANIFEST_WP10C9D6C7C3B5C4F25X_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PHYSICAL_DIMENSION = 162
EXACT_NONSTABLE_DIMENSION = 28
HIDDEN_ORDERS = (112, 120, 128, 130)
MAXIMUM_ONLINE_DIMENSION = 320
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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


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
        raise RuntimeError("constrained-Lyapunov parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("constrained-Lyapunov parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("constrained-Lyapunov parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != "two_anchor_effective_rank_unstable_exact_fiber_passed_constrained_lyapunov_stable_reduction_manifest_authorized"
        or summary["effective_real_rank"] != EXACT_NONSTABLE_DIMENSION
        or summary["minimum_remaining_stable_memory_budget"] != 130
        or summary["physical_failure_detected"]
        or summary["authorized_next"]
        != "definitions_only_constrained_lyapunov_stable_reduction_manifest"
    ):
        raise RuntimeError("constrained-Lyapunov authorization changed")
    for directory in (FIBER_DIRECTORY, R32_DIRECTORY, PRIMARY_GENERATOR_DIRECTORY, CROSS_ANCHOR_DIRECTORY):
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
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "realification_singular_values.npz")
        },
        "fiber_decisive_hashes": {
            name: _sha(FIBER_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "decisive_fibers.npz")
        },
        "saved_input_hashes": {
            "primary_generator": _sha(PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz"),
            "primary_output": _sha(PRIMARY_GENERATOR_DIRECTORY / "projection.npz"),
            "heldout_generator_and_output": _sha(CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz"),
            "R32_projection": _sha(R32_DIRECTORY / "R32_projection_promotion.npz"),
            "frequency_ladder": _sha(R32_DIRECTORY / "R32_transfer.npz"),
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_new_truth_anchors": 0,
            "maximum_wall_hours": 1.5,
            "candidate_hidden_orders": list(HIDDEN_ORDERS),
            "fail_fast_after_first_joint_pass": True,
        },
        "exact_architecture": {
            "truth_split": "x_equals_Uu_a_plus_S_y",
            "unstable_dynamics": "a_dot_equals_Au_a_exactly_without_reduction",
            "stable_lyapunov_equation": "As_transpose_P_plus_P_As_equals_minus_identity",
            "stable_conservative_map": "C_equals_R32_S",
            "P_minimum_conservative_lift": "Lq_equals_P_inverse_C_transpose_times_inverse_C_P_inverse_C_transpose",
            "hidden_trial_constraint": "C_Z_equals_zero",
            "trial": "V_equals_horizontal_concatenation_Lq_Z",
            "test": "W_equals_P_V_times_inverse_V_transpose_P_V",
            "required_identity": "first_162_columns_of_W_equal_C_transpose",
            "internal_stable_coordinate": "q_s_equals_q_total_minus_R32_Uu_a",
            "face_flux_single_valued_before_conservative_divergence": True,
        },
        "snapshot_selection": {
            "family": "frequency_limited_two_sided_snapshot_balancing",
            "training_frequencies": "unchanged_33_point_R32_ladder",
            "heldout_frequencies": "unchanged_log_midpoints_plus_DC",
            "exact_hidden_input": "W_h_transpose_As_Lq",
            "combined_output": "conservative_derivative_feedback_plus_face_flux",
            "empirical_gramians": "equal_weight_real_parts_of_primal_and_adjoint_resolvent_outer_products",
            "candidate_basis": "balanced_snapshot_trial_then_P_orthonormalization_in_kernel_C",
            "candidate_hidden_orders": list(HIDDEN_ORDERS),
        },
        "binding_gates": {
            "stable_Lyapunov_relative_residual_max": 1.0e-8,
            "stable_Lyapunov_minimum_eigenvalue_min": 0.0,
            "stable_Lyapunov_condition_number_max": 1.0e12,
            "conservative_lift_identity_defect_max": 5.0e-9,
            "conservative_test_identity_defect_max": 5.0e-9,
            "trial_test_biorthogonality_defect_max": 5.0e-9,
            "hidden_conservative_annihilation_defect_max": 5.0e-9,
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
            "hidden_orders": list(HIDDEN_ORDERS),
            "online_dimensions": [
                PHYSICAL_DIMENSION + EXACT_NONSTABLE_DIMENSION + order
                for order in HIDDEN_ORDERS
            ],
            "maximum_online_dimension": MAXIMUM_ONLINE_DIMENSION,
        },
        "decisions": {
            "first_joint_pass": (
                "two_anchor_conservative_constrained_lyapunov_reduction_passed_"
                "parametric_alignment_manifest_authorized"
            ),
            "no_order_through_130_passes": (
                "constrained_lyapunov_reduction_failed_within_R320_"
                "structured_basis_reassessment_required"
            ),
            "numerical_integrity_fails": (
                "constrained_lyapunov_reduction_numerical_failure_stop"
            ),
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
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "PROSPECTIVE"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_summary, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("constrained-Lyapunov manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("constrained-Lyapunov manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "parent_effective_real_rank": parent_summary["effective_real_rank"],
        "candidate_hidden_orders": list(HIDDEN_ORDERS),
        "candidate_online_dimensions": [PHYSICAL_DIMENSION + EXACT_NONSTABLE_DIMENSION + order for order in HIDDEN_ORDERS],
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": WORK_PACKAGE.replace("25x", "25y"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_commit": PARENT_COMMIT,
        "parent_parent": PARENT_PARENT,
        "parent_tree": PARENT_TREE,
        "parent_package_hashes": parent_hashes,
        "fiber_package_hashes": _checksums(FIBER_DIRECTORY),
        "R32_package_hashes": _checksums(R32_DIRECTORY),
    })
    _write(ARTIFACT_DIRECTORY / "provenance.json", {
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
        "source_hashes": {THIS_RUNNER: _sha(ROOT / THIS_RUNNER), THIS_TEST: _sha(ROOT / THIS_TEST)},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    })
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.write_text("\n".join((
        "# Constrained-Lyapunov reduction manifest WP10c9d6c7c3b5c4f25x",
        "",
        "## Classification",
        "",
        f"`{CLASSIFICATION}`",
        "",
        "This definitions-only package keeps all 28 nonstable coordinates exact, reduces only the strictly stable spectral complement, and uses a P-minimum conservative lift. The resulting P-weighted Petrov test basis must reproduce the 162 conservative test rows exactly while providing a strict reduced Lyapunov certificate.",
        "",
        "Hidden orders 112, 120, 128, and 130 are tested at both saved anchors. The closure must reproduce resolved self-energy and face-flux transfer, add no nonstable poles, remain cross-anchor alignable, and stay within R320.",
        "",
        "No truth assembly, nonlinear root, propagation, online integrator, or predictive cycle is authorized.",
        "",
    )), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze: raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
