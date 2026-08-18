#!/usr/bin/env python3
"""Freeze the square-root conservative transfer-seeded reduction preflight."""

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

import run_causal_inner_constrained_lyapunov_reduction_manifest_wp10c9d6c7c3b5c4f25x as prior_manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25z"
CLASSIFICATION = (
    "square_root_conservative_transfer_seeded_manifest_frozen_"
    "saved_generator_reassessment_authorized"
)
PARENT_COMMIT = "f2c7d14f1c81d392ad81449733c5ac5fe5a19041"
PARENT_PARENT = "c2be1dc25a0102ef03c98388b9ebefaa73ee9508"
PARENT_TREE = "4d125b4214aad98927cd29b2326fd9495a1841bb"

PRIOR_ARTIFACT = (
    "causal_inner_constrained_lyapunov_reduction_audit_"
    "wp10c9d6c7c3b5c4f25y"
)
PRIOR_DIRECTORY = ROOT / "results/canonical" / PRIOR_ARTIFACT
FIBER_DIRECTORY = prior_manifest.FIBER_DIRECTORY
R32_DIRECTORY = prior_manifest.R32_DIRECTORY
PRIMARY_GENERATOR_DIRECTORY = prior_manifest.PRIMARY_GENERATOR_DIRECTORY
CROSS_ANCHOR_DIRECTORY = prior_manifest.CROSS_ANCHOR_DIRECTORY
COMMON_BASIS_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_rank_adaptive_common_memory_audit_wp10c9d6c7c3b5c4f25q"
)

ARTIFACT = (
    "causal_inner_square_root_transfer_seeded_manifest_"
    "wp10c9d6c7c3b5c4f25z"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_square_root_transfer_seeded_manifest_"
    "wp10c9d6c7c3b5c4f25z.py"
)
THIS_TEST = (
    "tests/test_causal_inner_square_root_transfer_seeded_manifest_"
    "wp10c9d6c7c3b5c4f25z.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_square_root_transfer_seeded_audit_"
    "wp10c9d6c7c3b5c4f25aa.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_square_root_transfer_seeded_audit_"
    "wp10c9d6c7c3b5c4f25aa.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SQUARE_ROOT_TRANSFER_SEEDED_"
    "MANIFEST_WP10C9D6C7C3B5C4F25Z_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PHYSICAL_DIMENSION = 162
EXACT_NONSTABLE_DIMENSION = 28
STABLE_DIMENSION = 532
HIDDEN_DIMENSION = 370
HIDDEN_ORDERS = (112, 120, 124, 128, 130)
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
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent() -> tuple[dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("square-root reassessment parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("square-root reassessment parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("square-root reassessment parent tree changed")
    hashes = _checksums(PRIOR_DIRECTORY)
    summary = _read(PRIOR_DIRECTORY / "summary.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "constrained_lyapunov_reduction_numerical_failure_stop"
        or summary["physical_failure_detected"]
        or summary["authorized_next"] is not None
    ):
        raise RuntimeError("raw-P rejection changed")
    for directory in (
        FIBER_DIRECTORY,
        R32_DIRECTORY,
        PRIMARY_GENERATOR_DIRECTORY,
        CROSS_ANCHOR_DIRECTORY,
        COMMON_BASIS_DIRECTORY,
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
        "authority": {
            "source": "explicit_user_authorization_after_f25y_reassessment",
            "preserve_f25y_rejection": True,
            "may_reuse_only_saved_generators_and_canonical_bases": True,
        },
        "prior_decisive_hashes": {
            name: _sha(PRIOR_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "decisive_model.npz")
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
            "R196_common_basis": _sha(COMMON_BASIS_DIRECTORY / "decisive_basis.npz"),
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
            "exact_nonstable_fiber": "a_dot_equals_Au_a_without_reduction",
            "stable_lyapunov_equation": "As_transpose_P_plus_P_As_equals_minus_identity",
            "square_root": "P_equals_T_transpose_T_upper_Cholesky",
            "whitened_operator": "Ahat_equals_T_As_T_inverse",
            "whitened_conservative_map": "Chat_equals_R32_S_T_inverse",
            "minimum_norm_right_inverse": "Lhat_equals_Chat_Moore_Penrose_right_inverse",
            "hidden_space": "Nhat_equals_orthonormal_basis_of_kernel_Chat",
            "trial": "Vhat_equals_horizontal_concatenation_Lhat_Zhat",
            "test": "What_equals_horizontal_concatenation_Chat_transpose_Zhat",
            "metric": "G_equals_Vhat_transpose_Vhat",
            "required_identities": [
                "Chat_Lhat_equals_identity",
                "Chat_Zhat_equals_zero",
                "What_transpose_Vhat_equals_identity",
                "full_Vhat_What_transpose_equals_identity",
            ],
            "face_flux_single_valued_before_conservative_divergence": True,
        },
        "transfer_seed": {
            "source": "two_sided_balanced_trial_of_complete_R196_saved_generator",
            "seed_order": 130,
            "mapping": (
                "R196_stable_trial_to_truth_then_exact_nonstable_projection_then_"
                "stable_Lyapunov_whitening_then_kernel_Chat_projection"
            ),
            "nested_basis": "left_singular_vectors_of_projected_seed_coordinates",
            "candidate_hidden_orders": list(HIDDEN_ORDERS),
            "tangential_rational_Krylov_enrichment_allowed": False,
        },
        "binding_gates": {
            "stable_Lyapunov_relative_residual_max": 1.0e-8,
            "stable_Lyapunov_minimum_eigenvalue_min": 0.0,
            "stable_Lyapunov_condition_number_max": 1.0e12,
            "square_root_reconstruction_relative_defect_max": 5.0e-12,
            "whitened_Lyapunov_relative_defect_max": 1.0e-8,
            "conservative_map_rank_equal": PHYSICAL_DIMENSION,
            "conservative_lift_identity_defect_max": 5.0e-9,
            "conservative_test_identity_defect_max": 5.0e-9,
            "trial_test_biorthogonality_defect_max": 5.0e-9,
            "hidden_conservative_annihilation_defect_max": 5.0e-9,
            "full_coordinate_reconstruction_relative_defect_max": 5.0e-9,
            "projected_seed_effective_rank_min": max(HIDDEN_ORDERS),
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
            "stable": STABLE_DIMENSION,
            "hidden_available": HIDDEN_DIMENSION,
            "hidden_orders": list(HIDDEN_ORDERS),
            "online_dimensions": [
                PHYSICAL_DIMENSION + EXACT_NONSTABLE_DIMENSION + order
                for order in HIDDEN_ORDERS
            ],
            "maximum_online_dimension": MAXIMUM_ONLINE_DIMENSION,
        },
        "decisions": {
            "first_joint_pass": (
                "two_anchor_square_root_transfer_seeded_reduction_passed_"
                "parametric_alignment_manifest_authorized"
            ),
            "no_order_through_130_passes": (
                "square_root_transfer_seeded_reduction_failed_within_R320_"
                "structured_basis_reassessment_required"
            ),
            "numerical_integrity_fails": (
                "square_root_transfer_seeded_reduction_numerical_failure_stop"
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
    prior_summary, prior_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("square-root manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("square-root manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "preserved_prior_classification": prior_summary["classification"],
        "candidate_hidden_orders": list(HIDDEN_ORDERS),
        "candidate_online_dimensions": [
            PHYSICAL_DIMENSION + EXACT_NONSTABLE_DIMENSION + order
            for order in HIDDEN_ORDERS
        ],
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25aa",
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
            "prior_package_hashes": prior_hashes,
            "fiber_package_hashes": _checksums(FIBER_DIRECTORY),
            "R32_package_hashes": _checksums(R32_DIRECTORY),
            "common_basis_package_hashes": _checksums(COMMON_BASIS_DIRECTORY),
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
                "# Square-root transfer-seeded reduction manifest WP10c9d6c7c3b5c4f25z",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This definitions-only reassessment preserves the f25y rejection. It removes the raw-P inverse from the conservative Petrov construction by whitening the exact stable complement with an upper Cholesky factor of its Lyapunov certificate.",
                "",
                "In whitened coordinates, the trial is `[Chat^+ , Z]` and the test is `[Chat^T, Z]`, with `Z` orthonormal in `ker(Chat)`. The hidden seed is the saved complete-R196 two-sided balanced trial, projected through the exact stable fiber and conservative nullspace.",
                "",
                "Hidden orders 112, 120, 124, 128, and 130 are tested at both saved anchors. Stability, exact nonstable poles, conservative algebra, transfer accuracy, and cross-anchor alignment remain jointly binding.",
                "",
                "No truth assembly, nonlinear root, propagation, online integrator, or predictive cycle is authorized.",
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
    arguments = parser.parse_args()
    if not arguments.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
