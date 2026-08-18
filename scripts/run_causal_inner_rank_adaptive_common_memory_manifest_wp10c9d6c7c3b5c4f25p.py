#!/usr/bin/env python3
"""Freeze the saved-generator rank-adaptive common-memory reassessment."""

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

SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25p"
CLASSIFICATION = (
    "rank_adaptive_common_memory_reassessment_manifest_frozen_"
    "saved_generator_audit_authorized"
)
PARENT_COMMIT = "34ee36ca2ed33eacc72689a0282a6845caac7d10"
PARENT_PARENT = "da1d271942a0521adeea261833d91fb7263b78b6"
PARENT_TREE = "441b9c754d01e759c5415ebc60c66d3be9679ef2"

PARENT_ARTIFACT = (
    "causal_inner_common_resolved_subspace_cross_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25o"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
ARTIFACT = (
    "causal_inner_rank_adaptive_common_memory_manifest_"
    "wp10c9d6c7c3b5c4f25p"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_rank_adaptive_common_memory_manifest_"
    "wp10c9d6c7c3b5c4f25p.py"
)
THIS_TEST = (
    "tests/test_causal_inner_rank_adaptive_common_memory_manifest_"
    "wp10c9d6c7c3b5c4f25p.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_rank_adaptive_common_memory_audit_"
    "wp10c9d6c7c3b5c4f25q.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_rank_adaptive_common_memory_audit_"
    "wp10c9d6c7c3b5c4f25q.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_RANK_ADAPTIVE_COMMON_MEMORY_"
    "MANIFEST_WP10C9D6C7C3B5C4F25P_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TRUTH_DIMENSION = 560
PHYSICAL_R32_DIMENSION = 162
LOCAL_PROMOTED_DIMENSION = 18
MAXIMUM_ONLINE_CONTINUOUS_DIMENSION = 320
STABILITY_MARGIN_PER_SECOND = 1.0e-8
COMMON_RANK_CANDIDATES = tuple(range(18, 37, 2))
MEMORY_ORDER_START = 96
MEMORY_ORDER_INCREMENT = 8


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


def _memory_orders(common_rank: int) -> tuple[int, ...]:
    maximum = MAXIMUM_ONLINE_CONTINUOUS_DIMENSION - (
        PHYSICAL_R32_DIMENSION + int(common_rank)
    )
    if maximum < MEMORY_ORDER_START:
        return ()
    orders = list(range(MEMORY_ORDER_START, maximum + 1, MEMORY_ORDER_INCREMENT))
    if not orders or orders[-1] != maximum:
        orders.append(maximum)
    return tuple(orders)


def _validate_parent() -> tuple[dict, dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("parent cross-anchor result commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parent cross-anchor result parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parent cross-anchor result tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "common_metrics.json")
    if (
        summary["classification"]
        != "heldout_R32_R96_memory_failed_architecture_reassessment_required"
        or summary["passed"]
        or not summary["assembly_passed"]
        or not summary["common_numerical_passed"]
        or summary["memory_passed"]
        or not summary["global_chart_alignment_passed"]
        or summary["authorized_next"]
        != "definitions_only_reduced_variable_or_memory_architecture_reassessment_manifest"
        or summary["physical_failure_detected"]
        or metrics["common_promoted_union_dimension"] != 36
        or metrics["primary_local_promoted_dimension"] != LOCAL_PROMOTED_DIMENSION
        or metrics["heldout_local_promoted_dimension"] != LOCAL_PROMOTED_DIMENSION
    ):
        raise RuntimeError("parent architecture-reassessment authorization changed")
    return summary, metrics, hashes


def _transfer_gates() -> dict:
    return {
        "reduced_spectral_abscissa_per_second_max": -STABILITY_MARGIN_PER_SECOND,
        "lyapunov_dissipation_residual_max": 1.0e-8,
        "lyapunov_certificate_minimum_eigenvalue_min": 0.0,
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
            for name in (
                "summary.json",
                "assembly_metrics.json",
                "common_metrics.json",
                "heldout_generator.npz",
                "common_subspace.npz",
                "common_memory_models.npz",
            )
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_new_truth_anchors": 0,
            "saved_generator_rank_candidates": list(COMMON_RANK_CANDIDATES),
            "maximum_balanced_realizations_per_anchor": len(COMMON_RANK_CANDIDATES),
            "maximum_wall_hours": 1.5,
            "fail_fast_after_first_joint_rank_memory_pass": True,
        },
        "mathematical_diagnosis": {
            "local_promoted_dimensions": [LOCAL_PROMOTED_DIMENSION] * 2,
            "local_promoted_worst_principal_angle_degrees": 2.9204769695366193,
            "full_union_dimension": 36,
            "reason_full_union_is_not_automatically_selected": (
                "small_cross_anchor_difference_directions_need_only_be_retained_"
                "when_required_for_two_anchor_unresolved_stability"
            ),
            "binding_selection_principle": (
                "minimum_online_dimension_that_is_stable_and_transfer_accurate_"
                "at_both_hash_locked_anchors"
            ),
        },
        "common_rank_ladder": {
            "candidates": list(COMMON_RANK_CANDIDATES),
            "basis": (
                "leading_left_singular_vectors_of_the_saved_concatenated_local_"
                "promoted_bases_projected_into_each_anchor_physical_complement_"
                "and_orthogonal_Procrustes_aligned"
            ),
            "evaluate_in_ascending_rank": True,
            "pass_requires_at_both_anchors": {
                "remaining_unresolved_spectral_abscissa_per_second_max": (
                    -STABILITY_MARGIN_PER_SECOND
                ),
                "common_basis_orthogonality_defect_max": 5.0e-11,
                "augmented_restriction_lifting_identity_defect_max": 5.0e-10,
                "augmented_restriction_stable_annihilation_defect_max": 5.0e-10,
                "minimum_cross_anchor_basis_principal_cosine": 0.75,
            },
            "local_subspace_capture_is_diagnostic": True,
            "stability_is_binding": True,
        },
        "memory_order_ladder": {
            "policy": (
                "for_each_stable_rank_test_96_then_increment_by_8_then_the_"
                "largest_order_allowed_by_the_R320_cap"
            ),
            "orders_by_rank": {
                str(rank): list(_memory_orders(rank))
                for rank in COMMON_RANK_CANDIDATES
            },
            "family": "anchor_local_continuous_time_square_root_balanced",
            "training_frequencies": "same_33_parent_frequencies",
            "heldout_frequencies": "same_DC_plus_32_midpoint_frequencies",
            "pass_requires_at_both_anchors": _transfer_gates(),
        },
        "online_architecture": {
            "physical_R32_dimension": PHYSICAL_R32_DIMENSION,
            "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
            "dimension_formula": "162_plus_selected_common_rank_plus_selected_memory_order",
            "interior_memory_output_enters_as_single_valued_face_flux": True,
            "memory_update": "exponential_or_L_stable_implicit",
        },
        "decisions": {
            "joint_candidate_passes": (
                "two_anchor_rank_adaptive_common_memory_passed_"
                "online_prototype_manifest_authorized"
            ),
            "stable_rank_exists_but_no_memory_order_within_R320_passes": (
                "common_memory_cap_failed_local_fiber_parametric_memory_"
                "architecture_manifest_authorized"
            ),
            "no_common_rank_stabilizes_both": (
                "common_resolved_chart_failed_local_fiber_atlas_manifest_authorized"
            ),
            "saved_evidence_or_numerical_gate_fails": (
                "rank_adaptive_common_memory_numerical_failure_stop"
            ),
        },
        "claim_boundary": {
            "production_coefficients_authorized": False,
            "online_reduced_solver_implementation_authorized": False,
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
    parent_summary, parent_metrics, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("rank-adaptive manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("rank-adaptive manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "candidate_common_ranks": list(COMMON_RANK_CANDIDATES),
        "memory_order_start": MEMORY_ORDER_START,
        "memory_order_increment": MEMORY_ORDER_INCREMENT,
        "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
        "parent_classification_preserved": parent_summary["classification"],
        "parent_common_rank": parent_metrics["common_promoted_union_dimension"],
        "parent_primary_R96_maximum_dynamic_error": parent_metrics[
            "primary_memory_metrics"
        ]["training_maximum_normalized_dynamic_transfer_relative_error"],
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "production_coefficients_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": WORK_PACKAGE.replace("25p", "25q"),
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
                "# Rank-adaptive common-memory manifest WP10c9d6c7c3b5c4f25p",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The two local 18-dimensional promoted subspaces differ by at most 2.92 degrees, but the prior numerical-union rule retained all 36 directions. This manifest replaces exact union capture with the binding purpose of promotion: stable unresolved dynamics at both anchors.",
                "",
                "Saved common-basis ranks 18 through 36 are tested in increments of two. For every stable rank, balanced memory begins at order 96 and increases only while the complete R32 plus common modes plus memory remains at or below 320 states. The first joint two-anchor transfer pass is selected.",
                "",
                "No new truth generator, nonlinear root, or propagation is permitted. A failure is an architecture/cap result, not a physical failure.",
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
