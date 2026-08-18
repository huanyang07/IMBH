#!/usr/bin/env python3
"""Freeze the two-anchor common-resolved-subspace preflight."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25n"
CLASSIFICATION = (
    "common_resolved_subspace_cross_anchor_manifest_frozen_"
    "heldout_16ms_generator_preflight_authorized"
)
PARENT_COMMIT = "14caf62e8e5f3190f044d2aa2e87a97cc3015b80"
PARENT_PARENT = "4a2b820e1d1a632bd8120bb899293bd7789cfe9b"
PARENT_TREE = "963a0afdfb14477b5bed4e858793d9408a9f1f09"

PARENT_ARTIFACT = "causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
R32_ARTIFACT = "causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k"
R32_DIRECTORY = ROOT / "results/canonical" / R32_ARTIFACT
GENERATOR_ARTIFACT = "causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c"
GENERATOR_DIRECTORY = ROOT / "results/canonical" / GENERATOR_ARTIFACT

MIDDLE_PILOT_ARRAYS_RELATIVE = (
    "results/canonical/causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_"
    "wp10c9d6c7c3b5c4e1/decisive_arrays.npz"
)
MIDDLE_ARRAYS_RELATIVE = (
    "results/canonical/causal_inner_nonlinear_optimized_middle_20ms_completion_"
    "wp10c9d6c7c3b5c4e3/decisive_arrays.npz"
)
MIDDLE_PILOT_ARRAYS = ROOT / MIDDLE_PILOT_ARRAYS_RELATIVE
MIDDLE_ARRAYS = ROOT / MIDDLE_ARRAYS_RELATIVE

ARTIFACT = "causal_inner_common_resolved_subspace_cross_anchor_manifest_wp10c9d6c7c3b5c4f25n"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_common_resolved_subspace_cross_anchor_manifest_wp10c9d6c7c3b5c4f25n.py"
THIS_TEST = "tests/test_causal_inner_common_resolved_subspace_cross_anchor_manifest_wp10c9d6c7c3b5c4f25n.py"
NEXT_RUNNER = "scripts/run_causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o.py"
NEXT_TEST = "tests/test_causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COMMON_RESOLVED_SUBSPACE_"
    "CROSS_ANCHOR_MANIFEST_WP10C9D6C7C3B5C4F25N_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TRUTH_DIMENSION = 560
PHYSICAL_R32_DIMENSION = 162
PRIMARY_LOCAL_PROMOTED_DIMENSION = 18
MEMORY_ORDER = 96
MAXIMUM_COMMON_PROMOTED_DIMENSION = 62
MAXIMUM_ONLINE_CONTINUOUS_DIMENSION = 320
STABILITY_MARGIN_PER_SECOND = 1.0e-8


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
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _validate_parent() -> tuple[dict, dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("parent R32 memory-selection commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parent R32 memory-selection parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parent R32 memory-selection tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != "single_anchor_R32_global_balanced_order_96_selected_cross_anchor_preflight_authorized"
        or summary["authorized_next"]
        != "definitions_only_common_resolved_subspace_cross_anchor_preflight_manifest"
        or summary["selected_memory_order"] != MEMORY_ORDER
        or summary["selected_online_continuous_dimension"] != 276
        or summary["physical_failure_detected"]
        or not metrics["full_order_numerical_passed"]
    ):
        raise RuntimeError("parent common-subspace authorization changed")
    return summary, metrics, hashes


def _contract() -> dict:
    transfer_gates = {
        "reduced_spectral_abscissa_per_second_max": -STABILITY_MARGIN_PER_SECOND,
        "lyapunov_dissipation_residual_max": 1.0e-8,
        "lyapunov_certificate_minimum_eigenvalue_min": 0.0,
        "maximum_normalized_dynamic_transfer_relative_error_max": 0.25,
        "RMS_normalized_dynamic_transfer_relative_error_max": 0.10,
        "DC_normalized_dynamic_transfer_relative_error_max": 0.10,
        "maximum_normalized_total_transfer_relative_error_max": 0.25,
        "RMS_normalized_total_transfer_relative_error_max": 0.10,
        "DC_normalized_total_transfer_relative_error_max": 0.10,
        "database_roundtrip_bitwise": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": {
            "selected_R32_memory_models": _sha(PARENT_DIRECTORY / "candidate_models.npz"),
            "selected_R32_memory_errors": _sha(PARENT_DIRECTORY / "candidate_errors.npz"),
            "selected_R32_memory_metrics": _sha(PARENT_DIRECTORY / "metrics.json"),
            "primary_R32_projection_promotion": _sha(R32_DIRECTORY / "R32_projection_promotion.npz"),
            "primary_R32_transfer": _sha(R32_DIRECTORY / "R32_transfer.npz"),
            "primary_complete_fixed_Q_generator": _sha(GENERATOR_DIRECTORY / "descriptor_A.npz"),
            "middle_6ms_arrays": _sha(MIDDLE_PILOT_ARRAYS),
            "middle_20ms_arrays": _sha(MIDDLE_ARRAYS),
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 1,
            "allowed_new_truth_anchors": 1,
            "allowed_anchor_local_R32_projection_promotion_audits": 1,
            "allowed_common_basis_memory_fits": 2,
            "maximum_wall_hours": 2.5,
            "fail_fast": True,
        },
        "anchors": {
            "primary_20ms": {
                "role": "training_and_reference",
                "source": "hash_locked_existing_R32_generator_projection_and_transfer",
                "new_generator_assembly": False,
            },
            "heldout_16ms": {
                "role": "heldout_state_robustness",
                "source": "unique_exact_16ms_state_in_hash_locked_middle_trajectory",
                "time_seconds": 0.016,
                "layout": "same_committed_112_cell_middle_layout",
                "new_generator_assembly": True,
                "state_must_be_copied_bitwise_without_projection": True,
            },
        },
        "heldout_generator": {
            "construction": (
                "complete_state_dependent_fixed_Q_continuous_generator_from_"
                "monolithic_frozen_tangent_plus_complete_reaction_JVP"
            ),
            "reaction_mode": "same_certified_frozen_normalized_physical_action",
            "all_560_coordinate_directions": True,
            "pass_requires": {
                "complete_JVP_relative_defect_max": 5.0e-10,
                "constraint_differential_identity_relative_defect_max": 5.0e-10,
                "reaction_ledger_directional_relative_defect_max": 5.0e-10,
                "reaction_identity_directional_defect_max": 5.0e-8,
                "maximum_raw_Schur_condition_number": 1.0e8,
                "generator_and_state_database_roundtrip_bitwise": True,
            },
        },
        "anchor_local_R32_projection": {
            "construction": "same_nested_R32_mapped_plus_responsive_height_complete_QR",
            "promote_every_nonstable_compressed_coordinate": True,
            "stability_margin_per_second": STABILITY_MARGIN_PER_SECOND,
            "pass_requires": {
                "resolved_rank": PHYSICAL_R32_DIMENSION,
                "resolved_condition_number_max": 2.0e4,
                "restriction_lifting_identity_max": 5.0e-11,
                "restriction_complement_annihilation_max": 5.0e-11,
                "complement_orthogonality_max": 5.0e-11,
                "constraint_rowspace_relative_defect_max": 5.0e-10,
                "M_J_E_telescope_relative_defect_max": 5.0e-12,
                "remaining_unresolved_spectral_abscissa_per_second_max": -STABILITY_MARGIN_PER_SECOND,
            },
        },
        "common_resolved_subspace": {
            "input_spaces": "primary_and_heldout_local_promoted_truth_bases",
            "reference_basis": (
                "left_singular_vectors_of_concatenated_local_promoted_truth_bases"
            ),
            "union_numerical_rank_relative_cutoff": 1.0e-10,
            "maximum_common_promoted_dimension": MAXIMUM_COMMON_PROMOTED_DIMENSION,
            "anchor_local_realization": (
                "project_reference_union_into_each_anchor_physical_R32_complement_"
                "then_complete_QR_and_orthogonal_Procrustes_align_to_reference"
            ),
            "pass_requires": {
                "local_promoted_subspace_projection_relative_defect_max": 1.0e-8,
                "common_modal_basis_orthogonality_defect_max": 5.0e-11,
                "common_augmented_restriction_lifting_identity_defect_max": 5.0e-10,
                "common_augmented_restriction_stable_annihilation_defect_max": 5.0e-10,
                "remaining_common_unresolved_spectral_abscissa_per_second_max": -STABILITY_MARGIN_PER_SECOND,
                "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
            },
            "coordinate_policy": {
                "minimum_anchor_basis_principal_cosine_for_one_global_chart": 0.75,
                "below_threshold_selects_local_atlas_not_physical_failure": True,
                "raw_anchor_local_Schur_vectors_may_not_be_interpolated": True,
            },
        },
        "common_basis_memory": {
            "order": MEMORY_ORDER,
            "family": "global_continuous_time_square_root_balanced",
            "fit_separately_at_each_anchor_behind_common_resolved_coordinates": True,
            "training_frequencies": "same_33_parent_frequencies",
            "heldout_frequencies": "same_DC_plus_32_prospective_midpoint_ladder",
            "normalization": "anchor_local_forcing_observation_direct_two_norm_scaling",
            "pass_requires_at_each_anchor_on_training_and_heldout": transfer_gates,
            "direct_primary_coefficients_applied_at_heldout_anchor": "diagnostic_nonbinding",
        },
        "online_architecture": {
            "physical_R32_dimension": PHYSICAL_R32_DIMENSION,
            "common_promoted_dimension_max": MAXIMUM_COMMON_PROMOTED_DIMENSION,
            "stable_memory_dimension": MEMORY_ORDER,
            "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
            "memory_update_requirement": "exponential_or_L_stable_implicit",
            "interior_M_J_E_memory_output_must_enter_as_single_valued_face_flux": True,
        },
        "decisions": {
            "all_gates_and_global_chart_pass": (
                "two_anchor_common_subspace_R96_memory_passed_"
                "online_prototype_manifest_authorized"
            ),
            "local_gates_pass_global_chart_alignment_fails": (
                "two_anchor_local_models_passed_common_chart_failed_"
                "conservative_atlas_manifest_authorized"
            ),
            "heldout_memory_fails": (
                "heldout_R32_R96_memory_failed_architecture_reassessment_required"
            ),
            "generator_projection_or_stability_fails": (
                "heldout_generator_or_common_projection_numerical_failure_stop"
            ),
        },
        "claim_boundary": {
            "production_memory_coefficients_authorized": False,
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
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "PROSPECTIVE",
            })
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
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": PARENT_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_summary, parent_metrics, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("cross-anchor manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("cross-anchor manifest is already frozen")
    if not MIDDLE_PILOT_ARRAYS.exists() or not MIDDLE_ARRAYS.exists():
        raise RuntimeError("committed middle trajectory evidence is unavailable")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "primary_anchor_time_seconds": 0.020,
        "heldout_anchor_time_seconds": 0.016,
        "physical_R32_dimension": PHYSICAL_R32_DIMENSION,
        "primary_local_promoted_dimension": PRIMARY_LOCAL_PROMOTED_DIMENSION,
        "memory_order": MEMORY_ORDER,
        "maximum_common_promoted_dimension": MAXIMUM_COMMON_PROMOTED_DIMENSION,
        "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
        "heldout_16ms_generator_preflight_authorized": True,
        "parent_classification_preserved": parent_summary["classification"],
        "parent_selected_R32_memory_maximum_dynamic_error": next(
            item["training_maximum_normalized_dynamic_transfer_relative_error"]
            for item in parent_metrics["candidate_metrics"]
            if item["label"] == parent_summary["selected_label"]
        ),
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": WORK_PACKAGE.replace("25n", "25o"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_commit": PARENT_COMMIT,
        "parent_parent": PARENT_PARENT,
        "parent_tree": PARENT_TREE,
        "parent_package_hashes": parent_hashes,
        "R32_package_hashes": _checksums(R32_DIRECTORY),
        "generator_package_hashes": _checksums(GENERATOR_DIRECTORY),
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
        "source_hashes": {
            THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
            THIS_TEST: _sha(ROOT / THIS_TEST),
        },
        "heldout_state_source_hashes": {
            MIDDLE_PILOT_ARRAYS_RELATIVE: _sha(MIDDLE_PILOT_ARRAYS),
            MIDDLE_ARRAYS_RELATIVE: _sha(MIDDLE_ARRAYS),
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
    })
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join((
            "# Common-resolved-subspace cross-anchor manifest WP10c9d6c7c3b5c4f25n",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "This definitions-only package preserves the certified single-anchor R180 plus order-96 memory result. It freezes one held-out complete fixed-Q generator at the exact committed 16 ms middle-layout state; no new nonlinear root or propagation is permitted.",
            "",
            "The local promoted subspaces are combined in truth coordinates. Their exact numerical union is projected into each anchor's physical R32 complement and aligned by orthogonal Procrustes. Raw anchor-local Schur vectors may not be interpolated. The common promoted dimension is capped at 62 so the R32 plus common modes plus R96 memory remains at or below R320.",
            "",
            "An order-96 stable balanced memory is refit behind the common resolved coordinates at both anchors and must pass the original and midpoint frequency gates. Weak cross-anchor basis alignment selects a conservative local atlas rather than a physical-failure claim.",
            "",
            "No production coefficients, online solver, predictive cycle, or reduced slow evolution is authorized by this manifest.",
            "",
        )),
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
