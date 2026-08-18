#!/usr/bin/env python3
"""Freeze the unstable-exact conservative-fiber saved-generator audit."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25t"
CLASSIFICATION = (
    "unstable_exact_conservative_fiber_manifest_frozen_"
    "saved_generator_projector_audit_authorized"
)
PARENT_COMMIT = "aa046b1249a97aa86c0adb22799792222e29f418"
PARENT_PARENT = "b12e979f004c64e83fed72cdb45044ffc34eeea0"
PARENT_TREE = "dc373372c11bbc03ba7cb663e2e273bf7d5e9702"

PARENT_ARTIFACT = (
    "causal_inner_complete_resolved_closure_audit_wp10c9d6c7c3b5c4f25s"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
PRIMARY_GENERATOR_ARTIFACT = (
    "causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c"
)
PRIMARY_GENERATOR_DIRECTORY = ROOT / "results/canonical" / PRIMARY_GENERATOR_ARTIFACT
CROSS_ANCHOR_ARTIFACT = (
    "causal_inner_common_resolved_subspace_cross_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25o"
)
CROSS_ANCHOR_DIRECTORY = ROOT / "results/canonical" / CROSS_ANCHOR_ARTIFACT
R32_ARTIFACT = "causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k"
R32_DIRECTORY = ROOT / "results/canonical" / R32_ARTIFACT

ARTIFACT = (
    "causal_inner_unstable_exact_conservative_fiber_manifest_"
    "wp10c9d6c7c3b5c4f25t"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_unstable_exact_conservative_fiber_manifest_"
    "wp10c9d6c7c3b5c4f25t.py"
)
THIS_TEST = (
    "tests/test_causal_inner_unstable_exact_conservative_fiber_manifest_"
    "wp10c9d6c7c3b5c4f25t.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_UNSTABLE_EXACT_CONSERVATIVE_"
    "FIBER_MANIFEST_WP10C9D6C7C3B5C4F25T_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FULL_DIMENSION = 560
PHYSICAL_R32_DIMENSION = 162
EXPECTED_NONSTABLE_DIMENSION = 28
MAXIMUM_ONLINE_CONTINUOUS_DIMENSION = 320
MINIMUM_STABLE_MEMORY_BUDGET = 112
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
        raise RuntimeError("structured-closure parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("structured-closure parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("structured-closure parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    if (
        summary["classification"]
        != "complete_R196_memory_closure_failed_within_R320_structured_closure_reassessment_required"
        or summary["passed"]
        or not summary["numerical_passed"]
        or summary["physical_failure_detected"]
        or summary["authorized_next"]
        != "definitions_only_structured_resolved_closure_reassessment_manifest"
    ):
        raise RuntimeError("structured-closure reassessment authorization changed")
    for directory in (
        PRIMARY_GENERATOR_DIRECTORY,
        CROSS_ANCHOR_DIRECTORY,
        R32_DIRECTORY,
    ):
        _checksums(directory)
    return summary, hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "decisive_model.npz")
        },
        "saved_input_hashes": {
            "primary_generator": _sha(PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz"),
            "primary_projection": _sha(PRIMARY_GENERATOR_DIRECTORY / "projection.npz"),
            "heldout_generator": _sha(CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz"),
            "R32_projection": _sha(R32_DIRECTORY / "R32_projection_promotion.npz"),
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_new_truth_anchors": 0,
            "maximum_wall_minutes": 20.0,
        },
        "spectral_partition": {
            "algorithm": (
                "independent_ordered_complex_Schur_subspaces_of_A_and_A_transpose_"
                "followed_by_real_SVD_bases_and_left_right_biorthogonalization"
            ),
            "nonstable_threshold_per_second": NONSTABLE_THRESHOLD_PER_SECOND,
            "expected_nonstable_dimension_at_each_anchor": EXPECTED_NONSTABLE_DIMENSION,
            "reduce_nonstable_subspace": False,
            "ordinary_balanced_truncation_of_closed_feedback": False,
        },
        "binding_gates": {
            "selected_nonstable_dimension_equal": EXPECTED_NONSTABLE_DIMENSION,
            "realification_relative_defect_max": 5.0e-10,
            "left_right_overlap_condition_number_max": 1.0e8,
            "biorthogonality_defect_max": 5.0e-10,
            "spectral_projector_idempotence_relative_defect_max": 5.0e-9,
            "spectral_projector_commutator_relative_defect_max": 5.0e-9,
            "right_invariance_relative_defect_max": 5.0e-9,
            "left_invariance_relative_defect_max": 5.0e-9,
            "stable_complement_invariance_relative_defect_max": 5.0e-9,
            "stable_complement_spectral_abscissa_per_second_max": NONSTABLE_THRESHOLD_PER_SECOND,
            "R32_stable_coordinate_rank_equal": PHYSICAL_R32_DIMENSION,
            "R32_stable_coordinate_condition_number_max": 1.0e8,
            "stable_physical_lifting_identity_defect_max": 5.0e-9,
            "stable_physical_lifting_nonstable_annihilation_defect_max": 5.0e-9,
            "nonstable_residual_rank_max": EXPECTED_NONSTABLE_DIMENSION,
            "augmented_nonstable_capture_relative_defect_max": 5.0e-9,
            "cross_anchor_right_principal_cosine_min": 0.8,
            "cross_anchor_left_principal_cosine_min": 0.8,
            "remaining_stable_memory_budget_min": MINIMUM_STABLE_MEMORY_BUDGET,
        },
        "prospective_online_state_if_passed": {
            "coordinates": [
                "Q_R32_conservative_total",
                "a_u_exact_anchor_dependent_nonstable_fiber",
                "z_s_stability_preserving_stable_reduction",
            ],
            "maximum_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
            "nonstable_fiber_transport": (
                "biorthogonal_Grassmann_alignment_with_stability_preserving_"
                "factor_interpolation_not_raw_Schur_vector_interpolation"
            ),
            "stable_reduction": (
                "constrained_Lyapunov_or_dissipativity_preserving_Petrov_Galerkin_"
                "with_conservative_face_flux_rows_binding"
            ),
            "face_flux_single_valued_before_conservative_divergence": True,
        },
        "decisions": {
            "all_binding_gates_pass": (
                "two_anchor_unstable_exact_conservative_fiber_passed_"
                "constrained_lyapunov_stable_reduction_manifest_authorized"
            ),
            "spectral_or_coordinate_gate_fails": (
                "unstable_exact_conservative_fiber_failed_"
                "reduced_architecture_reassessment_required"
            ),
            "numerical_integrity_fails": (
                "unstable_exact_conservative_fiber_numerical_failure_stop"
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
        "passed": True,
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
    parent_summary, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("unstable-exact manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("unstable-exact manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "parent_classification": parent_summary["classification"],
        "physical_R32_dimension": PHYSICAL_R32_DIMENSION,
        "expected_nonstable_dimension": EXPECTED_NONSTABLE_DIMENSION,
        "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": WORK_PACKAGE.replace("25t", "25u"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_commit": PARENT_COMMIT,
        "parent_parent": PARENT_PARENT,
        "parent_tree": PARENT_TREE,
        "parent_package_hashes": parent_hashes,
        "primary_generator_package_hashes": _checksums(PRIMARY_GENERATOR_DIRECTORY),
        "cross_anchor_package_hashes": _checksums(CROSS_ANCHOR_DIRECTORY),
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
        "source_hashes": {
            THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
            THIS_TEST: _sha(ROOT / THIS_TEST),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in (
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    })
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text("\n".join((
        "# Unstable-exact conservative-fiber manifest WP10c9d6c7c3b5c4f25t",
        "",
        "## Classification",
        "",
        f"`{CLASSIFICATION}`",
        "",
        "Ordinary balanced memory reproduced the open-loop transfers but created extra unstable closed-loop poles. This definitions-only package therefore keeps the complete nonstable spectral fiber exact and permits reduction only after exact stable deflation.",
        "",
        "The audit uses only the two hash-locked 560-dimensional generators and the R32 conservative projection. It tests left/right spectral projectors, conservative-coordinate compatibility, cross-anchor alignment, and the remaining R320 memory budget.",
        "",
        "No truth assembly, nonlinear root, propagation, online integrator, or predictive cycle is authorized.",
        "",
    )), encoding="utf-8")
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
