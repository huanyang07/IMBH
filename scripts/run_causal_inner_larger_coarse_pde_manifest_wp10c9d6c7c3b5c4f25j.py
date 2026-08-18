#!/usr/bin/env python3
"""Freeze the 32-cell conservative coarse-PDE fallback audit."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25j"
CLASSIFICATION = "larger_conservative_coarse_PDE_manifest_frozen_R32_audit_authorized"
PARENT_COMMIT = "0117b58efb3b97af812dc1aa89ad0f96f9b49192"
PARENT_PARENT = "33a76ee1bf491f268c7ef6f57cf232cc9695bbd2"
PARENT_TREE = "250cda936350939e15ff1950c60a61d844c6cffb"

PARENT_ARTIFACT = "causal_inner_finite_memory_selection_audit_wp10c9d6c7c3b5c4f25i"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
GENERATOR_ARTIFACT = "causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c"
GENERATOR_DIRECTORY = ROOT / "results/canonical" / GENERATOR_ARTIFACT
PROMOTION_ARTIFACT = "causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g"
PROMOTION_DIRECTORY = ROOT / "results/canonical" / PROMOTION_ARTIFACT
ARTIFACT = "causal_inner_larger_coarse_pde_manifest_wp10c9d6c7c3b5c4f25j"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_larger_coarse_pde_manifest_wp10c9d6c7c3b5c4f25j.py"
THIS_TEST = "tests/test_causal_inner_larger_coarse_pde_manifest_wp10c9d6c7c3b5c4f25j.py"
NEXT_RUNNER = "scripts/run_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k.py"
NEXT_TEST = "tests/test_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_LARGER_COARSE_PDE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25J_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TRUTH_CELLS = 112
COARSE_CELLS = 32
FIELDS_PER_CELL = 5
STORAGE_DIMENSION = COARSE_CELLS * FIELDS_PER_CELL
EXPLICIT_A2_DIMENSION = 2
BASE_RESOLVED_DIMENSION = STORAGE_DIMENSION + EXPLICIT_A2_DIMENSION
MAXIMUM_PROMOTED_DIMENSION = 30
MAXIMUM_ONLINE_CONTINUOUS_DIMENSION = 192
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
        raise RuntimeError("parent compact-memory rejection commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parent compact-memory rejection parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parent compact-memory rejection tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "compact_finite_memory_failed_larger_conservative_coarse_PDE_fallback_required"
        or summary["authorized_next"]
        != "definitions_only_larger_conservative_coarse_PDE_manifest"
        or not summary["full_order_numerical_passed"]
        or summary["physical_failure_detected"]
        or metrics["selected_order"] is not None
    ):
        raise RuntimeError("parent larger-PDE authorization changed")
    return summary, metrics, hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": {
            "saved_generator": _sha(GENERATOR_DIRECTORY / "descriptor_A.npz"),
            "saved_descriptor": _sha(GENERATOR_DIRECTORY / "descriptor_E.npz"),
            "saved_a2_output": _sha(GENERATOR_DIRECTORY / "projection.npz"),
            "R16_promotion": _sha(PROMOTION_DIRECTORY / "promotion.npz"),
            "compact_memory_rejection": _sha(PARENT_DIRECTORY / "metrics.json"),
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_seed_local_mapped_height_descriptor_reconstructions": 1,
            "allowed_memory_coefficients_fit": 0,
            "maximum_wall_hours": 2.0,
        },
        "R32_conservative_grid": {
            "truth_cells": TRUTH_CELLS,
            "coarse_cells": COARSE_CELLS,
            "construction": "split_each_frozen_R16_group_into_two_contiguous_subgroups",
            "original_R16_boundaries_are_every_second_R32_boundary": True,
            "face_36_exterior_partition_remains_exact_group_boundary": True,
            "mapped_only_field_indices": (0, 2, 3),
            "mapped_plus_responsive_height_field_indices": (1, 4),
            "cellwise_storage_dimension": STORAGE_DIMENSION,
            "frozen_a2_dimension": EXPLICIT_A2_DIMENSION,
            "base_resolved_dimension": BASE_RESOLVED_DIMENSION,
            "interior_M_J_E_fluxes_must_telescope_exactly": True,
            "right_inverse_and_complement": "same_complete_QR_as_certified_R82_projection",
        },
        "ordered_schur_promotion": {
            "promote_every_nonstable_compressed_coordinate": True,
            "stability_margin_per_second": STABILITY_MARGIN_PER_SECOND,
            "maximum_promoted_dimension": MAXIMUM_PROMOTED_DIMENSION,
            "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
            "no_post_result_dimension_expansion": True,
        },
        "algebra_pass_requires": {
            "resolved_rank": BASE_RESOLVED_DIMENSION,
            "resolved_condition_number_max": 2.0e4,
            "restriction_lifting_identity_max": 5.0e-11,
            "restriction_complement_annihilation_max": 5.0e-11,
            "complement_orthogonality_max": 5.0e-11,
            "constraint_rowspace_relative_defect_max": 5.0e-10,
            "saved_complete_descriptor_relative_parity_max": 5.0e-12,
            "M_J_E_telescope_relative_defect_max": 5.0e-12,
            "remaining_unresolved_spectral_abscissa_per_second_max": -STABILITY_MARGIN_PER_SECOND,
        },
        "no_memory_closure_screen": {
            "reference": "exact_33_point_stable_complement_transfer",
            "candidate": "augmented_R32_direct_output_map_with_zero_auxiliary_memory",
            "normalization": "same_frozen_column_and_row_two_norm_scaling_as_compact_memory_screen",
            "pass_requires": {
                "maximum_normalized_total_transfer_relative_error_max": 0.25,
                "RMS_normalized_total_transfer_relative_error_max": 0.10,
                "DC_normalized_total_transfer_relative_error_max": 0.10,
                "maximum_frequency_solve_relative_residual_max": 1.0e-10,
                "database_roundtrip_bitwise": True,
            },
        },
        "decision": {
            "all_gates_pass": "R32_conservative_coarse_PDE_supported_cross_anchor_manifest_authorized",
            "dimension_budget_failed": "R32_promotion_exceeds_online_dimension_budget_stop",
            "closure_error_failed": "R32_no_memory_closure_insufficient_architecture_reassessment_required",
            "numerical_failure": "R32_conservative_projection_audit_failed_stop",
        },
        "claim_boundary": {
            "production_coefficients_authorized": False,
            "cross_anchor_campaign_authorized_only_after_pass": True,
            "online_solver_authorized": False,
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
        raise RuntimeError("larger coarse-PDE manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("larger coarse-PDE manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "coarse_cells": COARSE_CELLS,
        "base_resolved_dimension": BASE_RESOLVED_DIMENSION,
        "maximum_promoted_dimension": MAXIMUM_PROMOTED_DIMENSION,
        "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
        "R32_projection_promotion_and_no_memory_screen_authorized": True,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "memory_fit_executed": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "parent_classification_preserved": parent_summary["classification"],
        "parent_best_compact_RMS_dynamic_error": min(
            item["RMS_normalized_dynamic_transfer_relative_error"]
            for item in parent_metrics["candidate_metrics"]
        ),
        "authorized_next": WORK_PACKAGE.replace("25j", "25k"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_commit": PARENT_COMMIT,
        "parent_parent": PARENT_PARENT,
        "parent_tree": PARENT_TREE,
        "parent_package_hashes": parent_hashes,
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
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    })
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join((
            "# Larger conservative coarse-PDE manifest WP10c9d6c7c3b5c4f25j",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "The compact global memory orders 0/2/4/6 are preserved as rejected. This definitions-only fallback doubles the conservative radial grid from 16 to 32 cells by splitting every frozen R16 group, so every prior boundary and the face-36 exterior partition remain exact boundaries.",
            "",
            "The base state is R162. Every nonstable compressed Schur mode must be promoted, with at most 30 promotions and at most 192 online continuous states. The remaining block must be strictly stable. A zero-auxiliary-memory closure then must pass fixed total-transfer errors 0.25 maximum and 0.10 RMS/DC.",
            "",
            "No nonlinear root, propagation, generator assembly, memory fit, cross-anchor campaign, online solver, predictive cycle, or reduced slow evolution is authorized.",
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
