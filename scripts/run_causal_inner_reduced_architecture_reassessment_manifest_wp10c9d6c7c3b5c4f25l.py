#!/usr/bin/env python3
"""Freeze the saved-R32 reduced-architecture reassessment audit."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25l"
CLASSIFICATION = (
    "reduced_architecture_reassessment_manifest_frozen_"
    "saved_R32_memory_selection_authorized"
)
PARENT_COMMIT = "48000ca0168c1d548afbb7896356a851683ab459"
PARENT_PARENT = "4e6230a1e234d67804c8bdd961c2f02c5e851034"
PARENT_TREE = "8375e22ec671f846406e669b4f323c6272238276"

PARENT_ARTIFACT = "causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
ARTIFACT = "causal_inner_reduced_architecture_reassessment_manifest_wp10c9d6c7c3b5c4f25l"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_reduced_architecture_reassessment_manifest_wp10c9d6c7c3b5c4f25l.py"
THIS_TEST = "tests/test_causal_inner_reduced_architecture_reassessment_manifest_wp10c9d6c7c3b5c4f25l.py"
NEXT_RUNNER = "scripts/run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m.py"
NEXT_TEST = "tests/test_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_REDUCED_ARCHITECTURE_"
    "REASSESSMENT_MANIFEST_WP10C9D6C7C3B5C4F25L_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

BASE_ONLINE_DIMENSION = 180
GLOBAL_BALANCED_ORDERS = (24, 48, 96)
COHERENT_CHANNEL_RANKS = (1, 2, 3)
MAXIMUM_MEMORY_DIMENSION = 96
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
        raise RuntimeError("parent R32 rejection commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parent R32 rejection parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parent R32 rejection tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "R32_no_memory_closure_insufficient_architecture_reassessment_required"
        or summary["authorized_next"]
        != "definitions_only_reduced_architecture_reassessment_manifest"
        or not summary["projection_algebra_passed"]
        or not summary["remaining_unresolved_strictly_stable"]
        or summary["physical_failure_detected"]
        or metrics["augmented_resolved_dimension"] != BASE_ONLINE_DIMENSION
    ):
        raise RuntimeError("parent architecture-reassessment authorization changed")
    return summary, metrics, hashes


def _contract() -> dict:
    common_gates = {
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
            "R32_projection_promotion.npz": _sha(PARENT_DIRECTORY / "R32_projection_promotion.npz"),
            "R32_transfer.npz": _sha(PARENT_DIRECTORY / "R32_transfer.npz"),
            "metrics.json": _sha(PARENT_DIRECTORY / "metrics.json"),
            "summary.json": _sha(PARENT_DIRECTORY / "summary.json"),
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_new_truth_anchors": 0,
            "allowed_saved_generator_memory_fits": 12,
            "maximum_wall_hours": 2.0,
        },
        "resolved_backbone": {
            "physical_grid": "nested_conservative_R32",
            "physical_storage_dimension": 160,
            "explicit_a2_dimension": 2,
            "promoted_nonstable_dimension": 18,
            "base_online_dimension": BASE_ONLINE_DIMENSION,
            "remaining_stable_dimension": 380,
            "interior_M_J_E_fluxes_must_remain_single_valued_and_telescope": True,
            "promoted_coordinates_are_single_anchor_diagnostic_until_common_subspace_audit": True,
        },
        "candidate_families": {
            "global_balanced_controls": {
                "orders": GLOBAL_BALANCED_ORDERS,
                "construction": "continuous_time_square_root_balanced_truncation",
            },
            "coherent_spatial_channel_models": {
                "temporal_orders": GLOBAL_BALANCED_ORDERS,
                "spatial_channel_ranks": COHERENT_CHANNEL_RANKS,
                "construction": (
                    "project_balanced_input_and_output_maps_onto_leading_"
                    "aggregate_dynamic_transfer_singular_subspaces"
                ),
                "direct_map_is_never_projected": True,
            },
        },
        "normalization_and_validation": {
            "normalization": "frozen_R32_forcing_observation_direct_two_norm_scaling",
            "training_frequencies": "all_33_parent_frequencies",
            "heldout_frequencies": (
                "one_prospective_midpoint_per_adjacent_parent_interval; arithmetic_"
                "midpoint_for_DC_to_first_positive_and_geometric_thereafter"
            ),
            "heldout_reference": "direct_solve_of_hash_locked_saved_R32_stable_system",
            "candidate_pass_requires_training_and_heldout": common_gates,
            "full_order_numerical_pass_requires": {
                "controllability_gramian_relative_residual_max": 1.0e-8,
                "observability_gramian_relative_residual_max": 1.0e-8,
                "minimum_positive_hankel_singular_value_count": MAXIMUM_MEMORY_DIMENSION,
                "maximum_reference_frequency_solve_relative_residual": 1.0e-10,
            },
        },
        "online_budget": {
            "maximum_memory_dimension": MAXIMUM_MEMORY_DIMENSION,
            "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
            "dimension_ceiling_is_runtime_derived_not_a_physical_gate": True,
            "future_macro_memory_update": "exponential_or_L_stable_implicit_not_explicit_fast_substepping",
        },
        "selection": {
            "select_smallest_passing_memory_dimension": True,
            "prefer_coherent_spatial_channel_model_at_equal_dimension": True,
            "no_post_result_candidate_addition": True,
            "pass": "single_anchor_R32_memory_architecture_selected_cross_anchor_preflight_authorized",
            "no_candidate_pass": "saved_R32_memory_architectures_failed_reconsider_resolved_variables",
            "numerical_failure": "saved_R32_memory_selection_numerical_failure_stop",
        },
        "claim_boundary": {
            "production_memory_coefficients_authorized": False,
            "common_cross_anchor_subspace_certified": False,
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
        raise RuntimeError("architecture reassessment manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("architecture reassessment manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "base_online_dimension": BASE_ONLINE_DIMENSION,
        "maximum_memory_dimension": MAXIMUM_MEMORY_DIMENSION,
        "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
        "global_balanced_orders": GLOBAL_BALANCED_ORDERS,
        "coherent_channel_ranks": COHERENT_CHANNEL_RANKS,
        "saved_R32_memory_selection_authorized": True,
        "parent_classification_preserved": parent_summary["classification"],
        "parent_no_memory_RMS_error": parent_metrics[
            "RMS_normalized_total_transfer_relative_error"
        ],
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": WORK_PACKAGE.replace("25l", "25m"),
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
    })
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join((
            "# Reduced-architecture reassessment manifest WP10c9d6c7c3b5c4f25l",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "The rejected memoryless R32 closure is preserved. This definitions-only package freezes a saved-generator comparison between global balanced memories at orders 24/48/96 and coherent rank-1/2/3 spatial-channel variants at the same temporal orders.",
            "",
            "The certified R180 resolved backbone is unchanged. The memory ceiling is 96 and the prospective online continuous-state ceiling is 320; this supersedes the earlier non-cost-derived R192 ceiling without changing any physical equation or transfer-error gate.",
            "",
            "Both the 33 parent frequencies and 32 prospectively defined midpoint frequencies are binding. No truth root, propagation, generator assembly, cross-anchor campaign, online solver, predictive cycle, or reduced slow evolution is authorized.",
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
