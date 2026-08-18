#!/usr/bin/env python3
"""Freeze the complete resolved self-energy closure preflight."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25r"
CLASSIFICATION = (
    "complete_resolved_self_energy_closure_manifest_frozen_"
    "saved_generator_preflight_authorized"
)
PARENT_COMMIT = "5a0606720f66c03171a3d592e52a6ca7b746a8ce"
PARENT_PARENT = "9fe50603e08d7887f7a7fb057264a681a713a098"
PARENT_TREE = "a6dee7820992d38e64f57b6e7e2b0f39f3f51b91"

PARENT_ARTIFACT = (
    "causal_inner_rank_adaptive_common_memory_audit_"
    "wp10c9d6c7c3b5c4f25q"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
GENERATOR_ARTIFACT = (
    "causal_inner_common_resolved_subspace_cross_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25o"
)
GENERATOR_DIRECTORY = ROOT / "results/canonical" / GENERATOR_ARTIFACT
ARTIFACT = (
    "causal_inner_complete_resolved_closure_manifest_"
    "wp10c9d6c7c3b5c4f25r"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_complete_resolved_closure_manifest_"
    "wp10c9d6c7c3b5c4f25r.py"
)
THIS_TEST = (
    "tests/test_causal_inner_complete_resolved_closure_manifest_"
    "wp10c9d6c7c3b5c4f25r.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_complete_resolved_closure_audit_"
    "wp10c9d6c7c3b5c4f25s.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_complete_resolved_closure_audit_"
    "wp10c9d6c7c3b5c4f25s.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COMPLETE_RESOLVED_CLOSURE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25R_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PHYSICAL_R32_DIMENSION = 162
COMMON_RANK = 34
RESOLVED_DIMENSION = PHYSICAL_R32_DIMENSION + COMMON_RANK
MEMORY_ORDERS = (112, 120, 124)
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


def _validate_parent() -> tuple[dict, dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("parent rank-adaptive result commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parent rank-adaptive result parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parent rank-adaptive result tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != "two_anchor_rank_adaptive_common_memory_passed_online_prototype_manifest_authorized"
        or summary["selected_common_rank"] != COMMON_RANK
        or summary["selected_memory_order"] != MEMORY_ORDERS[0]
        or summary["selected_online_continuous_dimension"] != 308
        or summary["authorized_next"]
        != "definitions_only_R32_rank_adaptive_memory_online_prototype_manifest"
        or summary["physical_failure_detected"]
        or not metrics["numerical_passed"]
    ):
        raise RuntimeError("parent complete-closure authorization changed")
    return summary, metrics, hashes


def _error_gates() -> dict:
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
            for name in (
                "summary.json",
                "metrics.json",
                "decisive_basis.npz",
                "decisive_model.npz",
            )
        },
        "saved_generator_hashes": {
            name: _sha(GENERATOR_DIRECTORY / name)
            for name in ("heldout_generator.npz", "common_subspace.npz")
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_new_truth_anchors": 0,
            "balanced_realizations_per_anchor": 1,
            "candidate_memory_orders": list(MEMORY_ORDERS),
            "maximum_wall_hours": 0.75,
            "fail_fast_after_first_joint_pass": True,
        },
        "complete_closure": {
            "resolved_dimension": RESOLVED_DIMENSION,
            "stable_dimension": 560 - RESOLVED_DIMENSION,
            "exact_blocks": {
                "resolved_direct": "R_A_L",
                "stable_forcing": "S_transpose_A_L",
                "resolved_memory_observation": "R_A_S",
                "face_memory_observation": "O_face_S",
                "face_direct": "O_face_L",
                "stable_operator": "S_transpose_A_S",
            },
            "combined_memory_output": (
                "row_normalized_concatenation_of_resolved_derivative_feedback_"
                "and_conservative_face_flux"
            ),
            "candidate_orders": list(MEMORY_ORDERS),
            "online_dimensions": [
                RESOLVED_DIMENSION + order for order in MEMORY_ORDERS
            ],
            "pass_requires_at_both_anchors_on_training_and_heldout": {
                "resolved_self_energy": _error_gates(),
                "conservative_face_flux": _error_gates(),
            },
            "numerical_gates": {
                "coordinate_reconstruction_relative_defect_max": 5.0e-10,
                "controllability_gramian_relative_residual_max": 1.0e-8,
                "observability_gramian_relative_residual_max": 1.0e-8,
                "maximum_frequency_solve_relative_residual_max": 1.0e-10,
                "biorthogonality_defect_max": 5.0e-10,
                "memory_spectral_abscissa_per_second_max": -STABILITY_MARGIN_PER_SECOND,
                "lyapunov_dissipation_relative_residual_max": 1.0e-8,
                "lyapunov_certificate_minimum_eigenvalue_min": 0.0,
            },
            "spectral_fidelity": {
                "exact_and_reduced_nonstable_eigenvalue_count_must_match": True,
                "nonstable_threshold_per_second": -STABILITY_MARGIN_PER_SECOND,
                "bidirectional_nearest_eigenvalue_relative_defect_max": 0.10,
            },
        },
        "online_architecture_if_passed": {
            "state": "x_R196_plus_z_memory",
            "equations": [
                "x_dot_equals_D_resolved_x_plus_C_resolved_z",
                "z_dot_equals_A_memory_z_plus_B_memory_x",
                "face_flux_equals_D_face_x_plus_C_face_z",
            ],
            "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
            "memory_update": "exponential_or_L_stable_implicit",
            "face_flux_must_remain_single_valued_before_conservative_divergence": True,
        },
        "decisions": {
            "first_joint_complete_closure_pass": (
                "two_anchor_complete_R196_memory_closure_passed_"
                "bounded_online_prototype_manifest_authorized"
            ),
            "no_order_through_124_passes": (
                "complete_R196_memory_closure_failed_within_R320_"
                "structured_closure_reassessment_required"
            ),
            "numerical_or_coordinate_gate_fails": (
                "complete_resolved_closure_numerical_failure_stop"
            ),
        },
        "claim_boundary": {
            "production_coefficients_authorized": False,
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
    parent_summary, _, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("complete-closure manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("complete-closure manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "resolved_dimension": RESOLVED_DIMENSION,
        "candidate_memory_orders": list(MEMORY_ORDERS),
        "candidate_online_dimensions": [
            RESOLVED_DIMENSION + order for order in MEMORY_ORDERS
        ],
        "parent_selected_common_rank": parent_summary["selected_common_rank"],
        "parent_selected_face_flux_memory_order": parent_summary[
            "selected_memory_order"
        ],
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "production_coefficients_authorized": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": WORK_PACKAGE.replace("25r", "25s"),
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
            "generator_package_hashes": _checksums(GENERATOR_DIRECTORY),
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
                "# Complete resolved-closure manifest WP10c9d6c7c3b5c4f25r",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The selected R196 plus order-112 memory model certifies conservative face-flux transfer, but a closed online equation also requires stable-memory feedback into all R196 resolved derivatives. This definitions-only package makes that self-energy transfer binding before an integrator is implemented.",
                "",
                "Orders 112, 120, and 124 are tested at both saved anchors. The combined closure must reproduce resolved self-energy and face-flux transfer, preserve the nonstable eigenvalue count, and remain within the R320 online cap.",
                "",
                "No truth assembly, nonlinear root, propagation, online integrator, or predictive cycle is authorized by this manifest.",
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
