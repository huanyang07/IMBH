#!/usr/bin/env python3
"""Freeze the exact-hidden pointwise rank-capacity preflight."""

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

import run_causal_inner_square_root_transfer_seeded_manifest_wp10c9d6c7c3b5c4f25z as square_manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ab"
CLASSIFICATION = (
    "exact_hidden_pointwise_rank_capacity_manifest_frozen_"
    "saved_generator_lower_bound_authorized"
)
PARENT_COMMIT = "2c4523b0943eadb191b545f9e258cd0451d208bf"
PARENT_PARENT = "646c553e5d35ed2dff1b6b6d7cfdbdbf0f987f8a"
PARENT_TREE = "9fe5c6587fb6389897b7e6df3950cc5b1007f109"

PARENT_ARTIFACT = (
    "causal_inner_square_root_transfer_seeded_audit_"
    "wp10c9d6c7c3b5c4f25aa"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
FIBER_DIRECTORY = square_manifest.FIBER_DIRECTORY
R32_DIRECTORY = square_manifest.R32_DIRECTORY
PRIMARY_GENERATOR_DIRECTORY = square_manifest.PRIMARY_GENERATOR_DIRECTORY
CROSS_ANCHOR_DIRECTORY = square_manifest.CROSS_ANCHOR_DIRECTORY

ARTIFACT = (
    "causal_inner_hidden_rank_capacity_manifest_"
    "wp10c9d6c7c3b5c4f25ab"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_hidden_rank_capacity_manifest_"
    "wp10c9d6c7c3b5c4f25ab.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hidden_rank_capacity_manifest_"
    "wp10c9d6c7c3b5c4f25ab.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_hidden_rank_capacity_audit_"
    "wp10c9d6c7c3b5c4f25ac.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_hidden_rank_capacity_audit_"
    "wp10c9d6c7c3b5c4f25ac.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HIDDEN_RANK_CAPACITY_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AB_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PHYSICAL_DIMENSION = 162
EXACT_NONSTABLE_DIMENSION = 28
HIDDEN_ORDERS = (112, 120, 124, 128, 130)
TARGET_HIDDEN_ORDER = 130
SAFETY_FRACTION = 0.10


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
        raise RuntimeError("rank-capacity parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("rank-capacity parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("rank-capacity parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "square_root_transfer_seeded_reduction_failed_within_R320_structured_basis_reassessment_required"
        or not summary["base_architecture_passed"]
        or summary["physical_failure_detected"]
        or summary["authorized_next"] is not None
    ):
        raise RuntimeError("square-root basis rejection changed")
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
    transfer = _transfer_gates()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "authority": {
            "source": "explicit_user_authorization_after_f25aa_stop",
            "preserve_f25aa_rejection": True,
            "capacity_result_may_not_promote_a_model": True,
        },
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in (
                "summary.json",
                "metrics.json",
                "decisive_model.npz",
                "candidate_errors.npz",
            )
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
            "allowed_reduced_model_promotions": 0,
            "maximum_wall_hours": 0.5,
            "candidate_hidden_orders": list(HIDDEN_ORDERS),
        },
        "exact_lower_bound": {
            "dynamic_transfer": "Gdyn_b_iw_equals_C_b_times_iwI_minus_Ah_inverse_times_B",
            "pointwise_rank_constraint": "rank_of_any_order_r_dynamic_transfer_at_each_frequency_is_at_most_r",
            "eckart_young_tail": "sqrt_sum_sigma_k_squared_for_k_greater_than_r",
            "dynamic_relative_lower_bound": "tail_Frobenius_norm_divided_by_Gdyn_Frobenius_norm",
            "total_relative_lower_bound": "same_tail_divided_by_D_plus_Gdyn_Frobenius_norm",
            "frequency_sets": ["unchanged_training_ladder", "unchanged_heldout_midpoints_plus_DC"],
            "output_blocks": ["resolved_self_energy", "conservative_face_flux"],
            "target_hidden_order": TARGET_HIDDEN_ORDER,
            "interpretation": "necessary_pointwise_rank_condition_not_a_coherent_realization_witness",
        },
        "binding_gates": {
            "capacity_safety_fraction_of_transfer_gate_max": SAFETY_FRACTION,
            "maximum_frequency_solve_relative_residual_max": 1.0e-10,
            "resolved_self_energy": transfer,
            "conservative_face_flux": transfer,
        },
        "decisions": {
            "rank_130_lower_bound_with_safety_margin_passes": (
                "two_anchor_R130_pointwise_transfer_capacity_not_ruled_out_"
                "direct_structure_preserving_basis_manifest_authorized"
            ),
            "rank_130_lower_bound_exceeds_original_gate": (
                "R320_pointwise_transfer_rank_capacity_impossible_"
                "dimension_or_resolved_coordinates_must_change"
            ),
            "between_safety_and_original_gate": (
                "R320_pointwise_transfer_rank_capacity_marginal_"
                "no_basis_search_authorized"
            ),
            "numerical_integrity_fails": (
                "hidden_rank_capacity_numerical_failure_stop"
            ),
        },
        "claim_boundary": {
            "coherent_dynamic_realizability_certified": False,
            "structure_preserving_basis_certified": False,
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
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
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
        raise RuntimeError("rank-capacity manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("rank-capacity manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "preserved_parent_classification": parent_summary["classification"],
        "candidate_hidden_orders": list(HIDDEN_ORDERS),
        "target_hidden_order": TARGET_HIDDEN_ORDER,
        "capacity_safety_fraction": SAFETY_FRACTION,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25ac",
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
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text("\n".join((
        "# Hidden rank-capacity manifest WP10c9d6c7c3b5c4f25ab",
        "",
        "## Classification",
        "",
        f"`{CLASSIFICATION}`",
        "",
        "This definitions-only package preserves the successful square-root conservative architecture and the failed transplanted-basis classification. It asks only whether hidden order 130 is ruled out by the exact pointwise rank of the normalized transfer matrices.",
        "",
        "At every training and held-out frequency, the Eckart-Young singular-value tail is an exact lower bound for any order-r dynamic transfer matrix. A tenfold margin relative to every inherited transfer gate is required before a direct structure-preserving basis search can be authorized.",
        "",
        "A passing lower bound is necessary but not sufficient: it does not construct a coherent dynamical realization or promote a reduced model.",
        "",
    )), encoding="utf-8")
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
