#!/usr/bin/env python3
"""Freeze the invariant-compatible projection and saved-generator spectrum audit."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25d"
CLASSIFICATION = (
    "invariant_projection_spectrum_manifest_frozen_saved_generator_audit_authorized"
)
PARENT_COMMIT = "dc430aa4f19a52e0f3bee2e510707d8419ae3ea0"
PARENT_PARENT = "1d2d9154f4abfbbd1deefdd433dddf71c10128f4"
PARENT_TREE = "d718fdb00602d7701a5735b2b3c62929851d62f1"

PARENT_ARTIFACT = (
    "causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
ARTIFACT = (
    "causal_inner_invariant_projection_spectrum_manifest_wp10c9d6c7c3b5c4f25d"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_invariant_projection_spectrum_manifest_"
    "wp10c9d6c7c3b5c4f25d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_invariant_projection_spectrum_manifest_"
    "wp10c9d6c7c3b5c4f25d.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_invariant_projection_spectrum_audit_"
    "wp10c9d6c7c3b5c4f25e.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_invariant_projection_spectrum_audit_"
    "wp10c9d6c7c3b5c4f25e.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_INVARIANT_PROJECTION_SPECTRUM_"
    "MANIFEST_WP10C9D6C7C3B5C4F25D_2026-08-17.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TRUTH_DIMENSION = 560
PRIMARY_CELLS = 16
FIELDS_PER_CELL = 5
STORAGE_DIMENSION = PRIMARY_CELLS * FIELDS_PER_CELL
EXPLICIT_MODE_DIMENSION = 2
RESOLVED_DIMENSION = STORAGE_DIMENSION + EXPLICIT_MODE_DIMENSION
CONSERVATIVE_FIELDS = (0, 2, 3)
THERMAL_STRESS_FIELDS = (1, 4)
FREQUENCY_COUNT = 32
FIDUCIAL_CYCLE_SECONDS = 6.7 * 86_400.0
FAST_TIMESTEP_SECONDS = 1.0e-7


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
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _validate_parent() -> tuple[dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("parent projection-failure commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parent projection-failure parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parent projection-failure tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    assembly = _read(PARENT_DIRECTORY / "assembly_metrics.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "single_anchor_descriptor_schema_failed_database_campaign_blocked"
        or summary["physical_failure_detected"]
        or assembly["passed"]
        or assembly["constraint_rowspace_relative_defect"] <= 5.0e-10
        or assembly["new_nonlinear_roots"] != 0
        or assembly["propagated_states"] != 0
    ):
        raise RuntimeError("parent projection failure changed")
    return summary, hashes


def _frequency_grid() -> dict:
    low = 2.0 * math.pi / FIDUCIAL_CYCLE_SECONDS
    high = math.pi / FAST_TIMESTEP_SECONDS
    ratio = (high / low) ** (1.0 / (FREQUENCY_COUNT - 1))
    return {
        "count_excluding_DC": FREQUENCY_COUNT,
        "angular_frequency_min_per_second": low,
        "angular_frequency_max_per_second": high,
        "values_per_second": [low * ratio**index for index in range(FREQUENCY_COUNT)],
        "exact_DC_is_separate": True,
        "high_frequency_samples_are_diagnostic_not_explicit_online_step_constraints": True,
    }


def _contract() -> dict:
    parent_hashes = {
        name: _sha(PARENT_DIRECTORY / name)
        for name in ("descriptor_A.npz", "descriptor_E.npz", "projection.npz")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": parent_hashes,
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_descriptor_assemblies": 0,
            "saved_complete_generator_must_be_reused": True,
            "allowed_seed_local_mapped_height_descriptor_reconstructions": 1,
            "maximum_wall_hours": 2.0,
        },
        "stage_1_projection": {
            "candidate_id": "R82_mapped_MJE_complete_thermal_stress_plus_a2",
            "truth_dimension": TRUTH_DIMENSION,
            "primary_cells": PRIMARY_CELLS,
            "cellwise_storage_dimension": STORAGE_DIMENSION,
            "explicit_mode_dimension": EXPLICIT_MODE_DIMENSION,
            "resolved_dimension": RESOLVED_DIMENSION,
            "mapped_only_field_indices": CONSERVATIVE_FIELDS,
            "mapped_plus_responsive_height_field_indices": THERMAL_STRESS_FIELDS,
            "fixed_Q_rows_are_formed_from_mapped_storage_only": True,
            "coarse_face_36_must_remain_an_exact_group_boundary": True,
            "right_inverse_method": "complete_QR_of_R_transpose_plus_triangular_solve",
            "orthogonal_complement_from_same_complete_QR": True,
            "automatic_84_coordinate_rescue_forbidden": True,
            "failure_requires_a_new_definitions_only_coordinate_manifest": True,
            "pass_requires": {
                "resolved_rank": RESOLVED_DIMENSION,
                "resolved_condition_number_max": 1.0e4,
                "restriction_lifting_identity_max": 5.0e-12,
                "restriction_complement_annihilation_max": 5.0e-12,
                "complement_orthogonality_max": 5.0e-12,
                "constraint_rowspace_relative_defect_max": 5.0e-10,
                "saved_complete_descriptor_relative_parity_max": 5.0e-12,
                "M_J_E_telescope_relative_defect_max": 5.0e-12,
                "a2_dual_biorthogonality_defect_max": 5.0e-10,
                "a2_dual_reaction_annihilation_defect_max": 5.0e-10,
            },
        },
        "stage_2_spectrum_transfer": {
            "requires_stage_1_pass": True,
            "generator_source": "hash_locked_parent_descriptor_A_complete_fixed_Q_generator",
            "output_source": "hash_locked_parent_projection_all_17_coarse_face_M_J_E_rows",
            "frequency_grid": _frequency_grid(),
            "unresolved_operator": "A_z=Z_transpose_A_Z",
            "transfer": "G(s)=D+C_z*(sI-A_z)^(-1)*A_zr",
            "unstable_definition": "real_pole_greater_than_or_equal_to_zero",
            "cycle_scale_diagnostic": "negative_real_pole_with_decay_time_at_least_one_over_omega_min",
            "unstable_modes_may_not_be_fit_as_stable_memory": True,
            "no_memory_coefficients_are_fit_in_this_package": True,
            "pass_requires": {
                "frequency_count_including_DC": FREQUENCY_COUNT + 1,
                "frequency_solve_relative_residual_max": 1.0e-10,
                "transfer_conjugate_symmetry_relative_defect_max": 1.0e-10,
                "complex_schur_reconstruction_relative_defect_max": 1.0e-10,
                "complex_schur_unitarity_defect_max": 5.0e-12,
                "database_roundtrip_bitwise": True,
            },
            "decision": {
                "no_unstable_unresolved_modes": (
                    "definitions_only_mode_selection_and_finite_memory_manifest_authorized"
                ),
                "one_or_more_unstable_unresolved_modes": (
                    "definitions_only_resolved_mode_promotion_manifest_authorized"
                ),
                "numerical_transfer_gate_failure": "spectrum_transfer_audit_failed_stop",
            },
        },
        "claim_boundary": {
            "full_anchor_campaign_authorized": False,
            "memory_fit_authorized_in_this_package": False,
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
    parent_summary, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("projection/spectrum manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("projection/spectrum manifest is already frozen")
    contract = _contract()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "stage_1_projection_authorized": True,
        "stage_2_spectrum_transfer_authorized_only_after_stage_1_pass": True,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_full_descriptor_assembly_executed": False,
        "full_anchor_campaign_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "parent_classification_preserved": parent_summary["classification"],
        "authorized_next": WORK_PACKAGE.replace("25d", "25e"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", contract)
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
            "# Invariant-compatible projection and spectrum manifest WP10c9d6c7c3b5c4f25d",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "This definitions-only package preserves the rejected 82-coordinate pilot and authorizes one saved-generator repair audit. No nonlinear root, propagated state, new full 560-direction descriptor assembly, memory fit, anchor campaign, online solver, or cycle is authorized.",
            "",
            "Stage 1 prospectively rebuilds the 80 cellwise rows from mapped M/J/E and complete thermal/stress storage, then appends the two frozen stable-mode duals. It must preserve the fixed-Q row space without automatically appending two coordinates.",
            "",
            "Stage 2 runs only after Stage 1 passes. It reuses the hash-locked complete generator, measures the unresolved spectrum and 33-point transfer, and classifies unstable modes for explicit promotion rather than hiding them in a stable-memory fit.",
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
