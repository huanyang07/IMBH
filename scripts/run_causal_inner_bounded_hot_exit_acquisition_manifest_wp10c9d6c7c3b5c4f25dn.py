#!/usr/bin/env python3
"""Freeze a bounded full-y470 acquisition of the first hot-side exit candidate."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_branch_candidate_saved_array_screen_wp10c9d6c7c3b5c4f25dm as parent  # noqa: E402
import run_causal_inner_transition_hidden_tangent_wp10c9d6c7c3b5c4f25dk as tangent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dn"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25do"
PARENT_COMMIT = "623ef602837f46d674c78ff3ac58a78c06fb595d"
PARENT_PARENT = "dd1e4b2d8f95643446f44dbd95481a2f087316ab"
PARENT_TREE = "8c682318d306c6fa852f264297859a126e62e5b7"
CLASSIFICATION = (
    "bounded_full_y470_hot_exit_acquisition_manifest_frozen_"
    "stepwise_execution_authorized"
)

TIMESTEP_SECONDS = 1.0e-7
MAXIMUM_NEW_BDF2_ROOTS = 12
MAXIMUM_ROOTS_PER_COMMAND = 1
HIDDEN_SECANT_FRACTION_MAX = 0.25
HIDDEN_EXIT_PERSISTENCE_STEPS = 2
RANK16_HIDDEN_AMPLITUDE_MIN = 5.0e-2
MAXIMUM_MACRO_DRIFT_FROM_SEED = 2.0e-2
ROOT_RESIDUAL_TOLERANCE = 1.0e-10
CONSTRAINT_TOLERANCE = 1.0e-12
LEDGER_TOLERANCE = 1.0e-12
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12
MAXIMUM_SCHUR_CONDITION_NUMBER = 1.0e8
MAXIMUM_SCALED_PRIMITIVE_CHANGE = 5.0e-3
MAXIMUM_HEIGHT_RATIO = 0.12
MINIMUM_OPTICAL_DEPTH = 1.0
MAXIMUM_NEWTON_ITERATIONS = 8
MAXIMUM_LINE_SEARCH_ITERATIONS = 12
MAXIMUM_EXACT_REFRESHES_PER_ROOT = 1
ITERATION_RESERVE_TRIGGER = 6
FAILED_RELATIVE_BACKTRACK_TRIGGER = 4

ARTIFACT = "causal_inner_bounded_hot_exit_acquisition_manifest_wp10c9d6c7c3b5c4f25dn"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_bounded_hot_exit_acquisition_manifest_"
    "wp10c9d6c7c3b5c4f25dn.py"
)
THIS_TEST = (
    "tests/test_causal_inner_bounded_hot_exit_acquisition_manifest_"
    "wp10c9d6c7c3b5c4f25dn.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_bounded_hot_exit_acquisition_"
    "wp10c9d6c7c3b5c4f25do.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_bounded_hot_exit_acquisition_"
    "wp10c9d6c7c3b5c4f25do.py"
)
FIXED_Q_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
MONOLITHIC_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py"
LEGACY_CONTINUATION_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_BOUNDED_HOT_EXIT_ACQUISITION_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DN_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_SUMMARY = parent.CANONICAL_DIRECTORY / "summary.json"
PARENT_METRICS = parent.CANONICAL_DIRECTORY / "branch_candidate_screen_metrics.json"
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "branch_candidate_screen_arrays.npz"
PARENT_ACQUISITION = (
    parent.CANONICAL_DIRECTORY / "branch_candidate_acquisition_contract.json"
)
TANGENT_ARRAYS = tangent.CANONICAL_DIRECTORY / "transition_hidden_tangent_arrays.npz"
GEOMETRY_ARRAYS = parent.geometry.CANONICAL_DIRECTORY / "candidate_geometry_arrays.npz"
PRIMARY_RETRY_DIRECTORY = parent.PRIMARY_RETRY_DIRECTORY
SEED_CHECKPOINT = PRIMARY_RETRY_DIRECTORY / "checkpoint_warm_3.npz"
SEED_CHECKPOINT_JSON = PRIMARY_RETRY_DIRECTORY / "checkpoint_warm_3.json"
SEED_METRICS = PRIMARY_RETRY_DIRECTORY / "metrics_warm_3.json"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("hot-exit manifest parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("hot-exit manifest parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("hot-exit manifest parent tree changed")
    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    tangent_hashes = _checksums(tangent.CANONICAL_DIRECTORY)
    retry_hashes = _checksums(PRIMARY_RETRY_DIRECTORY)
    summary = _read(PARENT_SUMMARY)
    acquisition = _read(PARENT_ACQUISITION)
    seed_checkpoint_json = _read(SEED_CHECKPOINT_JSON)
    seed_metrics = _read(SEED_METRICS)
    seed = _load_npz(SEED_CHECKPOINT)
    if (
        summary["passed"]
        or not summary["screen_completed"]
        or not summary["infrastructure_passed"]
        or not summary["cold_candidate_supported"]
        or summary["hot_candidate_supported"]
        or summary["distinct_cold_hot_pair_supported"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["branch_truth_execution_authorized"]
        or summary["transition_truth_execution_authorized"]
        or not acquisition["authorization_boundaries"][
            "definitions_only_bounded_hot_exit_acquisition_manifest_authorized"
        ]
    ):
        raise RuntimeError("cold-only branch screen classification changed")
    if (
        not seed_checkpoint_json["bitwise_roundtrip"]
        or seed_checkpoint_json["sha256"] != _sha(SEED_CHECKPOINT)
        or not seed_metrics["accepted"]
        or not seed_metrics["acceptance"]["accepted"]
        or int(seed["current_order"]) != 2
        or int(seed["completed_steps"]) != 6
        or not np.isclose(float(seed["elapsed_time_seconds"]), 0.0200006)
        or not bool(seed["has_nonlinear_solver_state"])
    ):
        raise RuntimeError("warm_3 transition seed changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hot-exit manifest requires a clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "tangent_hashes": tangent_hashes,
        "retry_hashes": retry_hashes,
        "seed_elapsed_time_seconds": float(seed["elapsed_time_seconds"]),
        "seed_completed_steps": int(seed["completed_steps"]),
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "purpose": (
            "acquire_the_first_persistent_hot_side_candidate_after_the_saved_"
            "20ms_transition_anchor_using_the_full_exact_fixed_Q_y470_reference"
        ),
        "seed": {
            "checkpoint": str(SEED_CHECKPOINT.relative_to(ROOT)),
            "elapsed_time_seconds": 0.0200006,
            "completed_steps": 6,
            "state_role": "transition_internal_not_hot_branch",
            "checkpoint_sha256": _sha(SEED_CHECKPOINT),
        },
        "execution_order": {
            "one_root_per_command": True,
            "fixed_equal_BDF2_timestep_seconds": TIMESTEP_SECONDS,
            "maximum_new_BDF2_roots": MAXIMUM_NEW_BDF2_ROOTS,
            "checkpoint_every_accepted_root": True,
            "commit_each_stage_before_the_next_root": True,
            "rejected_root_never_propagates": True,
        },
        "coordinate_diagnostic": {
            "reference": "full_exact_y470_coordinate_map",
            "hidden_decomposition": "Z388_Q388_dual_consistent",
            "transition_feature_basis": "rank16_common_hidden_tangent_basis",
            "rank16_is_not_used_as_the_binding_dynamics": True,
            "full470_fallback_is_binding": True,
            "hot_distance_uses_hidden_not_macro_displacement": True,
            "reason": (
                "cold_and_hot_fast_states_share_the_same_slow_fiber_so_a_"
                "large_macro_displacement_is_not_a_valid_hot_exit_requirement"
            ),
        },
        "hot_exit_gate": {
            "saved_secant_hidden_fraction_max": HIDDEN_SECANT_FRACTION_MAX,
            "consecutive_accepted_secants_required": HIDDEN_EXIT_PERSISTENCE_STEPS,
            "rank16_hidden_amplitude_from_20ms_anchor_min": (
                RANK16_HIDDEN_AMPLITUDE_MIN
            ),
            "macro_drift_from_seed_max": MAXIMUM_MACRO_DRIFT_FROM_SEED,
            "all_step_acceptance_and_physical_gates_required": True,
            "candidate_is_unclassified_until_exact_branch_preflight": True,
        },
        "solver_contract": {
            "initial_matrix": "carried_exact_reset_v2_bordered_Broyden_state",
            "forced_exact_assembly_at_iteration_zero": False,
            "refresh_policy": "on_line_search_failure_or_iteration_reserve",
            "maximum_exact_refreshes_per_root": MAXIMUM_EXACT_REFRESHES_PER_ROOT,
            "iteration_reserve_trigger": ITERATION_RESERVE_TRIGGER,
            "failed_relative_backtrack_trigger": FAILED_RELATIVE_BACKTRACK_TRIGGER,
            "maximum_newton_iterations": MAXIMUM_NEWTON_ITERATIONS,
            "maximum_line_search_iterations": MAXIMUM_LINE_SEARCH_ITERATIONS,
            "residual_tolerance": ROOT_RESIDUAL_TOLERANCE,
            "constraint_tolerance": CONSTRAINT_TOLERANCE,
            "ledger_tolerance": LEDGER_TOLERANCE,
            "minimum_reconstruction_factor": MINIMUM_RECONSTRUCTION_FACTOR,
            "maximum_schur_condition_number": MAXIMUM_SCHUR_CONDITION_NUMBER,
            "maximum_scaled_primitive_change": MAXIMUM_SCALED_PRIMITIVE_CHANGE,
            "maximum_height_ratio": MAXIMUM_HEIGHT_RATIO,
            "minimum_optical_depth": MINIMUM_OPTICAL_DEPTH,
        },
        "decision": {
            "persistent_exit_reached": (
                "hot_side_candidate_supported_separate_exact_hot_branch_"
                "preflight_manifest_authorized"
            ),
            "root_or_physical_failure": (
                "hot_exit_acquisition_failed_no_branch_truth_authorized"
            ),
            "budget_exhausted_without_exit": (
                "hot_exit_not_reached_within_bound_trend_diagnosis_required"
            ),
            "branch_label_assigned_by_this_package": False,
        },
        "authorization_boundaries": {
            "execution_in_this_package": False,
            "stepwise_hot_exit_execution_in_next_package": True,
            "branch_root_execution_authorized": False,
            "transition_impulse_fit_authorized": False,
            "online_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "parent_summary": _sha(PARENT_SUMMARY),
            "parent_metrics": _sha(PARENT_METRICS),
            "parent_arrays": _sha(PARENT_ARRAYS),
            "parent_acquisition_contract": _sha(PARENT_ACQUISITION),
            "transition_tangent_arrays": _sha(TANGENT_ARRAYS),
            "candidate_geometry_arrays": _sha(GEOMETRY_ARRAYS),
            "seed_checkpoint": _sha(SEED_CHECKPOINT),
            "seed_checkpoint_json": _sha(SEED_CHECKPOINT_JSON),
            "seed_metrics": _sha(SEED_METRICS),
            "fixed_Q_source": _sha(ROOT / FIXED_Q_SOURCE),
            "monolithic_source": _sha(ROOT / MONOLITHIC_SOURCE),
            "legacy_continuation_runner": _sha(ROOT / LEGACY_CONTINUATION_RUNNER),
        },
    }


def _update_catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
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
    _write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("bounded hot-exit manifest already exists")
    locks = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "hot_exit_acquisition_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_tree": PARENT_TREE,
            "decisive_input_hashes": contract["decisive_input_hashes"],
            "validated_parent_hashes": locks,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "seed_checkpoint_accepted_and_bitwise": True,
        "maximum_new_BDF2_roots": MAXIMUM_NEW_BDF2_ROOTS,
        "one_root_per_command": True,
        "hidden_exit_fraction_gate": HIDDEN_SECANT_FRACTION_MAX,
        "hidden_exit_persistence_steps": HIDDEN_EXIT_PERSISTENCE_STEPS,
        "hot_distance_uses_hidden_not_macro_displacement": True,
        "full470_offline_transition_reference_preserved": True,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "branch_root_execution_authorized": False,
        "transition_impulse_fit_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "execution_runner": EXECUTION_RUNNER,
            "execution_test": EXECUTION_TEST,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": _git("rev-parse", "HEAD"),
            "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                FIXED_Q_SOURCE: _sha(ROOT / FIXED_Q_SOURCE),
                MONOLITHIC_SOURCE: _sha(ROOT / MONOLITHIC_SOURCE),
                LEGACY_CONTINUATION_RUNNER: _sha(ROOT / LEGACY_CONTINUATION_RUNNER),
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Bounded hot-exit acquisition manifest WP10c9d6c7c3b5c4f25dn",
                "",
                "The saved-array screen found a valid 12 ms cold-side candidate but no hot-side candidate. The accepted fixed-Q warm_3 checkpoint remains transition-internal and is frozen as the only execution seed.",
                "",
                f"The next package may execute at most {MAXIMUM_NEW_BDF2_ROOTS} equal BDF2 roots of {TIMESTEP_SECONDS:.1e} s, one root per committed stage. Every root retains the full exact y470 residual and all existing fixed-Q, storage, reaction, ledger, reconstruction, and physical gates.",
                "",
                f"A candidate exit requires {HIDDEN_EXIT_PERSISTENCE_STEPS} consecutive accepted secants with dual-consistent hidden fraction at most {HIDDEN_SECANT_FRACTION_MAX:.2f}, rank-16 hidden amplitude at least {RANK16_HIDDEN_AMPLITUDE_MIN:.2f}, and macro drift from the seed at most {MAXIMUM_MACRO_DRIFT_FROM_SEED:.2f}. Hidden rather than macro displacement defines hot-side separation because both fast branches lie on the same slow fiber.",
                "",
                "This manifest executes no root and authorizes no branch label, branch root, impulse fit, online solver, or reduced slow evolution.",
                "",
                f"Authorized next package: `{AUTHORIZED_NEXT}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if arguments.freeze:
        payload = _freeze()
    else:
        payload = {"input_validation": _validate_parent(require_clean=False), "contract": _contract()}
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
