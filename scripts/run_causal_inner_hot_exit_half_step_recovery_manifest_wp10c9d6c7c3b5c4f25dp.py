#!/usr/bin/env python3
"""Freeze trust-bound-preserving half-step recovery of hot-exit acquisition."""

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

import run_causal_inner_bounded_hot_exit_acquisition_manifest_wp10c9d6c7c3b5c4f25dn as original_manifest  # noqa: E402
import run_causal_inner_bounded_hot_exit_acquisition_wp10c9d6c7c3b5c4f25do as original  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dp"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dq"
PARENT_COMMIT = "7bed99b07afe5a302b5d4748c110a689c322d000"
PARENT_PARENT = "880b06db6e2c48350464ebba3f0f7b425618ebd8"
PARENT_TREE = "00e17b308a75bfd1b635b532f0ed8e1a621bd3b9"
CLASSIFICATION = (
    "full_step_trust_bound_rejection_preserved_half_step_hot_exit_"
    "recovery_manifest_frozen"
)

TIMESTEP_SECONDS = 5.0e-8
MAXIMUM_NEW_BDF2_ROOTS = 12
MAXIMUM_ROOTS_PER_COMMAND = 1
COLD_START_ROOTS = 2
HIDDEN_SECANT_FRACTION_MAX = original_manifest.HIDDEN_SECANT_FRACTION_MAX
HIDDEN_EXIT_PERSISTENCE_STEPS = original_manifest.HIDDEN_EXIT_PERSISTENCE_STEPS
RANK16_HIDDEN_AMPLITUDE_MIN = original_manifest.RANK16_HIDDEN_AMPLITUDE_MIN
MAXIMUM_MACRO_DRIFT_FROM_SEED = original_manifest.MAXIMUM_MACRO_DRIFT_FROM_SEED
ROOT_RESIDUAL_TOLERANCE = original_manifest.ROOT_RESIDUAL_TOLERANCE
CONSTRAINT_TOLERANCE = original_manifest.CONSTRAINT_TOLERANCE
LEDGER_TOLERANCE = original_manifest.LEDGER_TOLERANCE
MINIMUM_RECONSTRUCTION_FACTOR = original_manifest.MINIMUM_RECONSTRUCTION_FACTOR
MAXIMUM_SCHUR_CONDITION_NUMBER = original_manifest.MAXIMUM_SCHUR_CONDITION_NUMBER
MAXIMUM_SCALED_PRIMITIVE_CHANGE = original_manifest.MAXIMUM_SCALED_PRIMITIVE_CHANGE
MAXIMUM_HEIGHT_RATIO = original_manifest.MAXIMUM_HEIGHT_RATIO
MINIMUM_OPTICAL_DEPTH = original_manifest.MINIMUM_OPTICAL_DEPTH
MAXIMUM_NEWTON_ITERATIONS = original_manifest.MAXIMUM_NEWTON_ITERATIONS
MAXIMUM_LINE_SEARCH_ITERATIONS = original_manifest.MAXIMUM_LINE_SEARCH_ITERATIONS
ITERATION_RESERVE_TRIGGER = original_manifest.ITERATION_RESERVE_TRIGGER
FAILED_RELATIVE_BACKTRACK_TRIGGER = original_manifest.FAILED_RELATIVE_BACKTRACK_TRIGGER

ARTIFACT = "causal_inner_hot_exit_half_step_recovery_manifest_wp10c9d6c7c3b5c4f25dp"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_hot_exit_half_step_recovery_manifest_wp10c9d6c7c3b5c4f25dp.py"
THIS_TEST = "tests/test_causal_inner_hot_exit_half_step_recovery_manifest_wp10c9d6c7c3b5c4f25dp.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HOT_EXIT_HALF_STEP_RECOVERY_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DP_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_STEP_05 = original._stage_directory(5)
PARENT_STEP_06 = original._stage_directory(6)
SEED_CHECKPOINT = PARENT_STEP_05 / "checkpoint_step_05.npz"
SEED_CHECKPOINT_JSON = PARENT_STEP_05 / "checkpoint_step_05.json"
SEED_METRICS = PARENT_STEP_05 / "metrics_step_05.json"
FAILED_METRICS = PARENT_STEP_06 / "metrics_step_06.json"
TANGENT_ARRAYS = original_manifest.TANGENT_ARRAYS
GEOMETRY_ARRAYS = original_manifest.GEOMETRY_ARRAYS
PARENT_ARRAYS = original_manifest.PARENT_ARRAYS
FIXED_Q_SOURCE = original_manifest.FIXED_Q_SOURCE
MONOLITHIC_SOURCE = original_manifest.MONOLITHIC_SOURCE
LEGACY_CONTINUATION_RUNNER = original_manifest.LEGACY_CONTINUATION_RUNNER


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    result = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        result[name] = expected
    return result


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("half-step manifest parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("half-step manifest lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("half-step manifest tree changed")
    accepted_hashes = _checksums(PARENT_STEP_05)
    rejected_hashes = _checksums(PARENT_STEP_06)
    accepted = _read(PARENT_STEP_05 / "summary.json")
    rejected = _read(PARENT_STEP_06 / "summary.json")
    seed_metrics = _read(SEED_METRICS)
    failed = _read(FAILED_METRICS)
    if (
        not accepted["passed"]
        or not accepted["root_accepted"]
        or not accepted["checkpoint_roundtrip_bitwise"]
        or accepted["step_index"] != 5
        or rejected["passed"]
        or rejected["root_accepted"]
        or rejected["step_index"] != 6
        or rejected["next_step_authorized"]
        or failed["maximum_scaled_primitive_change"] != MAXIMUM_SCALED_PRIMITIVE_CHANGE
        or failed["maximum_scaled_residual"] <= ROOT_RESIDUAL_TOLERANCE
        or failed["acceptance"]["failure_reasons"] != ["nonlinear_root", "complete_residual", "Q3"]
        or not failed["acceptance"]["primitive_change_passed"]
        or not failed["acceptance"]["physical_height_passed"]
        or not failed["acceptance"]["physical_optical_depth_passed"]
        or not failed["acceptance"]["storage_parity_passed"]
        or not failed["acceptance"]["reconstruction_passed"]
        or not failed["acceptance"]["reaction_conditioning_passed"]
        or not seed_metrics["accepted"]
    ):
        raise RuntimeError("full-step trust-bound diagnosis changed")
    if (PARENT_STEP_06 / "checkpoint_step_06.npz").exists():
        raise RuntimeError("rejected full step unexpectedly produced history")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("half-step manifest requires a clean tracked tree")
    return {"accepted_hashes": accepted_hashes, "rejected_hashes": rejected_hashes}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_rejection": {
            "package": original.WORK_PACKAGE,
            "step": 6,
            "classification": original.FAILURE_CLASSIFICATION,
            "failure": "full_1e-7_s_BDF2_root_saturated_unchanged_0.005_scaled_increment_trust_bound",
            "physical_failure_selected": False,
            "rejected_state_propagated": False,
        },
        "seed": {
            "checkpoint": str(SEED_CHECKPOINT.relative_to(ROOT)),
            "accepted_step": 5,
            "elapsed_time_seconds": 0.0200011,
            "previous_timestep_seconds": original_manifest.TIMESTEP_SECONDS,
            "checkpoint_sha256": _sha(SEED_CHECKPOINT),
        },
        "execution_order": {
            "one_root_per_command": True,
            "new_timestep_seconds": TIMESTEP_SECONDS,
            "maximum_new_BDF2_roots": MAXIMUM_NEW_BDF2_ROOTS,
            "roots_1_and_2_use_cold_exact_initial_matrix": True,
            "reason_for_two_cold_roots": "variable_step_BDF2_matrix_not_reused_until_current_and_previous_steps_both_equal_5e-8_s",
            "roots_3_onward_may_use_carried_matrix": True,
            "checkpoint_every_accepted_root": True,
            "commit_each_stage_before_next_root": True,
            "rejected_root_never_propagates": True,
        },
        "solver_contract": {
            "full_y470_fixed_Q_residual_binding": True,
            "maximum_scaled_primitive_change": MAXIMUM_SCALED_PRIMITIVE_CHANGE,
            "trust_bound_relaxed": False,
            "cold_maximum_exact_assemblies": 2,
            "warm_maximum_exact_assemblies": 1,
            "warm_refresh_policy": "line_failure_or_iteration_reserve",
            "maximum_newton_iterations": MAXIMUM_NEWTON_ITERATIONS,
            "maximum_line_search_iterations": MAXIMUM_LINE_SEARCH_ITERATIONS,
            "residual_tolerance": ROOT_RESIDUAL_TOLERANCE,
            "constraint_tolerance": CONSTRAINT_TOLERANCE,
            "ledger_tolerance": LEDGER_TOLERANCE,
            "minimum_reconstruction_factor": MINIMUM_RECONSTRUCTION_FACTOR,
            "maximum_schur_condition_number": MAXIMUM_SCHUR_CONDITION_NUMBER,
            "maximum_height_ratio": MAXIMUM_HEIGHT_RATIO,
            "minimum_optical_depth": MINIMUM_OPTICAL_DEPTH,
        },
        "hot_exit_gate": {
            "hidden_secant_fraction_max": HIDDEN_SECANT_FRACTION_MAX,
            "consecutive_accepted_secants_required": HIDDEN_EXIT_PERSISTENCE_STEPS,
            "rank16_hidden_amplitude_min": RANK16_HIDDEN_AMPLITUDE_MIN,
            "macro_drift_from_seed_max": MAXIMUM_MACRO_DRIFT_FROM_SEED,
            "rank16_diagnostic_only_full470_fallback_binding": True,
        },
        "authorization_boundaries": {
            "execution_in_this_package": False,
            "half_step_execution_in_next_package": True,
            "branch_root_execution_authorized": False,
            "transition_impulse_fit_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "accepted_step_05_summary": _sha(PARENT_STEP_05 / "summary.json"),
            "accepted_step_05_checkpoint": _sha(SEED_CHECKPOINT),
            "accepted_step_05_metrics": _sha(SEED_METRICS),
            "rejected_step_06_summary": _sha(PARENT_STEP_06 / "summary.json"),
            "rejected_step_06_metrics": _sha(FAILED_METRICS),
            "original_execution_runner": _sha(ROOT / original.THIS_RUNNER),
            "fixed_Q_source": _sha(ROOT / FIXED_Q_SOURCE),
            "monolithic_source": _sha(ROOT / MONOLITHIC_SOURCE),
        },
    }


def _update_catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("half-step recovery manifest already exists")
    locks = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "half_step_recovery_contract.json", contract)
    _write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_commit": PARENT_COMMIT, "parent_tree": PARENT_TREE, "decisive_input_hashes": contract["decisive_input_hashes"], "validated_parent_hashes": locks})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "full_step_rejection_preserved": True, "trust_bound_unchanged": True, "half_timestep_seconds": TIMESTEP_SECONDS, "maximum_new_BDF2_roots": MAXIMUM_NEW_BDF2_ROOTS, "two_cold_roots_before_matrix_reuse": True, "new_nonlinear_roots": 0, "propagated_states": 0, "branch_root_execution_authorized": False, "reduced_slow_evolution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "runner": THIS_RUNNER, "test": THIS_TEST, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "implementation_commit": _git("rev-parse", "HEAD"), "implementation_tree": _git("rev-parse", "HEAD^{tree}"), "source_hashes": {THIS_RUNNER: _sha(ROOT / THIS_RUNNER), THIS_TEST: _sha(ROOT / THIS_TEST)}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(["# Hot-exit half-step recovery manifest WP10c9d6c7c3b5c4f25dp", "", "The rejected 1e-7 s step remains rejected. The unchanged 0.005 trust bound is recovered prospectively by 5e-8 s BDF2 subdivision from accepted step 5.", "", "The first two half-step roots use cold exact matrices; only later equal-step roots may carry Broyden state. No branch truth or reduced evolution is authorized.", ""]), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
