#!/usr/bin/env python3
"""Execute one trust-bound-preserving hot-exit half step per command."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_bounded_hot_exit_acquisition_wp10c9d6c7c3b5c4f25do as base  # noqa: E402
import run_causal_inner_hot_exit_half_step_recovery_manifest_wp10c9d6c7c3b5c4f25dp as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import load_causal_five_field_fixed_q_continuation_state  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dq"
MANIFEST_COMMIT = "16238f4e56974bd843b1defaf9e253d5607fd8ea"
LOCK_CLASSIFICATION = "half_step_hot_exit_recovery_execution_sources_locked_no_root_executed"
CONTINUE_CLASSIFICATION = "half_step_hot_exit_recovery_stage_passed_exit_not_yet_reached"
EXIT_CLASSIFICATION = "persistent_hot_side_exit_candidate_supported_exact_branch_preflight_manifest_authorized"
BUDGET_CLASSIFICATION = "half_step_hot_exit_recovery_budget_exhausted_exit_not_reached"
FAILURE_CLASSIFICATION = "half_step_hot_exit_recovery_failed_no_branch_truth_authorized"

ARTIFACT_PREFIX = "causal_inner_hot_exit_half_step_recovery_wp10c9d6c7c3b5c4f25dq"
LOCK_ARTIFACT = f"{ARTIFACT_PREFIX}_execution_lock"
LOCK_DIRECTORY = ROOT / "results/canonical" / LOCK_ARTIFACT
LOCK_REPORT_PATH = ROOT / "docs/reports/current/CODEX_CAUSAL_INNER_HOT_EXIT_HALF_STEP_RECOVERY_EXECUTION_LOCK_WP10C9D6C7C3B5C4F25DQ_2026-08-21.md"
THIS_RUNNER = "scripts/run_causal_inner_hot_exit_half_step_recovery_wp10c9d6c7c3b5c4f25dq.py"
THIS_TEST = "tests/test_causal_inner_hot_exit_half_step_recovery_wp10c9d6c7c3b5c4f25dq.py"
SCRATCH_ROOT = ROOT / "outputs/checkpoints" / ARTIFACT_PREFIX


def _stage_report_path(index: int) -> Path:
    return ROOT / (
        "docs/reports/current/CODEX_CAUSAL_INNER_HOT_EXIT_HALF_STEP_"
        f"RECOVERY_STEP_{index:02d}_WP10C9D6C7C3B5C4F25DQ_2026-08-21.md"
    )


def _configure_base() -> None:
    replacements = {
        "manifest": manifest,
        "WORK_PACKAGE": WORK_PACKAGE,
        "MANIFEST_COMMIT": MANIFEST_COMMIT,
        "LOCK_CLASSIFICATION": LOCK_CLASSIFICATION,
        "CONTINUE_CLASSIFICATION": CONTINUE_CLASSIFICATION,
        "EXIT_CLASSIFICATION": EXIT_CLASSIFICATION,
        "BUDGET_CLASSIFICATION": BUDGET_CLASSIFICATION,
        "FAILURE_CLASSIFICATION": FAILURE_CLASSIFICATION,
        "ARTIFACT_PREFIX": ARTIFACT_PREFIX,
        "LOCK_ARTIFACT": LOCK_ARTIFACT,
        "LOCK_DIRECTORY": LOCK_DIRECTORY,
        "LOCK_REPORT_PATH": LOCK_REPORT_PATH,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "SCRATCH_ROOT": SCRATCH_ROOT,
        "_stage_report_path": _stage_report_path,
    }
    for name, value in replacements.items():
        setattr(base, name, value)


_configure_base()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _clean_tracked_tree() -> bool:
    return not bool(_git("status", "--short", "--untracked-files=no"))


def _static_execution_contract() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "one_root_per_command": True,
        "timestep_seconds": manifest.TIMESTEP_SECONDS,
        "maximum_steps": manifest.MAXIMUM_NEW_BDF2_ROOTS,
        "cold_exact_initial_roots": [1, 2],
        "warm_carried_roots_begin_at": 3,
        "maximum_scaled_primitive_change": manifest.MAXIMUM_SCALED_PRIMITIVE_CHANGE,
        "trust_bound_relaxed": False,
        "rejected_full_step_propagated": False,
        "rejected_root_never_propagates": True,
        "full_y470_dynamics_binding": True,
        "rank16_coordinates_diagnostic_only": True,
    }


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("half-step recovery manifest commit changed")
    hashes = manifest._checksums(manifest.CANONICAL_DIRECTORY)
    summary = manifest._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = manifest._read(manifest.CANONICAL_DIRECTORY / "half_step_recovery_contract.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["full_step_rejection_preserved"]
        or not summary["trust_bound_unchanged"]
        or contract["execution_order"]["new_timestep_seconds"] != manifest.TIMESTEP_SECONDS
        or not contract["execution_order"]["roots_1_and_2_use_cold_exact_initial_matrix"]
        or contract["solver_contract"]["maximum_scaled_primitive_change"] != 5.0e-3
        or contract["solver_contract"]["trust_bound_relaxed"]
    ):
        raise RuntimeError("half-step recovery manifest changed")
    for name, path in {
        "accepted_step_05_summary": manifest.PARENT_STEP_05 / "summary.json",
        "accepted_step_05_checkpoint": manifest.SEED_CHECKPOINT,
        "accepted_step_05_metrics": manifest.SEED_METRICS,
        "rejected_step_06_summary": manifest.PARENT_STEP_06 / "summary.json",
        "rejected_step_06_metrics": manifest.FAILED_METRICS,
        "original_execution_runner": ROOT / original_manifest.EXECUTION_RUNNER,
        "fixed_Q_source": ROOT / manifest.FIXED_Q_SOURCE,
        "monolithic_source": ROOT / manifest.MONOLITHIC_SOURCE,
    }.items():
        if _sha(path) != contract["decisive_input_hashes"][name]:
            raise RuntimeError(f"half-step decisive input changed: {name}")
    if require_clean and not _clean_tracked_tree():
        raise RuntimeError("half-step execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _execution_lock_payload() -> dict:
    sources = sorted(set((*base.legacy.SOURCE_FILES, *base.FEATURE_SOURCE_FILES, manifest.THIS_RUNNER, THIS_RUNNER, THIS_TEST)))
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": LOCK_CLASSIFICATION,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_contract_sha256": _sha(manifest.CANONICAL_DIRECTORY / "half_step_recovery_contract.json"),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "coordinate_field_arrays_sha256": _sha(base.screen.geometry.FIELD_ARRAYS),
        "seed_checkpoint_sha256": _sha(manifest.SEED_CHECKPOINT),
        "static_execution_contract": _static_execution_contract(),
        "transitive_execution_source_hashes": {relative: _sha(ROOT / relative) for relative in sources},
    }


def _freeze_lock() -> dict:
    if LOCK_DIRECTORY.exists() or LOCK_REPORT_PATH.exists():
        raise RuntimeError("half-step execution lock already exists")
    _validate_manifest(require_clean=True)
    LOCK_DIRECTORY.mkdir(parents=True)
    base._write_json(LOCK_DIRECTORY / "execution_lock.json", _execution_lock_payload())
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": LOCK_CLASSIFICATION, "passed": True, "definitions_only": True, "new_nonlinear_roots": 0, "runner_and_test_hash_locked": True, "step_1_execution_authorized": True, "branch_root_execution_authorized": False, "reduced_slow_evolution_authorized": False}
    base._write_json(LOCK_DIRECTORY / "summary.json", summary)
    base._write_json(LOCK_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": _git("rev-parse", "HEAD"), "implementation_tree": _git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "source_hashes": {THIS_RUNNER: _sha(ROOT / THIS_RUNNER), THIS_TEST: _sha(ROOT / THIS_TEST)}})
    names = sorted(path.name for path in LOCK_DIRECTORY.iterdir())
    (LOCK_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(LOCK_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    LOCK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_REPORT_PATH.write_text("# Half-step hot-exit recovery execution lock\n\nThe runner, tests, full fixed-Q sources, accepted step-5 seed, and unchanged trust bound are locked before any half-step root.\n", encoding="utf-8")
    base._update_catalog(LOCK_ARTIFACT, LOCK_DIRECTORY, summary, "DEFINITIONS_ONLY")
    return summary


def _validate_lock(*, require_clean: bool) -> dict:
    _validate_manifest(require_clean=False)
    hashes = base._checksums(LOCK_DIRECTORY)
    summary = base._read(LOCK_DIRECTORY / "summary.json")
    lock = base._read(LOCK_DIRECTORY / "execution_lock.json")
    if not summary["passed"] or lock != _execution_lock_payload():
        raise RuntimeError("half-step execution lock changed")
    if require_clean and not _clean_tracked_tree():
        raise RuntimeError("half-step stage requires a clean tracked tree")
    return {"hashes": hashes, "lock": lock}


def _root_policy(label: str) -> dict:
    index = int(label.rsplit("_", 1)[1])
    cold = index <= manifest.COLD_START_ROOTS
    return {
        "cold": cold,
        "initial_exact_jacobian_required": cold,
        "maximum_exact_jacobian_refreshes": 2 if cold else 1,
        "use_carried_solver_state": not cold,
        "exact_jacobian_refresh_policy": "on_line_search_failure" if cold else "on_line_search_failure_or_iteration_reserve",
    }


@contextmanager
def _stage_runtime(scratch: Path):
    replacements = {"WORK_PACKAGE": WORK_PACKAGE, "ARTIFACT": ARTIFACT_PREFIX, "SCRATCH_DIRECTORY": scratch, "THIS_RUNNER": THIS_RUNNER, "THIS_TEST": THIS_TEST, "_root_policy": _root_policy}
    original = {name: getattr(base.legacy, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base.legacy, name, value)
        with base.legacy._legacy_runtime():
            yield
    finally:
        for name, value in original.items():
            setattr(base.legacy, name, value)


def _input_semantics(index: int, continuation) -> None:
    solver = continuation.nonlinear_solver_state
    if continuation.current_order != 2 or continuation.next_order != 2 or solver is None or continuation.next_reaction_channel_basis != "frozen_normalized":
        raise RuntimeError("half-step continuation semantics changed")
    if index == 1:
        expected_history = original_manifest.TIMESTEP_SECONDS
        expected_current = original_manifest.TIMESTEP_SECONDS
        expected_previous = original_manifest.TIMESTEP_SECONDS
    elif index == 2:
        expected_history = manifest.TIMESTEP_SECONDS
        expected_current = manifest.TIMESTEP_SECONDS
        expected_previous = original_manifest.TIMESTEP_SECONDS
    else:
        expected_history = manifest.TIMESTEP_SECONDS
        expected_current = manifest.TIMESTEP_SECONDS
        expected_previous = manifest.TIMESTEP_SECONDS
    if continuation.history.previous_timestep_seconds != expected_history or solver.current_timestep_seconds != expected_current or solver.previous_timestep_seconds != expected_previous:
        raise RuntimeError("half-step BDF/solver timestep lineage changed")


def _previous_coordinate(index: int, static: dict) -> np.ndarray:
    if index == 1:
        arrays = base._load_npz(manifest.PARENT_STEP_05 / "hot_exit_feature_arrays.npz")
        return np.asarray(arrays["current_coordinate470"])
    arrays = base._load_npz(base._stage_directory(index - 1) / "hot_exit_feature_arrays.npz")
    return np.asarray(arrays["current_coordinate470"])


def _static_feature_data() -> dict:
    tangent_arrays = base._load_npz(manifest.TANGENT_ARRAYS)
    geometry_arrays = base._load_npz(manifest.GEOMETRY_ARRAYS)
    screen_arrays = base._load_npz(manifest.PARENT_ARRAYS)
    field_arrays = base._load_npz(base.screen.geometry.FIELD_ARRAYS)
    field = base.screen.geometry.field_manifest.ForwardQuadraticAuthenticCenterField(
        field_arrays
    )
    return {
        "model": field.model,
        "macro_restriction": tangent_arrays["macro_restriction_R82"],
        "hidden_basis": tangent_arrays["hidden_basis_Z388"],
        "hidden_dual": tangent_arrays["hidden_dual_Q388"],
        "rank16_basis": tangent_arrays["selected_hidden_basis388"],
        "anchor_coordinate": geometry_arrays[
            "candidate_absolute_y470_coordinates"
        ][5],
        "seed_coordinate": screen_arrays[
            "candidate_absolute_y470_coordinates"
        ][-1],
    }


def _run_step(index: int) -> dict:
    if index < 1 or index > manifest.MAXIMUM_NEW_BDF2_ROOTS:
        raise ValueError("half-step index outside frozen budget")
    _validate_lock(require_clean=True)
    base._prior_stage(index)
    input_checkpoint = base._input_checkpoint(index)
    if not input_checkpoint.exists():
        raise RuntimeError("half-step input checkpoint missing")
    scratch = base._scratch_directory(index)
    if scratch.exists():
        raise RuntimeError("half-step scratch stage already exists")
    scratch.mkdir(parents=True)
    identity = base._execution_identity(index, input_checkpoint)
    base._write_json(scratch / "execution_identity.json", identity)
    data = base.legacy.e14d.e1._state_data("primary_20ms")
    continuation = load_causal_five_field_fixed_q_continuation_state(input_checkpoint, data["context"])
    _input_semantics(index, continuation)
    label = f"step_{index:02d}"
    root_passed = False; root_metrics = {}; next_continuation = None
    try:
        with _stage_runtime(scratch):
            result, next_continuation, root_metrics = base.legacy.e14d._advance(label, data, continuation, manifest.TIMESTEP_SECONDS, identity)
        root_passed = bool(root_metrics["root_passed"] and result.accepted)
    except base.legacy.e14d.BindingRootFailure as error:
        root_metrics = error.metrics
    feature_metrics = None; feature_arrays = None
    if root_passed and next_continuation is not None:
        static = _static_feature_data()
        feature_metrics, feature_arrays = base._exit_features(static, _previous_coordinate(index, static), next_continuation.current_primitive_charts)
        feature_metrics.update({"step_index": index, "elapsed_time_seconds": next_continuation.elapsed_time_seconds, "completed_steps": next_continuation.completed_steps})
        root_metrics["elapsed_time_seconds"] = next_continuation.elapsed_time_seconds
        base._write_json(scratch / f"metrics_{label}.json", root_metrics)
    else:
        root_metrics.setdefault("elapsed_time_seconds", None)
    return base._canonicalize_stage(index, scratch, input_checkpoint, root_metrics, feature_metrics, feature_arrays, root_passed=root_passed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-lock", action="store_true")
    parser.add_argument("--validate-lock", action="store_true")
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if sum((args.freeze_lock, args.validate_lock, args.next, args.step is not None)) != 1:
        parser.error("select exactly one mode")
    if args.freeze_lock:
        payload = _freeze_lock()
    elif args.validate_lock:
        payload = _validate_lock(require_clean=False)
    else:
        index = base._next_step_index() if args.next else int(args.step)
        payload = _run_step(index)
    print(json.dumps(base._plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
