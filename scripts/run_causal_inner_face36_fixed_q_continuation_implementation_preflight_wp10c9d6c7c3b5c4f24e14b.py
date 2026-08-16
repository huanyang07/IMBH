#!/usr/bin/env python3
"""Certify fixed-Q continuation infrastructure without advancing a state."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

e1 = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    CausalFiveFieldFixedQContinuationState,
    _validated_fixed_q_continuation_state,
    causal_five_field_fixed_q_continuation_states_equal,
    causal_five_field_fixed_q_reaction,
    evaluate_causal_five_field_fixed_q_bdf,
    load_causal_five_field_fixed_q_continuation_state,
    save_causal_five_field_fixed_q_continuation_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_bdf import (  # noqa: E402
    causal_five_field_monolithic_bdf_history,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14b"
ARTIFACT = (
    "causal_inner_face36_fixed_q_continuation_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14b"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PARENT_ARTIFACT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_bounded_continuation_cost_manifest_"
    "wp10c9d6c7c3b5c4f24e14a"
)
SEED_ARTIFACT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_continuation_implementation_"
    "preflight_wp10c9d6c7c3b5c4f24e14b.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_continuation_implementation_"
    "preflight_wp10c9d6c7c3b5c4f24e14b.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    "tests/test_causal_inner_fixed_q.py",
    "tests/test_causal_inner_face36_fixed_q_adaptive_refresh_refined_"
    "ladder_wp10c9d6c7c3b5c4f24e13.py",
)
FOCUSED_TESTS = (
    "tests/test_causal_inner_fixed_q.py",
    "tests/test_causal_inner_monolithic_bdf.py",
    THIS_TEST,
)
SEED_TIMESTEP_SECONDS = 1.0e-7


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"),
            cwd=ROOT,
        ).returncode
        == 0
    )


def _validate_checksums(directory: Path) -> None:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")


def _parent_contract() -> dict:
    _validate_checksums(PARENT_ARTIFACT)
    summary = _read(PARENT_ARTIFACT / "summary.json")
    if (
        not summary["passed"]
        or not summary["implementation_preflight_authorized"]
        or summary["bounded_continuation_execution_authorized"]
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("fixed-Q continuation implementation contract changed")
    return summary


def _run_focused_tests() -> dict:
    command = (sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    began = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_wall_seconds": time.perf_counter() - began,
        "passed": completed.returncode == 0,
    }


def _implementation_contract() -> dict:
    source = (ROOT / SOURCE_FILES[2]).read_text(encoding="utf-8")
    checks = {
        "continuation_after_any_accepted_order": (
            "class CausalFiveFieldFixedQContinuationState" in source
            and "result.order not in (1, 2)" in source
        ),
        "rejected_history_fails_closed": (
            "rejected fixed-Q step cannot define continuation" in source
        ),
        "raw_solver_matrix_serialization": (
            "bordered_matrix_raw_reaction_coordinates" in source
            and 'serialized_multiplier_basis="raw_reaction_channels"' in source
        ),
        "reaction_coordinate_rebase": (
            "causal_five_field_fixed_q_rebase_nonlinear_solver_state" in source
            and "matrix[:, dimensions:] @ transform" in source
        ),
        "warm_initial_exact_assembly_optional": (
            "initial_exact_jacobian_required" in source
            and "initial_nonlinear_solver_state" in source
        ),
        "arbitrary_BDF2_checkpoint": (
            "save_causal_five_field_fixed_q_continuation_state" in source
            and "load_causal_five_field_fixed_q_continuation_state" in source
        ),
        "solver_compatibility_hashes": (
            "_solver_state_compatibility_hashes" in source
            and 'provenance.get("compatibility_hashes")' in source
        ),
        "profiling_counters": (
            "class CausalFiveFieldFixedQStepProfiling" in source
            and "exact_jacobian_wall_seconds" in source
            and "bordered_linear_solve_wall_seconds" in source
        ),
        "legacy_BDF1_restart_preserved": (
            "fixed-Q replay restart must follow BDF1 startup" in source
        ),
        "module_remains_production_neutral": (
            "It does not advance a trajectory" in source
        ),
    }
    return {
        "schema_version": 1,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "physical_operator_changed": False,
        "production_defaults_changed": False,
        "canonical_solver_state_multiplier_basis": "raw_reaction_channels",
        "binding_step_multiplier_basis": "frozen_normalized",
        "warm_exact_refresh_policy": "line_search_failure_only",
    }


def _seed_arrays() -> dict[str, np.ndarray]:
    _validate_checksums(SEED_ARTIFACT)
    with np.load(SEED_ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _maximum_absolute(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def _reconstruct_seed(execution_commit: str) -> tuple[dict, Path]:
    data = e1._state_data("primary_20ms")
    arrays = _seed_arrays()
    start = np.asarray(data["state"], dtype=float)
    columns = np.asarray(data["columns"], dtype=float)
    rows = np.asarray(data["rows"], dtype=float)
    bdf1_state = np.asarray(arrays["bdf1_primitive_charts"], dtype=float)
    bdf1_increment = np.asarray(
        arrays["bdf1_primitive_increment"],
        dtype=float,
    )
    bdf2_state = np.asarray(arrays["bdf2_primitive_charts"], dtype=float)
    bdf2_increment = np.asarray(
        arrays["bdf2_primitive_increment"],
        dtype=float,
    )
    if not np.array_equal(start + bdf1_increment, bdf1_state):
        raise RuntimeError("canonical BDF1 primitive history does not close")
    if not np.array_equal(bdf1_state + bdf2_increment, bdf2_state):
        raise RuntimeError("canonical BDF2 primitive history does not close")
    common = {
        "primitive_column_scales": columns,
        "conservation_row_scales": rows,
        "parent_cell_indices": data["layout"].parent_cell_indices,
        "refinement_ratio": data["layout"].refinement_ratio,
        "exterior_parent_face": 36,
        "guard_end_parent_face": 48,
        "parent_cell_count": 64,
        "maximum_schur_condition_number": 1.0e8,
    }
    initial_reaction = data["reaction"]
    bdf1_evaluation = evaluate_causal_five_field_fixed_q_bdf(
        start,
        bdf1_state,
        arrays["bdf1_multipliers"],
        initial_reaction.q3_value,
        SEED_TIMESTEP_SECONDS,
        data["context"],
        order=1,
        constraint_row_scales=initial_reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=initial_reaction.raw_schur_inverse,
        scaled_primitive_increment=(bdf1_increment / columns).ravel(),
        **common,
    )
    bdf1_history = causal_five_field_monolithic_bdf_history(
        bdf1_increment,
        bdf1_evaluation.monolithic_evaluation.current_storage_increment,
        SEED_TIMESTEP_SECONDS,
    )
    bdf1_endpoint_reaction = causal_five_field_fixed_q_reaction(
        data["context"],
        bdf1_state,
        **common,
    )
    bdf2_evaluation = evaluate_causal_five_field_fixed_q_bdf(
        bdf1_state,
        bdf2_state,
        arrays["bdf2_multipliers"],
        initial_reaction.q3_value,
        SEED_TIMESTEP_SECONDS,
        data["context"],
        order=2,
        history=bdf1_history,
        constraint_row_scales=initial_reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=bdf1_endpoint_reaction.raw_schur_inverse,
        scaled_primitive_increment=(bdf2_increment / columns).ravel(),
        **common,
    )
    bdf2_history = causal_five_field_monolithic_bdf_history(
        bdf2_increment,
        bdf2_evaluation.monolithic_evaluation.current_storage_increment,
        SEED_TIMESTEP_SECONDS,
    )
    bdf2_endpoint_reaction = causal_five_field_fixed_q_reaction(
        data["context"],
        bdf2_state,
        **common,
    )
    minimum_factor = min(
        float(
            bdf2_evaluation.monolithic_evaluation.current_storage_increment
            .minimum_path_reconstruction_factor
        ),
        float(bdf2_evaluation.reaction.minimum_q3_reconstruction_factor),
    )
    provenance = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "execution_commit": execution_commit,
        "seed_arrays_sha256": _sha(SEED_ARTIFACT / "decisive_arrays.npz"),
        "history_origin": "exact_declared_path_reconstruction",
    }
    continuation = CausalFiveFieldFixedQContinuationState(
        current_primitive_charts=np.array(bdf2_state, copy=True),
        previous_primitive_charts=np.array(bdf1_state, copy=True),
        history=bdf2_history,
        q3_target=np.array(initial_reaction.q3_value, copy=True),
        constraint_row_scales=np.array(
            initial_reaction.q3_derivative_norms,
            copy=True,
        ),
        raw_multiplier_predictor=(
            bdf1_endpoint_reaction.raw_schur_inverse
            @ arrays["bdf2_multipliers"]
        ),
        next_reaction_channel_basis="frozen_normalized",
        next_reaction_channel_transform=np.array(
            bdf2_endpoint_reaction.raw_schur_inverse,
            copy=True,
        ),
        previous_minimum_path_reconstruction_factor=minimum_factor,
        elapsed_time_seconds=data["time_seconds"]
        + 2.0 * SEED_TIMESTEP_SECONDS,
        completed_steps=2,
        current_order=2,
        next_order=2,
        nonlinear_solver_state=None,
        provenance=provenance,
    )
    continuation = _validated_fixed_q_continuation_state(
        data["context"],
        continuation,
    )
    path = CANONICAL_DIRECTORY / "canonical_seed_continuation.npz"
    timing = {}
    save_causal_five_field_fixed_q_continuation_state(
        path,
        data["context"],
        continuation,
        timing_accumulator=timing,
    )
    loaded = load_causal_five_field_fixed_q_continuation_state(
        path,
        data["context"],
        expected_provenance=provenance,
        timing_accumulator=timing,
    )
    bdf1_canonical_residual = arrays["bdf1_augmented_scaled_residual"]
    bdf2_canonical_residual = arrays["bdf2_augmented_scaled_residual"]
    metrics = {
        "schema_version": 1,
        "trajectory_executed": False,
        "nonlinear_root_solved": False,
        "history_origin": "exact_declared_path_reconstruction",
        "BDF1_primitive_history_bitwise": bool(
            np.array_equal(start + bdf1_increment, bdf1_state)
        ),
        "BDF2_primitive_history_bitwise": bool(
            np.array_equal(bdf1_state + bdf2_increment, bdf2_state)
        ),
        "BDF1_residual_bitwise": bool(
            np.array_equal(
                bdf1_evaluation.augmented_scaled_residual,
                bdf1_canonical_residual,
            )
        ),
        "BDF2_residual_bitwise": bool(
            np.array_equal(
                bdf2_evaluation.augmented_scaled_residual,
                bdf2_canonical_residual,
            )
        ),
        "maximum_BDF1_residual_reconstruction_defect": _maximum_absolute(
            bdf1_evaluation.augmented_scaled_residual,
            bdf1_canonical_residual,
        ),
        "maximum_BDF2_residual_reconstruction_defect": _maximum_absolute(
            bdf2_evaluation.augmented_scaled_residual,
            bdf2_canonical_residual,
        ),
        "continuation_roundtrip_bitwise": (
            causal_five_field_fixed_q_continuation_states_equal(
                continuation,
                loaded,
            )
        ),
        "complete_mapped_history_finite": bool(
            np.all(np.isfinite(bdf2_history.previous_mapped_storage_increment))
        ),
        "complete_responsive_height_history_finite": bool(
            np.all(
                np.isfinite(
                    bdf2_history
                    .previous_responsive_height_storage_increment
                )
            )
        ),
        "minimum_path_reconstruction_factor": minimum_factor,
        "checkpoint_write_wall_seconds": timing.get(
            "checkpoint_write_wall_seconds",
            0.0,
        ),
        "checkpoint_read_wall_seconds": timing.get(
            "checkpoint_read_wall_seconds",
            0.0,
        ),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": _sha(path),
    }
    metrics["passed"] = bool(
        metrics["BDF1_primitive_history_bitwise"]
        and metrics["BDF2_primitive_history_bitwise"]
        and metrics["BDF1_residual_bitwise"]
        and metrics["BDF2_residual_bitwise"]
        and metrics["continuation_roundtrip_bitwise"]
        and metrics["complete_mapped_history_finite"]
        and metrics["complete_responsive_height_history_finite"]
        and minimum_factor >= 1.0 - 1.0e-12
    )
    return metrics, path


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": (
                        "SUPPORTED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def main() -> None:
    _parent_contract()
    if not _tracked_tree_is_clean():
        raise RuntimeError("continuation preflight requires a clean tracked tree")
    execution_commit = _git("rev-parse", "HEAD")
    execution_tree = _git("rev-parse", "HEAD^{tree}")
    contract = _implementation_contract()
    tests = _run_focused_tests()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    seed_metrics, seed_path = _reconstruct_seed(execution_commit)
    passed = bool(
        contract["all_checks_passed"]
        and tests["passed"]
        and seed_metrics["passed"]
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_continuation_implementation_preflight_certified_"
            "primary_pilot_manifest_authorized"
            if passed
            else "fixed_Q_continuation_implementation_preflight_rejected"
        ),
        "passed": passed,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "production_defaults_changed": False,
        "implementation_contract_passed": contract["all_checks_passed"],
        "focused_tests_passed": tests["passed"],
        "canonical_seed_history_reconstruction_passed": seed_metrics["passed"],
        "primary_pilot_execution_manifest_authorized": passed,
        "bounded_continuation_execution_authorized": False,
        "heldout_continuation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next_artifact": (
            "definitions_only_primary_bounded_continuation_execution_manifest"
            if passed
            else "stop_and_repair_continuation_implementation"
        ),
    }
    _write(CANONICAL_DIRECTORY / "implementation_contract.json", contract)
    _write(CANONICAL_DIRECTORY / "seed_reconstruction_metrics.json", seed_metrics)
    _write(CANONICAL_DIRECTORY / "test_results.json", tests)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": execution_commit,
            "execution_tree": execution_tree,
            "tracked_worktree_clean_at_start": True,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in SOURCE_FILES
            },
            "parent_artifact_hashes": {
                path.name: _sha(path)
                for path in sorted(PARENT_ARTIFACT.iterdir())
                if path.is_file()
            },
            "seed_artifact_hashes": {
                path.name: _sha(path)
                for path in sorted(SEED_ARTIFACT.iterdir())
                if path.is_file()
            },
            "canonical_seed_continuation_sha256": _sha(seed_path),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
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
    files = (
        "canonical_seed_continuation.npz",
        "implementation_contract.json",
        "provenance.json",
        "seed_reconstruction_metrics.json",
        "summary.json",
        "test_results.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
