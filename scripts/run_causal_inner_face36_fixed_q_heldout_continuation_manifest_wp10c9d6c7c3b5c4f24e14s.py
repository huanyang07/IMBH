#!/usr/bin/env python3
"""Freeze the short held-out fixed-Q continuation and reconstruct its seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
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


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14s"
ARTIFACT = (
    "causal_inner_face36_fixed_q_heldout_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14s"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SEED_PATH = ARTIFACT_DIRECTORY / "heldout_seed_continuation.npz"
PARENT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "wp10c9d6c7c3b5c4f24e14r"
)
SEED_ARTIFACT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_heldout_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14s.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_heldout_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14s.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_heldout_continuation_"
    "wp10c9d6c7c3b5c4f24e14t.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_face36_fixed_q_heldout_continuation_"
    "wp10c9d6c7c3b5c4f24e14t.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    EXECUTION_RUNNER,
    EXECUTION_TEST,
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py",
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
)
TIMESTEP_SECONDS = 1.0e-7


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "state": "heldout_16ms",
    "timestep_seconds": TIMESTEP_SECONDS,
    "seed": {
        "source": "adaptive_refresh_heldout_coarse_e12",
        "accepted_BDF1_then_BDF2_history": True,
        "exact_residual_replay_required": True,
        "nonlinear_solver_state_present": False,
    },
    "trajectory": {
        "root_order": ["cold_1", "warm_1", "warm_2"],
        "accepted_horizon_seconds": 3.0e-7,
        "checkpoint_every_accepted_root": True,
        "replay_from": "checkpoint_warm_1.npz",
        "replay_roots": ["warm_2"],
        "bitwise_result_and_continuation_replay": True,
        "accepted_history_only": True,
    },
    "solver_contract": {
        "cold_initial_exact_assembly": True,
        "cold_maximum_exact_assemblies": 2,
        "warm_carried_matrix_at_iteration_zero": True,
        "warm_maximum_exact_assemblies": 1,
        "warm_refresh_policy": "on_line_search_failure_or_iteration_reserve",
        "warm_iteration_reserve_trigger": 6,
        "warm_failed_relative_backtrack_trigger": 4,
        "maximum_newton_iterations": 8,
        "maximum_line_search_iterations": 12,
    },
    "binding_gates": {
        "maximum_scaled_residual": 1.0e-10,
        "maximum_Q3_relative_defect": 1.0e-12,
        "maximum_storage_parity_relative_defect": 1.0e-9,
        "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
        "maximum_reaction_ledger_relative_defect": 1.0e-12,
        "maximum_constraint_action_ledger_relative_defect": 1.0e-12,
        "maximum_raw_Schur_condition_number": 1.0e8,
        "maximum_H_over_R": 0.12,
        "minimum_scattering_optical_depth": 1.0,
        "maximum_scaled_primitive_change": 5.0e-3,
        "maximum_cumulative_absolute_ledger_defect": 3.0e-12,
        "incoming_excision_characteristics": 0,
    },
    "decision": {
        "pass": "heldout_bounded_continuation_certified",
        "fail": "heldout_bounded_continuation_failed",
        "pass_authorizes_only": "definitions_only_operational_timestep_manifest",
    },
    "hard_stops": {
        "no_operational_timestep_execution": True,
        "no_fixed_Q_micro_solver": True,
        "no_physical_microburst": True,
        "no_fast_averaging": True,
        "no_reduced_slow_evolution": True,
    },
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(("git", "diff", "--cached", "--quiet"), cwd=ROOT).returncode == 0
    )


def _checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _validate_parents() -> dict:
    parent_hashes = _checksums(PARENT_DIRECTORY)
    seed_hashes = _checksums(SEED_ARTIFACT)
    parent = _read(PARENT_DIRECTORY / "summary.json")
    seed = _read(SEED_ARTIFACT / "summary.json")
    if (
        parent["classification"]
        != "primary_bounded_continuation_evidence_certified"
        or not parent["passed"]
        or not parent["heldout_continuation_manifest_authorized"]
        or parent["heldout_continuation_execution_authorized"]
        or seed["classification"]
        != "adaptive_refresh_heldout_coarse_passed_refined_ladder_manifest_authorized"
        or not seed["passed"]
    ):
        raise RuntimeError("held-out continuation authorization changed")
    return {
        "primary_evidence_summary": parent,
        "heldout_seed_summary": seed,
        "package_hashes": {
            "primary_evidence": parent_hashes,
            "heldout_seed": seed_hashes,
        },
    }


def _reconstruct_seed(execution_commit: str) -> dict:
    with np.load(SEED_ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    data = e1._state_data("heldout_16ms")
    start = np.asarray(data["state"], dtype=float)
    columns = np.asarray(data["columns"], dtype=float)
    rows = np.asarray(data["rows"], dtype=float)
    bdf1_state = np.asarray(arrays["bdf1_primitive_charts"], dtype=float)
    bdf1_increment = np.asarray(arrays["bdf1_primitive_increment"], dtype=float)
    bdf2_state = np.asarray(arrays["bdf2_primitive_charts"], dtype=float)
    bdf2_increment = np.asarray(arrays["bdf2_primitive_increment"], dtype=float)
    if not np.array_equal(start + bdf1_increment, bdf1_state):
        raise RuntimeError("held-out BDF1 primitive history does not close")
    if not np.array_equal(bdf1_state + bdf2_increment, bdf2_state):
        raise RuntimeError("held-out BDF2 primitive history does not close")
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
        TIMESTEP_SECONDS,
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
        TIMESTEP_SECONDS,
    )
    bdf1_endpoint_reaction = causal_five_field_fixed_q_reaction(
        data["context"], bdf1_state, **common
    )
    bdf2_evaluation = evaluate_causal_five_field_fixed_q_bdf(
        bdf1_state,
        bdf2_state,
        arrays["bdf2_multipliers"],
        initial_reaction.q3_value,
        TIMESTEP_SECONDS,
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
        TIMESTEP_SECONDS,
    )
    bdf2_endpoint_reaction = causal_five_field_fixed_q_reaction(
        data["context"], bdf2_state, **common
    )
    bdf1_replay = np.array_equal(
        bdf1_evaluation.augmented_scaled_residual,
        arrays["bdf1_augmented_scaled_residual"],
    )
    bdf2_replay = np.array_equal(
        bdf2_evaluation.augmented_scaled_residual,
        arrays["bdf2_augmented_scaled_residual"],
    )
    if not bdf1_replay or not bdf2_replay:
        raise RuntimeError("held-out canonical seed residual replay changed")
    provenance = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "execution_commit": execution_commit,
        "seed_arrays_sha256": _sha(SEED_ARTIFACT / "decisive_arrays.npz"),
        "history_origin": "exact_declared_path_reconstruction",
        "state": "heldout_16ms",
    }
    continuation = CausalFiveFieldFixedQContinuationState(
        current_primitive_charts=np.array(bdf2_state, copy=True),
        previous_primitive_charts=np.array(bdf1_state, copy=True),
        history=bdf2_history,
        q3_target=np.array(initial_reaction.q3_value, copy=True),
        constraint_row_scales=np.array(initial_reaction.q3_derivative_norms, copy=True),
        raw_multiplier_predictor=(
            bdf1_endpoint_reaction.raw_schur_inverse @ arrays["bdf2_multipliers"]
        ),
        next_reaction_channel_basis="frozen_normalized",
        next_reaction_channel_transform=np.array(
            bdf2_endpoint_reaction.raw_schur_inverse, copy=True
        ),
        previous_minimum_path_reconstruction_factor=min(
            float(
                bdf2_evaluation.monolithic_evaluation.current_storage_increment
                .minimum_path_reconstruction_factor
            ),
            float(bdf2_evaluation.reaction.minimum_q3_reconstruction_factor),
        ),
        elapsed_time_seconds=data["time_seconds"] + 2.0 * TIMESTEP_SECONDS,
        completed_steps=2,
        current_order=2,
        next_order=2,
        nonlinear_solver_state=None,
        provenance=provenance,
    )
    continuation = _validated_fixed_q_continuation_state(data["context"], continuation)
    timing = {}
    save_causal_five_field_fixed_q_continuation_state(
        SEED_PATH, data["context"], continuation, timing_accumulator=timing
    )
    loaded = load_causal_five_field_fixed_q_continuation_state(
        SEED_PATH,
        data["context"],
        expected_provenance=provenance,
        timing_accumulator=timing,
    )
    roundtrip = causal_five_field_fixed_q_continuation_states_equal(
        continuation, loaded
    )
    if not roundtrip:
        raise RuntimeError("held-out continuation seed roundtrip changed")
    return {
        "schema_version": 1,
        "bdf1_residual_bitwise": bdf1_replay,
        "bdf2_residual_bitwise": bdf2_replay,
        "bdf1_maximum_scaled_residual": float(
            np.max(np.abs(bdf1_evaluation.augmented_scaled_residual))
        ),
        "bdf2_maximum_scaled_residual": float(
            np.max(np.abs(bdf2_evaluation.augmented_scaled_residual))
        ),
        "continuation_roundtrip_bitwise": roundtrip,
        "seed_bytes": SEED_PATH.stat().st_size,
        "seed_sha256": _sha(SEED_PATH),
        "checkpoint_timing": timing,
        "nonlinear_root_solved": False,
        "trajectory_horizon_seconds_added": 0.0,
    }


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
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
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
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


def _freeze() -> dict:
    parents = _validate_parents()
    if not _clean():
        raise RuntimeError("held-out continuation manifest requires a clean tree")
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    seed_metrics = _reconstruct_seed(_git("rev-parse", "HEAD"))
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": "heldout_continuation_manifest_frozen_execution_authorized",
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "heldout_continuation_execution_authorized": True,
        "operational_timestep_manifest_authorized": False,
        "operational_timestep_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write(ARTIFACT_DIRECTORY / "execution_manifest.json", CONTRACT)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", parents)
    _write(ARTIFACT_DIRECTORY / "seed_reconstruction.json", seed_metrics)
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": {relative: _sha(ROOT / relative) for relative in SOURCE_FILES},
            "seed_sha256": seed_metrics["seed_sha256"],
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
        "execution_manifest.json",
        "heldout_seed_continuation.npz",
        "parent_lock.json",
        "provenance.json",
        "seed_reconstruction.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in files),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("select --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
