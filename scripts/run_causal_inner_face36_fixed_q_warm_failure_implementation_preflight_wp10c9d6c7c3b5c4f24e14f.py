#!/usr/bin/env python3
"""Certify warm-failure repairs and replay the rejected endpoint residual."""

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
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

e14d = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    evaluate_causal_five_field_fixed_q_bdf,
    load_causal_five_field_fixed_q_continuation_state,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14f"
ARTIFACT = (
    "causal_inner_face36_fixed_q_warm_failure_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14f"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
E14E_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_failure_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f24e14e"
)
E14D_DIRECTORY = e14d.CANONICAL_DIRECTORY
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_warm_failure_implementation_"
    "preflight_wp10c9d6c7c3b5c4f24e14f.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_warm_failure_implementation_"
    "preflight_wp10c9d6c7c3b5c4f24e14f.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "tests/test_causal_inner_fixed_q.py",
    "tests/test_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
)
TIMESTEP_SECONDS = 1.0e-7
COMMITTED_RESIDUAL = 5.708109263036221e-9


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
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


def _validate_parent_contract() -> dict:
    _checksums(E14E_DIRECTORY)
    summary = _read(E14E_DIRECTORY / "summary.json")
    contract = _read(E14E_DIRECTORY / "diagnosis_contract.json")
    failure = _read(E14E_DIRECTORY / "parent_failure_lock.json")
    if (
        not summary["passed"]
        or not summary["accounting_repair_preflight_authorized"]
        or summary["endpoint_diagnostic_execution_authorized"]
        or summary["warm_policy_execution_authorized"]
        or contract["preserved_parent_classification"]
        != "bounded_continuation_failed"
        or not contract["repair_preflight"][
            "implementation_tests_must_run_without_a_nonlinear_root"
        ]
    ):
        raise RuntimeError("e14e repair authorization changed")
    parent_hashes = _checksums(E14D_DIRECTORY)
    for name, digest in failure["decisive_hashes"].items():
        if parent_hashes.get(name) != digest:
            raise RuntimeError(f"e14d decisive hash changed: {name}")
    return {"summary": summary, "contract": contract, "failure": failure}


def _implementation_checks() -> dict[str, bool]:
    fixed_q = (ROOT / SOURCE_FILES[2]).read_text(encoding="utf-8")
    runner = (ROOT / SOURCE_FILES[3]).read_text(encoding="utf-8")
    return {
        "total_and_since_exact_counters_are_distinct": (
            "total_broyden_updates" in fixed_q
            and "broyden_updates_since_last_exact" in fixed_q
        ),
        "exact_assembly_resets_since_counter": (
            "_broyden_counters_after_exact" in fixed_q
            and "return total, 0" in fixed_q
        ),
        "legacy_counter_semantics_are_explicit": (
            'counter_semantics = "legacy_untrusted_aggregate"' in fixed_q
        ),
        "new_counter_schema_is_trusted": (
            '"exact_reset_v2"' in fixed_q
            and "schema_version: int = 2" in fixed_q
        ),
        "failure_aware_root_accounting_is_committed": (
            "def _failure_aware_root_accounting" in runner
            and '"rejected_candidate_diagnostic_ledgers"' in runner
        ),
        "accepted_horizon_excludes_rejected_roots": (
            "len(accepted) * TIMESTEP_SECONDS" in runner
        ),
        "exclusive_profiling_and_counts_are_committed": (
            "activity_call_counts" in fixed_q
            and "exclusive_wall_seconds" in fixed_q
            and "_fixed_q_exclusive_profile" in fixed_q
        ),
    }


def _reconstruct_legacy_counter(metrics: dict) -> dict:
    cold = metrics["main_roots"]["cold_1"]
    events = cold["event_trace"]
    exact_indices = [
        index
        for index, event in enumerate(events)
        if event["stage"] == "exact_jacobian_refresh"
    ]
    if not exact_indices:
        raise RuntimeError("cold trace has no exact assembly")
    after_last_exact = events[exact_indices[-1] + 1 :]
    accepted_updates = 0
    for index, event in enumerate(after_last_exact[:-1]):
        following = after_last_exact[index + 1]
        if (
            event["stage"] == "line_search"
            and following["stage"] == "newton_iteration"
            and following["maximum_scaled_residual"]
            == event["maximum_scaled_residual"]
        ):
            accepted_updates += 1
    reconstructed = {
        "legacy_serialized_value": cold["Broyden_updates_since_exact"],
        "total_broyden_updates": cold["Broyden_updates"],
        "broyden_updates_since_last_exact": accepted_updates,
        "counter_semantics": "reconstructed_from_committed_event_trace",
        "exact_assemblies": len(exact_indices),
    }
    if reconstructed != {
        "legacy_serialized_value": 6,
        "total_broyden_updates": 6,
        "broyden_updates_since_last_exact": 1,
        "counter_semantics": "reconstructed_from_committed_event_trace",
        "exact_assemblies": 2,
    }:
        raise RuntimeError("committed cold counter reconstruction changed")
    return reconstructed


def _replay_rejected_endpoint(data: dict) -> tuple[dict, dict[str, np.ndarray]]:
    checkpoint = load_causal_five_field_fixed_q_continuation_state(
        E14D_DIRECTORY / "checkpoint_cold_1.npz",
        data["context"],
    )
    if checkpoint.nonlinear_solver_state is None:
        raise RuntimeError("cold checkpoint lacks nonlinear solver state")
    legacy = checkpoint.nonlinear_solver_state
    if (
        legacy.schema_version != 1
        or legacy.counter_semantics != "legacy_untrusted_aggregate"
    ):
        raise RuntimeError("legacy counter semantics were not explicit")
    with np.load(E14D_DIRECTORY / "result_warm_1.npz", allow_pickle=False) as source:
        saved = {name: np.array(source[name], copy=True) for name in source.files}
    old = checkpoint.current_primitive_charts
    new = np.asarray(saved["primitive_charts"], dtype=float)
    increment = np.asarray(saved["primitive_increment"], dtype=float)
    if not np.array_equal(new, old + increment):
        raise RuntimeError("saved rejected endpoint does not close its increment")
    scaled_increment = (increment / data["columns"]).ravel()
    began = time.perf_counter()
    evaluation = evaluate_causal_five_field_fixed_q_bdf(
        old,
        new,
        np.asarray(saved["multipliers"], dtype=float),
        checkpoint.q3_target,
        TIMESTEP_SECONDS,
        data["context"],
        order=2,
        history=checkpoint.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        constraint_row_scales=checkpoint.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=checkpoint.next_reaction_channel_transform,
        scaled_primitive_increment=scaled_increment,
        maximum_schur_condition_number=1.0e8,
    )
    replay = np.asarray(evaluation.augmented_scaled_residual, dtype=float)
    committed = np.asarray(saved["augmented_scaled_residual"], dtype=float)
    maximum = float(np.max(np.abs(replay)))
    metrics = {
        "bitwise_residual_reproduction": bool(np.array_equal(replay, committed)),
        "replayed_maximum_scaled_residual": maximum,
        "committed_maximum_scaled_residual": COMMITTED_RESIDUAL,
        "maximum_residual_array_absolute_difference": float(
            np.max(np.abs(replay - committed))
        ),
        "candidate_state_bitwise_increment_closure": True,
        "legacy_solver_schema_version": legacy.schema_version,
        "legacy_counter_semantics": legacy.counter_semantics,
        "elapsed_wall_seconds": time.perf_counter() - began,
        "nonlinear_root_executed": False,
        "exact_jacobian_assembled": False,
        "continuation_state_constructed": False,
    }
    if not metrics["bitwise_residual_reproduction"] or maximum != COMMITTED_RESIDUAL:
        raise RuntimeError("rejected endpoint residual did not replay bitwise")
    return metrics, {
        "replayed_augmented_scaled_residual": replay,
        "committed_augmented_scaled_residual": committed,
        "rejected_primitive_charts": new,
        "rejected_primitive_increment": increment,
        "rejected_multipliers": np.asarray(saved["multipliers"], dtype=float),
        "rejected_raw_solver_matrix": np.asarray(
            saved["raw_solver_matrix"], dtype=float
        ),
    }


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
                    "scientific_status": "PASS",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
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


def _run() -> dict:
    parent = _validate_parent_contract()
    if not _tracked_tree_is_clean():
        raise RuntimeError("warm-failure implementation preflight requires clean tree")
    checks = _implementation_checks()
    if not all(checks.values()):
        raise RuntimeError("warm-failure implementation checks failed")
    parent_metrics = _read(E14D_DIRECTORY / "metrics.json")
    failure_accounting = e14d._failure_aware_root_accounting(
        parent_metrics["main_roots"]
    )
    if (
        failure_accounting["accepted_roots"] != ["cold_1"]
        or failure_accounting["rejected_roots"] != ["warm_1"]
        or failure_accounting["accepted_trajectory_horizon_seconds"]
        != TIMESTEP_SECONDS
        or failure_accounting["planned_ladder_complete"]
    ):
        raise RuntimeError("failure-aware parent accounting changed")
    counter = _reconstruct_legacy_counter(parent_metrics)
    data = e14d.e1._state_data("primary_20ms")
    replay, arrays = _replay_rejected_endpoint(data)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "warm_failure_accounting_and_endpoint_replay_certified_"
            "endpoint_diagnostic_manifest_authorized"
        ),
        "passed": True,
        "trajectory_executed": False,
        "nonlinear_root_executed": False,
        "exact_jacobian_assembled": False,
        "parent_classification_preserved": "bounded_continuation_failed",
        "accounting_repair_certified": True,
        "endpoint_residual_replay_bitwise": True,
        "endpoint_diagnostic_manifest_authorized": True,
        "endpoint_diagnostic_execution_authorized": False,
        "warm_policy_execution_authorized": False,
        "full_primary_retry_authorized": False,
        "heldout_continuation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next_action": (
            "freeze_separate_nonpropagating_exact_endpoint_diagnostic_manifest"
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(
        CANONICAL_DIRECTORY / "implementation_checks.json",
        {"checks": checks, "all_passed": all(checks.values())},
    )
    _write(CANONICAL_DIRECTORY / "failure_aware_accounting.json", failure_accounting)
    _write(CANONICAL_DIRECTORY / "legacy_counter_reconstruction.json", counter)
    _write(CANONICAL_DIRECTORY / "endpoint_replay.json", replay)
    with (CANONICAL_DIRECTORY / "endpoint_replay_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in SOURCE_FILES
            },
            "parent_e14e_summary_sha256": _sha(
                E14E_DIRECTORY / "summary.json"
            ),
            "parent_e14d_decisive_hashes": parent["failure"]["decisive_hashes"],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in e14d.THREAD_ENVIRONMENT
            },
        },
    )
    files = (
        "endpoint_replay.json",
        "endpoint_replay_arrays.npz",
        "failure_aware_accounting.json",
        "implementation_checks.json",
        "legacy_counter_reconstruction.json",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        raise SystemExit("select --run")
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
