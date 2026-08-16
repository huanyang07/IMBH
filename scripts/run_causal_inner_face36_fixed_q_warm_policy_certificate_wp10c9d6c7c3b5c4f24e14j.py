#!/usr/bin/env python3
"""Certify one iteration-reserve warm root against a same-history cold root."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
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
    causal_five_field_fixed_q_continuation_state,
    causal_five_field_fixed_q_continuation_states_equal,
    load_causal_five_field_fixed_q_continuation_state,
    save_causal_five_field_fixed_q_continuation_state,
    solve_causal_five_field_fixed_q_bdf,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14j"
ARTIFACT = (
    "causal_inner_face36_fixed_q_warm_policy_certificate_"
    "wp10c9d6c7c3b5c4f24e14j"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
MANIFEST_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14i"
)
E14D_DIRECTORY = e14d.CANONICAL_DIRECTORY
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
TIMESTEP_SECONDS = 1.0e-7


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
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
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


def _validate_manifest() -> dict:
    _checksums(MANIFEST_DIRECTORY)
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "execution_contract.json")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["one_root_warm_policy_execution_authorized"]
        or summary["full_primary_retry_authorized"]
        or contract["warm_root"]["primary_trigger_iteration"] != 6
        or contract["warm_root"]["maximum_newton_iterations"] != 8
        or contract["warm_root"]["maximum_exact_assemblies"] != 1
        or contract["warm_root"]["maximum_scaled_residual"] != 1.0e-10
    ):
        raise RuntimeError("e14i warm-policy authorization changed")
    for relative, digest in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen warm-policy source changed: {relative}")
    return {"summary": summary, "contract": contract, "provenance": provenance}


def _identity(manifest: dict) -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "manifest_summary_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
        "source_hashes": manifest["provenance"]["source_hashes"],
    }


def _solve(label: str, data: dict, checkpoint, identity: dict, *, warm: bool):
    rate, multiplier = e14d._predictors(checkpoint, data["columns"])
    events = []

    def progress(payload: dict) -> None:
        event = _plain(payload)
        events.append(event)
        print(f"e14j {label}: {event}", flush=True)

    began_wall = time.perf_counter()
    began_process = time.process_time()
    result = solve_causal_five_field_fixed_q_bdf(
        data["context"],
        checkpoint.current_primitive_charts,
        TIMESTEP_SECONDS,
        rate,
        multiplier,
        None if warm else e14d._cold_top_left(data, checkpoint, TIMESTEP_SECONDS),
        order=2,
        history=checkpoint.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=checkpoint.q3_target,
        constraint_row_scales=checkpoint.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=checkpoint.next_reaction_channel_transform,
        residual_tolerance=1.0e-10,
        constraint_tolerance=1.0e-12,
        ledger_tolerance=1.0e-12,
        storage_parity_tolerance=1.0e-9,
        minimum_reconstruction_factor=1.0 - 1.0e-12,
        maximum_schur_condition_number=1.0e8,
        maximum_scaled_primitive_change=5.0e-3,
        maximum_newton_iterations=8,
        maximum_line_search_iterations=12,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=1 if warm else 2,
        exact_jacobian_refresh_policy=(
            "on_line_search_failure_or_iteration_reserve"
            if warm
            else "on_line_search_failure"
        ),
        initial_nonlinear_solver_state=(
            checkpoint.nonlinear_solver_state if warm else None
        ),
        initial_exact_jacobian_required=not warm,
        solver_state_provenance=identity,
        physical_state_audit=e14d.e1._state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=0.12,
        minimum_scattering_optical_depth=1.0,
        progress_callback=progress,
    )
    wall = time.perf_counter() - began_wall
    process = time.process_time() - began_process
    metrics = e14d._result_metrics(result, events, wall, process)
    metrics.update({"label": label, "warm": warm})
    return result, metrics


def _checkpoint_roundtrip(result, data: dict, start, identity: dict):
    continuation = causal_five_field_fixed_q_continuation_state(
        result,
        data["context"],
        start.current_primitive_charts,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        elapsed_time_seconds=start.elapsed_time_seconds + TIMESTEP_SECONDS,
        completed_steps=start.completed_steps + 1,
        provenance=identity,
    )
    path = SCRATCH_DIRECTORY / "checkpoint_warm_1.npz"
    timing = {}
    save_causal_five_field_fixed_q_continuation_state(
        path, data["context"], continuation, timing_accumulator=timing
    )
    loaded = load_causal_five_field_fixed_q_continuation_state(
        path,
        data["context"],
        expected_provenance=identity,
        timing_accumulator=timing,
    )
    return loaded, {
        "bitwise_roundtrip": causal_five_field_fixed_q_continuation_states_equal(
            continuation, loaded
        ),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        **timing,
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
                    "scientific_status": (
                        "PASS" if summary["scientific_passed"] else "FAIL"
                    ),
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
        "passed": summary["scientific_passed"],
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
    manifest = _validate_manifest()
    if not _tracked_tree_is_clean():
        raise RuntimeError("warm-policy certificate requires a clean tree")
    if SCRATCH_DIRECTORY.exists() or CANONICAL_DIRECTORY.exists():
        raise RuntimeError("warm-policy output directory already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    identity = _identity(manifest)
    data = e14d.e1._state_data("primary_20ms")
    start = load_causal_five_field_fixed_q_continuation_state(
        E14D_DIRECTORY / "checkpoint_cold_1.npz", data["context"]
    )
    warm_result, warm_metrics = _solve(
        "warm_1_iteration_reserve", data, start, identity, warm=True
    )
    e14d._save_result(
        SCRATCH_DIRECTORY / "result_warm_1.npz", warm_result, warm_metrics
    )
    trigger_reasons = warm_metrics["exact_Jacobian_reasons"]
    warm_scientific = bool(
        warm_result.accepted
        and warm_result.maximum_scaled_residual <= 1.0e-10
        and warm_result.exact_jacobian_assemblies <= 1
        and trigger_reasons == ["iteration_reserve"]
    )
    checkpoint_metrics = None
    cold_result = None
    cold_metrics = None
    endpoint = None
    if warm_scientific:
        _loaded, checkpoint_metrics = _checkpoint_roundtrip(
            warm_result, data, start, identity
        )
        warm_scientific = bool(checkpoint_metrics["bitwise_roundtrip"])
    if warm_scientific:
        cold_result, cold_metrics = _solve(
            "warm_1_same_history_cold_control",
            data,
            start,
            identity,
            warm=False,
        )
        e14d._save_result(
            SCRATCH_DIRECTORY / "result_same_history_cold.npz",
            cold_result,
            cold_metrics,
        )
        endpoint = {
            "maximum_scaled_state_difference": e14d._scaled_state_absolute(
                warm_result.primitive_charts,
                cold_result.primitive_charts,
                data["columns"],
            ),
            "reaction_action_relative_difference": e14d._relative(
                warm_result.scaled_reaction_rate_action_per_s,
                cold_result.scaled_reaction_rate_action_per_s,
            ),
        }
        endpoint["passed"] = bool(
            endpoint["maximum_scaled_state_difference"] <= 1.0e-8
            and endpoint["reaction_action_relative_difference"] <= 1.0e-8
        )
        warm_scientific = bool(cold_result.accepted and endpoint["passed"])
    cost = None
    cost_passed = False
    if warm_scientific and cold_metrics is not None:
        cost = {
            "warm_to_same_history_cold_wall_ratio": (
                warm_metrics["root_wall_seconds"]
                / cold_metrics["root_wall_seconds"]
            ),
            "warm_to_same_history_cold_residual_evaluation_ratio": (
                warm_metrics["function_evaluations"]
                / cold_metrics["function_evaluations"]
            ),
            "warm_wall_seconds": warm_metrics["root_wall_seconds"],
            "same_history_cold_wall_seconds": cold_metrics["root_wall_seconds"],
            "warm_function_evaluations": warm_metrics["function_evaluations"],
            "same_history_cold_function_evaluations": cold_metrics[
                "function_evaluations"
            ],
        }
        cost_passed = bool(
            cost["warm_to_same_history_cold_wall_ratio"] <= 0.75
        )
    if not warm_scientific:
        classification = "warm_policy_failed"
    elif cost_passed:
        classification = "warm_policy_scientific_and_cost_passed"
    else:
        classification = "warm_policy_scientific_passed_cost_failed"
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": warm_scientific,
        "scientific_passed": warm_scientific,
        "cost_passed": cost_passed,
        "one_warm_root_executed": True,
        "accepted_trajectory_horizon_seconds_added": (
            TIMESTEP_SECONDS if warm_scientific else 0.0
        ),
        "parent_classification_preserved": "bounded_continuation_failed",
        "full_primary_retry_manifest_authorized": bool(
            warm_scientific and cost_passed
        ),
        "solver_optimization_manifest_authorized": bool(
            warm_scientific and not cost_passed
        ),
        "full_primary_retry_execution_authorized": False,
        "heldout_continuation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    metrics = {
        "warm_root": warm_metrics,
        "checkpoint_roundtrip": checkpoint_metrics,
        "same_history_cold_control": cold_metrics,
        "endpoint_agreement": endpoint,
        "cost": cost,
        "scientific_passed": warm_scientific,
        "cost_passed": cost_passed,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True)
    for name in ("result_warm_1.npz", "result_same_history_cold.npz", "checkpoint_warm_1.npz"):
        source = SCRATCH_DIRECTORY / name
        if source.exists():
            shutil.copy2(source, CANONICAL_DIRECTORY / name)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(CANONICAL_DIRECTORY / "metrics.json", metrics)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            **identity,
            "tracked_worktree_clean_at_start": True,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name) for name in e14d.THREAD_ENVIRONMENT
            },
        },
    )
    files = tuple(
        path.name
        for path in sorted(CANONICAL_DIRECTORY.iterdir())
        if path.name != "SHA256SUMS.txt"
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files)
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
