#!/usr/bin/env python3
"""Fit, replay, and benchmark the truth-free hybrid phase engine."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.hybrid_phase_memory import (  # noqa: E402
    ConservativeHybridPhaseEngine,
    ConservativePhaseMode,
    HybridPhaseState,
)
import run_causal_inner_hybrid_phase_engine_manifest_wp10c9d6c7c3b5c4f25e2 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e3"
PASS_CLASSIFICATION = (
    "truth_free_hybrid_phase_engine_passed_observed_modes_"
    "online_cost_feasible_complete_cycle_calibration_missing"
)
FAIL_CLASSIFICATION = "truth_free_hybrid_phase_engine_failed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e4"

ARTIFACT = "causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3.py"
THIS_TEST = "tests/test_causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HYBRID_PHASE_ENGINE_"
    "WP10C9D6C7C3B5C4F25E3_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _rank_model(
    hidden: np.ndarray,
    times: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    tiny = np.finfo(float).tiny
    origin = hidden[0]
    departures = hidden[train] - origin
    rates = np.diff(hidden[train], axis=0) / np.diff(times[train])[:, None]
    combined = np.vstack(
        (
            departures / max(float(np.linalg.norm(departures)), tiny),
            rates / max(float(np.linalg.norm(rates)), tiny),
        )
    )
    _, singular, right = np.linalg.svd(combined, full_matrices=False)
    energy = np.cumsum(singular * singular) / np.sum(singular * singular)
    rank = int(np.searchsorted(energy, manifest.ENERGY_CAPTURE_TARGET) + 1)
    basis = right[:rank].T
    coefficients = departures @ basis
    capture = 1.0 - float(
        np.linalg.norm(departures - coefficients @ basis.T) ** 2
        / max(float(np.linalg.norm(departures) ** 2), tiny)
    )
    return origin, basis, coefficients, rank, capture


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = manifest.architecture.manifest.tube.manifest.geometry
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "engine_contract.json")
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("hybrid engine manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen hybrid engine source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hybrid engine execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _build_engine() -> tuple[ConservativeHybridPhaseEngine, dict, dict[str, np.ndarray]]:
    helper = manifest.architecture.manifest.tube.manifest.geometry
    candidates = helper._load_npz(manifest.architecture.manifest.CANDIDATE_DIRECTORY / "candidate_geometry_arrays.npz")
    tangent = helper._load_npz(manifest.architecture.manifest.TANGENT_DIRECTORY / "transition_hidden_tangent_arrays.npz")
    geometry = helper._load_npz(manifest.architecture.manifest.GEOMETRY_DIRECTORY / "geometry_arrays.npz")
    tube = helper._load_npz(
        manifest.architecture.manifest.tube.CANONICAL_DIRECTORY
        / "tube_model_and_validation.npz"
    )
    times = np.asarray(candidates["candidate_times_seconds"], dtype=float)
    coordinates = np.asarray(candidates["candidate_absolute_y470_coordinates"], dtype=float)
    restriction = np.asarray(tangent["macro_restriction_R82"], dtype=float)
    hidden_dual = np.asarray(tangent["hidden_dual_Q388"], dtype=float)
    hidden_lift = np.asarray(tangent["hidden_basis_Z388"], dtype=float)
    macro_lift = np.asarray(geometry["macro_lift_L470x82"], dtype=float)
    macro = (restriction @ coordinates.T).T
    hidden = (hidden_dual @ coordinates.T).T
    train = np.asarray(manifest.COLD_TRAIN_INDICES, dtype=int)
    holdout = np.asarray(manifest.COLD_HOLDOUT_INDICES, dtype=int)
    origin, basis, coefficients, rank, capture = _rank_model(hidden, times, train)
    cold_phase_all = (times - times[0]) / (times[-1] - times[0])
    cold_phase_train = cold_phase_all[train]
    cold_mode = ConservativePhaseMode(
        name="cold_observed",
        phase_knots=cold_phase_train,
        phase_speeds_per_second=np.full(
            len(train) - 1, 1.0 / (times[-1] - times[0]), dtype=float
        ),
        macro_ledger_knots=macro[train] - macro[[0]],
        hidden_coefficient_knots=coefficients,
        hidden_origin=origin,
        hidden_embedding_basis=basis,
        macro_lift=macro_lift,
        hidden_lift=hidden_lift,
        macro_restriction=restriction,
    )
    transition_mode = ConservativePhaseMode(
        name="fixed_Q_transition_observed",
        phase_knots=np.asarray(tube["progress_knots_s"], dtype=float),
        phase_speeds_per_second=np.asarray(
            tube["progress_speeds_per_second"], dtype=float
        ),
        macro_ledger_knots=np.asarray(tube["macro_ledger_table82"], dtype=float),
        hidden_coefficient_knots=np.asarray(
            tube["hidden_coefficient_table"], dtype=float
        ),
        hidden_origin=np.asarray(tube["hidden_origin388"], dtype=float),
        hidden_embedding_basis=np.asarray(
            tube["hidden_embedding_basis388xr"], dtype=float
        ),
        macro_lift=macro_lift,
        hidden_lift=hidden_lift,
        macro_restriction=restriction,
    )
    engine = ConservativeHybridPhaseEngine(
        {cold_mode.name: cold_mode, transition_mode.name: transition_mode},
        {cold_mode.name: transition_mode.name, transition_mode.name: None},
    )
    data = {
        "times": times,
        "coordinates": coordinates,
        "macro": macro,
        "hidden": hidden,
        "cold_phase_all": cold_phase_all,
        "train": train,
        "holdout": holdout,
        "transition_times": np.asarray(geometry["trajectory_times_seconds"], dtype=float),
        "transition_coordinates": np.asarray(
            geometry["trajectory_coordinates470"], dtype=float
        ),
        "transition_macro": np.asarray(
            geometry["trajectory_macro_coordinates82"], dtype=float
        ),
        "restriction": restriction,
        "macro_lift": macro_lift,
        "hidden_lift": hidden_lift,
    }
    fit = {
        "cold_hidden_embedding_rank": rank,
        "cold_training_displacement_capture": capture,
    }
    return engine, {**data, **fit}, {
        "cold_hidden_origin388": origin,
        "cold_hidden_embedding_basis388xr": basis,
        "cold_hidden_coefficient_table": coefficients,
        "cold_phase_knots": cold_phase_train,
        "cold_phase_speeds_per_second": cold_mode.phase_speeds_per_second,
        "cold_macro_ledger_table82": cold_mode.macro_ledger_knots,
        "transition_phase_knots": transition_mode.phase_knots,
        "transition_phase_speeds_per_second": transition_mode.phase_speeds_per_second,
        "transition_macro_ledger_table82": transition_mode.macro_ledger_knots,
        "transition_hidden_coefficient_table": transition_mode.hidden_coefficient_knots,
    }


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    engine, data, model_arrays = _build_engine()
    cold = engine.modes["cold_observed"]
    transition = engine.modes["fixed_Q_transition_observed"]
    coordinates = data["coordinates"]
    macro = data["macro"]
    holdout = data["holdout"]
    phases = data["cold_phase_all"]
    predicted_macro = np.stack([macro[0] + cold.ledger(s) for s in phases])
    predicted_coordinates = np.stack(
        [cold.decode(q, s) for q, s in zip(predicted_macro, phases, strict=True)]
    )
    cold_steps = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cold_path = float(np.sum(cold_steps))
    cold_macro_path = float(np.sum(np.linalg.norm(np.diff(macro, axis=0), axis=1)))
    holdout_errors = np.linalg.norm(
        predicted_coordinates[holdout] - coordinates[holdout], axis=1
    )
    macro_errors = np.linalg.norm(predicted_macro[holdout] - macro[holdout], axis=1)
    train = data["train"]
    local_chords = []
    for index in holdout:
        left = int(train[train < index][-1])
        right = int(train[train > index][0])
        local_chords.append(float(np.linalg.norm(coordinates[right] - coordinates[left])))
    local_chords = np.asarray(local_chords)

    start = HybridPhaseState(macro[0], 0.0, cold.name)
    event = engine.advance(start, cold.duration_seconds)
    event_decode = engine.decode(event.state)
    event_macro_discontinuity = float(np.linalg.norm(event.state.macro_state - macro[-1]))
    event_state_jump = float(np.linalg.norm(event_decode - coordinates[-1]))
    restart_payload = json.loads(json.dumps(event.state.to_payload()))
    restarted = HybridPhaseState.from_payload(restart_payload)
    restart_bitwise = bool(
        np.array_equal(restarted.macro_state, event.state.macro_state)
        and restarted.phase == event.state.phase
        and restarted.mode == event.state.mode
        and restarted.elapsed_seconds == event.state.elapsed_seconds
        and restarted.event_count == event.state.event_count
    )
    transition_result = engine.advance(restarted, transition.duration_seconds)
    transition_decode = engine.decode(transition_result.state)
    transition_truth = data["transition_coordinates"]
    transition_path = float(
        np.sum(np.linalg.norm(np.diff(transition_truth, axis=0), axis=1))
    )
    transition_endpoint_error = float(
        np.linalg.norm(transition_decode - transition_truth[-1])
    )
    single = engine.advance(
        start, cold.duration_seconds + transition.duration_seconds
    )
    single_equals_staged = bool(
        np.array_equal(single.state.macro_state, transition_result.state.macro_state)
        and single.state.phase == transition_result.state.phase
        and single.state.mode == transition_result.state.mode
        and single.state.elapsed_seconds == transition_result.state.elapsed_seconds
        and single.state.event_count == transition_result.state.event_count
    )
    all_decoded = np.vstack((predicted_coordinates, event_decode, transition_decode))
    all_macro = np.vstack((predicted_macro, event.state.macro_state, transition_result.state.macro_state))
    macro_closure = float(
        np.max(
            np.linalg.norm(
                (data["restriction"] @ all_decoded.T).T - all_macro, axis=1
            )
        )
    )

    update_iterations = 20_000
    decode_iterations = 2_000
    began = time.perf_counter()
    accumulator = 0.0
    for index in range(update_iterations):
        phase = (index % 1000) / 1000.0
        accumulator += float(cold.ledger(phase)[0])
    update_wall = time.perf_counter() - began
    began = time.perf_counter()
    for index in range(decode_iterations):
        phase = (index % 1000) / 1000.0
        accumulator += float(cold.decode(macro[0] + cold.ledger(phase), phase)[0])
    decode_wall = time.perf_counter() - began
    if not np.isfinite(accumulator):
        raise RuntimeError("online benchmark accumulator is nonfinite")
    projected_wall = manifest.ONLINE_MACROSTEPS * (
        update_wall / update_iterations + decode_wall / decode_iterations
    )
    table_bytes = int(
        sum(value.nbytes for value in model_arrays.values())
        + data["macro_lift"].nbytes
        + data["hidden_lift"].nbytes
    )
    gate_values = {
        "cold_hidden_embedding_rank": data["cold_hidden_embedding_rank"],
        "cold_training_displacement_capture": data[
            "cold_training_displacement_capture"
        ],
        "maximum_cold_holdout_error_over_path": float(
            np.max(holdout_errors) / cold_path
        ),
        "maximum_cold_holdout_error_over_local_chord": float(
            np.max(holdout_errors / local_chords)
        ),
        "maximum_cold_macro_ledger_error_over_path": float(
            np.max(macro_errors) / cold_macro_path
        ),
        "event_state_jump_over_cold_path": event_state_jump / cold_path,
        "event_macro_discontinuity": event_macro_discontinuity,
        "transition_endpoint_error_over_path": transition_endpoint_error
        / transition_path,
        "maximum_macro_decoder_closure": macro_closure,
        "projected_100k_step_wall_seconds": projected_wall,
        "online_table_bytes": table_bytes,
    }
    gates = {
        "cold_rank": gate_values["cold_hidden_embedding_rank"]
        <= manifest.MAXIMUM_HIDDEN_EMBEDDING_RANK,
        "cold_capture": gate_values["cold_training_displacement_capture"]
        >= manifest.ENERGY_CAPTURE_TARGET,
        "cold_holdout_path": gate_values["maximum_cold_holdout_error_over_path"]
        <= manifest.MAXIMUM_COLD_HOLDOUT_ERROR_OVER_PATH,
        "cold_holdout_local": gate_values[
            "maximum_cold_holdout_error_over_local_chord"
        ]
        <= manifest.MAXIMUM_COLD_HOLDOUT_ERROR_OVER_LOCAL_CHORD,
        "cold_macro_ledger": gate_values[
            "maximum_cold_macro_ledger_error_over_path"
        ]
        <= manifest.MAXIMUM_COLD_MACRO_LEDGER_ERROR_OVER_PATH,
        "event_state": gate_values["event_state_jump_over_cold_path"]
        <= manifest.MAXIMUM_EVENT_STATE_JUMP_OVER_COLD_PATH,
        "event_macro": event_macro_discontinuity
        <= manifest.MAXIMUM_EVENT_MACRO_DISCONTINUITY,
        "transition_endpoint": gate_values[
            "transition_endpoint_error_over_path"
        ]
        <= manifest.MAXIMUM_TRANSITION_ENDPOINT_ERROR_OVER_PATH,
        "macro_closure": macro_closure <= manifest.MAXIMUM_MACRO_CLOSURE,
        "restart_bitwise": restart_bitwise,
        "single_equals_staged": single_equals_staged,
        "online_cost": projected_wall
        <= manifest.MAXIMUM_PROJECTED_100K_STEP_WALL_SECONDS,
        "truth_free": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gates": gates,
        "gate_values": gate_values,
        "cold_duration_seconds": cold.duration_seconds,
        "transition_duration_seconds": transition.duration_seconds,
        "continuous_online_dimension": 83,
        "mode_count": 2,
        "event_count": event.state.event_count,
        "online_truth_calls": 0,
        "online_470_roots": 0,
        "online_fixed_Q_microsteps": 0,
        "benchmark_update_iterations": update_iterations,
        "benchmark_decode_iterations": decode_iterations,
        "benchmark_update_wall_seconds": update_wall,
        "benchmark_decode_wall_seconds": decode_wall,
        "complete_cycle_calibration_available": False,
        "predictive_cycle_authorized": False,
    }
    arrays = {
        **model_arrays,
        "cold_predicted_macro82": predicted_macro,
        "cold_predicted_coordinates470": predicted_coordinates,
        "cold_true_coordinates470": coordinates,
        "cold_holdout_errors": holdout_errors,
        "cold_macro_holdout_errors": macro_errors,
        "event_macro82": event.state.macro_state,
        "event_decoded_coordinate470": event_decode,
        "transition_terminal_macro82": transition_result.state.macro_state,
        "transition_terminal_decoded_coordinate470": transition_decode,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = manifest.architecture.manifest.tube.manifest.geometry
    with manifest.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
                }
            )
    with manifest.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(manifest.CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": manifest.PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(manifest.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = manifest.architecture.manifest.tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("hybrid phase engine result already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "engine_metrics.json", metrics)
    manifest.architecture._write_npz(CANONICAL_DIRECTORY / "engine_model_and_replay.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "working_offline_online_architecture_on_observed_modes": metrics["passed"],
        "online_cost_feasible": metrics["gates"]["online_cost"],
        "complete_cycle_calibration_missing": True,
        "predictive_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "engine_source": manifest.ENGINE_SOURCE,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hybrid phase engine WP10c9d6c7c3b5c4f25e3",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The engine retains 82 conservative macro coordinates, one scalar phase, and two observed modes. Its projected 100,000-step update-plus-full-decode cost is {metrics['gate_values']['projected_100k_step_wall_seconds']:.3f} wall seconds on this machine. Every online truth/root/microstep counter is zero.",
                "",
                f"Cold held-out error/path is {metrics['gate_values']['maximum_cold_holdout_error_over_path']:.6e}; event macro discontinuity is {metrics['gate_values']['event_macro_discontinuity']:.6e}; maximum decoder macro-closure defect is {metrics['gate_values']['maximum_macro_decoder_closure']:.6e}.",
                "",
                "This is a working offline/online architecture on the observed cold and transition modes. It is not yet a predictive astrophysical cycle: hot exit, complete impulse, and remaining cycle modes are absent and cannot be inferred from this replay.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    payload = _run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
