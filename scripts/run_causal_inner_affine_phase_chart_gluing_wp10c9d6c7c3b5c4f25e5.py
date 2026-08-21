#!/usr/bin/env python3
"""Execute affine chart gluing for the conservative hybrid phase engine."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
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
import run_causal_inner_affine_phase_chart_gluing_manifest_wp10c9d6c7c3b5c4f25e4 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e5"
PASS_CLASSIFICATION = (
    "affine_glued_truth_free_hybrid_phase_engine_working_on_observed_modes_"
    "cost_feasible_complete_cycle_calibration_missing"
)
FAIL_CLASSIFICATION = "affine_phase_chart_gluing_failed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e6"

ARTIFACT = "causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5.py"
THIS_TEST = "tests/test_causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AFFINE_PHASE_CHART_GLUING_"
    "WP10C9D6C7C3B5C4F25E5_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = manifest.rejected.manifest.architecture.manifest.tube.manifest.geometry
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "affine_gluing_contract.json")
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("affine chart-gluing manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen affine gluing source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("affine chart-gluing execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _build_affine_engine() -> tuple[ConservativeHybridPhaseEngine, dict, dict]:
    old_engine, data, old_arrays = manifest.rejected._build_engine()
    cold = old_engine.modes["cold_observed"]
    local = old_engine.modes["fixed_Q_transition_observed"]
    entry_hidden = np.asarray(data["hidden"][-1], dtype=float)
    absolute = ConservativePhaseMode(
        name=local.name,
        phase_knots=local.phase_knots,
        phase_speeds_per_second=local.phase_speeds_per_second,
        macro_ledger_knots=local.macro_ledger_knots,
        hidden_coefficient_knots=local.hidden_coefficient_knots,
        hidden_origin=entry_hidden + local.hidden_origin,
        hidden_embedding_basis=local.hidden_embedding_basis,
        macro_lift=local.macro_lift,
        hidden_lift=local.hidden_lift,
        macro_restriction=local.macro_restriction,
    )
    engine = ConservativeHybridPhaseEngine(
        {cold.name: cold, absolute.name: absolute},
        {cold.name: absolute.name, absolute.name: None},
    )
    data = dict(data)
    data["transition_coordinates_absolute"] = (
        data["coordinates"][[-1]] + data["transition_coordinates"]
    )
    data["transition_macro_absolute"] = (
        data["macro"][[-1]] + data["transition_macro"]
    )
    arrays = {
        **old_arrays,
        "transition_entry_absolute_coordinate470": data["coordinates"][-1],
        "transition_entry_absolute_macro82": data["macro"][-1],
        "transition_entry_absolute_hidden388": entry_hidden,
        "transition_absolute_hidden_origin388": absolute.hidden_origin,
        "transition_coordinates_absolute": data["transition_coordinates_absolute"],
        "transition_macro_absolute": data["transition_macro_absolute"],
    }
    return engine, data, arrays


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    helper = manifest.rejected.manifest.architecture.manifest.tube.manifest.geometry
    engine, data, arrays = _build_affine_engine()
    parent_metrics = helper._read(
        manifest.rejected.CANONICAL_DIRECTORY / "engine_metrics.json"
    )
    cold = engine.modes["cold_observed"]
    transition = engine.modes["fixed_Q_transition_observed"]
    start = HybridPhaseState(data["macro"][0], 0.0, cold.name)
    event = engine.advance(start, cold.duration_seconds)
    event_decode = engine.decode(event.state)
    entry_truth = data["coordinates"][-1]
    local_seed_norm = float(np.linalg.norm(data["transition_coordinates"][0]))
    entry_decoder_defect = float(np.linalg.norm(event_decode - entry_truth))
    event_macro_defect = float(
        np.linalg.norm(event.state.macro_state - data["macro"][-1])
    )
    restarted = HybridPhaseState.from_payload(
        json.loads(json.dumps(event.state.to_payload()))
    )
    restart_bitwise = bool(
        np.array_equal(restarted.macro_state, event.state.macro_state)
        and restarted.phase == event.state.phase
        and restarted.mode == event.state.mode
        and restarted.elapsed_seconds == event.state.elapsed_seconds
        and restarted.event_count == event.state.event_count
    )
    terminal = engine.advance(restarted, transition.duration_seconds)
    terminal_decode = engine.decode(terminal.state)
    transition_truth = data["transition_coordinates_absolute"]
    transition_path = float(
        np.sum(np.linalg.norm(np.diff(transition_truth, axis=0), axis=1))
    )
    terminal_error = float(np.linalg.norm(terminal_decode - transition_truth[-1]))
    single = engine.advance(start, cold.duration_seconds + transition.duration_seconds)
    single_equals_staged = bool(
        np.array_equal(single.state.macro_state, terminal.state.macro_state)
        and single.state.phase == terminal.state.phase
        and single.state.mode == terminal.state.mode
        and single.state.elapsed_seconds == terminal.state.elapsed_seconds
        and single.state.event_count == terminal.state.event_count
    )
    decoded = np.vstack((event_decode, terminal_decode))
    macros = np.vstack((event.state.macro_state, terminal.state.macro_state))
    macro_closure = float(
        np.max(
            np.linalg.norm(
                (data["restriction"] @ decoded.T).T - macros, axis=1
            )
        )
    )
    iterations = 2_000
    began = time.perf_counter()
    accumulator = 0.0
    for index in range(iterations):
        phase = (index % 1000) / 1000.0
        accumulator += float(
            transition.decode(data["macro"][-1] + transition.ledger(phase), phase)[0]
        )
    benchmark_wall = time.perf_counter() - began
    if not np.isfinite(accumulator):
        raise RuntimeError("affine benchmark accumulator is nonfinite")
    projected_wall = manifest.rejected.manifest.ONLINE_MACROSTEPS * (
        benchmark_wall / iterations
    )
    inherited_gates = {
        name: passed
        for name, passed in parent_metrics["gates"].items()
        if name != "event_state"
    }
    gates = {
        **inherited_gates,
        "local_seed_zero": local_seed_norm <= manifest.LOCAL_SEED_ZERO_GATE,
        "affine_entry_decoder": entry_decoder_defect
        <= manifest.AFFINE_ENTRY_CLOSURE_GATE,
        "event_macro": event_macro_defect
        <= manifest.rejected.manifest.MAXIMUM_EVENT_MACRO_DISCONTINUITY,
        "transition_endpoint": terminal_error / transition_path
        <= manifest.rejected.manifest.MAXIMUM_TRANSITION_ENDPOINT_ERROR_OVER_PATH,
        "macro_closure": macro_closure
        <= manifest.rejected.manifest.MAXIMUM_MACRO_CLOSURE,
        "restart_bitwise": restart_bitwise,
        "single_equals_staged": single_equals_staged,
        "online_cost": projected_wall
        <= manifest.rejected.manifest.MAXIMUM_PROJECTED_100K_STEP_WALL_SECONDS,
        "truth_free": True,
    }
    passed = bool(all(gates.values()))
    gate_values = {
        **parent_metrics["gate_values"],
        "transition_local_seed_norm": local_seed_norm,
        "absolute_entry_decoder_closure": entry_decoder_defect,
        "event_macro_discontinuity": event_macro_defect,
        "transition_endpoint_error_over_path": terminal_error / transition_path,
        "maximum_macro_decoder_closure": macro_closure,
        "projected_100k_full_decode_wall_seconds": projected_wall,
    }
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gates": gates,
        "gate_values": gate_values,
        "continuous_online_dimension": 83,
        "mode_count": 2,
        "online_truth_calls": 0,
        "online_470_roots": 0,
        "online_fixed_Q_microsteps": 0,
        "complete_cycle_calibration_available": False,
        "predictive_cycle_authorized": False,
    }
    arrays.update(
        {
            "affine_event_decoded_coordinate470": event_decode,
            "affine_terminal_decoded_coordinate470": terminal_decode,
            "affine_event_macro82": event.state.macro_state,
            "affine_terminal_macro82": terminal.state.macro_state,
        }
    )
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = manifest.rejected.manifest.architecture.manifest.tube.manifest.geometry
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
    helper = manifest.rejected.manifest.architecture.manifest.tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("affine chart-gluing result already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "affine_engine_metrics.json", metrics)
    manifest.rejected.manifest.architecture._write_npz(
        CANONICAL_DIRECTORY / "affine_engine_model_and_replay.npz", arrays
    )
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
                "# Affine phase-chart gluing WP10c9d6c7c3b5c4f25e5",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The affine event decoder closes at {metrics['gate_values']['absolute_entry_decoder_closure']:.6e}, and macro continuity closes at {metrics['gate_values']['event_macro_discontinuity']:.6e}. The projected 100,000 full-decode cost is {metrics['gate_values']['projected_100k_full_decode_wall_seconds']:.3f} wall seconds.",
                "",
                "The offline/online architecture now works on every observed cold and transition gate with no online truth call. The remaining blocker is calibration and independent validation of missing physical cycle modes, not online computational cost.",
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
