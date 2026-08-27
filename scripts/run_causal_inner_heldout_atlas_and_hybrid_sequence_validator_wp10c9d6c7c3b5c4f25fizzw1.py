#!/usr/bin/env python3
"""Certify heldout-validation logic and a two-event reduced sequence fixture."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import run_causal_inner_heldout_atlas_and_hybrid_sequence_validation_manifest_wp10c9d6c7c3b5c4f25fizzw as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_reduced_hybrid_cycle import (  # noqa: E402
    ReducedEventReset,
    ReducedHybridCheckpoint,
    ReducedHybridTransition,
    integrate_fixed_dopri5,
    integrate_reduced_hybrid,
    load_reduced_hybrid_checkpoint,
    save_reduced_hybrid_checkpoint,
    validate_heldout_atlas_and_sequence,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "heldout_atlas_and_hybrid_sequence_validator_certified_synthetic_fixture_only"
FAIL_CLASSIFICATION = "heldout_atlas_or_hybrid_sequence_validator_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = "causal_inner_heldout_atlas_and_hybrid_sequence_validator_wp10c9d6c7c3b5c4f25fizzw1"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_HELDOUT_ATLAS_AND_HYBRID_SEQUENCE_VALIDATOR_WP10C9D6C7C3B5C4F25FIZZW1_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_heldout_atlas_and_hybrid_sequence_validator_wp10c9d6c7c3b5c4f25fizzw1.py"
THIS_TEST = "tests/test_causal_inner_heldout_atlas_and_hybrid_sequence_validator_wp10c9d6c7c3b5c4f25fizzw1.py"
PHYSICAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_reduced_hybrid_cycle.py"
PHYSICAL_TEST = "tests/test_causal_inner_reduced_hybrid_cycle.py"
PARENT_SHA256 = "69e9997dd5412df2a8cf8c0fa65965f106f1a8915cd91a48a8d8b4109b212411"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return manifest._u()


def _relative(left, right):
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny))


def _validate_parent(*, require_clean=False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256: raise RuntimeError("heldout sequence manifest changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY); summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json"); contract = utility._read_json(manifest.CANONICAL_DIRECTORY / "heldout_and_sequence_contract.json")
    if not summary["passed"] or not summary["definitions_only"] or not summary["finite_event_phase_advance_frozen"] or summary["heldout_validator_certified"] or summary["hybrid_sequence_validator_certified"] or summary["authorized_next"] != WORK_PACKAGE or summary["complete_cycle_execution_authorized"] or contract["structure_certificate"]["complete_cycle_steps"] != 0: raise RuntimeError("heldout sequence contract changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"): raise RuntimeError("heldout sequence validator needs a clean tracked tree")
    return hashes, contract


def _smooth_order():
    rates = np.asarray((0.7, 1.1, 1.6, 2.2)); forcing = np.asarray((0.2, -0.1, 0.05, 0.3)); initial = np.asarray((0.4, -0.3, 0.2, 0.1, 0.2)); horizon = 5.0
    rhs = lambda _time, state, _mode: np.concatenate((-rates * state[:4] + forcing, [1.0]))
    exact = np.concatenate((initial[:4] * np.exp(-rates * horizon) + forcing / rates * (1.0 - np.exp(-rates * horizon)), [initial[4] + horizon]))
    counts = (16, 32, 64); states = [integrate_fixed_dopri5(rhs, initial, start_time=0.0, end_time=horizon, step_count=count, mode_index=0) for count in counts]; errors = [float(np.linalg.norm(state - exact) / np.linalg.norm(exact)) for state in states]; orders = [float(np.log(errors[index] / errors[index + 1]) / np.log(2.0)) for index in range(2)]
    return counts, states, exact, errors, orders


def _hybrid_fixture():
    rates = {0: np.asarray((0.01, 0.0, 0.0, 0.0, 1.0)), 1: np.asarray((0.0, 0.02, 0.0, 0.0, 1.0))}; rhs = lambda _time, _state, mode: rates[int(mode)]
    first = ReducedHybridTransition("cold_to_hot", 0, 1, 1, lambda state: float(state[4] - 1.0), lambda _time, _state: ReducedEventReset(np.asarray((0.1, 0.0, 0.0, 0.0)), 0.1, 0.1, 0.2))
    second = ReducedHybridTransition("hot_to_recovery", 1, 0, 1, lambda state: float(state[4] - 2.0), lambda _time, _state: ReducedEventReset(np.asarray((0.0, 0.2, 0.0, 0.0)), 0.2, 0.2, 0.2))
    initial = ReducedHybridCheckpoint(np.zeros(5), 0.0, 0, 0.03, np.zeros(4), np.zeros(4), 0, 0, 0); tolerances = np.full(5, 1.0e-10)
    full = integrate_reduced_hybrid(rhs, initial, end_time_seconds=2.5, transitions=(first, second), absolute_tolerance=tolerances, relative_tolerance=1.0e-9)
    truth = np.concatenate((1.3 * rates[0][:4] + 0.9 * rates[1][:4] + np.asarray((0.1, 0.2, 0.0, 0.0)), [2.5]))
    checkpoint = next(value for value in full.accepted_checkpoints if value.completed_events == 1)
    return rhs, (first, second), initial, tolerances, full, checkpoint, truth


def _heldout_fixture(restart_bitwise):
    rng = np.random.default_rng(2026082706)
    branch_truth = rng.normal(size=(5, 20)); branch_pred = branch_truth * (1.0 + 2.0e-3)
    rate_truth = rng.normal(size=(5, 5)); rate_pred = rate_truth * (1.0 - 3.0e-3)
    port_truth = rng.normal(size=(5, 11)); port_pred = port_truth * (1.0 + 4.0e-3)
    event_times = np.asarray((1.0, 2.0, 3.2)); event_time_pred = event_times * (1.0 + 1.0e-3)
    event_state_truth = rng.normal(size=(3, 20)); event_state_pred = event_state_truth * (1.0 - 5.0e-3)
    event_ledger_truth = rng.normal(size=(3, 4)); event_ledger_pred = event_ledger_truth * (1.0 + 2.0e-3)
    endpoint_truth = rng.normal(size=20); endpoint_pred = endpoint_truth * (1.0 + 4.0e-3)
    ledger_truth = rng.normal(size=4); ledger_pred = ledger_truth * (1.0 - 3.0e-3)
    audit = validate_heldout_atlas_and_sequence(branch_predicted_states=branch_pred, branch_truth_states=branch_truth, branch_predicted_rates=rate_pred, branch_truth_rates=rate_truth, predicted_port_actions=port_pred, truth_port_actions=port_truth, predicted_event_times=event_time_pred, truth_event_times=event_times, predicted_event_post_states=event_state_pred, truth_event_post_states=event_state_truth, predicted_event_ledgers=event_ledger_pred, truth_event_ledgers=event_ledger_truth, predicted_sequence_endpoint=endpoint_pred, truth_sequence_endpoint=endpoint_truth, predicted_sequence_ledger=ledger_pred, truth_sequence_ledger=ledger_truth, predicted_mode_sequence=(0, 1, 0, 1), truth_mode_sequence=(0, 1, 0, 1), all_structure_gates_passed=True, restart_suffix_replay_bitwise=restart_bitwise)
    rejected = validate_heldout_atlas_and_sequence(branch_predicted_states=branch_pred, branch_truth_states=branch_truth, branch_predicted_rates=rate_pred, branch_truth_rates=rate_truth, predicted_port_actions=port_pred, truth_port_actions=port_truth, predicted_event_times=event_time_pred, truth_event_times=event_times, predicted_event_post_states=event_state_pred, truth_event_post_states=event_state_truth, predicted_event_ledgers=event_ledger_pred, truth_event_ledgers=event_ledger_truth, predicted_sequence_endpoint=endpoint_pred, truth_sequence_endpoint=endpoint_truth, predicted_sequence_ledger=ledger_pred, truth_sequence_ledger=ledger_truth, predicted_mode_sequence=(0, 1, 1, 0), truth_mode_sequence=(0, 1, 0, 1), all_structure_gates_passed=True, restart_suffix_replay_bitwise=restart_bitwise)
    return audit, rejected


def _checkpoint_equal(left, right):
    return bool(np.array_equal(left.state5, right.state5) and left.time_seconds == right.time_seconds and left.mode_index == right.mode_index and left.next_timestep_seconds == right.next_timestep_seconds and np.array_equal(left.cumulative_smooth_ledger4, right.cumulative_smooth_ledger4) and np.array_equal(left.cumulative_event_ledger4, right.cumulative_event_ledger4) and left.accepted_steps == right.accepted_steps and left.rejected_steps == right.rejected_steps and left.completed_events == right.completed_events)


def _certificate():
    began = time.perf_counter(); _, contract = _validate_parent(); counts, smooth_states, smooth_exact, smooth_errors, smooth_orders = _smooth_order(); rhs, transitions, initial, tolerances, full, restart_point, hybrid_truth = _hybrid_fixture()
    with tempfile.TemporaryDirectory(prefix="reduced_hybrid_checkpoint_") as directory:
        path = Path(directory) / "checkpoint.npz"; save_reduced_hybrid_checkpoint(restart_point, path); loaded = load_reduced_hybrid_checkpoint(path); checkpoint_bitwise = _checkpoint_equal(loaded, restart_point); replay = integrate_reduced_hybrid(rhs, loaded, end_time_seconds=2.5, transitions=transitions, absolute_tolerance=tolerances, relative_tolerance=1.0e-9)
    suffix_bitwise = _checkpoint_equal(replay.checkpoint, full.checkpoint)
    heldout, heldout_rejected = _heldout_fixture(suffix_bitwise)
    event_entry_times = np.asarray([event.entry_time_seconds for event in full.events]); event_truth = np.asarray((1.0, 2.0)); event_time_relative = float(np.max(np.abs(event_entry_times - event_truth) / event_truth))
    reduced_ledger_defect = _relative(full.checkpoint.state5[:4] - initial.state5[:4], full.checkpoint.cumulative_smooth_ledger4 + full.checkpoint.cumulative_event_ledger4)
    event_phase_time_defect = max(abs((event.exit_state5[4] - event.entry_state5[4]) - event.phase_advance) for event in full.events)
    structure = contract["structure_certificate"]
    passed = bool(min(smooth_orders) >= structure["minimum_smooth_observed_order"] and len(full.accepted_checkpoints) >= structure["minimum_smooth_accepted_steps"] and len(full.events) >= structure["minimum_events"] and len({event.source_mode_index for event in full.events} | {event.destination_mode_index for event in full.events}) >= structure["minimum_modes"] and _relative(full.checkpoint.state5, hybrid_truth) <= 2.0e-10 and event_time_relative <= 1.0e-8 and reduced_ledger_defect <= structure["maximum_reduced_ledger_defect"] and event_phase_time_defect <= 2.0e-14 and checkpoint_bitwise and suffix_bitwise and heldout.passed and not heldout_rejected.passed)
    metrics = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION, "passed": passed, "synthetic_fixture_only": True, "smooth_step_counts": counts, "smooth_relative_errors": smooth_errors, "smooth_observed_orders": smooth_orders, "minimum_smooth_observed_order": min(smooth_orders), "hybrid_accepted_steps": len(full.accepted_checkpoints), "hybrid_rejected_steps": full.checkpoint.rejected_steps, "hybrid_event_count": len(full.events), "hybrid_event_names": [event.name for event in full.events], "hybrid_event_entry_times_seconds": event_entry_times.tolist(), "maximum_event_entry_time_relative_defect": event_time_relative, "hybrid_endpoint_relative_defect": _relative(full.checkpoint.state5, hybrid_truth), "reduced_ledger_relative_defect": reduced_ledger_defect, "maximum_event_phase_advance_defect": event_phase_time_defect, "checkpoint_roundtrip_bitwise": checkpoint_bitwise, "restart_suffix_replay_bitwise": suffix_bitwise, "heldout_audit": asdict(heldout), "heldout_audit_passed": heldout.passed, "wrong_event_order_rejected": not heldout_rejected.passed, "physical_model_complete": False, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "certificate_wall_seconds": time.perf_counter() - began, "authorized_next": AUTHORIZED_NEXT if passed else None}
    arrays = {"smooth_states": np.asarray(smooth_states), "smooth_exact": smooth_exact, "hybrid_final_state": full.checkpoint.state5, "hybrid_truth_state": hybrid_truth, "hybrid_smooth_ledger": full.checkpoint.cumulative_smooth_ledger4, "hybrid_event_ledger": full.checkpoint.cumulative_event_ledger4, "replayed_final_state": replay.checkpoint.state5, "event_entry_states": np.asarray([event.entry_state5 for event in full.events]), "event_exit_states": np.asarray([event.exit_state5 for event in full.events])}
    return metrics, arrays


def _update(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]; status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("heldout sequence validator already exists")
    hashes, _ = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); utility._write_json(CANONICAL_DIRECTORY / "validator_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "validator_arrays.npz", **arrays)
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "heldout_validator_structure_certified": metrics["passed"], "hybrid_sequence_validator_structure_certified": metrics["passed"], "finite_event_phase_advance_certified": metrics["passed"], "synthetic_fixture_only": True, "physical_model_complete": False, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": metrics["authorized_next"]}; utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Heldout atlas and hybrid-sequence validator certificate\n\n" f"Classification: `{metrics['classification']}`.\n\n" f"The smooth reduced fixture has minimum fifth-order convergence `{metrics['minimum_smooth_observed_order']:.6f}`. The two-event sequence localizes entries with maximum relative timing defect `{metrics['maximum_event_entry_time_relative_defect']:.6e}`, closes the reduced ledger at `{metrics['reduced_ledger_relative_defect']:.6e}`, advances event phase exactly, and replays its restarted suffix bitwise. The thresholded heldout audit passes and an incorrect event order is rejected.\n\n" "This certifies validation and reduced hybrid numerical structure only. The fixture is synthetic; external physical payloads and heldout physical evidence remain absent. No complete-cycle step occurred.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
