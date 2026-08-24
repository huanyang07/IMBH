#!/usr/bin/env python3
"""Freeze a nonpropagating diagnosis of the terminal 470-chart boundary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_curvature_adaptive_arclength_continuation_execution_wp10c9d6c7c3b5c4f25fi as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fia"
CLASSIFICATION = "terminal_coordinate_conditioning_diagnosis_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fib_coordinate_chart_conditioning_diagnosis"
)
ARTIFACT = (
    "causal_inner_coordinate_chart_conditioning_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fia"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COORDINATE_CHART_CONDITIONING_"
    "DIAGNOSIS_MANIFEST_WP10C9D6C7C3B5C4F25FIA_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_coordinate_chart_conditioning_diagnosis_"
    "manifest_wp10c9d6c7c3b5c4f25fia.py"
)
THIS_TEST = (
    "tests/test_causal_inner_coordinate_chart_conditioning_diagnosis_"
    "manifest_wp10c9d6c7c3b5c4f25fia.py"
)

WITNESS_ATTEMPTS = (78, 79, 80, 81, 82, 83)
EXPECTED_ACCEPTED = (True, False, False, False, True, False)
PHYSICAL_ROWS = 162
MEMORY_ROWS = 280
DEPARTURE_ROWS = 28
COORDINATE_ROWS = PHYSICAL_ROWS + MEMORY_ROWS + DEPARTURE_ROWS
PHYSICAL_COLUMNS = 560
RAW_CONDITION_GATE = 2.5e3
RECOMPUTED_CONDITION_RELATIVE_TOLERANCE = 1.0e-10
MAXIMUM_ROW_EQUILIBRATED_CONDITION = 2.5e3
MAXIMUM_BLOCK_WHITENED_CONDITION = 2.5e3
MAXIMUM_WHITENING_CLOSURE_DEFECT = 1.0e-9
MINIMUM_TERMINAL_NORMALIZED_SINGULAR_RATIO = 0.5
MAXIMUM_EXECUTION_WALL_HOURS = 1.0


def _helper():
    return parent._helper()


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "continuation_execution_metrics.json"
    )
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PHYSICAL_CLASSIFICATION
        or summary["passed"]
        or summary["cycle_observed"]
        or summary["equilibrium_candidate_observed"]
        or summary["authorized_next"] is not None
        or values["accepted_segments"] != 71
        or values["attempted_segments"] != 84
        or values["exact_free_field_calls"] != 102
        or values["terminal_elapsed_seconds"] != 0.11125000000000008
        or not values["restart_roundtrip_bitwise"]
        or not values["suffix_replay_bitwise"]
        or metrics["fixed_Q_physical_rate_calls"] != 0
        or metrics["fixed_Q_reaction_calls"] != 0
        or metrics["nonlinear_roots"] != 0
        or metrics["BDF_microsteps"] != 0
    ):
        raise RuntimeError("terminal continuation certificate changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("conditioning manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _other_physical_gates(field: dict) -> bool:
    free = field["free_field"]
    retraction = field["retraction"]
    return bool(
        retraction["passed"]
        and retraction["coordinate_residual_infinity"]
        <= parent.source.COORDINATE_RETRACTION_TOLERANCE
        and retraction["gauge_residual_infinity"]
        <= parent.source.GAUGE_RETRACTION_TOLERANCE
        and free["coordinate_jacobian_rank"] == COORDINATE_ROWS
        and free["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12
        and free["maximum_height_ratio"] <= 0.5
        and free["minimum_scattering_optical_depth"] >= 1.0
        and free["reaction_free_ledger_passed"]
    )


def _lock_witnesses() -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    scratch = parent.SCRATCH_DIRECTORY
    identity = helper._read(scratch / "execution_identity.json")
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        identity["implementation_commit"] != provenance["implementation_commit"]
        or identity["implementation_tree"] != provenance["implementation_tree"]
        or identity["source_hashes"] != provenance["source_hashes"]
    ):
        raise RuntimeError("scratch identity does not match canonical execution")

    attempts = []
    accepted = []
    elapsed = []
    spans = []
    states = []
    coordinates = []
    scaled_rates = []
    coordinate_rates = []
    conditions = []
    ranks = []
    physical_passed = []
    other_physical_passed = []
    source_hashes = {}
    for attempt_index, expected_accepted in zip(
        WITNESS_ATTEMPTS, EXPECTED_ACCEPTED, strict=True
    ):
        directory = scratch / f"attempt_{attempt_index:04d}"
        attempt_json = directory / "attempt.json"
        attempt_npz = directory / "attempt.npz"
        field_json = directory / "endpoint_field.json"
        field_npz = directory / "endpoint_field.npz"
        for path in (attempt_json, attempt_npz, field_json, field_npz):
            if not path.exists():
                raise RuntimeError(f"missing conditioning witness {path}")
            source_hashes[str(path.relative_to(ROOT))] = helper._sha(path)
        attempt = helper._read(attempt_json)
        field = helper._read(field_json)
        arrays = helper._load_npz(field_npz)
        if (
            attempt["attempt_index"] != attempt_index
            or bool(attempt["accepted"]) != expected_accepted
            or arrays["primitive_state"].shape != (112, 5)
            or arrays["requested_coordinate470"].shape != (COORDINATE_ROWS,)
            or arrays["scaled_free_rate560_per_s"].shape != (PHYSICAL_COLUMNS,)
            or arrays["coordinate_free_rate470_per_s"].shape
            != (COORDINATE_ROWS,)
            or not _other_physical_gates(field)
        ):
            raise RuntimeError(f"conditioning witness {attempt_index} changed")
        attempts.append(attempt_index)
        accepted.append(expected_accepted)
        elapsed.append(attempt["elapsed_seconds_after"])
        spans.append(attempt["span_seconds"])
        states.append(np.asarray(arrays["primitive_state"]))
        coordinates.append(np.asarray(arrays["requested_coordinate470"]))
        scaled_rates.append(np.asarray(arrays["scaled_free_rate560_per_s"]))
        coordinate_rates.append(
            np.asarray(arrays["coordinate_free_rate470_per_s"])
        )
        conditions.append(
            field["free_field"]["coordinate_jacobian_condition_number"]
        )
        ranks.append(field["free_field"]["coordinate_jacobian_rank"])
        physical_passed.append(field["physical_passed"])
        other_physical_passed.append(_other_physical_gates(field))

    if not (
        conditions[-2] < RAW_CONDITION_GATE
        and conditions[-1] > RAW_CONDITION_GATE
        and ranks == [COORDINATE_ROWS] * len(ranks)
        and all(other_physical_passed)
    ):
        raise RuntimeError("terminal chart crossing no longer reproduced")
    metrics = {
        "attempt_indices": attempts,
        "accepted": accepted,
        "elapsed_seconds_after": elapsed,
        "span_seconds": spans,
        "saved_condition_numbers": conditions,
        "saved_ranks": ranks,
        "saved_physical_passed": physical_passed,
        "all_nonconditioning_physical_gates_passed": bool(
            all(other_physical_passed)
        ),
        "scratch_source_hashes": source_hashes,
        "scratch_identity": identity,
    }
    arrays = {
        "attempt_indices": np.asarray(attempts, dtype=int),
        "accepted": np.asarray(accepted, dtype=bool),
        "elapsed_seconds_after": np.asarray(elapsed, dtype=float),
        "span_seconds": np.asarray(spans, dtype=float),
        "primitive_states": np.stack(states),
        "requested_coordinates470": np.stack(coordinates),
        "scaled_free_rates560_per_s": np.stack(scaled_rates),
        "coordinate_free_rates470_per_s": np.stack(coordinate_rates),
        "saved_condition_numbers": np.asarray(conditions, dtype=float),
        "saved_ranks": np.asarray(ranks, dtype=int),
        "saved_physical_passed": np.asarray(physical_passed, dtype=bool),
    }
    return metrics, arrays


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "authorized_execution": AUTHORIZED_NEXT,
        "scope": {
            "saved_attempts": list(WITNESS_ATTEMPTS),
            "new_exact_coordinate_jacobians": len(WITNESS_ATTEMPTS),
            "new_exact_free_field_calls": 0,
            "new_retractions": 0,
            "new_trajectory_segments": 0,
            "new_physical_time_seconds": 0.0,
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        },
        "coordinate_blocks": {
            "physical_rows": PHYSICAL_ROWS,
            "memory_rows": MEMORY_ROWS,
            "departure_rows": DEPARTURE_ROWS,
            "total_rows": COORDINATE_ROWS,
            "scaled_primitive_columns": PHYSICAL_COLUMNS,
        },
        "required_diagnostics": {
            "raw": "complete singular spectrum and critical left/right vectors",
            "row_equilibrated": "divide every row by its Euclidean norm",
            "block_whitened": (
                "independently inverse-square-root whiten the physical, memory, "
                "and departure row Gramians before stacking"
            ),
            "block_interaction": (
                "principal-angle singular values between the three row spaces"
            ),
            "rate_action": (
                "compare recomputed J times the saved scaled free rate with the "
                "saved coordinate free rate"
            ),
        },
        "gates": {
            "raw_rank_equal": COORDINATE_ROWS,
            "raw_condition_reproduction_relative_tolerance": (
                RECOMPUTED_CONDITION_RELATIVE_TOLERANCE
            ),
            "maximum_row_equilibrated_condition": (
                MAXIMUM_ROW_EQUILIBRATED_CONDITION
            ),
            "maximum_block_whitened_condition": (
                MAXIMUM_BLOCK_WHITENED_CONDITION
            ),
            "maximum_whitening_closure_defect": (
                MAXIMUM_WHITENING_CLOSURE_DEFECT
            ),
            "minimum_terminal_normalized_singular_ratio": (
                MINIMUM_TERMINAL_NORMALIZED_SINGULAR_RATIO
            ),
            "all_nonconditioning_physical_gates_passed": True,
        },
        "decision": {
            "equilibrated_and_block_whitened_pass": (
                "coordinate_metric_artifact_supported_atlas_manifest_authorized"
            ),
            "row_equilibrated_fails_but_block_whitened_passes": (
                "interblock_chart_exhaustion_supported_tangent_atlas_manifest_authorized"
            ),
            "block_whitened_or_rank_fails": (
                "intrinsic_chart_degeneracy_detected_no_continuation_authorized"
            ),
            "reproduction_or_method_gate_fails": (
                "conditioning_diagnosis_failed_no_continuation_authorized"
            ),
        },
        "forbidden": [
            "relax the historical raw condition gate",
            "retroactively accept attempt 83",
            "advance or retract a new state",
            "evaluate a new original free field",
            "authorize a cycle or reduced slow evolution",
        ],
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "DEFINITIONS_ONLY",
            })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(summary_path, catalog)


def _canonicalize(parent_lock: dict, witness_metrics: dict, arrays: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("conditioning diagnosis manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "diagnosis_contract.json", _contract())
    helper._write_json(CANONICAL_DIRECTORY / "witness_lock.json", witness_metrics)
    _save_npz(CANONICAL_DIRECTORY / "conditioning_witnesses.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "parent_hashes": parent_lock["hashes"],
        "parent_classification": parent_lock["summary"]["classification"],
        "scratch_source_hashes": witness_metrics["scratch_source_hashes"],
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "conditioning_diagnosis_authorized": True,
        "conditioning_diagnosis_executed": False,
        "trajectory_authorized": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Coordinate-chart conditioning diagnosis manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The 111.25 ms continuation certificate is preserved as a binding rejection of the original raw 470-coordinate condition gate. Six saved endpoint states from attempts 78--83 are hash-locked for a nonpropagating singular-spectrum and metric-sensitivity diagnosis.",
            "",
            "The diagnosis may assemble only six exact coordinate Jacobians. It may not evaluate a new original free field, retract or propagate a state, relax the historical gate, authorize a cycle, or authorize reduced slow evolution.",
            "",
            f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("--freeze is required")
    parent_lock = _validate_parent(require_clean=True)
    witness_metrics, arrays = _lock_witnesses()
    summary = _canonicalize(parent_lock, witness_metrics, arrays)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
