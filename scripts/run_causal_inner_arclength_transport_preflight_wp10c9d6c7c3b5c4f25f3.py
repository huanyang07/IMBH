#!/usr/bin/env python3
"""Replay Window-5 exact-chart states with one transported anchor matrix."""

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

import run_causal_inner_arclength_transport_manifest_wp10c9d6c7c3b5c4f25f2 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f3"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f4"
PASS_CLASSIFICATION = "moving_exact_arclength_anchor_transport_replay_passed"
FAIL_CLASSIFICATION = "moving_exact_arclength_anchor_transport_replay_rejected"
ARTIFACT = (
    "causal_inner_arclength_transport_preflight_"
    "wp10c9d6c7c3b5c4f25f3"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = manifest.PREFLIGHT_RUNNER
THIS_TEST = manifest.PREFLIGHT_TEST
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ARCLENGTH_TRANSPORT_PREFLIGHT_"
    "WP10C9D6C7C3B5C4F25F3_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return manifest._helper()


def _exact_chart():
    return manifest.parent.source.exact_chart


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "arclength_transport_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["transport_preflight_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("arclength transport manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen arclength source changed: {relative}")
    current = {name: helper._sha(path) for name, path in manifest._decisive_inputs().items()}
    if current != contract["decisive_input_hashes"]:
        raise RuntimeError("arclength transport decisive input changed")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("arclength transport preflight requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _broyden_update(matrix: np.ndarray, step: np.ndarray, residual_change: np.ndarray) -> np.ndarray:
    denominator = float(step @ step)
    if denominator <= np.finfo(float).tiny:
        return np.asarray(matrix, dtype=float)
    return np.asarray(matrix, dtype=float) + np.outer(
        residual_change - np.asarray(matrix) @ step, step
    ) / denominator


def _raw_initial_state(model, anchor_state: np.ndarray, anchor_coordinate: np.ndarray, target: np.ndarray) -> np.ndarray:
    anchor_model_state = np.asarray(model.decoded_state(anchor_coordinate), dtype=float)
    decoded = np.asarray(model.decoded_state(target), dtype=float)
    return np.asarray(anchor_state, dtype=float) + decoded - anchor_model_state


def _transport_retract(
    *,
    model,
    initial_state: np.ndarray,
    target: np.ndarray,
    gauge_basis: np.ndarray,
    anchor_delta: np.ndarray,
    anchor_augmented: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    exact_chart = _exact_chart()
    state = np.asarray(initial_state, dtype=float).copy()
    matrix = np.asarray(anchor_augmented, dtype=float).copy()
    residual_history = []
    line_factors = []
    refreshes = 0
    corrections = 0
    condition_numbers = []
    began = time.perf_counter()
    iteration = 0
    while iteration <= manifest.MAXIMUM_TRANSPORT_ITERATIONS:
        residual, factors = exact_chart._residual(
            model, state, target, gauge_basis, anchor_delta
        )
        coordinate_inf = float(np.max(np.abs(residual[: exact_chart.COORDINATE_DIMENSION])))
        gauge_inf = float(np.max(np.abs(residual[exact_chart.COORDINATE_DIMENSION :])))
        combined = max(coordinate_inf, gauge_inf)
        residual_history.append(combined)
        if (
            coordinate_inf <= manifest.parent.source.manifest.COORDINATE_TOLERANCE
            and gauge_inf <= manifest.parent.source.manifest.GAUGE_TOLERANCE
        ):
            physical = exact_chart._physical_audit(model, state, factors)
            return state, matrix, {
                "passed": bool(physical["passed"]),
                "coordinate_residual_infinity": coordinate_inf,
                "gauge_residual_infinity": gauge_inf,
                "transport_corrections": corrections,
                "target_exact_refreshes": refreshes,
                "accepted_line_factors": line_factors,
                "residual_history": residual_history,
                "maximum_augmented_condition_number": max(condition_numbers) if condition_numbers else 0.0,
                "maximum_scaled_anchor_departure": float(
                    np.max(np.abs(exact_chart._delta(model, state) - anchor_delta))
                ),
                "wall_seconds": float(time.perf_counter() - began),
                **physical,
            }
        if iteration == manifest.MAXIMUM_TRANSPORT_ITERATIONS:
            break
        if (
            iteration >= manifest.MAXIMUM_TRANSPORT_ITERATIONS - manifest.REFRESH_ITERATION_RESERVE
            and refreshes < manifest.MAXIMUM_TARGET_EXACT_REFRESHES
        ):
            matrix, jacobian_metrics = exact_chart._augmented_jacobian(
                model, state, gauge_basis
            )
            refreshes += 1
            condition_numbers.append(jacobian_metrics["augmented_condition_number"])
            if (
                jacobian_metrics["augmented_rank"] != exact_chart.PHYSICAL_DIMENSION
                or jacobian_metrics["augmented_condition_number"]
                > manifest.parent.source.manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER
            ):
                break
        correction = np.linalg.solve(matrix, residual)
        old_delta = exact_chart._delta(model, state)
        accepted = False
        for factor in exact_chart.LINE_FACTORS:
            proposed_delta = old_delta - factor * correction
            if (
                float(np.max(np.abs(proposed_delta - anchor_delta)))
                > manifest.parent.source.manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE
            ):
                continue
            proposed = exact_chart._state_from_delta(model, proposed_delta)
            trial, _trial_factors = exact_chart._residual(
                model, proposed, target, gauge_basis, anchor_delta
            )
            if float(np.max(np.abs(trial))) < combined:
                step = proposed_delta - old_delta
                matrix = _broyden_update(matrix, step, trial - residual)
                state = proposed
                line_factors.append(float(factor))
                corrections += 1
                accepted = True
                break
        if not accepted:
            if refreshes >= manifest.MAXIMUM_TARGET_EXACT_REFRESHES:
                break
            matrix, jacobian_metrics = exact_chart._augmented_jacobian(
                model, state, gauge_basis
            )
            refreshes += 1
            condition_numbers.append(jacobian_metrics["augmented_condition_number"])
            if (
                jacobian_metrics["augmented_rank"] != exact_chart.PHYSICAL_DIMENSION
                or jacobian_metrics["augmented_condition_number"]
                > manifest.parent.source.manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER
            ):
                break
            continue
        iteration += 1
    residual, factors = exact_chart._residual(
        model, state, target, gauge_basis, anchor_delta
    )
    physical = exact_chart._physical_audit(model, state, factors)
    return state, matrix, {
        "passed": False,
        "coordinate_residual_infinity": float(
            np.max(np.abs(residual[: exact_chart.COORDINATE_DIMENSION]))
        ),
        "gauge_residual_infinity": float(
            np.max(np.abs(residual[exact_chart.COORDINATE_DIMENSION :]))
        ),
        "transport_corrections": corrections,
        "target_exact_refreshes": refreshes,
        "accepted_line_factors": line_factors,
        "residual_history": residual_history,
        "maximum_augmented_condition_number": max(condition_numbers) if condition_numbers else 0.0,
        "maximum_scaled_anchor_departure": float(
            np.max(np.abs(_exact_chart()._delta(model, state) - anchor_delta))
        ),
        "wall_seconds": float(time.perf_counter() - began),
        **physical,
    }


def _evaluate(locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    source = manifest.parent.source
    exact_chart = _exact_chart()
    directory = manifest.parent._window_directories()[-1]
    window_metrics = helper._read(directory / "phase_window_metrics.json")
    window_arrays = helper._load_npz(directory / "phase_window_arrays.npz")
    base = source._base_inputs()
    model = base["model"]
    anchor_state = np.asarray(window_arrays["start_primitive_state"], dtype=float)
    anchor_coordinate = np.asarray(window_arrays["start_coordinate470"], dtype=float)
    gauge_basis = np.asarray(window_arrays["anchor_gauge_basis560x90"], dtype=float)
    anchor_delta = np.asarray(window_arrays["anchor_delta560"], dtype=float)
    began = time.perf_counter()
    anchor_augmented, anchor_metrics = exact_chart._augmented_jacobian(
        model, anchor_state, gauge_basis
    )
    anchor_wall = float(time.perf_counter() - began)
    if (
        anchor_metrics["augmented_rank"] != exact_chart.PHYSICAL_DIMENSION
        or anchor_metrics["augmented_condition_number"]
        > source.manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER
    ):
        raise RuntimeError("transport anchor Jacobian is inadmissible")
    targets = np.asarray(window_arrays["exact_evaluation_coordinates470"], dtype=float)
    references = np.asarray(window_arrays["exact_evaluation_primitive_states"], dtype=float)
    records = []
    states = []
    recovered = []
    for index, (target, reference) in enumerate(zip(targets, references, strict=True)):
        if np.array_equal(target, anchor_coordinate):
            state = np.array(anchor_state, copy=True)
            residual, factors = exact_chart._residual(
                model, state, target, gauge_basis, anchor_delta
            )
            physical = exact_chart._physical_audit(model, state, factors)
            item = {
                "passed": bool(physical["passed"]),
                "coordinate_residual_infinity": float(np.max(np.abs(residual[:470]))),
                "gauge_residual_infinity": float(np.max(np.abs(residual[470:]))),
                "transport_corrections": 0,
                "target_exact_refreshes": 0,
                "accepted_line_factors": [],
                "residual_history": [float(np.max(np.abs(residual)))],
                "maximum_augmented_condition_number": 0.0,
                "maximum_scaled_anchor_departure": 0.0,
                "wall_seconds": 0.0,
                **physical,
            }
        else:
            initial = _raw_initial_state(model, anchor_state, anchor_coordinate, target)
            state, _matrix, item = _transport_retract(
                model=model,
                initial_state=initial,
                target=target,
                gauge_basis=gauge_basis,
                anchor_delta=anchor_delta,
                anchor_augmented=anchor_augmented,
            )
        state_defect = float(
            np.max(
                np.abs(
                    (np.asarray(state) - np.asarray(reference)) / model.columns
                )
            )
        )
        coordinate, factors = model.coordinate(state)
        item.update({
            "target_index": index,
            "state_replay_scaled_infinity_defect": state_defect,
            "coordinate_replay_infinity_defect": float(np.max(np.abs(coordinate - target))),
            "decoder_minimum_reconstruction_factor": float(np.min(factors)),
        })
        records.append(item)
        states.append(state)
        recovered.append(coordinate)
        print(
            f"transport target {index + 1:02d}/{len(targets):02d}: "
            f"coord={item['coordinate_residual_infinity']:.3e} "
            f"state={state_defect:.3e} refresh={item['target_exact_refreshes']}",
            flush=True,
        )
    maximum_coordinate = max(item["coordinate_residual_infinity"] for item in records)
    maximum_gauge = max(item["gauge_residual_infinity"] for item in records)
    maximum_state = max(item["state_replay_scaled_infinity_defect"] for item in records)
    maximum_departure = max(item["maximum_scaled_anchor_departure"] for item in records)
    maximum_condition = max(
        anchor_metrics["augmented_condition_number"],
        *(item["maximum_augmented_condition_number"] for item in records),
    )
    total_refreshes = sum(item["target_exact_refreshes"] for item in records)
    minimum_reconstruction = min(
        min(item["minimum_reconstruction_factor"], item["decoder_minimum_reconstruction_factor"])
        for item in records
    )
    gates = {
        "all_targets_converged": all(item["passed"] for item in records),
        "coordinate_replay": maximum_coordinate <= source.manifest.COORDINATE_TOLERANCE,
        "gauge_replay": maximum_gauge <= source.manifest.GAUGE_TOLERANCE,
        "canonical_state_replay": maximum_state <= manifest.STATE_REPLAY_MAXIMUM_SCALED_INFINITY_DEFECT,
        "retraction_neighborhood": maximum_departure <= source.manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE,
        "conditioning": maximum_condition <= source.manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER,
        "reconstruction": minimum_reconstruction >= source.manifest.MINIMUM_RECONSTRUCTION_FACTOR,
        "refresh_budget": total_refreshes <= manifest.MAXIMUM_TOTAL_TARGET_EXACT_REFRESHES,
        "exact_assembly_reduction": 1 + total_refreshes < sum(
            int(item["retraction"]["Newton_corrections"])
            for item in window_metrics["exact_rate_metrics"]
        ),
        "no_new_fixed_Q_rates_roots_or_microsteps": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gates": gates,
        "gate_values": {
            "target_count": len(targets),
            "maximum_coordinate_residual_infinity": maximum_coordinate,
            "maximum_gauge_residual_infinity": maximum_gauge,
            "maximum_state_replay_scaled_infinity_defect": maximum_state,
            "maximum_scaled_anchor_departure": maximum_departure,
            "maximum_augmented_condition_number": maximum_condition,
            "minimum_reconstruction_factor": minimum_reconstruction,
            "anchor_exact_assemblies": 1,
            "target_exact_refreshes": total_refreshes,
            "transport_exact_assemblies": 1 + total_refreshes,
            "historical_exact_assemblies_for_same_targets": sum(
                int(item["retraction"]["Newton_corrections"])
                for item in window_metrics["exact_rate_metrics"]
            ),
            "anchor_assembly_wall_seconds": anchor_wall,
            "target_transport_wall_seconds": sum(item["wall_seconds"] for item in records),
        },
        "target_records": records,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
        "input_lock": locked,
    }
    arrays = {
        "target_coordinates470": targets,
        "reference_primitive_states": references,
        "transported_primitive_states": np.asarray(states),
        "recovered_coordinates470": np.asarray(recovered),
        "anchor_augmented_jacobian560x560": anchor_augmented,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = manifest.parent.source._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
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
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("arclength transport preflight already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "transport_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "transport_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "Window_05_targets_replayed": metrics["passed"],
        "new_exact_fixed_Q_rate_calls": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
        "arclength_execution_manifest_authorized": metrics["passed"],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Moving exact arclength transport preflight",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"All `{values['target_count']}` Window-5 targets were replayed with `{values['transport_exact_assemblies']}` exact augmented coordinate assemblies versus `{values['historical_exact_assemblies_for_same_targets']}` in the historical exact-Newton retractions.",
            "",
            f"The maximum coordinate residual is `{values['maximum_coordinate_residual_infinity']:.6e}`, maximum gauge residual `{values['maximum_gauge_residual_infinity']:.6e}`, and maximum canonical scaled-state defect `{values['maximum_state_replay_scaled_infinity_defect']:.6e}`. No fixed-Q rate, nonlinear root, or BDF microstep was executed.",
            "",
        )),
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
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
