#!/usr/bin/env python3
"""Fit and validate the conservative scalar transition tube."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_conservative_transition_tube_manifest_wp10c9d6c7c3b5c4f25dt as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25du"
PASS_CLASSIFICATION = (
    "train_only_rank_adaptive_conservative_scalar_transition_tube_"
    "validated_local_observed_segment"
)
FAIL_CLASSIFICATION = "conservative_scalar_transition_tube_surrogate_rejected"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dv"

ARTIFACT = "causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du.py"
THIS_TEST = "tests/test_causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSERVATIVE_TRANSITION_TUBE_"
    "WP10C9D6C7C3B5C4F25DU_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _validate_lock(*, require_clean: bool) -> dict:
    hashes = manifest.geometry._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = manifest.geometry._read(manifest.CANONICAL_DIRECTORY / "tube_contract.json")
    summary = manifest.geometry._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("conservative tube manifest classification changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if manifest.geometry._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen conservative tube source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and manifest.geometry._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("conservative tube execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _minimum_rank(matrix: np.ndarray, target: float) -> tuple[int, np.ndarray, np.ndarray]:
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    energy = singular_values**2
    if not np.any(energy):
        return 0, singular_values, right.T[:, :0]
    rank = int(np.searchsorted(np.cumsum(energy) / np.sum(energy), target) + 1)
    return rank, singular_values, right.T[:, :rank]


def _interpolate_training_table(
    times: np.ndarray,
    table: np.ndarray,
    train_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_indices, dtype=int)
    predicted = []
    brackets = []
    for index, time in enumerate(times):
        if index in train_indices:
            local = int(np.flatnonzero(train == index)[0])
            predicted.append(table[local])
            brackets.append((index, index))
            continue
        left = int(train[train < index][-1])
        right = int(train[train > index][0])
        left_local = int(np.flatnonzero(train == left)[0])
        right_local = int(np.flatnonzero(train == right)[0])
        weight = float((time - times[left]) / (times[right] - times[left]))
        predicted.append((1.0 - weight) * table[left_local] + weight * table[right_local])
        brackets.append((left, right))
    return np.asarray(predicted), np.asarray(brackets)


def _fit_tube(
    times: np.ndarray,
    coordinates: np.ndarray,
    macro: np.ndarray,
    hidden: np.ndarray,
    macro_restriction: np.ndarray,
    macro_lift: np.ndarray,
    hidden_basis: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    tiny = np.finfo(float).tiny
    train_indices = manifest.geometry_manifest.TRAIN_STATE_INDICES
    holdout_indices = manifest.geometry_manifest.HOLDOUT_STATE_INDICES
    train = np.asarray(train_indices, dtype=int)
    holdout = np.asarray(holdout_indices, dtype=int)
    hidden_origin = hidden[0]
    hidden_departures_train = hidden[train] - hidden_origin
    train_durations = np.diff(times[train])
    hidden_train_rates = np.diff(hidden[train], axis=0) / train_durations[:, None]
    scaled_displacements = hidden_departures_train / max(
        float(np.linalg.norm(hidden_departures_train)), tiny
    )
    scaled_rates = hidden_train_rates / max(float(np.linalg.norm(hidden_train_rates)), tiny)
    combined_training_matrix = np.vstack((scaled_displacements, scaled_rates))
    selected_rank, combined_singular_values, hidden_modes = _minimum_rank(
        combined_training_matrix, manifest.ENERGY_CAPTURE_TARGET
    )

    coefficient_table = hidden_departures_train @ hidden_modes
    coefficient_steps = np.linalg.norm(np.diff(coefficient_table, axis=0), axis=1)
    progress_knots = np.concatenate(([0.0], np.cumsum(coefficient_steps)))
    progress_knots /= max(float(progress_knots[-1]), tiny)
    progress_speeds = np.diff(progress_knots) / train_durations
    ledger_table = macro[train] - macro[[0]]

    coefficient_prediction, brackets = _interpolate_training_table(
        times, coefficient_table, train_indices
    )
    ledger_prediction, ledger_brackets = _interpolate_training_table(
        times, ledger_table, train_indices
    )
    if not np.array_equal(brackets, ledger_brackets):
        raise RuntimeError("coefficient and ledger interpolation brackets differ")
    hidden_prediction = hidden_origin + coefficient_prediction @ hidden_modes.T
    macro_prediction = macro[[0]] + ledger_prediction
    coordinate_prediction = (
        macro_lift @ macro_prediction.T + hidden_basis @ hidden_prediction.T
    ).T
    conditioned_prediction = (
        macro_lift @ macro.T + hidden_basis @ hidden_prediction.T
    ).T

    true_increments = np.diff(coordinates, axis=0)
    true_path_length = float(np.sum(np.linalg.norm(true_increments, axis=1)))
    full_errors = np.linalg.norm(coordinate_prediction[holdout] - coordinates[holdout], axis=1)
    conditioned_errors = np.linalg.norm(
        conditioned_prediction[holdout] - coordinates[holdout], axis=1
    )
    hidden_errors = np.linalg.norm(hidden_prediction[holdout] - hidden[holdout], axis=1)
    hidden_path_length = float(np.sum(np.linalg.norm(np.diff(hidden, axis=0), axis=1)))
    local_chords = np.asarray(
        [
            np.linalg.norm(coordinates[right] - coordinates[left])
            for left, right in brackets[holdout]
        ]
    )
    macro_ledger_errors = np.linalg.norm(
        ledger_prediction[holdout] - (macro[holdout] - macro[[0]]), axis=1
    )

    predicted_secants = np.diff(coordinate_prediction, axis=0) / np.diff(times)[:, None]
    true_secants = true_increments / np.diff(times)[:, None]
    secant_errors = np.linalg.norm(predicted_secants - true_secants, axis=1)
    true_secant_norms = np.linalg.norm(true_secants, axis=1)
    predicted_secant_norms = np.linalg.norm(predicted_secants, axis=1)
    relative_secant_errors = secant_errors / np.maximum(true_secant_norms, tiny)
    secant_cosines = np.sum(predicted_secants * true_secants, axis=1) / np.maximum(
        predicted_secant_norms * true_secant_norms, tiny
    )

    macro_decoder_defect = np.max(
        np.linalg.norm(
            (macro_restriction @ coordinate_prediction.T).T - macro_prediction,
            axis=1,
        )
    )
    ledger_telescoping_defect = float(
        np.linalg.norm(np.sum(np.diff(ledger_table, axis=0), axis=0) - ledger_table[-1])
    )
    displacement_capture = 1.0 - float(
        np.linalg.norm(hidden_departures_train - coefficient_table @ hidden_modes.T) ** 2
        / max(float(np.linalg.norm(hidden_departures_train) ** 2), tiny)
    )
    rate_coefficients = hidden_train_rates @ hidden_modes
    rate_capture = 1.0 - float(
        np.linalg.norm(hidden_train_rates - rate_coefficients @ hidden_modes.T) ** 2
        / max(float(np.linalg.norm(hidden_train_rates) ** 2), tiny)
    )
    lifted_modes = hidden_basis @ hidden_modes
    table_bytes = int(
        coefficient_table.nbytes
        + ledger_table.nbytes
        + progress_knots.nbytes
        + progress_speeds.nbytes
        + lifted_modes.nbytes
        + macro_lift.nbytes
    )
    lift_flops = int(
        2 * macro_lift.shape[0] * macro_lift.shape[1]
        + 2 * lifted_modes.shape[0] * max(lifted_modes.shape[1], 1)
    )
    gate_values = {
        "selected_hidden_embedding_rank": selected_rank,
        "training_displacement_energy_capture": displacement_capture,
        "training_rate_energy_capture": rate_capture,
        "maximum_holdout_hidden_error_over_path": float(
            np.max(hidden_errors) / max(hidden_path_length, tiny)
        ),
        "maximum_holdout_full_error_over_path": float(
            np.max(full_errors) / max(true_path_length, tiny)
        ),
        "maximum_conditioned_holdout_error_over_path": float(
            np.max(conditioned_errors) / max(true_path_length, tiny)
        ),
        "maximum_holdout_full_error_over_local_chord": float(
            np.max(full_errors / np.maximum(local_chords, tiny))
        ),
        "maximum_macro_ledger_holdout_error": float(np.max(macro_ledger_errors)),
        "maximum_fine_secant_relative_error": float(np.max(relative_secant_errors)),
        "minimum_fine_secant_direction_cosine": float(np.min(secant_cosines)),
        "macro_decoder_closure": float(macro_decoder_defect),
        "ledger_telescoping_closure": ledger_telescoping_defect,
        "online_lift_flops": lift_flops,
        "online_table_bytes": table_bytes,
    }
    gates = {
        "embedding_rank": selected_rank <= manifest.MAXIMUM_HIDDEN_EMBEDDING_RANK,
        "training_displacement_energy": displacement_capture
        >= manifest.ENERGY_CAPTURE_TARGET,
        "training_rate_energy": rate_capture >= manifest.ENERGY_CAPTURE_TARGET,
        "holdout_hidden": gate_values["maximum_holdout_hidden_error_over_path"]
        <= manifest.MAXIMUM_HOLDOUT_HIDDEN_ERROR_OVER_PATH,
        "holdout_full": gate_values["maximum_holdout_full_error_over_path"]
        <= manifest.MAXIMUM_HOLDOUT_FULL_ERROR_OVER_PATH,
        "holdout_local": gate_values["maximum_holdout_full_error_over_local_chord"]
        <= manifest.MAXIMUM_HOLDOUT_FULL_ERROR_OVER_LOCAL_CHORD,
        "macro_ledger_holdout": gate_values["maximum_macro_ledger_holdout_error"]
        <= manifest.MAXIMUM_MACRO_LEDGER_HOLDOUT_ERROR,
        "fine_secant_error": gate_values["maximum_fine_secant_relative_error"]
        <= manifest.MAXIMUM_FINE_SECANT_RELATIVE_ERROR,
        "fine_secant_direction": gate_values["minimum_fine_secant_direction_cosine"]
        >= manifest.MINIMUM_FINE_SECANT_DIRECTION_COSINE,
        "macro_decoder_closure": macro_decoder_defect
        <= manifest.CONSERVATIVE_CLOSURE_TOLERANCE,
        "ledger_telescoping_closure": ledger_telescoping_defect
        <= manifest.CONSERVATIVE_CLOSURE_TOLERANCE,
        "online_lift_cost": lift_flops <= manifest.MAXIMUM_ONLINE_LIFT_FLOPS,
        "online_table_size": table_bytes <= manifest.MAXIMUM_ONLINE_TABLE_BYTES,
        "positive_progress_speed": bool(np.all(progress_speeds > 0.0)),
    }
    passed = all(gates.values())
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "selected_hidden_embedding_rank": selected_rank,
        "training_state_count": len(train),
        "held_out_state_count": len(holdout),
        "observed_tube_duration_seconds": float(times[-1] - times[0]),
        "observed_partial_macro_ledger_norm": float(np.linalg.norm(ledger_table[-1])),
        "progress_speed_min_per_second": float(np.min(progress_speeds)),
        "progress_speed_max_per_second": float(np.max(progress_speeds)),
        "gate_values": gate_values,
        "gates": gates,
        "failed_gates": [name for name, passed_gate in gates.items() if not passed_gate],
        "new_truth_calls": 0,
        "hot_exit_observed": False,
        "partial_endpoint_is_complete_impulse": False,
        "complete_impulse_fit_authorized": False,
    }
    arrays = {
        "training_state_indices": train,
        "held_out_state_indices": holdout,
        "training_times_seconds": times[train],
        "progress_knots_s": progress_knots,
        "progress_speeds_per_second": progress_speeds,
        "hidden_origin388": hidden_origin,
        "hidden_embedding_basis388xr": hidden_modes,
        "lifted_hidden_basis470xr": lifted_modes,
        "hidden_coefficient_table": coefficient_table,
        "macro_ledger_table82": ledger_table,
        "partial_terminal_macro_reset82": ledger_table[-1],
        "predicted_coordinates470": coordinate_prediction,
        "conditioned_predicted_coordinates470": conditioned_prediction,
        "true_coordinates470": coordinates,
        "predicted_hidden_coordinates388": hidden_prediction,
        "predicted_macro_ledger82": ledger_prediction,
        "holdout_full_errors": full_errors,
        "holdout_conditioned_errors": conditioned_errors,
        "holdout_hidden_errors": hidden_errors,
        "holdout_macro_ledger_errors": macro_ledger_errors,
        "predicted_secants470_per_s": predicted_secants,
        "true_secants470_per_s": true_secants,
        "fine_secant_relative_errors": relative_secant_errors,
        "fine_secant_direction_cosines": secant_cosines,
        "combined_training_singular_values": combined_singular_values,
        "interpolation_brackets": brackets,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": manifest.geometry._sha(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = manifest.geometry._read(CANONICAL_SUMMARY)
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
    manifest.geometry._write_json(CANONICAL_SUMMARY, catalog)


def _execute() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("conservative transition-tube result already exists")
    lock = _validate_lock(require_clean=True)
    geometry_arrays = manifest.geometry._load_npz(
        manifest.geometry.CANONICAL_DIRECTORY / "geometry_arrays.npz"
    )
    tangent = manifest.geometry._load_npz(manifest.geometry_manifest.TANGENT_ARRAYS)
    metrics, arrays = _fit_tube(
        np.asarray(geometry_arrays["trajectory_times_seconds"], dtype=float),
        np.asarray(geometry_arrays["trajectory_coordinates470"], dtype=float),
        np.asarray(geometry_arrays["trajectory_macro_coordinates82"], dtype=float),
        np.asarray(geometry_arrays["trajectory_hidden_coordinates388"], dtype=float),
        np.asarray(tangent["macro_restriction_R82"], dtype=float),
        np.asarray(geometry_arrays["macro_lift_L470x82"], dtype=float),
        np.asarray(tangent["hidden_basis_Z388"], dtype=float),
    )
    CANONICAL_DIRECTORY.mkdir(parents=True)
    manifest.geometry._write_json(CANONICAL_DIRECTORY / "tube_metrics.json", metrics)
    np.savez(CANONICAL_DIRECTORY / "tube_model_and_validation.npz", **arrays)
    manifest.geometry._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_directory": str(manifest.CANONICAL_DIRECTORY.relative_to(ROOT)),
            "manifest_hashes": lock["manifest_hashes"],
            "geometry_arrays_sha256": manifest.geometry._sha(
                manifest.geometry.CANONICAL_DIRECTORY / "geometry_arrays.npz"
            ),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "transition_dynamic_dimension": 1 if metrics["passed"] else None,
        "selected_hidden_embedding_rank": metrics["selected_hidden_embedding_rank"],
        "failed_gates": metrics["failed_gates"],
        "observed_segment_validated": metrics["passed"],
        "hot_exit_observed": False,
        "complete_impulse_fit_authorized": False,
        "reduced_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    manifest.geometry._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    manifest.geometry._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": manifest.geometry._git("rev-parse", "HEAD"),
            "implementation_tree": manifest.geometry._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": lock["contract"]["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{manifest.geometry._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Conservative transition tube WP10c9d6c7c3b5c4f25du",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"The train-only model selected hidden rank `{metrics['selected_hidden_embedding_rank']}` with one dynamic progress coordinate. Maximum held-out full error/path is `{metrics['gate_values']['maximum_holdout_full_error_over_path']:.6e}`; maximum fine-secant relative error is `{metrics['gate_values']['maximum_fine_secant_relative_error']:.6e}`.",
                "",
                f"The online lift requires approximately `{metrics['gate_values']['online_lift_flops']}` flops and `{metrics['gate_values']['online_table_bytes']}` stored bytes. The 82-coordinate conservative decoder and partial ledger telescope close within `{max(metrics['gate_values']['macro_decoder_closure'], metrics['gate_values']['ledger_telescoping_closure']):.6e}`.",
                "",
                f"Failed gates: `{metrics['failed_gates']}`.",
                "",
                "The observed terminal state is not a hot exit. The partial macro reset must not be used as a complete transition impulse.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    if not args.fit:
        parser.error("use --fit")
    payload = _execute()
    print(json.dumps(manifest.geometry._plain(payload), indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
