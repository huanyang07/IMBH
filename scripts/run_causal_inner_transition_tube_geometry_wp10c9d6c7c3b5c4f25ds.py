#!/usr/bin/env python3
"""Audit the accepted transition trajectory as a conservative scalar tube."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_tube_geometry_manifest_wp10c9d6c7c3b5c4f25dr as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ds"
PASS_CLASSIFICATION = (
    "one_scalar_conservative_transition_tube_geometry_supported_"
    "hot_exit_unobserved"
)
FAIL_CLASSIFICATION = "scalar_transition_tube_geometry_rejected"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dt"

ARTIFACT = "causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds.py"
THIS_TEST = "tests/test_causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_TUBE_GEOMETRY_"
    "WP10C9D6C7C3B5C4F25DS_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_checksums(directory: Path) -> dict[str, str]:
    hashes = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        hashes[name] = expected
    return hashes


def _validate_lock(*, require_clean: bool) -> dict:
    hashes = _validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = _read(manifest.CANONICAL_DIRECTORY / "geometry_contract.json")
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
        or contract["trajectory"]["state_count_including_seed"]
        != manifest.STATE_COUNT
    ):
        raise RuntimeError("transition-tube manifest classification changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen transition-tube source changed: {relative}")
    manifest._validate_parents(require_clean=False)
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("transition-tube analysis requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _ordered_trajectory() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    directories = manifest._accepted_stage_directories()
    first = _load_npz(directories[0] / "hot_exit_feature_arrays.npz")
    coordinates = [np.asarray(first["previous_coordinate470"], dtype=float)]
    times = [0.0]
    labels = ["warm_3_seed"]
    hidden_fractions = []
    elapsed = 0.0
    for position, directory in enumerate(directories, 1):
        arrays = _load_npz(directory / "hot_exit_feature_arrays.npz")
        metrics = _read(directory / "hot_exit_feature_metrics.json")
        previous = np.asarray(arrays["previous_coordinate470"], dtype=float)
        if not np.array_equal(previous, coordinates[-1]):
            raise RuntimeError(f"coordinate lineage changed: {directory}")
        coordinates.append(np.asarray(arrays["current_coordinate470"], dtype=float))
        timestep = 1.0e-7 if position <= manifest.FULL_STEP_COUNT else 5.0e-8
        elapsed += timestep
        times.append(elapsed)
        family = "full" if position <= manifest.FULL_STEP_COUNT else "half"
        local = position if family == "full" else position - manifest.FULL_STEP_COUNT
        labels.append(f"{family}_step_{local:02d}")
        hidden_fractions.append(float(metrics["hidden_secant_fraction"]))
    return (
        np.asarray(times),
        np.asarray(coordinates),
        labels,
        np.asarray(hidden_fractions),
    )


def _rank_for_energy(matrix: np.ndarray, target: float) -> tuple[int, np.ndarray, np.ndarray]:
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    energy = singular_values**2
    if not np.any(energy):
        return 0, singular_values, right.T[:, :0]
    cumulative = np.cumsum(energy) / np.sum(energy)
    rank = int(np.searchsorted(cumulative, target, side="left") + 1)
    return rank, singular_values, right.T[:, :rank]


def _holdout_interpolation(
    times: np.ndarray,
    coordinates: np.ndarray,
    train_indices: tuple[int, ...],
    holdout_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = []
    local_chords = []
    brackets = []
    train = np.asarray(train_indices, dtype=int)
    for index in holdout_indices:
        left_candidates = train[train < index]
        right_candidates = train[train > index]
        if not len(left_candidates) or not len(right_candidates):
            raise RuntimeError("held-out state is not bracketed by training states")
        left = int(left_candidates[-1])
        right = int(right_candidates[0])
        weight = float((times[index] - times[left]) / (times[right] - times[left]))
        predictions.append((1.0 - weight) * coordinates[left] + weight * coordinates[right])
        local_chords.append(float(np.linalg.norm(coordinates[right] - coordinates[left])))
        brackets.append((left, right))
    return np.asarray(predictions), np.asarray(local_chords), np.asarray(brackets)


def _nonlocal_separation(coordinates: np.ndarray) -> float:
    minimum = math.inf
    for left in range(len(coordinates)):
        for right in range(left + 3, len(coordinates)):
            minimum = min(minimum, float(np.linalg.norm(coordinates[right] - coordinates[left])))
    return minimum


def _analyze_geometry(
    times: np.ndarray,
    coordinates: np.ndarray,
    hidden_fractions: np.ndarray,
    macro_restriction: np.ndarray,
    hidden_basis: np.ndarray,
    hidden_dual: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    tiny = np.finfo(float).tiny
    decomposition = np.vstack((macro_restriction, hidden_dual))
    inverse = np.linalg.solve(decomposition, np.eye(decomposition.shape[0]))
    macro_lift = inverse[:, : macro_restriction.shape[0]]
    inverse_hidden_basis = inverse[:, macro_restriction.shape[0] :]

    macro = (macro_restriction @ coordinates.T).T
    hidden = (hidden_dual @ coordinates.T).T
    reconstructed = (macro_lift @ macro.T + hidden_basis @ hidden.T).T
    fixed_macro_coordinates = (
        macro_lift @ np.repeat(macro[[0]], len(macro), axis=0).T
        + hidden_basis @ hidden.T
    ).T

    increments = np.diff(coordinates, axis=0)
    durations = np.diff(times)
    step_lengths = np.linalg.norm(increments, axis=1)
    secants = increments / durations[:, None]
    secant_norms = np.linalg.norm(secants, axis=1)
    unit_secants = secants / np.maximum(secant_norms[:, None], tiny)
    adjacent_cosines = np.sum(unit_secants[:-1] * unit_secants[1:], axis=1)
    adjacent_cosines = np.clip(adjacent_cosines, -1.0, 1.0)
    turn_angles = np.degrees(np.arccos(adjacent_cosines))
    path_length = float(np.sum(step_lengths))
    chord = coordinates[-1] - coordinates[0]
    chord_norm = float(np.linalg.norm(chord))
    chord_unit = chord / max(chord_norm, tiny)
    forward_chord_cosines = unit_secants @ chord_unit
    progress = np.concatenate(([0.0], np.cumsum(step_lengths))) / max(path_length, tiny)

    hidden_increments = np.diff(hidden, axis=0)
    displacement_rank, displacement_singular_values, displacement_modes = _rank_for_energy(
        hidden[1:] - hidden[[0]], manifest.ENERGY_CAPTURE_TARGET
    )
    secant_rank, secant_singular_values, secant_modes = _rank_for_energy(
        hidden_increments / durations[:, None], manifest.ENERGY_CAPTURE_TARGET
    )
    selected_rank = max(displacement_rank, secant_rank)

    predictions, local_chords, brackets = _holdout_interpolation(
        times,
        fixed_macro_coordinates,
        manifest.TRAIN_STATE_INDICES,
        manifest.HOLDOUT_STATE_INDICES,
    )
    truth = fixed_macro_coordinates[np.asarray(manifest.HOLDOUT_STATE_INDICES)]
    errors = np.linalg.norm(predictions - truth, axis=1)
    path_relative_errors = errors / max(path_length, tiny)
    local_relative_errors = errors / np.maximum(local_chords, tiny)
    predicted_macro = (macro_restriction @ predictions.T).T
    truth_macro = (macro_restriction @ truth.T).T
    macro_errors = np.linalg.norm(predicted_macro - truth_macro, axis=1)

    minimum_step = float(np.min(step_lengths))
    nonlocal_separation = _nonlocal_separation(fixed_macro_coordinates)
    table_bytes = int(
        len(manifest.TRAIN_STATE_INDICES)
        * (selected_rank + 1)
        * np.dtype(np.float64).itemsize
        + hidden_basis.shape[0] * selected_rank * np.dtype(np.float64).itemsize
    )
    lift_flops = int(
        2 * hidden_basis.shape[0] * max(selected_rank, 1)
        + 2 * macro_lift.shape[0] * macro_lift.shape[1]
    )

    closures = {
        "R_L_minus_I_inf": float(
            np.linalg.norm(
                macro_restriction @ macro_lift - np.eye(macro_lift.shape[1]),
                ord=np.inf,
            )
        ),
        "R_Z_inf": float(np.linalg.norm(macro_restriction @ hidden_basis, ord=np.inf)),
        "Q_L_inf": float(np.linalg.norm(hidden_dual @ macro_lift, ord=np.inf)),
        "Q_Z_minus_I_inf": float(
            np.linalg.norm(
                hidden_dual @ hidden_basis - np.eye(hidden_basis.shape[1]),
                ord=np.inf,
            )
        ),
        "inverse_hidden_basis_relative_error": float(
            np.linalg.norm(inverse_hidden_basis - hidden_basis)
            / max(float(np.linalg.norm(hidden_basis)), tiny)
        ),
        "trajectory_reconstruction_relative_error": float(
            np.linalg.norm(reconstructed - coordinates)
            / max(float(np.linalg.norm(coordinates)), tiny)
        ),
    }
    gate_values = {
        "maximum_decomposition_closure": max(closures.values()),
        "selected_hidden_embedding_rank": selected_rank,
        "maximum_turn_angle_degrees": float(np.max(turn_angles)),
        "minimum_forward_chord_cosine": float(np.min(forward_chord_cosines)),
        "maximum_holdout_error_over_path_length": float(np.max(path_relative_errors)),
        "maximum_holdout_error_over_local_chord": float(np.max(local_relative_errors)),
        "maximum_holdout_macro_error": float(np.max(macro_errors)),
        "nonlocal_separation_over_minimum_step": float(nonlocal_separation / max(minimum_step, tiny)),
        "maximum_macro_drift_from_seed": float(
            np.max(np.linalg.norm(macro - macro[[0]], axis=1))
        ),
        "minimum_transition_hidden_fraction": float(np.min(hidden_fractions)),
        "online_lift_flops": lift_flops,
        "online_table_bytes": table_bytes,
    }
    gates = {
        "decomposition_closure": gate_values["maximum_decomposition_closure"]
        <= manifest.DECOMPOSITION_CLOSURE_TOLERANCE,
        "embedding_rank": selected_rank <= manifest.MAXIMUM_HIDDEN_EMBEDDING_RANK,
        "turn_angle": gate_values["maximum_turn_angle_degrees"]
        <= manifest.MAXIMUM_TURN_ANGLE_DEGREES,
        "forward_progress": gate_values["minimum_forward_chord_cosine"]
        >= manifest.MINIMUM_FORWARD_CHORD_COSINE,
        "holdout_path_error": gate_values["maximum_holdout_error_over_path_length"]
        <= manifest.MAXIMUM_HOLDOUT_ERROR_OVER_PATH_LENGTH,
        "holdout_local_error": gate_values["maximum_holdout_error_over_local_chord"]
        <= manifest.MAXIMUM_HOLDOUT_ERROR_OVER_LOCAL_CHORD,
        "holdout_macro_error": gate_values["maximum_holdout_macro_error"]
        <= manifest.MAXIMUM_HOLDOUT_MACRO_ERROR,
        "no_self_intersection": gate_values["nonlocal_separation_over_minimum_step"]
        >= manifest.MINIMUM_NONLOCAL_SEPARATION_OVER_MINIMUM_STEP,
        "macro_drift": gate_values["maximum_macro_drift_from_seed"]
        <= manifest.MAXIMUM_MACRO_DRIFT_FROM_SEED,
        "transition_sector": gate_values["minimum_transition_hidden_fraction"]
        >= manifest.MINIMUM_TRANSITION_HIDDEN_FRACTION,
        "online_lift_cost": lift_flops <= manifest.MAXIMUM_ONLINE_LIFT_FLOPS,
        "online_table_size": table_bytes <= manifest.MAXIMUM_ONLINE_TABLE_BYTES,
    }
    passed = all(gates.values())
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "state_count": len(coordinates),
        "interval_count": len(secants),
        "trajectory_duration_seconds": float(times[-1] - times[0]),
        "path_length": path_length,
        "endpoint_chord_length": chord_norm,
        "path_tortuosity": path_length / max(chord_norm, tiny),
        "minimum_step_length": minimum_step,
        "maximum_step_length": float(np.max(step_lengths)),
        "displacement_rank_at_target": displacement_rank,
        "secant_rank_at_target": secant_rank,
        "selected_hidden_embedding_rank": selected_rank,
        "decomposition_closures": closures,
        "gate_values": gate_values,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "transition_dynamic_dimension": 1 if passed else None,
        "hot_exit_observed": False,
        "transition_impulse_fit_authorized": False,
        "new_truth_calls": 0,
    }
    arrays = {
        "trajectory_times_seconds": times,
        "trajectory_coordinates470": coordinates,
        "trajectory_macro_coordinates82": macro,
        "trajectory_hidden_coordinates388": hidden,
        "fixed_seed_macro_trajectory_coordinates470": fixed_macro_coordinates,
        "trajectory_secants470_per_s": secants,
        "normalized_progress_s": progress,
        "adjacent_secant_cosines": adjacent_cosines,
        "turn_angles_degrees": turn_angles,
        "forward_chord_cosines": forward_chord_cosines,
        "hidden_displacement_singular_values": displacement_singular_values,
        "hidden_secant_singular_values": secant_singular_values,
        "selected_displacement_modes388xr": displacement_modes,
        "selected_secant_modes388xr": secant_modes,
        "macro_lift_L470x82": macro_lift,
        "holdout_predictions470": predictions,
        "holdout_truth470": truth,
        "holdout_errors": errors,
        "holdout_path_relative_errors": path_relative_errors,
        "holdout_local_relative_errors": local_relative_errors,
        "holdout_brackets": brackets,
        "hidden_secant_fractions": hidden_fractions,
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
                    "sha256": _sha(path),
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
    catalog = _read(CANONICAL_SUMMARY)
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
    _write_json(CANONICAL_SUMMARY, catalog)


def _execute() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("transition-tube geometry result already exists")
    lock = _validate_lock(require_clean=True)
    times, coordinates, labels, hidden_fractions = _ordered_trajectory()
    tangent = _load_npz(manifest.TANGENT_ARRAYS)
    metrics, arrays = _analyze_geometry(
        times,
        coordinates,
        hidden_fractions,
        np.asarray(tangent["macro_restriction_R82"], dtype=float),
        np.asarray(tangent["hidden_basis_Z388"], dtype=float),
        np.asarray(tangent["hidden_dual_Q388"], dtype=float),
    )
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "geometry_metrics.json", metrics)
    np.savez(CANONICAL_DIRECTORY / "geometry_arrays.npz", **arrays)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_directory": str(manifest.CANONICAL_DIRECTORY.relative_to(ROOT)),
            "manifest_hashes": lock["manifest_hashes"],
            "trajectory_labels": labels,
            "decisive_input_hashes": lock["contract"]["decisive_input_hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "transition_dynamic_dimension": metrics["transition_dynamic_dimension"],
        "selected_hidden_embedding_rank": metrics["selected_hidden_embedding_rank"],
        "trajectory_duration_seconds": metrics["trajectory_duration_seconds"],
        "held_out_state_count": len(manifest.HOLDOUT_STATE_INDICES),
        "failed_gates": metrics["failed_gates"],
        "hot_exit_observed": False,
        "transition_impulse_fit_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": _git("rev-parse", "HEAD"),
            "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": lock["contract"]["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Transition-tube geometry WP10c9d6c7c3b5c4f25ds",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"The accepted trajectory contains {metrics['state_count']} states over {metrics['trajectory_duration_seconds']:.9e} s. The selected hidden embedding rank is {metrics['selected_hidden_embedding_rank']}; the dynamic transition coordinate is {'one scalar s' if metrics['passed'] else 'not certified'}.",
                "",
                f"Maximum held-out error/path length: `{metrics['gate_values']['maximum_holdout_error_over_path_length']:.6e}`. Maximum held-out error/local chord: `{metrics['gate_values']['maximum_holdout_error_over_local_chord']:.6e}`. Maximum turn: `{metrics['gate_values']['maximum_turn_angle_degrees']:.6e}` degrees.",
                "",
                f"Failed gates: `{metrics['failed_gates']}`.",
                "",
                "The hot exit remains unobserved. This result cannot authorize an impulse fit, hot branch, or reduced slow evolution.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if not args.analyze:
        parser.error("use --analyze")
    payload = _execute()
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
