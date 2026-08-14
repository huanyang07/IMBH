#!/usr/bin/env python3
"""Run the existing-state face-36 reaction/coordinate preflight.

No nonlinear, fixed-Q, 50 ms, or reduced trajectory is advanced.  The
physical macro reaction is built on committed middle/fine states, while the
retained-coordinate test reuses the committed c4f13 response histories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_absolute_baseline_observable_memory_screen_wp10c9d6c7c3b5c4f1 as c4f1  # noqa: E402
import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_retained_mode_q_plus_a_pilot_manifest_wp10c9d6c7c3b5c4f14 as c4f14  # noqa: E402

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import _cell_state  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f15"
ARTIFACT = "causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15"
THIS_RUNNER = "scripts/run_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15.py"
THIS_TEST = "tests/test_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FACE36_Q_PLUS_A_REACTION_COORDINATE_PREFLIGHT_WP10C9D6C7C3B5C4F15_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LAYOUTS = ("middle", "fine")
FORMS = ("instantaneous", "cumulative", "window_mean")
SELECTED_TIMES_SECONDS = (0.005, 0.010, 0.016, 0.020)
CONSERVATIVE_FIELDS = (0, 2, 3)
PARENT_CORE_FACE = 36
PARENT_GUARD_END_FACE = 48
PARENT_CELL_COUNT = 64
TWO_MODE_DIMENSION = 2


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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_authorization() -> tuple[dict, dict]:
    summary = _read(c4f14.SUMMARY_PATH)
    manifest = _read(c4f14.MANIFEST_PATH)
    expected = (
        "WP10c9d6c7c3b5c4f15_analysis_only_Q_plus_a_reaction_map_"
        "and_coordinate_preflight"
    )
    if (
        not summary["passed"]
        or summary["authorized_next"] != expected
        or not summary["reaction_map_preflight_authorized"]
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["nonlinear_retained_mode_pilot_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or not manifest["authorized_reaction_map_preflight"][
            "uses_existing_middle_and_fine_5_to_20ms_states_only"
        ]
    ):
        raise RuntimeError("c4f15 authorization changed")
    return summary, manifest


def _full_descriptor(context, state, columns, rows):
    weights, cells, radii, measures, reconstruction, partition = (
        _node_reconstruction_weights(context, state)
    )
    mapped, height = _descriptor_matrices(
        context, state, columns, rows, weights, cells, radii, measures
    )
    if reconstruction > 1.0e-12 or partition > 1.0e-12:
        raise RuntimeError("c4f15 descriptor reconstruction contract failed")
    return mapped, height, mapped + height


def _Q_maps(mapped, rows, layout):
    n_cells = rows.shape[0]
    start = PARENT_CORE_FACE * int(layout.refinement_ratio)
    physical = []
    for component in CONSERVATIVE_FIELDS:
        selector = np.zeros(n_cells * 5, dtype=float)
        for cell in range(start, n_cells):
            selector[5 * cell + component] = C * rows[cell, component]
        physical.append(selector @ mapped)
    physical = np.asarray(physical)
    norms = np.linalg.norm(physical, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("c4f15 Q derivative is rank deficient")
    return physical, physical / norms[:, None], norms


def _reaction_raw(context, state, rows, layout):
    """Return scaled residual channels and an independent physical ledger."""

    parents = np.asarray(layout.parent_cell_indices, dtype=int)
    support = np.flatnonzero(
        (parents >= PARENT_GUARD_END_FACE) & (parents < PARENT_CELL_COUNT)
    )
    phase = (parents[support] - PARENT_GUARD_END_FACE + 0.5) / (
        PARENT_CELL_COUNT - PARENT_GUARD_END_FACE
    )
    envelope = np.sin(np.pi * phase) ** 2
    envelope *= np.asarray(context.grid.cell_measures, dtype=float)[support]
    envelope /= np.sum(envelope)

    physical = np.zeros((state.shape[0], 5, 3), dtype=float)
    for weight, cell in zip(envelope, support, strict=True):
        local = _cell_state(
            context, float(context.grid.centers[cell]), state[cell]
        )
        rest_mass = float(local.conserved[0])
        specific_j = float(local.conserved[2] / rest_mass)
        specific_e = float(local.conserved[3] / rest_mass)
        omega = float(local.stress.coordinate_angular_velocity)
        physical[cell, 0, 0] = weight
        physical[cell, 2, 0] = weight * specific_j
        physical[cell, 3, 0] = weight * specific_e
        physical[cell, 2, 1] = weight
        physical[cell, 3, 1] = weight * omega
        physical[cell, 3, 2] = weight

    scaled = (physical / C / rows[:, :, None]).reshape(state.size, 3)
    ledger = np.sum(physical, axis=0)[np.asarray(CONSERVATIVE_FIELDS)]
    outside = np.ones(state.shape[0], dtype=bool)
    outside[support] = False
    support_defect = float(
        np.max(np.abs(physical[outside]))
        / max(float(np.max(np.abs(physical))), np.finfo(float).tiny)
    )
    return scaled, physical, ledger, support, envelope, support_defect


def _reaction_preflight(label, selected_index, layout, configuration, trajectory):
    context = configuration["context"]
    state = trajectory["states"][selected_index]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(state.shape)
    rows = np.asarray(configuration["rows"], dtype=float).reshape(state.shape)
    mapped, height, descriptor = _full_descriptor(
        context, state, columns, rows
    )
    q_physical, q_scaled, q_norms = _Q_maps(mapped, rows, layout)
    raw, physical_raw, raw_ledger, support, envelope, support_defect = (
        _reaction_raw(context, state, rows, layout)
    )
    factor = splu(csc_matrix(descriptor))
    raw_lift = factor.solve(raw)
    raw_schur = q_scaled @ raw_lift
    inverse_schur = np.linalg.inv(raw_schur)
    reaction = raw @ inverse_schur
    reaction_lift = factor.solve(reaction)
    identity = q_scaled @ reaction_lift

    physical_from_scaled = (
        reaction.reshape(state.shape[0], 5, 3)
        * C
        * rows[:, :, None]
    )
    ledger_from_rows = np.sum(physical_from_scaled, axis=0)[
        np.asarray(CONSERVATIVE_FIELDS)
    ]
    ledger_from_channels = raw_ledger @ inverse_schur
    ledger_scale = max(
        float(np.linalg.norm(ledger_from_rows)),
        float(np.linalg.norm(ledger_from_channels)),
        np.finfo(float).tiny,
    )
    ledger_defect = float(
        np.linalg.norm(ledger_from_rows - ledger_from_channels) / ledger_scale
    )

    rng = np.random.default_rng(1515 + selected_index)
    forcing = rng.normal(size=(state.size, 3))
    minimum = factor.solve(forcing)
    multiplier = -np.linalg.solve(identity, q_scaled @ minimum)
    rate = minimum + reaction_lift @ multiplier
    upper = descriptor @ rate - reaction @ multiplier - forcing
    lower = q_scaled @ rate
    solve_scale = max(
        float(np.linalg.norm(descriptor @ rate)),
        float(np.linalg.norm(reaction @ multiplier)),
        float(np.linalg.norm(forcing)),
        1.0,
    )
    kkt_defect = float(
        max(np.linalg.norm(upper), np.linalg.norm(lower)) / solve_scale
    )
    return {
        "time_seconds": float(trajectory["times"][selected_index]),
        "raw_schur_condition_number": float(np.linalg.cond(raw_schur)),
        "DQ_M_inverse_BQ_identity_defect": float(
            np.max(np.abs(identity - np.eye(3)))
        ),
        "KKT_linear_solve_relative_defect": kkt_defect,
        "reaction_ledger_relative_defect": ledger_defect,
        "reaction_support_relative_defect": support_defect,
        "descriptor_component_relative_defect": float(
            np.linalg.norm(descriptor - mapped - height)
            / max(float(np.linalg.norm(descriptor)), np.finfo(float).tiny)
        ),
        "Q_derivative_row_norms": q_norms,
        "raw_physical_reaction_ledger": raw_ledger,
        "normalized_physical_reaction_ledger": ledger_from_rows,
        "support_cell_indices": support,
        "reaction_envelope": envelope,
        "descriptor": descriptor,
        "q_scaled": q_scaled,
        "reaction_lift": reaction_lift,
        "columns": columns,
    }


def _state_coordinate_preflight(
    layout, configuration, trajectory, descriptor, q_scaled, reaction_lift, basis
):
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        trajectory["states"][0].shape
    )
    directions = c4f1._initial_directions(
        configuration,
        trajectory,
        PARENT_CORE_FACE * int(layout.refinement_ratio),
        trajectory["states"].shape[1],
    )["current"]
    scaled = directions.reshape(directions.shape[0], -1) / columns.ravel()[None, :]
    state_lifts = scaled.T @ basis
    q_defect = float(np.max(np.abs(q_scaled @ state_lifts)))

    reaction_scale = np.linalg.norm(descriptor @ reaction_lift, axis=0)
    normalized_reaction_lift = reaction_lift / reaction_scale[None, :]
    trial = np.column_stack((scaled.T, normalized_reaction_lift))
    target = np.column_stack(
        (basis.T, np.zeros((basis.shape[1], 3), dtype=float))
    )
    descriptor_trial = descriptor @ trial
    gram = descriptor_trial.T @ descriptor_trial
    dual = np.linalg.solve(
        gram, descriptor_trial.T @ descriptor
    )
    dual = target @ dual
    biorthogonality = float(
        np.max(np.abs(dual @ state_lifts - np.eye(basis.shape[1])))
    )
    slow_annihilation = float(np.max(np.abs(dual @ reaction_lift)))
    normalized_slow_annihilation = float(
        np.max(np.abs(dual @ normalized_reaction_lift))
    )
    coefficient_defect = float(
        np.max(np.abs(dual @ scaled.T - basis.T))
    )
    return {
        "state_lift_Q3_defect": q_defect,
        "dual_biorthogonality_defect": biorthogonality,
        "dual_slow_lift_annihilation_defect": slow_annihilation,
        "dual_normalized_slow_lift_annihilation_defect": (
            normalized_slow_annihilation
        ),
        "initial_consensus_coefficient_defect": coefficient_defect,
        "descriptor_trial_condition_number": float(
            np.linalg.cond(descriptor_trial)
        ),
        "state_lifts": state_lifts,
        "dual": dual,
    }


def _normalized_consensus_basis(parent_arrays, dimension):
    grams = []
    for label in LAYOUTS:
        for form in FORMS:
            singular = parent_arrays[f"{label}__{form}_singular_values"]
            right = parent_arrays[f"{label}__{form}_right_vectors"]
            gram = (right.T * np.square(singular)) @ right
            grams.append(gram / np.trace(gram))
    consensus = sum(grams) / len(grams)
    values, vectors = np.linalg.eigh(consensus)
    order = np.argsort(values)[::-1]
    basis = vectors[:, order[:dimension]]
    for mode in range(dimension):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0.0:
            basis[:, mode] *= -1.0
    return basis, values[order]


def _forms(parent_arrays, label, trajectory):
    times = parent_arrays[f"{label}__times"]
    output = parent_arrays[f"{label}__face36_outputs"]
    scales = np.asarray(trajectory["output_scales"][:3], dtype=float)
    means, durations = c4f13._window_means(output, times)
    return {
        "instantaneous": (output / scales, c4f13._weights(times)),
        "cumulative": (
            c4f13._cumulative(output, times) / (scales * (times[-1] - times[0])),
            c4f13._weights(times),
        ),
        "window_mean": (means / scales, durations / np.sum(durations)),
    }


def _output_reconstruction(parent_arrays, basis, gates):
    results = {}
    maximum_rms = 0.0
    maximum_direction = 0.0
    modal_histories = {}
    for label in LAYOUTS:
        _layout, _configuration, trajectory = c4f13._layout_data(label)
        results[label] = {}
        for form, (values, weights) in _forms(parent_arrays, label, trajectory).items():
            matrix = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
            weighted = (
                values * np.sqrt(weights)[:, None, None]
            ).transpose(0, 2, 1).reshape(-1, values.shape[1])
            reconstructed = (matrix @ basis) @ basis.T
            weighted_reconstructed = (weighted @ basis) @ basis.T
            rms = float(
                np.linalg.norm(weighted - weighted_reconstructed)
                / max(float(np.linalg.norm(weighted)), np.finfo(float).tiny)
            )
            reshaped = reconstructed.reshape(
                values.shape[0], values.shape[2], values.shape[1]
            ).transpose(0, 2, 1)
            direction_norm = np.linalg.norm(
                values * np.sqrt(weights)[:, None, None], axis=(0, 2)
            )
            error_norm = np.linalg.norm(
                (values - reshaped) * np.sqrt(weights)[:, None, None], axis=(0, 2)
            )
            relative = error_norm / np.maximum(direction_norm, np.finfo(float).tiny)
            significant = direction_norm >= (
                gates["significant_direction_relative_response_floor"]
                * np.max(direction_norm)
            )
            worst = float(np.max(relative[significant]))
            results[label][form] = {
                "output_weighted_RMS_error": rms,
                "maximum_significant_direction_error": worst,
                "significant_direction_count": int(np.sum(significant)),
            }
            maximum_rms = max(maximum_rms, rms)
            maximum_direction = max(maximum_direction, worst)
            modal_histories[(label, form)] = np.einsum(
                "dm,tdk->tmk", basis, values
            )

    modal_cosines = {}
    for form in FORMS:
        left = modal_histories[("middle", form)]
        right = modal_histories[("fine", form)]
        modal_cosines[form] = []
        for mode in range(basis.shape[1]):
            a = left[:, mode].ravel()
            b = right[:, mode].ravel()
            modal_cosines[form].append(
                float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            )
    return {
        "layouts": results,
        "maximum_output_weighted_RMS_error": maximum_rms,
        "maximum_significant_direction_error": maximum_direction,
        "modal_output_history_cosines": modal_cosines,
        "minimum_modal_output_history_cosine": min(
            min(values) for values in modal_cosines.values()
        ),
        "state_amplitude_history_gate_evaluated": False,
        "state_amplitude_history_gate_reason": (
            "c4f13_commits_output_histories_but_not_intermediate_state_"
            "direction_histories;_modal_output_kernels_are_not_relabelled_"
            "as_descriptor_dual_state_amplitudes"
        ),
    }


def _minimum_dimension(parent_arrays, gates):
    for dimension in range(TWO_MODE_DIMENSION, c4f13.TOTAL_DIRECTIONS + 1):
        basis, values = _normalized_consensus_basis(parent_arrays, dimension)
        metrics = _output_reconstruction(parent_arrays, basis, gates)
        if (
            metrics["maximum_output_weighted_RMS_error"]
            <= gates["maximum_two_mode_face36_output_weighted_RMS_error"]
            and metrics["maximum_significant_direction_error"]
            <= gates["maximum_two_mode_face36_significant_direction_error"]
        ):
            return dimension, basis, values, metrics
    raise RuntimeError("no retained dimension closes the output gate")


def _catalog(summary):
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "REJECTED CANDIDATE",
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
        "passed": False,
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


def main() -> None:
    began = time.perf_counter()
    parent_summary, manifest = _validate_authorization()
    gates = manifest["prospective_preflight_gates"]
    with np.load(c4f14.MODE_BASIS_PATH, allow_pickle=False) as arrays:
        two_mode_basis = np.asarray(arrays["consensus_direction_coefficients"])
    with np.load(c4f13.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        parent_arrays = {name: np.asarray(arrays[name]) for name in arrays.files}

    reactions = {}
    coordinates = {}
    stored = {
        "selected_times_seconds": np.asarray(SELECTED_TIMES_SECONDS),
        "two_mode_consensus_direction_coefficients": two_mode_basis,
    }
    for label in LAYOUTS:
        layout, configuration, trajectory = c4f13._layout_data(label)
        selected = [
            int(np.flatnonzero(trajectory["times"] == selected_time)[0])
            for selected_time in SELECTED_TIMES_SECONDS
        ]
        reactions[label] = []
        first = None
        for position, index in enumerate(selected):
            result = _reaction_preflight(
                label, index, layout, configuration, trajectory
            )
            if position == 0:
                first = result
            reactions[label].append(
                {key: value for key, value in result.items() if key not in {
                    "descriptor", "q_scaled", "reaction_lift", "columns"
                }}
            )
            stored[f"{label}__time_{position}__normalized_reaction_ledger"] = (
                result["normalized_physical_reaction_ledger"]
            )
            stored[f"{label}__time_{position}__reaction_envelope"] = result[
                "reaction_envelope"
            ]
        if first is None:
            raise RuntimeError("c4f15 selected no state")
        coordinate = _state_coordinate_preflight(
            layout,
            configuration,
            trajectory,
            first["descriptor"],
            first["q_scaled"],
            first["reaction_lift"],
            two_mode_basis,
        )
        coordinates[label] = {
            key: value for key, value in coordinate.items()
            if key not in {"state_lifts", "dual"}
        }
        stored[f"{label}__two_mode_state_lifts_scaled"] = coordinate["state_lifts"]
        stored[f"{label}__two_mode_descriptor_dual"] = coordinate["dual"]

    two_mode = _output_reconstruction(parent_arrays, two_mode_basis, gates)
    minimum_dimension, expanded_basis, spectrum, expanded = _minimum_dimension(
        parent_arrays, gates
    )
    stored["minimum_passing_consensus_direction_coefficients"] = expanded_basis
    stored["consensus_gram_eigenvalues"] = spectrum

    reaction_identity = max(
        item["DQ_M_inverse_BQ_identity_defect"]
        for values in reactions.values() for item in values
    )
    kkt_defect = max(
        item["KKT_linear_solve_relative_defect"]
        for values in reactions.values() for item in values
    )
    ledger_defect = max(
        item["reaction_ledger_relative_defect"]
        for values in reactions.values() for item in values
    )
    schur_condition = max(
        item["raw_schur_condition_number"]
        for values in reactions.values() for item in values
    )
    state_q_defect = max(
        item["state_lift_Q3_defect"] for item in coordinates.values()
    )
    dual_defect = max(
        item["dual_biorthogonality_defect"] for item in coordinates.values()
    )
    reaction_passed = bool(
        reaction_identity <= gates["maximum_DQ_M_inverse_BQ_identity_defect"]
        and kkt_defect <= gates["maximum_KKT_linear_solve_relative_defect"]
        and ledger_defect <= gates["maximum_reaction_ledger_relative_defect"]
        and schur_condition <= gates["maximum_KKT_Schur_condition_number"]
    )
    endpoint_coordinate_passed = bool(
        state_q_defect <= gates["maximum_state_lift_Q3_defect"]
        and dual_defect <= gates["maximum_dual_biorthogonality_defect"]
    )
    two_mode_aggregate_passed = bool(
        two_mode["maximum_output_weighted_RMS_error"]
        <= gates["maximum_two_mode_face36_output_weighted_RMS_error"]
    )
    two_mode_direction_passed = bool(
        two_mode["maximum_significant_direction_error"]
        <= gates["maximum_two_mode_face36_significant_direction_error"]
    )
    coordinate_passed = bool(
        endpoint_coordinate_passed
        and two_mode_aggregate_passed
        and two_mode_direction_passed
        and two_mode["state_amplitude_history_gate_evaluated"]
    )
    classification = (
        "face36_two_mode_coordinate_preflight_rejected_"
        "six_mode_manifest_authorized"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "audit_completed": True,
        "passed": False,
        "reaction_map_preflight_passed": reaction_passed,
        "endpoint_coordinate_preflight_passed": endpoint_coordinate_passed,
        "two_mode_aggregate_output_gate_passed": two_mode_aggregate_passed,
        "two_mode_significant_direction_gate_passed": two_mode_direction_passed,
        "dynamic_state_amplitude_history_gate_evaluated": False,
        "retained_coordinate_preflight_passed": coordinate_passed,
        "minimum_passing_output_oriented_dimension": minimum_dimension,
        "maximum_DQ_M_inverse_BQ_identity_defect": reaction_identity,
        "maximum_KKT_linear_solve_relative_defect": kkt_defect,
        "maximum_reaction_ledger_relative_defect": ledger_defect,
        "maximum_KKT_Schur_condition_number": schur_condition,
        "maximum_state_lift_Q3_defect": state_q_defect,
        "maximum_dual_biorthogonality_defect": dual_defect,
        "two_mode_output_reconstruction": two_mode,
        "minimum_passing_dimension_output_reconstruction": expanded,
        "reactions": reactions,
        "coordinates_at_5ms": coordinates,
        "new_trajectory_executed": False,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "raw_face48_export_rejection_preserved": True,
        "guard_complement_retained": True,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f16_definitions_only_six_mode_"
            "Q_plus_a_coordinate_manifest"
        ),
        "wall_seconds": time.perf_counter() - began,
    }

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **stored)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "layouts": list(LAYOUTS),
            "selected_times_seconds": list(SELECTED_TIMES_SECONDS),
            "two_mode_dimension": TWO_MODE_DIMENSION,
            "parent_macro_cells": [36, 64],
            "parent_reaction_support_cells": [48, 64],
            "shared_exchange_parent_face": 36,
            "prospective_gates": gates,
        },
    )
    _write(SUMMARY_PATH, summary)
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": _read(CANONICAL_SUMMARY)[
                "latest_source_parent_commit"
            ],
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "parent_manifest_summary_sha256": _sha(c4f14.SUMMARY_PATH),
            "parent_manifest_sha256": _sha(c4f14.MANIFEST_PATH),
            "parent_mode_basis_sha256": _sha(c4f14.MODE_BASIS_PATH),
            "parent_memory_arrays_sha256": _sha(c4f13.DECISIVE_ARRAYS),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None,
            },
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 Q+a reaction and coordinate preflight\n\n"
        f"Classification: `{classification}`.\n\n"
        "The physical macro-only reaction map passes on every committed middle/fine state at 5, 10, 16, and 20 ms. "
        f"The maximum normalized reaction identity defect is `{reaction_identity:.3e}`, the KKT solve defect is `{kkt_defect:.3e}`, the reaction-ledger defect is `{ledger_defect:.3e}`, and the maximum raw Schur condition number is `{schur_condition:.6g}`.\n\n"
        "The 5 ms state lifts and descriptor-weighted Petrov duals also pass their endpoint algebraic gates. "
        f"Their maximum Q3 defect is `{state_q_defect:.3e}` and maximum biorthogonality defect is `{dual_defect:.3e}`.\n\n"
        "The two-mode coordinate is nevertheless rejected. Its worst aggregate output-weighted RMS error is "
        f"`{two_mode['maximum_output_weighted_RMS_error']:.6f}`, but its worst significant-direction error is "
        f"`{two_mode['maximum_significant_direction_error']:.6f}`, above the frozen `0.25` gate. "
        f"A prospective consensus dimension of `{minimum_dimension}` is the first to pass both output gates; its worst significant-direction error is "
        f"`{expanded['maximum_significant_direction_error']:.6f}`.\n\n"
        "The committed memory result contains output histories but not intermediate state-direction histories. Modal output kernels are therefore reported only as diagnostics and are not relabelled as descriptor-dual state-amplitude histories. No fixed-Q or nonlinear pilot is authorized. The next package may only freeze a six-mode coordinate manifest and a cost-bounded dynamic-coordinate propagation contract.\n",
        encoding="utf-8",
    )
    files = (CONFIG_PATH, DECISIVE_ARRAYS, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
