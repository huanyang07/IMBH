"""Audit region-selective causal slow closures at the certified WP10c8b state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigvals, eigvalsh, expm, solve

import run_causal_slow_mode_audit_wp10c8a as wp10c8a
import run_causal_stress_time_audit_wp10c8b as wp10c8b
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_reduced_descriptor_matrices,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "748d0ad1420c90ecb0efd2aac38c8af5cd8dd62f"
WP10C8B_OUTPUT = (
    ROOT / "outputs/tables/causal_stress_time_audit_wp10c8b.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_region_selective_closure_audit_wp10c8c.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_region_selective_closure_audit_wp10c8c_arrays.npz"
)
RESOLUTIONS = (64, 128)
CERTIFIED_LABEL = "t_0p125"
SECANT_LABEL = "t_0p10"
CERTIFIED_TIME_SECONDS = 0.125
REGIONS_RG = (
    ("horizon_to_3rg", 0.0, 3.0),
    ("horizon_to_6rg", 0.0, 6.0),
    ("6_to_20rg", 6.0, 20.0),
    ("20_to_60rg", 20.0, 60.0),
    ("6_to_60rg", 6.0, 60.0),
    ("horizon_to_60rg", 0.0, 60.0),
    ("60_to_200rg", 60.0, 200.0),
    ("200rg_to_outer", 200.0, np.inf),
    ("full_domain", 0.0, np.inf),
)
ELIMINATION_CHARTS = (
    ("radial_momentum", (1,)),
    ("causal_stress", (4,)),
    ("radial_momentum_and_stress", (1, 4)),
)
STABILITY_TOLERANCE_PER_S = 1.0e-8
MAXIMUM_PHYSICAL_WEIGHTED_RMS = 0.15
MAXIMUM_PHYSICAL_95TH_PERCENTILE = 0.30
MINIMUM_FAST_TO_RETAINED_GAP = 3.0
MAXIMUM_FAST_EIGENVECTOR_CONDITION = 1.0e12
MAXIMUM_FAST_TRANSIENT_AMPLIFICATION = 2.0
MAXIMUM_SCHUR_SOLVE_RELATIVE_DEFECT = 1.0e-10
MAXIMUM_STATE_PROJECTION_RELATIVE_ERROR = 0.10
MAXIMUM_STATE_TANGENT_RELATIVE_ERROR = 0.10
MAXIMUM_MANIFOLD_INVARIANCE_RELATIVE_DEFECT = 0.10
MAXIMUM_OBSERVABLE_RELATIVE_ERROR = 0.10
OBSERVABLE_ACTIVITY_FLOOR = 1.0e-6
TRANSIENT_SAMPLE_SECONDS = (0.01, 0.025, 0.05, 0.125)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate WP10c8b evidence and certified checkpoints.",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_wp10c8b() -> tuple[dict, str]:
    if not WP10C8B_OUTPUT.exists():
        raise RuntimeError("WP10c8c requires canonical WP10c8b evidence")
    evidence = json.loads(WP10C8B_OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / str(evidence.get("artifacts", {}).get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c8b"
        and evidence.get("decision")
        == "wp10c8b_stress_time_spatial_stop"
        and evidence.get("latest_spatially_certified_time_seconds")
        == CERTIFIED_TIME_SECONDS
        and evidence.get("next_authorization")
        == "wp10c8c_certified_state_operator_only_closure_audit"
        and evidence.get("gates", {}).get(
            "certified_operator_audit_authorized",
            False,
        )
        and not evidence.get("gates", {}).get("wp10c8b_passed", True)
        and arrays.exists()
        and _sha256(arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c8b did not authorize WP10c8c")
    return evidence, _sha256(WP10C8B_OUTPUT)


def _load_certified_states(
    evidence: dict,
) -> tuple[dict[int, dict], dict[int, dict[str, np.ndarray]], dict]:
    (
        spectral,
        spectral_sha256,
        reference,
        reference_sha256,
    ) = wp10c8b._validate_authorization()
    initial, wp10c7k_evidence, wp10c7k_sha256 = (
        wp10c8b._initial_bundles(reference)
    )
    vectors = {}
    provenance = {}
    for n_cells in RESOLUTIONS:
        parent, parent_entry = wp10c8b._parent_restart(
            initial[n_cells],
            n_cells,
            "production",
            wp10c7k_evidence,
            wp10c7k_sha256,
            reference,
        )
        del parent
        vectors[n_cells] = {}
        provenance[str(n_cells)] = {}
        for label in (SECANT_LABEL, CERTIFIED_LABEL):
            restart = wp10c8b._load_snapshot(
                initial[n_cells],
                "production",
                label,
                parent_entry,
                spectral_sha256,
                reference_sha256,
            )
            path = wp10c8b._checkpoint_path(
                n_cells,
                "production",
                label,
            )
            vectors[n_cells][label] = np.asarray(
                restart.state_vector,
                dtype=float,
            )
            provenance[str(n_cells)][label] = {
                "path": _relative(path),
                "sha256": _sha256(path),
                "state_vector_sha256": _array_sha256(
                    restart.state_vector
                ),
                "elapsed_time_seconds": restart.elapsed_time,
            }
    return initial, vectors, {
        "wp10c8a_decision": spectral["decision"],
        "wp10c7n_decision": reference["decision"],
        "states": provenance,
    }


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    normalized = np.asarray(weights, dtype=float)
    normalized /= float(np.sum(normalized))
    return float(np.sqrt(np.sum(normalized * np.asarray(values) ** 2)))


def _weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ordering = np.argsort(values)
    ordered_values = values[ordering]
    cumulative = np.cumsum(weights[ordering])
    cumulative /= float(cumulative[-1])
    index = min(
        int(np.searchsorted(cumulative, quantile, side="left")),
        ordered_values.size - 1,
    )
    return float(ordered_values[index])


def _physical_component_metrics(
    components: tuple[int, ...],
    mask: np.ndarray,
    measures: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> dict:
    component_values = {}
    if 1 in components:
        component_values["radial_momentum_stationary_balance"] = (
            diagnostics["radial_momentum_stationary_balance"][mask]
        )
    if 4 in components:
        component_values["stress_target_departure"] = diagnostics[
            "stress_target_relative_departure"
        ][mask]
        component_values["stress_stationary_balance"] = diagnostics[
            "stress_stationary_balance"
        ][mask]
    rows = {}
    for name, values in component_values.items():
        rows[name] = {
            "maximum": float(np.max(values)),
            "weighted_rms": _weighted_rms(values, measures[mask]),
            "weighted_95th_percentile": _weighted_percentile(
                values,
                measures[mask],
                0.95,
            ),
        }
    weighted_rms_passed = all(
        row["weighted_rms"] <= MAXIMUM_PHYSICAL_WEIGHTED_RMS
        for row in rows.values()
    )
    percentile_passed = all(
        row["weighted_95th_percentile"]
        <= MAXIMUM_PHYSICAL_95TH_PERCENTILE
        for row in rows.values()
    )
    return {
        "components": list(components),
        "metrics": rows,
        "weighted_rms_passed": weighted_rms_passed,
        "weighted_95th_percentile_passed": percentile_passed,
        "passed": bool(weighted_rms_passed and percentile_passed),
    }


def _component_indices(
    n_cells: int,
    components: tuple[int, ...],
    cell_mask: np.ndarray,
) -> np.ndarray:
    cells = np.flatnonzero(cell_mask)
    return np.sort(
        np.concatenate(
            [5 * cells + component for component in components]
        )
    )


def _schur_closure(
    dynamic: np.ndarray,
    fast_indices: np.ndarray,
) -> dict:
    dynamic = np.asarray(dynamic, dtype=float)
    fast_indices = np.asarray(fast_indices, dtype=int)
    retained_indices = np.setdiff1d(
        np.arange(dynamic.shape[0]),
        fast_indices,
        assume_unique=True,
    )
    a_ff = dynamic[np.ix_(fast_indices, fast_indices)]
    a_fs = dynamic[np.ix_(fast_indices, retained_indices)]
    a_sf = dynamic[np.ix_(retained_indices, fast_indices)]
    a_ss = dynamic[np.ix_(retained_indices, retained_indices)]
    manifold = -solve(
        a_ff,
        a_fs,
        assume_a="gen",
        check_finite=True,
    )
    solve_residual = a_ff @ manifold + a_fs
    solve_scale = max(
        float(np.linalg.norm(a_ff, ord=2) * np.linalg.norm(manifold, ord=2)),
        float(np.linalg.norm(a_fs, ord=2)),
        np.finfo(float).tiny,
    )
    effective = a_ss + a_sf @ manifold
    return {
        "fast_indices": fast_indices,
        "retained_indices": retained_indices,
        "fast_operator": a_ff,
        "fast_from_retained": a_fs,
        "retained_from_fast": a_sf,
        "retained_operator": a_ss,
        "manifold": manifold,
        "effective_operator": effective,
        "solve_relative_defect": float(
            np.linalg.norm(solve_residual, ord=2) / solve_scale
        ),
    }


def _lift(
    retained_values: np.ndarray,
    closure: dict,
) -> np.ndarray:
    values = np.empty(
        closure["fast_indices"].size + closure["retained_indices"].size,
        dtype=float,
    )
    values[closure["retained_indices"]] = retained_values
    values[closure["fast_indices"]] = (
        closure["manifold"] @ retained_values
    )
    return values


def _smooth_window(
    radius_rg: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    selected = (radius_rg >= lower) & (radius_rg < upper)
    values = np.zeros_like(radius_rg, dtype=float)
    if not np.any(selected):
        return values
    coordinates = (
        radius_rg[selected] - float(np.min(radius_rg[selected]))
    )
    span = float(np.max(coordinates))
    if span <= np.finfo(float).tiny:
        values[selected] = 1.0
    else:
        values[selected] = np.sin(np.pi * coordinates / span) ** 2
    return values


def _normalize_direction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(float).tiny:
        raise RuntimeError("WP10c8c perturbation direction is zero")
    return values / norm


def _perturbation_directions(
    initial: dict,
    secant_vector: np.ndarray,
    certified_vector: np.ndarray,
    primitive_scales: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n_cells = initial["state"].n_cells
    before = unpack_causal_five_field_state(secant_vector, n_cells)
    after = unpack_causal_five_field_state(certified_vector, n_cells)
    radius_rg = diagnostics["radius_rg"]

    trajectory = (
        np.asarray(after.primitives - before.primitives, dtype=float).ravel()
        / primitive_scales
    )
    thermal = np.zeros((n_cells, 5), dtype=float)
    thermal[:, 3] = 1.0e-2 * _smooth_window(radius_rg, 6.0, 60.0)
    density = np.zeros((n_cells, 5), dtype=float)
    density[:, 0] = 1.0e-2 * _smooth_window(radius_rg, 6.0, 60.0)
    source = np.zeros((n_cells, 5), dtype=float)
    source[:, 0] = 1.0e-2 * _smooth_window(radius_rg, 200.0, 280.0)
    source[:, 2] = 5.0e-3 * _smooth_window(radius_rg, 200.0, 280.0)
    stress = np.zeros((n_cells, 5), dtype=float)
    stress[:, 4] = (
        diagnostics["target_specific_stress"]
        - diagnostics["specific_stress"]
    )
    physical = {
        "trajectory_secant_0p10_to_0p125": trajectory,
        "thermal_6_to_60rg": thermal.ravel() / primitive_scales,
        "surface_density_6_to_60rg": density.ravel() / primitive_scales,
        "source_band_loading_200_to_280rg": (
            source.ravel() / primitive_scales
        ),
        "stress_target_adjustment": stress.ravel() / primitive_scales,
    }
    return {
        name: _normalize_direction(values)
        for name, values in physical.items()
    }


def _observable_value(
    operator: np.ndarray,
    vector: np.ndarray,
) -> float:
    response = np.asarray(operator @ vector)
    return float(np.max(np.abs(response)))


def _observable_error_rows(
    operators: dict,
    full: np.ndarray,
    reduced: np.ndarray,
) -> dict:
    rows = {}
    for name, operator in operators.items():
        if name == "baseline":
            continue
        operator = np.asarray(operator, dtype=float)
        full_value = _observable_value(operator, full)
        reduced_value = _observable_value(operator, reduced)
        difference = _observable_value(operator, reduced - full)
        activity_scale = (
            OBSERVABLE_ACTIVITY_FLOOR
            * float(np.linalg.norm(operator))
            * max(float(np.linalg.norm(full)), np.finfo(float).tiny)
        )
        active = bool(full_value >= activity_scale)
        rows[name] = {
            "full_magnitude": full_value,
            "reduced_magnitude": reduced_value,
            "absolute_difference": difference,
            "activity_floor": activity_scale,
            "active": active,
            "relative_error": (
                float(difference / full_value) if active else None
            ),
        }
    active_errors = [
        row["relative_error"]
        for row in rows.values()
        if row["active"]
    ]
    return {
        "observables": rows,
        "active_observable_count": len(active_errors),
        "maximum_active_relative_error": (
            max(active_errors) if active_errors else 0.0
        ),
    }


def _direction_audit(
    dynamic: np.ndarray,
    closure: dict,
    operators: dict,
    direction: np.ndarray,
) -> dict:
    retained = direction[closure["retained_indices"]]
    lifted = _lift(retained, closure)
    full_tangent = dynamic @ direction
    full_manifold_tangent = dynamic @ lifted
    reduced_retained_tangent = (
        closure["effective_operator"] @ retained
    )
    lifted_tangent = _lift(reduced_retained_tangent, closure)
    projection_error = float(
        np.linalg.norm(lifted - direction)
        / max(float(np.linalg.norm(direction)), np.finfo(float).tiny)
    )
    tangent_error = float(
        np.linalg.norm(lifted_tangent - full_tangent)
        / max(float(np.linalg.norm(full_tangent)), np.finfo(float).tiny)
    )
    invariance_defect = float(
        np.linalg.norm(lifted_tangent - full_manifold_tangent)
        / max(
            float(np.linalg.norm(full_manifold_tangent)),
            np.finfo(float).tiny,
        )
    )
    projection_observables = _observable_error_rows(
        operators,
        direction,
        lifted,
    )
    tangent_observables = _observable_error_rows(
        operators,
        full_tangent,
        lifted_tangent,
    )
    passed = bool(
        projection_error <= MAXIMUM_STATE_PROJECTION_RELATIVE_ERROR
        and tangent_error <= MAXIMUM_STATE_TANGENT_RELATIVE_ERROR
        and invariance_defect
        <= MAXIMUM_MANIFOLD_INVARIANCE_RELATIVE_DEFECT
        and projection_observables["maximum_active_relative_error"]
        <= MAXIMUM_OBSERVABLE_RELATIVE_ERROR
        and tangent_observables["maximum_active_relative_error"]
        <= MAXIMUM_OBSERVABLE_RELATIVE_ERROR
    )
    return {
        "state_projection_relative_error": projection_error,
        "state_tangent_relative_error": tangent_error,
        "manifold_invariance_relative_defect": invariance_defect,
        "projection_observables": projection_observables,
        "tangent_observables": tangent_observables,
        "passed": passed,
    }


def _spectral_audit(closure: dict) -> dict:
    fast = closure["fast_operator"]
    effective = closure["effective_operator"]
    fast_values, fast_vectors = np.linalg.eig(fast)
    effective_values = eigvals(effective)
    stable_fast = bool(
        np.max(np.real(fast_values)) < -STABILITY_TOLERANCE_PER_S
    )
    stable_effective = bool(
        np.max(np.real(effective_values)) < STABILITY_TOLERANCE_PER_S
    )
    fast_damping = np.asarray(
        [
            1.0 / abs(float(np.real(value)))
            for value in fast_values
            if float(np.real(value)) < -STABILITY_TOLERANCE_PER_S
        ],
        dtype=float,
    )
    retained_damping = np.asarray(
        [
            1.0 / abs(float(np.real(value)))
            for value in effective_values
            if float(np.real(value)) < -STABILITY_TOLERANCE_PER_S
        ],
        dtype=float,
    )
    slowest_fast = (
        float(np.max(fast_damping)) if fast_damping.size else np.inf
    )
    fastest_retained = (
        float(np.min(retained_damping))
        if retained_damping.size
        else np.inf
    )
    gap = fastest_retained / slowest_fast
    transient_rows = []
    maximum_amplification = 1.0
    for elapsed in TRANSIENT_SAMPLE_SECONDS:
        propagator = expm(float(elapsed) * fast)
        amplification = float(np.linalg.norm(propagator, ord=2))
        maximum_amplification = max(maximum_amplification, amplification)
        transient_rows.append(
            {
                "elapsed_time_seconds": elapsed,
                "two_norm_amplification": amplification,
            }
        )
    eigenvector_condition = float(np.linalg.cond(fast_vectors))
    numerical_abscissa = float(
        np.max(eigvalsh(0.5 * (fast + fast.T)))
    )
    passed = bool(
        stable_fast
        and stable_effective
        and gap >= MINIMUM_FAST_TO_RETAINED_GAP
        and eigenvector_condition <= MAXIMUM_FAST_EIGENVECTOR_CONDITION
        and maximum_amplification <= MAXIMUM_FAST_TRANSIENT_AMPLIFICATION
        and closure["solve_relative_defect"]
        <= MAXIMUM_SCHUR_SOLVE_RELATIVE_DEFECT
    )
    return {
        "fast_dimension": int(fast.shape[0]),
        "retained_dimension": int(effective.shape[0]),
        "fast_stable": stable_fast,
        "effective_stable": stable_effective,
        "fast_maximum_real_eigenvalue_per_s": float(
            np.max(np.real(fast_values))
        ),
        "effective_maximum_real_eigenvalue_per_s": float(
            np.max(np.real(effective_values))
        ),
        "fast_minimum_damping_time_seconds": (
            float(np.min(fast_damping)) if fast_damping.size else None
        ),
        "fast_maximum_damping_time_seconds": (
            slowest_fast if np.isfinite(slowest_fast) else None
        ),
        "effective_fastest_damping_time_seconds": (
            fastest_retained if np.isfinite(fastest_retained) else None
        ),
        "fast_to_retained_timescale_gap": (
            gap if np.isfinite(gap) else None
        ),
        "fast_numerical_abscissa_per_s": numerical_abscissa,
        "fast_eigenvector_condition_estimate": eigenvector_condition,
        "maximum_sampled_transient_amplification": maximum_amplification,
        "transient_amplification_samples": transient_rows,
        "schur_solve_relative_defect": closure[
            "solve_relative_defect"
        ],
        "passed": passed,
    }


def _candidate_audit(
    dynamic: np.ndarray,
    operators: dict,
    directions: dict[str, np.ndarray],
    radius_rg: np.ndarray,
    measures: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    region: tuple[str, float, float],
    chart: tuple[str, tuple[int, ...]],
) -> dict:
    region_name, lower, upper = region
    chart_name, components = chart
    cell_mask = (radius_rg >= lower) & (radius_rg < upper)
    if not np.any(cell_mask):
        raise RuntimeError(f"WP10c8c region {region_name} has no cells")
    fast_indices = _component_indices(
        radius_rg.size,
        components,
        cell_mask,
    )
    closure = _schur_closure(dynamic, fast_indices)
    physical = _physical_component_metrics(
        components,
        cell_mask,
        measures,
        diagnostics,
    )
    spectral = _spectral_audit(closure)
    direction_rows = {
        name: _direction_audit(
            dynamic,
            closure,
            operators,
            direction,
        )
        for name, direction in directions.items()
    }
    response_passed = all(
        row["passed"] for row in direction_rows.values()
    )
    return {
        "region": region_name,
        "radius_bounds_rg": [
            lower,
            upper if np.isfinite(upper) else None,
        ],
        "cell_count": int(np.count_nonzero(cell_mask)),
        "elimination_chart": chart_name,
        "components": list(components),
        "physical_slaving": physical,
        "spectral_and_schur": spectral,
        "directional_response": direction_rows,
        "response_passed": response_passed,
        "passed": bool(
            physical["passed"]
            and spectral["passed"]
            and response_passed
        ),
    }


def _resolution_audit(
    initial: dict,
    secant_vector: np.ndarray,
    certified_vector: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    started = time.perf_counter()
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        certified_vector,
    )
    descriptor_wall = time.perf_counter() - started
    stationary = np.asarray(
        reduced["stationary_reduced_scaled_jacobian"],
        dtype=float,
    )
    descriptor = np.asarray(
        reduced["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    dynamic = solve(
        descriptor,
        -stationary,
        assume_a="gen",
        check_finite=True,
    )
    dynamic_defect = float(
        np.linalg.norm(descriptor @ dynamic + stationary, ord=2)
        / max(float(np.linalg.norm(stationary, ord=2)), np.finfo(float).tiny)
    )
    operators = wp10c8a._observable_operators(
        initial,
        certified_vector,
        reduced,
    )
    _, diagnostics = wp10c8b._off_manifold_diagnostics(
        initial,
        certified_vector,
    )
    primitive_scales = np.asarray(
        reduced["primitive_column_scales"],
        dtype=float,
    )
    directions = _perturbation_directions(
        initial,
        secant_vector,
        certified_vector,
        primitive_scales,
        diagnostics,
    )
    radius_rg = np.asarray(diagnostics["radius_rg"], dtype=float)
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    candidates = {}
    for region in REGIONS_RG:
        for chart in ELIMINATION_CHARTS:
            key = f"{region[0]}__{chart[0]}"
            candidate_started = time.perf_counter()
            candidates[key] = _candidate_audit(
                dynamic,
                operators,
                directions,
                radius_rg,
                measures,
                diagnostics,
                region,
                chart,
            )
            candidates[key]["wall_seconds"] = (
                time.perf_counter() - candidate_started
            )
    passed = [name for name, row in candidates.items() if row["passed"]]
    arrays = {
        "radius_rg": radius_rg,
        "primitive_column_scales": primitive_scales,
        **{
            f"direction_{name}": values
            for name, values in directions.items()
        },
        **diagnostics,
    }
    return {
        "n_cells": n_cells,
        "descriptor_dimensions": reduced["dimensions"],
        "descriptor_rank": int(np.linalg.matrix_rank(descriptor)),
        "descriptor_condition_estimate": float(np.linalg.cond(descriptor)),
        "descriptor_dynamic_solve_relative_defect": dynamic_defect,
        "maximum_scaled_descriptor_algebraic_row": reduced[
            "maximum_scaled_descriptor_algebraic_row"
        ],
        "descriptor_wall_seconds": descriptor_wall,
        "candidate_count": len(candidates),
        "authorized_candidate_count": len(passed),
        "authorized_candidates": passed,
        "candidates": candidates,
    }, arrays


def _cross_mesh_summary(results: dict[str, dict]) -> dict:
    candidate_names = list(results["64"]["candidates"])
    rows = {}
    for name in candidate_names:
        coarse = results["64"]["candidates"][name]
        fine = results["128"]["candidates"][name]
        rows[name] = {
            "n64_passed": coarse["passed"],
            "n128_passed": fine["passed"],
            "both_passed": bool(coarse["passed"] and fine["passed"]),
            "n64_physical_passed": coarse["physical_slaving"]["passed"],
            "n128_physical_passed": fine["physical_slaving"]["passed"],
            "n64_spectral_passed": coarse["spectral_and_schur"][
                "passed"
            ],
            "n128_spectral_passed": fine["spectral_and_schur"][
                "passed"
            ],
            "n64_response_passed": coarse["response_passed"],
            "n128_response_passed": fine["response_passed"],
        }
    authorized = [
        name for name, row in rows.items() if row["both_passed"]
    ]
    return {
        "candidate_count": len(rows),
        "authorized_candidate_count": len(authorized),
        "authorized_candidates": authorized,
        "candidates": rows,
    }


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    evidence, evidence_sha256 = _validate_wp10c8b()
    initial, vectors, provenance = _load_certified_states(evidence)
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8c",
                    "preflight_passed": True,
                    "wp10c8b_evidence_sha256": evidence_sha256,
                    "certified_time_seconds": CERTIFIED_TIME_SECONDS,
                    "states": provenance["states"],
                },
                sort_keys=True,
            )
        )
        return

    results = {}
    arrays = {}
    for n_cells in RESOLUTIONS:
        result, resolution_arrays = _resolution_audit(
            initial[n_cells],
            vectors[n_cells][SECANT_LABEL],
            vectors[n_cells][CERTIFIED_LABEL],
        )
        results[str(n_cells)] = result
        for name, values in resolution_arrays.items():
            arrays[f"n{n_cells}_{name}"] = values
    cross_mesh = _cross_mesh_summary(results)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    nonlinear_authorized = bool(
        cross_mesh["authorized_candidate_count"] > 0
    )
    payload = {
        "schema_version": 1,
        "work_package": "WP10c8c",
        "generated_at_utc": "2026-07-19T00:00:00Z",
        "base_commit": BASE_COMMIT,
        "scope": (
            "operator-only region-selective Schur closure audit at the "
            "spatially certified 0.125 s N64/N128 causal states"
        ),
        "wp10c8b_evidence": {
            "path": _relative(WP10C8B_OUTPUT),
            "sha256": evidence_sha256,
            "decision": evidence["decision"],
            "latest_spatially_certified_time_seconds": evidence[
                "latest_spatially_certified_time_seconds"
            ],
        },
        "selected_state_provenance": provenance,
        "contract": {
            "regions_rg": [
                {
                    "name": name,
                    "lower": lower,
                    "upper": upper if np.isfinite(upper) else None,
                }
                for name, lower, upper in REGIONS_RG
            ],
            "elimination_charts": {
                name: list(components)
                for name, components in ELIMINATION_CHARTS
            },
            "maximum_physical_weighted_rms": (
                MAXIMUM_PHYSICAL_WEIGHTED_RMS
            ),
            "maximum_physical_95th_percentile": (
                MAXIMUM_PHYSICAL_95TH_PERCENTILE
            ),
            "minimum_fast_to_retained_gap": (
                MINIMUM_FAST_TO_RETAINED_GAP
            ),
            "maximum_fast_eigenvector_condition": (
                MAXIMUM_FAST_EIGENVECTOR_CONDITION
            ),
            "maximum_fast_transient_amplification": (
                MAXIMUM_FAST_TRANSIENT_AMPLIFICATION
            ),
            "maximum_schur_solve_relative_defect": (
                MAXIMUM_SCHUR_SOLVE_RELATIVE_DEFECT
            ),
            "maximum_state_projection_relative_error": (
                MAXIMUM_STATE_PROJECTION_RELATIVE_ERROR
            ),
            "maximum_state_tangent_relative_error": (
                MAXIMUM_STATE_TANGENT_RELATIVE_ERROR
            ),
            "maximum_manifold_invariance_relative_defect": (
                MAXIMUM_MANIFOLD_INVARIANCE_RELATIVE_DEFECT
            ),
            "maximum_observable_relative_error": (
                MAXIMUM_OBSERVABLE_RELATIVE_ERROR
            ),
            "transient_sample_seconds": list(
                TRANSIENT_SAMPLE_SECONDS
            ),
        },
        "resolutions": results,
        "cross_mesh_contract": cross_mesh,
        "gates": {
            "both_descriptor_ranks_full": all(
                results[str(n_cells)]["descriptor_rank"]
                == 5 * n_cells
                for n_cells in RESOLUTIONS
            ),
            "both_descriptor_dynamic_solves_passed": all(
                results[str(n_cells)][
                    "descriptor_dynamic_solve_relative_defect"
                ]
                <= 1.0e-8
                for n_cells in RESOLUTIONS
            ),
            "region_selective_candidate_authorized": (
                nonlinear_authorized
            ),
            "nonlinear_reduced_trajectory_authorized": (
                nonlinear_authorized
            ),
            "wp10c8c_passed": nonlinear_authorized,
        },
        "decision": (
            "wp10c8c_region_selective_reduction_authorized"
            if nonlinear_authorized
            else "wp10c8c_region_selective_reduction_not_authorized"
        ),
        "next_authorization": (
            "implement_one_certified_region_selective_nonlinear_closure"
            if nonlinear_authorized
            else "retain_full_causal_dae_and_design_alternative_secular_coordinates"
        ),
        "hard_stops": [
            "no global algebraic P_R/chi elimination",
            "no nonlinear reduced trajectory without a two-mesh candidate",
            "no loading-time macrosteps",
            "no tide or wind",
            "no hot-state or cycle claim",
        ],
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "authorized_candidate_count": cross_mesh[
                    "authorized_candidate_count"
                ],
                "authorized_candidates": cross_mesh[
                    "authorized_candidates"
                ],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
