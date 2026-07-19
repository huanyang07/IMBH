"""Compute selected-state finite descriptor spectra for WP10c8a."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eig, eigvals, eigvalsh, solve
from scipy.optimize import linear_sum_assignment

import run_causal_characteristic_extension_wp10c7l as wp10c7l
import run_causal_n128_reference_wp10c7n as wp10c7n
import run_causal_spatial_balance_adaptive_wp10c7k as wp10c7k
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    audit_causal_five_field_state_gates,
    causal_diffusion_cooling_rate,
    causal_five_field_cell_states,
    causal_five_field_observable_snapshot,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_regression_seed_parameters,
    make_causal_five_field_regression_context,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "183afa13b21762d7fef49addc297172013981e8b"
WP10C7N_OUTPUT = (
    ROOT / "outputs/tables/causal_n128_reference_wp10c7n.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_slow_mode_audit_wp10c8a.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_slow_mode_audit_wp10c8a_arrays.npz"
)
RESOLUTIONS = (64, 128)
CHECKPOINTS = (
    ("t_0", 0.0),
    ("t_0p0375", 3.75e-2),
    ("t_0p05", 5.0e-2),
)
FAST_COMPONENTS = (1, 4)
SLOW_COMPONENTS = (0, 2, 3)
STABILITY_TOLERANCE_PER_S = 1.0e-8
MINIMUM_PROTOTYPE_FAST_SLOW_GAP = 3.0
MAXIMUM_EIGENPAIR_RELATIVE_DEFECT = 2.0e-7
MAXIMUM_LOW_MODE_MEDIAN_RELATIVE_MISMATCH = 0.25
LOW_MODE_COUNT = 32
OBSERVABLE_RELATIVE_PROJECTION_FLOOR = 1.0e-3
COOLING_INNER_CUTOFF_RG = 6.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate WP10c7n evidence and selected checkpoints.",
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


def _validate_wp10c7n() -> tuple[dict, str]:
    if not WP10C7N_OUTPUT.exists():
        raise RuntimeError("WP10c8a requires canonical WP10c7n evidence")
    evidence = json.loads(WP10C7N_OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / str(evidence.get("artifacts", {}).get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c7n"
        and evidence.get("decision")
        == "wp10c7n_n128_0p05_reference_certified"
        and evidence.get("next_authorization")
        == "selected_state_slow_mode_audit"
        and evidence.get("gates", {}).get("wp10c7n_passed", False)
        and evidence.get("primary_log_h_over_r_contract", {}).get(
            "all_common_times_passed",
            False,
        )
        and arrays.exists()
        and _sha256(arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c7n did not authorize WP10c8a")
    return evidence, _sha256(WP10C7N_OUTPUT)


def _fresh_initial(n_cells: int) -> dict:
    baseline = make_causal_five_field_regression_context(
        16,
        spatial_reconstruction="plm_smooth",
    )
    seed_parameters = causal_five_field_regression_seed_parameters(
        baseline
    )
    return wp10c7k._initial_bundle(
        n_cells,
        seed_parameters,
    )


def _load_states(evidence: dict) -> tuple[dict, dict]:
    wp10c7k_evidence, wp10c7k_sha256 = wp10c7l._validate_wp10c7k()
    initial64 = wp10c7l._initial_bundles(wp10c7k_evidence)[64]
    initial128 = _fresh_initial(128)
    if (
        initial128["vector_sha256"]
        != evidence["initialization"]["n128_initial_state_sha256"]
    ):
        raise RuntimeError("WP10c8a N128 initial state differs")
    states = {
        64: {"t_0": np.asarray(initial64["vector"], dtype=float)},
        128: {"t_0": np.asarray(initial128["vector"], dtype=float)},
    }
    provenance = {
        64: {
            "t_0": {
                "state_vector_sha256": initial64["vector_sha256"],
                "source": "fresh deterministic initial state",
            }
        },
        128: {
            "t_0": {
                "state_vector_sha256": initial128["vector_sha256"],
                "source": "fresh deterministic initial state",
            }
        },
    }
    parent64 = wp10c7l._parent_checkpoint_entry(wp10c7k_evidence, 64)
    authorization_sha256 = evidence["wp10c7m_authorization"]["sha256"]
    for label in ("t_0p0375", "t_0p05"):
        restart64 = wp10c7l._load_snapshot(
            initial64,
            wp10c7k_sha256,
            parent64,
            "production",
            label,
        )
        path64 = wp10c7l._checkpoint_path(64, "production", label)
        restart128 = wp10c7n._load_snapshot(
            initial128,
            authorization_sha256,
            "production",
            label,
        )
        path128 = wp10c7n._checkpoint_path("production", label)
        states[64][label] = np.asarray(
            restart64.state_vector,
            dtype=float,
        )
        states[128][label] = np.asarray(
            restart128.state_vector,
            dtype=float,
        )
        provenance[64][label] = {
            "path": _relative(path64),
            "sha256": _sha256(path64),
            "state_vector_sha256": _array_sha256(
                restart64.state_vector
            ),
        }
        provenance[128][label] = {
            "path": _relative(path128),
            "sha256": _sha256(path128),
            "state_vector_sha256": _array_sha256(
                restart128.state_vector
            ),
        }
    return (
        {
            64: initial64,
            128: initial128,
        },
        {
            "vectors": states,
            "provenance": provenance,
        },
    )


def _observable_operators(
    initial: dict,
    vector: np.ndarray,
    reduced: dict,
) -> dict:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    n_reduced = 5 * n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    cells = causal_five_field_cell_states(context, vector)
    primitive_scale = np.asarray(
        reduced["primitive_column_scales"],
        dtype=float,
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    radius = np.asarray(context.grid.centers, dtype=float)
    cutoff = COOLING_INNER_CUTOFF_RG * context.grid.gravitational_radius

    log_h = np.zeros((n_cells, n_reduced), dtype=float)
    cooling = np.empty(n_cells, dtype=float)
    for index, cell in enumerate(cells):
        derivatives = context.vertical_frequency.eos(
            float(radius[index])
        ).derivatives(
            cell.thermodynamics.surface_density,
            cell.thermodynamics.temperature,
        )
        sigma_column = 5 * index
        temperature_column = sigma_column + 3
        log_h[index, sigma_column] = (
            derivatives.height_log_surface_density
            * primitive_scale[sigma_column]
        )
        log_h[index, temperature_column] = (
            derivatives.height_log_temperature
            * primitive_scale[temperature_column]
        )
        cooling[index] = causal_diffusion_cooling_rate(
            cell.thermodynamics
        )[0]

    weighted = measures * cooling
    exterior = radius >= cutoff
    cooling_gradient = np.zeros(n_reduced, dtype=float)
    exterior_gradient = np.zeros(n_reduced, dtype=float)
    for index in range(n_cells):
        sigma_column = 5 * index
        temperature_column = sigma_column + 3
        cooling_gradient[sigma_column] = (
            -weighted[index] * primitive_scale[sigma_column]
        )
        cooling_gradient[temperature_column] = (
            4.0 * weighted[index] * primitive_scale[temperature_column]
        )
        if exterior[index]:
            exterior_gradient[sigma_column] = (
                -weighted[index] * primitive_scale[sigma_column]
            )
            exterior_gradient[temperature_column] = (
                4.0 * weighted[index]
                * primitive_scale[temperature_column]
            )
    cooling_gradient /= float(np.sum(weighted))
    exterior_gradient /= float(np.sum(weighted[exterior]))

    response_scaled = np.asarray(
        reduced["algebraic_response_scaled"],
        dtype=float,
    )
    algebraic_scale = np.asarray(
        reduced["algebraic_column_scales"],
        dtype=float,
    )
    response_physical = algebraic_scale[:, None] * response_scaled
    conserved_response = response_physical[:n_reduced].reshape(
        n_cells,
        5,
        n_reduced,
    )
    face_response = response_physical[n_reduced:].reshape(
        n_cells + 1,
        5,
        n_reduced,
    )
    integrated = np.sum(
        measures[:, None] * state.conserved,
        axis=0,
    )
    integrated_response = np.sum(
        measures[:, None, None] * conserved_response,
        axis=0,
    )
    face_rates = C * state.weighted_face_fluxes_over_c
    inner_rate = float(-face_rates[0, 0])
    inner_gradient = -C * face_response[0, 0]
    snapshot = causal_five_field_observable_snapshot(
        context,
        vector,
        cooling_inner_cutoff=cutoff,
    )
    return {
        "log_h_over_r_profile": log_h,
        "cooling_relative": cooling_gradient,
        "cooling_outside_6rg_relative": exterior_gradient,
        "inner_accretion_relative": (
            inner_gradient / max(abs(inner_rate), np.finfo(float).tiny)
        ),
        "integrated_mass_relative": (
            integrated_response[0]
            / max(abs(integrated[0]), np.finfo(float).tiny)
        ),
        "integrated_angular_momentum_relative": (
            integrated_response[2]
            / max(abs(integrated[2]), np.finfo(float).tiny)
        ),
        "integrated_killing_energy_relative": (
            integrated_response[3]
            / max(abs(integrated[3]), np.finfo(float).tiny)
        ),
        "baseline": {
            "cooling_power_proxy_erg_s": (
                snapshot.cooling_power_proxy_erg_s
            ),
            "cooling_power_proxy_outside_cutoff_erg_s": (
                snapshot.cooling_power_proxy_outside_cutoff_erg_s
            ),
            "inner_accretion_rate_g_s": inner_rate,
            "integrated_conserved": integrated,
        },
    }


def _component_fractions(
    vector: np.ndarray,
    measures: np.ndarray,
) -> list[float]:
    values = np.asarray(vector).reshape(measures.size, 5)
    normalized_measures = measures / float(np.sum(measures))
    powers = np.sum(
        normalized_measures[:, None] * np.abs(values) ** 2,
        axis=0,
    )
    total = max(float(np.sum(powers)), np.finfo(float).tiny)
    return [float(value / total) for value in powers]


def _mode_observable_projections(
    right: np.ndarray,
    operators: dict,
) -> dict:
    return {
        "maximum_log_h_over_r": float(
            np.max(
                np.abs(
                    operators["log_h_over_r_profile"] @ right
                )
            )
        ),
        "cooling_relative": float(
            abs(operators["cooling_relative"] @ right)
        ),
        "cooling_outside_6rg_relative": float(
            abs(operators["cooling_outside_6rg_relative"] @ right)
        ),
        "inner_accretion_relative": float(
            abs(operators["inner_accretion_relative"] @ right)
        ),
        "integrated_mass_relative": float(
            abs(operators["integrated_mass_relative"] @ right)
        ),
        "integrated_angular_momentum_relative": float(
            abs(
                operators["integrated_angular_momentum_relative"]
                @ right
            )
        ),
        "integrated_killing_energy_relative": float(
            abs(
                operators["integrated_killing_energy_relative"]
                @ right
            )
        ),
    }


def _damping_time(value: complex) -> float | None:
    real = float(np.real(value))
    if abs(real) <= STABILITY_TOLERANCE_PER_S:
        return None
    return float(1.0 / abs(real))


def _oscillation_period(value: complex) -> float | None:
    imaginary = float(abs(np.imag(value)))
    if imaginary <= STABILITY_TOLERANCE_PER_S:
        return None
    return float(2.0 * np.pi / imaginary)


def _spectrum(
    initial: dict,
    vector: np.ndarray,
) -> tuple[dict, dict]:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    started = time.perf_counter()
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
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
    singular = np.linalg.svd(descriptor, compute_uv=False)
    threshold = max(
        1.0e-11 * float(singular[0]),
        np.finfo(float).eps * descriptor.shape[0] * float(singular[0]),
    )
    rank = int(np.count_nonzero(singular > threshold))
    if rank != descriptor.shape[0]:
        raise RuntimeError(f"N{n_cells} reduced descriptor is singular")

    eig_started = time.perf_counter()
    eigenvalues, left_vectors, right_vectors = eig(
        -stationary,
        descriptor,
        left=True,
        right=True,
        check_finite=True,
    )
    eig_wall = time.perf_counter() - eig_started
    if np.any(~np.isfinite(eigenvalues)):
        raise RuntimeError(f"N{n_cells} finite spectrum is non-finite")
    ordering = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[ordering]
    left_vectors = left_vectors[:, ordering]
    right_vectors = right_vectors[:, ordering]
    dynamic = solve(
        descriptor,
        -stationary,
        assume_a="gen",
        check_finite=True,
    )
    operators = _observable_operators(initial, vector, reduced)
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )

    rows = []
    maximum_defect = 0.0
    for index, eigenvalue in enumerate(eigenvalues):
        right = np.asarray(right_vectors[:, index], dtype=complex)
        left = np.asarray(left_vectors[:, index], dtype=complex)
        right /= max(float(np.linalg.norm(right)), np.finfo(float).tiny)
        left /= max(float(np.linalg.norm(left)), np.finfo(float).tiny)
        residual = (
            -stationary @ right
            - eigenvalue * (descriptor @ right)
        )
        residual_scale = max(
            float(np.linalg.norm(stationary @ right)),
            abs(eigenvalue)
            * float(np.linalg.norm(descriptor @ right)),
            np.finfo(float).tiny,
        )
        defect = float(np.linalg.norm(residual) / residual_scale)
        maximum_defect = max(maximum_defect, defect)
        right_fractions = _component_fractions(right, measures)
        left_fractions = _component_fractions(left, measures)
        reshaped = right.reshape(n_cells, 5)
        radial_power = np.sum(np.abs(reshaped) ** 2, axis=1)
        peak_cell = int(np.argmax(radial_power))
        peak_component = int(
            np.argmax(np.abs(reshaped[peak_cell]))
        )
        overlap = abs(np.vdot(left, descriptor @ right))
        rows.append(
            {
                "index": index,
                "eigenvalue_real_per_s": float(np.real(eigenvalue)),
                "eigenvalue_imaginary_per_s": float(
                    np.imag(eigenvalue)
                ),
                "damping_time_seconds": _damping_time(eigenvalue),
                "oscillation_period_seconds": (
                    _oscillation_period(eigenvalue)
                ),
                "classification": (
                    "stable"
                    if np.real(eigenvalue) < -STABILITY_TOLERANCE_PER_S
                    else (
                        "unstable"
                        if np.real(eigenvalue)
                        > STABILITY_TOLERANCE_PER_S
                        else "neutral"
                    )
                ),
                "right_component_fractions": right_fractions,
                "left_component_fractions": left_fractions,
                "right_fast_fraction": float(
                    sum(right_fractions[item] for item in FAST_COMPONENTS)
                ),
                "left_fast_fraction": float(
                    sum(left_fractions[item] for item in FAST_COMPONENTS)
                ),
                "peak_radius_rg": float(radius_rg[peak_cell]),
                "peak_component": peak_component,
                "observable_projections": (
                    _mode_observable_projections(right, operators)
                ),
                "generalized_biorthogonality_reciprocal": float(
                    1.0 / max(overlap, np.finfo(float).tiny)
                ),
                "eigenpair_relative_defect": defect,
            }
        )

    component_indices = {
        component: np.arange(component, 5 * n_cells, 5)
        for component in range(5)
    }
    fast_indices = np.sort(
        np.concatenate(
            [component_indices[item] for item in FAST_COMPONENTS]
        )
    )
    slow_indices = np.sort(
        np.concatenate(
            [component_indices[item] for item in SLOW_COMPONENTS]
        )
    )
    a_ff = dynamic[np.ix_(fast_indices, fast_indices)]
    a_fs = dynamic[np.ix_(fast_indices, slow_indices)]
    a_sf = dynamic[np.ix_(slow_indices, fast_indices)]
    a_ss = dynamic[np.ix_(slow_indices, slow_indices)]
    fast_eigenvalues = eigvals(a_ff)
    fast_stable = bool(
        np.max(np.real(fast_eigenvalues))
        < -STABILITY_TOLERANCE_PER_S
    )
    fast_damping = np.asarray(
        [
            1.0 / abs(float(np.real(value)))
            for value in fast_eigenvalues
            if abs(float(np.real(value))) > STABILITY_TOLERANCE_PER_S
        ],
        dtype=float,
    )
    manifold = -solve(a_ff, a_fs, assume_a="gen")
    effective_slow = a_ss + a_sf @ manifold
    slow_eigenvalues = eigvals(effective_slow)
    stable_slow_damping = np.asarray(
        [
            1.0 / abs(float(np.real(value)))
            for value in slow_eigenvalues
            if float(np.real(value)) < -STABILITY_TOLERANCE_PER_S
        ],
        dtype=float,
    )
    slowest_fast = (
        float(np.max(fast_damping)) if fast_damping.size else np.inf
    )
    fastest_stable_slow = (
        float(np.min(stable_slow_damping))
        if stable_slow_damping.size
        else np.inf
    )
    gap = fastest_stable_slow / slowest_fast
    fast_numerical_abscissa = float(
        np.max(eigvalsh(0.5 * (a_ff + a_ff.T)))
    )

    projections = {
        name: np.asarray(
            [row["observable_projections"][name] for row in rows],
            dtype=float,
        )
        for name in rows[0]["observable_projections"]
    }
    shortest_observable = {}
    for name, values in projections.items():
        threshold_value = (
            OBSERVABLE_RELATIVE_PROJECTION_FLOOR
            * max(float(np.max(values)), np.finfo(float).tiny)
        )
        selected = [
            row
            for row, projection in zip(rows, values, strict=True)
            if projection >= threshold_value
            and row["classification"] != "neutral"
        ]
        damping = [
            row["damping_time_seconds"]
            for row in selected
            if row["damping_time_seconds"] is not None
        ]
        periods = [
            row["oscillation_period_seconds"]
            for row in selected
            if row["oscillation_period_seconds"] is not None
        ]
        shortest_observable[name] = {
            "relative_projection_threshold": threshold_value,
            "selected_mode_count": len(selected),
            "shortest_damping_time_seconds": (
                min(damping) if damping else None
            ),
            "shortest_oscillation_period_seconds": (
                min(periods) if periods else None
            ),
        }

    stable_count = sum(row["classification"] == "stable" for row in rows)
    unstable_count = sum(
        row["classification"] == "unstable" for row in rows
    )
    neutral_count = len(rows) - stable_count - unstable_count
    prototype_passed = bool(
        fast_stable
        and gap >= MINIMUM_PROTOTYPE_FAST_SLOW_GAP
        and maximum_defect <= MAXIMUM_EIGENPAIR_RELATIVE_DEFECT
        and reduced["maximum_scaled_descriptor_algebraic_row"] <= 1.0e-9
    )
    summary = {
        "n_cells": n_cells,
        "state_gates": audit_causal_five_field_state_gates(
            context,
            vector,
        ),
        "descriptor": {
            "dimensions": reduced["dimensions"],
            "rank": rank,
            "smallest_singular_value": float(singular[-1]),
            "largest_singular_value": float(singular[0]),
            "condition_estimate": float(singular[0] / singular[-1]),
            "algebraic_solve_relative_defect": reduced[
                "algebraic_solve_relative_defect"
            ],
            "maximum_scaled_algebraic_reconstruction_defect": reduced[
                "maximum_scaled_algebraic_reconstruction_defect"
            ],
            "maximum_scaled_descriptor_algebraic_row": reduced[
                "maximum_scaled_descriptor_algebraic_row"
            ],
            "stationary_nonzeros": reduced["stationary_nonzeros"],
            "descriptor_nonzeros": reduced["descriptor_nonzeros"],
        },
        "finite_spectrum": {
            "mode_count": len(rows),
            "stable_mode_count": stable_count,
            "unstable_mode_count": unstable_count,
            "neutral_mode_count": neutral_count,
            "maximum_eigenpair_relative_defect": maximum_defect,
            "right_eigenvector_condition_estimate": float(
                np.linalg.cond(right_vectors)
            ),
            "dynamic_numerical_abscissa_per_s": float(
                np.max(eigvalsh(0.5 * (dynamic + dynamic.T)))
            ),
            "maximum_real_eigenvalue_per_s": float(
                np.max(np.real(eigenvalues))
            ),
            "minimum_real_eigenvalue_per_s": float(
                np.min(np.real(eigenvalues))
            ),
            "modes": rows,
        },
        "fast_subsystem": {
            "components": list(FAST_COMPONENTS),
            "dimension": int(fast_indices.size),
            "stable": fast_stable,
            "maximum_real_eigenvalue_per_s": float(
                np.max(np.real(fast_eigenvalues))
            ),
            "minimum_damping_time_seconds": (
                float(np.min(fast_damping))
                if fast_damping.size
                else None
            ),
            "maximum_damping_time_seconds": (
                slowest_fast if np.isfinite(slowest_fast) else None
            ),
            "numerical_abscissa_per_s": fast_numerical_abscissa,
            "eigenvector_condition_estimate": float(
                np.linalg.cond(np.linalg.eig(a_ff)[1])
            ),
        },
        "quasi_steady_slow_operator": {
            "components": list(SLOW_COMPONENTS),
            "dimension": int(slow_indices.size),
            "stable_mode_count": int(
                np.count_nonzero(
                    np.real(slow_eigenvalues)
                    < -STABILITY_TOLERANCE_PER_S
                )
            ),
            "unstable_mode_count": int(
                np.count_nonzero(
                    np.real(slow_eigenvalues)
                    > STABILITY_TOLERANCE_PER_S
                )
            ),
            "maximum_real_eigenvalue_per_s": float(
                np.max(np.real(slow_eigenvalues))
            ),
            "fastest_stable_damping_time_seconds": (
                fastest_stable_slow
                if np.isfinite(fastest_stable_slow)
                else None
            ),
            "fast_to_slow_timescale_gap": (
                gap if np.isfinite(gap) else None
            ),
        },
        "observable_timescales": shortest_observable,
        "wall_seconds": {
            "descriptor_construction": descriptor_wall,
            "generalized_eigendecomposition": eig_wall,
        },
        "prototype_reduction_gate": {
            "minimum_fast_slow_gap": MINIMUM_PROTOTYPE_FAST_SLOW_GAP,
            "maximum_eigenpair_relative_defect": (
                MAXIMUM_EIGENPAIR_RELATIVE_DEFECT
            ),
            "passed": prototype_passed,
        },
    }
    arrays = {
        "eigenvalue_real_per_s": np.real(eigenvalues),
        "eigenvalue_imaginary_per_s": np.imag(eigenvalues),
        "fast_eigenvalue_real_per_s": np.real(fast_eigenvalues),
        "fast_eigenvalue_imaginary_per_s": np.imag(fast_eigenvalues),
        "slow_eigenvalue_real_per_s": np.real(slow_eigenvalues),
        "slow_eigenvalue_imaginary_per_s": np.imag(slow_eigenvalues),
        "slow_manifold_matrix": manifold,
        "right_fast_fraction": np.asarray(
            [row["right_fast_fraction"] for row in rows],
            dtype=float,
        ),
        "left_fast_fraction": np.asarray(
            [row["left_fast_fraction"] for row in rows],
            dtype=float,
        ),
    }
    for name, values in projections.items():
        arrays[f"projection_{name}"] = values
    return summary, arrays


def _match_low_modes(coarse: dict, fine: dict) -> dict:
    coarse_values = np.asarray(
        [
            complex(
                row["eigenvalue_real_per_s"],
                row["eigenvalue_imaginary_per_s"],
            )
            for row in coarse["finite_spectrum"]["modes"][:LOW_MODE_COUNT]
        ],
        dtype=complex,
    )
    fine_values = np.asarray(
        [
            complex(
                row["eigenvalue_real_per_s"],
                row["eigenvalue_imaginary_per_s"],
            )
            for row in fine["finite_spectrum"]["modes"][:LOW_MODE_COUNT]
        ],
        dtype=complex,
    )
    scale = np.maximum(
        np.maximum(
            np.abs(coarse_values)[:, None],
            np.abs(fine_values)[None, :],
        ),
        1.0e-12,
    )
    relative = np.abs(
        coarse_values[:, None] - fine_values[None, :]
    ) / scale
    coarse_indices, fine_indices = linear_sum_assignment(relative)
    matched = relative[coarse_indices, fine_indices]
    return {
        "mode_count": int(matched.size),
        "median_relative_eigenvalue_mismatch": float(
            np.median(matched)
        ),
        "maximum_relative_eigenvalue_mismatch": float(np.max(matched)),
        "coarse_indices": coarse_indices,
        "fine_indices": fine_indices,
        "relative_mismatches": matched,
        "gate": MAXIMUM_LOW_MODE_MEDIAN_RELATIVE_MISMATCH,
        "passed": bool(
            np.median(matched)
            <= MAXIMUM_LOW_MODE_MEDIAN_RELATIVE_MISMATCH
        ),
    }


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    evidence, evidence_sha256 = _validate_wp10c7n()
    initial, loaded = _load_states(evidence)
    if args.preflight:
        rows = {
            str(n_cells): {
                label: {
                    "state_vector_sha256": _array_sha256(
                        loaded["vectors"][n_cells][label]
                    ),
                    "state_gates_passed": audit_causal_five_field_state_gates(
                        initial[n_cells]["context"],
                        loaded["vectors"][n_cells][label],
                    )["passed"],
                }
                for label, _ in CHECKPOINTS
            }
            for n_cells in RESOLUTIONS
        }
        print(
            json.dumps(
                {
                    "work_package": "WP10c8a",
                    "preflight_passed": all(
                        row["state_gates_passed"]
                        for mesh in rows.values()
                        for row in mesh.values()
                    ),
                    "wp10c7n_evidence_sha256": evidence_sha256,
                    "selected_states": rows,
                },
                sort_keys=True,
            )
        )
        return

    spectra = {str(n_cells): {} for n_cells in RESOLUTIONS}
    arrays = {}
    for label, elapsed in CHECKPOINTS:
        for n_cells in RESOLUTIONS:
            summary, state_arrays = _spectrum(
                initial[n_cells],
                loaded["vectors"][n_cells][label],
            )
            spectra[str(n_cells)][label] = summary
            for name, values in state_arrays.items():
                arrays[f"n{n_cells}_{label}_{name}"] = values
            print(
                json.dumps(
                    {
                        "work_package": "WP10c8a",
                        "n_cells": n_cells,
                        "checkpoint": label,
                        "elapsed_time_seconds": elapsed,
                        "fast_stable": summary["fast_subsystem"][
                            "stable"
                        ],
                        "fast_slow_gap": summary[
                            "quasi_steady_slow_operator"
                        ]["fast_to_slow_timescale_gap"],
                        "unstable_modes": summary["finite_spectrum"][
                            "unstable_mode_count"
                        ],
                        "prototype_gate": summary[
                            "prototype_reduction_gate"
                        ]["passed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    matching = {
        label: _match_low_modes(
            spectra["64"][label],
            spectra["128"][label],
        )
        for label, _ in CHECKPOINTS
    }
    prototype_passed = all(
        spectra[str(n_cells)][label]["prototype_reduction_gate"][
            "passed"
        ]
        for n_cells in RESOLUTIONS
        for label, _ in CHECKPOINTS
    )
    matching_passed = all(row["passed"] for row in matching.values())
    passed = bool(prototype_passed and matching_passed)

    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "work_package": "WP10c8a",
        "base_commit": BASE_COMMIT,
        "scope": (
            "frozen-coefficient finite descriptor spectra at spatially "
            "certified N64/N128 t=0, 0.0375, and 0.05 s states"
        ),
        "spatial_options": dict(wp10c7l.SPATIAL_OPTIONS),
        "wp10c7n_evidence": {
            "path": _relative(WP10C7N_OUTPUT),
            "sha256": evidence_sha256,
            "decision": evidence["decision"],
        },
        "selected_state_provenance": loaded["provenance"],
        "descriptor_contract": {
            "finite_equation": "M dp/dt + K dp = 0",
            "primitive_component_order": [
                "log_surface_density",
                "radial_velocity_over_c",
                "azimuthal_velocity_over_c",
                "log_temperature",
                "specific_stress",
            ],
            "slow_components": list(SLOW_COMPONENTS),
            "fast_components": list(FAST_COMPONENTS),
            "qualification": (
                "frozen-coefficient local spectrum at a generally "
                "nonstationary state; nonlinear slaving remains subject "
                "to WP10c8c trajectory validation"
            ),
        },
        "spectra": spectra,
        "low_mode_n64_n128_matching": matching,
        "gates": {
            "all_fast_subsystems_stable_and_separated": prototype_passed,
            "low_mode_mesh_matching_passed": matching_passed,
            "wp10c8a_passed": passed,
        },
        "decision": (
            "wp10c8a_conservative_reduced_prototype_authorized"
            if passed
            else "wp10c8a_slow_manifold_not_authorized"
        ),
        "next_authorization": (
            "wp10c8b_conservative_mje_reduced_prototype"
            if passed
            else "retain_full_causal_dae_and_diagnose_modes"
        ),
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
                "prototype_gate": prototype_passed,
                "low_mode_matching_gate": matching_passed,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
