"""Run the WP10c9b family-resolved characteristic-dissipation audit.

WP10c9a isolated the remaining bounded inner-phase failure to the scalar
maximum-speed Rusanov penalty, with inward causal shear as the binding family.
This package derives the complete coordinate principal pencil, implements one
audit-only five-family matrix penalty, certifies its method/Jacobian/storage
contracts, and reruns the five packet ladders.  The mixed WP10c8z common mode
is rerun only when every pure-family gate passes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_characteristic_phase_audit_wp10c9a as wp10c9a
import run_causal_inner_embedded_patch_preflight_wp10c8z as wp10c8z
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    CausalFiveFieldAdaptiveStepConfig,
    causal_embedded_patch_flux_audit,
    causal_five_field_characteristic_dissipation,
    causal_five_field_colored_central_jacobian,
    causal_five_field_coordinate_principal_basis,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    evolve_causal_five_field_fixed_bdf2,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


BASE_COMMIT = "6764fc117ce453b4deb5c6b1c275a19c7352b4be"
WORK_PACKAGE = "WP10c9b"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_characteristic_dissipation_audit_wp10c9b.py"
)
CORE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_characteristic_dissipation.py"
)
WP10C9A_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a.json"
)
WP10C9A_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a_arrays.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c9b"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_dissipation_audit_wp10c9b.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_dissipation_audit_wp10c9b_arrays.npz"
)

MINIMUM_SMOOTH_ORDER = 1.8
MINIMUM_PACKET_PHASE_ORDER = 0.75
MINIMUM_PACKET_DAMPING_ORDER = 0.75
MINIMUM_SIGNED_COSINE = 0.90
MAXIMUM_ANALYTIC_SPEED_DEFECT = 2.5e-3
MAXIMUM_EIGENPAIR_DEFECT = 1.0e-10
MAXIMUM_BIORTHOGONALITY_DEFECT = 1.0e-10
MAXIMUM_BASIS_CONDITION = 1.0e4
MINIMUM_CROSS_FACE_CONTINUITY = 0.995
MAXIMUM_SHARED_FLUX_DEFECT = 1.0e-12
MAXIMUM_STORAGE_ACTION_DEFECT = 2.0e-5
MAXIMUM_DENSE_COLORED_DEFECT = 1.0e-10
MAXIMUM_EQUAL_SPEED_DEFECT = 1.0e-10
MAXIMUM_RESTART_DEFECT = 2.0e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _observed_order(coarse: float, fine: float) -> float | None:
    if not (
        np.isfinite(coarse)
        and np.isfinite(fine)
        and coarse > 0.0
        and fine > 0.0
    ):
        return None
    return float(np.log2(coarse / fine))


def _face_midpoint_charts(configuration: dict) -> tuple[np.ndarray, np.ndarray]:
    context = configuration["context"]
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        configuration["base_primitives"],
    )
    faces = np.arange(context.grid.edges.size, dtype=int)
    midpoints = 0.5 * (
        reconstruction.left_face_charts
        + reconstruction.right_face_charts
    )
    return faces, np.asarray(midpoints, dtype=float)


def _principal_report(
    configuration: dict,
    active_outer_rg: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    faces, charts = _face_midpoint_charts(configuration)
    radius_rg = context.grid.edges / context.grid.gravitational_radius
    active = radius_rg <= active_outer_rg * (1.0 + 2.0e-14)
    selected_faces = faces[active]
    analytic = []
    numerical = []
    eigenpair = []
    biorthogonality = []
    conditions = []
    imaginary = []
    incoming = []
    continuity = []
    previous = None
    for face in selected_faces:
        basis = causal_five_field_coordinate_principal_basis(
            context,
            float(context.grid.edges[face]),
            charts[face],
        )
        analytic.append(basis.analytic_speeds_over_c)
        numerical.append(basis.numerical_speeds_over_c)
        eigenpair.append(basis.maximum_eigenpair_defect)
        biorthogonality.append(basis.maximum_biorthogonality_defect)
        conditions.append(basis.descriptor_condition_number)
        imaginary.append(basis.maximum_imaginary_part)
        incoming.append(basis.incoming_inner_characteristics)
        right = basis.descriptor_right_eigenvectors
        if previous is None:
            continuity.append(1.0)
        else:
            continuity.append(
                min(
                    abs(
                        float(np.dot(previous[:, family], right[:, family]))
                        / max(
                            float(np.linalg.norm(previous[:, family])),
                            np.finfo(float).tiny,
                        )
                        / max(
                            float(np.linalg.norm(right[:, family])),
                            np.finfo(float).tiny,
                        )
                    )
                    for family in range(5)
                )
            )
        previous = right
    analytic = np.asarray(analytic, dtype=float)
    numerical = np.asarray(numerical, dtype=float)
    maximum_speed = float(np.max(np.abs(numerical - analytic)))
    report = {
        "active_faces": int(selected_faces.size),
        "maximum_analytic_valencia_speed_defect_over_c": maximum_speed,
        "maximum_eigenpair_defect": float(np.max(eigenpair)),
        "maximum_biorthogonality_defect": float(
            np.max(biorthogonality)
        ),
        "maximum_descriptor_condition_number": float(
            np.max(conditions)
        ),
        "maximum_imaginary_part": float(np.max(imaginary)),
        "minimum_cross_face_family_continuity": float(
            np.min(continuity)
        ),
        "maximum_incoming_inner_characteristics": int(np.max(incoming)),
    }
    report["passed"] = bool(
        report["maximum_analytic_valencia_speed_defect_over_c"]
        <= MAXIMUM_ANALYTIC_SPEED_DEFECT
        and report["maximum_eigenpair_defect"]
        <= MAXIMUM_EIGENPAIR_DEFECT
        and report["maximum_biorthogonality_defect"]
        <= MAXIMUM_BIORTHOGONALITY_DEFECT
        and report["maximum_descriptor_condition_number"]
        <= MAXIMUM_BASIS_CONDITION
        and report["maximum_imaginary_part"]
        <= MAXIMUM_EIGENPAIR_DEFECT
        and report["minimum_cross_face_family_continuity"]
        >= MINIMUM_CROSS_FACE_CONTINUITY
        and report["maximum_incoming_inner_characteristics"] == 0
    )
    arrays = {
        "face_indices": selected_faces,
        "radius_rg": radius_rg[selected_faces],
        "analytic_speeds_over_c": analytic,
        "numerical_speeds_over_c": numerical,
        "cross_face_continuity": np.asarray(continuity, dtype=float),
        "condition_number": np.asarray(conditions, dtype=float),
    }
    return report, arrays


def _operator_paths(label: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / f"{label}_operator.json",
        CHECKPOINT_DIRECTORY / f"{label}_operator_arrays.npz",
    )


def _candidate_operator(
    configuration: dict,
    *,
    force: bool,
) -> tuple[dict, np.ndarray, object]:
    label = f"{configuration['label']}_characteristic_matrix"
    context = replace(
        configuration["context"],
        interior_dissipation_mode="characteristic_matrix_audit",
    ).validated()
    base = np.asarray(configuration["base_primitives"], dtype=float)
    json_path, arrays_path = _operator_paths(label)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "label": label,
        "grid_edges_sha256": _array_sha256(context.grid.edges),
        "base_primitives_sha256": _array_sha256(base),
        "interior_dissipation_mode": context.interior_dissipation_mode,
        "core_file_sha256": _sha256(ROOT / CORE_FILE),
    }
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
            and payload.get("passed") is True
        ):
            arrays = _load_npz(arrays_path)
            return payload, np.asarray(arrays["generator"], dtype=float), context

    print(f"WP10c9b: building {label}", flush=True)
    started = time.perf_counter()
    state = causal_five_field_state_from_primitives(context, base)
    vector = pack_causal_five_field_state(state)
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
        finite_difference_step=wp10c8v.FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=wp10c8v.DESCRIPTOR_TIMESTEP_SECONDS,
    )
    evolving = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        primitive_rate_per_s=None,
        reduced_descriptor=reduced,
        finite_difference_step=wp10c8v.FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=wp10c8v.DESCRIPTOR_TIMESTEP_SECONDS,
        storage_difference_step=wp10c8v.STORAGE_DIFFERENCE_STEP,
        storage_rate_derivative_step=(
            wp10c8v.STORAGE_RATE_DERIVATIVE_STEP
        ),
        storage_quadrature_order=wp10c8v.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8v.STORAGE_DIRECTIONAL_STEP,
    )
    generator = wp10c8v._similarity_rescale_generator(
        np.asarray(evolving["evolving_scaled_generator_per_s"], dtype=float),
        np.asarray(evolving["primitive_column_scales"], dtype=float),
        np.asarray(configuration["amplitudes"], dtype=float),
    )
    arrays = {
        "generator": generator,
        "primitive_column_scales": np.asarray(
            evolving["primitive_column_scales"],
            dtype=float,
        ),
    }
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "maximum_scaled_generator_factorization_defect": float(
            evolving["maximum_scaled_generator_factorization_defect"]
        ),
        "maximum_relative_storage_action_defect": float(
            evolving["maximum_relative_storage_action_defect"]
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    payload["passed"] = bool(
        payload["maximum_scaled_generator_factorization_defect"]
        <= wp10c8z.MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
        and payload["maximum_relative_storage_action_defect"]
        <= MAXIMUM_STORAGE_ACTION_DEFECT
    )
    json_path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload, generator, context


def _candidate_configuration(
    production: dict,
    *,
    force: bool,
) -> dict:
    report, generator, context = _candidate_operator(
        production,
        force=force,
    )
    result = dict(production)
    result.update(
        {
            "label": f"{production['label']}_characteristic_matrix",
            "context": context,
            "generator": generator,
            "candidate_operator_report": report,
        }
    )
    return result


def _smooth_dissipation_norm(
    configuration: dict,
    packet: np.ndarray,
) -> float:
    context = configuration["context"]
    base = np.asarray(configuration["base_primitives"], dtype=float)
    physical = np.asarray(configuration["amplitudes"], dtype=float) * packet
    step = wp10c9a.RECONSTRUCTION_DIRECTIONAL_STEP

    def dissipative_faces(primitives: np.ndarray) -> np.ndarray:
        reconstruction = causal_five_field_reconstruct_face_charts(
            context,
            primitives,
        )
        result = np.zeros(
            (context.grid.edges.size, 5),
            dtype=float,
        )
        for face in range(1, context.grid.centers.size):
            candidate = causal_five_field_characteristic_dissipation(
                context,
                float(context.grid.edges[face]),
                reconstruction.left_face_charts[face],
                reconstruction.right_face_charts[face],
                face_measure=float(context.grid.face_measures[face]),
            )
            result[face] = candidate.dissipative_flux_over_c
        return result

    tangent = (
        dissipative_faces(base + step * physical)
        - dissipative_faces(base - step * physical)
    ) / (2.0 * step)
    divergence = tangent[1:] - tangent[:-1]
    restricted = wp10c9a._integrated_restriction(
        divergence,
        configuration,
    )
    active = wp10c8z.ACTIVE_OUTER_PARENT_FACE
    return float(np.linalg.norm(restricted[:active]))


def _packet_ladder(
    parent: dict,
    configurations: dict[int, dict],
) -> tuple[dict, dict[str, np.ndarray], bool]:
    reports = {}
    arrays = {}
    all_passed = True
    for family in CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES:
        print(f"WP10c9b: propagating {family}", flush=True)
        histories = {}
        moments = {}
        smooth = {}
        projection = {}
        for ratio, configuration in configurations.items():
            packet, bases, projected = wp10c9a._project_packet(
                configuration,
                family,
            )
            history, moment = wp10c9a._propagate_packet(
                configuration,
                packet,
                bases,
                family,
            )
            histories[ratio] = history
            moments[ratio] = moment
            projection[ratio] = projected
            smooth[ratio] = _smooth_dissipation_norm(
                configuration,
                packet,
            )
            arrays[f"{family}_ratio{ratio}_initial"] = packet
            arrays[f"{family}_ratio{ratio}_state_history"] = history["state"]
            arrays[f"{family}_ratio{ratio}_rate_history"] = history["rate"]
            arrays[f"{family}_ratio{ratio}_amplitude"] = moment["arrays"][
                "l2_amplitude"
            ]
            arrays[f"{family}_ratio{ratio}_centroid"] = moment["arrays"][
                "log_radius_centroid"
            ]
        restricted = {
            ratio: wp10c8z._restrict_history(
                histories[ratio],
                configurations[ratio],
            )
            for ratio in configurations
        }
        coarse_medium = wp10c8z._history_metrics(
            restricted[1],
            restricted[2],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=parent["active_outer_rg"],
        )
        medium_fine = wp10c8z._history_metrics(
            restricted[2],
            restricted[4],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=parent["active_outer_rg"],
        )
        moment_coarse_medium = wp10c9a._moment_pair(
            moments[1],
            moments[2],
        )
        moment_medium_fine = wp10c9a._moment_pair(
            moments[2],
            moments[4],
        )
        orders = {
            "state_history": _observed_order(
                coarse_medium["state"]["maximum_relative_l2_difference"],
                medium_fine["state"]["maximum_relative_l2_difference"],
            ),
            "rate_history": _observed_order(
                coarse_medium["rate"]["maximum_relative_l2_difference"],
                medium_fine["rate"]["maximum_relative_l2_difference"],
            ),
            "phase_centroid": _observed_order(
                moment_coarse_medium[
                    "maximum_log_radius_centroid_defect"
                ],
                moment_medium_fine[
                    "maximum_log_radius_centroid_defect"
                ],
            ),
            "damping": _observed_order(
                moment_coarse_medium["maximum_log_amplitude_defect"],
                moment_medium_fine["maximum_log_amplitude_defect"],
            ),
            "smooth_dissipation": _observed_order(smooth[2], smooth[4]),
        }
        fine_cosine = min(
            medium_fine["state"]["minimum_signed_cosine"],
            medium_fine["rate"]["minimum_signed_cosine"],
        )
        coupling_fraction = max(
            wp10c8z._coupling_signal_fraction(
                histories[ratio],
                configurations[ratio],
            )
            for ratio in configurations
        )
        passed = bool(
            orders["phase_centroid"] is not None
            and orders["phase_centroid"] >= MINIMUM_PACKET_PHASE_ORDER
            and orders["damping"] is not None
            and orders["damping"] >= MINIMUM_PACKET_DAMPING_ORDER
            and orders["smooth_dissipation"] is not None
            and orders["smooth_dissipation"] >= MINIMUM_SMOOTH_ORDER
            and fine_cosine >= MINIMUM_SIGNED_COSINE
            and coupling_fraction
            <= wp10c8z.MAXIMUM_COUPLING_SIGNAL_FRACTION
            and all(row["passed"] for row in projection.values())
        )
        reports[family] = {
            "projection": projection,
            "history_pairs": {
                "N128_N256patch": coarse_medium,
                "N256patch_N512patch": medium_fine,
            },
            "moment_pairs": {
                "N128_N256patch": moment_coarse_medium,
                "N256patch_N512patch": moment_medium_fine,
            },
            "smooth_dissipation_norm_by_ratio": smooth,
            "observed_orders": orders,
            "fine_minimum_signed_cosine": fine_cosine,
            "maximum_coupling_signal_fraction": coupling_fraction,
            "passed": passed,
        }
        all_passed = all_passed and passed
    return reports, arrays, all_passed


def _dense_colored_parity() -> dict:
    context = replace(
        make_causal_five_field_regression_context(4),
        interior_dissipation_mode="characteristic_matrix_audit",
    ).validated()
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    step = 2.0e-6
    zero = np.zeros_like(vector)

    def residual(increment: np.ndarray) -> np.ndarray:
        trial = vector + scaling.column_scales * increment
        return (
            evaluate_causal_five_field_dae(trial, context).residual
            / scaling.row_scales
        )

    dense = np.empty((zero.size, zero.size), dtype=float)
    for column in range(zero.size):
        plus = np.array(zero, copy=True)
        minus = np.array(zero, copy=True)
        plus[column] += step
        minus[column] -= step
        dense[:, column] = (
            residual(plus) - residual(minus)
        ) / (2.0 * step)
    pattern = causal_five_field_dae_jacobian_sparsity(
        4,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=(
            context.boundary_trace_reconstruction
        ),
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )
    colored = causal_five_field_colored_central_jacobian(
        residual,
        zero,
        pattern,
        finite_difference_step=step,
    ).toarray()
    row_scale = np.maximum(np.max(np.abs(dense), axis=1), 1.0e-14)
    allowed = pattern.toarray().astype(bool)
    omitted = float(
        np.max(
            np.abs(np.where(allowed, 0.0, dense))
            / row_scale[:, None]
        )
    )
    parity = float(
        np.max(np.abs(colored - dense) / row_scale[:, None])
    )
    return {
        "maximum_omitted_relative_entry": omitted,
        "maximum_dense_colored_relative_defect": parity,
        "passed": bool(
            omitted <= MAXIMUM_DENSE_COLORED_DEFECT
            and parity <= MAXIMUM_DENSE_COLORED_DEFECT
        ),
    }


def _bdf_replay() -> dict:
    context = replace(
        make_causal_five_field_regression_context(4),
        interior_dissipation_mode="characteristic_matrix_audit",
    ).validated()
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=1.0e-10,
        maximum_dt=1.0e-5,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        residual_tolerance=1.0e-10,
        algebraic_residual_tolerance=1.0e-10,
        conservation_tolerance=1.0e-9,
        maximum_newton_iterations=12,
    ).validated()
    split = {}

    def capture(completed, _total, state, history) -> None:
        if completed == 2:
            split["state"] = np.array(state, copy=True)
            split["history"] = history

    full = evolve_causal_five_field_fixed_bdf2(
        context,
        vector,
        np.zeros_like(vector),
        1.0e-8,
        4.0e-8,
        4,
        config,
        progress=capture,
    )
    history = split.get("history")
    if history is None:
        return {
            "full_passed": full.passed,
            "completed_steps": full.completed_steps,
            "maximum_scaled_residual": full.maximum_scaled_residual,
            "maximum_scaled_algebraic_residual": (
                full.maximum_scaled_algebraic_residual
            ),
            "maximum_discrete_ledger_relative_defect": (
                full.maximum_discrete_ledger_relative_defect
            ),
            "maximum_newton_iterations": full.maximum_newton_iterations,
            "message": full.message,
            "binding_residual_tolerance": config.residual_tolerance,
            "passed": False,
            "reason": (
                "candidate nonlinear step stagnated above the unchanged "
                "residual tolerance before a replayable split history"
            ),
        }
    replay = evolve_causal_five_field_fixed_bdf2(
        context,
        np.asarray(split["state"], dtype=float),
        history.previous_physical_increment,
        history.previous_timestep_seconds,
        2.0e-8,
        2,
        config,
        startup_with_bdf1=False,
        initial_history=history,
    )
    state_equal = np.array_equal(replay.state_vector, full.state_vector)
    increment_equal = np.array_equal(
        replay.history.previous_physical_increment,
        full.history.previous_physical_increment,
    )
    vertical_equal = np.array_equal(
        replay.history.previous_vertical_killing_increment,
        full.history.previous_vertical_killing_increment,
    )
    return {
        "full_passed": full.passed,
        "replay_passed": replay.passed,
        "state_bitwise_equal": state_equal,
        "increment_bitwise_equal": increment_equal,
        "vertical_history_bitwise_equal": vertical_equal,
        "passed": bool(
            full.passed
            and replay.passed
            and state_equal
            and increment_equal
            and vertical_equal
        ),
    }


def _common_mode(
    parent: dict,
    configurations: dict[int, dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    histories = {
        ratio: wp10c8z._propagate(configuration)
        for ratio, configuration in configurations.items()
    }
    restricted = {
        ratio: wp10c8z._restrict_history(
            histories[ratio],
            configurations[ratio],
        )
        for ratio in configurations
    }
    coarse_medium = wp10c8z._history_metrics(
        restricted[1],
        restricted[2],
        parent["parent_grid"],
        lower_rg=None,
        upper_rg=parent["active_outer_rg"],
    )
    medium_fine = wp10c8z._history_metrics(
        restricted[2],
        restricted[4],
        parent["parent_grid"],
        lower_rg=None,
        upper_rg=parent["active_outer_rg"],
    )
    orders = {
        "state_history": _observed_order(
            coarse_medium["state"]["maximum_relative_l2_difference"],
            medium_fine["state"]["maximum_relative_l2_difference"],
        ),
        "rate_history": _observed_order(
            coarse_medium["rate"]["maximum_relative_l2_difference"],
            medium_fine["rate"]["maximum_relative_l2_difference"],
        ),
    }
    fine_cosine = min(
        medium_fine["state"]["minimum_signed_cosine"],
        medium_fine["rate"]["minimum_signed_cosine"],
    )
    passed = bool(
        all(
            value is not None and value >= MINIMUM_PACKET_PHASE_ORDER
            for value in orders.values()
        )
        and fine_cosine >= MINIMUM_SIGNED_COSINE
    )
    arrays = {}
    for ratio, history in histories.items():
        arrays[f"common_ratio{ratio}_state_history"] = history["state"]
        arrays[f"common_ratio{ratio}_rate_history"] = history["rate"]
    return {
        "history_pairs": {
            "N128_N256patch": coarse_medium,
            "N256patch_N512patch": medium_fine,
        },
        "observed_orders": orders,
        "fine_minimum_signed_cosine": fine_cosine,
        "passed": passed,
    }, arrays


def run(*, force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    if not WP10C9A_OUTPUT.exists() or not WP10C9A_ARRAYS.exists():
        raise FileNotFoundError("WP10c9b requires WP10c9a evidence")
    parent_payload = json.loads(WP10C9A_OUTPUT.read_text(encoding="utf-8"))
    if parent_payload.get("classification") != (
        "characteristic_rate_phase_unresolved_operator_redesign_required"
    ):
        raise RuntimeError("WP10c9a stop classification changed")

    parent, production_by_label, labels = wp10c9a._configurations()
    principal = {}
    arrays = {}
    for ratio, label in labels.items():
        report, basis_arrays = _principal_report(
            production_by_label[label],
            parent["active_outer_rg"],
        )
        principal[f"ratio_{ratio}"] = report
        for name, values in basis_arrays.items():
            arrays[f"principal_ratio{ratio}_{name}"] = values
    principal_passed = all(row["passed"] for row in principal.values())

    candidate_configurations = {}
    operator_reports = {}
    for ratio, label in labels.items():
        candidate = _candidate_configuration(
            production_by_label[label],
            force=force,
        )
        candidate["active_outer_rg"] = parent["active_outer_rg"]
        candidate_configurations[ratio] = candidate
        operator_reports[f"ratio_{ratio}"] = candidate[
            "candidate_operator_report"
        ]

    print("WP10c9b: certifying method/Jacobian/restart contracts", flush=True)
    parity = _dense_colored_parity()
    bdf = _bdf_replay()
    shared_flux = max(
        causal_embedded_patch_flux_audit(
            configuration["context"],
            pack_causal_five_field_state(
                causal_five_field_state_from_primitives(
                    configuration["context"],
                    configuration["base_primitives"],
                )
            ),
            configuration["layout"],
        ).maximum_state_flux_defect
        for configuration in candidate_configurations.values()
    )
    maximum_storage = max(
        report["maximum_relative_storage_action_defect"]
        for report in operator_reports.values()
    )
    equal_speed = 0.0
    minimum_quadratic = float("inf")
    constant_defect = 0.0
    for configuration in candidate_configurations.values():
        context = configuration["context"]
        reconstruction = causal_five_field_reconstruct_face_charts(
            context,
            configuration["base_primitives"],
        )
        representative = np.linspace(
            1,
            context.grid.centers.size - 1,
            8,
            dtype=int,
        )
        for face in representative:
            candidate = causal_five_field_characteristic_dissipation(
                context,
                float(context.grid.edges[face]),
                reconstruction.left_face_charts[face],
                reconstruction.right_face_charts[face],
                face_measure=float(context.grid.face_measures[face]),
            )
            equal_speed = max(
                equal_speed,
                candidate.scalar_equal_speed_defect,
            )
            minimum_quadratic = min(
                minimum_quadratic,
                candidate.quadratic_dissipation,
            )
            constant = causal_five_field_characteristic_dissipation(
                context,
                float(context.grid.edges[face]),
                reconstruction.left_face_charts[face],
                reconstruction.left_face_charts[face],
                face_measure=float(context.grid.face_measures[face]),
            )
            constant_defect = max(
                constant_defect,
                float(np.max(np.abs(constant.dissipative_flux_over_c))),
            )
    method_pre_packet_passed = bool(
        principal_passed
        and all(report["passed"] for report in operator_reports.values())
        and parity["passed"]
        and bdf["passed"]
        and shared_flux <= MAXIMUM_SHARED_FLUX_DEFECT
        and maximum_storage <= MAXIMUM_STORAGE_ACTION_DEFECT
        and equal_speed <= MAXIMUM_EQUAL_SPEED_DEFECT
        and minimum_quadratic >= 0.0
        and constant_defect == 0.0
    )

    packet_reports, packet_arrays, packets_passed = _packet_ladder(
        parent,
        candidate_configurations,
    )
    arrays.update(packet_arrays)
    common_report = None
    common_arrays = {}
    if method_pre_packet_passed and packets_passed:
        print("WP10c9b: all packets passed; rerunning common mode", flush=True)
        common_report, common_arrays = _common_mode(
            parent,
            candidate_configurations,
        )
        arrays.update(common_arrays)
    common_passed = bool(
        common_report is not None and common_report["passed"]
    )

    if not principal_passed:
        classification = "coordinate_principal_basis_failed"
    elif not method_pre_packet_passed and not packets_passed:
        classification = (
            "characteristic_matrix_rejected_bdf_noise_and_"
            "inward_shear_damping_unresolved"
        )
    elif not method_pre_packet_passed:
        classification = "characteristic_matrix_method_contract_failed"
    elif not packets_passed:
        classification = (
            "characteristic_matrix_packets_failed_"
            "path_conservative_shear_required"
        )
    elif not common_passed:
        classification = (
            "characteristic_packets_passed_common_mode_"
            "family_coupling_unresolved"
        )
    else:
        classification = (
            "characteristic_matrix_common_mode_converged_"
            "nonlinear_patch_truth_authorized"
        )

    arrays["times"] = np.linspace(
        0.0,
        wp10c9a.TARGET_SECONDS,
        wp10c9a.TIME_SAMPLES,
    )
    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "passed": common_passed,
        "scope": {
            "production_operator_changed": False,
            "audit_only_characteristic_matrix_built": True,
            "pure_packet_ladder_run": True,
            "common_mode_rerun": common_report is not None,
            "bounded_nonlinear_patch_truth_run": False,
            "fixed_q_averaging_run": False,
            "reduced_model_run": False,
        },
        "frozen_parent_evidence": {
            "wp10c9a_json": _relative(WP10C9A_OUTPUT),
            "wp10c9a_json_sha256": _sha256(WP10C9A_OUTPUT),
            "wp10c9a_arrays": _relative(WP10C9A_ARRAYS),
            "wp10c9a_arrays_sha256": _sha256(WP10C9A_ARRAYS),
            "wp10c9a_classification": parent_payload["classification"],
        },
        "coordinate_principal_basis": {
            "by_refinement_ratio": principal,
            "passed": principal_passed,
            "interpretation": (
                "the exact implemented pencil includes a bounded "
                "background-stress correction to the ideal Valencia cones; "
                "family ordering, reality, completeness, continuity, and "
                "zero incoming excision characteristics are binding"
            ),
        },
        "candidate_operator": {
            "name": "full_five_family_characteristic_matrix",
            "central_flux": "unchanged_production_physical_central_flux",
            "dissipation": "R_absLambda_L_descriptor_path_jump",
            "by_refinement_ratio": operator_reports,
        },
        "method_contract": {
            "constant_state_dissipative_flux_defect": constant_defect,
            "minimum_characteristic_quadratic_dissipation": (
                minimum_quadratic
            ),
            "maximum_equal_speed_reduction_defect": equal_speed,
            "maximum_shared_flux_defect": shared_flux,
            "maximum_storage_action_defect": maximum_storage,
            "dense_colored_jacobian": parity,
            "bdf2_split_replay": bdf,
            "no_incoming_excision_characteristic": True,
            "passed_before_packets": method_pre_packet_passed,
        },
        "packet_results": packet_reports,
        "all_packet_gates_passed": packets_passed,
        "conditional_common_mode": common_report,
        "machine_evidence": {
            "arrays_path": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
        },
        "gates": {
            "minimum_smooth_order": MINIMUM_SMOOTH_ORDER,
            "minimum_packet_phase_order": MINIMUM_PACKET_PHASE_ORDER,
            "minimum_packet_damping_order": MINIMUM_PACKET_DAMPING_ORDER,
            "minimum_same_time_signed_cosine": MINIMUM_SIGNED_COSINE,
            "maximum_analytic_speed_defect_over_c": (
                MAXIMUM_ANALYTIC_SPEED_DEFECT
            ),
            "maximum_eigenpair_defect": MAXIMUM_EIGENPAIR_DEFECT,
            "maximum_biorthogonality_defect": (
                MAXIMUM_BIORTHOGONALITY_DEFECT
            ),
            "maximum_shared_flux_defect": MAXIMUM_SHARED_FLUX_DEFECT,
            "maximum_storage_action_defect": (
                MAXIMUM_STORAGE_ACTION_DEFECT
            ),
            "maximum_dense_colored_defect": (
                MAXIMUM_DENSE_COLORED_DEFECT
            ),
        },
        "decision": {
            "candidate_promoted_to_production": False,
            "pure_packets_passed": packets_passed,
            "common_mode_rerun_authorized_and_run": (
                common_report is not None
            ),
            "bounded_nonlinear_patch_truth_authorized": common_passed,
            "one_more_brute_force_refinement_authorized": False,
            "fixed_q_averaging_authorized": False,
            "reduced_model_authorized": False,
            "next_step": (
                "run one bounded nonlinear embedded-patch truth experiment"
                if common_passed
                else (
                    "diagnose nonlinear family coupling"
                    if packets_passed
                    else (
                        "derive a path-conservative causal-shear Riemann "
                        "coupling or redesign the near-horizon bulk operator"
                    )
                )
            ),
        },
        "provenance": {
            "runner": THIS_RUNNER,
            "core_file": CORE_FILE,
            "core_file_sha256": _sha256(ROOT / CORE_FILE),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    DEFAULT_OUTPUT.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    payload, _arrays = run(force=arguments.force)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "passed": payload["passed"],
                "all_packet_gates_passed": payload[
                    "all_packet_gates_passed"
                ],
                "common_mode_run": payload["scope"]["common_mode_rerun"],
                "output": str(DEFAULT_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
