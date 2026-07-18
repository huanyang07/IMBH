"""Run the bounded N16 causal clock and timestep-ceiling audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    SchwarzschildCurvatureVerticalFrequency,
    ValenciaPerfectFluidPrimitive,
    advance_causal_five_field_increment_backward_euler,
    audit_causal_five_field_state_gates,
    causal_backward_euler_step_doubling_factor,
    causal_five_field_dae_scaling,
    causal_five_field_local_timescale_audit,
    causal_five_field_observable_snapshot,
    causal_five_field_physical_step_ledger,
    causal_five_field_temporal_error_ratio,
    compare_causal_five_field_observables,
    evaluate_causal_five_field_dae,
    exact_kerr_schild_compact_stream_sources,
    fiducial_hill_roche_nozzle_geometry,
    kerr_schild_column_geometry,
    kerr_schild_stream_injection,
    load_causal_five_field_adaptive_restart,
    make_kerr_schild_column_grid,
    unpack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESTART_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c5k"
)
N16_RESULT = (
    ROOT
    / "outputs/tables"
    / "causal_timescale_timestep_audit_wp10c6a.json"
)
SUPPORTED_CELL_COUNTS = (16, 32)
STREAM_CENTER_RG = 240.0
STREAM_LOG_WIDTH = 0.08
STREAM_MDOT_EDD = 5.0
STREAM_SURFACE_DENSITY = 1.0e5
STREAM_TEMPERATURE = 1.0e6
TIMESTEP_FACTOR = 2.0
MAXIMUM_RUNGS = 12
COOLING_INNER_CUTOFF_RG = 6.0
FINITE_DIFFERENCE_STEP = 2.0e-6

TEMPORAL_ACCURACY_GATES = {
    "cooling_power_proxy_relative": 1.0e-3,
    "cooling_power_proxy_outside_cutoff_relative": 1.0e-3,
    "inner_accretion_rate_relative": 1.0e-3,
    "maximum_log_h_over_r_profile": 2.0e-3,
    "maximum_integrated_conserved_relative": 1.0e-3,
    "maximum_baseline_scaled_state_difference": 2.0e-3,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-cells",
        type=int,
        choices=SUPPORTED_CELL_COUNTS,
        default=16,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--maximum-rungs",
        type=int,
        default=MAXIMUM_RUNGS,
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _default_checkpoint(n_cells: int) -> Path:
    return (
        DEFAULT_RESTART_DIRECTORY
        / f"causal_wp10c5q_N{n_cells:03d}_final.npz"
    )


def _default_output(n_cells: int) -> Path:
    label = "wp10c6a" if n_cells == 16 else "wp10c6b"
    return (
        ROOT
        / "outputs/tables"
        / f"causal_timescale_timestep_audit_{label}.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_regression_stream(
    context: CausalFiveFieldDAEContext,
    mass: float,
    gravitational_radius: float,
):
    radius = STREAM_CENTER_RG * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    thermodynamics = context.vertical_frequency.eos(
        radius
    ).from_surface_density_temperature(
        STREAM_SURFACE_DENSITY,
        STREAM_TEMPERATURE,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=STREAM_SURFACE_DENSITY,
        radial_velocity_over_c=(
            2.0 * gravitational_radius / radius
        ),
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=STREAM_MDOT_EDD * eddington_mdot(mass),
    )
    return exact_kerr_schild_compact_stream_sources(
        context.grid,
        injection,
        center=radius,
        log_width=STREAM_LOG_WIDTH,
        shape="compact_c2",
    )


def _context(n_cells: int) -> CausalFiveFieldDAEContext:
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        n_cells,
        gravitational_radius,
    )
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    context = CausalFiveFieldDAEContext(
        grid=grid,
        vertical_frequency=SchwarzschildCurvatureVerticalFrequency(
            gravitational_radius
        ),
        outer_boundary_provider=GasRadiationHillRocheNozzleProvider(
            geometry,
            transverse_quadrature_zones=24,
        ),
        include_radiative_cooling=True,
    ).validated()
    return replace(
        context,
        stream_sources=_exact_regression_stream(
            context,
            mass,
            gravitational_radius,
        ),
    ).validated()


def _array(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def _nullable_array(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if np.isfinite(value) else None
        for value in np.asarray(values, dtype=float)
    ]


def _observable_dict(snapshot) -> dict:
    result = asdict(snapshot)
    result["h_over_r"] = _array(snapshot.h_over_r)
    result["integrated_conserved"] = _array(
        snapshot.integrated_conserved
    )
    return result


def _clock_summary(context, clocks) -> dict:
    radius_rg = (
        context.grid.centers / context.grid.gravitational_radius
    )
    names = (
        "characteristic_crossing_seconds",
        "stress_relaxation_seconds",
        "thermal_response_seconds",
        "luminosity_response_seconds",
        "radial_advection_seconds",
        "local_loading_seconds",
    )
    profiles = {}
    minima = {}
    for name in names:
        values = np.asarray(getattr(clocks, name), dtype=float)
        index = int(np.argmin(values))
        profiles[name] = _nullable_array(values)
        minima[name] = {
            "seconds": float(values[index]),
            "cell_index": index,
            "radius_rg": float(radius_rg[index]),
        }
    finite_physical = {
        name: entry
        for name, entry in minima.items()
        if name != "local_loading_seconds"
    }
    shortest_name = min(
        finite_physical,
        key=lambda name: finite_physical[name]["seconds"],
    )
    return {
        "radius_rg": _array(radius_rg),
        "profiles": profiles,
        "cooling_log_temperature_derivative": _array(
            clocks.cooling_log_temperature_derivative
        ),
        "minima": minima,
        "global_loading_seconds": float(clocks.global_loading_seconds),
        "shortest_physical_clock": {
            "name": shortest_name,
            **finite_physical[shortest_name],
        },
    }


def _physical_ledger_summary(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    physical_increment: np.ndarray,
    timestep: float,
) -> dict:
    ledger = causal_five_field_physical_step_ledger(
        context,
        old_vector,
        physical_increment,
        timestep,
    )
    scale = (
        np.abs(ledger.conserved_storage_change)
        + np.abs(ledger.vertical_storage_change)
        + np.abs(ledger.boundary_transport)
        + np.abs(ledger.endogenous_source)
        + np.abs(ledger.prescribed_stream_source)
    )
    relative = np.abs(ledger.closure_defect) / np.maximum(
        scale,
        np.finfo(float).tiny,
    )
    return {
        "conserved_storage_change": _array(
            ledger.conserved_storage_change
        ),
        "vertical_storage_change": _array(
            ledger.vertical_storage_change
        ),
        "boundary_transport": _array(ledger.boundary_transport),
        "endogenous_source": _array(ledger.endogenous_source),
        "prescribed_stream_source": _array(
            ledger.prescribed_stream_source
        ),
        "closure_defect": _array(ledger.closure_defect),
        "component_relative_defect": _array(relative),
        "maximum_relative_defect": float(np.max(relative)),
    }


def _step_summary(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    step,
) -> dict:
    result = {
        "accepted": bool(step.accepted),
        "timestep_seconds": float(step.timestep_seconds),
        "maximum_scaled_residual": float(
            step.maximum_scaled_residual
        ),
        "maximum_scaled_algebraic_residual": float(
            step.maximum_scaled_algebraic_residual
        ),
        "maximum_scaled_primitive_change": float(
            step.maximum_scaled_primitive_change
        ),
        "maximum_scaled_total_change": float(
            step.maximum_scaled_total_change
        ),
        "conservation_telescoping_relative_defect": float(
            step.conservation_telescoping_relative_defect
        ),
        "component_conservation_defects": [
            float(value)
            for value in step.component_conservation_defects
        ],
        "minimum_scattering_optical_depth": float(
            step.minimum_scattering_optical_depth
        ),
        "outer_boundary_choked_before": bool(
            step.outer_boundary_choked_before
        ),
        "outer_boundary_choked_after": bool(
            step.outer_boundary_choked_after
        ),
        "iterations": int(step.iterations),
        "function_evaluations": int(step.function_evaluations),
        "jacobian_evaluations": int(step.jacobian_evaluations),
        "maximum_linear_residual": float(
            step.maximum_linear_residual
        ),
        "message": step.message,
    }
    if step.accepted:
        result["physical_ledger"] = _physical_ledger_summary(
            context,
            old_vector,
            step.physical_increment,
            step.timestep_seconds,
        )
        result["state_gates"] = audit_causal_five_field_state_gates(
            context,
            step.state_vector,
        )
    else:
        result["physical_ledger"] = None
        result["state_gates"] = None
    return result


def _step_contract_passed(summary: dict) -> bool:
    return bool(
        summary["accepted"]
        and summary["physical_ledger"]["maximum_relative_defect"]
        <= 1.0e-10
        and summary["state_gates"]["passed"]
    )


def _accuracy_audit(errors: dict) -> dict[str, object]:
    return causal_five_field_temporal_error_ratio(
        errors,
        TEMPORAL_ACCURACY_GATES,
    )


def _failure_class(
    full: dict,
    half_one: dict,
    half_two: dict | None,
    *,
    accuracy_passed: bool | None,
) -> str:
    steps = [full, half_one]
    if half_two is not None:
        steps.append(half_two)
    if any(not step["accepted"] for step in steps):
        if any(
            "acceptance gates" in step["message"]
            and (
                step["maximum_scaled_primitive_change"] > 0.2
                or step["maximum_scaled_total_change"] > 0.25
            )
            for step in steps
        ):
            return "audit_safety_change_gate"
        return "nonlinear_or_step_acceptance"
    if any(
        not step["state_gates"]["passed"]
        for step in steps
    ):
        return "physical_state_gate"
    if any(
        step["physical_ledger"]["maximum_relative_defect"] > 1.0e-10
        for step in steps
    ):
        return "physical_conservation_gate"
    if accuracy_passed is False:
        return "temporal_accuracy_gate"
    return "none"


def _run_rung(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    previous_increment: np.ndarray,
    previous_dt: float,
    timestep: float,
    config: CausalFiveFieldAdaptiveStepConfig,
    baseline_column_scales: np.ndarray,
    cooling_cutoff: float,
) -> dict:
    full_predictor = previous_increment * (timestep / previous_dt)
    half_timestep = 0.5 * timestep
    half_predictor = previous_increment * (
        half_timestep / previous_dt
    )
    full_step = advance_causal_five_field_increment_backward_euler(
        context,
        old_vector,
        timestep,
        full_predictor,
        config,
    )
    half_one_step = advance_causal_five_field_increment_backward_euler(
        context,
        old_vector,
        half_timestep,
        half_predictor,
        config,
    )
    half_two_step = None
    if half_one_step.accepted:
        half_two_step = (
            advance_causal_five_field_increment_backward_euler(
                context,
                half_one_step.state_vector,
                half_timestep,
                half_one_step.physical_increment,
                config,
            )
        )
    full = _step_summary(context, old_vector, full_step)
    half_one = _step_summary(context, old_vector, half_one_step)
    half_two = (
        _step_summary(
            context,
            half_one_step.state_vector,
            half_two_step,
        )
        if half_two_step is not None
        else None
    )
    contracts_passed = bool(
        _step_contract_passed(full)
        and _step_contract_passed(half_one)
        and half_two is not None
        and _step_contract_passed(half_two)
    )
    errors = None
    accuracy_audit = None
    accuracy_passed = None
    if contracts_passed:
        full_observables = causal_five_field_observable_snapshot(
            context,
            full_step.state_vector,
            cooling_inner_cutoff=cooling_cutoff,
        )
        half_observables = causal_five_field_observable_snapshot(
            context,
            half_two_step.state_vector,
            cooling_inner_cutoff=cooling_cutoff,
        )
        errors = compare_causal_five_field_observables(
            full_observables,
            half_observables,
        )
        errors["maximum_baseline_scaled_state_difference"] = float(
            np.max(
                np.abs(
                    (
                        full_step.state_vector
                        - half_two_step.state_vector
                    )
                    / baseline_column_scales
                )
            )
        )
        accuracy_audit = _accuracy_audit(errors)
        accuracy_passed = bool(accuracy_audit["passed"])
    failure = _failure_class(
        full,
        half_one,
        half_two,
        accuracy_passed=accuracy_passed,
    )
    passed = bool(contracts_passed and accuracy_passed)
    return {
        "timestep_seconds": float(timestep),
        "half_timestep_seconds": float(half_timestep),
        "full_step": full,
        "first_half_step": half_one,
        "second_half_step": half_two,
        "all_step_contracts_passed": contracts_passed,
        "step_doubling_errors": errors,
        "temporal_gate_audit": accuracy_audit,
        "temporal_accuracy_passed": accuracy_passed,
        "passed": passed,
        "failure_class": failure,
    }


def _temporal_mesh_comparison(
    *,
    n32_previous_dt: float,
    n32_largest_passed: float,
    n32_first_failed: float,
    n32_rungs: list[dict],
) -> dict:
    if not N16_RESULT.exists():
        raise FileNotFoundError(
            "WP10c6a N16 result is required for N32 comparison"
        )
    parent = json.loads(N16_RESULT.read_text(encoding="utf-8"))
    parent_result = parent["result"]
    parent_failure = next(
        row for row in parent["rungs"] if not row["passed"]
    )
    current_failure = next(
        row for row in n32_rungs if not row["passed"]
    )
    parent_failure_audit = causal_five_field_temporal_error_ratio(
        parent_failure["step_doubling_errors"],
        parent["temporal_accuracy_gates"],
    )
    current_failure_audit = current_failure["temporal_gate_audit"]
    parent_largest = float(
        parent_result["largest_passing_timestep_seconds"]
    )
    parent_first_failed = float(
        parent_result["first_failing_timestep_seconds"]
    )
    ceiling_ratio = n32_largest_passed / parent_largest
    failure_ratio = n32_first_failed / parent_first_failed
    parent_violated = set(
        parent_failure_audit["violated_observables"]
    )
    current_violated = set(
        current_failure_audit["violated_observables"]
    )
    common_violated = sorted(parent_violated & current_violated)
    same_schema = bool(
        parent["construction"]["observable_schema_version"]
        == CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
    )
    same_gates = bool(
        parent["temporal_accuracy_gates"]
        == TEMPORAL_ACCURACY_GATES
    )
    same_starting_timestep = bool(
        np.isclose(
            float(parent["checkpoint"]["previous_dt_seconds"]),
            n32_previous_dt,
            rtol=0.0,
            atol=0.0,
        )
    )
    ceiling_within_one_rung = bool(0.5 <= ceiling_ratio <= 2.0)
    failure_within_one_rung = bool(0.5 <= failure_ratio <= 2.0)
    passed = bool(
        parent_result["passed"]
        and same_schema
        and same_gates
        and same_starting_timestep
        and ceiling_within_one_rung
        and failure_within_one_rung
        and common_violated
        and parent_result["first_failure_class"]
        == "temporal_accuracy_gate"
        and current_failure["failure_class"]
        == "temporal_accuracy_gate"
    )
    shared_ceiling = min(parent_largest, n32_largest_passed)
    last_passing_n32 = next(
        row for row in reversed(n32_rungs) if row["passed"]
    )
    normalized_error = float(
        last_passing_n32["temporal_gate_audit"][
            "maximum_normalized_error"
        ]
    )
    return {
        "parent_result": str(N16_RESULT.relative_to(ROOT)),
        "n16_largest_passing_timestep_seconds": parent_largest,
        "n32_largest_passing_timestep_seconds": n32_largest_passed,
        "n16_first_failing_timestep_seconds": parent_first_failed,
        "n32_first_failing_timestep_seconds": n32_first_failed,
        "n32_over_n16_ceiling_ratio": float(ceiling_ratio),
        "n32_over_n16_failure_ratio": float(failure_ratio),
        "same_observable_schema": same_schema,
        "same_temporal_accuracy_gates": same_gates,
        "same_starting_timestep": same_starting_timestep,
        "ceiling_within_one_factor_two_rung": (
            ceiling_within_one_rung
        ),
        "failure_within_one_factor_two_rung": (
            failure_within_one_rung
        ),
        "n16_violated_observables": sorted(parent_violated),
        "n32_violated_observables": sorted(current_violated),
        "common_violated_observables": common_violated,
        "shared_conservative_ceiling_seconds": float(shared_ceiling),
        "passed": passed,
        "controller_contract": {
            "authorized": passed,
            "accepted_state": "two_half_step_state",
            "error_estimator": (
                "one_full_step_minus_two_half_steps"
            ),
            "normalized_error": (
                "maximum declared observable error divided by its gate"
            ),
            "backward_euler_error_exponent": 0.5,
            "safety_factor": 0.8,
            "minimum_timestep_factor": 0.25,
            "maximum_timestep_factor": 2.0,
            "factor_rule": (
                "clip(0.8/sqrt(normalized_error), 0.25, 2.0)"
            ),
            "initial_timestep_seconds": float(0.5 * shared_ceiling),
            "n32_last_passing_normalized_error": normalized_error,
            "n32_last_passing_proposed_factor": (
                causal_backward_euler_step_doubling_factor(
                    normalized_error
                )
            ),
            "all_existing_state_and_ledger_gates_remain_mandatory": (
                True
            ),
        },
    }


def main() -> None:
    args = _arguments()
    n_cells = int(args.n_cells)
    work_package = "WP10c6a" if n_cells == 16 else "WP10c6b"
    if args.maximum_rungs < 2 or args.maximum_rungs > MAXIMUM_RUNGS:
        raise ValueError(
            f"maximum rungs must lie in [2, {MAXIMUM_RUNGS}]"
        )
    checkpoint_path = (
        _default_checkpoint(n_cells)
        if args.checkpoint is None
        else _absolute(args.checkpoint)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    context = _context(n_cells)
    restart = load_causal_five_field_adaptive_restart(
        checkpoint_path,
        context,
    )
    checkpoint_provenance_passed = bool(
        restart.provenance.get("work_package") == "WP10c5q"
        and restart.provenance.get("n_cells") == n_cells
        and "exact circularized regression stream"
        in str(restart.provenance.get("source", ""))
    )
    initial_gates = audit_causal_five_field_state_gates(
        context,
        restart.state_vector,
    )
    if not checkpoint_provenance_passed or not initial_gates["passed"]:
        raise RuntimeError("WP10c5q checkpoint prerequisite failed")
    state = unpack_causal_five_field_state(
        restart.state_vector,
        n_cells,
    )
    evaluation = evaluate_causal_five_field_dae(
        restart.state_vector,
        context,
    )
    baseline_scales = causal_five_field_dae_scaling(
        state,
        evaluation,
    ).column_scales
    cooling_cutoff = (
        COOLING_INNER_CUTOFF_RG * context.grid.gravitational_radius
    )
    initial_observables = causal_five_field_observable_snapshot(
        context,
        restart.state_vector,
        cooling_inner_cutoff=cooling_cutoff,
    )
    clocks = causal_five_field_local_timescale_audit(
        context,
        restart.state_vector,
    )
    clock_summary = _clock_summary(context, clocks)
    config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=restart.previous_dt,
        maximum_dt=(
            restart.previous_dt
            * TIMESTEP_FACTOR ** (args.maximum_rungs - 1)
        ),
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=0,
        easy_iterations=3,
        residual_tolerance=1.0e-11,
        algebraic_residual_tolerance=1.0e-11,
        conservation_tolerance=1.0e-10,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()
    rungs = []
    timestep = float(restart.previous_dt)
    for _index in range(args.maximum_rungs):
        rung = _run_rung(
            context,
            restart.state_vector,
            restart.previous_physical_increment,
            restart.previous_dt,
            timestep,
            config,
            baseline_scales,
            cooling_cutoff,
        )
        rungs.append(rung)
        print(
            json.dumps(
                {
                    "timestep_seconds": rung["timestep_seconds"],
                    "passed": rung["passed"],
                    "failure_class": rung["failure_class"],
                    "step_doubling_errors": (
                        rung["step_doubling_errors"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not rung["passed"]:
            break
        timestep *= TIMESTEP_FACTOR
    passing = [row for row in rungs if row["passed"]]
    failing = [row for row in rungs if not row["passed"]]
    largest_passed = (
        float(passing[-1]["timestep_seconds"]) if passing else None
    )
    first_failed = (
        float(failing[0]["timestep_seconds"]) if failing else None
    )
    first_failure_class = (
        failing[0]["failure_class"]
        if failing
        else "ladder_upper_bound_without_failure"
    )
    shortest_clock = float(
        clock_summary["shortest_physical_clock"]["seconds"]
    )
    if first_failure_class == "temporal_accuracy_gate":
        classification = (
            "temporal_accuracy_limited_below_shortest_physical_clock"
        )
    elif first_failure_class == "ladder_upper_bound_without_failure":
        classification = "ceiling_not_bracketed_by_bounded_ladder"
    else:
        classification = first_failure_class
    individual_passed = bool(
        checkpoint_provenance_passed
        and initial_gates["passed"]
        and largest_passed is not None
        and first_failed is not None
        and first_failure_class == "temporal_accuracy_gate"
        and largest_passed < first_failed
        and first_failed < shortest_clock
    )
    mesh_comparison = None
    if n_cells == 32 and individual_passed:
        assert largest_passed is not None
        assert first_failed is not None
        mesh_comparison = _temporal_mesh_comparison(
            n32_previous_dt=restart.previous_dt,
            n32_largest_passed=largest_passed,
            n32_first_failed=first_failed,
            n32_rungs=rungs,
        )
    passed = bool(
        individual_passed
        and (
            n_cells == 16
            or (
                mesh_comparison is not None
                and mesh_comparison["passed"]
            )
        )
    )
    output = {
        "work_package": work_package,
        "scope": (
            f"bounded N{n_cells} one-full-versus-two-half "
            "backward-Euler timescale and timestep-ceiling audit"
        ),
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": _sha256(checkpoint_path),
            "elapsed_time_seconds": float(restart.elapsed_time),
            "previous_dt_seconds": float(restart.previous_dt),
            "dt_next_seconds": float(restart.dt_next),
            "accepted_steps": int(restart.accepted_steps),
            "rejected_attempts": int(restart.rejected_attempts),
            "provenance": restart.provenance,
            "provenance_passed": checkpoint_provenance_passed,
        },
        "construction": {
            "n_cells": n_cells,
            "observable_schema_version": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "timestep_factor": TIMESTEP_FACTOR,
            "maximum_rungs": int(args.maximum_rungs),
            "each_rung_restarts_from_identical_checkpoint": True,
            "one_full_versus_two_half_steps": True,
            "backward_euler_order": 1,
            "cooling_inner_cutoff_rg": COOLING_INNER_CUTOFF_RG,
            "stream_role": (
                "exact circularized regression stream; not ballistic "
                "Layer-1 calibration"
            ),
            "audit_only_change_bounds": {
                "maximum_scaled_primitive_change": 0.2,
                "maximum_scaled_total_change": 0.25,
                "role": (
                    "nonphysical emergency bounds; they do not define "
                    "the accepted timestep ceiling"
                ),
            },
            "no_n32_run": n_cells == 16,
            "no_n64_n128_run": True,
            "no_physics_change": True,
        },
        "step_config": asdict(config),
        "temporal_accuracy_gates": TEMPORAL_ACCURACY_GATES,
        "initial_state_gates": initial_gates,
        "initial_observables": _observable_dict(initial_observables),
        "local_timescales": clock_summary,
        "rungs": rungs,
        "result": {
            "largest_passing_timestep_seconds": largest_passed,
            "first_failing_timestep_seconds": first_failed,
            "first_failure_class": first_failure_class,
            "classification": classification,
            "shortest_physical_clock_seconds": shortest_clock,
            "largest_passing_over_shortest_physical_clock": (
                None
                if largest_passed is None
                else largest_passed / shortest_clock
            ),
            "first_failing_over_shortest_physical_clock": (
                None
                if first_failed is None
                else first_failed / shortest_clock
            ),
            "current_controller_dt_over_shortest_physical_clock": (
                restart.previous_dt / shortest_clock
            ),
            "ceiling_gain_over_current_controller_dt": (
                None
                if largest_passed is None
                else largest_passed / restart.previous_dt
            ),
            "individual_mesh_passed": individual_passed,
            "passed": passed,
        },
        "mesh_comparison": mesh_comparison,
        "authorization": {
            "n16_timescale_and_timestep_ceiling_certified": (
                bool(
                    passed
                    if n_cells == 16
                    else mesh_comparison is not None
                    and mesh_comparison["passed"]
                )
            ),
            "n32_timestep_audit_authorized": (
                passed if n_cells == 16 else True
            ),
            "n32_timestep_audit_performed": n_cells == 32,
            "n32_timescale_and_timestep_ceiling_certified": (
                passed if n_cells == 32 else False
            ),
            "production_temporal_controller_contract_authorized": (
                bool(
                    n_cells == 32
                    and mesh_comparison is not None
                    and mesh_comparison["controller_contract"][
                        "authorized"
                    ]
                )
            ),
            "production_temporal_controller_implemented": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_certified": False,
            "hot_state_certified": False,
            "limit_cycle_certified": False,
        },
        "decision": (
            (
                "bounded_n16_temporal_ceiling_bracketed"
                if n_cells == 16
                else "mesh_supported_temporal_controller_contract"
            )
            if passed
            else f"stop_after_bounded_n{n_cells}_temporal_audit"
        ),
    }
    output_path = (
        _default_output(n_cells)
        if args.output is None
        else _absolute(args.output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
