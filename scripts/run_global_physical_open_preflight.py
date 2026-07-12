"""Map the canonical physical open control into the one-domain solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalConservativeState,
    PaczynskiWiitaPotential,
    advance_global_backward_euler,
    evaluate_global_rusanov_profile,
    evaluate_coupled_open_overflow_residual,
    global_compact_stream_cell_sources,
    global_radiative_cooling_rate_cells,
    make_log_grid,
    recover_global_primitives,
    state_from_thermodynamic_primitives,
    vertical_state,
)
from imri_qpe.scales import eddington_luminosity, eddington_mdot

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import _open_context, _target_mesh


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"
OUTPUT = ROOT / "outputs/tables/global_physical_open_preflight.json"
DT_LOADING_FRACTION = 1.0e-9


def _canonical_open_evaluation():
    base, _wall_state = _load_source()
    context_96 = _open_context(base, 1.0)
    with np.load(CANONICAL / "Ninner96_Nouter64.npz") as data:
        state_96 = np.asarray(data["state"], dtype=float)
    context_144, _seed = _target_mesh(context_96, state_96, 144, 96)
    with np.load(CANONICAL / "Ninner144_Nouter96.npz") as data:
        state_144 = np.asarray(data["state"], dtype=float)
    return context_144, evaluate_coupled_open_overflow_residual(
        state_144, context_144
    )


def _mapped_global_state(context, evaluation, n_cells: int):
    inner = evaluation.base.inner_profile
    outer = evaluation.base.outer_transport
    energy = evaluation.base.outer_energy_profile
    outer_grid = context.base.outer_grid
    radius = np.concatenate((inner.R, outer_grid.centers))
    sigma = np.concatenate((inner.Sigma, outer.surface_density))
    radial_velocity = np.concatenate((-inner.u, energy.radial_velocity))
    omega = np.concatenate((inner.Omega, outer.omega))
    temperature = np.concatenate((inner.T, energy.temperature))
    grid = make_log_grid(inner.R[0], outer_grid.edges[-1], n_cells)
    log_radius = np.log(radius)
    mapped_sigma = np.exp(
        np.interp(np.log(grid.centers), log_radius, np.log(sigma))
    )
    mapped_temperature = np.exp(
        np.interp(np.log(grid.centers), log_radius, np.log(temperature))
    )
    mapped_omega = np.exp(
        np.interp(np.log(grid.centers), log_radius, np.log(omega))
    )
    mapped_velocity = np.interp(
        np.log(grid.centers), log_radius, radial_velocity
    )
    state = state_from_thermodynamic_primitives(
        grid,
        mapped_sigma,
        mapped_velocity,
        mapped_omega,
        mapped_temperature,
        context.base.inner_params.M2_g,
    )
    return grid, state


def _extrapolating_interpolation(query, nodes, values, *, positive: bool):
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    query = np.asarray(query, dtype=float)
    work = np.log(values) if positive else values
    result = np.interp(query, nodes, work)
    left = query < nodes[0]
    right = query > nodes[-1]
    if np.any(left):
        slope = (work[1] - work[0]) / (nodes[1] - nodes[0])
        result[left] = work[0] + slope * (query[left] - nodes[0])
    if np.any(right):
        slope = (work[-1] - work[-2]) / (nodes[-1] - nodes[-2])
        result[right] = work[-1] + slope * (query[right] - nodes[-1])
    return np.exp(result) if positive else result


def _conservatively_mapped_global_state(
    context, evaluation, n_cells: int, *, quadrature_order: int = 8
):
    inner = evaluation.base.inner_profile
    outer = evaluation.base.outer_transport
    energy = evaluation.base.outer_energy_profile
    outer_grid = context.base.outer_grid
    radius = np.concatenate((inner.R, outer_grid.centers))
    log_radius = np.log(radius)
    sigma = np.concatenate((inner.Sigma, outer.surface_density))
    radial_velocity = np.concatenate((-inner.u, energy.radial_velocity))
    omega = np.concatenate((inner.Omega, outer.omega))
    temperature = np.concatenate((inner.T, energy.temperature))
    grid = make_log_grid(inner.R[0], outer_grid.edges[-1], n_cells)
    mass = context.base.inner_params.M2_g
    potential = PaczynskiWiitaPotential(mass)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    components = [np.empty(n_cells, dtype=float) for _ in range(4)]
    for index, (left, right) in enumerate(
        zip(np.log(grid.edges[:-1]), np.log(grid.edges[1:]))
    ):
        x = 0.5 * (left + right) + 0.5 * (right - left) * nodes
        r = np.exp(x)
        sigma_q = _extrapolating_interpolation(
            x, log_radius, sigma, positive=True
        )
        velocity_q = _extrapolating_interpolation(
            x, log_radius, radial_velocity, positive=False
        )
        omega_q = _extrapolating_interpolation(
            x, log_radius, omega, positive=True
        )
        temperature_q = _extrapolating_interpolation(
            x, log_radius, temperature, positive=True
        )
        vertical = vertical_state(
            sigma_q, temperature_q, r, potential
        )
        specific_total = (
            np.asarray(potential.phi(r), dtype=float)
            + 0.5 * velocity_q**2
            + 0.5 * (r * omega_q) ** 2
            + np.asarray(vertical.e, dtype=float)
        )
        measure = 2.0 * np.pi * r**2 * sigma_q
        integrands = (
            measure,
            measure * velocity_q,
            measure * r**2 * omega_q,
            measure * specific_total,
        )
        for component, integrand in zip(components, integrands):
            component[index] = (
                0.5 * (right - left) * np.sum(weights * integrand)
            )
    return grid, GlobalConservativeState(*components).validated()


def _initial_case(
    context, evaluation, n_cells: int, *, mapping_mode: str = "conservative"
):
    if mapping_mode == "conservative":
        grid, initial = _conservatively_mapped_global_state(
            context, evaluation, n_cells
        )
    elif mapping_mode == "pointwise":
        grid, initial = _mapped_global_state(context, evaluation, n_cells)
    else:
        raise ValueError("mapping_mode must be conservative or pointwise")
    mass = context.base.inner_params.M2_g
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0 * eddington_mdot(mass)
    circularization_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(circularization_radius))
    stream_energy = float(
        potential.phi(circularization_radius)
        + 0.5 * (stream_l / circularization_radius) ** 2
    )
    stream = global_compact_stream_cell_sources(
        grid,
        stream_rate,
        center=240.0 * potential.r_g,
        log_width=0.08,
        specific_radial_velocity=0.0,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_energy,
    )
    initial_primitives = recover_global_primitives(grid, initial, mass)
    profile_without_stream = evaluate_global_rusanov_profile(
        grid,
        initial,
        mass,
        reference_state=initial,
        boundary_mode="open_no_inflow",
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
    )
    profile_with_stream = evaluate_global_rusanov_profile(
        grid,
        initial,
        mass,
        reference_state=initial,
        boundary_mode="open_no_inflow",
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        external_sources=stream,
    )
    mapping = {
        "mapping_mode": mapping_mode,
        "n_cells": n_cells,
        "active_source_cells": int(np.count_nonzero(stream.mass)),
        "inner_mass_flux_over_supply": float(
            profile_with_stream.face_fluxes.mass[0] / stream_rate
        ),
        "outer_mass_flux_over_supply": float(
            profile_with_stream.face_fluxes.mass[-1] / stream_rate
        ),
        "maximum_face_mass_flux_change_from_stream_source": float(
            np.max(
                np.abs(
                    profile_with_stream.face_fluxes.mass
                    - profile_without_stream.face_fluxes.mass
                )
            )
            / stream_rate
        ),
        "inner_cell_radial_velocity_c": float(
            initial_primitives.radial_velocity[0] / 2.99792458e10
        ),
        "outer_cell_radial_velocity_c": float(
            initial_primitives.radial_velocity[-1] / 2.99792458e10
        ),
    }
    return (
        grid,
        initial,
        mass,
        potential,
        stream_rate,
        stream,
        initial_primitives,
        mapping,
    )


def _mapping_attempt(
    context, evaluation, n_cells: int, *, mapping_mode: str
):
    try:
        return {
            "accepted": True,
            **_initial_case(
                context,
                evaluation,
                n_cells,
                mapping_mode=mapping_mode,
            )[-1],
        }
    except ValueError as error:
        return {
            "accepted": False,
            "mapping_mode": mapping_mode,
            "n_cells": n_cells,
            "message": str(error),
        }


def _run_case(
    context,
    evaluation,
    n_cells: int,
    n_steps: int,
    *,
    mapping_mode: str = "pointwise",
    jacobian_mode: str = "dense",
):
    (
        grid,
        initial,
        mass,
        potential,
        stream_rate,
        stream,
        initial_primitives,
        mapping,
    ) = _initial_case(
        context, evaluation, n_cells, mapping_mode=mapping_mode
    )
    loading_time = float(np.sum(initial.mass) / stream_rate)
    dt = DT_LOADING_FRACTION * loading_time / n_steps
    current = initial
    results = []
    for _ in range(n_steps):
        result = advance_global_backward_euler(
            grid,
            current,
            mass,
            dt,
            alpha=context.base.alpha,
            reference_state=initial,
            boundary_mode="open_no_inflow",
            stress_boundary_mode="outer_zero_torque",
            include_radiative_cooling=True,
            external_sources=stream,
            jacobian_mode=jacobian_mode,
            max_nfev=300,
        )
        results.append(result)
        if not result.accepted:
            break
        current = result.state
    final_result = results[-1]
    final = recover_global_primitives(grid, final_result.state, mass)
    return {
        "initial": {
            **mapping,
            "inner_radius_rg": float(grid.edges[0] / potential.r_g),
            "loading_time_seconds": loading_time,
            "maximum_H_over_R": float(
                np.max(np.asarray(initial_primitives.vertical.H) / grid.centers)
            ),
            "radiative_luminosity_over_eddington": float(
                np.sum(
                    global_radiative_cooling_rate_cells(
                        grid, initial_primitives
                    )
                )
                / eddington_luminosity(mass)
            ),
        },
        "step": {
            "dt_over_loading_time": DT_LOADING_FRACTION,
            "dt_seconds": dt,
            "steps_attempted": n_steps,
            "steps_accepted": sum(result.accepted for result in results),
            "all_steps_accepted": len(results) == n_steps
            and all(result.accepted for result in results),
            "maximum_nfev": max(result.nfev for result in results),
            "messages": [result.message for result in results],
            "jacobian_audits": [
                None
                if result.jacobian_audit is None
                else {
                    "directions": result.jacobian_audit.directions,
                    "pattern_nonzeros": result.jacobian_audit.pattern_nonzeros,
                    "maximum_absolute_defect": (
                        result.jacobian_audit.maximum_absolute_defect
                    ),
                    "maximum_relative_defect": (
                        result.jacobian_audit.maximum_relative_defect
                    ),
                    "accepted": result.jacobian_audit.accepted,
                }
                for result in results
            ],
            "maximum_scaled_residual": max(
                result.maximum_scaled_residual for result in results
            ),
            "maximum_storage_scaled_ledger_defect": max(
                result.maximum_storage_scaled_ledger_defect
                for result in results
            ),
            "minimum_surface_density": float(
                np.min(final.surface_density)
            ),
            "minimum_temperature": float(np.min(final.temperature)),
            "maximum_H_over_R": float(
                np.max(np.asarray(final.vertical.H) / grid.centers)
            ),
            "inner_mass_flux_over_supply": float(
                final_result.profile.face_fluxes.mass[0] / stream_rate
            ),
            "outer_mass_flux_over_supply": float(
                final_result.profile.face_fluxes.mass[-1] / stream_rate
            ),
        },
    }


def main() -> None:
    context, evaluation = _canonical_open_evaluation()
    potential = PaczynskiWiitaPotential(context.base.inner_params.M2_g)
    circularization_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(circularization_radius))
    stream_energy = float(
        potential.phi(circularization_radius)
        + 0.5 * (stream_l / circularization_radius) ** 2
    )
    coarse = _run_case(context, evaluation, 16, 1)
    temporal = _run_case(context, evaluation, 16, 2)
    medium = _run_case(context, evaluation, 24, 2)
    refined = _run_case(context, evaluation, 32, 2)
    pointwise_mapping_only = [
        _mapping_attempt(
            context, evaluation, n_cells, mapping_mode="pointwise"
        )
        for n_cells in (16, 24, 32, 48)
    ]
    conservative_mapping_only = [
        _mapping_attempt(
            context, evaluation, n_cells, mapping_mode="conservative"
        )
        for n_cells in (16, 24, 32, 48, 64, 96, 128)
    ]
    conservative_step = _run_case(
        context,
        evaluation,
        64,
        1,
        mapping_mode="conservative",
    )
    conservative_temporal = _run_case(
        context,
        evaluation,
        64,
        2,
        mapping_mode="conservative",
    )
    sparse_64 = _run_case(
        context,
        evaluation,
        64,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
    )
    sparse_96 = _run_case(
        context,
        evaluation,
        96,
        1,
        mapping_mode="conservative",
        jacobian_mode="sparse_forward",
    )
    target_inner = float(
        -evaluation.base.outer_transport.mdot_faces[0]
        / (5.0 * eddington_mdot(context.base.inner_params.M2_g))
    )
    target_outer = float(
        -evaluation.base.outer_transport.mdot_faces[-1]
        / (5.0 * eddington_mdot(context.base.inner_params.M2_g))
    )
    report = {
        "input_closure": {
            "M2_g": context.base.inner_params.M2_g,
            "stream_rate_over_eddington": 5.0,
            "source_center_rg": 240.0,
            "source_log_width": 0.08,
            "circularization_radius_rg": 248.96693,
            "specific_radial_velocity": 0.0,
            "specific_angular_momentum": stream_l,
            "specific_total_energy": stream_energy,
            "alpha": context.base.alpha,
            "outer_radius_rg": 335.0,
            "target_inner_outward_mass_flux_over_supply": target_inner,
            "target_outer_outward_mass_flux_over_supply": target_outer,
        },
        "runs": [coarse, temporal, medium, refined],
        "pointwise_mapping_only_meshes": pointwise_mapping_only,
        "conservative_mapping_only_meshes": conservative_mapping_only,
        "conservative_N64_step": conservative_step,
        "conservative_N64_two_half_steps": conservative_temporal,
        "conservative_N64_temporal_comparison": {
            "inner_flux_difference": float(
                conservative_temporal["step"][
                    "inner_mass_flux_over_supply"
                ]
                - conservative_step["step"][
                    "inner_mass_flux_over_supply"
                ]
            ),
            "outer_flux_difference": float(
                conservative_temporal["step"][
                    "outer_mass_flux_over_supply"
                ]
                - conservative_step["step"][
                    "outer_mass_flux_over_supply"
                ]
            ),
            "maximum_H_over_R_relative_difference": float(
                conservative_temporal["step"]["maximum_H_over_R"]
                / conservative_step["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
        "sparse_evolved_mesh_runs": [sparse_64, sparse_96],
        "sparse_evolved_N64_N96_comparison": {
            "inner_flux_difference_over_supply": float(
                sparse_96["step"]["inner_mass_flux_over_supply"]
                - sparse_64["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference_over_supply": float(
                sparse_96["step"]["outer_mass_flux_over_supply"]
                - sparse_64["step"]["outer_mass_flux_over_supply"]
            ),
            "outer_flux_relative_difference": float(
                sparse_96["step"]["outer_mass_flux_over_supply"]
                / sparse_64["step"]["outer_mass_flux_over_supply"]
                - 1.0
            ),
            "maximum_H_over_R_relative_difference": float(
                sparse_96["step"]["maximum_H_over_R"]
                / sparse_64["step"]["maximum_H_over_R"]
                - 1.0
            ),
            "flux_mesh_gate_pass": bool(
                abs(
                    sparse_96["step"]["outer_mass_flux_over_supply"]
                    - sparse_64["step"]["outer_mass_flux_over_supply"]
                )
                <= 0.01
            ),
        },
        "mapping_only_conclusion": (
            "Pointwise coarse mapping creates the boundary-flux sensitivity. "
            "Strict conservative remapping rejects N<=48 on thermal primitive "
            "recovery, while N64 and N96 are admissible and approach the "
            "canonical coupled boundary fluxes."
        ),
        "temporal_comparison_N16": {
            "inner_flux_difference": float(
                temporal["step"]["inner_mass_flux_over_supply"]
                - coarse["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference": float(
                temporal["step"]["outer_mass_flux_over_supply"]
                - coarse["step"]["outer_mass_flux_over_supply"]
            ),
            "maximum_H_over_R_relative_difference": float(
                temporal["step"]["maximum_H_over_R"]
                / coarse["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
        "mesh_comparison_N16_N24_two_steps": {
            "inner_flux_difference": float(
                medium["step"]["inner_mass_flux_over_supply"]
                - temporal["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference": float(
                medium["step"]["outer_mass_flux_over_supply"]
                - temporal["step"]["outer_mass_flux_over_supply"]
            ),
            "maximum_H_over_R_relative_difference": float(
                medium["step"]["maximum_H_over_R"]
                / temporal["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
        "mesh_comparison_N24_N32_two_steps": {
            "inner_flux_difference": float(
                refined["step"]["inner_mass_flux_over_supply"]
                - medium["step"]["inner_mass_flux_over_supply"]
            ),
            "outer_flux_difference": float(
                refined["step"]["outer_mass_flux_over_supply"]
                - medium["step"]["outer_mass_flux_over_supply"]
            ),
            "maximum_H_over_R_relative_difference": float(
                refined["step"]["maximum_H_over_R"]
                / medium["step"]["maximum_H_over_R"]
                - 1.0
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
