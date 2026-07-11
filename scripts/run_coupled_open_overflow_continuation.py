"""Continue the fully coupled finite minidisk from a mass wall to overflow."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    ConservedInterfaceFlux,
    CoupledInnerOuterContext,
    CoupledOpenOverflowContext,
    audit_coupled_open_rank,
    build_nonkeplerian_residual_scales,
    evaluate_coupled_open_overflow_residual,
    hill_outer_torque_weights,
    fiducial_hill_tidal_geometry,
    interpolate_coupled_state_components,
    pack_coupled_open_state,
    pack_coupled_state,
    select_sonic_compatibility_pivot,
    solve_coupled_open_overflow_steady,
    unpack_coupled_open_state,
)
from imri_qpe.scales import eddington_luminosity

from run_common_stress_interface_sweep import _build_case
from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_inner_outer_rank_prototype import _jsonable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/coupled_open_overflow_continuation.json"
CHECKPOINTS = ROOT / "outputs/checkpoints/coupled_open_overflow_continuation"
BOUNDARY_STAGES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
OPEN_MESH_SEQUENCE = ((144, 96), (168, 112), (192, 128))


def _open_context(base: CoupledInnerOuterContext, fraction: float):
    mass_scale = float(np.sum(base.outer_template.source_mass_rate_cells))
    torque_scale = max(
        abs(float(base.outer_template.viscous_torque_faces[-1])),
        float(np.sum(np.abs(base.outer_template.source_angular_rate_cells))),
        1.0,
    )
    return CoupledOpenOverflowContext(
        base=base,
        boundary_fraction=float(fraction),
        mass_flux_scale=mass_scale,
        torque_scale=torque_scale,
    )


def _rank_summary(audit):
    values = asdict(audit)
    singular = np.asarray(values.pop("singular_values"), dtype=float)
    values["largest_singular_value"] = float(singular[0])
    values["smallest_six_singular_values"] = singular[-6:].tolist()
    return _jsonable(values)


def _stagnation_radius_rg(evaluation, context) -> float | None:
    mdot = np.asarray(evaluation.base.outer_transport.mdot_faces, dtype=float)
    edges = np.asarray(context.base.outer_grid.edges, dtype=float)
    crossing = np.flatnonzero(mdot[:-1] * mdot[1:] <= 0.0)
    if crossing.size == 0:
        return None
    index = int(crossing[0])
    left, right = mdot[index], mdot[index + 1]
    if left == right:
        radius = edges[index]
    else:
        weight = -left / (right - left)
        radius = np.exp(
            np.log(edges[index])
            + weight * (np.log(edges[index + 1]) - np.log(edges[index]))
        )
    return float(radius / context.base.inner_params.r_g)


def _result_row(result, context, geometry, weights):
    evaluation = result.evaluation
    inner = evaluation.base.inner_profile
    outer = evaluation.base.outer_energy_profile
    luminosity = float(
        (
            np.trapezoid(2.0 * np.pi * inner.R * inner.Q_rad, inner.R)
            + np.sum(outer.radiative_loss_rate_cells)
        )
        / eddington_luminosity(context.base.inner_params.M2_g)
    )
    H_over_R = outer.H / context.base.outer_grid.centers
    active = weights > 0.0
    mass_scale = context.mass_flux_scale
    base = evaluation.base
    return {
        "boundary_fraction": context.boundary_fraction,
        "accepted": result.accepted,
        "maximum_residual": result.maximum_residual,
        "nfev": result.nfev,
        "message": result.message,
        "mdot_inner_over_stream": evaluation.mdot_inner / mass_scale,
        "mdot_outer_over_stream": evaluation.mdot_outer / mass_scale,
        "overflow_fraction": max(-evaluation.mdot_outer / mass_scale, 0.0),
        "outer_torque_relative": evaluation.outer_torque / context.torque_scale,
        "edge_boundary_residual": evaluation.edge_boundary,
        "sonic_radius_rg": float(
            inner.R[0] / context.base.inner_params.r_g
        ),
        "lambda0": float(inner.lambda0),
        "composite_Lrad_over_LEdd": luminosity,
        "max_outer_H_over_R": float(np.max(H_over_R)),
        "max_tidal_band_H_over_R": float(np.max(H_over_R[active])),
        "stagnation_radius_rg": _stagnation_radius_rg(evaluation, context),
        "block_maximum_residuals": {
            "inner_core": float(np.max(np.abs(base.inner_core))),
            "outer_stress": float(np.max(np.abs(base.outer_stress))),
            "outer_radial": float(np.max(np.abs(base.outer_radial))),
            "outer_energy": float(np.max(np.abs(base.outer_energy))),
            "flux_extraction": float(np.max(np.abs(base.flux_extraction))),
            "interface_boundary": float(
                np.max(np.abs(base.interface_boundary))
            ),
            "edge_boundary": abs(evaluation.edge_boundary),
        },
    }


def _target_mesh(
    source_context: CoupledOpenOverflowContext,
    source_state,
    n_inner: int,
    n_outer: int,
):
    source_base_state, mdot_inner = unpack_coupled_open_state(
        source_state,
        source_context,
    )
    source_trial = evaluate_coupled_open_overflow_residual(
        source_state,
        source_context,
        include_inner_profile=False,
    ).trial_context
    (
        _profile,
        params,
        _potential,
        _index,
        interface_radius,
        _prescribed,
        outer_grid,
        _stream_rate,
        _stream_l,
        closure,
        baseline,
    ) = _build_case(40.0, n_outer)
    target_inner_params = replace(
        source_trial.inner_params,
        n_nodes=int(n_inner),
        custom_grid_xi=None,
    )
    if not np.isclose(
        interface_radius,
        target_inner_params.R_out,
        rtol=2.0e-13,
    ):
        raise RuntimeError("target mesh moved the interface")
    (
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
    ) = interpolate_coupled_state_components(
        source_base_state,
        source_trial,
        target_inner_params,
        outer_grid,
    )
    interface_flux = ConservedInterfaceFlux(
        mdot=mdot_inner,
        angular_momentum=interface_angular,
        total_energy=interface_energy,
    )
    delta = mdot_inner - float(baseline.transport.mdot_faces[0])
    shifted_template = replace(
        baseline.transport,
        mdot_faces=np.asarray(
            baseline.transport.mdot_faces + delta,
            dtype=float,
        ),
    )
    scales = build_nonkeplerian_residual_scales(
        outer_grid,
        shifted_template,
        sigma,
        temperature,
        omega,
        params.M2_g,
        closure=closure,
        prescribed_inner_flux=interface_flux,
    )
    pivot = select_sonic_compatibility_pivot(inner_state, target_inner_params)
    base = CoupledInnerOuterContext(
        inner_params=target_inner_params,
        outer_grid=outer_grid,
        outer_template=baseline.transport,
        outer_closure=closure,
        outer_scales=scales,
        anchor_log_surface_density=(
            source_context.base.anchor_log_surface_density
        ),
        anchor_log_temperature=source_context.base.anchor_log_temperature,
        reference_log_surface_density_jump=(
            source_context.base.reference_log_surface_density_jump
        ),
        reference_log_temperature_jump=(
            source_context.base.reference_log_temperature_jump
        ),
        angular_flux_scale=source_context.base.angular_flux_scale,
        energy_flux_scale=source_context.base.energy_flux_scale,
        coupling_fraction=1.0,
        sonic_pivot=pivot,
        alpha=source_context.base.alpha,
        mu_stress=source_context.base.mu_stress,
        stress_factor=source_context.base.stress_factor,
    )
    base_state = pack_coupled_state(
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
        base,
    )
    context = _open_context(base, 1.0)
    return context, pack_coupled_open_state(base_state, mdot_inner, context)


def run():
    base, base_state = _load_source()
    context = _open_context(base, 0.0)
    state = pack_coupled_open_state(
        base_state,
        float(base.outer_template.mdot_faces[0]),
        context,
    )
    geometry = fiducial_hill_tidal_geometry()
    weights = hill_outer_torque_weights(
        base.outer_grid,
        geometry.hill_radius,
    )
    stages = []
    rank_audits = {}
    result = None
    for fraction in BOUNDARY_STAGES:
        context = replace(context, boundary_fraction=fraction)
        result = solve_coupled_open_overflow_steady(
            state,
            context,
            tolerance=1.0e-7,
            max_nfev=100,
        )
        stages.append(_result_row(result, context, geometry, weights))
        if not result.accepted:
            break
        state = result.state
        if fraction in {0.0, 1.0}:
            rank_audits[f"chi_{fraction:g}"] = _rank_summary(
                audit_coupled_open_rank(state, context)
            )

    mesh_rows = []
    if result is not None and result.accepted and context.boundary_fraction == 1.0:
        coarse_row = dict(stages[-1])
        coarse_row["n_inner"] = base.inner_params.n_nodes
        coarse_row["n_outer"] = base.outer_grid.centers.size
        mesh_rows.append(coarse_row)
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CHECKPOINTS / "Ninner96_Nouter64.npz",
            state=state,
            mdot_inner=result.evaluation.mdot_inner,
        )
        for n_inner, n_outer in OPEN_MESH_SEQUENCE:
            context, initial = _target_mesh(
                context,
                state,
                n_inner,
                n_outer,
            )
            weights = hill_outer_torque_weights(
                context.base.outer_grid,
                geometry.hill_radius,
            )
            result = solve_coupled_open_overflow_steady(
                initial,
                context,
                tolerance=1.0e-7,
                max_nfev=100,
            )
            row = _result_row(result, context, geometry, weights)
            row["n_inner"] = n_inner
            row["n_outer"] = n_outer
            mesh_rows.append(row)
            np.savez_compressed(
                CHECKPOINTS / f"Ninner{n_inner}_Nouter{n_outer}.npz",
                state=result.state,
                mdot_inner=result.evaluation.mdot_inner,
                accepted=result.accepted,
                maximum_residual=result.maximum_residual,
            )
            if not result.accepted:
                break
            state = result.state
            rank_audits[f"mesh_{n_inner}_{n_outer}"] = _rank_summary(
                audit_coupled_open_rank(state, context)
            )

    reached_open = bool(
        stages
        and stages[-1]["boundary_fraction"] == 1.0
        and stages[-1]["accepted"]
    )
    mesh_gate = bool(
        reached_open
        and len(mesh_rows) == 1 + len(OPEN_MESH_SEQUENCE)
        and all(row["accepted"] for row in mesh_rows)
        and all(
            audit["ranks_by_relative_threshold"]["1e-10"]
            == audit["jacobian_shape"][1]
            and audit["preboundary_nullity"] == 2
            and audit["interface_response_rank"] == 2
            and audit["sonic_rank"] == 2
            for key, audit in rank_audits.items()
            if key.startswith("mesh_")
        )
    )
    output = {
        "interface_rg": base.inner_params.R_out_rg,
        "reservoir_outer_radius_rg": (
            base.outer_grid.edges[-1] / base.inner_params.r_g
        ),
        "boundary_stages_requested": list(BOUNDARY_STAGES),
        "open_mesh_sequence_requested": [
            list(mesh) for mesh in OPEN_MESH_SEQUENCE
        ],
        "stages": stages,
        "mesh_rows": mesh_rows,
        "rank_audits": rank_audits,
        "reached_open_boundary": reached_open,
        "mesh_gate": mesh_gate,
        "next_stage": (
            "interface_invariance_and_distributed_tide"
            if mesh_gate
            else "coupled_mass_energy_time_evolution"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    print(json.dumps(_jsonable(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
