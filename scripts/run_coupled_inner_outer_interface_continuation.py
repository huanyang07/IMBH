"""Continue the certified coupled root across interface radius controls."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    ConservedInterfaceFlux,
    CoupledInnerOuterContext,
    audit_coupled_rank,
    build_nonkeplerian_residual_scales,
    interpolate_coupled_state_across_interface,
    pack_coupled_state,
    select_sonic_compatibility_pivot,
    solve_coupled_inner_outer_steady,
)

from run_common_stress_interface_sweep import _build_case
from run_coupled_inner_outer_mesh_certification import (
    CHECKPOINTS as MESH_CHECKPOINTS,
    OUTPUT as MESH_OUTPUT,
    SOURCE_MESH,
    TARGET_MESHES,
    _load_source,
    _result_summary,
    _target_context_and_state,
)
from run_coupled_inner_outer_rank_prototype import TARGET_RG, _jsonable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/coupled_inner_outer_interface_continuation.json"
CHECKPOINTS = ROOT / "outputs/checkpoints/coupled_inner_outer_interface_continuation"
N_INNER = TARGET_MESHES[-1][0]
N_OUTER = TARGET_MESHES[-1][1]
INWARD_TARGET = 35.0
OUTWARD_TARGETS = (45.0, 50.0)


def _load_finest_mesh():
    context, state = _load_source()
    for n_inner, n_outer in TARGET_MESHES:
        context, _initial = _target_context_and_state(
            context,
            state,
            n_inner,
            n_outer,
        )
        checkpoint = (
            MESH_CHECKPOINTS / f"Ninner{n_inner}_Nouter{n_outer}.npz"
        )
        with np.load(checkpoint, allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=float)
    return context, state


def _target_radius_context_and_state(
    source_context: CoupledInnerOuterContext,
    source_state,
    target_rg: float,
):
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
    ) = _build_case(target_rg, N_OUTER)
    target_inner_params = replace(
        source_context.inner_params,
        R_out_rg=interface_radius / source_context.inner_params.r_g,
        n_nodes=N_INNER,
        custom_grid_xi=None,
    )
    (
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
    ) = interpolate_coupled_state_across_interface(
        source_state,
        source_context,
        target_inner_params,
        outer_grid,
    )
    interface_flux = ConservedInterfaceFlux(
        mdot=target_inner_params.Mdot_g_s,
        angular_momentum=interface_angular,
        total_energy=interface_energy,
    )
    scales = build_nonkeplerian_residual_scales(
        outer_grid,
        baseline.transport,
        sigma,
        temperature,
        omega,
        params.M2_g,
        closure=closure,
        prescribed_inner_flux=interface_flux,
    )
    pivot = select_sonic_compatibility_pivot(
        inner_state,
        target_inner_params,
    )
    context = CoupledInnerOuterContext(
        inner_params=target_inner_params,
        outer_grid=outer_grid,
        outer_template=baseline.transport,
        outer_closure=closure,
        outer_scales=scales,
        anchor_log_surface_density=(
            source_context.anchor_log_surface_density
        ),
        anchor_log_temperature=source_context.anchor_log_temperature,
        reference_log_surface_density_jump=(
            source_context.reference_log_surface_density_jump
        ),
        reference_log_temperature_jump=(
            source_context.reference_log_temperature_jump
        ),
        angular_flux_scale=source_context.angular_flux_scale,
        energy_flux_scale=source_context.energy_flux_scale,
        coupling_fraction=1.0,
        sonic_pivot=pivot,
        alpha=source_context.alpha,
        mu_stress=source_context.mu_stress,
        stress_factor=source_context.stress_factor,
    )
    state = pack_coupled_state(
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
        context,
    )
    return context, state


def _solve_target(source_context, source_state, target_rg: float):
    context, initial = _target_radius_context_and_state(
        source_context,
        source_state,
        target_rg,
    )
    result = solve_coupled_inner_outer_steady(
        initial,
        context,
        tolerance=1.0e-7,
        max_nfev=100,
    )
    if not result.accepted:
        raise RuntimeError(
            f"interface target {target_rg:g} rg failed: "
            f"{result.maximum_residual:.3e} ({result.message})"
        )
    rank = audit_coupled_rank(result.state, context)
    row = _result_summary(result, context, rank)
    row["target_interface_rg"] = target_rg
    row["actual_interface_rg"] = context.inner_params.R_out_rg
    row["source_interface_rg"] = source_context.inner_params.R_out_rg
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINTS / f"R{target_rg:g}_Ninner{N_INNER}_Nouter{N_OUTER}.npz",
        state=result.state,
        inner_radius=result.evaluation.inner_profile.R,
        outer_radius=context.outer_grid.centers,
        target_interface_rg=target_rg,
        actual_interface_rg=context.inner_params.R_out_rg,
        n_inner=N_INNER,
        n_outer=N_OUTER,
    )
    return context, result.state, row


def run():
    mesh_summary = json.loads(MESH_OUTPUT.read_text())
    if not mesh_summary["mesh_certification_gate"]:
        raise RuntimeError("mesh certification must pass before radius continuation")
    base_context, base_state = _load_finest_mesh()
    base_row = dict(mesh_summary["rows"][-1])
    base_row.update(
        {
            "target_interface_rg": TARGET_RG,
            "actual_interface_rg": base_context.inner_params.R_out_rg,
            "source_interface_rg": None,
        }
    )

    _inward_context, _inward_state, inward_row = _solve_target(
        base_context,
        base_state,
        INWARD_TARGET,
    )
    outward_rows = []
    outward_context = base_context
    outward_state = base_state
    for target_rg in OUTWARD_TARGETS:
        outward_context, outward_state, row = _solve_target(
            outward_context,
            outward_state,
            target_rg,
        )
        outward_rows.append(row)

    rows = [inward_row, base_row, *outward_rows]
    luminosities = np.asarray(
        [row["composite_Lrad_over_LEdd"] for row in rows],
        dtype=float,
    )
    raw_outer_thicknesses = np.asarray(
        [row["max_outer_H_over_R"] for row in rows],
        dtype=float,
    )
    common_band_thicknesses = np.asarray(
        [row["max_common_band_H_over_R"] for row in rows],
        dtype=float,
    )
    luminosity_spread = float(
        (np.max(luminosities) - np.min(luminosities))
        / np.mean(luminosities)
    )
    raw_outer_thickness_spread = float(
        (np.max(raw_outer_thicknesses) - np.min(raw_outer_thicknesses))
        / np.mean(raw_outer_thicknesses)
    )
    common_band_thickness_spread = float(
        (np.max(common_band_thicknesses) - np.min(common_band_thicknesses))
        / np.mean(common_band_thicknesses)
    )
    gate = bool(
        all(row["accepted"] for row in rows)
        and all(row["rank_gate"] for row in rows)
        and all(row["primitive_gate"] for row in rows)
        and luminosity_spread <= 0.01
        and common_band_thickness_spread <= 0.02
    )
    output = {
        "mesh": [N_INNER, N_OUTER],
        "targets_requested_rg": [INWARD_TARGET, TARGET_RG, *OUTWARD_TARGETS],
        "continuation_policy": (
            "fork inward from 40; continue outward 40 to 45 to 50"
        ),
        "rows": rows,
        "relative_composite_luminosity_spread": luminosity_spread,
        "relative_max_outer_H_over_R_spread": raw_outer_thickness_spread,
        "relative_max_common_band_H_over_R_spread": (
            common_band_thickness_spread
        ),
        "thickness_invariance_metric": "maximum H/R over fixed R >= 60 rg band",
        "interface_position_gate": gate,
        "next_stage": (
            "physical_tidal_torque_and_power"
            if gate
            else "global_signed_transonic_fallback"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    print(json.dumps(_jsonable(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
