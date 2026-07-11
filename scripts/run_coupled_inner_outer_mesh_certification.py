"""Prolongate the fully coupled 40 rg root through its mesh gate."""

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
    evaluate_coupled_inner_outer_residual,
    interpolate_coupled_state_components,
    pack_coupled_state,
    select_sonic_compatibility_pivot,
    solve_coupled_inner_outer_steady,
)
from imri_qpe.scales import eddington_luminosity

from run_common_stress_interface_sweep import _build_case
from run_coupled_inner_outer_rank_prototype import (
    TARGET_RG,
    _jsonable,
    _rank_summary,
    _stage_summary,
    build_coupled_case,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CASE = ROOT / "results/canonical/coupled_inner_outer_rank_prototype"
SOURCE_STATE = SOURCE_CASE / "state.npz"
SOURCE_SUMMARY = SOURCE_CASE / "summary.json"
OUTPUT = ROOT / "outputs/tables/coupled_inner_outer_mesh_certification.json"
CHECKPOINTS = ROOT / "outputs/checkpoints/coupled_inner_outer_mesh_certification"
SOURCE_MESH = (96, 64)
TARGET_MESHES = ((144, 96), (192, 128))


def _load_source():
    _canonical, _index, _radius, context, _seed = build_coupled_case(
        *SOURCE_MESH
    )
    context = replace(context, coupling_fraction=1.0)
    with np.load(SOURCE_STATE, allow_pickle=False) as data:
        state = np.asarray(data["state"], dtype=float)
    evaluation = evaluate_coupled_inner_outer_residual(state, context)
    maximum = float(np.max(np.abs(evaluation.residual)))
    if maximum > 1.0e-7:
        raise RuntimeError(
            f"canonical coupled source no longer closes: {maximum:.3e}"
        )
    return context, state


def _target_context_and_state(
    source_context: CoupledInnerOuterContext,
    source_state,
    n_inner: int,
    n_outer: int,
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
    ) = _build_case(TARGET_RG, n_outer)
    target_inner_params = replace(
        source_context.inner_params,
        n_nodes=int(n_inner),
        custom_grid_xi=None,
    )
    if not np.isclose(
        interface_radius,
        target_inner_params.R_out,
        rtol=2.0e-13,
    ):
        raise RuntimeError("target inner and outer interface radii differ")
    (
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
    ) = interpolate_coupled_state_components(
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


def _result_summary(result, context, rank):
    stage = _stage_summary(result, context)
    inner_luminosity = float(
        np.trapezoid(
            2.0 * np.pi
            * result.evaluation.inner_profile.R
            * result.evaluation.inner_profile.Q_rad,
            result.evaluation.inner_profile.R,
        )
        / eddington_luminosity(context.inner_params.M2_g)
    )
    outer_luminosity = float(
        np.sum(
            result.evaluation.outer_energy_profile.radiative_loss_rate_cells
        )
        / eddington_luminosity(context.inner_params.M2_g)
    )
    inner_radius_rg = (
        result.evaluation.inner_profile.R / context.inner_params.r_g
    )
    outer_radius_rg = context.outer_grid.centers / context.inner_params.r_g
    composite_radius_rg = np.concatenate((inner_radius_rg, outer_radius_rg))
    composite_H_over_R = np.concatenate(
        (
            result.evaluation.inner_profile.H
            / result.evaluation.inner_profile.R,
            result.evaluation.outer_energy_profile.H
            / context.outer_grid.centers,
        )
    )
    common_band = composite_radius_rg >= 60.0
    fixed_radius_H_over_R = {
        f"{radius_rg:g}": float(
            np.interp(
                np.log(radius_rg),
                np.log(composite_radius_rg),
                composite_H_over_R,
            )
        )
        for radius_rg in (60.0, 100.0, 200.0, 240.0)
    }
    full_rank = (
        rank.ranks_by_relative_threshold["1e-10"] == result.state.size
    )
    rank_gate = bool(
        full_rank
        and rank.preboundary_nullity == 2
        and rank.interface_response_rank == 2
        and rank.sonic_rank == 2
    )
    primitive_gate = bool(
        max(abs(value) for value in stage["primitive_audits"].values())
        <= 0.01
    )
    return {
        "n_inner": context.inner_params.n_nodes,
        "n_outer": context.outer_grid.centers.size,
        "unknown_count": result.state.size,
        "accepted": result.accepted,
        "maximum_residual": result.maximum_residual,
        "continuity_residual": stage["continuity_residual"],
        "primitive_audits": stage["primitive_audits"],
        "lambda0": stage["lambda0"],
        "sonic_radius_rg": stage["sonic_radius_rg"],
        "inner_Lrad_over_LEdd": inner_luminosity,
        "outer_Lrad_over_LEdd": outer_luminosity,
        "composite_Lrad_over_LEdd": inner_luminosity + outer_luminosity,
        "max_outer_H_over_R": float(
            np.max(
                result.evaluation.outer_energy_profile.H
                / context.outer_grid.centers
            )
        ),
        "max_composite_H_over_R": float(np.max(composite_H_over_R)),
        "max_common_band_H_over_R": float(
            np.max(composite_H_over_R[common_band])
        ),
        "fixed_radius_H_over_R": fixed_radius_H_over_R,
        "rank_gate": rank_gate,
        "primitive_gate": primitive_gate,
        "rank_audit": _rank_summary(rank),
    }


def run():
    source_context, source_state = _load_source()
    source_summary = json.loads(SOURCE_SUMMARY.read_text())
    source_final = source_summary["stages"][-1]
    rows = [
        {
            "n_inner": SOURCE_MESH[0],
            "n_outer": SOURCE_MESH[1],
            "unknown_count": source_summary["unknown_count"],
            "accepted": source_final["accepted"],
            "maximum_residual": source_final["maximum_residual"],
            "continuity_residual": source_final["continuity_residual"],
            "primitive_audits": source_final["primitive_audits"],
            "lambda0": source_final["lambda0"],
            "sonic_radius_rg": source_final["sonic_radius_rg"],
            "inner_Lrad_over_LEdd": source_summary["inner_Lrad_over_LEdd"],
            "outer_Lrad_over_LEdd": source_summary["outer_Lrad_over_LEdd"],
            "composite_Lrad_over_LEdd": source_summary[
                "composite_Lrad_over_LEdd"
            ],
            "max_outer_H_over_R": source_summary["max_outer_H_over_R"],
            "rank_gate": True,
            "primitive_gate": True,
            "rank_audit": source_summary["rank_audits"]["mu_1"],
        }
    ]
    state = source_state
    context = source_context
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    for n_inner, n_outer in TARGET_MESHES:
        context, initial = _target_context_and_state(
            context,
            state,
            n_inner,
            n_outer,
        )
        result = solve_coupled_inner_outer_steady(
            initial,
            context,
            tolerance=1.0e-7,
            max_nfev=100,
        )
        if not result.accepted:
            raise RuntimeError(
                f"coupled mesh {n_inner}/{n_outer} failed: "
                f"{result.maximum_residual:.3e} ({result.message})"
            )
        rank = audit_coupled_rank(result.state, context)
        row = _result_summary(result, context, rank)
        rows.append(row)
        np.savez_compressed(
            CHECKPOINTS / f"Ninner{n_inner}_Nouter{n_outer}.npz",
            state=result.state,
            inner_radius=result.evaluation.inner_profile.R,
            outer_radius=context.outer_grid.centers,
            n_inner=n_inner,
            n_outer=n_outer,
        )
        if not row["rank_gate"] or not row["primitive_gate"]:
            break
        state = result.state

    finest = rows[-1]
    comparator = rows[-2]
    luminosity_shift = abs(
        finest["composite_Lrad_over_LEdd"]
        / comparator["composite_Lrad_over_LEdd"]
        - 1.0
    )
    thickness_shift = abs(
        finest["max_outer_H_over_R"]
        / comparator["max_outer_H_over_R"]
        - 1.0
    )
    mesh_gate = bool(
        len(rows) == 1 + len(TARGET_MESHES)
        and all(row["accepted"] for row in rows)
        and all(row["rank_gate"] for row in rows)
        and all(row["primitive_gate"] for row in rows)
        and luminosity_shift <= 0.01
        and thickness_shift <= 0.02
    )
    output = {
        "interface_rg": source_summary["actual_interface_rg"],
        "source_checkpoint": SOURCE_STATE.relative_to(ROOT).as_posix(),
        "restart_policy": "prolongate_previous_full_mu1_root",
        "meshes_requested": [list(SOURCE_MESH), *map(list, TARGET_MESHES)],
        "rows": rows,
        "finest_pair_relative_luminosity_shift": luminosity_shift,
        "finest_pair_relative_max_H_over_R_shift": thickness_shift,
        "mesh_certification_gate": mesh_gate,
        "next_stage": (
            "interface_radius_continuation"
            if mesh_gate
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
