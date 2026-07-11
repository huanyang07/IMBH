"""Run the canonical-anchor coupled inner/outer rank prototype at 40 rg."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CoupledInnerOuterContext,
    audit_coupled_rank,
    build_nonkeplerian_residual_scales,
    computational_grid,
    pack_coupled_state,
    pack_state,
    positive_edge_reconstruction,
    select_sonic_compatibility_pivot,
    solve_canonical_anchored_inner,
    solve_coupled_inner_outer_steady,
    solve_nonkeplerian_common_stress_steady,
    transonic_profile_from_state_vector,
    transonic_profile_interface_flux,
    unused_sonic_compatibility,
)
from imri_qpe.scales import eddington_luminosity

from run_common_stress_interface_sweep import _build_case, _load_transonic


ROOT = Path(__file__).resolve().parents[1]
OUTER_SEED = (
    ROOT
    / "outputs/checkpoints/nonkeplerian_common_stress_sweep/"
    "R40_N256.npz"
)
OUTPUT = ROOT / "outputs/tables/coupled_inner_outer_rank_prototype.json"
CHECKPOINT = ROOT / "outputs/checkpoints/coupled_inner_outer_rank_prototype.npz"
TARGET_RG = 40.0
N_INNER = 96
N_OUTER = 64
COUPLING_STAGES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def _positive_interpolate(radius, values, target):
    return np.exp(np.interp(np.log(target), np.log(radius), np.log(values)))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_coupled_case(n_inner: int = N_INNER, n_outer: int = N_OUTER):
    canonical, canonical_params = _load_transonic()
    index = int(
        np.argmin(np.abs(canonical.R / canonical_params.r_g - TARGET_RG))
    )
    interface_radius = float(canonical.R[index])
    interface_rg = interface_radius / canonical_params.r_g
    inner_params = replace(
        canonical_params,
        R_out_rg=interface_rg,
        n_nodes=int(n_inner),
        custom_grid_xi=None,
        R_son_bounds_rg=(2.05, 12.0),
        max_nfev=1,
    )
    inner_log_radius = computational_grid(inner_params, np.log(canonical.R[0]))
    inner_radius = np.exp(inner_log_radius)
    inner_state = pack_state(
        np.interp(
            inner_log_radius,
            np.log(canonical.R[: index + 1]),
            np.log(canonical.u[: index + 1]),
        ),
        np.interp(
            inner_log_radius,
            np.log(canonical.R[: index + 1]),
            np.log(canonical.T[: index + 1]),
        ),
        np.log(canonical.R[0]),
        canonical.lambda0,
    )
    pivot = select_sonic_compatibility_pivot(inner_state, inner_params)
    anchor_log_sigma = float(np.log(canonical.Sigma[index]))
    anchor_log_temperature = float(np.log(canonical.T[index]))
    inner_state, inner_accepted, inner_maximum = solve_canonical_anchored_inner(
        inner_state,
        inner_params,
        anchor_log_sigma,
        anchor_log_temperature,
        sonic_pivot=pivot,
        tolerance=1.0e-7,
        max_nfev=1000,
    )
    if not inner_accepted:
        raise RuntimeError(
            f"canonical inner anchor failed with residual {inner_maximum:.3e}"
        )
    anchored_profile = transonic_profile_from_state_vector(
        inner_state,
        inner_params,
    )
    extracted = transonic_profile_interface_flux(
        anchored_profile,
        inner_params.M2_g,
        inner_params.Mdot_g_s,
        -1,
    )

    (
        _profile,
        params,
        _potential,
        _index,
        outer_interface_radius,
        prescribed,
        outer_grid,
        _stream_rate,
        _stream_l,
        closure,
        baseline,
    ) = _build_case(TARGET_RG, n_outer)
    if not np.isclose(interface_radius, outer_interface_radius, rtol=2.0e-13):
        raise RuntimeError("inner and outer interface radii differ")
    with np.load(OUTER_SEED, allow_pickle=False) as seed:
        seed_radius = np.asarray(seed["radius"], dtype=float)
        sigma = _positive_interpolate(
            seed_radius,
            np.asarray(seed["surface_density"], dtype=float),
            outer_grid.centers,
        )
        temperature = _positive_interpolate(
            seed_radius,
            np.asarray(seed["temperature"], dtype=float),
            outer_grid.centers,
        )
        omega = _positive_interpolate(
            seed_radius,
            np.asarray(seed["omega"], dtype=float),
            outer_grid.centers,
        )
    outer_root = solve_nonkeplerian_common_stress_steady(
        outer_grid,
        baseline.transport,
        sigma,
        temperature,
        omega,
        params.M2_g,
        alpha=0.01,
        closure=closure,
        prescribed_inner_flux=extracted,
        radial_support_fraction=1.0,
        mu_stress=0.0,
        stress_factor=1.0,
        tolerance=1.0e-7,
        max_nfev=2000,
    )
    if not outer_root.accepted:
        raise RuntimeError(
            "outer anchor solve failed: "
            f"{outer_root.maximum_stress_residual:.3e}, "
            f"{outer_root.maximum_radial_residual:.3e}, "
            f"{outer_root.maximum_energy_residual:.3e}"
        )
    sigma = outer_root.surface_density
    temperature = outer_root.temperature
    omega = outer_root.omega
    scales = build_nonkeplerian_residual_scales(
        outer_grid,
        baseline.transport,
        sigma,
        temperature,
        omega,
        params.M2_g,
        closure=closure,
        prescribed_inner_flux=extracted,
    )
    reference_log_surface_density_jump = float(
        np.log(positive_edge_reconstruction(outer_grid, sigma)[0])
        - anchor_log_sigma
    )
    reference_log_temperature_jump = float(
        np.log(positive_edge_reconstruction(outer_grid, temperature)[0])
        - anchor_log_temperature
    )
    context = CoupledInnerOuterContext(
        inner_params=inner_params,
        outer_grid=outer_grid,
        outer_template=baseline.transport,
        outer_closure=closure,
        outer_scales=scales,
        anchor_log_surface_density=anchor_log_sigma,
        anchor_log_temperature=anchor_log_temperature,
        reference_log_surface_density_jump=reference_log_surface_density_jump,
        reference_log_temperature_jump=reference_log_temperature_jump,
        angular_flux_scale=max(abs(extracted.angular_momentum), 1.0),
        energy_flux_scale=max(abs(extracted.total_energy), 1.0),
        coupling_fraction=0.0,
        sonic_pivot=pivot,
        alpha=0.01,
        mu_stress=0.0,
        stress_factor=1.0,
    )
    state = pack_coupled_state(
        inner_state,
        sigma,
        temperature,
        omega,
        extracted.angular_momentum,
        extracted.total_energy,
        context,
    )
    return canonical, index, inner_radius, context, state


def _rank_summary(audit):
    values = asdict(audit)
    singular = np.asarray(values.pop("singular_values"), dtype=float)
    values["largest_singular_value"] = float(singular[0])
    values["smallest_six_singular_values"] = singular[-6:].tolist()
    return _jsonable(values)


def _stage_summary(result, context):
    evaluation = result.evaluation
    inner = evaluation.inner_profile
    outer = evaluation.outer_energy_profile
    outer_pi = float(
        positive_edge_reconstruction(
            context.outer_grid,
            outer.vertically_integrated_pressure,
        )[0]
    )
    outer_H = float(
        positive_edge_reconstruction(context.outer_grid, outer.H)[0]
    )
    outer_speed = float(
        positive_edge_reconstruction(
            context.outer_grid,
            np.maximum(-outer.radial_velocity, 1.0e-300),
        )[0]
    )
    audits = {
        "log_integrated_pressure": float(np.log(outer_pi / inner.Pi[-1])),
        "omega_relative": float(
            (evaluation.outer_edge_omega - inner.Omega[-1]) / inner.Omega[-1]
        ),
        "log_scale_height": float(np.log(outer_H / inner.H[-1])),
        "log_inflow_speed": float(np.log(outer_speed / inner.u[-1])),
    }
    return {
        "mu": context.coupling_fraction,
        "accepted": result.accepted,
        "nfev": result.nfev,
        "maximum_residual": result.maximum_residual,
        "message": result.message,
        "lambda0": float(inner.lambda0),
        "sonic_radius_rg": float(inner.R[0] / context.inner_params.r_g),
        "interface_angular_flux": evaluation.interface_flux.angular_momentum,
        "interface_energy_flux": evaluation.interface_flux.total_energy,
        "block_maximum_residuals": {
            "inner_core": float(np.max(np.abs(evaluation.inner_core))),
            "outer_stress": float(np.max(np.abs(evaluation.outer_stress))),
            "outer_radial": float(np.max(np.abs(evaluation.outer_radial))),
            "outer_energy": float(np.max(np.abs(evaluation.outer_energy))),
            "flux_extraction": float(np.max(np.abs(evaluation.flux_extraction))),
            "interface_boundary": float(np.max(np.abs(evaluation.interface_boundary))),
        },
        "anchor_residual": evaluation.anchor_boundary.tolist(),
        "continuity_residual": evaluation.continuity_boundary.tolist(),
        "primitive_audits": audits,
        "unused_sonic_compatibility": float(
            unused_sonic_compatibility(
                result.state[: 2 * context.inner_params.n_nodes + 2],
                context.inner_params,
                pivot=context.sonic_pivot,
            )
        ),
    }


def run(n_inner: int = N_INNER, n_outer: int = N_OUTER):
    canonical, index, _inner_radius, context, state = build_coupled_case(
        n_inner,
        n_outer,
    )
    stages = []
    rank_audits = {}
    result = None
    for mu in COUPLING_STAGES:
        context = replace(context, coupling_fraction=mu)
        result = solve_coupled_inner_outer_steady(
            state,
            context,
            tolerance=1.0e-7,
            max_nfev=1500,
        )
        stages.append(_stage_summary(result, context))
        if not result.accepted:
            break
        state = result.state
        if mu in {0.0, 1.0}:
            rank_audits[f"mu_{mu:g}"] = _rank_summary(
                audit_coupled_rank(state, context)
            )
            audit = rank_audits[f"mu_{mu:g}"]
            full_rank = audit["ranks_by_relative_threshold"]["1e-10"] == len(state)
            if (
                not full_rank
                or audit["preboundary_nullity"] != 2
                or audit["interface_response_rank"] != 2
                or audit["sonic_rank"] != 2
            ):
                break

    assert result is not None
    inner_luminosity = float(
        np.trapezoid(
            2.0 * np.pi * result.evaluation.inner_profile.R
            * result.evaluation.inner_profile.Q_rad,
            result.evaluation.inner_profile.R,
        )
        / eddington_luminosity(context.inner_params.M2_g)
    )
    outer_luminosity = float(
        np.sum(result.evaluation.outer_energy_profile.radiative_loss_rate_cells)
        / eddington_luminosity(context.inner_params.M2_g)
    )
    reached_full = bool(stages[-1]["mu"] == 1.0 and stages[-1]["accepted"])
    output = {
        "target_interface_rg": TARGET_RG,
        "actual_interface_rg": float(
            canonical.R[index] / context.inner_params.r_g
        ),
        "n_inner": int(n_inner),
        "n_outer": int(n_outer),
        "sonic_pivot": context.sonic_pivot,
        "unknown_count": int(len(state)),
        "residual_count": int(len(result.evaluation.residual)),
        "canonical_anchor": "log_surface_density_and_log_temperature",
        "interface_homotopy": "reference_primitive_jump_to_zero",
        "reference_log_surface_density_jump": (
            context.reference_log_surface_density_jump
        ),
        "reference_log_temperature_jump": (
            context.reference_log_temperature_jump
        ),
        "coupling_stages_requested": list(COUPLING_STAGES),
        "reached_full_coupling": reached_full,
        "rank_audits": rank_audits,
        "stages": stages,
        "inner_Lrad_over_LEdd": inner_luminosity,
        "outer_Lrad_over_LEdd": outer_luminosity,
        "composite_Lrad_over_LEdd": inner_luminosity + outer_luminosity,
        "max_outer_H_over_R": float(
            np.max(
                result.evaluation.outer_energy_profile.H
                / context.outer_grid.centers
            )
        ),
        "next_stage": (
            "mesh_certification"
            if reached_full
            else "global_signed_transonic_fallback"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT,
        state=result.state,
        inner_radius=result.evaluation.inner_profile.R,
        outer_radius=context.outer_grid.centers,
        coupling_fraction=context.coupling_fraction,
        n_inner=n_inner,
        n_outer=n_outer,
    )
    return output


def main() -> None:
    print(json.dumps(_jsonable(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
