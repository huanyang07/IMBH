"""Solve outer wall reservoirs from transonic conserved interface fluxes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    SignedThermalClosure,
    TransonicSlimParams,
    make_log_grid,
    normalized_stream_injection_state,
    rotation_profile_from_omega,
    signed_inner_interface_flux,
    solve_signed_total_energy_thermoviscous_steady,
    transonic_profile_from_state_vector,
    transonic_profile_interface_flux,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_luminosity, eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
TRANSONIC = ROOT / "results/canonical/no_wind_mdot5/state.npz"
OUTPUT = ROOT / "outputs/tables/two_domain_interface_sweep.json"
INTERFACE_TARGETS_RG = (30.0, 40.0, 50.0, 60.0)
RESERVOIR_RESOLUTIONS = (128, 256)
RESERVOIR_OUTER_RADIUS_RG = 335.0


def _optional_pair(data, key: str) -> tuple[float, float] | None:
    if key not in data:
        return None
    values = np.asarray(data[key], dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        return None
    return float(values[0]), float(values[1])


def _custom_grid(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    values = np.asarray(data["custom_grid_xi"], dtype=float)
    if values.shape != (int(data["n_nodes"]),):
        return None
    return tuple(float(value) for value in values)


def _load_transonic():
    fiducial = FiducialParams()
    with np.load(TRANSONIC) as data:
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=float(data["ratio"]) * eddington_mdot(fiducial.M2_g),
            alpha=0.01,
            mu_stress=0.0,
            stress_factor=1.0,
            R_out_rg=float(data["R_out_rg"]),
            n_nodes=int(data["n_nodes"]),
            grid_power=float(data["grid_power"]),
            custom_grid_xi=_custom_grid(data),
            outer_closure=str(np.asarray(data["outer_closure"]).item()),
            outer_match_log_slopes=_optional_pair(data, "outer_match_log_slopes"),
            residual_tol=1.0e-8,
            max_nfev=1,
        )
        state = np.asarray(data["z"], dtype=float)
    return transonic_profile_from_state_vector(state, params), params


def _interpolate_positive(radius, values, target: float) -> float:
    return float(
        np.exp(np.interp(np.log(target), np.log(radius), np.log(values)))
    )


def _primitive_mismatch(profile, signed, target_radius: float) -> dict[str, float]:
    outer = signed.energy.profile
    transport = signed.transport
    radius = float(target_radius)
    inner_sigma = _interpolate_positive(profile.R, profile.Sigma, radius)
    inner_temperature = _interpolate_positive(profile.R, profile.T, radius)
    inner_H = _interpolate_positive(profile.R, profile.H, radius)
    inner_Pi = _interpolate_positive(profile.R, profile.Pi, radius)
    inner_omega = _interpolate_positive(profile.R, profile.Omega, radius)
    inner_velocity = -_interpolate_positive(profile.R, profile.u, radius)
    outer_sigma = float(transport.surface_density[0])
    outer_temperature = float(signed.energy.temperature[0])
    outer_H = float(outer.H[0])
    outer_Pi = float(outer.vertically_integrated_pressure[0])
    outer_omega = float(transport.omega[0])
    outer_velocity = float(outer.radial_velocity[0])
    sound_speed = max(abs(outer_omega * outer_H), 1.0)
    values = {
        "log_surface_density": float(np.log(outer_sigma / inner_sigma)),
        "log_temperature": float(np.log(outer_temperature / inner_temperature)),
        "log_integrated_pressure": float(np.log(outer_Pi / inner_Pi)),
        "log_scale_height": float(np.log(outer_H / inner_H)),
        "omega_relative": float((outer_omega - inner_omega) / inner_omega),
        "radial_velocity_over_sound_speed": float(
            (outer_velocity - inner_velocity) / sound_speed
        ),
    }
    values["maximum_absolute"] = float(max(abs(value) for value in values.values()))
    return values


def _solve_one(
    target_rg: float,
    n_reservoir: int,
    *,
    pressure_supported: bool = False,
    pressure_damping: float = 0.05,
    pressure_smoothing_log_width: float = 0.08,
    pressure_max_iterations: int = 400,
) -> dict[str, object]:
    transonic, params = _load_transonic()
    potential = PaczynskiWiitaPotential(params.M2_g)
    index = int(np.argmin(np.abs(transonic.R / potential.r_g - target_rg)))
    interface_radius = float(transonic.R[index])
    prescribed = transonic_profile_interface_flux(
        transonic,
        params.M2_g,
        params.Mdot_g_s,
        index,
    )
    grid = make_log_grid(
        interface_radius,
        RESERVOIR_OUTER_RADIUS_RG * potential.r_g,
        n_reservoir,
    )
    stream_rate = 5.0 * eddington_mdot(params.M2_g)
    stream_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(stream_radius))
    stream_B = float(
        potential.phi(stream_radius)
        + 0.5 * (stream_l / stream_radius) ** 2
    )
    stream = normalized_stream_injection_state(
        grid,
        stream_rate,
        center=240.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_B,
    )
    boundary = SignedFluxBoundary(
        inner_mode="prescribed_flux",
        outer_mode="tidal_wall",
    )
    closure = SignedThermalClosure(temperature_bounds=(1.0e3, 1.0e9))
    temperature_seed = np.interp(grid.centers, transonic.R, transonic.T)
    warm_start = None
    if pressure_supported:
        warm_start = solve_signed_total_energy_thermoviscous_steady(
            grid,
            params.M2_g,
            alpha=0.01,
            boundary=boundary,
            stream_state=stream,
            closure=closure,
            temperature_seed=temperature_seed,
            prescribed_inner_flux=prescribed,
            damping=0.2,
            tolerance=2.0e-3,
            max_iterations=60,
            energy_tolerance=1.0e-6,
            energy_max_nfev=1000,
        )
        if not warm_start.converged:
            raise RuntimeError("Keplerian warm start did not converge")
        temperature_seed = warm_start.energy.temperature
    solved = warm_start
    support_stages = (
        (0.10, 0.25, 0.50, 0.75, 1.0)
        if pressure_supported
        else (0.0,)
    )
    for support_fraction in support_stages:
        solved = solve_signed_total_energy_thermoviscous_steady(
            grid,
            params.M2_g,
            alpha=0.01,
            boundary=boundary,
            stream_state=stream,
            closure=closure,
            temperature_seed=temperature_seed,
            prescribed_inner_flux=prescribed,
            damping=pressure_damping if pressure_supported else 0.2,
            tolerance=2.0e-3,
            max_iterations=(
                pressure_max_iterations if pressure_supported else 100
            ),
            energy_tolerance=1.0e-6,
            energy_max_nfev=1000,
            pressure_supported_rotation=pressure_supported,
            pressure_support_fraction=support_fraction,
            pressure_smoothing_log_width=pressure_smoothing_log_width,
            viscosity_seed=(None if solved is None else solved.viscosity),
            rotation_seed=(
                None
                if solved is None
                else rotation_profile_from_omega(grid, solved.transport.omega)
            ),
        )
        if not solved.converged:
            break
        temperature_seed = solved.energy.temperature
    assert solved is not None
    recovered = signed_inner_interface_flux(solved.transport, solved.energy.profile)
    log_radius = np.log(grid.centers)
    dln_l = np.gradient(
        np.log(solved.transport.specific_angular_momentum),
        log_radius,
        edge_order=2,
    )
    dln_omega = np.gradient(
        np.log(solved.transport.omega),
        log_radius,
        edge_order=2,
    )
    pressure_gradient = np.gradient(
        solved.energy.profile.vertically_integrated_pressure,
        grid.centers,
        edge_order=2,
    )
    force_target = potential.omega_k(grid.centers) ** 2 + pressure_gradient / (
        grid.centers * solved.transport.surface_density
    )
    force_balance_mismatch = np.max(
        np.abs(solved.transport.omega**2 - force_target)
        / potential.omega_k(grid.centers) ** 2
    )
    inner_luminosity = float(
        np.trapezoid(
            2.0 * np.pi * transonic.R[: index + 1] * transonic.Q_rad[: index + 1],
            transonic.R[: index + 1],
        )
        / eddington_luminosity(params.M2_g)
    )
    outer_luminosity = float(
        np.sum(solved.energy.profile.radiative_loss_rate_cells)
        / eddington_luminosity(params.M2_g)
    )
    flux_scales = {
        "mdot": max(abs(prescribed.mdot), 1.0),
        "angular_momentum": max(abs(prescribed.angular_momentum), 1.0),
        "total_energy": max(abs(prescribed.total_energy), 1.0),
    }
    flux_mismatch = {
        "mdot": float((recovered.mdot - prescribed.mdot) / flux_scales["mdot"]),
        "angular_momentum": float(
            (recovered.angular_momentum - prescribed.angular_momentum)
            / flux_scales["angular_momentum"]
        ),
        "total_energy": float(
            (recovered.total_energy - prescribed.total_energy)
            / flux_scales["total_energy"]
        ),
    }
    return {
        "target_interface_rg": target_rg,
        "actual_interface_rg": interface_radius / potential.r_g,
        "N_reservoir": int(n_reservoir),
        "rotation_mode": (
            "pressure_supported" if pressure_supported else "keplerian"
        ),
        "pressure_support_fraction_reached": float(support_fraction),
        "pressure_damping": float(pressure_damping),
        "pressure_smoothing_log_width": float(pressure_smoothing_log_width),
        "converged": solved.converged,
        "iterations": solved.iterations,
        "maximum_log_viscosity_change": solved.maximum_log_viscosity_change,
        "maximum_log_rotation_change": solved.maximum_log_rotation_change,
        "iteration_history_tail": solved.history[-10:].tolist(),
        "maximum_energy_residual": solved.energy.maximum_normalized_residual,
        "flux_mismatch_relative": flux_mismatch,
        "primitive_mismatch": _primitive_mismatch(
            transonic,
            solved,
            float(grid.centers[0]),
        ),
        "max_H_over_R": float(np.max(solved.energy.profile.H / grid.centers)),
        "inner_Lrad_over_LEdd": inner_luminosity,
        "outer_Lrad_over_LEdd": outer_luminosity,
        "composite_Lrad_over_LEdd": inner_luminosity + outer_luminosity,
        "outer_torque_over_stream_J": float(
            solved.transport.viscous_torque_faces[-1] / (stream_rate * stream_l)
        ),
        "max_radial_pressure_fraction": float(
            np.max(solved.energy.profile.radial_pressure_force_fraction)
        ),
        "minimum_dln_l_dln_R": float(np.min(dln_l)),
        "maximum_dln_omega_dln_R": float(np.max(dln_omega)),
        "maximum_radial_force_balance_mismatch": float(force_balance_mismatch),
    }


def run() -> dict[str, object]:
    rows = [
        _solve_one(target, resolution)
        for resolution in RESERVOIR_RESOLUTIONS
        for target in INTERFACE_TARGETS_RG
    ]
    by_resolution = {}
    for resolution in RESERVOIR_RESOLUTIONS:
        selected = [row for row in rows if row["N_reservoir"] == resolution]
        relative_spreads = {}
        for key in (
            "max_H_over_R",
            "composite_Lrad_over_LEdd",
            "outer_torque_over_stream_J",
        ):
            values = np.asarray([row[key] for row in selected], dtype=float)
            relative_spreads[key] = float(
                np.ptp(values) / max(abs(np.mean(values)), 1.0e-300)
            )
        maximum_flux_mismatch = max(
            abs(value)
            for row in selected
            for value in row["flux_mismatch_relative"].values()
        )
        maximum_primitive_mismatch = max(
            row["primitive_mismatch"]["maximum_absolute"] for row in selected
        )
        by_resolution[str(resolution)] = {
            "relative_spreads": relative_spreads,
            "maximum_flux_mismatch": maximum_flux_mismatch,
            "maximum_primitive_mismatch": maximum_primitive_mismatch,
            "flux_gate": maximum_flux_mismatch <= 1.0e-10,
            "interface_position_gate": (
                relative_spreads["composite_Lrad_over_LEdd"] <= 0.01
            ),
            "primitive_continuity_gate": maximum_primitive_mismatch <= 0.10,
        }
    mesh_differences = {}
    for target in INTERFACE_TARGETS_RG:
        coarse = next(
            row
            for row in rows
            if row["target_interface_rg"] == target
            and row["N_reservoir"] == RESERVOIR_RESOLUTIONS[0]
        )
        fine = next(
            row
            for row in rows
            if row["target_interface_rg"] == target
            and row["N_reservoir"] == RESERVOIR_RESOLUTIONS[-1]
        )
        mesh_differences[str(target)] = {
            key: float(abs(fine[key] - coarse[key]) / max(abs(fine[key]), 1.0e-300))
            for key in ("composite_Lrad_over_LEdd", "max_H_over_R")
        }
    result: dict[str, object] = {
        "reservoir_resolutions": list(RESERVOIR_RESOLUTIONS),
        "outer_boundary": "tidal_wall",
        "interface_targets_rg": list(INTERFACE_TARGETS_RG),
        "all_converged": all(bool(row["converged"]) for row in rows),
        "by_resolution": by_resolution,
        "mesh_relative_differences": mesh_differences,
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
