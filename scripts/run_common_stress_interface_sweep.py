"""Audit a common alpha-stress reservoir across the candidate interfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    SignedThermalClosure,
    common_alpha_stress_torque,
    diffusive_alpha_torque,
    make_log_grid,
    normalized_stream_injection_state,
    signed_inner_interface_flux,
    solve_common_stress_total_energy_steady,
    solve_signed_total_energy_thermoviscous_steady,
    transonic_profile_interface_flux,
)
from imri_qpe.scales import eddington_luminosity, eddington_mdot

from run_two_domain_interface_sweep import _interpolate_positive, _load_transonic


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/common_stress_interface_sweep.json"
STATE_OUTPUT = ROOT / "outputs/checkpoints/common_stress_interface_sweep"
INTERFACE_TARGETS_RG = (30.0, 40.0, 50.0, 60.0)
RESERVOIR_RESOLUTIONS = (64, 128, 256)
STRESS_STAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA = 0.01
MU_STRESS = 0.0
STRESS_FACTOR = 1.0
RESERVOIR_OUTER_RADIUS_RG = 335.0


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _positive_interpolate(old_radius, values, new_radius):
    return np.exp(
        np.interp(np.log(new_radius), np.log(old_radius), np.log(values))
    )


def _primitive_mismatch(profile, solved, target_radius: float) -> dict[str, float]:
    outer = solved.energy_profile
    transport = solved.transport
    radius = float(target_radius)
    inner_sigma = _interpolate_positive(profile.R, profile.Sigma, radius)
    inner_temperature = _interpolate_positive(profile.R, profile.T, radius)
    inner_H = _interpolate_positive(profile.R, profile.H, radius)
    inner_Pi = _interpolate_positive(profile.R, profile.Pi, radius)
    inner_omega = _interpolate_positive(profile.R, profile.Omega, radius)
    inner_velocity = -_interpolate_positive(profile.R, profile.u, radius)
    outer_sigma = float(transport.surface_density[0])
    outer_temperature = float(solved.temperature[0])
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
    values["maximum_absolute"] = float(
        max(abs(value) for value in values.values())
    )
    return values


def _build_case(target_rg: float, n_reservoir: int):
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
    baseline = solve_signed_total_energy_thermoviscous_steady(
        grid,
        params.M2_g,
        alpha=ALPHA,
        boundary=boundary,
        stream_state=stream,
        closure=closure,
        temperature_seed=np.interp(grid.centers, transonic.R, transonic.T),
        prescribed_inner_flux=prescribed,
        damping=0.2,
        tolerance=2.0e-3,
        max_iterations=100,
        energy_tolerance=1.0e-6,
        energy_max_nfev=1000,
    )
    if not baseline.converged:
        raise RuntimeError(
            f"baseline failed at R={target_rg:g} rg, N={n_reservoir}"
        )
    return (
        transonic,
        params,
        potential,
        index,
        interface_radius,
        prescribed,
        grid,
        stream_rate,
        stream_l,
        closure,
        baseline,
    )


def _solve_one(target_rg: float, n_reservoir: int, coarse_seed=None):
    (
        transonic,
        params,
        potential,
        index,
        interface_radius,
        prescribed,
        grid,
        stream_rate,
        stream_l,
        closure,
        baseline,
    ) = _build_case(target_rg, n_reservoir)
    if coarse_seed is None:
        sigma = baseline.transport.surface_density
        temperature = baseline.energy.temperature
        stages = STRESS_STAGES
    else:
        old_radius, old_sigma, old_temperature = coarse_seed
        sigma = _positive_interpolate(old_radius, old_sigma, grid.centers)
        temperature = _positive_interpolate(
            old_radius, old_temperature, grid.centers
        )
        stages = (1.0,)

    stage_rows = []
    solved = None
    for fraction in stages:
        solved = solve_common_stress_total_energy_steady(
            grid,
            baseline.transport,
            sigma,
            temperature,
            params.M2_g,
            alpha=ALPHA,
            closure=closure,
            prescribed_inner_flux=prescribed,
            stress_fraction=fraction,
            mu_stress=MU_STRESS,
            stress_factor=STRESS_FACTOR,
            tolerance=1.0e-7,
            max_nfev=2000,
        )
        stage_rows.append(
            {
                "stress_fraction": fraction,
                "accepted": solved.accepted,
                "nfev": solved.nfev,
                "maximum_stress_residual": solved.maximum_stress_residual,
                "maximum_energy_residual": solved.maximum_energy_residual,
                "message": solved.message,
            }
        )
        if not solved.accepted:
            break
        sigma = solved.surface_density
        temperature = solved.temperature
    assert solved is not None

    recovered = signed_inner_interface_flux(
        solved.transport, solved.energy_profile
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
    old_diffusive = diffusive_alpha_torque(
        grid,
        solved.surface_density,
        solved.temperature,
        params.M2_g,
        alpha=ALPHA,
        closure=closure,
    )
    common = common_alpha_stress_torque(
        grid,
        solved.surface_density,
        solved.temperature,
        params.M2_g,
        alpha=ALPHA,
        closure=closure,
        mu_stress=MU_STRESS,
        stress_factor=STRESS_FACTOR,
    )
    chi_shear = old_diffusive / common
    inner_Pi = _interpolate_positive(
        transonic.R, transonic.Pi, float(grid.centers[0])
    )
    inner_alpha_torque = (
        2.0 * np.pi * ALPHA * grid.centers[0] ** 2 * inner_Pi
    )
    inner_luminosity = float(
        np.trapezoid(
            2.0 * np.pi * transonic.R[: index + 1] * transonic.Q_rad[: index + 1],
            transonic.R[: index + 1],
        )
        / eddington_luminosity(params.M2_g)
    )
    outer_luminosity = float(
        np.sum(solved.energy_profile.radiative_loss_rate_cells)
        / eddington_luminosity(params.M2_g)
    )
    primitive = _primitive_mismatch(
        transonic, solved, float(grid.centers[0])
    )
    row = {
        "target_interface_rg": target_rg,
        "actual_interface_rg": interface_radius / potential.r_g,
        "N_reservoir": int(n_reservoir),
        "accepted": solved.accepted,
        "stress_stages": stage_rows,
        "maximum_stress_residual": solved.maximum_stress_residual,
        "maximum_energy_residual": solved.maximum_energy_residual,
        "flux_mismatch_relative": flux_mismatch,
        "primitive_mismatch": primitive,
        "chi_shear_inner_cell": float(chi_shear[0]),
        "inner_alpha_stress_torque": float(inner_alpha_torque),
        "required_torque_inner_cell": float(
            baseline.transport.viscous_torque_centers[0]
        ),
        "old_diffusive_torque_inner_cell": float(old_diffusive[0]),
        "new_common_stress_torque_inner_cell": float(common[0]),
        "max_H_over_R": float(np.max(solved.energy_profile.H / grid.centers)),
        "inner_Lrad_over_LEdd": inner_luminosity,
        "outer_Lrad_over_LEdd": outer_luminosity,
        "composite_Lrad_over_LEdd": inner_luminosity + outer_luminosity,
        "outer_torque_over_stream_J": float(
            solved.transport.viscous_torque_faces[-1]
            / (stream_rate * stream_l)
        ),
        "max_radial_pressure_fraction": float(
            np.max(solved.energy_profile.radial_pressure_force_fraction)
        ),
    }
    return row, (grid.centers, solved.surface_density, solved.temperature), solved


def _save_state(target_rg: float, n_reservoir: int, radius, solved) -> None:
    STATE_OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        STATE_OUTPUT / f"R{target_rg:g}_N{n_reservoir}.npz",
        radius=np.asarray(radius, dtype=float),
        surface_density=solved.surface_density,
        temperature=solved.temperature,
        omega=solved.transport.omega,
        mdot_faces=solved.transport.mdot_faces,
        angular_flux_faces=solved.transport.angular_flux_faces,
        viscous_torque_faces=solved.transport.viscous_torque_faces,
        total_energy_flux_faces=solved.energy_profile.total_energy_flux_faces,
        H=solved.energy_profile.H,
        integrated_pressure=solved.energy_profile.vertically_integrated_pressure,
        radial_velocity=solved.energy_profile.radial_velocity,
    )


def run() -> dict[str, object]:
    rows = []
    for target in INTERFACE_TARGETS_RG:
        seed = None
        for resolution in RESERVOIR_RESOLUTIONS:
            row, seed, solved = _solve_one(target, resolution, seed)
            rows.append(row)
            if resolution == RESERVOIR_RESOLUTIONS[-1]:
                _save_state(target, resolution, seed[0], solved)
            if not row["accepted"]:
                break

    by_resolution = {}
    for resolution in RESERVOIR_RESOLUTIONS:
        selected = [row for row in rows if row["N_reservoir"] == resolution]
        if len(selected) != len(INTERFACE_TARGETS_RG):
            continue
        luminosities = np.asarray(
            [row["composite_Lrad_over_LEdd"] for row in selected], dtype=float
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
            "maximum_flux_mismatch": maximum_flux_mismatch,
            "maximum_primitive_mismatch": maximum_primitive_mismatch,
            "composite_luminosity_relative_spread": float(
                np.ptp(luminosities) / abs(np.mean(luminosities))
            ),
            "flux_gate": maximum_flux_mismatch <= 1.0e-10,
            "primitive_continuity_gate": maximum_primitive_mismatch <= 0.10,
            "interface_position_gate": (
                np.ptp(luminosities) / abs(np.mean(luminosities)) <= 0.01
            ),
        }
    mesh_differences = {}
    for target in INTERFACE_TARGETS_RG:
        selected = sorted(
            [row for row in rows if row["target_interface_rg"] == target],
            key=lambda row: row["N_reservoir"],
        )
        if len(selected) < 2:
            continue
        fine = selected[-1]
        previous = selected[-2]
        mesh_differences[str(target)] = {
            key: float(
                abs(fine[key] - previous[key]) / max(abs(fine[key]), 1.0e-300)
            )
            for key in ("composite_Lrad_over_LEdd", "max_H_over_R")
        }
        mesh_differences[str(target)]["primitive_maximum_absolute"] = float(
            abs(
                fine["primitive_mismatch"]["maximum_absolute"]
                - previous["primitive_mismatch"]["maximum_absolute"]
            )
        )
    fine_rows = [
        row for row in rows if row["N_reservoir"] == RESERVOIR_RESOLUTIONS[-1]
    ]
    decision_gate = bool(
        len(fine_rows) == len(INTERFACE_TARGETS_RG)
        and all(row["accepted"] for row in fine_rows)
        and max(
            row["primitive_mismatch"]["maximum_absolute"] for row in fine_rows
        )
        <= 0.10
        and by_resolution[str(RESERVOIR_RESOLUTIONS[-1])][
            "interface_position_gate"
        ]
    )
    result: dict[str, object] = {
        "stress_closure": {
            "alpha": ALPHA,
            "mu_stress": MU_STRESS,
            "stress_factor": STRESS_FACTOR,
            "torque": "2*pi*R^2*integrated_stress",
            "viscous_heating_source_added": False,
        },
        "reservoir_resolutions": list(RESERVOIR_RESOLUTIONS),
        "interface_targets_rg": list(INTERFACE_TARGETS_RG),
        "stress_homotopy_stages": list(STRESS_STAGES),
        "all_accepted": all(bool(row["accepted"]) for row in rows),
        "by_resolution": by_resolution,
        "mesh_relative_differences": mesh_differences,
        "primitive_gate_passed": decision_gate,
        "next_stage": (
            "fully_coupled_inner_outer"
            if decision_gate
            else "simultaneous_non_keplerian_reservoir"
        ),
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
