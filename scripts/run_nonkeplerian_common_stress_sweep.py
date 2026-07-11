"""Run the simultaneous non-Keplerian common-stress reservoir gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    signed_inner_interface_flux,
    solve_common_stress_total_energy_steady,
    solve_nonkeplerian_common_stress_steady,
)
from imri_qpe.scales import eddington_luminosity

from run_common_stress_interface_sweep import (
    ALPHA,
    INTERFACE_TARGETS_RG,
    MU_STRESS,
    RESERVOIR_RESOLUTIONS,
    STRESS_FACTOR,
    STRESS_STAGES,
    _build_case,
    _json_default,
    _positive_interpolate,
    _primitive_mismatch,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/nonkeplerian_common_stress_sweep.json"
STATE_OUTPUT = ROOT / "outputs/checkpoints/nonkeplerian_common_stress_sweep"
RADIAL_SUPPORT_STAGES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def _common_root(grid, baseline, params, closure, prescribed):
    sigma = baseline.transport.surface_density
    temperature = baseline.energy.temperature
    solved = None
    for fraction in STRESS_STAGES:
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
        if not solved.accepted:
            raise RuntimeError(
                f"common-stress homotopy failed at fraction {fraction:g}"
            )
        sigma = solved.surface_density
        temperature = solved.temperature
    assert solved is not None
    return solved


def _flux_mismatch(prescribed, recovered) -> dict[str, float]:
    scales = {
        "mdot": max(abs(prescribed.mdot), 1.0),
        "angular_momentum": max(abs(prescribed.angular_momentum), 1.0),
        "total_energy": max(abs(prescribed.total_energy), 1.0),
    }
    return {
        "mdot": float((recovered.mdot - prescribed.mdot) / scales["mdot"]),
        "angular_momentum": float(
            (recovered.angular_momentum - prescribed.angular_momentum)
            / scales["angular_momentum"]
        ),
        "total_energy": float(
            (recovered.total_energy - prescribed.total_energy)
            / scales["total_energy"]
        ),
    }


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
    stage_rows = []
    if coarse_seed is None:
        common = _common_root(grid, baseline, params, closure, prescribed)
        sigma = common.surface_density
        temperature = common.temperature
        omega = common.transport.omega
        stages = RADIAL_SUPPORT_STAGES
    else:
        old_radius, old_sigma, old_temperature, old_omega = coarse_seed
        sigma = _positive_interpolate(old_radius, old_sigma, grid.centers)
        temperature = _positive_interpolate(
            old_radius, old_temperature, grid.centers
        )
        omega = _positive_interpolate(old_radius, old_omega, grid.centers)
        stages = (1.0,)

    solved = None
    for fraction in stages:
        solved = solve_nonkeplerian_common_stress_steady(
            grid,
            baseline.transport,
            sigma,
            temperature,
            omega,
            params.M2_g,
            alpha=ALPHA,
            closure=closure,
            prescribed_inner_flux=prescribed,
            radial_support_fraction=fraction,
            mu_stress=MU_STRESS,
            stress_factor=STRESS_FACTOR,
            tolerance=1.0e-7,
            max_nfev=3000,
        )
        stage_rows.append(
            {
                "radial_support_fraction": fraction,
                "accepted": solved.accepted,
                "nfev": solved.nfev,
                "maximum_stress_residual": solved.maximum_stress_residual,
                "maximum_radial_residual": solved.maximum_radial_residual,
                "maximum_energy_residual": solved.maximum_energy_residual,
                "minimum_dln_l_dln_R": solved.minimum_dln_l_dln_R,
                "maximum_dln_omega_dln_R": solved.maximum_dln_omega_dln_R,
                "message": solved.message,
            }
        )
        if not solved.accepted:
            break
        sigma = solved.surface_density
        temperature = solved.temperature
        omega = solved.omega
    assert solved is not None

    recovered = signed_inner_interface_flux(
        solved.transport, solved.energy_profile
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
        "radial_support_stages": stage_rows,
        "maximum_stress_residual": solved.maximum_stress_residual,
        "maximum_radial_residual": solved.maximum_radial_residual,
        "maximum_energy_residual": solved.maximum_energy_residual,
        "minimum_dln_l_dln_R": solved.minimum_dln_l_dln_R,
        "maximum_dln_omega_dln_R": solved.maximum_dln_omega_dln_R,
        "flux_mismatch_relative": _flux_mismatch(prescribed, recovered),
        "primitive_mismatch": primitive,
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
    seed = (grid.centers, solved.surface_density, solved.temperature, solved.omega)
    return row, seed, solved


def _save_state(target_rg: float, n_reservoir: int, radius, solved) -> None:
    STATE_OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        STATE_OUTPUT / f"R{target_rg:g}_N{n_reservoir}.npz",
        radius=np.asarray(radius, dtype=float),
        surface_density=solved.surface_density,
        temperature=solved.temperature,
        omega=solved.omega,
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
        maximum_flux = max(
            abs(value)
            for row in selected
            for value in row["flux_mismatch_relative"].values()
        )
        maximum_primitive = max(
            row["primitive_mismatch"]["maximum_absolute"] for row in selected
        )
        by_resolution[str(resolution)] = {
            "maximum_flux_mismatch": maximum_flux,
            "maximum_primitive_mismatch": maximum_primitive,
            "composite_luminosity_relative_spread": float(
                np.ptp(luminosities) / abs(np.mean(luminosities))
            ),
            "flux_gate": maximum_flux <= 1.0e-8,
            "primitive_continuity_gate": maximum_primitive <= 0.05,
            "interface_position_gate": bool(
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
        fine, previous = selected[-1], selected[-2]
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
    production_gate = bool(
        len(fine_rows) == len(INTERFACE_TARGETS_RG)
        and all(row["accepted"] for row in fine_rows)
        and max(
            row["primitive_mismatch"]["maximum_absolute"] for row in fine_rows
        )
        <= 0.05
        and by_resolution[str(RESERVOIR_RESOLUTIONS[-1])][
            "interface_position_gate"
        ]
    )
    result: dict[str, object] = {
        "model": {
            "unknowns": ["log_surface_density", "log_temperature", "log_omega"],
            "equations": ["common_alpha_stress", "radial_momentum", "total_energy"],
            "alpha": ALPHA,
            "mu_stress": MU_STRESS,
            "stress_factor": STRESS_FACTOR,
            "projection": False,
            "smoothing": False,
            "accepted_state_clipping": False,
        },
        "reservoir_resolutions": list(RESERVOIR_RESOLUTIONS),
        "interface_targets_rg": list(INTERFACE_TARGETS_RG),
        "radial_support_stages": list(RADIAL_SUPPORT_STAGES),
        "all_accepted": all(bool(row["accepted"]) for row in rows),
        "by_resolution": by_resolution,
        "mesh_relative_differences": mesh_differences,
        "production_gate_passed": production_gate,
        "next_stage": (
            "fully_coupled_inner_outer"
            if production_gate
            else "fully_coupled_inner_outer_at_40rg_then_global_fallback"
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
