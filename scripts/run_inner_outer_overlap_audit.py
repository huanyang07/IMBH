"""Audit a common physical overlap between inner and reservoir solutions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    OverlapDiagnostics,
    OverlapGateConfig,
    TransonicSlimParams,
    contiguous_passing_bands,
    intersect_bands,
    overlap_diagnostics,
    transonic_profile_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import gravitational_radius


ROOT = Path(__file__).resolve().parents[1]
TRANSONIC = ROOT / "results/canonical/no_wind_mdot5/state.npz"
RESERVOIR = ROOT / "results/canonical/signed_flux_total_energy_rin10_N512"
OUTPUT = ROOT / "outputs/tables/inner_outer_overlap_audit.json"
R_MIN_RG = 12.0
R_MAX_RG = 60.0


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


def _restrict(diagnostics: OverlapDiagnostics, r_g: float) -> OverlapDiagnostics:
    radius_rg = diagnostics.radius / r_g
    mask = (radius_rg >= R_MIN_RG) & (radius_rg <= R_MAX_RG)
    values = {
        field: np.asarray(getattr(diagnostics, field))[mask]
        for field in diagnostics.__dataclass_fields__
    }
    values["radius"] = values["radius"] / r_g
    return OverlapDiagnostics(**values)


def _transonic_diagnostics(config: OverlapGateConfig) -> OverlapDiagnostics:
    profile, params = _load_transonic()
    pressure_acceleration = np.abs(np.gradient(profile.Pi, profile.R, edge_order=2) / profile.Sigma)
    pressure_fraction = pressure_acceleration / (profile.R * profile.Omega_K**2)
    diagnostics = overlap_diagnostics(
        profile.R,
        profile.Sigma,
        profile.T,
        profile.H,
        profile.rho,
        -profile.u,
        pressure_fraction,
        params.M2_g,
        tau_scattering=profile.tau,
        source_fraction=0.0,
        config=config,
    )
    return _restrict(diagnostics, params.r_g)


def _reservoir_diagnostics(
    filename: str,
    config: OverlapGateConfig,
) -> OverlapDiagnostics:
    fiducial = FiducialParams()
    r_g = gravitational_radius(fiducial.M2_g)
    with np.load(RESERVOIR / filename) as data:
        radius = np.asarray(data["radius_centers"], dtype=float)
        sigma = np.asarray(data["surface_density"], dtype=float)
        temperature = np.asarray(data["temperature"], dtype=float)
        H = np.asarray(data["H"], dtype=float)
        mdot_faces = np.asarray(data["mdot_faces"], dtype=float)
        pressure_fraction = np.asarray(data["radial_pressure_force_fraction"], dtype=float)
        tau = np.asarray(data["tau_scattering"], dtype=float)
        stream_energy = np.abs(np.asarray(data["stream_energy_rate_cells"], dtype=float))
    mdot = 0.5 * (mdot_faces[:-1] + mdot_faces[1:])
    velocity = -mdot / (2.0 * np.pi * radius * sigma)
    source_total = float(np.sum(stream_energy))
    source_fraction = (
        stream_energy / source_total if source_total > 0.0 else np.zeros_like(radius)
    )
    diagnostics = overlap_diagnostics(
        radius,
        sigma,
        temperature,
        H,
        sigma / (2.0 * H),
        velocity,
        pressure_fraction,
        fiducial.M2_g,
        tau_scattering=tau,
        source_fraction=source_fraction,
        config=config,
    )
    return _restrict(diagnostics, r_g)


def _summary(diagnostics: OverlapDiagnostics, config: OverlapGateConfig) -> dict[str, object]:
    failures = {
        "radial_pressure_fraction": diagnostics.radial_pressure_fraction
        > config.max_radial_pressure_fraction,
        "dln_l_k_dln_R": diagnostics.dln_l_k_dln_R < config.min_dln_l_k_dln_R,
        "H_over_R": diagnostics.H_over_R > config.max_H_over_R,
        "radial_mach": diagnostics.radial_mach > config.max_radial_mach,
        "tau_scattering": diagnostics.tau_scattering < config.min_tau_scattering,
        "tau_effective_low": diagnostics.tau_effective_low < config.min_tau_effective,
        "gradient_length_over_H": diagnostics.gradient_length_over_H
        < config.min_gradient_length_over_H,
        "source_fraction": diagnostics.source_fraction > config.max_source_fraction,
    }
    return {
        "bands_rg": contiguous_passing_bands(diagnostics),
        "passing_fraction": float(np.mean(diagnostics.passes)),
        "failure_counts": {key: int(np.count_nonzero(value)) for key, value in failures.items()},
        "extrema": {
            "max_radial_pressure_fraction": float(np.max(diagnostics.radial_pressure_fraction)),
            "min_dln_l_k_dln_R": float(np.min(diagnostics.dln_l_k_dln_R)),
            "max_H_over_R": float(np.max(diagnostics.H_over_R)),
            "max_radial_mach": float(np.max(diagnostics.radial_mach)),
            "min_tau_scattering": float(np.min(diagnostics.tau_scattering)),
            "min_tau_effective_low": float(np.min(diagnostics.tau_effective_low)),
            "min_tau_effective_high": float(np.min(diagnostics.tau_effective_high)),
            "min_gradient_length_over_H": float(np.min(diagnostics.gradient_length_over_H)),
            "max_source_fraction": float(np.max(diagnostics.source_fraction)),
        },
    }


def _audit_tier(config: OverlapGateConfig) -> dict[str, object]:
    transonic = _transonic_diagnostics(config)
    wall = _reservoir_diagnostics("tidal_wall.npz", config)
    open_edge = _reservoir_diagnostics("zero_torque.npz", config)
    transonic_bands = contiguous_passing_bands(transonic)
    wall_bands = contiguous_passing_bands(wall)
    open_bands = contiguous_passing_bands(open_edge)
    return {
        "gate_config": asdict(config),
        "transonic": _summary(transonic, config),
        "tidal_wall": _summary(wall, config),
        "open_zero_torque": _summary(open_edge, config),
        "common_transonic_wall_bands_rg": intersect_bands(transonic_bands, wall_bands),
        "common_transonic_open_bands_rg": intersect_bands(transonic_bands, open_bands),
    }


def run() -> dict[str, object]:
    strict = OverlapGateConfig(max_radial_pressure_fraction=0.05)
    sensitivity = OverlapGateConfig(max_radial_pressure_fraction=0.10)
    result: dict[str, object] = {
        "window_rg": [R_MIN_RG, R_MAX_RG],
        "absorption_opacity_bracket_cm2_g": [6.4e22, 5.0e24],
        "absorption_status": "diagnostic bracket; not used by the cooling closure",
        "strict": _audit_tier(strict),
        "pressure_tolerance_sensitivity": _audit_tier(sensitivity),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
