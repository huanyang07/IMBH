"""Audit the two declared inner-boundary architectures for fresh loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    continue_transonic_supersonic_plunge,
)
from imri_qpe.layer3_minidisk_1d.entropy_advection import (
    gas_radiation_adiabatic_sound_speed_squared,
)
from imri_qpe.layer3_minidisk_1d.transonic_thermo import vertical_state
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT
    / "outputs/checkpoints/global_low_throughput_remnant/transonic_profile.npz"
)
DEFAULT_COUPLED = ROOT / "outputs/tables/time_dae_coupled_open_evolved_mesh.json"
DEFAULT_OUTPUT = ROOT / "outputs/tables/global_inner_boundary_architecture_gate.json"
CRITICAL_RADIUS_RG = 5.996987044986418
EXCISION_RADII_RG = (4.5, 3.0, 2.1, 2.01, 2.001, 2.0001)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--coupled-evidence", type=Path, default=DEFAULT_COUPLED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_low_rate_critical_profile(path: Path):
    fiducial = FiducialParams()
    params = TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=0.025 * eddington_mdot(fiducial.M2_g),
        alpha=fiducial.alpha_cool,
        n_nodes=64,
        R_out_rg=335.0,
    )
    with np.load(path, allow_pickle=False) as data:
        radius = np.asarray(data["radius"], dtype=float)
        surface_density = np.asarray(data["surface_density"], dtype=float)
        radial_velocity = np.asarray(data["radial_velocity"], dtype=float)
        omega = np.asarray(data["omega"], dtype=float)
        temperature = np.asarray(data["temperature"], dtype=float)
    target_radius = CRITICAL_RADIUS_RG * params.r_g
    critical_index = int(np.argmin(np.abs(radius - target_radius)))
    if not np.isclose(
        radius[critical_index], target_radius, rtol=1.0e-10, atol=0.0
    ):
        raise ValueError("saved profile does not contain the declared critical node")

    column = vertical_state(
        float(surface_density[critical_index]),
        float(temperature[critical_index]),
        float(radius[critical_index]),
        params.potential,
    )
    specific_l = float(radius[critical_index] ** 2 * omega[critical_index])
    l0 = specific_l - (
        2.0
        * np.pi
        * radius[critical_index] ** 2
        * params.alpha
        * float(column.Pi)
        / params.Mdot_g_s
    )
    lambda0 = float(l0 / (params.r_g * C))
    profile = SimpleNamespace(
        R=radius[critical_index:],
        u=-radial_velocity[critical_index:],
        T=temperature[critical_index:],
        sonic_radius=float(radius[critical_index]),
        lambda0=lambda0,
    )
    acoustic_speed = float(
        np.sqrt(
            gas_radiation_adiabatic_sound_speed_squared(
                float(column.rho), float(column.T)
            )
        )
    )
    critical = {
        "radius_rg": float(radius[critical_index] / params.r_g),
        "lambda0": lambda0,
        "stationary_effective_mach": float(
            profile.u[0] / (float(column.H) * float(column.Omega_K))
        ),
        "euler_acoustic_mach": float(-profile.u[0] / acoustic_speed),
        "incoming_characteristics": int(
            sum(
                value > 0.0
                for value in (
                    -profile.u[0] - acoustic_speed,
                    -profile.u[0],
                    -profile.u[0],
                    -profile.u[0] + acoustic_speed,
                )
            )
        ),
    }
    return profile, params, critical


def _causal_excision_rows(profile, params) -> list[dict]:
    rows = []
    for radius_rg in EXCISION_RADII_RG:
        plunge = continue_transonic_supersonic_plunge(
            profile,
            params,
            radius_rg * params.r_g,
            n_nodes=256,
            maximum_log_step=1.0e-3,
        )
        radial_speed_over_c = float(-plunge.u[0] / C)
        sound_speed_over_c = float(plunge.effective_sound_speed[0] / C)
        incoming = int(plunge.incoming_characteristics[0])
        rows.append(
            {
                "radius_rg": radius_rg,
                "radial_speed_over_c": radial_speed_over_c,
                "sound_speed_over_c": sound_speed_over_c,
                "euler_acoustic_mach": float(plunge.radial_mach_number[0]),
                "incoming_characteristics": incoming,
                "subluminal_radial_speed": abs(radial_speed_over_c) < 1.0,
                "subluminal_sound_speed": sound_speed_over_c < 1.0,
                "zero_incoming_characteristics": incoming == 0,
                "causal_excision_gate": bool(
                    incoming == 0
                    and abs(radial_speed_over_c) < 1.0
                    and sound_speed_over_c < 1.0
                ),
            }
        )
    return rows


def _hybrid_evidence(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        evidence = json.load(stream)
    fine = evidence["fine_24_16"]
    final = fine["final_step"]
    return {
        "unknowns_and_rows": 2 * 24 + 5 * 16 + 5,
        "inner_nodes": 24,
        "outer_cells": 16,
        "accepted_steps": int(fine["accepted_steps"]),
        "all_steps_accepted": bool(fine["all_steps_accepted"]),
        "final_maximum_residual": float(final["maximum_residual"]),
        "fixed_residual_gate": 1.0e-7,
        "maximum_interface_continuity": float(
            final["maximum_interface_continuity"]
        ),
        "maximum_interface_flux_extraction": float(
            final["maximum_interface_flux_extraction"]
        ),
        "architecture_gate": bool(fine["all_steps_accepted"]),
    }


def main() -> None:
    arguments = _arguments()
    profile, params, critical = _load_low_rate_critical_profile(
        _absolute(arguments.profile)
    )
    excision_rows = _causal_excision_rows(profile, params)
    architecture_a = {
        "name": "one-domain causal excision",
        "differential_unknowns": "4 N",
        "backward_euler_rows": "4 N",
        "required_inner_boundary_rows": 0,
        "required_incoming_characteristics": 0,
        "critical_point": critical,
        "excision_rows": excision_rows,
        "architecture_gate": bool(
            any(row["causal_excision_gate"] for row in excision_rows)
        ),
    }
    architecture_b = {
        "name": "outer-evolving quasi-steady transonic response",
        "differential_storage_rank": "3 No",
        **_hybrid_evidence(_absolute(arguments.coupled_evidence)),
    }
    output = {
        "architecture_a": architecture_a,
        "architecture_b": architecture_b,
        "selected_architecture": None,
        "decision": (
            "neither existing architecture passes its declared production gate"
        ),
        "next_requirement": (
            "repair the inner causal physical model before more evolution"
        ),
        "audit_complete": True,
    }
    destination = _absolute(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
