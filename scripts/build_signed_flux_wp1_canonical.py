"""Build compact pre-WP1 and angularly closed signed-flux checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    SignedThermalClosure,
    normalized_stream_injection_state,
    solve_signed_flux_steady,
    solve_signed_flux_steady_legacy,
    solve_signed_thermoviscous_steady,
)
from imri_qpe.layer3_minidisk_1d.grid import make_log_grid
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
N = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _thermal_summary(
    result, grid, potential, stream_rate, mass
) -> dict[str, float | bool]:
    profile = result.thermal.profile
    viscous = float(np.sum(profile.viscous_heating_rate_cells))
    stream = float(np.sum(profile.stream_heating_rate_cells))
    radiative = float(np.sum(profile.radiative_cooling_rate_cells))
    internal_export = float(-np.sum(profile.advective_rate_cells))
    return {
        "converged": bool(result.converged),
        "inner_mdot_over_stream": float(result.transport.mdot_faces[0] / stream_rate),
        "outer_mdot_over_stream": float(result.transport.mdot_faces[-1] / stream_rate),
        "outer_torque_over_stream_J": float(
            result.transport.viscous_torque_faces[-1]
            / np.sum(result.transport.source_angular_rate_cells)
        ),
        "unmodeled_angular_defect_relative": float(
            result.transport.angular_momentum_budget_defect
            / np.sum(result.transport.source_angular_rate_cells)
        ),
        "internal_energy_export_fraction": internal_export / (viscous + stream),
        "max_H_over_R": float(np.max(profile.H / grid.centers)),
        "Lrad_over_LEdd": radiative / eddington_luminosity(mass),
        "minimum_tau_scattering": float(np.min(profile.tau)),
        "maximum_radial_pressure_force_fraction": float(
            np.max(profile.radial_pressure_force_fraction)
        ),
        "radial_pressure_force_fraction_10rg": float(
            np.interp(
                np.log(10.0 * potential.r_g),
                np.log(grid.centers),
                profile.radial_pressure_force_fraction,
            )
        ),
        "final_viscosity_mismatch": float(result.maximum_log_viscosity_change),
        "thermal_residual": float(result.thermal.maximum_normalized_residual),
    }


def _save_state(path: Path, transport, grid, *, thermal=None, viscosity=None) -> None:
    payload = {
        "radius_centers": grid.centers,
        "radius_edges": grid.edges,
        "surface_density": transport.surface_density,
        "viscosity": transport.viscosity if viscosity is None else viscosity,
        "mdot_faces": transport.mdot_faces,
        "angular_flux_faces": transport.angular_flux_faces,
        "viscous_torque_faces": transport.viscous_torque_faces,
        "source_mass_rate_cells": transport.source_mass_rate_cells,
        "source_angular_rate_cells": transport.source_angular_rate_cells,
        "source_total_energy_rate_cells": transport.source_total_energy_rate_cells,
    }
    if thermal is not None:
        payload.update(
            temperature=thermal.temperature,
            H=thermal.profile.H,
            tau_scattering=thermal.profile.tau,
            radial_pressure_force_fraction=thermal.profile.radial_pressure_force_fraction,
            dln_l_k_dln_R=thermal.profile.dln_l_k_dln_R,
        )
    np.savez_compressed(path, **payload)


def _finalize(directory: Path, provenance: dict) -> None:
    payload = sorted(
        path
        for path in directory.iterdir()
        if path.name not in {"provenance.json", "SHA256SUMS.txt"}
    )
    provenance["payload_sha256"] = {path.name: _sha256(path) for path in payload}
    _write_json(directory / "provenance.json", provenance)
    files = sorted(
        path for path in directory.iterdir() if path.name != "SHA256SUMS.txt"
    )
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files)
    )


def _rebuild_manifest() -> None:
    rows = []
    for case in sorted(path for path in CANONICAL.iterdir() if path.is_dir()):
        provenance = json.loads((case / "provenance.json").read_text())
        status = provenance.get(
            "scientific_status", provenance.get("numerical_status")
        )
        for path in sorted(item for item in case.iterdir() if item.is_file()):
            rows.append(
                {
                    "case": case.name,
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.1 * potential.r_g, 335.0 * potential.r_g, N)
    stream_rate = 5.0 * eddington_mdot(mass)
    stream_radius = 248.96693 * potential.r_g
    stream_l = float(potential.l_k(stream_radius))
    stream_B = float(
        potential.phi(stream_radius) + 0.5 * (stream_l / stream_radius) ** 2
    )
    stream = normalized_stream_injection_state(
        grid,
        stream_rate,
        center=240.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_B,
    )
    source_l = np.divide(
        stream.angular_momentum_rate_cells,
        stream.mass_rate_cells,
        out=np.full(N, stream_l),
        where=stream.mass_rate_cells > 0.0,
    )
    prescribed_viscosity = (
        0.01 * 0.1**2 * grid.centers**2 * potential.omega_k(grid.centers)
    )
    config = {
        "N": N,
        "M_g": mass,
        "R_in_rg": 6.1,
        "R_out_rg": 335.0,
        "stream_rate_over_edd": 5.0,
        "stream_center_rg": 240.0,
        "stream_circularization_rg": 248.96693,
        "stream_log_width": 0.08,
        "alpha": 0.01,
    }

    for closure_name in ("legacy_53566fa", "angular_closed_wp1"):
        directory = CANONICAL / f"signed_flux_{closure_name}_N512"
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
        summaries = {}
        for outer_mode in ("tidal_wall", "zero_torque"):
            boundary = SignedFluxBoundary(outer_mode=outer_mode)
            if closure_name == "legacy_53566fa":
                prescribed = solve_signed_flux_steady_legacy(
                    grid,
                    prescribed_viscosity,
                    mass,
                    boundary=boundary,
                    source_mass_rate_cells=stream.mass_rate_cells,
                    source_specific_angular_momentum=source_l,
                )
                thermal_closure = SignedThermalClosure(
                    stream_specific_angular_momentum=stream_l,
                    stream_specific_total_energy=stream_B,
                    temperature_bounds=(1.0e3, 1.0e9),
                )
                angular_closure = "legacy_mass_only"
            else:
                prescribed = solve_signed_flux_steady(
                    grid,
                    prescribed_viscosity,
                    mass,
                    boundary=boundary,
                    stream_state=stream,
                )
                thermal_closure = SignedThermalClosure(
                    temperature_bounds=(1.0e3, 1.0e9)
                )
                angular_closure = "conservative"
            _save_state(
                directory / f"prescribed_{outer_mode}.npz", prescribed, grid
            )
            thermal = solve_signed_thermoviscous_steady(
                grid,
                mass,
                alpha=0.01,
                boundary=boundary,
                stream_state=stream,
                thermal_closure=thermal_closure,
                temperature_seed=np.full(N, 1.0e6),
                angular_closure=angular_closure,
                damping=0.2,
                tolerance=2.0e-3,
                max_iterations=60,
            )
            _save_state(
                directory / f"thermoviscous_{outer_mode}.npz",
                thermal.transport,
                grid,
                thermal=thermal.thermal,
                viscosity=thermal.viscosity,
            )
            summaries[outer_mode] = _thermal_summary(
                thermal, grid, potential, stream_rate, mass
            )
        _write_json(directory / "config.json", config)
        _write_json(directory / "summary.json", summaries)
        is_legacy = closure_name == "legacy_53566fa"
        _finalize(
            directory,
            {
                "generation_command": (
                    "PYTHONPATH=src python3 scripts/build_signed_flux_wp1_canonical.py"
                ),
                "source_parent_commit": "53566fa",
                "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
                "physical_status": "DIAGNOSTIC ONLY",
                "claim_scope": (
                    "53566fa mass-only angular closure baseline"
                    if is_legacy
                    else "WP1 fixed-Keplerian angularly closed internal-energy model"
                ),
                "does_not_establish": (
                    "A physical hot branch, total-energy closure, or inner transonic match."
                ),
            },
        )
    _rebuild_manifest()


if __name__ == "__main__":
    run()
