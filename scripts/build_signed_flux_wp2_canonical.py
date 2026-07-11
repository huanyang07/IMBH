"""Build compact total-energy controls and near-ISCO failure evidence."""

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
    make_log_grid,
    normalized_stream_injection_state,
    solve_signed_total_energy_thermoviscous_steady,
)
from imri_qpe.scales import eddington_luminosity, eddington_mdot
from imri_qpe.units import solar_masses_to_g


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
PILOT = ROOT / "outputs/tables/signed_flux_total_energy_pilot.json"
N = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _prepare(name: str) -> Path:
    directory = CANONICAL / name
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
    return directory


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
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary(result, grid, mass, stream_rate, stream_l) -> dict[str, object]:
    profile = result.energy.profile
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "maximum_log_viscosity_change": result.maximum_log_viscosity_change,
        "total_energy_residual": result.energy.maximum_normalized_residual,
        "inner_mdot_over_stream": float(result.transport.mdot_faces[0] / stream_rate),
        "outer_mdot_over_stream": float(result.transport.mdot_faces[-1] / stream_rate),
        "outer_torque_over_stream_J": float(
            result.transport.viscous_torque_faces[-1] / (stream_rate * stream_l)
        ),
        "unmodeled_angular_defect_relative": float(
            result.transport.angular_momentum_budget_defect
            / (stream_rate * stream_l)
        ),
        "max_H_over_R": float(np.max(profile.H / grid.centers)),
        "Lrad_over_LEdd": float(
            np.sum(profile.radiative_loss_rate_cells) / eddington_luminosity(mass)
        ),
        "minimum_tau_scattering": float(np.min(profile.tau)),
        "maximum_radial_pressure_force_fraction": float(
            np.max(profile.radial_pressure_force_fraction)
        ),
    }


def _save_state(path: Path, result, grid) -> None:
    profile = result.energy.profile
    np.savez_compressed(
        path,
        radius_centers=grid.centers,
        radius_edges=grid.edges,
        surface_density=result.transport.surface_density,
        viscosity=result.viscosity,
        mdot_faces=result.transport.mdot_faces,
        angular_flux_faces=result.transport.angular_flux_faces,
        viscous_torque_faces=result.transport.viscous_torque_faces,
        temperature=result.energy.temperature,
        H=profile.H,
        tau_scattering=profile.tau,
        radial_pressure_force_fraction=profile.radial_pressure_force_fraction,
        column_bernoulli=profile.column_bernoulli,
        total_energy_flux_faces=profile.total_energy_flux_faces,
        vertical_work_rate_cells=profile.vertical_work_rate_cells,
        radiative_loss_rate_cells=profile.radiative_loss_rate_cells,
        stream_energy_rate_cells=profile.stream_energy_rate_cells,
        history=result.history,
        converged=np.asarray(result.converged),
        maximum_log_viscosity_change=np.asarray(
            result.maximum_log_viscosity_change
        ),
        total_energy_residual=np.asarray(result.energy.maximum_normalized_residual),
    )


def run() -> None:
    if not PILOT.is_file():
        raise FileNotFoundError(
            "run scripts/run_signed_flux_total_energy_pilot.py before this builder"
        )
    pilot_rows = json.loads(PILOT.read_text())
    near_isco_failure = [
        dict(row)
        for row in pilot_rows
        if row["inner_radius_rg"] == 6.1 and row["N"] in {256, 512}
    ]
    for row in near_isco_failure:
        legacy_key = "total_energy_ledger_defect_relative"
        if legacy_key in row:
            row["total_energy_telescoping_defect_relative"] = row.pop(legacy_key)
    failure = _prepare("signed_flux_total_energy_near_isco_failure")
    _write_json(failure / "failure_summary.json", near_isco_failure)
    _finalize(
        failure,
        {
            "generation_command": (
                "PYTHONPATH=src python3 scripts/run_signed_flux_total_energy_pilot.py && "
                "PYTHONPATH=src python3 scripts/build_signed_flux_wp2_canonical.py"
            ),
            "source_parent_commit": "248e43c",
            "energy_identity_revision": "enthalpy_vertical_work_v2",
            "supersedes": "the mixed enthalpy/internal-work payload in 248e43c",
            "numerical_status": "REJECTED",
            "physical_status": "REJECTED",
            "claim_scope": "Near-ISCO fixed-Keplerian total-energy coupling",
            "establishes": (
                "At N>=256 the alpha-viscosity mismatch is localized to the "
                "invalid near-ISCO cells even though the total-energy row closes."
            ),
        },
    )

    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(10.0 * potential.r_g, 335.0 * potential.r_g, N)
    stream_rate = 5.0 * eddington_mdot(mass)
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
    closure = SignedThermalClosure(temperature_bounds=(1.0e3, 1.0e9))
    accepted = _prepare("signed_flux_total_energy_rin10_N512")
    summaries = {}
    for outer_mode in ("tidal_wall", "zero_torque"):
        result = solve_signed_total_energy_thermoviscous_steady(
            grid,
            mass,
            alpha=0.01,
            boundary=SignedFluxBoundary(outer_mode=outer_mode),
            stream_state=stream,
            closure=closure,
            temperature_seed=np.full(N, 1.0e6),
            damping=0.2,
            tolerance=2.0e-3,
            max_iterations=60,
            energy_tolerance=1.0e-6,
            energy_max_nfev=1000,
        )
        if not result.converged:
            raise RuntimeError(f"Rin=10 rg {outer_mode} control did not converge")
        _save_state(accepted / f"{outer_mode}.npz", result, grid)
        summaries[outer_mode] = _summary(
            result, grid, mass, stream_rate, stream_l
        )
    _write_json(
        accepted / "config.json",
        {
            "N": N,
            "R_in_rg": 10.0,
            "R_out_rg": 335.0,
            "stream_rate_over_edd": 5.0,
            "stream_center_rg": 240.0,
            "stream_circularization_rg": 248.96693,
            "alpha": 0.01,
        },
    )
    _write_json(accepted / "summary.json", summaries)
    _finalize(
        accepted,
        {
            "generation_command": (
                "PYTHONPATH=src python3 scripts/build_signed_flux_wp2_canonical.py"
            ),
            "source_parent_commit": "248e43c",
            "energy_identity_revision": "enthalpy_vertical_work_v2",
            "supersedes": "the mixed enthalpy/internal-work payload in 248e43c",
            "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
            "physical_status": "DIAGNOSTIC ONLY",
            "claim_scope": "Rin=10 rg total-energy interface controls",
            "does_not_establish": (
                "A physical hot branch or a valid inner match; radial pressure "
                "support still exceeds the production gate near the interface."
            ),
        },
    )
    _rebuild_manifest()


if __name__ == "__main__":
    run()
