"""Package the one-domain signed descriptor prototype canonically."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from build_time_dae_flux_primary_canonical import _rebuild_manifest, _sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tables/global_signed_descriptor_rank.json"
STREAM_SOURCE = ROOT / "outputs/tables/global_signed_stream_preflight.json"
PHYSICAL_OPEN_SOURCE = ROOT / "outputs/tables/global_physical_open_preflight.json"
TARGET = ROOT / "results/canonical/global_signed_descriptor_prototype"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError("run the global signed descriptor prototype first")
    if not STREAM_SOURCE.exists():
        raise FileNotFoundError("run the global signed stream preflight first")
    if not PHYSICAL_OPEN_SOURCE.exists():
        raise FileNotFoundError("run the global physical open preflight first")
    TARGET.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, TARGET / "rank_and_ledger.json")
    shutil.copyfile(STREAM_SOURCE, TARGET / "stream_preflight.json")
    shutil.copyfile(
        PHYSICAL_OPEN_SOURCE, TARGET / "physical_open_preflight.json"
    )
    config = {
        "architecture": "one_domain_flux_primary_conservative_evolution",
        "differential_fields": [
            "cell_mass",
            "cell_radial_momentum",
            "cell_angular_momentum",
            "cell_total_energy",
        ],
        "algebraic_fields": [
            "face_mass_flux",
            "face_radial_momentum_flux",
            "face_angular_momentum_flux",
            "face_total_energy_flux",
        ],
        "state_count": "8*N + 4",
        "meshes": [8, 16, 32],
        "source_free_meshes": [16, 32, 64],
        "primitive_recovery": (
            "shared Paczynski-Wiita potential and one-zone vertical closure"
        ),
        "inviscid_radial_operator": (
            "smooth reconstructed Euler flux with paired cylindrical source"
        ),
        "shock_flux": (
            "gravity-compatible residual-equilibrium Rusanov diagnostic"
        ),
        "common_stress": (
            "shared alpha stress with outward +G and +Omega*G flux pairing; "
            "monolithic physical-state backward Euler with thermal-energy "
            "scaling and a rejected colored-sparsity audit"
        ),
        "radiative_cooling": "shared two-face diffusion sink, implicit",
        "stream_source": (
            "exact compact-C2 cell moments for one constant injected state"
        ),
        "physical_open_remap": (
            "32-point conservative annular remap with fixed mass-weighted "
            "mechanical reference correction; N16-N128 positive primitive "
            "recovery; N64 temporal gate; "
            "certified sparse N64/N96 evolved comparison; bounded conserved-"
            "donor outer-face reconstruction"
        ),
        "tide": False,
        "wind": False,
    }
    provenance = {
        "numerical_status": "DIAGNOSTIC ONLY",
        "physical_status": "DIAGNOSTIC ONLY",
        "claim_scope": (
            "exact four-field finite-volume ledgers, signed mass-flux crossing, "
            "descriptor rank, manufactured backward-Euler rank, and "
            "conservative/primitive thermodynamic round trip plus a smooth "
            "inviscid radial equilibrium audit, bounded Rusanov step, and "
            "source-free temporal and interior mesh-convergence preflights; "
            "the common-stress flux pair and two-mesh, eight-step monolithic "
            "implicit preflight plus radiative-cooling certification are "
            "implemented; exact constant-state stream moments and one source-"
            "bearing step are certified; the physical absolute-supply open "
            "control has an admissible conservative remap from N16, matches "
            "the coupled boundary fluxes at N96, and passes N64 timestep "
            "refinement; certified sparse N64/N96 steps pass individually "
            "but the legacy outer flux mesh gate fails by 0.02846 supply; one "
            "conserved-donor correction improves this to 0.01150 but narrowly "
            "fails the fixed gate; the characteristic audit finds one "
            "incoming acoustic mode but the 335 rg edge is only 0.4485 R_H "
            "and has no declared exterior thermodynamic state; a physical "
            "truncation invariant or modeled Hill/Roche overflow layer, "
            "remains open; the reference-state inner characteristic absorber "
            "and enthalpy-compatible "
            "radial/temporal column-energy identity is implemented and passes "
            "the physical N64/N96 tiny-step gates; the fixed mechanical "
            "reference passes 32/64-point quadrature and N64/N96 evolved-mesh "
            "gates without floors; calibrated "
            "physical tide/wind closures are not implemented"
        ),
        "source_parent_commit": "b9b1bc1",
        "generation_command": (
            "python scripts/run_global_signed_descriptor_rank_prototype.py"
        ),
        "stream_generation_command": (
            "python scripts/run_global_signed_stream_preflight.py"
        ),
        "physical_open_generation_command": (
            "PYTHONPATH=src:scripts python "
            "scripts/run_global_physical_open_preflight.py"
        ),
        "packaging_command": (
            "python scripts/build_global_signed_descriptor_canonical.py"
        ),
    }
    (TARGET / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (TARGET / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in TARGET.iterdir() if path.name != "SHA256SUMS.txt")
    (TARGET / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files)
    )
    _rebuild_manifest()
    print(f"Wrote {len(files)} canonical files to {TARGET}")


if __name__ == "__main__":
    main()
