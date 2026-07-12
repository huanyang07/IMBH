"""Package the directly coupled open time-DAE evidence canonically."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from build_time_dae_flux_primary_canonical import _rebuild_manifest, _sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "summary.json": ROOT / "outputs/tables/time_dae_coupled_open_evolution.json",
    "repeated_steps.json": ROOT
    / "outputs/tables/time_dae_coupled_open_repeated.json",
    "evolved_mesh.json": ROOT
    / "outputs/tables/time_dae_coupled_open_evolved_mesh.json",
}
TARGET = ROOT / "results/canonical/time_dae_coupled_open_prototype"


def main() -> None:
    missing = [path for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing coupled time-DAE outputs: {missing}")
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        shutil.copyfile(source, TARGET / name)
    config = {
        "architecture": "direct_inner_flux_primary_outer_backward_euler",
        "unknown_count": "2*Ni + 5*No + 5",
        "meshes": [[16, 8], [24, 16]],
        "dt_over_loading_time": 1.0e-9,
        "repeated_step_dt_over_loading_time": 1.25e-8,
        "repeated_steps": 8,
        "interface_stencil_fraction": 1.0,
        "interface_stencil_homotopy": [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.4,
            0.7,
            1.0,
        ],
        "absolute_stream_source": True,
        "radiative_cooling": True,
        "tide": False,
        "wind": False,
    }
    provenance = {
        "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
        "physical_status": "DIAGNOSTIC ONLY",
        "claim_scope": (
            "accepted small-mesh direct inner-outer backward-Euler steps, "
            "resolved timestep convergence, coarse restart control, and a "
            "cross-interface radial stencil; fine evolved-mesh and long "
            "evolution are not certified"
        ),
        "source_parent_commit": "b9b1bc1",
        "generation_command": (
            "python scripts/run_time_dae_coupled_open_evolution.py"
            "; python scripts/run_time_dae_coupled_open_repeated.py"
            "; python scripts/run_time_dae_coupled_open_evolved_mesh.py"
        ),
        "packaging_command": (
            "python scripts/build_time_dae_coupled_open_canonical.py"
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
