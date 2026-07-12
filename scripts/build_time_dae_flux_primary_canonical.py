"""Package the time-DAE boundary, rank, and step evidence canonically."""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tables"
TARGET = ROOT / "results/canonical/time_dae_flux_primary_prototype"
CANONICAL = ROOT / "results/canonical"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FILES = {
    "endpoint_audit.json": "coupled_open_edge_asymptotic_audit.json",
    "boundary_rank_comparison.json": "time_dae_boundary_rank_prototype.json",
    "flux_primary_rank.json": "time_dae_flux_primary_rank_prototype.json",
    "backward_euler_steps.json": "time_dae_flux_primary_step_prototype.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    existing = json.loads(CANONICAL_SUMMARY.read_text())
    existing.update(
        {
            "latest_source_parent_commit": "b9b1bc1",
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
        }
    )
    CANONICAL_SUMMARY.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    missing = [source for source in FILES.values() if not (SOURCE / source).exists()]
    if missing:
        raise FileNotFoundError(f"missing time-DAE source outputs: {missing}")
    TARGET.mkdir(parents=True, exist_ok=True)
    for destination, source in FILES.items():
        shutil.copyfile(SOURCE / source, TARGET / destination)
    config = {
        "architecture": "flux_primary_low_mach_dae",
        "outer_face_torque": "G = Mdot*l - J",
        "coupled_unknown_count": "2*Ni + 5*No + 5",
        "small_rank_meshes": [8, 12, 16],
        "backward_euler_meshes": [16, 32],
        "wind": False,
        "tide": False,
    }
    provenance = {
        "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
        "physical_status": "DIAGNOSTIC ONLY",
        "claim_scope": "outer-only flux-primary DAE architecture prototype",
        "source_parent_commit": "b9b1bc1",
        "generation_commands": [
            "python scripts/run_coupled_open_edge_asymptotic_audit.py",
            "python scripts/run_time_dae_boundary_rank_prototype.py",
            "python scripts/run_time_dae_flux_primary_rank_prototype.py",
            "python scripts/run_time_dae_flux_primary_step_prototype.py",
        ],
        "packaging_command": "python scripts/build_time_dae_flux_primary_canonical.py",
    }
    (TARGET / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (TARGET / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    checksum_files = sorted(
        path for path in TARGET.iterdir() if path.name != "SHA256SUMS.txt"
    )
    checksums = "".join(
        f"{_sha256(path)}  {path.name}\n" for path in checksum_files
    )
    (TARGET / "SHA256SUMS.txt").write_text(checksums)
    _rebuild_manifest()
    print(f"Wrote {len(checksum_files)} canonical files to {TARGET}")


if __name__ == "__main__":
    main()
