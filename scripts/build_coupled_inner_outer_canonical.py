"""Package the first fully coupled inner/outer rank prototype."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "coupled_inner_outer_rank_prototype"
SUMMARY_SOURCE = ROOT / "outputs/tables/coupled_inner_outer_rank_prototype.json"
STATE_SOURCE = ROOT / "outputs/checkpoints/coupled_inner_outer_rank_prototype.npz"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rebuild_manifest() -> None:
    rows = []
    for case in sorted(path for path in CANONICAL.iterdir() if path.is_dir()):
        provenance = json.loads((case / "provenance.json").read_text())
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status"),
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
            handle,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    existing = json.loads(CANONICAL_SUMMARY.read_text())
    existing.update(
        {
            "latest_source_parent_commit": "1146e67",
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
        }
    )
    _write_json(CANONICAL_SUMMARY, existing)


def run() -> None:
    for source in (SUMMARY_SOURCE, STATE_SOURCE):
        if not source.is_file():
            raise FileNotFoundError(f"missing production result: {source}")
    CASE.mkdir(parents=True, exist_ok=True)
    for path in CASE.iterdir():
        if path.is_file():
            path.unlink()

    shutil.copy2(SUMMARY_SOURCE, CASE / "summary.json")
    shutil.copy2(STATE_SOURCE, CASE / "state.npz")
    _write_json(
        CASE / "config.json",
        {
            "interface_rg": 40.04153642035986,
            "inner_nodes": 96,
            "outer_cells": 64,
            "coupling_stages": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
            "hard_interface_conditions": ["log_surface_density", "log_temperature"],
            "wind": False,
            "outer_boundary": "ideal_tidal_wall_control",
            "stress": "shared_integrated_alpha_pressure",
        },
    )
    payload = sorted(path for path in CASE.iterdir() if path.is_file())
    _write_json(
        CASE / "provenance.json",
        {
            "solver_generation_command": (
                "PYTHONPATH=src python3 "
                "scripts/run_coupled_inner_outer_rank_prototype.py"
            ),
            "canonical_packaging_command": (
                "PYTHONPATH=src python3 "
                "scripts/build_coupled_inner_outer_canonical.py"
            ),
            "source_parent_commit": "1146e67",
            "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
            "physical_status": "DIAGNOSTIC ONLY",
            "claim_scope": (
                "First fully coupled no-wind inner-transonic/outer-reservoir "
                "control at 40.041536 rg"
            ),
            "establishes": (
                "A square 388-variable root with full scaled Jacobian rank, "
                "rank-two interface and sonic responses, conserved fluxes, and "
                "continuous Sigma and T at Ninner=96/Nouter=64."
            ),
            "does_not_establish": (
                "N128/N256 mesh certification, interface-position invariance, "
                "a physical tidal torque and power, stability, time evolution, "
                "or wind."
            ),
            "payload_sha256": {path.name: _sha256(path) for path in payload},
        },
    )
    files = sorted(
        path for path in CASE.iterdir() if path.name != "SHA256SUMS.txt"
    )
    (CASE / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files)
    )
    _rebuild_manifest()


if __name__ == "__main__":
    run()
