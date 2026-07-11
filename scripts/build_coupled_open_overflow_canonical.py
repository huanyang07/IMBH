"""Package the coupled open-overflow eigenvalue decision gate."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "coupled_open_overflow_eigenvalue"
SUMMARY_SOURCE = ROOT / "outputs/tables/coupled_open_overflow_continuation.json"
STATE_SOURCES = ROOT / "outputs/checkpoints/coupled_open_overflow_continuation"
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
            "latest_source_parent_commit": "0700292",
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
        }
    )
    _write_json(CANONICAL_SUMMARY, existing)


def run() -> None:
    if not SUMMARY_SOURCE.is_file():
        raise FileNotFoundError(f"missing production result: {SUMMARY_SOURCE}")
    summary = json.loads(SUMMARY_SOURCE.read_text())
    if not summary["reached_open_boundary"]:
        raise RuntimeError("open boundary was not reached")
    if summary["mesh_gate"]:
        raise RuntimeError("expected retained mesh-failure witness is absent")
    if summary["next_stage"] != "coupled_mass_energy_time_evolution":
        raise RuntimeError("open-overflow decision did not select time evolution")

    CASE.mkdir(parents=True, exist_ok=True)
    for path in CASE.iterdir():
        if path.is_file():
            path.unlink()
    shutil.copy2(SUMMARY_SOURCE, CASE / "summary.json")
    for name in (
        "Ninner96_Nouter64.npz",
        "Ninner144_Nouter96.npz",
        "Ninner168_Nouter112.npz",
    ):
        source = STATE_SOURCES / name
        if not source.is_file():
            raise FileNotFoundError(f"missing open-overflow state: {source}")
        shutil.copy2(source, CASE / name)
    _write_json(
        CASE / "config.json",
        {
            "interface_rg": summary["interface_rg"],
            "reservoir_outer_radius_rg": summary["reservoir_outer_radius_rg"],
            "boundary_stages": summary["boundary_stages_requested"],
            "open_mesh_sequence": summary["open_mesh_sequence_requested"],
            "stream_supply_mdot_edd": 5.0,
            "wind": False,
            "outer_inflow_allowed": False,
        },
    )
    payload = sorted(path for path in CASE.iterdir() if path.is_file())
    _write_json(
        CASE / "provenance.json",
        {
            "solver_generation_command": (
                "PYTHONPATH=src python3 "
                "scripts/run_coupled_open_overflow_continuation.py"
            ),
            "canonical_packaging_command": (
                "PYTHONPATH=src python3 "
                "scripts/build_coupled_open_overflow_canonical.py"
            ),
            "source_parent_commit": "0700292",
            "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
            "physical_status": "DIAGNOSTIC ONLY",
            "claim_scope": (
                "Fully coupled finite-minidisk open-overflow control with "
                "Mdot_inner as an eigenvalue"
            ),
            "establishes": (
                "Full-rank 96/64 and 144/96 open roots with about 83.1% "
                "overflow, finite-density stagnation, and a thin Hill band."
            ),
            "does_not_establish": (
                "A mesh-certified steady open branch: the controlled 168/112 "
                "refinement fails in the outer stress/energy endpoint cells."
            ),
            "retained_negative_diagnostic": (
                "The Ninner168/Nouter112 state is a rejected mesh-refinement "
                "witness and selects coupled mass-energy time evolution."
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
