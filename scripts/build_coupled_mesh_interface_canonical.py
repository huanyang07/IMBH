"""Package coupled mesh and interface-position certification evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "coupled_mesh_interface_certification"
MESH_SUMMARY = ROOT / "outputs/tables/coupled_inner_outer_mesh_certification.json"
INTERFACE_SUMMARY = ROOT / "outputs/tables/coupled_inner_outer_interface_continuation.json"
MESH_STATES = ROOT / "outputs/checkpoints/coupled_inner_outer_mesh_certification"
INTERFACE_STATES = ROOT / "outputs/checkpoints/coupled_inner_outer_interface_continuation"
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
            "latest_source_parent_commit": "ab3f751",
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
        }
    )
    _write_json(CANONICAL_SUMMARY, existing)


def run() -> None:
    for source in (MESH_SUMMARY, INTERFACE_SUMMARY):
        if not source.is_file():
            raise FileNotFoundError(f"missing production result: {source}")
    mesh = json.loads(MESH_SUMMARY.read_text())
    interface = json.loads(INTERFACE_SUMMARY.read_text())
    if not mesh["mesh_certification_gate"]:
        raise RuntimeError("mesh result does not pass its declared gate")
    if not interface["interface_position_gate"]:
        raise RuntimeError("interface result does not pass its declared gate")

    CASE.mkdir(parents=True, exist_ok=True)
    for path in CASE.iterdir():
        if path.is_file():
            path.unlink()
    shutil.copy2(MESH_SUMMARY, CASE / "mesh_summary.json")
    shutil.copy2(INTERFACE_SUMMARY, CASE / "interface_summary.json")
    for source in sorted(MESH_STATES.glob("*.npz")):
        shutil.copy2(source, CASE / f"mesh_{source.name}")
    for source in sorted(INTERFACE_STATES.glob("*.npz")):
        shutil.copy2(source, CASE / f"interface_{source.name}")

    _write_json(
        CASE / "config.json",
        {
            "mesh_sequence": [[96, 64], [144, 96], [192, 128]],
            "interface_targets_rg": [35.0, 40.0, 45.0, 50.0],
            "mesh_restart_policy": "prolongate_previous_full_mu1_root",
            "interface_continuation_policy": (
                "fork inward from 40; continue outward 40 to 45 to 50"
            ),
            "luminosity_spread_gate": 0.01,
            "thickness_spread_gate": 0.02,
            "thickness_invariance_metric": (
                "maximum H/R over fixed R >= 60 rg band"
            ),
            "primitive_audit_gate": 0.01,
            "wind": False,
            "outer_boundary": "ideal_tidal_wall_control",
        },
    )
    payload = sorted(path for path in CASE.iterdir() if path.is_file())
    _write_json(
        CASE / "provenance.json",
        {
            "solver_generation_commands": [
                "PYTHONPATH=src python3 scripts/run_coupled_inner_outer_mesh_certification.py",
                "PYTHONPATH=src python3 scripts/run_coupled_inner_outer_interface_continuation.py",
            ],
            "canonical_packaging_command": (
                "PYTHONPATH=src python3 "
                "scripts/build_coupled_mesh_interface_canonical.py"
            ),
            "source_parent_commit": "ab3f751",
            "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
            "physical_status": "DIAGNOSTIC ONLY",
            "claim_scope": (
                "Mesh and numerical-interface certification of the fully "
                "coupled no-wind ideal-wall control"
            ),
            "establishes": (
                "Chained full-root convergence through Ninner192/Nouter128 and "
                "full-rank roots at 35-50 rg with invariant luminosity and "
                "fixed-band thickness metrics."
            ),
            "does_not_establish": (
                "A physical tidal torque and power, endpoint-stencil "
                "independence, stability, time evolution, or wind."
            ),
            "retained_negative_diagnostic": (
                "The raw maximum over the moving outer domain varies by 3.90%; "
                "it is not used as an interface-invariance metric because the "
                "domain itself changes."
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
