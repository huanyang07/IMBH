"""Package the pressure-supported interface pilot as compact evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tables/pressure_supported_interface_pilot.json"
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "pressure_supported_interface_pilot"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"


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
            handle,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            "run scripts/run_pressure_supported_interface_pilot.py first"
        )
    CASE.mkdir(parents=True, exist_ok=True)
    for path in CASE.iterdir():
        if path.is_file():
            path.unlink()
    result = json.loads(SOURCE.read_text())
    for row in result["rows"]:
        row.pop("iteration_history_tail", None)
    _write_json(CASE / "summary.json", result)
    _write_json(
        CASE / "config.json",
        {
            "interface_rg": 40.0,
            "resolutions": [64, 128],
            "pressure_support_stages": [0.10, 0.25, 0.50, 0.75, 1.0],
            "damping_values": [0.05, 0.10, 0.20],
            "smoothing_log_width_values": [0.04, 0.08, 0.16],
            "viscosity_and_rotation_tolerance": 0.002,
        },
    )
    payload = [CASE / "config.json", CASE / "summary.json"]
    provenance = {
        "solver_generation_command": (
            "PYTHONPATH=src python3 scripts/run_pressure_supported_interface_pilot.py"
        ),
        "canonical_packaging_command": (
            "PYTHONPATH=src python3 "
            "scripts/build_pressure_supported_interface_pilot_canonical.py"
        ),
        "source_parent_commit": "4b11435",
        "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
        "physical_status": "DIAGNOSTIC ONLY",
        "claim_scope": "Projected pressure-supported staggered reservoir iteration",
        "establishes": (
            "Full-pressure coarse N64 roots exist, but the closure is not mesh-supported at N128."
        ),
        "does_not_establish": (
            "A smooth transonic/reservoir match or a converged production pressure-supported disk."
        ),
        "payload_sha256": {path.name: _sha256(path) for path in payload},
    }
    _write_json(CASE / "provenance.json", provenance)
    files = sorted(path for path in CASE.iterdir() if path.name != "SHA256SUMS.txt")
    (CASE / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files)
    )
    _rebuild_manifest()


if __name__ == "__main__":
    run()
