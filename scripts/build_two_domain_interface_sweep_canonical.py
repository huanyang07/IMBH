"""Package the prescribed-flux two-domain interface sweep."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from run_two_domain_interface_sweep import run as run_interface_sweep


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "two_domain_interface_sweep"
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
    CASE.mkdir(parents=True, exist_ok=True)
    for path in CASE.iterdir():
        if path.is_file():
            path.unlink()
    summary = run_interface_sweep()
    _write_json(CASE / "summary.json", summary)
    _write_json(
        CASE / "config.json",
        {
            "interface_targets_rg": [30.0, 40.0, 50.0, 60.0],
            "reservoir_resolutions": [128, 256],
            "outer_boundary": "tidal_wall",
            "flux_gate": 1.0e-10,
            "composite_luminosity_position_spread_gate": 0.01,
            "primitive_log_mismatch_gate": 0.10,
        },
    )
    payload = [CASE / "config.json", CASE / "summary.json"]
    provenance = {
        "solver_generation_command": (
            "PYTHONPATH=src python3 scripts/run_two_domain_interface_sweep.py"
        ),
        "canonical_packaging_command": (
            "PYTHONPATH=src python3 scripts/build_two_domain_interface_sweep_canonical.py"
        ),
        "source_parent_commit": "02d7336",
        "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
        "physical_status": "DIAGNOSTIC ONLY",
        "claim_scope": "One-way no-wind transonic-to-tidal-wall conserved-flux composite",
        "establishes": (
            "Flux closure, mesh support, and interface-position-stable composite luminosity."
        ),
        "does_not_establish": (
            "A smooth physical domain match; the integrated-pressure state mismatch fails."
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
