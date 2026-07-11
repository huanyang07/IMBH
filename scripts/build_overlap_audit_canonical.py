"""Package the inner/reservoir overlap audit as a canonical result."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from run_inner_outer_overlap_audit import run as run_overlap_audit


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "inner_outer_overlap_audit"
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
    summary = run_overlap_audit()
    _write_json(CASE / "summary.json", summary)
    _write_json(
        CASE / "config.json",
        {
            "window_rg": [12.0, 60.0],
            "strict_radial_pressure_fraction": 0.05,
            "sensitivity_radial_pressure_fraction": 0.10,
            "effective_optical_depth_definition": "sqrt(tau_abs*(tau_abs+tau_es))",
            "absorption_opacity_status": "diagnostic Kramers bracket only",
        },
    )
    payload = [CASE / "config.json", CASE / "summary.json"]
    provenance = {
        "solver_generation_command": (
            "PYTHONPATH=src python3 scripts/run_inner_outer_overlap_audit.py"
        ),
        "canonical_packaging_command": (
            "PYTHONPATH=src python3 scripts/build_overlap_audit_canonical.py"
        ),
        "source_parent_commit": "c03f640",
        "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
        "physical_status": "DIAGNOSTIC ONLY",
        "claim_scope": "Inner transonic and corrected Rin=10 rg reservoir overlap audit",
        "establishes": (
            "No common band passes the 5% radial-pressure gate; a threshold-sensitive "
            "candidate band exists for the 10% pressure sensitivity test."
        ),
        "does_not_establish": (
            "A physical two-domain match, calibrated absorption opacity, or primitive-state continuity."
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
