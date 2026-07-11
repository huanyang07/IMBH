"""Package common-stress and simultaneous-reservoir evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "common_stress_simultaneous_reservoir"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
SUMMARY = ROOT / "results/manifests/canonical_summary.json"
COMMON_SUMMARY = ROOT / "outputs/tables/common_stress_interface_sweep.json"
NONKEPLERIAN_SUMMARY = (
    ROOT / "outputs/tables/nonkeplerian_common_stress_sweep.json"
)
COMMON_STATES = ROOT / "outputs/checkpoints/common_stress_interface_sweep"
NONKEPLERIAN_STATES = (
    ROOT / "outputs/checkpoints/nonkeplerian_common_stress_sweep"
)


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
    _write_json(
        SUMMARY,
        {
            "legacy_source_commit": "0a000767a915880c0710b8f4ec03eb0c64aa168a",
            "legacy_source_tag": "pre-cleanup-p0-2026-07-11",
            "latest_source_parent_commit": "5d36c24",
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
        },
    )


def run() -> None:
    for source in (COMMON_SUMMARY, NONKEPLERIAN_SUMMARY):
        if not source.is_file():
            raise FileNotFoundError(f"missing production result: {source}")
    CASE.mkdir(parents=True, exist_ok=True)
    for path in CASE.iterdir():
        if path.is_file():
            path.unlink()

    shutil.copy2(COMMON_SUMMARY, CASE / "common_stress_summary.json")
    shutil.copy2(
        NONKEPLERIAN_SUMMARY, CASE / "nonkeplerian_summary.json"
    )
    for source in sorted(COMMON_STATES.glob("R*_N256.npz")):
        shutil.copy2(source, CASE / f"common_{source.name}")
    for source in sorted(NONKEPLERIAN_STATES.glob("R*_N256.npz")):
        shutil.copy2(source, CASE / f"nonkeplerian_{source.name}")

    _write_json(
        CASE / "config.json",
        {
            "alpha": 0.01,
            "mu_stress": 0.0,
            "stress_factor": 1.0,
            "interface_targets_rg": [30.0, 40.0, 50.0, 60.0],
            "reservoir_resolutions": [64, 128, 256],
            "common_stress_stages": [0.0, 0.25, 0.5, 0.75, 1.0],
            "radial_support_stages": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
            "primitive_gate_common": 0.10,
            "primitive_gate_simultaneous": 0.05,
        },
    )
    payload = sorted(path for path in CASE.iterdir() if path.is_file())
    provenance = {
        "solver_generation_commands": [
            "PYTHONPATH=src python3 scripts/run_common_stress_interface_sweep.py",
            "PYTHONPATH=src python3 scripts/run_nonkeplerian_common_stress_sweep.py",
        ],
        "canonical_packaging_command": (
            "PYTHONPATH=src python3 scripts/build_common_stress_audit_canonical.py"
        ),
        "source_parent_commit": "5d36c24",
        "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
        "physical_status": "DIAGNOSTIC ONLY",
        "claim_scope": (
            "Common-stress fixed-Keplerian control and simultaneous "
            "non-Keplerian prescribed-flux reservoir"
        ),
        "establishes": (
            "Stress parity explains much of the old pressure jump; simultaneous "
            "40-60 rg roots close stress, radial momentum, energy, and fluxes."
        ),
        "does_not_establish": (
            "A fully coupled smooth inner-outer branch, physical tidal closure, "
            "stability, time evolution, or wind."
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
