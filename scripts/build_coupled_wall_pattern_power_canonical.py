"""Package the finite-minidisk wall pattern-power decision gate."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
CASE = CANONICAL / "coupled_wall_pattern_power"
SUMMARY_SOURCE = (
    ROOT / "outputs/tables/coupled_wall_pattern_power_continuation.json"
)
STATE_SOURCE = ROOT / "outputs/checkpoints/coupled_wall_pattern_power_full.npz"
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
            "latest_source_parent_commit": "0667263",
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
    summary = json.loads(SUMMARY_SOURCE.read_text())
    if summary["next_stage"] != "promote_inner_mdot_and_test_open_overflow":
        raise RuntimeError("pattern-power decision gate has not selected overflow")

    CASE.mkdir(parents=True, exist_ok=True)
    for path in CASE.iterdir():
        if path.is_file():
            path.unlink()
    shutil.copy2(SUMMARY_SOURCE, CASE / "summary.json")
    shutil.copy2(STATE_SOURCE, CASE / "last_accepted_state.npz")
    _write_json(
        CASE / "config.json",
        {
            "reservoir_outer_radius_rg": 335.0,
            "hill_radius_rg": summary["hill_radius_secondary_rg"],
            "pattern_omega_s_inverse": summary["rows"][0][
                "binary_pattern_omega"
            ],
            "tidal_kernel_onset_hill_fraction": 0.35,
            "power_fractions": summary["power_stages_requested"],
            "tidal_band_validity_gate_H_over_R": 0.3,
            "wind": False,
        },
    )
    payload = sorted(path for path in CASE.iterdir() if path.is_file())
    _write_json(
        CASE / "provenance.json",
        {
            "solver_generation_command": (
                "PYTHONPATH=src python3 "
                "scripts/run_coupled_wall_pattern_power_continuation.py"
            ),
            "canonical_packaging_command": (
                "PYTHONPATH=src python3 "
                "scripts/build_coupled_wall_pattern_power_canonical.py"
            ),
            "source_parent_commit": "0667263",
            "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
            "physical_status": "REJECTED",
            "claim_scope": (
                "Paired wall torque/pattern-speed power continuation on the "
                "fully coupled finite 335 rg minidisk"
            ),
            "establishes": (
                "The disk-rate wall control is mesh supported, but depositing "
                "25% of the differential pattern work makes the tidal band "
                "geometrically thick; the confined one-zone continuation is "
                "outside its declared validity regime before full power."
            ),
            "does_not_establish": (
                "Nonexistence of a physical tidally interacting state, because "
                "overflow, a free inner accretion rate, and multidimensional "
                "tidal transport are not represented."
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
