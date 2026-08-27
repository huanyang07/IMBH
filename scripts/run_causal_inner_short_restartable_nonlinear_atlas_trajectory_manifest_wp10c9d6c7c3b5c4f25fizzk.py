#!/usr/bin/env python3
"""Freeze the short restartable nonlinear equilibrium-core trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_conservative_entropy_projection_microstep_kernel_wp10c9d6c7c3b5c4f25fizzj2 as parent  # noqa: E402


WORK_PACKAGE = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzk_"
    "short_restartable_nonlinear_atlas_trajectory_manifest"
)
CLASSIFICATION = "short_restartable_nonlinear_atlas_trajectory_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzk1_"
    "short_restartable_equilibrium_core_trajectory"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzl_"
    "physical_entropy_congruence_and_AP_macrostep_manifest"
)
ARTIFACT = (
    "causal_inner_short_restartable_nonlinear_atlas_trajectory_manifest_"
    "wp10c9d6c7c3b5c4f25fizzk"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_SHORT_RESTARTABLE_NONLINEAR_ATLAS_TRAJECTORY_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZZK_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_short_restartable_nonlinear_atlas_trajectory_"
    "manifest_wp10c9d6c7c3b5c4f25fizzk.py"
)
THIS_TEST = (
    "tests/test_causal_inner_short_restartable_nonlinear_atlas_trajectory_"
    "manifest_wp10c9d6c7c3b5c4f25fizzk.py"
)
PARENT_SHA256 = "1a148f12f22f795b65f3a60320b449a3066f014a1557ee26e4acbb9543a34d6b"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(require_clean=False):
    utility = _u()
    checksum = utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
    if checksum != PARENT_SHA256:
        raise RuntimeError("projection microstep certificate checksum changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["conservative_entropy_projection_microstep_certified"]
        or not summary[
            "short_restartable_nonlinear_atlas_trajectory_manifest_authorized"
        ]
        or summary["trajectory_authorized"]
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("projection microstep certificate classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("short trajectory manifest needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "scope": {
            "advanced_state": "three-cell periodic four-current equilibrium core",
            "nonlinear_flux": "certified compensated discrete-gradient flux",
            "time_step": "certified conservative entropy-projected midpoint RK2",
            "atlas_policy": "recover and reanchor every accepted endpoint",
            "excluded": (
                "the five shear and two height ports remain algebraic/split-kernel "
                "certificates until a physical entropy congruence is constructed"
            ),
        },
        "cases": {
            "primary": {"witness_index": 0, "patch_index": 0},
            "held_out": {"witness_index": 30, "patch_index": 1},
            "dimensionless_horizon": 0.32,
            "matched_ladder": [
                {"courant": 0.02, "steps": 16},
                {"courant": 0.01, "steps": 32},
                {"courant": 0.005, "steps": 64},
            ],
        },
        "restart": {
            "checkpoint": "middle-ladder state after 16 accepted steps",
            "payload": (
                "density, temperature, radial and azimuthal velocity in every "
                "cell; height; accepted-step count; accumulated Courant time; "
                "initial conserved totals and entropy; source/config hashes"
            ),
            "roundtrip": "bitwise primitive payload",
            "suffix_replay": "remaining 16 middle-ladder steps bitwise",
            "rejected_state_propagation": False,
        },
        "gates": {
            "all_microsteps_pass": True,
            "minimum_matched_endpoint_order": 1.8,
            "maximum_cumulative_conservation_defect": 2.0e-10,
            "maximum_cumulative_entropy_defect": 2.0e-10,
            "maximum_trust_radius_fraction": 1.0,
            "checkpoint_roundtrip_bitwise": True,
            "suffix_replay_bitwise": True,
            "state_robustness": "both primary and held-out cases pass",
        },
        "decision": {
            "pass_classification": (
                "short_restartable_equilibrium_core_trajectory_certified"
            ),
            "failure_classification": (
                "short_restartable_equilibrium_core_trajectory_failed"
            ),
            "pass_authorized_next": PASS_NEXT,
        },
        "claim_boundary": {
            "full_eleven_field_trajectory_certified": False,
            "physical_time_horizon_claimed": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary):
    utility = _u()
    rows = list(
        csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))
    )
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utility._sha256(path),
                    "scientific_status": "SUPPORTED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": CLASSIFICATION,
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": utility._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("short trajectory manifest exists")
    hashes = _validate_parent(require_clean=True)
    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "trajectory_contract.json", _contract())
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "equilibrium_core_trajectory_certified": False,
        "full_eleven_field_trajectory_certified": False,
        "physical_time_horizon_claimed": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_SHA256,
            "parent_hashes": hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Short restartable nonlinear-atlas trajectory manifest\n\n"
        "This definitions-only package freezes two 0.32-Courant-time, "
        "three-cell equilibrium-core trajectories with matched 16/32/64-step "
        "ladders and a bitwise halfway restart. It advances only the certified "
        "nonlinear four-current core.\n\n"
        "The five shear and two height ports are not silently identified with "
        "the normalized atlas coordinates: their physical entropy-congruence "
        "bridge remains the next architecture gate. No physical time horizon "
        "or complete-cycle execution is authorized.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {
                source: utility._sha256(ROOT / source) for source in sources
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
