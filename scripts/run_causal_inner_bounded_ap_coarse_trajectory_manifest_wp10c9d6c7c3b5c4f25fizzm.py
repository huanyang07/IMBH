#!/usr/bin/env python3
"""Freeze the bounded eleven-field AP coarse-trajectory experiment."""

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
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

import run_causal_inner_physical_entropy_congruence_and_ap_kernel_wp10c9d6c7c3b5c4f25fizzl1 as parent  # noqa: E402


WORK_PACKAGE = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzm_"
    "bounded_AP_coarse_trajectory_manifest"
)
CLASSIFICATION = "bounded_AP_coarse_trajectory_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzm1_bounded_AP_coarse_trajectory_kernel"
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzn_"
    "complete_cycle_preexecution_manifest"
)
ARTIFACT = (
    "causal_inner_bounded_ap_coarse_trajectory_manifest_"
    "wp10c9d6c7c3b5c4f25fizzm"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_BOUNDED_AP_COARSE_TRAJECTORY_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZZM_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_bounded_ap_coarse_trajectory_manifest_"
    "wp10c9d6c7c3b5c4f25fizzm.py"
)
THIS_TEST = (
    "tests/test_causal_inner_bounded_ap_coarse_trajectory_manifest_"
    "wp10c9d6c7c3b5c4f25fizzm.py"
)
PARENT_SHA256 = "294f08584f496ce90cf01ba65bbd4f8c027f5e581a6576017f66205a7853f799"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("physical congruence/AP kernel checksum changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["physical_entropy_congruence_certified"]
        or not summary["corrected_Kerr_Schild_eleven_field_port_certified"]
        or not summary["AP_macrostep_kernel_certified"]
        or not summary["bounded_AP_coarse_trajectory_manifest_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("physical congruence/AP kernel classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("bounded AP trajectory manifest needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "offline_physical_atlas": {
            "case_anchor_indices": {"primary": [0, 10], "held_out": [20, 30]},
            "online_truth_calls": 0,
            "interpolation": "convex midpoint interpolation of hash-locked symmetric A and dissipative S",
            "claim": "local two-anchor interpolation only; no cycle-wide atlas is implied",
        },
        "trajectory": {
            "coordinates": "complex Fourier amplitudes in the corrected eleven-field entropy port",
            "generator": "L(t)=-i*kappa(t)*A(t)+stiffness*S(t)",
            "integrator": "exponential midpoint affine macrostep",
            "normalized_horizon": 2.0,
            "stiffness_ratios": [1.0, 100.0, 1000.0],
            "step_counts": [8, 16, 32],
            "reference_step_count": 128,
            "checkpoint_fraction": 0.5,
            "initial_fast_perturbation_nonzero": True,
            "smooth_slow_forcing_nonzero": True,
        },
        "gates": {
            "minimum_matched_refinement_order": 1.7,
            "maximum_homogeneous_step_expansivity": 2.0e-10,
            "maximum_state_norm": 0.5,
            "maximum_stiff_fast_slaving_defect": 2.0e-2,
            "required_source_nullity": 4,
            "checkpoint_roundtrip": "bitwise",
            "suffix_replay": "bitwise",
            "maximum_projected_100k_step_wall_days": 3.0,
            "online_truth_calls": 0,
        },
        "decision": {
            "pass_classification": "bounded_AP_coarse_trajectory_kernel_certified",
            "failure_classification": "bounded_AP_coarse_trajectory_kernel_failed",
            "pass_authorized_next": PASS_NEXT,
        },
        "claim_boundary": {
            "physical_nonlinear_cycle_trajectory_certified": False,
            "cycle_wide_coefficient_atlas_complete": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("bounded AP trajectory manifest exists")
    hashes = _validate_parent(require_clean=True); utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "trajectory_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "bounded_AP_coarse_trajectory_certified": False, "cycle_wide_coefficient_atlas_complete": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("# Bounded AP coarse-trajectory manifest\n\nThis package freezes a two-anchor, eleven-field, exponentially integrated AP trajectory test at primary and held-out physical states. It binds matched refinement, source-nullity preservation, stiff fast-manifold slaving, bitwise arbitrary-step restart, zero online truth calls, and a measured 100,000-step cost projection.\n\nThe test is local and linearized. It does not supply a cycle-wide physical coefficient atlas and does not authorize complete-cycle execution.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); arguments = parser.parse_args()
    if not arguments.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
