#!/usr/bin/env python3
"""Supersede the event-sheet simplex dimension before interpolation evidence."""

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

import run_causal_inner_cycle_physical_driver_branch_and_event_interpolator_manifest_wp10c9d6c7c3b5c4f25fizzv as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "cycle_interpolator_guard_sheet_dimension_correction_manifest_frozen"
AUTHORIZED_NEXT = parent.AUTHORIZED_NEXT
PASS_NEXT = parent.PASS_NEXT
ARTIFACT = (
    "causal_inner_cycle_interpolator_guard_sheet_dimension_correction_manifest_"
    "wp10c9d6c7c3b5c4f25fizzv0"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_INTERPOLATOR_GUARD_SHEET_"
    "DIMENSION_CORRECTION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZV0_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_cycle_interpolator_guard_sheet_dimension_"
    "correction_manifest_wp10c9d6c7c3b5c4f25fizzv0.py"
)
THIS_TEST = (
    "tests/test_causal_inner_cycle_interpolator_guard_sheet_dimension_"
    "correction_manifest_wp10c9d6c7c3b5c4f25fizzv0.py"
)
PARENT_SHA256 = "3060ed914a25d90354b18d09aab6623f0b92b145a34a7bc0a1ec3d193532a9f1"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return parent._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256: raise RuntimeError("cycle interpolator manifest changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY); summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json"); contract = utility._read_json(parent.CANONICAL_DIRECTORY / "interpolator_contract.json")
    if not summary["passed"] or not summary["definitions_only"] or not summary["schema_v2_guard_geometry_frozen"] or summary["cycle_interpolator_certified"] or summary["authorized_next"] != WORK_PACKAGE or summary["complete_cycle_execution_authorized"] or contract["schema_v2_extension"]["event_additions"]["event_simplices6"] is None: raise RuntimeError("cycle interpolator manifest classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"): raise RuntimeError("guard-sheet correction needs a clean tracked tree")
    return hashes, contract


def _contract() -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "supersedes_artifact": parent.ARTIFACT,
        "reason": "an event guard is codimension one in the five-dimensional reduced coordinate space",
        "preserved_without_change": [
            "periodic driver interpolation",
            "four-dimensional driver q simplices with five vertices",
            "five-dimensional mode-pure branch simplices with six vertices",
            "invariant-exact state correction",
            "convex radial/source/forcing interpolation",
            "common four-coordinate source kernel",
            "fail-closed hull, trust, gap, boundary, reset, and provenance gates",
            "all numerical tolerances and the no-cycle-execution boundary",
        ],
        "binding_correction": {
            "event_guard_ambient_dimension": 5,
            "event_guard_intrinsic_dimension": 4,
            "event_simplices5": "five truth vertices per class-pure piecewise-affine guard sheet element",
            "event_simplices6_forbidden": True,
            "locator": "nonnegative barycentric weights in the four-dimensional affine hull",
            "affine_hull_gate": "ambient reconstruction residual <= 2e-12",
            "oriented_normal": "null vector of the four edge tangents, oriented by the truth normals",
            "signed_guard": "oriented normal dot (query-guard projection)",
            "transversality": "oriented normal dot reduced flow has absolute value >= 1e-8",
            "curvature_policy": "adjacent elements may rotate normals; no off-sheet volumetric interpolation",
        },
        "claim_boundary": {
            "numerical_interpolator_executed": False,
            "physical_payloads_acquired": False,
            "heldout_physical_validation_complete": False,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
        },
        "authorized_next": AUTHORIZED_NEXT,
        "pass_authorized_next": PASS_NEXT,
    }


def _update(summary: dict):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("guard-sheet correction already exists")
    hashes, _ = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); utility._write_json(CANONICAL_DIRECTORY / "corrected_interpolator_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "supersedes_prior_interpolator_manifest": True, "event_guard_intrinsic_dimension": 4, "event_simplex_vertex_count": 5, "cycle_interpolator_certified": False, "physical_payloads_acquired": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": AUTHORIZED_NEXT}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(
        "# Cycle interpolator guard-sheet dimension correction\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "A transition guard is a four-dimensional sheet in the five-dimensional reduced coordinate space. The earlier definitions-only manifest incorrectly assigned six vertices, which span a five-dimensional volume. This prospective correction requires five-vertex, class-pure sheet simplices, an affine-hull residual gate, and an oriented normal used for signed guard values and transversality.\n\n"
        "No interpolator evidence or cycle step existed, so no numerical result is amended. All other interpolation, physical-input, and no-execution boundaries remain unchanged.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
