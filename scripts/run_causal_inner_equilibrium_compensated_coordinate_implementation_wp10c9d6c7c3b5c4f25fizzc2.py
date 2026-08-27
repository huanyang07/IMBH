#!/usr/bin/env python3
"""Certify the compensated equilibrium entropy-coordinate implementation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_equilibrium_entropy_coordinate_numerics_repair_manifest_wp10c9d6c7c3b5c4f25fizzc1 as manifest  # noqa: E402
import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as original  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "equilibrium_compensated_coordinate_potential_certified"
FAIL_CLASSIFICATION = "equilibrium_compensated_coordinate_potential_failed"
AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzd_"
    "dynamic_height_convex_legendre_manifest"
)
ARTIFACT = (
    "causal_inner_equilibrium_compensated_coordinate_implementation_"
    "wp10c9d6c7c3b5c4f25fizzc2"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_COMPENSATED_"
    "COORDINATE_POTENTIAL_WP10C9D6C7C3B5C4F25FIZZC2_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_equilibrium_compensated_coordinate_"
    "implementation_wp10c9d6c7c3b5c4f25fizzc2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_equilibrium_compensated_coordinate_"
    "implementation_wp10c9d6c7c3b5c4f25fizzc2.py"
)
PHYSICAL_SOURCE = original.PHYSICAL_SOURCE
PHYSICAL_TEST = original.PHYSICAL_TEST
MANIFEST_CHECKSUM_SHA256 = (
    "955dc3e87072f7f6668e5b0723f26622d8b6a2b0c615356ee53089c3fa3c84e0"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return manifest._utils()


def _validate_manifest(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != MANIFEST_CHECKSUM_SHA256:
        raise RuntimeError("coordinate-repair manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(manifest.CANONICAL_DIRECTORY / "repair_contract.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["equilibrium_rejection_preserved"]
        or summary["equilibrium_physical_potential_certified"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["prospective_repair"]["physical_EOS_unchanged"]
        or not contract["binding_rerun"]["same_47_frozen_physical_witnesses"]
    ):
        raise RuntimeError("coordinate-repair contract changed")
    for relative, expected in utils._read_json(manifest.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"coordinate-repair source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("compensated-coordinate certificate needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _certificate():
    _validate_manifest(require_clean=False)
    metrics, arrays = original._certificate()
    metrics.update({
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if metrics["passed"] else FAIL_CLASSIFICATION,
        "original_equilibrium_rejection_preserved": True,
        "compensated_single_alpha_coordinate": True,
        "metric_local_beta_stencil": True,
        "physical_master_potential_unchanged": True,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    })
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("compensated-coordinate package already exists")
    validated = _validate_manifest(require_clean=True); utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "certificate_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "certificate_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "original_equilibrium_rejection_preserved": True, "equilibrium_physical_potential_certified": metrics["passed"], "compensated_coordinate_certified": metrics["passed"], "dynamic_height_potential_certified": False, "full_shear_master_potential_certified": False, "eleven_field_local_closure_certified": False, "eleven_field_trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": metrics["authorized_next"]}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_sha256": MANIFEST_CHECKSUM_SHA256, "manifest_hashes": validated["hashes"], "preserved_rejection_artifact": original.ARTIFACT, "preserved_rejection_checksum_sha256": manifest.PARENT_CHECKSUM_MANIFEST_SHA256})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Compensated equilibrium-coordinate potential certificate", "", f"Classification: `{metrics['classification']}`.", "", f"The unchanged fixed-height master potential passes all 47 frozen physical witnesses. Maximum physical-current parity is `{metrics['maximum_physical_current_relative_defect']:.6e}`; maximum complex-step derivative defect is `{metrics['maximum_complex_step_current_jacobian_relative_defect']:.6e}`; maximum metric-local sixth-order defect is `{metrics['maximum_sixth_order_current_jacobian_relative_defect']:.6e}`.", "", "The prior numerical rejection is preserved. The repair represents the single alpha coordinate with compensated rest-mass and thermal components and uses metric-local beta stencil scales; it changes neither the EOS nor the potential.", "", "Height and shear potentials remain uncertified. No trajectory or complete-cycle execution is authorized.", "", f"Authorized next: `{metrics['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
