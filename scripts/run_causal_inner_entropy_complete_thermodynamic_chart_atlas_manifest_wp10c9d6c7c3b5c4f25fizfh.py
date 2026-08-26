#!/usr/bin/env python3
"""Freeze thermodynamic-chart recovery of the conservative macro atlas."""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_conservative_macro_atlas_execution_wp10c9d6c7c3b5c4f25fizfg as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fizfh_entropy_complete_thermodynamic_chart_atlas_recovery_manifest"
CLASSIFICATION = "entropy_complete_thermodynamic_chart_conservative_atlas_recovery_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizfi_entropy_complete_thermodynamic_chart_atlas_execution"
ARTIFACT = "causal_inner_entropy_complete_thermodynamic_chart_atlas_manifest_wp10c9d6c7c3b5c4f25fizfh"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_THERMODYNAMIC_CHART_ATLAS_MANIFEST_WP10C9D6C7C3B5C4F25FIZFH_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_thermodynamic_chart_atlas_manifest_wp10c9d6c7c3b5c4f25fizfh.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_thermodynamic_chart_atlas_manifest_wp10c9d6c7c3b5c4f25fizfh.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "f491e80fcfbbe0c37d547e7bd83c87cd108848c014501cb45be81503e149ae72"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils(): return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256: raise RuntimeError("raw-coordinate atlas rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "macro_atlas_metrics.json")
    failure = metrics.get("failure", {})
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["authorized_next"] is not None
        or not summary["hydrostatic_implicit_inverse_tangent_preserved"]
        or failure.get("stage") != "first_colored_raw_M_coordinate_plus_lift"
        or failure.get("type") != "LinAlgError"
        or metrics["new_truth_operator_calls"] != 0
        or metrics["propagated_states"] != 0
    ): raise RuntimeError("raw-coordinate atlas rejection classification changed")
    for relative, expected in utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected: raise RuntimeError(f"raw-coordinate atlas source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"): raise RuntimeError("thermodynamic-chart manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_rejection": {
            "raw_independent_MJE_coordinate_atlas_rejected": True,
            "conservative_16_cell_macro_state_rejected": False,
            "hydrostatic_inverse_rejected": False,
            "new_truth_operator_calls_in_rejected_run": 0,
        },
        "coordinate_redesign": {
            "stored_online_state": "exact_16_cell_(M,J,E,beta_r,chi)",
            "local_atlas_coordinates": ("delta_lnSigma", "delta_beta_phi", "delta_lnT", "delta_beta_r", "delta_chi"),
            "fine_chart_scales": (1.0, 0.1, 1.0, 0.1, 1.0e-4),
            "same_shift_applied_to_seven_fine_cells_in_each_macro_block": True,
            "height_recomputed_by_hydrostatic_EOS": True,
            "vertical_velocity_over_c": 0.0,
            "online_macro_to_chart_pullback": "inverse_of_cellwise_5x5_normalized_macro_chart_tangent",
            "pullback_derivative_step": 1.0e-5,
            "maximum_pullback_condition_number": 1.0e5,
            "exact_MJE_storage_and_face_flux_conservation_unchanged": True,
        },
        "atlas_audit": {
            "primary_training_anchor": "primary_20ms_base",
            "strict_blind_profiles": ("heldout_16ms_base", "heldout_16ms_perturbed"),
            "cell_colors_per_field": 3,
            "central_colored_chart_step": 2.0e-2,
            "colored_truth_calls": 30,
            "independent_JVP_directions": 4,
            "independent_JVP_chart_step": 1.0e-2,
            "independent_JVP_truth_calls": 8,
            "maximum_new_truth_operator_calls": 38,
            "maximum_independent_JVP_relative_defect": 5.0e-2,
            "maximum_saved_near_witness_output_relative_defect": 1.0e-2,
            "maximum_blind_output_relative_defect_per_block": 5.0e-2,
            "maximum_blind_macro_rate_relative_defect_per_field": 5.0e-2,
            "maximum_blind_inferred_chart_coordinate_infinity": 1.5e-1,
            "maximum_truth_state_constraint_relative_defect": 1.0e-10,
            "fail_on_any_truth_physical_or_hyperbolicity_gate": True,
        },
        "online_cost": {
            "state_dimension": 80,
            "truth_calls_per_macrostep": 0,
            "benchmark_evaluations": 100000,
            "maximum_benchmark_wall_seconds": 10.0,
            "maximum_macrosteps_per_cycle": 100000,
        },
        "decision": {
            "pass": "authorize_definitions_only_structure_preserving_macro_integrator_manifest",
            "fail": "require_multiple_entropy_complete_training_anchors_or_nonlinear_local_chart_atlas",
        },
        "claim_boundary": {
            "chart_atlas_execution_authorized": True,
            "state_propagation_authorized": False,
            "macro_integrator_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("thermodynamic-chart atlas manifest already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "thermodynamic_chart_atlas_contract.json", _contract())
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "raw_coordinate_atlas_rejection_preserved": True, "thermodynamic_chart_atlas_execution_authorized": True, "state_propagation_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"]})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete thermodynamic-chart atlas recovery manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The raw independent M/J/E perturbation remains rejected. Exact M/J/E remain the online storage, but atlas samples are generated in admissible local thermodynamic charts and pulled back through a certified 5x5 cellwise tangent.", "", "The same 38-call, held-out, conservation, physics, and online-cost gates remain binding. No propagation is authorized.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
