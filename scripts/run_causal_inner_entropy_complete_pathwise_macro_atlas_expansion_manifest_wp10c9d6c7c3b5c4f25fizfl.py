#!/usr/bin/env python3
"""Freeze one prospective pathwise thermodynamic macro-atlas expansion."""

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
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_structure_preserving_macro_integrator_implementation_wp10c9d6c7c3b5c4f25fizfk as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfl_"
    "entropy_complete_pathwise_macro_atlas_expansion_manifest"
)
CLASSIFICATION = (
    "entropy_complete_second_pathwise_macro_patch_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfm_"
    "entropy_complete_second_pathwise_macro_patch_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_pathwise_macro_atlas_expansion_manifest_"
    "wp10c9d6c7c3b5c4f25fizfl"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_PATHWISE_MACRO_"
    "ATLAS_EXPANSION_MANIFEST_WP10C9D6C7C3B5C4F25FIZFL_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_pathwise_macro_atlas_expansion_"
    "manifest_wp10c9d6c7c3b5c4f25fizfl.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_pathwise_macro_atlas_expansion_"
    "manifest_wp10c9d6c7c3b5c4f25fizfl.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "35a5a45e5485d496ec801a570ab7d92f52ee9edae9ce59aff1e168ef116a09a3"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("bounded macro-integrator checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "macro_integrator_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["exact_affine_macro_integrator_certified"]
        or not summary["online_cost_gate_passed"]
        or summary["accepted_macrosteps"] != 4
        or summary["accepted_horizon_seconds"] != 4.0e-3
        or not summary["pathwise_macro_atlas_expansion_manifest_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != (
            "definitions_only_WP10c9d6c7c3b5c4f25fizfl_"
            "entropy_complete_pathwise_macro_atlas_expansion_manifest"
        )
        or not metrics["suffix_replay_bitwise"]
        or not metrics["endpoint_truth_all_physical_gates_passed"]
        or metrics["endpoint_maximum_macro_rate_relative_defect"] > 5.0e-2
    ):
        raise RuntimeError("pathwise patch authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"bounded integrator source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("pathwise patch manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "anchor": {
            "patch_1": "certified_primary_20ms_thermodynamic_chart_atlas",
            "patch_2_macro_state": "certified_patch_1_exact_affine_4ms_endpoint",
            "patch_2_primitive_charts": "certified_exact_thermodynamic_reconstruction_at_4ms",
            "patch_2_base_output": "certified_full_truth_operator_at_4ms",
            "no_synthetic_or_projected_anchor": True,
        },
        "patch_2_construction": {
            "coordinate_system": "local_(lnSigma,beta_phi,lnT,beta_r,chi)",
            "pullback_derivative_step": 1.0e-5,
            "maximum_pullback_condition_number": 1.0e5,
            "cell_colors_per_field": 3,
            "central_colored_chart_step": 2.0e-2,
            "colored_truth_calls": 30,
            "independent_JVP_directions": 4,
            "independent_JVP_chart_step": 1.0e-2,
            "independent_JVP_truth_calls": 8,
            "maximum_independent_JVP_relative_defect": 5.0e-2,
            "maximum_colored_support_leakage_ratio": 1.0e-12,
            "atlas_trust_coordinate_infinity": 1.5e-1,
            "all_truth_physical_gates_binding": True,
        },
        "overlap_and_dynamic_validation": {
            "overlap_witness": "certified_patch_1_3ms_state",
            "maximum_interpatch_output_relative_defect_per_block": 1.0e-1,
            "maximum_interpatch_macro_rate_relative_defect_per_field": 1.0e-1,
            "patch_2_fixed_macrostep_seconds": 1.0e-3,
            "patch_2_macrosteps": 4,
            "patch_2_horizon_seconds": 4.0e-3,
            "absolute_elapsed_endpoint_seconds": 8.0e-3,
            "reserved_trust_coordinate_infinity": 1.2e-1,
            "one_new_dynamic_endpoint_truth_call": True,
            "maximum_endpoint_truth_output_relative_defect_per_block": 5.0e-2,
            "maximum_endpoint_truth_macro_rate_relative_defect_per_field": 5.0e-2,
            "maximum_endpoint_macro_roundtrip_relative_defect": 1.0e-10,
            "maximum_local_spectral_abscissa_per_second": 0.0,
            "exact_integrated_ledger_relative_defect_max": 1.0e-12,
        },
        "budgets": {
            "maximum_new_truth_operator_calls": 39,
            "new_global_roots": 0,
            "accepted_new_macrosteps_max": 4,
            "complete_cycle_execution_authorized": False,
        },
        "decision": {
            "pass": "authorize_definitions_only_bounded_multi_patch_growth_and_fast_slaving_manifest",
            "failure": "stop_and_reassess_patch_coordinates_or_nonlinear_curvature",
            "no_retrospective_gate_change": True,
        },
        "claim_boundary": {
            "second_patch_execution_authorized": True,
            "unbounded_pathwise_continuation_authorized": False,
            "fixed_Q_invariant_object_certified": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("pathwise patch manifest already exists")
    validated = _validate_parent(require_clean=True); utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "pathwise_patch_contract.json", _contract())
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "bounded_macro_integrator_preserved": True, "second_patch_execution_authorized": True, "unbounded_pathwise_continuation_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"]})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete pathwise macro-atlas expansion manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The certified 4 ms endpoint becomes a second exact thermodynamic anchor. The same 38-call colored/JVP construction is repeated prospectively, followed by one full-truth 8 ms endpoint and an overlap witness.", "", "A pass establishes only two-patch local continuation and its growth rate. It does not authorize extrapolation, an invariant object, or a complete cycle.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
