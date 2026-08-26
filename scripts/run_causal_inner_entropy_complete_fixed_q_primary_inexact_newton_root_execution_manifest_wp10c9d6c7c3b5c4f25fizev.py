#!/usr/bin/env python3
"""Freeze the primary fixed-Q inexact-Newton root execution."""

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

import run_causal_inner_entropy_complete_fixed_q_inexact_trust_trial_execution_wp10c9d6c7c3b5c4f25fizeu as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizev_"
    "entropy_complete_fixed_Q_primary_inexact_Newton_root_execution_manifest"
)
CLASSIFICATION = "entropy_complete_fixed_Q_primary_inexact_Newton_root_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizew_"
    "entropy_complete_fixed_Q_primary_inexact_Newton_root_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_primary_inexact_newton_root_"
    "execution_manifest_wp10c9d6c7c3b5c4f25fizev"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "PRIMARY_INEXACT_NEWTON_ROOT_MANIFEST_WP10C9D6C7C3B5C4F25FIZEV_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_fixed_q_primary_inexact_"
    "newton_root_execution_manifest_wp10c9d6c7c3b5c4f25fizev.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_fixed_q_primary_inexact_"
    "newton_root_execution_manifest_wp10c9d6c7c3b5c4f25fizev.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "023951f1be0b91ff3905b0a0483a6aa5fc1abddd56f6333e63ca8164ade9c806"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("inexact-trust physical trial checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "trial_metrics.json")
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["nonpropagating_inexact_trial_passed"]
        or not summary["primary_inexact_Newton_root_execution_manifest_authorized"]
        or summary["primary_root_execution_authorized"]
        or summary["new_nonlinear_roots"] != 0
        or summary["propagated_states"] != 0
        or summary["authorized_next"] != f"definitions_only_{WORK_PACKAGE}"
        or metrics["selected_step_factor"] != 0.25
        or metrics["physical_field_calls"] != 3
    ):
        raise RuntimeError("primary inexact-Newton root manifest authorization changed")
    for relative, expected in utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"inexact-trust trial source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("primary root manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "objective": "find_and_classify_the_primary_cellwise_fixed_Q_fast_equilibrium",
        "initial_state": {
            "base_equation_Jacobian": "hash_locked_fizes_colored_Jacobian",
            "first_accepted_trial": "hash_locked_fizeu_quarter_step",
            "initial_Broyden_matrix": "good_Broyden_update_from_base_to_quarter_step",
            "fixed_slow_targets_and_equation_scales_unchanged": True,
            "first_trial_counts_as_nonlinear_correction_one": True,
        },
        "inexact_Newton_solver": {
            "maximum_total_nonlinear_corrections": 12,
            "linear_subproblem": "500_iteration_bounded_TRF_descent_direction",
            "maximum_scaled_step": 0.25,
            "maximum_inexact_forcing_two_norm": 0.95,
            "negative_directional_derivative_required": True,
            "ordered_line_search_factors": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
            "two_norm_Armijo_coefficient": 1.0e-4,
            "strict_infinity_norm_decrease_required": True,
            "good_Broyden_secant_update_after_each_accepted_step": True,
            "maximum_solver_colored_Jacobian_assemblies": 2,
            "canonical_base_assembly_counts_as_one": True,
            "one_refresh_trigger": [
                "complete_physical_line_search_failure",
                "beginning_of_correction_10_if_no_refresh_has_occurred",
                "inexact_direction_forcing_or_descent_failure",
            ],
            "refreshed_correction_retries_same_iteration": True,
            "failed_or_nonphysical_candidate_never_updates_Broyden_or_state": True,
        },
        "root_gates": {
            "maximum_normalized_equation_residual_infinity": 1.0e-8,
            "maximum_physical_fast_coordinate_rate_infinity_per_second": 1.0e-8,
            "maximum_equation_rate_parity_relative_defect": 1.0e-10,
            "all_parent_physical_and_fixed_slow_gates_required": True,
            "root_is_saved_but_not_propagated": True,
        },
        "post_root_certification": {
            "one_fresh_12_color_equation_Jacobian": True,
            "certification_assembly_is_not_a_solver_refresh": True,
            "certification_assembly_cannot_change_the_root": True,
            "independent_direct_physical_rate_central_JVP_directions": 4,
            "central_relative_step": 2.0e-6,
            "maximum_physical_tangent_JVP_relative_defect": 2.0e-5,
            "physical_tangent_transform": "C_times_A_inverse_times_equation_Jacobian_with_similarity_chart_scaling",
            "all_eigenvalues_finite": True,
        },
        "normal_attraction": {
            "maximum_spectral_abscissa_per_second": -1.0,
            "slow_relative_rate": "max_abs_slow_drift_per_second_divided_by_abs_slow_target",
            "minimum_attraction_to_slow_relative_rate_ratio": 10.0,
            "solver_normalized_equation_eigenvalues_are_nonphysical": True,
        },
        "decision": {
            "root_and_normal_attraction_pass": "authorize_definitions_only_heldout_root_replication_manifest",
            "root_passes_but_normal_attraction_fails": "authorize_only_fixed_Q_invariant_measure_diagnosis_manifest",
            "root_or_physical_gate_fails": "stop_without_heldout_or_slow_atlas",
        },
        "claim_boundary": {
            "primary_root_execution_authorized": True,
            "heldout_root_execution_authorized": False,
            "slow_flux_atlas_authorized": False,
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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("primary root manifest already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "primary_root_execution_contract.json", _contract()); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "selected_step_factor": validated["metrics"]["selected_step_factor"]})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "nonpropagating_inexact_trial_preserved": True, "primary_root_execution_authorized": True, "heldout_root_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete fixed-Q primary inexact-Newton root manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The root solve reuses the hash-locked base colored Jacobian and accepted quarter-step trial, permits one solver refresh, and reserves a separate post-root derivative/spectral audit that cannot alter the root.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
