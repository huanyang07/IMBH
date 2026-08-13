#!/usr/bin/env python3
"""Freeze the optimized middle-layout continuation from 6 to 20 ms."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_wp10c9d6c7c3b5c4e1 as c4e1  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e2"
ANALYZED_BASE_COMMIT = "132857f4acb0c71185bd2096fc6a4bbd87ac0675"
ANALYZED_BASE_PARENT = "e0154f7c4f97be51f6b8efce507bdb8b1a14f9d0"
ANALYZED_BASE_TREE = "91f3cc17443e4cf25e652a5159b251420b764ba8"

OUTPUT_TARGET_MICROSECONDS = (
    6000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    18800,
    19600,
    19800,
    20000,
)
AUDIT_TARGET_MICROSECONDS = (6400, 7200, 9200, 13200, 17200, 20000)
TIMESTEP_CAP_CANDIDATES_SECONDS = (4.0e-4, 8.0e-4, 1.2e-3)
EXTRACTION_JVP_RELATIVE_STEP = 0.1
EXTRACTION_JVP_STEP_SWEEP = (0.05, 0.1, 0.2)

ARTIFACT = (
    "causal_inner_nonlinear_optimized_middle_20ms_completion_manifest_"
    "wp10c9d6c7c3b5c4e2"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_optimized_middle_20ms_completion_"
    "manifest_wp10c9d6c7c3b5c4e2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_optimized_middle_20ms_completion_"
    "manifest_wp10c9d6c7c3b5c4e2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_OPTIMIZED_MIDDLE_20MS_"
    "COMPLETION_MANIFEST_WP10C9D6C7C3B5C4E2_2026-08-10.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "completion_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> dict:
    summary = _read_json(c4e1.SUMMARY_PATH)
    if (
        not summary["passed"]
        or not summary["scientific_gates_passed"]
        or not summary["middle_twenty_ms_completion_manifest_authorized"]
        or summary["middle_twenty_ms_propagation_authorized"]
        or summary["fine_twenty_ms_propagation_authorized"]
        or summary["fixed_q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c4e2 parent authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e2 analyzed identity changed")
    return summary


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "optimized_middle_6_to_20ms_completion_manifest_frozen_"
            "propagation_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "scientific_scope": {
            "layout": c4e1.h2b1.MIDDLE_LAYOUT,
            "start_seconds": 6.0e-3,
            "stop_seconds": 20.0e-3,
            "profiles": c4e1.h2b1.PROFILES,
            "generic_nonlinear_anchor": c4e1.h2b1.GENERIC_PROFILE,
            "coupling_face": c4e1.COUPLING_FACE,
            "extraction_face": c4e1.EXTRACTION_FACE,
            "extraction_radius_rg": c4e1.c4e.EXTRACTION_RADIUS_RG,
            "output_target_microseconds": OUTPUT_TARGET_MICROSECONDS,
            "full_step_doubling_audit_target_microseconds": (
                AUDIT_TARGET_MICROSECONDS
            ),
            "raw_inner_face_is_not_a_binding_slow_export": True,
        },
        "optimization_contract": {
            "timestep_cap_candidates_seconds": TIMESTEP_CAP_CANDIDATES_SECONDS,
            "largest_passing_cap_is_selected_prospectively": True,
            "maximum_BDF2_step_ratio": 2.0,
            "routine_base_step": "one_full_nonlinear_BDF2_solve",
            "routine_error_bound": (
                "four_times_last_audited_full_vs_two_half_error_scaled_by_dt_cubed"
            ),
            "routine_error_bound_safety_factor": 4.0,
            "audit_step": "one_full_step_plus_two_half_steps",
            "failed_cap_falls_back_to_next_smaller_passing_cap": True,
            "generic_anchor_is_full_nonlinear_on_every_accepted_base_step": True,
            "generic_anchor_uses_tangent_Newton_predictor": True,
            "generic_anchor_step_doubling_only_at_declared_audits": True,
            "all_five_profiles_share_one_tangent_matrix_factorization": True,
            "complete_residual_tangent_JVP_only_at_declared_audits": True,
        },
        "extraction_tangent_contract": {
            "observable": "certified_conservative_exterior_partition",
            "central_relative_step": EXTRACTION_JVP_RELATIVE_STEP,
            "step_sweep": EXTRACTION_JVP_STEP_SWEEP,
            "all_five_profile_directions_required": True,
            "instantaneous_response_required": True,
            "cumulative_response_required": True,
            "window_mean_response_windows_seconds": ((0.010, 0.020), (0.016, 0.020)),
            "maximum_step_sensitivity_fraction_of_response": 1.0e-4,
            "maximum_generic_discrepancy_fraction_of_response": 0.01,
            "maximum_surrogate_fraction_of_spatial_difference_deferred": 0.10,
        },
        "method_gates": {
            "maximum_scaled_nonlinear_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure_defect": 1.0e-9,
            "maximum_extraction_identity_defect": 1.0e-12,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
            "maximum_local_error_estimate": 2.5e-4,
            "maximum_sum_of_local_error_bounds": 5.0e-3,
            "minimum_audit_error_margin_factor": 10.0,
            "maximum_tangent_linear_solve_relative_defect": 1.0e-10,
            "maximum_tangent_matrix_JVP_relative_defect": 1.0e-8,
            "same_target_final_restart_replay_bitwise": True,
        },
        "durability": {
            "checkpoint_after_every_accepted_base_step": True,
            "checkpoint_after_every_tangent_and_anchor_step": True,
            "checkpoint_payload_hashes_verified_on_resume": True,
            "source_dependency_hashes_binding": True,
            "final_base_and_anchor_replay_bitwise": True,
        },
        "decision_branches": {
            "all_gates_pass": "authorize_coarse_middle_20ms_checkpoint_analysis",
            "1p2ms_cap_fails": "continue_with_largest_smaller_passing_cap",
            "extraction_tangent_fails": "stop_before_spatial_analysis",
            "middle_completion_fails": "stop_before_fine",
        },
        "hard_stops": (
            "do_not_use_raw_inner_face_flux_as_slow_export",
            "do_not_run_fine_before_middle_checkpoint_analysis",
            "do_not_start_50ms_fixed_Q_or_reduced_evolution",
            "do_not_relax_scientific_gates_for_runtime",
            "do_not_change_the_operator_or_profiles",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c4e3_optimized_middle_6_to_20ms_completion"
        ),
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    classification = manifest["classification"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "middle_twenty_ms_optimized_propagation_authorized": True,
        "fine_twenty_ms_propagation_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "output_target_microseconds": OUTPUT_TARGET_MICROSECONDS,
            "audit_target_microseconds": AUDIT_TARGET_MICROSECONDS,
            "timestep_cap_candidates_seconds": TIMESTEP_CAP_CANDIDATES_SECONDS,
            "extraction_jvp_relative_step": EXTRACTION_JVP_RELATIVE_STEP,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "middle_6ms_summary": _sha256(c4e1.SUMMARY_PATH),
                "middle_6ms_arrays": _sha256(c4e1.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Optimized middle 20 ms completion manifest WP10c9d6c7c3b5c4e2",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This definitions-only package authorizes one middle nonlinear base, one full generic nonlinear anchor, and a five-profile block tangent from 6 to 20 ms.",
                "",
                "Routine base steps use one nonlinear BDF2 solve. Full-versus-two-half comparisons remain binding at the declared 0.4, 0.8, and 1.2 ms preflight steps and at later audit windows.",
                "",
                "The tangent of the certified extraction partition is binding in instantaneous, cumulative, and window-mean form. The rejected raw inner-face flux remains excluded.",
                "",
                "Fine propagation, 50 ms propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("completion_manifest.json", "config.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
