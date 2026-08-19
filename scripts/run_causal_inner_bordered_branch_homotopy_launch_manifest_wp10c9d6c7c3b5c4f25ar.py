#!/usr/bin/env python3
"""Freeze the first bordered hidden-residual homotopy launch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_first_conditional_branch_seed_preflight_wp10c9d6c7c3b5c4f25aq as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ar"
CLASSIFICATION = (
    "bordered_hidden_residual_homotopy_launch_manifest_frozen_"
    "single_tau_1_over_64_rung_only"
)
PARENT_COMMIT = "e231232bedfa80744531ba1b63fa97bc33d5cd1c"
PARENT_PARENT = "988cff4c345ffd5d824d6c9f2ca303b677456740"
PARENT_TREE = "33ed9e8cd1b1d4214d7760d6a7cf63e46a5f7c30"

ARTIFACT = (
    "causal_inner_bordered_branch_homotopy_launch_manifest_"
    "wp10c9d6c7c3b5c4f25ar"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_bordered_branch_homotopy_launch_manifest_"
    "wp10c9d6c7c3b5c4f25ar.py"
)
THIS_TEST = (
    "tests/test_causal_inner_bordered_branch_homotopy_launch_manifest_"
    "wp10c9d6c7c3b5c4f25ar.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_bordered_branch_homotopy_launch_"
    "wp10c9d6c7c3b5c4f25as.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_bordered_branch_homotopy_launch_"
    "wp10c9d6c7c3b5c4f25as.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_BORDERED_BRANCH_HOMOTOPY_"
    "LAUNCH_MANIFEST_WP10C9D6C7C3B5C4F25AR_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TAU_START = 0.0
TAU_TARGET = 1.0 / 64.0
RATE_SCALE_DEFINITION = "RMS_norm_of_saved_anchor_fixed_Q_rate"
MAXIMUM_NEW_RATE_EVALUATIONS = 17


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("homotopy-launch parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("homotopy-launch parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("homotopy-launch parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["authorized_next"]
        != "definitions_only_bordered_hidden_residual_homotopy_manifest"
        or not summary["homotopy_required"]
        or summary["direct_root_attempted"]
        or summary["physical_branch_found"]
    ):
        raise RuntimeError("homotopy authorization changed")
    return {"summary": summary, "hashes": hashes}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "anchor": {
            "label": "primary_20ms",
            "state": "hash_locked_preflight_primitive_anchor_x0",
            "resolved_coordinates": "C0_equals_C_phys_x0",
            "fixed_Q3": True,
        },
        "bordered_homotopy": {
            "unknowns": {
                "scaled_primitive_state_x": 560,
                "dimensionless_coordinate_multiplier_mu": 162,
                "total": 722,
            },
            "coordinate_equations": "C_phys_x_minus_C0_equals_zero_in_R162",
            "stationarity_equations": (
                "F_Q_x_divided_by_omega_minus_D_C_phys_x_transpose_mu_"
                "minus_one_minus_tau_times_r0_equals_zero_in_R560"
            ),
            "rate_scale_omega": RATE_SCALE_DEFINITION,
            "mu0": (
                "solve_D_C0_D_C0_transpose_mu0_equals_"
                "D_C0_F_Q_x0_divided_by_omega"
            ),
            "r0": (
                "F_Q_x0_divided_by_omega_minus_D_C0_transpose_mu0"
            ),
            "tau_start": TAU_START,
            "tau_target": TAU_TARGET,
            "tau_zero_anchor_residual_required_exact": True,
            "tau_one_is_conditional_branch_but_is_not_attempted_here": True,
            "continuation_parameter_is_physical_time": False,
            "forward_BDF_history_is_used": False,
        },
        "nonlinear_policy": {
            "initial_matrix": (
                "frozen_anchor_Gauss_Newton_KKT_seed_using_saved_exact_A0_"
                "with_no_coordinate_Hessian_term"
            ),
            "complete_residual_is_evaluated_at_every_candidate": True,
            "matrix_updates": "dense_good_Broyden_after_each_accepted_trial",
            "linear_equilibration": "eight_deterministic_Ruiz_row_column_passes",
            "maximum_iterations": 4,
            "line_search_relative_factors": [1.0, 0.5, 0.25, 0.125],
            "line_search_merit": "Euclidean_norm_of_complete_722_residual",
            "maximum_new_fixed_Q_rate_evaluations": MAXIMUM_NEW_RATE_EVALUATIONS,
            "maximum_scaled_anchor_departure": 5.0e-3,
            "rejected_candidate_may_define_future_history": False,
        },
        "binding_gates": {
            "tau_zero_complete_residual_infinity_max": 1.0e-12,
            "equilibrated_initial_matrix_condition_number_max": 1.0e6,
            "linear_predictor_maximum_scaled_component_max": 5.0e-3,
            "complete_target_residual_infinity_max": 1.0e-8,
            "coordinate_residual_infinity_max": 1.0e-8,
            "stationarity_residual_infinity_max": 1.0e-8,
            "normalized_Q3_defect_max": 1.0e-10,
            "minimum_reconstruction_factor_min": 1.0 - 1.0e-12,
            "raw_schur_rank_equal": 3,
            "raw_schur_condition_number_max": 1.0e8,
            "raw_schur_solve_relative_defect_max": 1.0e-10,
            "maximum_h_over_r_max": 0.3,
            "minimum_scattering_optical_depth_min": 1.0,
            "maximum_new_rate_evaluations": MAXIMUM_NEW_RATE_EVALUATIONS,
        },
        "decision": {
            "pass": (
                "bordered_homotopy_launch_tau_1_over_64_passed_"
                "definitions_only_adaptive_homotopy_continuation_manifest_authorized"
            ),
            "fail": (
                "bordered_homotopy_launch_failed_"
                "conditional_branch_path_requires_diagnosis"
            ),
            "pass_authorizes_only": (
                "definitions_only_adaptive_bordered_homotopy_continuation_manifest"
            ),
        },
        "claim_boundary": {
            "tau_one_reached": False,
            "physical_conditional_branch_found": False,
            "normal_hyperbolicity_certified": False,
            "transition_orbit_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "preflight_diagnostics": _sha(
                parent.CANONICAL_DIRECTORY / "preflight_diagnostics.npz"
            ),
            "preflight_metrics": _sha(parent.CANONICAL_DIRECTORY / "metrics.json"),
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
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
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": _git("rev-parse", PARENT_COMMIT),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    validated = _validate_parent()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("homotopy-launch manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("homotopy-launch manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "tau_start": TAU_START,
        "tau_target": TAU_TARGET,
        "maximum_new_rate_evaluations": MAXIMUM_NEW_RATE_EVALUATIONS,
        "tau_one_reached": False,
        "physical_conditional_branch_found": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25as",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": _git("rev-parse", PARENT_COMMIT),
            "parent_parent": _git("rev-parse", PARENT_PARENT),
            "parent_tree": _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}"),
            "parent_package_hashes": validated["hashes"],
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_next_runner": NEXT_RUNNER,
            "authorized_next_test": NEXT_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": parent.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Bordered branch homotopy launch manifest WP10c9d6c7c3b5c4f25ar",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The direct predictor is replaced by a square 722-variable KKT homotopy. The anchor is an exact tau=0 solution; the sole authorized physical execution advances to tau=1/64 at fixed exact mapped-storage/stable coordinates.",
                "",
                "The fixed-Q rate is divided by its anchor RMS norm and the bordered matrix is deterministically equilibrated. The initial matrix is explicitly a Gauss-Newton/Broyden seed, not a claimed complete branch Jacobian.",
                "",
                "A pass certifies only launch of the continuation path and authorizes a new definitions-only adaptive continuation manifest. Tau=1, a physical conditional branch, normal hyperbolicity, transitions, and reduced slow evolution remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
