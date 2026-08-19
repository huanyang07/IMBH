#!/usr/bin/env python3
"""Freeze the first conditional-branch coordinate and predictor preflight."""

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

import run_causal_inner_equilibrium_centered_hybrid_architecture_audit_wp10c9d6c7c3b5c4f25ao as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ap"
CLASSIFICATION = (
    "first_conditional_branch_seed_preflight_manifest_frozen_"
    "exact_integrable_coordinates_only"
)
PARENT_COMMIT = "3f62e1ac2f9c8f5683ad6fc5e9a1409215516d8f"
PARENT_PARENT = "d5e7b228e1d50bd494a08ef2353271f10996ccf7"
PARENT_TREE = "7bd6dbe11eeb2a3e08bde6b404ab1f2711b86305"

ARTIFACT = (
    "causal_inner_first_conditional_branch_seed_manifest_"
    "wp10c9d6c7c3b5c4f25ap"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_first_conditional_branch_seed_manifest_"
    "wp10c9d6c7c3b5c4f25ap.py"
)
THIS_TEST = (
    "tests/test_causal_inner_first_conditional_branch_seed_manifest_"
    "wp10c9d6c7c3b5c4f25ap.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_first_conditional_branch_seed_preflight_"
    "wp10c9d6c7c3b5c4f25aq.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_first_conditional_branch_seed_preflight_"
    "wp10c9d6c7c3b5c4f25aq.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FIRST_CONDITIONAL_BRANCH_SEED_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AP_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

GENERATOR_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c"
)
R32_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_larger_coarse_pde_audit_"
    "wp10c9d6c7c3b5c4f25k"
)
GENERATOR_PATH = GENERATOR_DIRECTORY / "descriptor_A.npz"
R32_PATH = R32_DIRECTORY / "R32_projection_promotion.npz"

FULL_DIMENSION = 560
MAPPED_COORDINATES = 160
STABLE_COORDINATES = 2
RESOLVED_DIMENSION = 162
HIDDEN_DIMENSION = 398
COARSE_CELLS = 32
FIELDS_PER_CELL = 5


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
        raise RuntimeError("branch-seed parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("branch-seed parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("branch-seed parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["authorized_next"]
        != "definitions_only_first_conditional_fast_branch_seed_manifest"
        or summary["physical_conditional_branch_found"]
        or summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("hybrid architecture authorization changed")
    return {"summary": summary, "hashes": hashes}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "anchor": {
            "label": "primary_20ms",
            "spatial_layout": "middle_112_cells",
            "fixed_Q3": True,
            "moving_checkpoint_is_assumed_to_be_a_branch_root": False,
        },
        "finite_amplitude_coordinate_map": {
            "symbol": "C_phys",
            "dimension": RESOLVED_DIMENSION,
            "mapped_storage_coordinates": MAPPED_COORDINATES,
            "explicit_stable_coordinates": STABLE_COORDINATES,
            "mapped_storage_definition": (
                "five_exact_cell_integrated_mapped_conserved_states_"
                "summed_over_each_of_32_nested_R32_cells"
            ),
            "mapped_row_normalization": (
                "frozen_anchor_Euclidean_norm_of_each_scaled_derivative_row"
            ),
            "stable_coordinate_definition": (
                "the_two_frozen_linear_stable_duals_from_the_R32_projection"
            ),
            "responsive_height_one_form_is_a_finite_amplitude_coordinate": False,
            "reason_height_is_excluded": (
                "responsive_height_storage_is_a_non_exact_temporal_one_form"
            ),
            "required_rank": RESOLVED_DIMENSION,
        },
        "direct_predictor": {
            "fixed_Q_rate": "saved_exact_primary_continuous_fixed_Q_rate_F0",
            "fixed_Q_generator": "saved_exact_complete_primary_generator_A0",
            "hidden_basis": "orthonormal_null_space_H0_of_D_C_phys_x0",
            "linear_system": "stack_D_C_phys_x0_and_H0_transpose_A0",
            "right_hand_side": "zero_then_minus_H0_transpose_F0",
            "direct_root_may_run_in_this_work_package": False,
            "nonbase_physical_truth_calls": 0,
        },
        "binding_gates": {
            "mapped_reconstruction_relative_defect_max": 1.0e-12,
            "reconstruction_partition_defect_max": 1.0e-12,
            "coordinate_rank_equal": RESOLVED_DIMENSION,
            "coordinate_condition_number_max": 1.0e4,
            "coordinate_directional_derivative_relative_defect_max": 1.0e-6,
            "Q3_rowspace_relative_defect_max": 1.0e-10,
            "direct_branch_linear_condition_number_max": 1.0e8,
            "direct_predictor_maximum_scaled_component_max": 5.0e-3,
            "direct_predictor_relative_linear_residual_max": 1.0e-10,
        },
        "decision": {
            "coordinate_failure": (
                "exact_integrable_coordinate_preflight_failed_"
                "conditional_branch_architecture_blocked"
            ),
            "direct_safe": (
                "direct_branch_predictor_safe_"
                "definitions_only_single_direct_root_manifest_authorized"
            ),
            "direct_unsafe": (
                "direct_branch_predictor_rejected_"
                "definitions_only_bordered_hidden_residual_homotopy_manifest_authorized"
            ),
            "direct_unsafe_is_not": [
                "a_physical_branch_nonexistence_result",
                "a_failure_of_the_fixed_Q_equations",
                "authorization_to_relax_any_physical_gate",
            ],
        },
        "homotopy_if_needed": {
            "unknowns": "x_in_R560_and_lambda_in_R162",
            "coordinate_equations": "C_phys_x_minus_C_phys_x0_equals_zero",
            "stationarity_equations": (
                "F_Q_x_minus_D_C_phys_x_transpose_lambda_"
                "minus_one_minus_tau_times_r0_equals_zero"
            ),
            "anchor_multiplier": (
                "lambda0_solves_D_C_D_C_transpose_lambda0_equals_D_C_F0"
            ),
            "anchor_hidden_residual": "r0_equals_F0_minus_D_C_transpose_lambda0",
            "tau_zero_anchor_is_exact": True,
            "tau_one_is_conditional_branch_stationarity": True,
            "continuation_parameter_is_physical_time": False,
            "tiny_forward_BDF_steps_are_used": False,
        },
        "claim_boundary": {
            "physical_branch_root_attempted": False,
            "physical_branch_found": False,
            "normal_hyperbolicity_certified": False,
            "transition_orbit_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "primary_generator": _sha(GENERATOR_PATH),
            "R32_projection": _sha(R32_PATH),
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    validated = _validate_parent()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("branch-seed manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("branch-seed manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "coordinate_dimension": RESOLVED_DIMENSION,
        "hidden_dimension": HIDDEN_DIMENSION,
        "nonbase_physical_truth_calls": 0,
        "physical_branch_root_attempted": False,
        "physical_branch_found": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25aq",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
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
                "# First conditional branch seed manifest WP10c9d6c7c3b5c4f25ap",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The first branch gate uses 160 exact finite-amplitude mapped-storage observables on the nested R32 grid plus two frozen linear stable coordinates. The responsive-height temporal one-form is explicitly excluded because it is not an integrable state coordinate.",
                "",
                "The authorized execution is a zero-new-truth-call rank, conditioning, Q3-rowspace, derivative-parity, and direct-predictor preflight at the primary 20 ms state. A direct root is not authorized inside this work package.",
                "",
                "If the coordinate structure passes but the direct predictor is unsafe, the only authorized successor is a definitions-only bordered hidden-residual homotopy. This is a solver-path decision, not evidence that a physical branch is absent.",
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
