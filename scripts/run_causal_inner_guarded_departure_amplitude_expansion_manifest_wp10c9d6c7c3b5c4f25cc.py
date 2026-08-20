#!/usr/bin/env python3
"""Freeze a guarded exact-geometry departure-amplitude expansion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_fixed_q_fast_attractor_screen_wp10c9d6c7c3b5c4f25cb as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cc"
PARENT_COMMIT = "476ff40b1e84d209072ebd3c94b7c3c466a04cfa"
PARENT_PARENT = "fc084916a418d71d85d3b64a3950140461a13c67"
PARENT_TREE = "3696fdec83b63e04d8b83d2e6d0d129a19b7fa5d"
CLASSIFICATION = "guarded_departure_amplitude_expansion_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cd"

AMPLITUDE_RUNGS = (1.5e-2, 2.0e-2, 3.0e-2)
SIGNS = (-1, 1)
ENERGY_DIRECTION_COUNT = 8
TARGETED_DIRECTION_COUNT = 2
DIRECTION_COUNT = ENERGY_DIRECTION_COUNT + TARGETED_DIRECTION_COUNT
CANDIDATES_PER_RUNG = DIRECTION_COUNT * len(SIGNS)
PLANNED_CANDIDATES = len(AMPLITUDE_RUNGS) * CANDIDATES_PER_RUNG

ARTIFACT = (
    "causal_inner_guarded_departure_amplitude_expansion_manifest_"
    "wp10c9d6c7c3b5c4f25cc"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_guarded_departure_amplitude_expansion_manifest_"
    "wp10c9d6c7c3b5c4f25cc.py"
)
THIS_TEST = (
    "tests/test_causal_inner_guarded_departure_amplitude_expansion_manifest_"
    "wp10c9d6c7c3b5c4f25cc.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_guarded_departure_amplitude_expansion_preflight_"
    "wp10c9d6c7c3b5c4f25cd.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_guarded_departure_amplitude_expansion_preflight_"
    "wp10c9d6c7c3b5c4f25cd.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_GUARDED_DEPARTURE_AMPLITUDE_"
    "EXPANSION_MANIFEST_WP10C9D6C7C3B5C4F25CC_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

SHORT_VECTOR_DIRECTORY = parent.vector_field.CANONICAL_DIRECTORY
RATE_DATABASE = parent.vector_field.manifest.parent.DATABASE_PATH

_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_sha = parent._sha
_checksums = parent._checksums


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("fast-attractor screen commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("fast-attractor screen lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("fast-attractor screen tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "screen_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    decision = metrics["decision"]
    if (
        not summary["passed"]
        or summary["classification"] != parent.NONCLOSURE_CLASSIFICATION
        or summary["authorized_next"]
        != "definitions_only_guarded_departure_amplitude_expansion_manifest"
        or summary["accepted_root_count"] != 0
        or summary["new_truth_calls"] != 0
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or not all(decision["structure_checks"].values())
        or not all(decision["clear_nonclosure_checks"].values())
        or decision["decoded_state_audit_pass_fraction"] != 0.0
    ):
        raise RuntimeError("fast-attractor nonclosure authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"fast-attractor screen source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("amplitude-expansion manifest requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("departure expansion direction vanished")
    return array / norms[:, None]


def _direction_design() -> tuple[np.ndarray, tuple[str, ...], dict]:
    with np.load(RATE_DATABASE, allow_pickle=False) as source:
        energy = np.asarray(source["energy_directions"], dtype=float).T
    with np.load(
        parent.CANONICAL_DIRECTORY / "screen_arrays.npz", allow_pickle=False
    ) as source:
        solutions = np.asarray(source["departure_solutions"], dtype=float)
        rates = np.asarray(source["reduced_departure_rates_per_second"], dtype=float)
    best_index = int(np.argmin(np.linalg.norm(rates, axis=1)))
    escape = solutions[best_index]
    with np.load(
        SHORT_VECTOR_DIRECTORY / "validation_arrays.npz", allow_pickle=False
    ) as source:
        forward = np.asarray(source["truth_BDF_coordinate_rate"], dtype=float)[-28:]
    directions = _normalize_rows(np.vstack((energy, escape, forward)))
    labels = tuple(
        [f"energy_{index}" for index in range(ENERGY_DIRECTION_COUNT)]
        + ["screen_escape", "accepted_forward_rate"]
    )
    if directions.shape != (DIRECTION_COUNT, 28) or len(labels) != DIRECTION_COUNT:
        raise RuntimeError("departure expansion direction design changed")
    gram = np.abs(directions @ directions.T - np.eye(DIRECTION_COUNT))
    off_diagonal = gram[~np.eye(DIRECTION_COUNT, dtype=bool)]
    diagnostics = {
        "screen_best_start_index": best_index,
        "maximum_pairwise_absolute_cosine": float(np.max(off_diagonal)),
        "screen_escape_to_energy_maximum_absolute_cosine": float(
            np.max(np.abs(energy @ directions[-2]))
        ),
        "forward_to_energy_maximum_absolute_cosine": float(
            np.max(np.abs(energy @ directions[-1]))
        ),
        "escape_forward_absolute_cosine": float(
            abs(directions[-2] @ directions[-1])
        ),
    }
    return directions, labels, diagnostics


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "determine_the_exact_C_phys_geometric_reach_of_the_current_"
            "departure_chart_before_any_new_rate_or_trajectory_call"
        ),
        "scientific_interpretation": {
            "current_local_field": "validated_short_transient_not_stationary_fast_graph",
            "stationary_eliminated_memory_states_are_physical": False,
            "old_polynomial_extrapolation_is_binding": False,
            "departure_amplitude_expansion_alone_proves_attractor": False,
            "stable_memory_must_remain_dynamic": True,
        },
        "candidate_family": {
            "anchor": "accepted_primary_warm_3",
            "component_bound_rungs": list(AMPLITUDE_RUNGS),
            "rung_order": "strictly_increasing_fail_fast",
            "directions": DIRECTION_COUNT,
            "direction_labels": [
                *[f"energy_{index}" for index in range(ENERGY_DIRECTION_COUNT)],
                "screen_escape",
                "accepted_forward_rate",
            ],
            "signs": list(SIGNS),
            "candidates_per_rung": CANDIDATES_PER_RUNG,
            "maximum_planned_candidates": PLANNED_CANDIDATES,
        },
        "exact_geometric_retraction": {
            "equations": "C_phys_x_equals_C_phys_x_anchor",
            "state_normal": "exact_state_local_derivative_of_C_phys",
            "departure_seed": "signed_frozen_departure_basis_direction",
            "stable_memory_seed": "zero",
            "rate_reaction_lift_used": False,
            "maximum_Newton_iterations": 8,
            "maximum_line_search_iterations": 12,
            "maximum_radius_rescalings": 6,
            "coordinate_residual_tolerance": 1.0e-10,
        },
        "binding_per_rung_gates": {
            "completed_candidate_count_equal": CANDIDATES_PER_RUNG,
            "failed_candidate_count_equal": 0,
            "maximum_coordinate_residual_infinity": 1.0e-10,
            "maximum_normalized_Q3_defect": 1.0e-10,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_coordinate_Jacobian_condition_number": 1.0e4,
            "minimum_departure_direction_alignment_cosine": 0.99,
            "maximum_departure_transverse_fraction": 0.05,
            "maximum_coordinate_odd_symmetry_defect": 0.05,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
        },
        "cost_and_truth_budget": {
            "new_nonbase_continuous_rate_evaluations_equal": 0,
            "new_full_generator_assemblies_equal": 0,
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "propagated_physical_states_equal": 0,
        },
        "fail_fast_decision": {
            "all_rungs_pass": {
                "classification": "exact_departure_chart_geometry_reaches_component_bound_0p03",
                "authorizes_only": "definitions_only_minimal_expanded_departure_rate_design_manifest",
            },
            "partial_rungs_pass": {
                "classification": "exact_departure_chart_geometry_has_finite_guarded_amplitude_limit",
                "authorizes_only": "definitions_only_rate_design_within_largest_passing_departure_bound",
            },
            "first_rung_fails": {
                "classification": "radial_departure_chart_expansion_failed_at_0p015",
                "authorizes_only": "definitions_only_authentic_trajectory_recentered_chart_manifest",
            },
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "required_mathematical_architecture": {
            "active_state": "q162_physical_coordinates",
            "stable_memory": "z280_dynamic_exponential_or_L_stable_update",
            "departure": "a28_nonlinear_multichart_transient_or_invariant_measure",
            "eventual_cycle_closure": (
                "multi_anchor_conservative_q_flux_plus_dynamic_stable_memory_"
                "plus_eliminated_or_averaged_departure"
            ),
            "maximum_online_truth_calls_per_macrostep": 0,
            "target_maximum_cycle_macrosteps": 100_000,
        },
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
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
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
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
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
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("amplitude-expansion manifest already canonicalized")
    directions, labels, diagnostics = _direction_design()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "direction_design.npz",
        directions=directions,
        amplitude_rungs=np.asarray(AMPLITUDE_RUNGS),
        signs=np.asarray(SIGNS, dtype=int),
    )
    _write_json(
        CANONICAL_DIRECTORY / "direction_design.json",
        {"labels": labels, "diagnostics": diagnostics},
    )
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "short_vector_arrays_sha256": _sha(
                SHORT_VECTOR_DIRECTORY / "validation_arrays.npz"
            ),
            "rate_database_sha256": _sha(RATE_DATABASE),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "maximum_amplitude_bound": max(AMPLITUDE_RUNGS),
        "planned_candidate_count": PLANNED_CANDIDATES,
        "new_truth_calls": 0,
        "stable_memory_remains_dynamic": True,
        "old_polynomial_extrapolation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                parent.THIS_RUNNER: _sha(ROOT / parent.THIS_RUNNER),
                parent.THIS_TEST: _sha(ROOT / parent.THIS_TEST),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Guarded departure-amplitude expansion manifest WP10c9d6c7c3b5c4f25cc",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The prior screen found no stationary fast graph inside the 0.01 departure chart. "
                "This package freezes a geometry-only, fail-fast chart reachability ladder at "
                "0.015, 0.02, and 0.03 before any new rate evaluation.",
                "",
                "The ten directions comprise eight hash-locked energy directions, the best boundary "
                "escape direction from the nonclosure screen, and the accepted warm_4 forward-rate direction.",
                "",
                "The 280D stable memory remains dynamic. Its far, unphysical stationary elimination "
                "is not used to generate any candidate. The old polynomial is not extrapolated as evidence.",
                "",
                "The target cycle architecture is q162 physical evolution plus an exponential/L-stable "
                "z280 update and a multichart nonlinear/averaged a28 closure, with zero online truth calls "
                "and at most 100,000 macrosteps per cycle.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
