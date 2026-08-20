#!/usr/bin/env python3
"""Freeze the fixed-Q fast-attractor and normal-hyperbolicity screen."""

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

import run_causal_inner_departure28_short_vector_field_validation_wp10c9d6c7c3b5c4f25bz as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ca"
PARENT_COMMIT = "632d63fe665b3ad29e649366920c15bb9f9c1968"
PARENT_PARENT = "0d35744ccea8c1b5884407231bd2e9f173fe6220"
PARENT_TREE = "e6f5d7a0a1924e0aef839b315e528b811a5e3258"
CLASSIFICATION = "fixed_Q_fast_attractor_normal_hyperbolicity_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cb"

ARTIFACT = (
    "causal_inner_fixed_q_fast_attractor_manifest_"
    "wp10c9d6c7c3b5c4f25ca"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_fixed_q_fast_attractor_manifest_"
    "wp10c9d6c7c3b5c4f25ca.py"
)
THIS_TEST = (
    "tests/test_causal_inner_fixed_q_fast_attractor_manifest_"
    "wp10c9d6c7c3b5c4f25ca.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_fixed_q_fast_attractor_screen_"
    "wp10c9d6c7c3b5c4f25cb.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_fixed_q_fast_attractor_screen_"
    "wp10c9d6c7c3b5c4f25cb.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FIXED_Q_FAST_ATTRACTOR_"
    "MANIFEST_WP10C9D6C7C3B5C4F25CA_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PHYSICAL_DIMENSION = 162
STABLE_MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28
ONLINE_DIMENSION = 470
DEPARTURE_COMPONENT_BOUND = 1.0e-2
SEARCH_AMPLITUDES = (2.5e-3, 5.0e-3, 8.0e-3)
STARTS_PER_AMPLITUDE = 4
SEARCH_START_COUNT = 1 + len(SEARCH_AMPLITUDES) * STARTS_PER_AMPLITUDE
SEARCH_SEED = 20_260_819


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
        raise RuntimeError("short-vector-field certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("short-vector-field certificate lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("short-vector-field certificate tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "validation_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.PASS_CLASSIFICATION
        or summary["authorized_next"]
        != "definitions_only_fixed_Q_fast_attractor_and_normal_hyperbolicity_manifest"
        or not summary["retrospective_readiness_passed"]
        or not summary["prospective_forecast_passed"]
        or summary["new_truth_roots"] != 1
        or summary["model_470_role"]
        != "offline_fast_transient_and_closure_model"
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or not all(metrics["checks"].values())
    ):
        raise RuntimeError("short-vector-field authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"short-vector-field source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("fast-attractor manifest requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _search_design() -> np.ndarray:
    generator = np.random.default_rng(SEARCH_SEED)
    starts = [np.zeros(DEPARTURE_DIMENSION)]
    for amplitude in SEARCH_AMPLITUDES:
        for _ in range(STARTS_PER_AMPLITUDE):
            direction = generator.normal(size=DEPARTURE_DIMENSION)
            direction /= np.max(np.abs(direction))
            starts.append(amplitude * direction)
    result = np.asarray(starts, dtype=float)
    if result.shape != (SEARCH_START_COUNT, DEPARTURE_DIMENSION):
        raise RuntimeError("fast-attractor search design changed")
    return result


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "determine_whether_the_certified_local_470_field_contains_an_"
            "in_chart_normally_attracting_fixed_Q_fast_graph"
        ),
        "mathematical_split": {
            "active_physical_q_dimension": PHYSICAL_DIMENSION,
            "strictly_stable_memory_z_dimension": STABLE_MEMORY_DIMENSION,
            "nonlinear_departure_a_dimension": DEPARTURE_DIMENSION,
            "identity": "470_equals_162_plus_280_plus_28",
            "q_is_held_fixed_during_this_screen": True,
            "z_is_the_only_block_authorized_for_linear_elimination": True,
            "a_remains_explicitly_nonlinear": True,
            "naive_96_slow_plus_374_fast_split_authorized": False,
            "reason_96_plus_374_is_blocked": (
                "the_346_constitutive_memory_subblock_is_not_certified_Hurwitz"
            ),
        },
        "cheap_polynomial_field": {
            "formula": (
                "F_tilde_y_equals_R_r0_plus_R_G_L_y_plus_R_G_B4_kappa_a_"
                "plus_R_D28_N28_a"
            ),
            "uses_only_frozen_certified_coefficients": True,
            "new_truth_calls": 0,
            "new_full_generator_assemblies": 0,
            "new_nonlinear_fixed_Q_roots": 0,
            "propagated_physical_states": 0,
            "claim": "local_fast_attractor_screen_not_cycle_prediction",
        },
        "stable_memory_elimination": {
            "equation": (
                "z_star_a_equals_minus_Azz_inverse_times_"
                "b_z_plus_Aza_a_plus_Kz_kappa_a_plus_Pz_N28_a"
            ),
            "method": "one_frozen_pivoted_LU_factorization_and_reused_solves",
            "reduced_equation_dimension": DEPARTURE_DIMENSION,
            "reduced_equation": (
                "g_a_a_equals_b_a_plus_Aaz_z_star_a_plus_Aaa_a_"
                "plus_Ka_kappa_a_plus_Pa_N28_a_equals_zero"
            ),
        },
        "deterministic_search": {
            "departure_component_bound": DEPARTURE_COMPONENT_BOUND,
            "start_count": SEARCH_START_COUNT,
            "starts": "zero_plus_four_seeded_directions_at_each_of_three_amplitudes",
            "amplitudes": list(SEARCH_AMPLITUDES),
            "seed": SEARCH_SEED,
            "algorithm": "bounded_scipy_least_squares_trust_region_reflective",
            "maximum_function_evaluations_per_start": 1_000,
            "xtol": 1.0e-12,
            "ftol": 1.0e-12,
            "gtol": 1.0e-12,
            "duplicate_root_distance": 1.0e-6,
        },
        "binding_structure_gates": {
            "stable_memory_dimension_equal": STABLE_MEMORY_DIMENSION,
            "stable_memory_spectral_abscissa_max_per_second": -1.0,
            "stable_memory_condition_number_max": 1.0e6,
            "stable_memory_minimum_decay_rate_per_second": 1.0,
            "departure_projection_cross_block_norm_max": 1.0e-10,
            "search_start_count_equal": SEARCH_START_COUNT,
            "maximum_search_start_component": max(SEARCH_AMPLITUDES),
            "new_truth_calls_equal": 0,
            "new_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "binding_fixed_graph_gates": {
            "root_reduced_residual_relative_to_base_max": 1.0e-8,
            "root_maximum_absolute_departure_component": 9.9e-3,
            "root_full_fast_residual_relative_to_base_max": 1.0e-8,
            "full_fast_spectral_abscissa_max_per_second": -1.0,
            "full_fast_minimum_attraction_rate_per_second": 1.0,
            "minimum_distinct_root_separation": 1.0e-6,
        },
        "clear_nonclosure_gates": {
            "accepted_root_count_equal": 0,
            "minimum_screened_residual_relative_to_base_min": 5.0e-2,
            "minimum_boundary_limited_solution_fraction": 0.75,
            "boundary_fraction_threshold": 0.999,
        },
        "decision": {
            "stable_graph": {
                "classification": "in_chart_normally_attracting_fixed_Q_fast_graph_candidate_found",
                "authorizes_only": (
                    "definitions_only_local_fast_graph_continuation_and_"
                    "slow_flux_closure_manifest"
                ),
            },
            "clear_nonclosure": {
                "classification": (
                    "no_in_chart_stationary_fast_graph_found_departure_"
                    "amplitude_expansion_required"
                ),
                "authorizes_only": (
                    "definitions_only_guarded_departure_amplitude_expansion_manifest"
                ),
            },
            "root_not_attracting": {
                "classification": (
                    "in_chart_fast_root_not_normally_attracting_"
                    "invariant_measure_or_chart_expansion_required"
                ),
                "authorizes_only": (
                    "definitions_only_fast_invariant_measure_and_departure_"
                    "chart_expansion_manifest"
                ),
            },
            "inconclusive": {
                "classification": "fixed_Q_fast_attractor_screen_inconclusive",
                "authorizes_only": None,
            },
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "final_cycle_architecture_boundary": {
            "direct_microsecond_marching": False,
            "single_anchor_470_field_is_final_cycle_model": False,
            "required_end_state": (
                "multi_anchor_conservative_q162_slow_flux_closure_with_"
                "stable_memory_exponential_update_and_eliminated_or_averaged_a28"
            ),
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
        raise RuntimeError("fast-attractor manifest already canonicalized")
    starts = _search_design()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(CANONICAL_DIRECTORY / "search_design.npz", starts=starts)
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "selected_split": "q162_active_plus_z280_stable_plus_a28_nonlinear",
        "naive_96_plus_374_split_authorized": False,
        "new_truth_calls": 0,
        "fast_graph_found": False,
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
                "# Fixed-Q fast-attractor manifest WP10c9d6c7c3b5c4f25ca",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The frozen split is `q162 active + z280 strictly stable memory + a28 nonlinear departure`.",
                "",
                "Only the 280D stable-memory block may be eliminated. The tempting 96-slow/374-fast split is explicitly blocked because its proposed 346D eliminated block is not certified Hurwitz.",
                "",
                f"The next screen uses `{SEARCH_START_COUNT}` deterministic bounded starts and zero new truth calls.",
                "",
                "A stable in-chart root may authorize local fast-graph continuation. Clear nonclosure may authorize only a guarded departure-amplitude expansion. Neither branch authorizes a physical microburst or cycle evolution.",
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
