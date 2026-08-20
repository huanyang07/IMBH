#!/usr/bin/env python3
"""Execute the frozen guarded departure-amplitude geometry preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_expanded_departure_chart_preflight_wp10c9d6c7c3b5c4f25bc as prior_geometry  # noqa: E402
import run_causal_inner_guarded_departure_amplitude_expansion_manifest_wp10c9d6c7c3b5c4f25cc as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cd"
MANIFEST_COMMIT = "4edc39c9fe058b62c6d214b6529450c059fd383b"
MANIFEST_PARENT = "476ff40b1e84d209072ebd3c94b7c3c466a04cfa"
MANIFEST_TREE = "2280f739f526efdd49b0d67af6d2d1fd0a2395c0"

FULL_PASS_CLASSIFICATION = (
    "exact_departure_chart_geometry_reaches_component_bound_0p03"
)
PARTIAL_CLASSIFICATION = (
    "exact_departure_chart_geometry_has_finite_guarded_amplitude_limit"
)
FIRST_FAILURE_CLASSIFICATION = (
    "radial_departure_chart_expansion_failed_at_0p015"
)

ARTIFACT = (
    "causal_inner_guarded_departure_amplitude_expansion_preflight_"
    "wp10c9d6c7c3b5c4f25cd"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_guarded_departure_amplitude_expansion_preflight_"
    "wp10c9d6c7c3b5c4f25cd.py"
)
THIS_TEST = (
    "tests/test_causal_inner_guarded_departure_amplitude_expansion_preflight_"
    "wp10c9d6c7c3b5c4f25cd.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_GUARDED_DEPARTURE_AMPLITUDE_"
    "EXPANSION_PREFLIGHT_WP10C9D6C7C3B5C4F25CD_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LINE_FACTORS = tuple(0.5**index for index in range(12))

_plain = manifest._plain
_read = manifest._read
_write_json = manifest._write_json
_sha = manifest._sha
_checksums = manifest._checksums


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        prior_geometry.THIS_RUNNER,
        prior_geometry.THIS_TEST,
        prior_geometry.chart_tools.THIS_RUNNER,
        prior_geometry.chart_tools.THIS_TEST,
    )
    return {relative: _sha(ROOT / relative) for relative in files}


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("amplitude-expansion manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("amplitude-expansion manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("amplitude-expansion manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    design_json = _read(manifest.CANONICAL_DIRECTORY / "direction_design.json")
    with np.load(
        manifest.CANONICAL_DIRECTORY / "direction_design.npz", allow_pickle=False
    ) as source:
        directions = np.asarray(source["directions"], dtype=float)
        amplitudes = np.asarray(source["amplitude_rungs"], dtype=float)
        signs = np.asarray(source["signs"], dtype=int)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["new_truth_calls"] != 0
        or not summary["stable_memory_remains_dynamic"]
        or summary["old_polynomial_extrapolation_authorized"]
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or not np.array_equal(amplitudes, np.asarray(manifest.AMPLITUDE_RUNGS))
        or not np.array_equal(signs, np.asarray(manifest.SIGNS))
        or directions.shape != (manifest.DIRECTION_COUNT, 28)
        or len(design_json["labels"]) != manifest.DIRECTION_COUNT
    ):
        raise RuntimeError("amplitude-expansion frozen contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"amplitude-expansion manifest source changed: {relative}")
    for name, expected in manifest.parent.vector_field.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("amplitude-expansion preflight requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "hashes": hashes,
        "directions": directions,
        "amplitudes": amplitudes,
        "signs": signs,
        "labels": tuple(design_json["labels"]),
    }


def _retraction_contract(contract: dict, component_bound: float) -> dict:
    return {
        "binding_preflight_gates": {
            **contract["binding_per_rung_gates"],
            "maximum_final_scaled_component": float(component_bound),
        },
        "exact_geometric_retraction": {
            **contract["exact_geometric_retraction"],
            "line_factors": list(LINE_FACTORS),
        },
    }


def _direction_consistency(
    directions: np.ndarray, family: dict[str, np.ndarray]
) -> dict:
    energy = np.asarray(family["energy_directions"], dtype=float).T
    energy /= np.linalg.norm(energy, axis=1)[:, None]
    return {
        "frozen_energy_direction_maximum_absolute_defect": float(
            np.max(np.abs(directions[: manifest.ENERGY_DIRECTION_COUNT] - energy))
        ),
        "all_direction_norm_maximum_absolute_defect": float(
            np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0))
        ),
    }


def _aggregate(candidates: list[dict], failures: list[dict]) -> dict:
    def maximum(name: str, default=math.inf) -> float:
        values = [item.get(name, default) for item in candidates]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item.get(name, default) for item in candidates]
        return float(min(values)) if values else float(default)

    return {
        "completed_candidate_count": len(candidates),
        "failed_candidate_count": len(failures),
        "failures": failures,
        "maximum_coordinate_residual_infinity": maximum(
            "coordinate_residual_infinity"
        ),
        "maximum_normalized_Q3_defect": maximum("normalized_Q3_defect"),
        "maximum_final_scaled_component": maximum("final_scaled_component"),
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum(
            "maximum_reconstruction_factor"
        ),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "maximum_coordinate_Jacobian_condition_number"
        ),
        "minimum_departure_direction_alignment_cosine": minimum(
            "departure_direction_alignment_cosine"
        ),
        "maximum_departure_transverse_fraction": maximum(
            "departure_transverse_fraction"
        ),
        "maximum_coordinate_odd_symmetry_defect": maximum(
            "pair_coordinate_odd_symmetry_defect"
        ),
        "maximum_stable_memory_coordinate_leakage_norm": maximum(
            "stable_memory_coordinate_leakage_norm"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "maximum_Newton_corrections": maximum("Newton_corrections"),
        "maximum_radius_rescalings": maximum("radius_rescalings"),
        "candidates": candidates,
    }


def _gate_checks(metrics: dict, gates: dict, component_bound: float) -> dict:
    return {
        "candidate_count": metrics["completed_candidate_count"]
        == gates["completed_candidate_count_equal"],
        "failure_count": metrics["failed_candidate_count"]
        == gates["failed_candidate_count_equal"],
        "coordinate_closure": metrics["maximum_coordinate_residual_infinity"]
        <= gates["maximum_coordinate_residual_infinity"],
        "Q3_closure": metrics["maximum_normalized_Q3_defect"]
        <= gates["maximum_normalized_Q3_defect"],
        "component_trust": metrics["maximum_final_scaled_component"]
        <= component_bound * (1.0 + 1.0e-12),
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "coordinate_condition": metrics[
            "maximum_coordinate_Jacobian_condition_number"
        ]
        <= gates["maximum_coordinate_Jacobian_condition_number"],
        "direction_alignment": metrics[
            "minimum_departure_direction_alignment_cosine"
        ]
        >= gates["minimum_departure_direction_alignment_cosine"],
        "transverse_distortion": metrics["maximum_departure_transverse_fraction"]
        <= gates["maximum_departure_transverse_fraction"],
        "odd_symmetry": metrics["maximum_coordinate_odd_symmetry_defect"]
        <= gates["maximum_coordinate_odd_symmetry_defect"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
    }


def _execute(frozen: dict) -> tuple[dict, dict[str, np.ndarray]]:
    components = prior_geometry._prepare_components()
    family_metrics, family = prior_geometry.chart_tools._departure_family()
    direction_metrics = _direction_consistency(frozen["directions"], family)
    if (
        direction_metrics["frozen_energy_direction_maximum_absolute_defect"]
        > 1.0e-14
        or direction_metrics["all_direction_norm_maximum_absolute_defect"]
        > 1.0e-14
    ):
        raise RuntimeError("frozen departure directions changed")
    all_candidates = []
    all_failures = []
    rung_records = []
    states = []
    deltas = []
    coordinates = []
    amplitudes = []
    direction_indices = []
    signs = []
    began = time.perf_counter()
    for rung_index, component_bound in enumerate(frozen["amplitudes"]):
        candidates = []
        failures = []
        local_contract = _retraction_contract(
            frozen["contract"], float(component_bound)
        )
        for direction_index, direction in enumerate(frozen["directions"]):
            pair_indices = []
            for sign in frozen["signs"]:
                candidate_index = len(all_candidates) + len(all_failures)
                try:
                    candidate, arrays = prior_geometry.chart_tools._retract_candidate(
                        components,
                        family["departure_basis"],
                        family["stable_memory_basis"],
                        direction,
                        int(sign),
                        float(component_bound),
                        local_contract,
                    )
                    candidate.update(
                        {
                            "candidate_index": candidate_index,
                            "rung_index": rung_index,
                            "direction_index": direction_index,
                            "direction_label": frozen["labels"][direction_index],
                        }
                    )
                    candidates.append(candidate)
                    all_candidates.append(candidate)
                    pair_indices.append(len(candidates) - 1)
                    states.append(arrays["primitive_state"])
                    deltas.append(arrays["scaled_delta"])
                    coordinates.append(arrays["departure_coordinates"])
                    amplitudes.append(component_bound)
                    direction_indices.append(direction_index)
                    signs.append(sign)
                    status = "accepted"
                except prior_geometry.chart_tools.ChartRetractionFailure as error:
                    failure = {
                        "candidate_index": candidate_index,
                        "rung_index": rung_index,
                        "direction_index": direction_index,
                        "direction_label": frozen["labels"][direction_index],
                        "sign": int(sign),
                        "reason": str(error),
                        "diagnostics": error.diagnostics,
                    }
                    failures.append(failure)
                    all_failures.append(failure)
                    status = "failed"
                except (ValueError, FloatingPointError) as error:
                    failure = {
                        "candidate_index": candidate_index,
                        "rung_index": rung_index,
                        "direction_index": direction_index,
                        "direction_label": frozen["labels"][direction_index],
                        "sign": int(sign),
                        "reason": f"{type(error).__name__}: {error}",
                        "diagnostics": {},
                    }
                    failures.append(failure)
                    all_failures.append(failure)
                    status = "failed"
                print(
                    json.dumps(
                        {
                            "rung": rung_index,
                            "component_bound": float(component_bound),
                            "direction": frozen["labels"][direction_index],
                            "sign": int(sign),
                            "status": status,
                            "elapsed_seconds": time.perf_counter() - began,
                        }
                    ),
                    flush=True,
                )
                if failures:
                    break
            if failures:
                break
            if len(pair_indices) != 2:
                raise RuntimeError("signed departure pair changed")
            left = np.asarray(
                coordinates[-2], dtype=float
            )
            right = np.asarray(coordinates[-1], dtype=float)
            denominator = max(
                float(np.linalg.norm(left)) + float(np.linalg.norm(right)),
                np.finfo(float).tiny,
            )
            odd = float(np.linalg.norm(left + right) / denominator)
            candidates[pair_indices[0]]["pair_coordinate_odd_symmetry_defect"] = odd
            candidates[pair_indices[1]]["pair_coordinate_odd_symmetry_defect"] = odd
        aggregate = _aggregate(candidates, failures)
        checks = _gate_checks(
            aggregate,
            frozen["contract"]["binding_per_rung_gates"],
            float(component_bound),
        )
        rung_passed = all(checks.values())
        rung_records.append(
            {
                "rung_index": rung_index,
                "component_bound": float(component_bound),
                "passed": rung_passed,
                "checks": checks,
                **aggregate,
            }
        )
        if not rung_passed:
            break
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "departure_family": family_metrics,
        "direction_consistency": direction_metrics,
        "rungs": rung_records,
        "attempted_rung_count": len(rung_records),
        "passing_rung_count": sum(record["passed"] for record in rung_records),
        "completed_candidate_count": len(all_candidates),
        "failed_candidate_count": len(all_failures),
        "failures": all_failures,
        "new_nonbase_continuous_rate_evaluations": 0,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_physical_states": 0,
        "total_wall_seconds": time.perf_counter() - began,
    }
    arrays = {
        "candidate_primitive_states": np.asarray(states, dtype=float),
        "candidate_scaled_deltas": np.asarray(deltas, dtype=float),
        "candidate_departure_coordinates": np.asarray(coordinates, dtype=float),
        "candidate_component_bounds": np.asarray(amplitudes, dtype=float),
        "candidate_direction_indices": np.asarray(direction_indices, dtype=int),
        "candidate_signs": np.asarray(signs, dtype=int),
        "frozen_directions": frozen["directions"],
    }
    return metrics, arrays


def _classify(metrics: dict, contract: dict) -> dict:
    passing = int(metrics["passing_rung_count"])
    total = len(manifest.AMPLITUDE_RUNGS)
    if passing == total:
        branch = contract["fail_fast_decision"]["all_rungs_pass"]
        largest = max(manifest.AMPLITUDE_RUNGS)
    elif passing > 0:
        branch = contract["fail_fast_decision"]["partial_rungs_pass"]
        largest = manifest.AMPLITUDE_RUNGS[passing - 1]
    else:
        branch = contract["fail_fast_decision"]["first_rung_fails"]
        largest = 1.0e-2
    budgets = contract["cost_and_truth_budget"]
    budget_checks = {
        "rate_evaluations": metrics["new_nonbase_continuous_rate_evaluations"]
        == budgets["new_nonbase_continuous_rate_evaluations_equal"],
        "generator_assemblies": metrics["new_full_generator_assemblies"]
        == budgets["new_full_generator_assemblies_equal"],
        "nonlinear_roots": metrics["new_nonlinear_fixed_Q_roots"]
        == budgets["new_nonlinear_fixed_Q_roots_equal"],
        "propagated_states": metrics["propagated_physical_states"]
        == budgets["propagated_physical_states_equal"],
    }
    return {
        "classification": branch["classification"],
        "passed": all(budget_checks.values()),
        "authorized_next": branch["authorizes_only"],
        "largest_passing_component_bound": largest,
        "budget_checks": budget_checks,
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "FAILED",
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("amplitude-expansion preflight already canonicalized")
    metrics, arrays = _execute(frozen)
    decision = _classify(metrics, frozen["contract"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": decision["classification"],
        "passed": decision["passed"],
        "attempted_rung_count": metrics["attempted_rung_count"],
        "passing_rung_count": metrics["passing_rung_count"],
        "largest_passing_component_bound": decision[
            "largest_passing_component_bound"
        ],
        "completed_candidate_count": metrics["completed_candidate_count"],
        "failed_candidate_count": metrics["failed_candidate_count"],
        "new_truth_calls": 0,
        "stable_memory_remains_dynamic": True,
        "old_polynomial_extrapolation_used": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": decision["authorized_next"],
    }
    payload = {**metrics, "decision": decision, "classification": decision["classification"], "passed": decision["passed"]}
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "preflight_metrics.json", payload)
    _write_npz(CANONICAL_DIRECTORY / "expanded_chart_states.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "direction_design_sha256": _sha(
                manifest.CANONICAL_DIRECTORY / "direction_design.npz"
            ),
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if summary["passed"] else "FAILED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": _source_hashes(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in manifest.parent.vector_field.THREAD_ENVIRONMENT
            },
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
                "# Guarded departure-amplitude expansion preflight WP10c9d6c7c3b5c4f25cd",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"Passing amplitude rungs: `{summary['passing_rung_count']}` of `{len(manifest.AMPLITUDE_RUNGS)}`. "
                f"Largest passing component bound: `{summary['largest_passing_component_bound']:.6g}`.",
                "",
                f"Completed candidates: `{summary['completed_candidate_count']}`; failures: `{summary['failed_candidate_count']}`. "
                "No new rate, generator, nonlinear-root, or propagated-state evaluation was made.",
                "",
                "The 280D stable memory remains a dynamic exponential/L-stable subsystem. "
                "The old local polynomial was not extrapolated as evidence.",
                "",
                f"The only authorized next artifact is `{summary['authorized_next']}`. "
                "No cycle evolution or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
