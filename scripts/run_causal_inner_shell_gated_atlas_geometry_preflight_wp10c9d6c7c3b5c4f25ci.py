#!/usr/bin/env python3
"""Execute mixed forward-sector geometry holdouts for the local atlas."""

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

import run_causal_inner_shell_gated_atlas_geometry_manifest_wp10c9d6c7c3b5c4f25ch as manifest  # noqa: E402
import run_causal_inner_guarded_departure_amplitude_expansion_preflight_wp10c9d6c7c3b5c4f25cd as geometry_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ci"
MANIFEST_COMMIT = "47c8db0a35a46a64d71acd6bd155be9bf9c3eace"
MANIFEST_PARENT = "7251f59456cef29363cb8b217a844e9c0805b2f7"
MANIFEST_TREE = "eefdebbefbda3bd9e09d52ab50b801becc511f39"

FULL_CLASSIFICATION = "shell_gated_atlas_mixed_geometry_valid_to_0p015"
PARTIAL_CLASSIFICATION = "shell_gated_atlas_mixed_geometry_valid_to_0p0125"
FAIL_CLASSIFICATION = (
    "shell_gated_atlas_mixed_geometry_failed_before_recenter_margin"
)

ARTIFACT = (
    "causal_inner_shell_gated_atlas_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25ci"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_shell_gated_atlas_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25ci.py"
)
THIS_TEST = (
    "tests/test_causal_inner_shell_gated_atlas_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25ci.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SHELL_GATED_ATLAS_"
    "GEOMETRY_PREFLIGHT_WP10C9D6C7C3B5C4F25CI_2026-08-19.md"
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
_load_npz = manifest._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("atlas geometry manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("atlas geometry manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("atlas geometry manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    atlas_hashes = _checksums(manifest.parent.CANONICAL_DIRECTORY)
    directions = _load_npz(manifest.PARENT_HOLDOUT)
    labels = tuple(
        _read(manifest.parent.CANONICAL_DIRECTORY / "holdout_design.json")[
            "labels"
        ]
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_geometry_candidates"] != 8
        or summary["new_truth_calls"] != 0
        or summary["trajectory_authorized"]
        or directions["directions"].shape != (4, 28)
        or not np.array_equal(
            directions["component_bounds"],
            np.asarray(manifest.parent.HOLDOUT_COMPONENT_BOUNDS),
        )
        or len(labels) != 4
    ):
        raise RuntimeError("atlas geometry execution contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"atlas geometry manifest source changed: {relative}")
    for name, expected in geometry_tools.manifest.parent.vector_field.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("atlas geometry preflight requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "hashes": hashes,
        "atlas_hashes": atlas_hashes,
        "directions": np.asarray(directions["directions"], dtype=float),
        "bounds": np.asarray(directions["component_bounds"], dtype=float),
        "labels": labels,
    }


def _local_contract(contract: dict, component_bound: float) -> dict:
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
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "candidates": candidates,
    }


def _gate_checks(metrics: dict, gates: dict, component_bound: float) -> dict:
    return {
        "completed": metrics["completed_candidate_count"]
        == gates["completed_candidate_count_equal"],
        "failed": metrics["failed_candidate_count"]
        == gates["failed_candidate_count_equal"],
        "coordinate_residual": metrics["maximum_coordinate_residual_infinity"]
        <= gates["maximum_coordinate_residual_infinity"],
        "Q3": metrics["maximum_normalized_Q3_defect"]
        <= gates["maximum_normalized_Q3_defect"],
        "component_bound": metrics["maximum_final_scaled_component"]
        <= component_bound + 1.0e-12,
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "coordinate_condition": metrics[
            "maximum_coordinate_Jacobian_condition_number"
        ] <= gates["maximum_coordinate_Jacobian_condition_number"],
        "direction_alignment": metrics[
            "minimum_departure_direction_alignment_cosine"
        ] >= gates["minimum_departure_direction_alignment_cosine"],
        "transverse_fraction": metrics["maximum_departure_transverse_fraction"]
        <= gates["maximum_departure_transverse_fraction"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
    }


def _execute(frozen: dict) -> tuple[dict, dict[str, np.ndarray]]:
    components = geometry_tools.prior_geometry._prepare_components()
    family_metrics, family = geometry_tools.prior_geometry.chart_tools._departure_family()
    all_candidates = []
    all_failures = []
    rung_records = []
    states = []
    deltas = []
    coordinates = []
    amplitudes = []
    direction_indices = []
    began = time.perf_counter()
    for rung_index, component_bound in enumerate(frozen["bounds"]):
        candidates = []
        failures = []
        local = _local_contract(frozen["contract"], float(component_bound))
        for direction_index, direction in enumerate(frozen["directions"]):
            candidate_index = len(all_candidates) + len(all_failures)
            try:
                candidate, arrays = geometry_tools.prior_geometry.chart_tools._retract_candidate(
                    components,
                    family["departure_basis"],
                    family["stable_memory_basis"],
                    direction,
                    1,
                    float(component_bound),
                    local,
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
                states.append(arrays["primitive_state"])
                deltas.append(arrays["scaled_delta"])
                coordinates.append(arrays["departure_coordinates"])
                amplitudes.append(component_bound)
                direction_indices.append(direction_index)
                status = "accepted"
            except geometry_tools.prior_geometry.chart_tools.ChartRetractionFailure as error:
                failure = {
                    "candidate_index": candidate_index,
                    "rung_index": rung_index,
                    "direction_index": direction_index,
                    "direction_label": frozen["labels"][direction_index],
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
                        "status": status,
                        "elapsed_seconds": time.perf_counter() - began,
                    }
                ),
                flush=True,
            )
            if failures:
                break
        aggregate = _aggregate(candidates, failures)
        checks = _gate_checks(
            aggregate,
            frozen["contract"]["binding_per_rung_gates"],
            float(component_bound),
        )
        rung_records.append(
            {
                "rung_index": rung_index,
                "component_bound": float(component_bound),
                "passed": all(checks.values()),
                "checks": checks,
                **aggregate,
            }
        )
        if not rung_records[-1]["passed"]:
            break
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "departure_family": family_metrics,
        "rungs": rung_records,
        "attempted_rung_count": len(rung_records),
        "passing_rung_count": sum(item["passed"] for item in rung_records),
        "completed_candidate_count": len(all_candidates),
        "failed_candidate_count": len(all_failures),
        "failures": all_failures,
        "new_nonbase_continuous_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_physical_states": 0,
        "wall_seconds": time.perf_counter() - began,
    }
    arrays = {
        "candidate_primitive_states": np.asarray(states, dtype=float),
        "candidate_scaled_deltas": np.asarray(deltas, dtype=float),
        "candidate_departure_coordinates": np.asarray(coordinates, dtype=float),
        "candidate_component_bounds": np.asarray(amplitudes, dtype=float),
        "candidate_direction_indices": np.asarray(direction_indices, dtype=int),
        "frozen_directions": frozen["directions"],
    }
    return metrics, arrays


def _classify(metrics: dict, contract: dict) -> dict:
    passing = int(metrics["passing_rung_count"])
    if passing == 2:
        branch = contract["decision"]["both_rungs_pass"]
        largest = 0.015
    elif passing == 1:
        branch = contract["decision"]["only_first_rung_passes"]
        largest = 0.0125
    else:
        branch = contract["decision"]["first_rung_fails"]
        largest = 0.01
    budget = contract["cost_budget"]
    budget_checks = {
        "rate": metrics["new_nonbase_continuous_rate_evaluations"]
        == budget["new_nonbase_continuous_rate_evaluations_equal"],
        "generator": metrics["new_complete_generator_assemblies"]
        == budget["new_complete_generator_assemblies_equal"],
        "root": metrics["new_nonlinear_fixed_Q_roots"]
        == budget["new_nonlinear_fixed_Q_roots_equal"],
        "propagation": metrics["propagated_physical_states"]
        == budget["propagated_physical_states_equal"],
    }
    return {
        "classification": branch["classification"],
        "authorized_next": branch["authorizes_only"],
        "largest_passing_component_bound": largest,
        "passed": all(budget_checks.values()),
        "budget_checks": budget_checks,
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": MANIFEST_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("atlas geometry preflight already canonicalized")
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
        "geometry_candidate_became_atlas_center": False,
        "trajectory_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": decision["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "geometry_metrics.json", {**metrics, "decision": decision})
    _write_npz(CANONICAL_DIRECTORY / "holdout_geometry.npz", arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_commit": MANIFEST_COMMIT, "manifest_parent": MANIFEST_PARENT, "manifest_tree": MANIFEST_TREE, "manifest_hashes": frozen["hashes"], "atlas_hashes": frozen["atlas_hashes"], "holdout_design_sha256": _sha(manifest.PARENT_HOLDOUT)})
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST, geometry_tools.THIS_RUNNER, geometry_tools.THIS_TEST)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED", "execution_commit": _git("rev-parse", "HEAD"), "execution_tree": _git("rev-parse", "HEAD^{tree}"), "tracked_worktree_clean_at_start": True, "runner": THIS_RUNNER, "test": THIS_TEST, "report": REPORT_RELATIVE, "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name) for name in geometry_tools.manifest.parent.vector_field.THREAD_ENVIRONMENT}})
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(("# Shell-gated atlas geometry preflight WP10c9d6c7c3b5c4f25ci", "", "## Classification", "", f"`{summary['classification']}`", "", f"Passing geometry rungs: `{summary['passing_rung_count']}` of `2`; largest passing bound: `{summary['largest_passing_component_bound']:.6g}`.", "", f"Completed `{summary['completed_candidate_count']}` mixed forward-sector candidates with `{summary['failed_candidate_count']}` failures and zero rate calls.", "", f"Authorized next artifact: `{summary['authorized_next']}`. No geometry candidate became an atlas center and no trajectory or cycle evolution is authorized.", "")), encoding="utf-8")
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)
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
