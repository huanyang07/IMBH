#!/usr/bin/env python3
"""Validate the compensated decoder on frozen independent geometry."""

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

import run_causal_inner_compensated_decoder_repair_manifest_wp10c9d6c7c3b5c4f25cl as manifest  # noqa: E402
import run_causal_inner_shell_gated_atlas_geometry_preflight_wp10c9d6c7c3b5c4f25ci as prior_geometry  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cm"
MANIFEST_COMMIT = "cce685eeed3be66a7918bd7718c279989eb01c7e"
MANIFEST_PARENT = "2e8e0368b53d338e8d093ddd3d79ae001caec2ad"
MANIFEST_TREE = "61a4b27acc6fce0c118dbe3bdfd898717b2d9a5c"

FULL_CLASSIFICATION = "compensated_decoder_independent_geometry_valid_to_0p015"
PARTIAL_CLASSIFICATION = (
    "compensated_decoder_independent_geometry_valid_to_0p0125"
)
FAIL_CLASSIFICATION = "compensated_decoder_independent_geometry_failed"
FULL_AUTHORIZED_NEXT = "definitions_only_recentered_transition_forecast_manifest"
FAIL_AUTHORIZED_NEXT = "definitions_only_decoder_architecture_revision_manifest"

ARTIFACT = (
    "causal_inner_compensated_decoder_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cm"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_compensated_decoder_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cm.py"
)
THIS_TEST = (
    "tests/test_causal_inner_compensated_decoder_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cm.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COMPENSATED_DECODER_"
    "GEOMETRY_PREFLIGHT_WP10C9D6C7C3B5C4F25CM_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

REPAIR_ARRAYS = (
    manifest.CANONICAL_DIRECTORY / "compensated_decoder_repair.npz"
)
HOLDOUT_DESIGN = manifest.CANONICAL_DIRECTORY / "holdout_design.npz"

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


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("compensated decoder manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("compensated decoder manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("compensated decoder manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    metrics = _read(manifest.CANONICAL_DIRECTORY / "design_metrics.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    directions = _load_npz(HOLDOUT_DESIGN)
    labels = tuple(
        _read(manifest.CANONICAL_DIRECTORY / "holdout_design.json")["labels"]
    )
    repair = _load_npz(REPAIR_ARRAYS)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_independent_geometry_candidates"] != 8
        or not summary[
            "independently_validated_rate_field_preserved_algebraically"
        ]
        or summary["new_truth_rate_calls"] != 0
        or summary["trajectory_authorized"]
        or not all(metrics["checks"].values())
        or directions["directions"].shape != (4, 28)
        or not np.array_equal(
            directions["component_bounds"],
            np.asarray(manifest.HOLDOUT_COMPONENT_BOUNDS),
        )
        or len(labels) != 4
        or repair["decoder_repair_centers"].shape != (16, 28)
        or repair["decoder_repair_coefficients"].shape != (16, 560)
        or contract["decision"]["full_pass_classification"]
        != FULL_CLASSIFICATION
        or contract["decision"]["full_pass_authorizes_only"]
        != FULL_AUTHORIZED_NEXT
    ):
        raise RuntimeError("compensated decoder geometry contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"compensated decoder source changed: {relative}")
    thread_environment = prior_geometry.geometry_tools.manifest.parent.vector_field.THREAD_ENVIRONMENT
    for name, expected in thread_environment.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("compensated decoder preflight requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "hashes": hashes,
        "directions": np.asarray(directions["directions"], dtype=float),
        "bounds": np.asarray(directions["component_bounds"], dtype=float),
        "labels": labels,
        "repair": repair,
    }


def _old_extended_delta(model, old_extension: dict, exact_delta, departure):
    online = np.concatenate(
        (
            np.zeros(162, dtype=float),
            model.memory_basis.T @ np.asarray(exact_delta, dtype=float),
            np.asarray(departure, dtype=float),
        )
    )
    old_delta = model.decoded_delta(online)
    weight = manifest.parent.atlas._shell_weight(
        float(np.max(np.abs(old_delta)))
    )
    shell_correction = manifest.parent.atlas._extension_value(
        departure,
        old_extension["extension_center_directions"],
        old_extension["decoder_even4_coefficients"],
        old_extension["decoder_odd5_coefficients"],
    )
    return online, old_delta + weight * shell_correction, weight


def _evaluate_decoder(
    model,
    old_extension: dict,
    repair: dict,
    exact_delta: np.ndarray,
    departure: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    online, old_extended, weight = _old_extended_delta(
        model, old_extension, exact_delta, departure
    )
    repair_value = manifest._repair_value(
        departure,
        repair["decoder_repair_centers"],
        repair["decoder_repair_coefficients"],
    )
    applied_repair = weight * repair_value
    repaired_delta = old_extended + applied_repair
    repaired_state = model.base_state + (
        model.columns.ravel() * repaired_delta
    ).reshape(model.base_state.shape)
    repaired_coordinate, factors = model.coordinate(repaired_state)
    physical = manifest.parent.vector_field.manifest.parent.geometry.chart_tools._state_audit(
        model.components["context"], repaired_state
    )

    old_rate_extension = manifest.parent.atlas._extension_value(
        departure,
        old_extension["extension_center_directions"],
        old_extension["full_state_rate_even4_coefficients"],
        old_extension["full_state_rate_odd5_coefficients"],
    )
    old_full_rate = (
        model.base_rate
        + model.generator @ old_extended
        + model.departure_basis @ model.nonlinear_departure(departure)
        + weight * old_rate_extension
    )
    uncompensated = old_full_rate + model.generator @ applied_repair
    compensated = uncompensated - model.generator @ applied_repair
    metrics = {
        "shell_weight": weight,
        "old_decoder_full_state_relative_error": _relative_error(
            old_extended, exact_delta
        ),
        "repaired_decoder_full_state_relative_error": _relative_error(
            repaired_delta, exact_delta
        ),
        "repaired_decoder_coordinate_relative_mismatch": _relative_error(
            repaired_coordinate, online
        ),
        "compensated_full_rate_invariance_defect": _relative_error(
            compensated, old_full_rate
        ),
        "repaired_minimum_reconstruction_factor": min(
            float(np.min(factors)), physical["minimum_reconstruction_factor"]
        ),
        "repaired_maximum_H_over_R": physical["maximum_h_over_r"],
        "repaired_minimum_scattering_optical_depth": physical[
            "minimum_scattering_optical_depth"
        ],
    }
    arrays = {
        "online_coordinate": online,
        "old_extended_delta": old_extended,
        "decoder_repair": applied_repair,
        "repaired_delta": repaired_delta,
        "repaired_coordinate": repaired_coordinate,
        "old_full_state_rate": old_full_rate,
        "compensated_full_state_rate": compensated,
    }
    return metrics, arrays


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
        "maximum_old_decoder_full_state_relative_error": maximum(
            "old_decoder_full_state_relative_error"
        ),
        "maximum_repaired_decoder_full_state_relative_error": maximum(
            "repaired_decoder_full_state_relative_error"
        ),
        "maximum_repaired_decoder_coordinate_relative_mismatch": maximum(
            "repaired_decoder_coordinate_relative_mismatch"
        ),
        "maximum_compensated_full_rate_invariance_defect": maximum(
            "compensated_full_rate_invariance_defect"
        ),
        "minimum_repaired_reconstruction_factor": minimum(
            "repaired_minimum_reconstruction_factor", math.inf
        ),
        "maximum_repaired_H_over_R": maximum("repaired_maximum_H_over_R"),
        "minimum_repaired_scattering_optical_depth": minimum(
            "repaired_minimum_scattering_optical_depth"
        ),
        "candidates": candidates,
    }


def _gate_checks(metrics: dict, gates: dict, component_bound: float) -> dict:
    return {
        "completed": metrics["completed_candidate_count"]
        == gates["completed_candidate_count_equal"] // 2,
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
        "decoder_error": metrics[
            "maximum_repaired_decoder_full_state_relative_error"
        ] <= gates["maximum_repaired_decoder_full_state_relative_error"],
        "decoder_coordinate": metrics[
            "maximum_repaired_decoder_coordinate_relative_mismatch"
        ] <= gates["maximum_repaired_decoder_coordinate_relative_mismatch"],
        "rate_invariance": metrics[
            "maximum_compensated_full_rate_invariance_defect"
        ] <= gates["maximum_compensated_full_rate_invariance_defect"],
        "repaired_reconstruction": metrics[
            "minimum_repaired_reconstruction_factor"
        ] >= gates["minimum_reconstruction_factor"],
        "repaired_height": metrics["maximum_repaired_H_over_R"]
        <= gates["maximum_H_over_R"],
        "repaired_optical_depth": metrics[
            "minimum_repaired_scattering_optical_depth"
        ] >= gates["minimum_scattering_optical_depth"],
    }


def _execute(frozen: dict) -> tuple[dict, dict[str, np.ndarray]]:
    model = manifest.parent.vector_field.ReducedVectorField()
    old_extension = _load_npz(manifest.OLD_EXTENSION)
    components = prior_geometry.geometry_tools.prior_geometry._prepare_components()
    family_metrics, family = prior_geometry.geometry_tools.prior_geometry.chart_tools._departure_family()
    all_candidates = []
    all_failures = []
    rung_records = []
    arrays = {
        "candidate_primitive_states": [],
        "candidate_scaled_deltas": [],
        "candidate_departure_coordinates": [],
        "candidate_component_bounds": [],
        "candidate_direction_indices": [],
        "online_coordinates": [],
        "old_extended_scaled_deltas": [],
        "decoder_repairs": [],
        "repaired_scaled_deltas": [],
        "repaired_online_coordinates": [],
        "old_full_state_rates_per_second": [],
        "compensated_full_state_rates_per_second": [],
    }
    began = time.perf_counter()
    for rung_index, component_bound in enumerate(frozen["bounds"]):
        candidates = []
        failures = []
        local = prior_geometry._local_contract(
            prior_geometry.manifest._contract(), float(component_bound)
        )
        for direction_index, direction in enumerate(frozen["directions"]):
            candidate_index = len(all_candidates) + len(all_failures)
            try:
                candidate, exact_arrays = prior_geometry.geometry_tools.prior_geometry.chart_tools._retract_candidate(
                    components,
                    family["departure_basis"],
                    family["stable_memory_basis"],
                    direction,
                    1,
                    float(component_bound),
                    local,
                )
                decoder_metrics, decoder_arrays = _evaluate_decoder(
                    model,
                    old_extension,
                    frozen["repair"],
                    exact_arrays["scaled_delta"],
                    exact_arrays["departure_coordinates"],
                )
                candidate.update(
                    {
                        "candidate_index": candidate_index,
                        "rung_index": rung_index,
                        "direction_index": direction_index,
                        "direction_label": frozen["labels"][direction_index],
                        **decoder_metrics,
                    }
                )
                candidates.append(candidate)
                all_candidates.append(candidate)
                arrays["candidate_primitive_states"].append(
                    exact_arrays["primitive_state"]
                )
                arrays["candidate_scaled_deltas"].append(
                    exact_arrays["scaled_delta"]
                )
                arrays["candidate_departure_coordinates"].append(
                    exact_arrays["departure_coordinates"]
                )
                arrays["candidate_component_bounds"].append(component_bound)
                arrays["candidate_direction_indices"].append(direction_index)
                for name, source in (
                    ("online_coordinates", "online_coordinate"),
                    ("old_extended_scaled_deltas", "old_extended_delta"),
                    ("decoder_repairs", "decoder_repair"),
                    ("repaired_scaled_deltas", "repaired_delta"),
                    ("repaired_online_coordinates", "repaired_coordinate"),
                    ("old_full_state_rates_per_second", "old_full_state_rate"),
                    (
                        "compensated_full_state_rates_per_second",
                        "compensated_full_state_rate",
                    ),
                ):
                    arrays[name].append(decoder_arrays[source])
                status = "accepted"
            except Exception as error:  # fail closed on first candidate failure
                failure = {
                    "candidate_index": candidate_index,
                    "rung_index": rung_index,
                    "direction_index": direction_index,
                    "direction_label": frozen["labels"][direction_index],
                    "reason": f"{type(error).__name__}: {error}",
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
            frozen["contract"]["future_independent_geometry_gates"],
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
        "new_truth_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_physical_states": 0,
        "wall_seconds": time.perf_counter() - began,
    }
    packed = {
        name: np.asarray(values, dtype=(int if "indices" in name else float))
        for name, values in arrays.items()
    }
    packed["frozen_directions"] = frozen["directions"]
    return metrics, packed


def _classify(metrics: dict) -> dict:
    passing = int(metrics["passing_rung_count"])
    if passing == 2:
        return {
            "classification": FULL_CLASSIFICATION,
            "authorized_next": FULL_AUTHORIZED_NEXT,
            "largest_passing_component_bound": 0.015,
            "passed": True,
        }
    if passing == 1:
        return {
            "classification": PARTIAL_CLASSIFICATION,
            "authorized_next": FAIL_AUTHORIZED_NEXT,
            "largest_passing_component_bound": 0.0125,
            "passed": False,
        }
    return {
        "classification": FAIL_CLASSIFICATION,
        "authorized_next": FAIL_AUTHORIZED_NEXT,
        "largest_passing_component_bound": 0.01,
        "passed": False,
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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
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
        raise RuntimeError("compensated decoder geometry already canonicalized")
    metrics, arrays = _execute(frozen)
    decision = _classify(metrics)
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
        "new_truth_rate_calls": 0,
        "independently_validated_rate_field_preserved_algebraically": True,
        "geometry_candidate_became_atlas_center": False,
        "trajectory_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": decision["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "geometry_metrics.json", metrics)
    _write_npz(CANONICAL_DIRECTORY / "holdout_geometry.npz", arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "repair_arrays_sha256": _sha(REPAIR_ARRAYS),
            "holdout_design_sha256": _sha(HOLDOUT_DESIGN),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        prior_geometry.THIS_RUNNER,
        prior_geometry.THIS_TEST,
    )
    thread_environment = prior_geometry.geometry_tools.manifest.parent.vector_field.THREAD_ENVIRONMENT
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name) for name in thread_environment
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Compensated decoder geometry preflight WP10c9d6c7c3b5c4f25cm",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"Passing independent geometry rungs: `{summary['passing_rung_count']}` of `2`; largest passing component bound: `{summary['largest_passing_component_bound']:.6g}`.",
                "",
                f"Completed `{summary['completed_candidate_count']}` new mixed-corner candidates with `{summary['failed_candidate_count']}` failures and zero rate calls. The previously validated physical rate field is preserved algebraically.",
                "",
                f"Authorized next artifact: `{summary['authorized_next']}`. No holdout became an atlas center and no trajectory, cycle evolution, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
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
