#!/usr/bin/env python3
"""Execute the frozen forward-quadratic blind geometry preflight."""

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

import run_causal_inner_authentic_center_geometry_preflight_wp10c9d6c7c3b5c4f25cs as geometry_parent  # noqa: E402
import run_causal_inner_forward_quadratic_field_revision_manifest_wp10c9d6c7c3b5c4f25cx as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cy"
MANIFEST_COMMIT = "2c1c3545e37e939ef402c094a39f614550507342"
MANIFEST_PARENT = "75d3083671f36dcd69524df0ead95ea3a9c9f516"
MANIFEST_TREE = "ba0e6fc0a5a1982e6a101c805e89db5fb4c300c1"

PASS_CLASSIFICATION = "forward_quadratic_blind_geometry_passed"
FAIL_CLASSIFICATION = "forward_quadratic_blind_geometry_failed"
PASS_AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cz"
FAIL_AUTHORIZED_NEXT = "definitions_only_forward_quadratic_geometry_revision"

ARTIFACT = (
    "causal_inner_forward_quadratic_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cy"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_forward_quadratic_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cy.py"
)
THIS_TEST = (
    "tests/test_causal_inner_forward_quadratic_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cy.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FORWARD_QUADRATIC_GEOMETRY_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25CY_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FIELD_ARRAYS = manifest.CANONICAL_DIRECTORY / "forward_quadratic_local_field.npz"

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("forward-quadratic manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("forward-quadratic manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("forward-quadratic manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    metrics = _read(manifest.CANONICAL_DIRECTORY / "design_metrics.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    closure = _load_npz(FIELD_ARRAYS)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["new_exact_rate_calls"] != 0
        or summary["prospective_geometry_candidate_count"]
        != manifest.NEW_GEOMETRY_COUNT
        or not summary["coefficients_frozen_before_new_geometry_and_truth"]
        or not summary["directions_frozen_before_geometry_and_truth"]
        or not metrics["passed"]
        or not all(metrics["checks"].values())
        or contract["geometry_preflight"]["work_package"] != WORK_PACKAGE
        or closure["blind_directions"].shape
        != (manifest.NEW_GEOMETRY_COUNT, manifest.DEPARTURE_DIMENSION)
        or closure["full_rate_forward_quadratic_coefficients"].shape != (5, 560)
        or closure["q162_Jacobian_affine_coefficients"].shape
        != (4, manifest.PHYSICAL_DIMENSION, 560)
    ):
        raise RuntimeError("forward-quadratic geometry authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"forward-quadratic manifest source changed: {relative}")
    if _sha(FIELD_ARRAYS) != hashes["forward_quadratic_local_field.npz"]:
        raise RuntimeError("forward-quadratic field arrays changed")
    for name, expected in manifest.training._thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("forward-quadratic geometry requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "metrics": metrics,
        "hashes": hashes,
        "closure": closure,
    }


def _candidate_specs(closure: dict[str, np.ndarray]) -> list[dict]:
    return [
        {
            "candidate_index": index,
            "direction": np.asarray(direction, dtype=float),
            "component_bound": float(bound),
            "mixing_magnitude": float(mixing),
            "azimuth_radians": float(azimuth),
        }
        for index, (direction, bound, mixing, azimuth) in enumerate(
            zip(
                closure["blind_directions"],
                closure["blind_component_bounds"],
                closure["blind_mixing_magnitudes"],
                closure["blind_azimuths_radians"],
            )
        )
    ]


def _execute(frozen: dict) -> tuple[dict, dict[str, np.ndarray]]:
    closure = frozen["closure"]
    components = geometry_parent._prepare_center_components(closure)
    center_coordinate = np.asarray(
        closure["authentic_center_absolute_coordinate"], dtype=float
    )
    center_delta = np.asarray(closure["authentic_center_scaled_delta"], dtype=float)
    restriction = np.asarray(
        closure["authentic_center_fixed_restriction"], dtype=float
    )
    memory_basis = restriction[
        manifest.PHYSICAL_DIMENSION : (
            manifest.PHYSICAL_DIMENSION + manifest.MEMORY_DIMENSION
        )
    ].T
    departure_basis = restriction[-manifest.DEPARTURE_DIMENSION :].T
    chart_tools = geometry_parent.prior_geometry.geometry_tools.prior_geometry.chart_tools
    field = manifest.ForwardQuadraticAuthenticCenterField(closure)
    candidates = []
    failures = []
    arrays: dict[str, list[np.ndarray | float | int]] = {
        "candidate_primitive_states": [],
        "candidate_local_scaled_deltas": [],
        "candidate_absolute_scaled_deltas": [],
        "candidate_local_coordinates": [],
        "candidate_absolute_coordinates": [],
        "candidate_directions": [],
        "candidate_component_bounds": [],
        "candidate_mixing_magnitudes": [],
        "candidate_azimuths_radians": [],
        "candidate_active_coordinates": [],
        "decoded_scaled_deltas": [],
        "decoded_absolute_coordinates": [],
        "predicted_full_rates_per_second": [],
        "predicted_coordinate_rates_per_second": [],
        "predicted_q162_Jacobians": [],
    }
    began = time.perf_counter()
    for spec in _candidate_specs(closure):
        started = time.perf_counter()
        try:
            candidate, result_arrays = chart_tools._retract_candidate(
                components,
                departure_basis,
                memory_basis,
                spec["direction"],
                1,
                spec["component_bound"],
                geometry_parent._local_contract(spec["component_bound"]),
            )
            local_delta = np.asarray(result_arrays["scaled_delta"], dtype=float)
            state = np.asarray(result_arrays["primitive_state"], dtype=float)
            local_coordinate = geometry_parent._local_coordinate(
                state,
                local_delta,
                components,
                memory_basis,
                departure_basis,
            )
            absolute_coordinate = center_coordinate + local_coordinate
            absolute_delta = center_delta + local_delta
            decoded_delta = field.decoded_delta(local_coordinate)
            decoded_state = field.decoded_state(local_coordinate)
            decoded_coordinate, decoded_factors = field.model.coordinate(decoded_state)
            decoded_physical = chart_tools._state_audit(
                field.model.components["context"], decoded_state
            )
            exact_physical = chart_tools._state_audit(components["context"], state)
            active = field._active(local_coordinate)
            partition_weight = field.weight(local_coordinate)
            q_jacobian = field.q162_jacobian(local_coordinate)
            singular = np.linalg.svd(q_jacobian, compute_uv=False)
            predicted_full = field.full_state_rate(local_coordinate)
            predicted_coordinate = field.field(local_coordinate)
            record = {
                "candidate_index": spec["candidate_index"],
                "component_bound": spec["component_bound"],
                "mixing_magnitude": spec["mixing_magnitude"],
                "azimuth_radians": spec["azimuth_radians"],
                "new_scaled_state_load": float(np.max(np.abs(local_delta))),
                "local_coordinate_infinity_load": float(
                    np.max(np.abs(local_coordinate))
                ),
                "forward_active_coordinate": float(active[0]),
                "transverse_active_radius": float(np.linalg.norm(active[1:])),
                "partition_weight": partition_weight,
                "decoder_relative_error": _relative_error(
                    decoded_delta, absolute_delta
                ),
                "decoder_coordinate_relative_mismatch": _relative_error(
                    decoded_coordinate, absolute_coordinate
                ),
                "minimum_reconstruction_factor": float(
                    min(
                        np.min(decoded_factors),
                        decoded_physical["minimum_reconstruction_factor"],
                        exact_physical["minimum_reconstruction_factor"],
                    )
                ),
                "maximum_H_over_R": float(
                    max(
                        decoded_physical["maximum_h_over_r"],
                        exact_physical["maximum_h_over_r"],
                    )
                ),
                "minimum_scattering_optical_depth": float(
                    min(
                        decoded_physical["minimum_scattering_optical_depth"],
                        exact_physical["minimum_scattering_optical_depth"],
                    )
                ),
                "predicted_q162_Jacobian_rank": int(
                    np.linalg.matrix_rank(q_jacobian)
                ),
                "predicted_q162_Jacobian_condition_number": float(
                    singular[0] / singular[-1]
                ),
                "wall_seconds": time.perf_counter() - started,
            }
            candidate.update(record)
            candidates.append(candidate)
            arrays["candidate_primitive_states"].append(state)
            arrays["candidate_local_scaled_deltas"].append(local_delta)
            arrays["candidate_absolute_scaled_deltas"].append(absolute_delta)
            arrays["candidate_local_coordinates"].append(local_coordinate)
            arrays["candidate_absolute_coordinates"].append(absolute_coordinate)
            arrays["candidate_directions"].append(spec["direction"])
            arrays["candidate_component_bounds"].append(spec["component_bound"])
            arrays["candidate_mixing_magnitudes"].append(spec["mixing_magnitude"])
            arrays["candidate_azimuths_radians"].append(spec["azimuth_radians"])
            arrays["candidate_active_coordinates"].append(active)
            arrays["decoded_scaled_deltas"].append(decoded_delta)
            arrays["decoded_absolute_coordinates"].append(decoded_coordinate)
            arrays["predicted_full_rates_per_second"].append(predicted_full)
            arrays["predicted_coordinate_rates_per_second"].append(
                predicted_coordinate
            )
            arrays["predicted_q162_Jacobians"].append(q_jacobian)
            status = "accepted"
        except geometry_parent.prior_geometry.geometry_tools.prior_geometry.chart_tools.ChartRetractionFailure as error:
            failures.append(
                {
                    "candidate_index": spec["candidate_index"],
                    "reason": str(error),
                    "diagnostics": error.diagnostics,
                }
            )
            status = "failed"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "candidate_index": spec["candidate_index"],
                    "reason": f"{type(error).__name__}: {error}",
                    "diagnostics": {},
                }
            )
            status = "failed"
        print(
            json.dumps(
                {
                    "candidate": spec["candidate_index"],
                    "mixing": spec["mixing_magnitude"],
                    "status": status,
                    "elapsed_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if failures:
            break

    def minimum(name: str) -> float:
        return float(min((item[name] for item in candidates), default=-math.inf))

    def maximum(name: str) -> float:
        return float(max((item[name] for item in candidates), default=math.inf))

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "completed_candidate_count": len(candidates),
        "failed_candidate_count": len(failures),
        "candidates": candidates,
        "failures": failures,
        "minimum_partition_weight": minimum("partition_weight"),
        "minimum_forward_active_coordinate": minimum(
            "forward_active_coordinate"
        ),
        "maximum_decoder_relative_error": maximum("decoder_relative_error"),
        "maximum_decoder_coordinate_relative_mismatch": maximum(
            "decoder_coordinate_relative_mismatch"
        ),
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "minimum_predicted_q162_Jacobian_rank": int(
            minimum("predicted_q162_Jacobian_rank")
        ),
        "maximum_predicted_q162_Jacobian_condition_number": maximum(
            "predicted_q162_Jacobian_condition_number"
        ),
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "wall_seconds": time.perf_counter() - began,
    }
    array_payload = {
        name: np.asarray(values, dtype=float) for name, values in arrays.items()
    }
    array_payload.update(
        {
            "authentic_center_primitive_state": closure[
                "authentic_center_primitive_state"
            ],
            "authentic_center_scaled_delta": center_delta,
            "authentic_center_absolute_coordinate": center_coordinate,
            "authentic_center_fixed_restriction": restriction,
            "active_departure_basis": closure["active_departure_basis"],
            "frozen_blind_directions": closure["blind_directions"],
            "frozen_blind_component_bounds": closure["blind_component_bounds"],
        }
    )
    return metrics, array_payload


def _checks(metrics: dict, gates: dict) -> dict[str, bool]:
    return {
        "completed": metrics["completed_candidate_count"] == gates["count"],
        "failed": metrics["failed_candidate_count"] == 0,
        "partition": metrics["minimum_partition_weight"]
        >= gates["minimum_partition_weight"],
        "forward_sector": metrics["minimum_forward_active_coordinate"]
        >= manifest.partition.FORWARD_FULL_COORDINATE,
        "decoder": metrics["maximum_decoder_relative_error"]
        <= gates["maximum_decoder_relative_error"],
        "decoder_coordinate": metrics[
            "maximum_decoder_coordinate_relative_mismatch"
        ]
        <= gates["maximum_decoder_coordinate_relative_mismatch"],
        "reconstruction": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "q162_Jacobian_rank": metrics["minimum_predicted_q162_Jacobian_rank"]
        == manifest.PHYSICAL_DIMENSION,
        "q162_Jacobian_condition": metrics[
            "maximum_predicted_q162_Jacobian_condition_number"
        ]
        <= 5.0e3,
        "truth_budget": metrics["new_exact_rate_calls"]
        == gates["new_exact_rate_calls_equal"],
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_fixed_Q_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
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
        raise RuntimeError("forward-quadratic geometry artifact already exists")
    metrics, arrays = _execute(frozen)
    gates = frozen["contract"]["geometry_preflight"]
    checks = _checks(metrics, gates)
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = PASS_AUTHORIZED_NEXT if passed else FAIL_AUTHORIZED_NEXT
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "geometry_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "geometry_metrics.json",
        {"checks": checks, "passed": passed, **metrics},
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_execution_contract.json", frozen["contract"]
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "field_arrays_sha256": _sha(FIELD_ARRAYS),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "completed_candidate_count": metrics["completed_candidate_count"],
        "failed_candidate_count": metrics["failed_candidate_count"],
        "directions_changed_after_manifest": False,
        "coefficients_changed_after_manifest": False,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative)
                for relative in (
                    THIS_RUNNER,
                    THIS_TEST,
                    manifest.THIS_RUNNER,
                    manifest.THIS_TEST,
                    geometry_parent.THIS_RUNNER,
                    geometry_parent.THIS_TEST,
                )
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in manifest.training._thread_environment()
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
                "# Forward-quadratic blind geometry preflight WP10c9d6c7c3b5c4f25cy",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Accepted `{metrics['completed_candidate_count']}` of four frozen directions; failures: `{metrics['failed_candidate_count']}`.",
                "",
                f"The minimum forward active coordinate and partition weight are `{metrics['minimum_forward_active_coordinate']:.6e}` and `{metrics['minimum_partition_weight']:.6e}`. Maximum decoder and coordinate-roundtrip errors are `{metrics['maximum_decoder_relative_error']:.6e}` and `{metrics['maximum_decoder_coordinate_relative_mismatch']:.6e}`.",
                "",
                "No exact rate, complete generator, nonlinear fixed-Q root, or propagated state was evaluated. Directions and field coefficients remained frozen.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No trajectory, microburst, predictive cycle, or reduced slow evolution is authorized.",
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
