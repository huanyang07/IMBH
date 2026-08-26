#!/usr/bin/env python3
"""Execute the frozen four-step seven-field radial crossing experiment."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizej as parent  # noqa: E402
import run_causal_inner_invariant_cluster_local_structural_audit_wp10c9d6c7c3b5c4f25fizee7 as local_parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_ssprk2_step,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (  # noqa: E402
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "entropy_complete_bounded_radial_crossing_passed"
NUMERICAL_FAILURE = "entropy_complete_bounded_radial_crossing_numerical_failed"
PHYSICAL_FAILURE = "entropy_complete_bounded_radial_crossing_physical_failed"
REPLAY_FAILURE = "entropy_complete_bounded_radial_crossing_replay_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizel_"
    "entropy_complete_fixed_Q_invariant_object_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_bounded_radial_crossing_execution_"
    "wp10c9d6c7c3b5c4f25fizek"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_BOUNDED_"
    "RADIAL_CROSSING_EXECUTION_WP10C9D6C7C3B5C4F25FIZEK_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizek.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizek.py"
RADIAL_SOURCE = parent.RADIAL_SOURCE
RADIAL_TEST = parent.RADIAL_TEST
PARENT_CHECKSUM_MANIFEST_SHA256 = "7704a16a4f205853d09fa5bd0771762f010349d37e039686e8c02de7185c37b2"
RADIAL_SOURCE_SHA256 = "9e1b432c6a9b54bf51181e02eae9ee056268bc554fa2a964b0bd69fa31bf9e07"
RADIAL_TEST_SHA256 = "c88d36371f654ebf3674fdecbafb3c8b7b69bb6b2cd1a4631fe82214e58957a6"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHART_SCALES = np.asarray((1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03))


def _utils(): return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("crossing manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "crossing_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["bounded_crossing_execution_authorized"]
        or summary["maximum_new_trajectory_steps"] != parent.ACCEPTED_STEPS
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
    ):
        raise RuntimeError("crossing execution authorization changed")
    if utils._sha256(ROOT / RADIAL_SOURCE) != RADIAL_SOURCE_SHA256: raise RuntimeError("radial source changed")
    if utils._sha256(ROOT / RADIAL_TEST) != RADIAL_TEST_SHA256: raise RuntimeError("radial source test changed")
    local_hashes = utils._validate_checksums(local_parent.CANONICAL_DIRECTORY)
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("crossing execution requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract, "local_hashes": local_hashes}


def _context_and_seed():
    stage2 = local_parent.frozen_audit.parent.parent.parent.parent
    with np.load(stage2.CANONICAL_DIRECTORY / "audit_envelope.npz", allow_pickle=False) as archive:
        profile = np.asarray(archive["accepted_terminal_base_charts5"], dtype=float)
    source = (
        local_parent.frozen_audit.parent.parent.parent.boundary_diagnostic
        .manifest.parent.engine.execution.source
    )
    context = source._initial_inputs()["base"]["configuration"]["context"]
    charts = np.asarray([
        generalized_maxwell_cattaneo_hydrostatic_embedding(
            chart,
            proper_vertical_frequency=float(context.vertical_frequency.frequency(float(radius))),
        )
        for radius, chart in zip(context.grid.centers, profile, strict=True)
    ])
    return context, charts


def _roundtrip(values: np.ndarray) -> np.ndarray:
    buffer = BytesIO(); np.savez(buffer, primitive_charts=np.asarray(values, dtype=float))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        return np.array(archive["primitive_charts"], copy=True)


def _step_metrics(step, gates: dict) -> dict:
    operators = (step.initial_operator, step.euler_operator, step.accepted_operator)
    metrics = {
        "maximum_imaginary_speed_over_c": max(item.maximum_imaginary_speed_over_c for item in operators),
        "maximum_light_cone_excess_over_c": max(item.maximum_light_cone_excess_over_c for item in operators),
        "maximum_eigenvector_condition_number": max(item.maximum_eigenvector_condition_number for item in operators),
        "maximum_CFL": max(float(item.maximum_CFL_for_timestep) for item in operators),
        "maximum_scaled_chart_change": step.maximum_scaled_chart_change,
        "minimum_height_over_radius": min(item.minimum_height_over_radius for item in operators),
        "maximum_height_over_radius": max(item.maximum_height_over_radius for item in operators),
        "minimum_optical_depth": min(item.minimum_optical_depth for item in operators),
        "maximum_temporal_solve_relative_residual": max(float(np.max(item.temporal_solve_relative_residuals)) for item in operators),
        "exact_flux_balance_relative_defect": step.exact_flux_balance_relative_defect,
        "maximum_incoming_inner_characteristics": max(item.incoming_inner_characteristics for item in operators),
        "maximum_incoming_outer_characteristics": max(item.incoming_outer_characteristics for item in operators),
        "checkpoint_roundtrip_bitwise": bool(np.array_equal(step.accepted_charts, _roundtrip(step.accepted_charts))),
    }
    reasons = []
    checks_max = (
        ("numerical:imaginary_speed", metrics["maximum_imaginary_speed_over_c"], gates["maximum_imaginary_speed_over_c"]),
        ("numerical:light_cone", metrics["maximum_light_cone_excess_over_c"], gates["maximum_light_cone_excess_over_c"]),
        ("numerical:eigenbasis_condition", metrics["maximum_eigenvector_condition_number"], gates["eigenvector_condition_number_max"]),
        ("numerical:CFL", metrics["maximum_CFL"], gates["maximum_CFL"]),
        ("physical:chart_change", metrics["maximum_scaled_chart_change"], gates["maximum_scaled_chart_change_per_step"]),
        ("physical:height_max", metrics["maximum_height_over_radius"], gates["maximum_height_over_radius"]),
        ("numerical:temporal_solve", metrics["maximum_temporal_solve_relative_residual"], gates["maximum_temporal_solve_relative_residual"]),
        ("numerical:flux_balance", metrics["exact_flux_balance_relative_defect"], gates["maximum_exact_flux_balance_relative_defect"]),
        ("physical:inner_incoming", metrics["maximum_incoming_inner_characteristics"], 0),
    )
    for reason, value, maximum in checks_max:
        if value > maximum: reasons.append(reason)
    if metrics["minimum_height_over_radius"] < gates["minimum_height_over_radius"]: reasons.append("physical:height_min")
    if metrics["minimum_optical_depth"] < gates["minimum_optical_depth"]: reasons.append("physical:optical_depth")
    if not metrics["checkpoint_roundtrip_bitwise"]: reasons.append("replay:checkpoint_roundtrip")
    metrics["passed"] = not reasons; metrics["failure_reasons"] = reasons
    return metrics


def _advance(context, seed: np.ndarray, count: int, timestep: float, gates: dict, *, label: str):
    endpoints = [np.array(seed, copy=True)]; euler_stages = []; metrics = []; current = np.array(seed, copy=True)
    failure = None
    for index in range(count):
        start = time.perf_counter()
        step = generalized_maxwell_cattaneo_ssprk2_step(context, current, timestep_seconds=timestep, quadrature_order=8)
        item = _step_metrics(step, gates); item["step"] = index + 1; item["wall_seconds"] = time.perf_counter() - start
        metrics.append(item); euler_stages.append(step.euler_stage_charts)
        print(f"{label} step {index + 1}/{count}: {'passed' if item['passed'] else 'failed'} ({item['wall_seconds']:.2f} s)", flush=True)
        if not item["passed"]:
            failure = item; break
        current = np.array(step.accepted_charts, copy=True); endpoints.append(current)
    return endpoints, euler_stages, metrics, failure


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False); gates = validated["contract"]["binding_gates"]
    start = time.perf_counter(); context, seed = _context_and_seed(); context_seconds = time.perf_counter() - start
    endpoints, euler, steps, failure = _advance(context, seed, parent.ACCEPTED_STEPS, parent.TIMESTEP_SECONDS, gates, label="main")
    matched = None; full_endpoint = None
    if failure is None:
        full, _full_euler, full_metrics, full_failure = _advance(context, seed, 1, 2.0 * parent.TIMESTEP_SECONDS, gates, label="matched-full")
        if full_failure is None:
            full_endpoint = full[-1]
            defect = float(np.max(np.abs((full_endpoint - endpoints[2]) / CHART_SCALES)) / max(float(np.max(np.abs(endpoints[2] / CHART_SCALES))), 1.0))
            matched = {"scaled_state_defect": defect, "maximum_allowed": gates["maximum_matched_endpoint_scaled_state_defect"], "passed": defect <= gates["maximum_matched_endpoint_scaled_state_defect"], "full_step_metrics": full_metrics[0]}
            if not matched["passed"]: failure = {"failure_reasons": ["numerical:matched_endpoint"], **matched}
        else:
            matched = {"passed": False, "failure_reasons": full_failure["failure_reasons"]}; failure = full_failure
    replay_bitwise = False; replay_endpoints = []
    if failure is None:
        replay_endpoints, _replay_euler, replay_metrics, replay_failure = _advance(context, _roundtrip(seed), parent.ACCEPTED_STEPS, parent.TIMESTEP_SECONDS, gates, label="replay")
        replay_bitwise = replay_failure is None and len(replay_endpoints) == len(endpoints) and all(np.array_equal(a, b) for a, b in zip(endpoints, replay_endpoints, strict=True))
        if not replay_bitwise: failure = {"failure_reasons": ["replay:suffix_bitwise"]}
    reasons = [] if failure is None else list(failure.get("failure_reasons", ()))
    if not reasons: classification = PASS_CLASSIFICATION
    elif any(reason.startswith("replay:") for reason in reasons): classification = REPLAY_FAILURE
    elif any(reason.startswith("physical:") for reason in reasons): classification = PHYSICAL_FAILURE
    else: classification = NUMERICAL_FAILURE
    passed = failure is None and len(endpoints) == parent.ACCEPTED_STEPS + 1
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "legacy_seed_elapsed_seconds": 0.18587500000000012,
        "old_rejected_candidate_elapsed_seconds": 0.186,
        "accepted_new_steps": len(endpoints) - 1,
        "accepted_new_horizon_seconds": (len(endpoints) - 1) * parent.TIMESTEP_SECONDS,
        "accepted_terminal_elapsed_seconds": 0.18587500000000012 + (len(endpoints) - 1) * parent.TIMESTEP_SECONDS,
        "crossed_old_rejected_time": bool(len(endpoints) >= 3),
        "context_construction_wall_seconds": context_seconds,
        "execution_wall_seconds": time.perf_counter() - start,
        "steps": steps,
        "matched_endpoint": matched,
        "suffix_replay_bitwise": replay_bitwise,
        "first_failure": failure,
    }
    arrays = {
        "trajectory_charts7": np.asarray(endpoints, dtype=float),
        "euler_stage_charts7": np.asarray(euler, dtype=float).reshape(-1, seed.shape[0], 7),
        "replay_trajectory_charts7": np.asarray(replay_endpoints, dtype=float).reshape(-1, seed.shape[0], 7),
        "matched_full_step_endpoint_charts7": np.asarray(full_endpoint if full_endpoint is not None else np.empty((0, 7)), dtype=float),
        "grid_centers_cm": np.asarray(context.grid.centers, dtype=float),
        "grid_edges_cm": np.asarray(context.grid.edges, dtype=float),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("crossing execution already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "execution_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "execution_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": bool(metrics["passed"]), "all_parent_results_preserved": True, "accepted_new_steps": metrics["accepted_new_steps"], "crossed_old_rejected_time": metrics["crossed_old_rejected_time"], "suffix_replay_bitwise": metrics["suffix_replay_bitwise"], "bounded_radial_crossing_certified": bool(metrics["passed"]), "fixed_Q_invariant_object_manifest_authorized": bool(metrics["passed"]), "fixed_Q_invariant_object_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "local_parent_artifact": local_parent.ARTIFACT, "local_parent_hashes": validated["local_hashes"], "radial_source_sha256": RADIAL_SOURCE_SHA256, "radial_test_sha256": RADIAL_TEST_SHA256})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete bounded radial crossing execution", "", f"Classification: `{metrics['classification']}`.", "", f"Accepted new steps: `{metrics['accepted_new_steps']}`; crossed old rejected time: `{metrics['crossed_old_rejected_time']}`; suffix replay bitwise: `{metrics['suffix_replay_bitwise']}`.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, RADIAL_SOURCE, RADIAL_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _audit(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
