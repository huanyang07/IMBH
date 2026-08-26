#!/usr/bin/env python3
"""Execute the corrected same-horizon seven-field radial crossing."""

from __future__ import annotations

import argparse
import csv
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

import run_causal_inner_entropy_complete_corrected_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizen as parent  # noqa: E402
import run_causal_inner_entropy_complete_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizek as crossing  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_ssprk2_step,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizeo_"
    "entropy_complete_corrected_bounded_radial_crossing_execution"
)
PASS_CLASSIFICATION = "entropy_complete_corrected_bounded_radial_crossing_certified"
NUMERICAL_FAILURE = "entropy_complete_corrected_bounded_radial_crossing_numerical_failed"
PHYSICAL_FAILURE = "entropy_complete_corrected_bounded_radial_crossing_physical_failed"
REPLAY_FAILURE = "entropy_complete_corrected_bounded_radial_crossing_replay_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizep_"
    "entropy_complete_fixed_Q_invariant_object_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_corrected_bounded_radial_crossing_execution_"
    "wp10c9d6c7c3b5c4f25fizeo"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_CORRECTED_"
    "BOUNDED_RADIAL_CROSSING_EXECUTION_WP10C9D6C7C3B5C4F25FIZEO_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_corrected_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizeo.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_corrected_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizeo.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "2afbca34fef4f8462c79ccee0c94342f13b1da1ebf5193d9e6362545f429061c"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
EXACT_ROWS = np.asarray((0, 1, 2, 3, 5, 6), dtype=int)


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("corrected crossing manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "crossing_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["corrected_crossing_execution_authorized"]
        or summary["maximum_new_trajectory_steps"] != parent.ACCEPTED_STEPS
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["additional_gates"]["fail_closed"]
    ):
        raise RuntimeError("corrected crossing execution authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"corrected crossing manifest source changed: {relative}")
    diagnosis = parent.parent
    diagnosis_hashes = utils._validate_checksums(diagnosis.CANONICAL_DIRECTORY)
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("corrected crossing execution requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract, "diagnosis_hashes": diagnosis_hashes}


def _inventory(operator) -> np.ndarray:
    return np.sum(operator.exact_integrated_states[:, EXACT_ROWS], axis=0)


def _step_exact_balance(step) -> tuple[np.ndarray, np.ndarray]:
    expected = 0.5 * step.timestep_seconds * C * (
        step.initial_operator.exact_global_boundary_source_rate_per_ct
        + step.euler_operator.exact_global_boundary_source_rate_per_ct
    )
    defect = _inventory(step.accepted_operator) - _inventory(step.initial_operator) - expected
    return defect, expected


def _relative(defect: np.ndarray, *references: np.ndarray) -> float:
    scale = max(*(float(np.max(np.abs(item))) for item in references), np.finfo(float).tiny)
    return float(np.max(np.abs(defect)) / scale)


def _advance(
    context,
    seed: np.ndarray,
    count: int,
    timestep: float,
    gates: dict,
    cumulative_gate: float,
    *,
    label: str,
    first_endpoint_reference: np.ndarray | None = None,
):
    endpoints = [np.array(seed, copy=True)]
    euler_stages = []
    metrics = []
    current = np.array(seed, copy=True)
    cumulative_defect = np.zeros(len(EXACT_ROWS), dtype=float)
    cumulative_expected = np.zeros(len(EXACT_ROWS), dtype=float)
    initial_inventory = None
    failure = None
    for index in range(count):
        start = time.perf_counter()
        step = generalized_maxwell_cattaneo_ssprk2_step(
            context,
            current,
            timestep_seconds=timestep,
            quadrature_order=8,
        )
        item = crossing._step_metrics(step, gates)
        defect, expected = _step_exact_balance(step)
        if initial_inventory is None:
            initial_inventory = _inventory(step.initial_operator)
        candidate_defect = cumulative_defect + defect
        candidate_expected = cumulative_expected + expected
        actual = _inventory(step.accepted_operator) - initial_inventory
        cumulative_relative = _relative(candidate_defect, actual, candidate_expected)
        first_parity = True
        if index == 0 and first_endpoint_reference is not None:
            first_parity = bool(np.array_equal(step.accepted_charts, first_endpoint_reference))
        reasons = list(item["failure_reasons"])
        if cumulative_relative > cumulative_gate:
            reasons.append("numerical:cumulative_flux_balance")
        if not first_parity:
            reasons.append("replay:diagnostic_first_endpoint")
        item.update({
            "step": index + 1,
            "wall_seconds": time.perf_counter() - start,
            "cumulative_exact_flux_balance_relative_defect": cumulative_relative,
            "first_endpoint_matches_diagnostic_bitwise": first_parity,
            "passed": not reasons,
            "failure_reasons": reasons,
        })
        metrics.append(item)
        euler_stages.append(step.euler_stage_charts)
        print(
            f"{label} step {index + 1}/{count}: "
            f"{'passed' if item['passed'] else 'failed'} "
            f"change={item['maximum_scaled_chart_change']:.5g} "
            f"({item['wall_seconds']:.2f} s)",
            flush=True,
        )
        if not item["passed"]:
            failure = item
            break
        cumulative_defect = candidate_defect
        cumulative_expected = candidate_expected
        current = np.array(step.accepted_charts, copy=True)
        endpoints.append(current)
    return endpoints, euler_stages, metrics, failure


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    gates = contract["binding_gates"]
    cumulative_gate = contract["additional_gates"]["maximum_cumulative_exact_flux_balance_relative_defect"]
    context_start = time.perf_counter()
    context, seed = crossing._context_and_seed()
    context_seconds = time.perf_counter() - context_start
    with np.load(parent.parent.CANONICAL_DIRECTORY / "execution_arrays.npz", allow_pickle=False) as archive:
        diagnosed_first = np.asarray(archive["trial_accepted_charts7"][2], dtype=float)
    run_start = time.perf_counter()
    endpoints, euler, steps, failure = _advance(
        context,
        seed,
        parent.ACCEPTED_STEPS,
        parent.TIMESTEP_SECONDS,
        gates,
        cumulative_gate,
        label="main",
        first_endpoint_reference=diagnosed_first,
    )
    matched = None
    matched_full_endpoint = None
    if failure is None:
        full, _full_euler, full_metrics, full_failure = _advance(
            context,
            seed,
            1,
            2.0 * parent.TIMESTEP_SECONDS,
            gates,
            cumulative_gate,
            label="matched-full",
        )
        if full_failure is None:
            matched_full_endpoint = full[-1]
            defect = float(np.max(np.abs((matched_full_endpoint - endpoints[2]) / parent.parent.parent.CHART_SCALES)) / max(float(np.max(np.abs(endpoints[2] / parent.parent.parent.CHART_SCALES))), 1.0))
            matched = {"scaled_state_defect": defect, "maximum_allowed": gates["maximum_matched_endpoint_scaled_state_defect"], "passed": defect <= gates["maximum_matched_endpoint_scaled_state_defect"], "full_step_metrics": full_metrics[0]}
            if not matched["passed"]:
                failure = {"failure_reasons": ["numerical:matched_endpoint"], **matched}
        else:
            matched = {"passed": False, "failure_reasons": full_failure["failure_reasons"]}
            failure = full_failure
    replay_endpoints = []
    replay_bitwise = False
    if failure is None:
        checkpoint = crossing._roundtrip(endpoints[parent.REPLAY_CHECKPOINT_STEP])
        replay_endpoints, _replay_euler, _replay_metrics, replay_failure = _advance(
            context,
            checkpoint,
            parent.REPLAY_SUFFIX_STEPS,
            parent.TIMESTEP_SECONDS,
            gates,
            cumulative_gate,
            label="suffix-replay",
        )
        reference = endpoints[parent.REPLAY_CHECKPOINT_STEP:]
        replay_bitwise = replay_failure is None and len(replay_endpoints) == len(reference) and all(np.array_equal(a, b) for a, b in zip(replay_endpoints, reference, strict=True))
        if not replay_bitwise:
            failure = {"failure_reasons": ["replay:suffix_bitwise"]}
    accepted_steps = len(endpoints) - 1
    terminal_elapsed = 0.18587500000000012 + accepted_steps * parent.TIMESTEP_SECONDS
    crossed_old = terminal_elapsed > 0.186
    if failure is None and not crossed_old:
        failure = {"failure_reasons": ["physical:old_rejected_time_not_crossed"]}
    reasons = [] if failure is None else list(failure.get("failure_reasons", ()))
    if not reasons: classification = PASS_CLASSIFICATION
    elif any(reason.startswith("replay:") for reason in reasons): classification = REPLAY_FAILURE
    elif any(reason.startswith("physical:") for reason in reasons): classification = PHYSICAL_FAILURE
    else: classification = NUMERICAL_FAILURE
    passed = failure is None and accepted_steps == parent.ACCEPTED_STEPS
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "legacy_seed_elapsed_seconds": 0.18587500000000012,
        "old_rejected_candidate_elapsed_seconds": 0.186,
        "accepted_new_steps": accepted_steps,
        "accepted_new_horizon_seconds": accepted_steps * parent.TIMESTEP_SECONDS,
        "accepted_terminal_elapsed_seconds": terminal_elapsed,
        "crossed_old_rejected_time": crossed_old,
        "context_construction_wall_seconds": context_seconds,
        "execution_wall_seconds": time.perf_counter() - run_start,
        "steps": steps,
        "matched_endpoint": matched,
        "replay_checkpoint_step": parent.REPLAY_CHECKPOINT_STEP,
        "replay_suffix_steps": parent.REPLAY_SUFFIX_STEPS,
        "suffix_replay_bitwise": replay_bitwise,
        "first_failure": failure,
    }
    arrays = {
        "trajectory_charts7": np.asarray(endpoints),
        "euler_stage_charts7": np.asarray(euler).reshape(-1, seed.shape[0], 7),
        "suffix_replay_trajectory_charts7": np.asarray(replay_endpoints).reshape(-1, seed.shape[0], 7),
        "matched_full_step_endpoint_charts7": np.asarray(matched_full_endpoint if matched_full_endpoint is not None else np.empty((0, 7))),
        "grid_centers_cm": np.asarray(context.grid.centers),
        "grid_edges_cm": np.asarray(context.grid.edges),
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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("corrected crossing execution already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "execution_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "execution_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": bool(metrics["passed"]), "coarse_crossing_rejection_preserved": True, "radial_shear_substep_certificate_preserved": True, "accepted_new_steps": metrics["accepted_new_steps"], "crossed_old_rejected_time": metrics["crossed_old_rejected_time"], "first_endpoint_matches_diagnostic_bitwise": bool(metrics["steps"] and metrics["steps"][0]["first_endpoint_matches_diagnostic_bitwise"]), "suffix_replay_bitwise": metrics["suffix_replay_bitwise"], "bounded_radial_crossing_certified": bool(metrics["passed"]), "fixed_Q_invariant_object_manifest_authorized": bool(metrics["passed"]), "fixed_Q_invariant_object_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "diagnosis_hashes": validated["diagnosis_hashes"]})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete corrected bounded radial crossing execution", "", f"Classification: `{metrics['classification']}`.", "", f"Accepted new steps: `{metrics['accepted_new_steps']}`; crossed old rejected time: `{metrics['crossed_old_rejected_time']}`; suffix replay bitwise: `{metrics['suffix_replay_bitwise']}`.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, crossing.RADIAL_SOURCE, crossing.RADIAL_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _audit(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
