#!/usr/bin/env python3
"""Execute the frozen nonpropagating radial shear-substep diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

import run_causal_inner_entropy_complete_radial_substep_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizel as parent  # noqa: E402
import run_causal_inner_entropy_complete_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizek as crossing  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_ssprk2_step,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizem_"
    "entropy_complete_radial_substep_diagnosis_execution"
)
PASS_CLASSIFICATION = "entropy_complete_radial_shear_substep_certified"
FAIL_CLASSIFICATION = "entropy_complete_radial_shear_substep_diagnosis_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizen_"
    "entropy_complete_corrected_bounded_radial_crossing_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_radial_substep_diagnosis_execution_"
    "wp10c9d6c7c3b5c4f25fizem"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_RADIAL_"
    "SUBSTEP_DIAGNOSIS_EXECUTION_WP10C9D6C7C3B5C4F25FIZEM_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_radial_substep_diagnosis_execution_wp10c9d6c7c3b5c4f25fizem.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_radial_substep_diagnosis_execution_wp10c9d6c7c3b5c4f25fizem.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "11085798da106996fabbb4c25a4c2f98d4d24177b818f915e07a9fdcc5144914"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
COMPONENT_NAMES = ("logSigma", "radial_velocity", "azimuthal_velocity", "logPi", "chi", "logH", "betaH")


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("substep diagnosis manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "diagnosis_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["diagnosis_execution_authorized"]
        or summary["maximum_new_trajectory_steps"] != 0
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
    ):
        raise RuntimeError("substep diagnosis authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"substep diagnosis source changed: {relative}")
    crossing_lock = utils._read_json(crossing.CANONICAL_DIRECTORY / "input_lock.json")
    if utils._sha256(ROOT / crossing.RADIAL_SOURCE) != crossing_lock["radial_source_sha256"]:
        raise RuntimeError("radial source changed")
    if utils._sha256(ROOT / crossing.RADIAL_TEST) != crossing_lock["radial_test_sha256"]:
        raise RuntimeError("radial source test changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("substep diagnosis execution requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    gates = crossing.parent._utils()._read_json(
        crossing.parent.CANONICAL_DIRECTORY / "crossing_contract.json"
    )["binding_gates"]
    context_start = time.perf_counter()
    context, seed = crossing._context_and_seed()
    context_seconds = time.perf_counter() - context_start
    trials = []
    accepted_candidates = []
    euler_candidates = []
    component_changes = []
    run_start = time.perf_counter()
    for timestep in parent.TIMESTEPS_SECONDS:
        start = time.perf_counter()
        step = generalized_maxwell_cattaneo_ssprk2_step(
            context,
            seed,
            timestep_seconds=timestep,
            quadrature_order=8,
        )
        item = crossing._step_metrics(step, gates)
        scaled = np.max(
            np.abs((step.accepted_charts - seed) / parent.CHART_SCALES), axis=0
        )
        dominant = int(np.argmax(scaled))
        nonchart_reasons = [
            reason for reason in item["failure_reasons"]
            if reason != "physical:chart_change"
        ]
        item.update({
            "timestep_seconds": timestep,
            "wall_seconds": time.perf_counter() - start,
            "componentwise_maximum_scaled_chart_change": scaled.tolist(),
            "dominant_component": COMPONENT_NAMES[dominant],
            "dominant_component_index": dominant,
            "nonchart_failure_reasons": nonchart_reasons,
            "original_chart_gate_passed": item["maximum_scaled_chart_change"] <= gates["maximum_scaled_chart_change_per_step"],
            "headroom_gate_passed": item["maximum_scaled_chart_change"] <= parent.HEADROOM_CHANGE,
        })
        trials.append(item)
        accepted_candidates.append(step.accepted_charts)
        euler_candidates.append(step.euler_stage_charts)
        component_changes.append(scaled)
        print(
            f"substep {timestep:.7g} s: change={item['maximum_scaled_chart_change']:.6g}, "
            f"dominant={item['dominant_component']}, reasons={item['failure_reasons']} "
            f"({item['wall_seconds']:.2f} s)",
            flush=True,
        )
    changes = [item["maximum_scaled_chart_change"] for item in trials]
    orders = [math.log(changes[index] / changes[index + 1], 2.0) for index in range(len(changes) - 1)]
    eligible = [
        item for item in trials
        if not item["nonchart_failure_reasons"]
        and item["original_chart_gate_passed"]
        and item["headroom_gate_passed"]
        and item["checkpoint_roundtrip_bitwise"]
    ]
    selected = eligible[0]["timestep_seconds"] if eligible else None
    order_gate = all(
        contract["binding_gates"]["minimum_adjacent_chart_change_order"]
        <= order
        <= contract["binding_gates"]["maximum_adjacent_chart_change_order"]
        for order in orders
    )
    monotone = all(changes[index + 1] < changes[index] for index in range(len(changes) - 1))
    all_nonchart = all(not item["nonchart_failure_reasons"] for item in trials)
    passed = selected is not None and order_gate and monotone and all_nonchart
    reasons = []
    if selected is None: reasons.append("no_headroom_substep")
    if not order_gate: reasons.append("chart_change_order")
    if not monotone: reasons.append("chart_change_not_monotone")
    if not all_nonchart: reasons.append("nonchart_gate")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "new_trajectory_steps": 0,
        "trial_endpoints_propagated": False,
        "context_construction_wall_seconds": context_seconds,
        "execution_wall_seconds": time.perf_counter() - run_start,
        "trials": trials,
        "adjacent_chart_change_orders": orders,
        "chart_change_monotone": monotone,
        "all_nonchart_gates_passed": all_nonchart,
        "selected_timestep_seconds": selected,
        "selection_headroom_change": parent.HEADROOM_CHANGE,
        "failure_reasons": reasons,
    }
    arrays = {
        "seed_charts7": np.asarray(seed),
        "trial_euler_stage_charts7": np.asarray(euler_candidates),
        "trial_accepted_charts7": np.asarray(accepted_candidates),
        "componentwise_maximum_scaled_chart_changes": np.asarray(component_changes),
        "timesteps_seconds": np.asarray(parent.TIMESTEPS_SECONDS),
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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("substep diagnosis execution already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "execution_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "execution_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": bool(metrics["passed"]), "bounded_crossing_rejection_preserved": True, "new_trajectory_steps": 0, "trial_endpoints_propagated": False, "radial_shear_substep_certified": bool(metrics["passed"]), "selected_timestep_seconds": metrics["selected_timestep_seconds"], "bounded_crossing_retry_manifest_authorized": bool(metrics["passed"]), "bounded_crossing_retry_execution_authorized": False, "fixed_Q_invariant_object_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"]})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete radial substep diagnosis execution", "", f"Classification: `{metrics['classification']}`.", "", f"Selected timestep: `{metrics['selected_timestep_seconds']}` s; trial endpoints propagated: `False`.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, crossing.RADIAL_SOURCE, crossing.RADIAL_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _audit(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
