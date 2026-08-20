#!/usr/bin/env python3
"""Authenticate the frozen two-root first recenter transition."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_recenter_transition_forecast_manifest_wp10c9d6c7c3b5c4f25cp as manifest  # noqa: E402
import run_causal_inner_face36_fixed_q_primary_bounded_continuation_wp10c9d6c7c3b5c4f24e14d as continuation_tools  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_continuation_state,
    causal_five_field_fixed_q_continuation_states_equal,
    load_causal_five_field_fixed_q_continuation_state,
    save_causal_five_field_fixed_q_continuation_state,
    solve_causal_five_field_fixed_q_bdf,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cq"
MANIFEST_COMMIT = "7919fcf7dd568e48bfa94cfe18bec96466958f79"
MANIFEST_PARENT = "3855c50e1480e0b0c2136455d5ba6afb246e84a0"
MANIFEST_TREE = "2830944083f4385d8918d60d419d31f3a653e672"
PASS_CLASSIFICATION = "one_authentic_recenter_transition_validated"
FAIL_CLASSIFICATION = "recenter_transition_forecast_or_truth_failed"
PASS_AUTHORIZED_NEXT = (
    "definitions_only_authentic_center_local_field_and_overlap_manifest"
)
FAIL_AUTHORIZED_NEXT = "definitions_only_transition_diagnosis_manifest"

ARTIFACT = (
    "causal_inner_recenter_transition_validation_"
    "wp10c9d6c7c3b5c4f25cq"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_recenter_transition_validation_"
    "wp10c9d6c7c3b5c4f25cq.py"
)
THIS_TEST = (
    "tests/test_causal_inner_recenter_transition_validation_"
    "wp10c9d6c7c3b5c4f25cq.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_RECENTER_TRANSITION_"
    "VALIDATION_WP10C9D6C7C3B5C4F25CQ_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

ROOT_LABELS = ("warm_5", "warm_6")
THREAD_ENVIRONMENT = warm4_environment = (
    manifest.warm4.manifest.parent.geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
)

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


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _root_policy() -> dict:
    return {
        "order": 2,
        "timestep_seconds": manifest.TIMESTEP_SECONDS,
        "residual_tolerance": 1.0e-10,
        "constraint_tolerance": 1.0e-12,
        "ledger_tolerance": 1.0e-12,
        "storage_parity_tolerance": 1.0e-9,
        "minimum_reconstruction_factor": 1.0 - 1.0e-12,
        "maximum_schur_condition_number": 1.0e8,
        "maximum_scaled_primitive_change": 5.0e-3,
        "maximum_newton_iterations": 8,
        "maximum_line_search_iterations": 12,
        "refresh_exact_jacobian": True,
        "maximum_exact_jacobian_refreshes": 1,
        "exact_jacobian_refresh_policy": (
            "on_line_search_failure_or_iteration_reserve"
        ),
        "initial_exact_jacobian_required": False,
    }


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("recenter forecast manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("recenter forecast manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("recenter forecast manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    forecast_metrics = _read(
        manifest.CANONICAL_DIRECTORY / "forecast_metrics.json"
    )
    lock = _read(manifest.CANONICAL_DIRECTORY / "input_lock.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["prospective_truth_root_budget"]
        != manifest.AUTHENTIC_ROOT_BUDGET
        or summary["new_truth_roots"] != 0
        or not summary["retrospective_direct_forecast_passed"]
        or not forecast_metrics["passed"]
        or not all(forecast_metrics["checks"].values())
        or contract["decision"]["pass_classification"] != PASS_CLASSIFICATION
        or contract["decision"]["fail_classification"] != FAIL_CLASSIFICATION
        or contract["decision"]["pass_authorizes_only"] != PASS_AUTHORIZED_NEXT
        or contract["decision"]["fail_authorizes_only"] != FAIL_AUTHORIZED_NEXT
    ):
        raise RuntimeError("recenter transition execution contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"recenter forecast source changed: {relative}")
    decisive = {
        "direct_field_sha256": manifest.DIRECT_FIELD,
        "direct_certificate_sha256": manifest.DIRECT_CERTIFICATE,
        "warm4_checkpoint_sha256": manifest.WARM4_CHECKPOINT,
        "warm4_validation_arrays_sha256": manifest.WARM4_VALIDATION_ARRAYS,
        "warm4_validation_metrics_sha256": manifest.WARM4_VALIDATION_METRICS,
        "warm4_metrics_sha256": manifest.WARM4_METRICS,
    }
    for name, path in decisive.items():
        if _sha(path) != lock[name]:
            raise RuntimeError(f"recenter transition decisive input changed: {path}")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("recenter transition validation requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "forecast_metrics": forecast_metrics,
        "lock": lock,
        "hashes": hashes,
    }


def _source_hashes() -> dict[str, str]:
    files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
        manifest.warm4.THIS_RUNNER,
        manifest.warm4.THIS_TEST,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    )
    return {relative: _sha(ROOT / relative) for relative in files}


def _execution_identity() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "source_hashes": _source_hashes(),
        "forecast_sha256": _sha(manifest.CANONICAL_DIRECTORY / "forecast.npz"),
    }


def _paths(label: str) -> dict[str, Path]:
    return {
        "result": SCRATCH_DIRECTORY / f"result_{label}.npz",
        "metrics": SCRATCH_DIRECTORY / f"metrics_{label}.json",
        "checkpoint": SCRATCH_DIRECTORY / f"checkpoint_{label}.npz",
    }


def _load_start(data: dict, identity: dict):
    return load_causal_five_field_fixed_q_continuation_state(
        manifest.WARM4_CHECKPOINT, data["context"]
    )


def _save_checkpoint(result, start, data: dict, identity: dict, label: str):
    continuation = causal_five_field_fixed_q_continuation_state(
        result,
        data["context"],
        start.current_primitive_charts,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        elapsed_time_seconds=start.elapsed_time_seconds + manifest.TIMESTEP_SECONDS,
        completed_steps=start.completed_steps + 1,
        provenance=identity,
    )
    timings = {}
    path = _paths(label)["checkpoint"]
    save_causal_five_field_fixed_q_continuation_state(
        path, data["context"], continuation, timing_accumulator=timings
    )
    loaded = load_causal_five_field_fixed_q_continuation_state(
        path,
        data["context"],
        expected_provenance=identity,
        timing_accumulator=timings,
    )
    checkpoint = {
        "bitwise_roundtrip": causal_five_field_fixed_q_continuation_states_equal(
            continuation, loaded
        ),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        **timings,
    }
    return loaded, checkpoint


def _execute_root(data: dict, start, identity: dict, label: str):
    paths = _paths(label)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"partial or prior root output exists: {label}")
    rate, multiplier = continuation_tools._predictors(start, data["columns"])
    events = []

    def progress(payload: dict) -> None:
        plain = _plain(payload)
        events.append(plain)
        print(f"f25cq {label}: {plain}", flush=True)

    policy = _root_policy()
    began_wall = time.perf_counter()
    began_process = time.process_time()
    result = solve_causal_five_field_fixed_q_bdf(
        data["context"],
        start.current_primitive_charts,
        policy["timestep_seconds"],
        rate,
        multiplier,
        None,
        order=policy["order"],
        history=start.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=start.q3_target,
        constraint_row_scales=start.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=start.next_reaction_channel_transform,
        residual_tolerance=policy["residual_tolerance"],
        constraint_tolerance=policy["constraint_tolerance"],
        ledger_tolerance=policy["ledger_tolerance"],
        storage_parity_tolerance=policy["storage_parity_tolerance"],
        minimum_reconstruction_factor=policy["minimum_reconstruction_factor"],
        maximum_schur_condition_number=policy["maximum_schur_condition_number"],
        maximum_scaled_primitive_change=policy["maximum_scaled_primitive_change"],
        maximum_newton_iterations=policy["maximum_newton_iterations"],
        maximum_line_search_iterations=policy["maximum_line_search_iterations"],
        refresh_exact_jacobian=policy["refresh_exact_jacobian"],
        maximum_exact_jacobian_refreshes=policy[
            "maximum_exact_jacobian_refreshes"
        ],
        exact_jacobian_refresh_policy=policy["exact_jacobian_refresh_policy"],
        initial_nonlinear_solver_state=start.nonlinear_solver_state,
        initial_exact_jacobian_required=policy["initial_exact_jacobian_required"],
        solver_state_provenance=identity,
        physical_state_audit=continuation_tools.e1._state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=0.12,
        minimum_scattering_optical_depth=1.0,
        progress_callback=progress,
    )
    metrics = continuation_tools._result_metrics(
        result,
        events,
        time.perf_counter() - began_wall,
        time.process_time() - began_process,
    )
    metrics.update({"label": label, "timestep_seconds": policy["timestep_seconds"], "policy": policy})
    continuation_tools._save_result(paths["result"], result, metrics)
    checkpoint = {"bitwise_roundtrip": False}
    next_state = None
    if result.accepted:
        next_state, checkpoint = _save_checkpoint(
            result, start, data, identity, label
        )
    metrics["checkpoint"] = checkpoint
    _write_json(paths["metrics"], metrics)
    return result, metrics, next_state


def _endpoint_errors(
    predicted: np.ndarray, truth: np.ndarray, start: np.ndarray
) -> dict:
    return manifest._endpoint_errors(predicted, truth, start)


def _rate_errors(predicted: np.ndarray, truth: np.ndarray) -> dict:
    slices = {
        "full": slice(None),
        "q162": slice(0, manifest.parent.manifest.PHYSICAL_DIMENSION),
        "z280": slice(
            manifest.parent.manifest.PHYSICAL_DIMENSION,
            manifest.parent.manifest.PHYSICAL_DIMENSION
            + manifest.parent.manifest.MEMORY_DIMENSION,
        ),
        "a28": slice(-manifest.parent.manifest.DEPARTURE_DIMENSION, None),
    }
    return {
        name: float(
            np.linalg.norm(predicted[selection] - truth[selection])
            / max(float(np.linalg.norm(truth[selection])), np.finfo(float).tiny)
        )
        for name, selection in slices.items()
    }


def _root_diagnostics(
    result,
    metrics: dict,
    direct,
    forecast_coordinate: np.ndarray,
    warm4_coordinate: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    truth_coordinate, factors = direct.model.coordinate(result.primitive_charts)
    exact_delta = (
        (result.primitive_charts - direct.model.base_state) / direct.model.columns
    ).ravel()
    exact_load = float(np.max(np.abs(exact_delta)))
    decoder_load = manifest._load(direct, truth_coordinate)
    endpoint = _endpoint_errors(
        forecast_coordinate, truth_coordinate, warm4_coordinate
    )
    predicted_rate = direct.field(truth_coordinate)
    coordinate_jacobian, coordinate_metrics = (
        manifest.warm4.manifest.parent.geometry.chart_tools._coordinate_jacobian(
            result.primitive_charts, direct.model.components
        )
    )
    truth_rate = np.concatenate(
        (
            coordinate_jacobian @ result.scaled_rate_per_s,
            direct.model.memory_basis.T @ result.scaled_rate_per_s,
            direct.model.departure_basis.T @ result.scaled_rate_per_s,
        )
    )
    diagnostics = {
        "accepted": bool(result.accepted),
        "forecast_endpoint_coordinate_relative_errors": endpoint,
        "direct_field_vs_BDF_coordinate_rate_relative_errors": _rate_errors(
            predicted_rate, truth_rate
        ),
        "exact_scaled_state_load": exact_load,
        "old_decoder_load": decoder_load,
        "conservative_transition_load": max(exact_load, decoder_load),
        "coordinate_Jacobian_rank": coordinate_metrics["rank"],
        "coordinate_Jacobian_condition_number": coordinate_metrics[
            "condition_number"
        ],
        "minimum_coordinate_reconstruction_factor": float(np.min(factors)),
        "maximum_scaled_residual": metrics["maximum_scaled_residual"],
        "maximum_Q3_relative_defect": metrics["maximum_Q3_relative_defect"],
        "minimum_path_reconstruction_factor": metrics[
            "minimum_path_reconstruction_factor"
        ],
        "maximum_H_over_R": metrics["maximum_H_over_R"],
        "minimum_scattering_optical_depth": metrics[
            "minimum_scattering_optical_depth"
        ],
        "exact_Jacobian_assemblies": metrics["exact_Jacobian_assemblies"],
        "checkpoint_bitwise_roundtrip": metrics["checkpoint"][
            "bitwise_roundtrip"
        ],
    }
    arrays = {
        "truth_coordinate": truth_coordinate,
        "truth_primitive_state": result.primitive_charts,
        "truth_scaled_delta": exact_delta,
        "truth_BDF_coordinate_rate": truth_rate,
        "predicted_coordinate_rate_at_truth": predicted_rate,
    }
    return diagnostics, arrays


def _translation_defect(points: np.ndarray, center: np.ndarray) -> float:
    translated = np.asarray(points) - np.asarray(center)
    restored = translated + np.asarray(center)
    return float(np.max(np.abs(restored - np.asarray(points))))


def _execution_checks(records: list[dict], translation_defect: float, gates: dict) -> dict:
    if len(records) < 2 or not all(record.get("accepted", False) for record in records):
        return {
            "accepted_roots": False,
            "root_contracts": False,
            "checkpoint_roundtrips": False,
            "forecast_endpoints": False,
            "trigger_bracket": False,
            "hard_limit": False,
            "translation_roundtrip": False,
        }
    root_contracts = all(
        record["maximum_scaled_residual"] <= gates["maximum_scaled_residual"]
        and record["maximum_Q3_relative_defect"]
        <= gates["maximum_Q3_relative_defect"]
        and record["minimum_path_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"]
        and record["maximum_H_over_R"] <= gates["maximum_H_over_R"]
        and record["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"]
        and record["exact_Jacobian_assemblies"]
        <= gates["maximum_exact_Jacobian_assemblies_per_root"]
        for record in records
    )
    forecast_endpoints = all(
        record["forecast_endpoint_coordinate_relative_errors"]["full"]
        <= gates["forecast_endpoint_full_coordinate_relative_error_max"]
        and record["forecast_endpoint_coordinate_relative_errors"]["q162"]
        <= gates["forecast_endpoint_q162_relative_error_max"]
        and record["forecast_endpoint_coordinate_relative_errors"]["z280"]
        <= gates["forecast_endpoint_z280_relative_error_max"]
        and record["forecast_endpoint_coordinate_relative_errors"]["a28"]
        <= gates["forecast_endpoint_a28_relative_error_max"]
        for record in records
    )
    return {
        "accepted_roots": len(records) == gates["accepted_roots_equal"]
        and all(record["accepted"] for record in records),
        "root_contracts": root_contracts,
        "checkpoint_roundtrips": all(
            record["checkpoint_bitwise_roundtrip"] for record in records
        ),
        "forecast_endpoints": forecast_endpoints,
        "trigger_bracket": records[0]["exact_scaled_state_load"]
        < gates["warm5_exact_load_below_trigger"]
        and records[1]["exact_scaled_state_load"]
        >= gates["warm6_exact_load_at_least_trigger"],
        "hard_limit": records[1]["exact_scaled_state_load"]
        < gates["warm6_exact_load_below_hard_limit"],
        "translation_roundtrip": translation_defect
        <= gates["translation_roundtrip_infinity_defect_max"],
    }


def _canonicalize(
    frozen: dict,
    records: list[dict],
    arrays: list[dict[str, np.ndarray]],
    checks: dict,
    translation_defect: float,
) -> dict:
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = PASS_AUTHORIZED_NEXT if passed else FAIL_AUTHORIZED_NEXT
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    for label in ROOT_LABELS:
        for path in _paths(label).values():
            if path.is_file():
                shutil.copy2(path, CANONICAL_DIRECTORY / path.name)
    _write_json(
        CANONICAL_DIRECTORY / "validation_metrics.json",
        {
            "checks": checks,
            "passed": passed,
            "root_records": records,
            "translation_roundtrip_infinity_defect": translation_defect,
            "new_truth_roots_attempted": len(records),
            "accepted_truth_roots": sum(record["accepted"] for record in records),
        },
    )
    combined = {}
    for index, payload in enumerate(arrays):
        for name, value in payload.items():
            combined[f"{ROOT_LABELS[index]}_{name}"] = value
    if len(arrays) == 2:
        combined["authentic_center_primitive_state"] = arrays[1][
            "truth_primitive_state"
        ]
        combined["authentic_center_old_coordinate"] = arrays[1][
            "truth_coordinate"
        ]
        combined["authentic_center_scaled_delta"] = arrays[1][
            "truth_scaled_delta"
        ]
    _write_npz(CANONICAL_DIRECTORY / "transition_arrays.npz", **combined)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "attempted_truth_roots": len(records),
        "accepted_truth_roots": sum(record["accepted"] for record in records),
        "first_trigger_root_index": 2 if passed else None,
        "authentic_center_established": passed,
        "predicted_center_used_as_center": False,
        "new_continuous_rate_calls": 0,
        "new_generator_assemblies": 0,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "forecast_sha256": _sha(manifest.CANONICAL_DIRECTORY / "forecast.npz"),
            "warm4_checkpoint_sha256": _sha(manifest.WARM4_CHECKPOINT),
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": _source_hashes(),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT
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
    loads = [record.get("exact_scaled_state_load") for record in records]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Recenter-transition validation WP10c9d6c7c3b5c4f25cq",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Attempted `{len(records)}` authentic BDF2 roots and accepted `{sum(record['accepted'] for record in records)}`.",
                "",
                f"Authentic exact scaled-state loads: `{loads}`. The frozen forecast placed the first trigger in root two.",
                "",
                f"Coordinate translation roundtrip defect: `{translation_defect:.6e}`.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No physical microburst, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


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
    if CANONICAL_DIRECTORY.exists() or SCRATCH_DIRECTORY.exists():
        raise RuntimeError("recenter transition validation already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    data = continuation_tools.e1._state_data("primary_20ms")
    identity = _execution_identity()
    start = _load_start(data, identity)
    direct = manifest.parent.manifest.DirectCoordinateField(
        _load_npz(manifest.DIRECT_FIELD)
    )
    forecast = _load_npz(manifest.CANONICAL_DIRECTORY / "forecast.npz")
    warm4_coordinate = np.asarray(forecast["warm4_truth_coordinate"], dtype=float)
    predicted = np.asarray(
        forecast["prospective_refined_coordinates"], dtype=float
    )
    records = []
    arrays = []
    for index, label in enumerate(ROOT_LABELS):
        result, metrics, next_state = _execute_root(data, start, identity, label)
        if not result.accepted:
            records.append(
                {
                    "accepted": False,
                    "label": label,
                    "maximum_scaled_residual": metrics["maximum_scaled_residual"],
                }
            )
            break
        diagnostics, payload = _root_diagnostics(
            result, metrics, direct, predicted[index], warm4_coordinate
        )
        diagnostics["label"] = label
        records.append(diagnostics)
        arrays.append(payload)
        start = next_state
        if index == 0 and diagnostics["conservative_transition_load"] >= manifest.RECENTER_TRIGGER_LOAD:
            break
    translation_defect = math.inf
    if len(arrays) == 2:
        points = np.asarray(
            (
                warm4_coordinate,
                arrays[0]["truth_coordinate"],
                arrays[1]["truth_coordinate"],
            )
        )
        translation_defect = _translation_defect(
            points, arrays[1]["truth_coordinate"]
        )
    gates = frozen["contract"]["binding_execution_gates"]
    checks = _execution_checks(records, translation_defect, gates)
    summary = _canonicalize(
        frozen, records, arrays, checks, translation_defect
    )
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
