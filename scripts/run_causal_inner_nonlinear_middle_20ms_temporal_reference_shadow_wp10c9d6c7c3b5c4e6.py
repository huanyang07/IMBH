#!/usr/bin/env python3
"""Run the short coarse/middle response-specific temporal-reference shadow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_20ms_temporal_reference_manifest_wp10c9d6c7c3b5c4e5 as c4e5  # noqa: E402
import run_causal_inner_nonlinear_coarse_middle_20ms_checkpoint_analysis_wp10c9d6c7c3b5c4e4 as c4e4  # noqa: E402
import run_causal_inner_nonlinear_optimized_middle_20ms_completion_wp10c9d6c7c3b5c4e3 as c4e3  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_completion_wp10c9d6c7c3b5c4c1 as c4c1  # noqa: E402
import run_causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2 as c4b2  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
    load_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e6"
ANALYZED_BASE_COMMIT = c4e5.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e5.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e5.ANALYZED_BASE_TREE

ARTIFACT = (
    "causal_inner_nonlinear_middle_20ms_temporal_reference_shadow_"
    "wp10c9d6c7c3b5c4e6"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_20ms_temporal_reference_"
    "shadow_wp10c9d6c7c3b5c4e6.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_20ms_temporal_reference_"
    "shadow_wp10c9d6c7c3b5c4e6.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_20MS_TEMPORAL_"
    "REFERENCE_SHADOW_WP10C9D6C7C3B5C4E6_2026-08-11.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "progress.json"

STAGE_ORDER = (
    "coarse_base_main",
    "coarse_perturbed_main",
    "coarse_base_strict",
    "coarse_perturbed_strict",
    "middle_base_strict",
    "middle_anchor_strict",
)
COARSE_RESTART_DIRECTORY = c4c1.PROGRESS_DIRECTORY
MIDDLE_BASE_PATH = c4e3.BASE_PATH
MIDDLE_ANCHOR_PATH = c4e3.ANCHOR_PATH
MIDDLE_EXTRACTION_PATH = c4e3.EXTRACTION_PATH


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    dependencies = (
        THIS_RUNNER,
        THIS_TEST,
        c4e5.THIS_RUNNER,
        c4e5.THIS_TEST,
        c4e4.THIS_RUNNER,
        c4e3.THIS_RUNNER,
        c4c1.THIS_RUNNER,
        c4b2.THIS_RUNNER,
        c4e3.h2b1.CONTROLLER_RELATIVE,
        c4e3.h2b1.MODULE_RELATIVE,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    )
    return {
        path: _sha256(ROOT / path)
        for path in dependencies
        if (ROOT / path).exists()
    }


def _stage_path(stage: str) -> Path:
    return CHECKPOINT_DIRECTORY / f"{stage}.npz"


def _stage_hashes() -> dict[str, str]:
    return {
        stage: _sha256(_stage_path(stage))
        for stage in STAGE_ORDER
        if _stage_path(stage).exists()
    }


def _progress() -> dict:
    identity = _source_identity()
    manifest_hash = _sha256(c4e5.MANIFEST_PATH)
    if PROGRESS_PATH.exists():
        payload = _read_json(PROGRESS_PATH)
        if payload.get("source_identity") != identity:
            raise RuntimeError("c4e6 checkpoint source identity changed")
        if payload.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("c4e6 temporal manifest changed")
        if payload.get("stage_hashes") != _stage_hashes():
            raise RuntimeError("c4e6 stage payload hash changed")
        return payload
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_identity": identity,
        "manifest_sha256": manifest_hash,
        "completed_stages": [],
        "stage_reports": {},
        "stage_hashes": {},
    }


def _save_stage(stage: str, arrays: dict, report: dict, progress: dict) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_stage_path(stage), **arrays)
    if stage not in progress["completed_stages"]:
        progress["completed_stages"].append(stage)
    progress["stage_reports"][stage] = report
    progress["stage_hashes"] = _stage_hashes()
    _write_json(PROGRESS_PATH, progress)


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(c4e5.SUMMARY_PATH)
    manifest = _read_json(c4e5.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["definitions_only"]
        or not parent["temporal_reference_shadow_authorized"]
        or parent["fine_twenty_ms_manifest_authorized"]
        or parent["fine_twenty_ms_propagation_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
        or parent["authorized_next"]
        != f"{WORK_PACKAGE}_middle_20ms_response_temporal_reference_shadow"
    ):
        raise RuntimeError("c4e6 parent authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e6 analyzed identity changed")
    return parent, manifest


def _indices(times: np.ndarray, target_microseconds) -> np.ndarray:
    source = np.rint(np.asarray(times) * 1.0e6).astype(int)
    result = []
    for target in target_microseconds:
        matches = np.flatnonzero(source == int(target))
        if matches.size != 1:
            raise RuntimeError(f"c4e6 target {target} us is not unique")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=int)


def _coarse_configuration():
    return c4b2.c3b1a._configurations()[c4b2.c2.LAYOUT]


def _coarse_tangent(configuration):
    return causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )


def _middle_configuration():
    return c4e3.h2b1._configuration()


def _middle_tangent(configuration):
    return c4e3.h2b1._build_frozen_tangent(configuration)[0]


def _coarse_seed(stage: str, context):
    branch = "base" if "base" in stage else "perturbed"
    path = (
        COARSE_RESTART_DIRECTORY
        / f"{branch}_main"
        / f"restart_{c4e5.START_MICROSECONDS}us.npz"
    )
    restart = load_causal_five_field_monolithic_bdf_restart(path, context)
    if (
        abs(restart.elapsed_time_seconds - c4e5.START_MICROSECONDS * 1.0e-6)
        > 1.0e-15
        or restart.next_order != 2
    ):
        raise RuntimeError(f"c4e6 {stage} coarse restart changed")
    return restart.primitive_charts, restart.history


def _middle_seed(stage: str):
    base = _load_npz(MIDDLE_BASE_PATH)
    anchor = _load_npz(MIDDLE_ANCHOR_PATH)
    index = int(_indices(base["accepted_times"], (c4e5.START_MICROSECONDS,))[0])
    if "base" in stage:
        state = base["accepted_states"][index]
        primitive = base["accepted_primitive_histories"][index]
        mapped = base["accepted_mapped_histories"][index]
        height = base["accepted_height_histories"][index]
        previous = base["accepted_previous_timesteps"][index]
    else:
        state = anchor["anchor_states"][index]
        primitive = anchor["anchor_primitive_histories"][index]
        mapped = anchor["anchor_mapped_histories"][index]
        height = anchor["anchor_height_histories"][index]
        previous = anchor["anchor_previous_timesteps"][index]
    history = c4e3.h2b1.h2a2._history(primitive, mapped, height, previous)
    return np.asarray(state, dtype=float), history


def _extraction_history(layout: str, context, states: np.ndarray):
    if layout == "coarse":
        values, audits = c4b2._exterior_history(context, states)
        unified = np.column_stack(
            (audits[:, 6], audits[:, 0], audits[:, 1], audits[:, 2], audits[:, 4])
        )
        return np.asarray(values), np.asarray(unified)
    values = []
    audits = []
    for state in states:
        value, identity, local = c4e3._extraction_value(context, state)
        values.append(value)
        audits.append((identity, local[0], local[1], local[2], local[3]))
    return np.asarray(values), np.asarray(audits)


def _segment_passed(report: dict, audits: np.ndarray, manifest: dict, strict: bool):
    gates = manifest["method_gates"]
    local_gate = gates[
        "strict_local_error_maximum" if strict else "main_local_error_maximum"
    ]
    return bool(
        report["method_passed"]
        and report["maximum_local_error_estimate"] <= local_gate
        and report["maximum_scaled_residual"]
        <= gates["maximum_scaled_nonlinear_residual"]
        and report["maximum_discrete_ledger_defect"]
        <= gates["maximum_discrete_ledger_defect"]
        and report["maximum_mapped_endpoint_path_closure_defect"]
        <= gates["maximum_mapped_endpoint_path_closure_defect"]
        and report["minimum_path_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"]
        and report["maximum_incoming_excision_characteristics"]
        <= gates["maximum_incoming_excision_characteristics"]
        and float(np.max(audits[:, 0]))
        <= gates["maximum_extraction_identity_defect"]
        and float(np.max(audits[:, 1]))
        <= gates["maximum_shared_conservative_face_defect"]
        and float(np.max(audits[:, 2])) <= 1.0e-12
        and float(np.max(audits[:, 3]))
        <= gates["maximum_source_double_count_defect"]
        and int(np.max(audits[:, 4]))
        <= gates["maximum_incoming_excision_characteristics"]
    )


def _run_stage(
    stage: str,
    configuration: dict,
    tangent,
    state: np.ndarray,
    history,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    coupling_face: int,
    controller_contract: dict,
    manifest: dict,
) -> tuple[dict, dict]:
    strict = stage.endswith("strict")
    targets_us = (
        np.asarray(c4e5.STRICT_TARGET_MICROSECONDS, dtype=int)
        if strict
        else np.asarray((c4e5.START_MICROSECONDS, c4e5.STOP_MICROSECONDS), dtype=int)
    )
    output_times = targets_us.astype(float) * 1.0e-6
    candidate = (
        c4e5.STRICT_TIMESTEP_SECONDS if strict else c4e5.MAIN_TIMESTEP_SECONDS
    )
    began = time.perf_counter()
    segment = c4b2.c2._controller_segment(
        configuration,
        tangent,
        state,
        history,
        c4e5.START_MICROSECONDS * 1.0e-6,
        candidate,
        field_scales,
        export_scales,
        coupling_face,
        controller_contract,
        output_times=output_times,
        stop_time=c4e5.STOP_MICROSECONDS * 1.0e-6,
        include_initial_output=True,
        log_prefix=f"c4e6-{stage}",
    )
    wall = time.perf_counter() - began
    report = c4b2.c3b._segment_report(segment, controller_contract)
    layout = "middle" if stage.startswith("middle") else "coarse"
    extraction, audits = _extraction_history(
        layout, configuration["context"], segment["output_states"]
    )
    readiness = c4b2.c3b1a._state_audit(
        configuration["context"], segment["output_states"][-1]
    )
    passed = bool(
        _segment_passed(report, audits, manifest, strict)
        and readiness["minimum_scattering_optical_depth"] >= 1.0
        and readiness["maximum_h_over_r"] <= 0.12
        and readiness["minimum_reconstruction_factor"] >= 1.0
    )
    report.update(
        {
            "passed": passed,
            "wall_seconds": wall,
            "maximum_extraction_identity_defect": float(np.max(audits[:, 0])),
            "maximum_shared_conservative_face_defect": float(
                np.max(audits[:, 1])
            ),
            "maximum_local_block_ledger_defect": float(np.max(audits[:, 2])),
            "maximum_source_double_count_defect": float(np.max(audits[:, 3])),
            "maximum_extraction_incoming_characteristics": int(
                np.max(audits[:, 4])
            ),
            "final_state_readiness": readiness,
        }
    )
    if not passed:
        raise RuntimeError(f"c4e6 {stage} failed its method/readiness gates")
    arrays = {
        "output_times": np.asarray(segment["output_times"]),
        "output_states": np.asarray(segment["output_states"]),
        "output_extraction_partition": extraction,
        "output_extraction_audits": audits,
        "accepted_times": np.asarray(segment["accepted_times"]),
        "accepted_timesteps": np.asarray(segment["accepted_timesteps"]),
        "local_error_estimates": np.asarray(segment["local_error_estimates"]),
        "retries": np.asarray(segment["retries"]),
        "accepted_step_wall_seconds": np.asarray(
            segment["accepted_step_wall_seconds"]
        ),
    }
    return arrays, report


def _scaled_difference(left, right, scales) -> float:
    shape = (1,) * (np.asarray(left).ndim - 1) + (len(scales),)
    return float(
        np.max(np.abs(np.asarray(left) - np.asarray(right)) / np.reshape(scales, shape))
    )


def _integral(values, times):
    return np.trapezoid(np.asarray(values), np.asarray(times), axis=0)


def _window_metrics(
    main_times,
    main_state_response,
    main_extraction_response,
    strict_times,
    strict_state_response,
    strict_extraction_response,
    field_scales,
    extraction_scales,
):
    endpoint_indices = _indices(strict_times, np.rint(main_times * 1.0e6).astype(int))
    state = _scaled_difference(
        main_state_response,
        strict_state_response[endpoint_indices],
        field_scales,
    )
    instantaneous = _scaled_difference(
        main_extraction_response,
        strict_extraction_response[endpoint_indices],
        extraction_scales,
    )
    main_integral = _integral(main_extraction_response, main_times)
    strict_integral = _integral(strict_extraction_response, strict_times)
    cumulative = _scaled_difference(
        main_integral,
        strict_integral,
        extraction_scales * c4e5.FULL_INTERVAL_SECONDS,
    )
    window = float(main_times[-1] - main_times[0])
    mean = _scaled_difference(
        main_integral / window,
        strict_integral / window,
        extraction_scales,
    )
    return {
        "state": state,
        "instantaneous_extraction": instantaneous,
        "cumulative_extraction": cumulative,
        "window_mean_extraction": mean,
    }


def _existing_coarse_metrics(payload, field_scales, extraction_scales):
    main_times = payload["base_main__output_times"]
    strict_times = payload["base_strict__output_times"]
    indices = c4e4._indices(main_times, strict_times)
    main_state = (
        payload["perturbed_main__output_states"][indices]
        - payload["base_main__output_states"][indices]
    )
    strict_state = (
        payload["perturbed_strict__output_states"]
        - payload["base_strict__output_states"]
    )
    main_extraction = (
        payload["perturbed_main__output_extraction_partition"][indices]
        - payload["base_main__output_extraction_partition"][indices]
    )
    strict_extraction = (
        payload["perturbed_strict__output_extraction_partition"]
        - payload["base_strict__output_extraction_partition"]
    )
    return _window_metrics(
        strict_times,
        main_state,
        main_extraction,
        strict_times,
        strict_state,
        strict_extraction,
        field_scales,
        extraction_scales,
    )


def _new_stage_metrics(stages, layout, field_scales, extraction_scales):
    if layout == "coarse":
        main_base = stages["coarse_base_main"]
        main_anchor = stages["coarse_perturbed_main"]
        strict_base = stages["coarse_base_strict"]
        strict_anchor = stages["coarse_perturbed_strict"]
        main_times = main_base["output_times"]
        main_state = main_anchor["output_states"] - main_base["output_states"]
        main_extraction = (
            main_anchor["output_extraction_partition"]
            - main_base["output_extraction_partition"]
        )
    else:
        base = _load_npz(MIDDLE_BASE_PATH)
        anchor = _load_npz(MIDDLE_ANCHOR_PATH)
        extraction = _load_npz(MIDDLE_EXTRACTION_PATH)
        indices = _indices(
            base["accepted_times"],
            (c4e5.START_MICROSECONDS, c4e5.STOP_MICROSECONDS),
        )
        extraction_indices = _indices(
            extraction["accepted_times"],
            (c4e5.START_MICROSECONDS, c4e5.STOP_MICROSECONDS),
        )
        main_times = base["accepted_times"][indices]
        main_state = anchor["anchor_states"][indices] - base["accepted_states"][indices]
        main_extraction = (
            extraction["anchor_values"][extraction_indices]
            - extraction["base_values"][extraction_indices]
        )
        strict_base = stages["middle_base_strict"]
        strict_anchor = stages["middle_anchor_strict"]
    strict_times = strict_base["output_times"]
    strict_state = strict_anchor["output_states"] - strict_base["output_states"]
    strict_extraction = (
        strict_anchor["output_extraction_partition"]
        - strict_base["output_extraction_partition"]
    )
    if layout == "middle":
        _parent, layouts, _contexts = c4e4.b2b._layouts_and_contexts(
            c4e4.b2b._input_arrays()
        )
        middle_layout = layouts[c4e4.MIDDLE_LAYOUT]
        main_state = np.asarray(
            [
                restrict_causal_embedded_patch_cell_averages(value, middle_layout)
                for value in main_state
            ]
        )
        strict_state = np.asarray(
            [
                restrict_causal_embedded_patch_cell_averages(value, middle_layout)
                for value in strict_state
            ]
        )
    return _window_metrics(
        main_times,
        main_state,
        main_extraction,
        strict_times,
        strict_state,
        strict_extraction,
        field_scales,
        extraction_scales,
    )


def _analyze(stages: dict[str, dict], manifest: dict):
    checkpoint = _read_json(c4e4.SUMMARY_PATH)["analysis"]
    c4e3_arrays = _load_npz(c4e3.DECISIVE_ARRAYS)
    c4c1_arrays = _load_npz(c4c1.DECISIVE_ARRAYS)
    c4b2_arrays = _load_npz(c4b2.DECISIVE_ARRAYS)
    field_scales = np.asarray(c4e3_arrays["tangent__field_scales"], dtype=float)
    extraction_scales = np.asarray(
        c4c1_arrays["extraction_partition_scales"], dtype=float
    )
    coarse_ten = _existing_coarse_metrics(
        c4b2_arrays, field_scales, extraction_scales
    )
    coarse_twenty = _existing_coarse_metrics(
        c4c1_arrays, field_scales, extraction_scales
    )
    coarse_interior = _new_stage_metrics(
        stages, "coarse", field_scales, extraction_scales
    )
    middle_interior = _new_stage_metrics(
        stages, "middle", field_scales, extraction_scales
    )
    spatial = {
        "state": checkpoint["state"]["maximum_normalized_difference"],
        "instantaneous_extraction": checkpoint["instantaneous_extraction"][
            "maximum_normalized_difference"
        ],
        "cumulative_extraction": checkpoint["cumulative_extraction"][
            "maximum_normalized_difference"
        ],
        "window_mean_extraction": checkpoint["window_mean_extraction"][
            "maximum_normalized_difference"
        ],
    }
    observables = tuple(spatial)
    result = {}
    for name in observables:
        coarse = max(
            coarse_ten[name], coarse_twenty[name], coarse_interior[name]
        )
        middle = middle_interior[name]
        uncertainty = (
            manifest["temporal_uncertainty"]["safety_factor"] * (coarse + middle)
        )
        ratio = uncertainty / max(spatial[name], np.finfo(float).tiny)
        result[name] = {
            "coarse_10ms_response_discrepancy": coarse_ten[name],
            "coarse_20ms_response_discrepancy": coarse_twenty[name],
            "coarse_16ms_response_discrepancy": coarse_interior[name],
            "coarse_maximum_response_discrepancy": coarse,
            "middle_16ms_response_discrepancy": middle,
            "safety_factor": manifest["temporal_uncertainty"]["safety_factor"],
            "combined_temporal_uncertainty": uncertainty,
            "spatial_difference": spatial[name],
            "temporal_to_spatial_fraction": ratio,
            "passed": bool(
                ratio
                <= manifest["temporal_uncertainty"][
                    "maximum_fraction_of_spatial_difference"
                ]
            ),
        }
    return {
        "passed": all(result[name]["passed"] for name in observables),
        "observables": result,
        "coarse_10ms": coarse_ten,
        "coarse_20ms": coarse_twenty,
        "coarse_16ms": coarse_interior,
        "middle_16ms": middle_interior,
    }, field_scales, extraction_scales


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["analysis_completed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
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
    catalog = _read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _finalize(parent, manifest, progress, stages, started):
    analysis, field_scales, extraction_scales = _analyze(stages, manifest)
    hardened = bool(analysis["passed"])
    classification = (
        "middle_20ms_response_temporal_reference_hardened_cost_bounded_fine_manifest_authorized"
        if hardened
        else "middle_20ms_response_temporal_reference_still_insufficient_fine_blocked"
    )
    authorized_next = (
        "cost_bounded_fine_20ms_spatial_certificate_manifest_only"
        if hardened
        else "one_additional_shorter_timestep_temporal_shadow_only"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": True,
        "analysis_completed": True,
        "temporal_reference_hardened": hardened,
        "analysis": analysis,
        "stage_reports": progress["stage_reports"],
        "elapsed_seconds": time.perf_counter() - started,
        "fine_twenty_ms_manifest_authorized": hardened,
        "fine_twenty_ms_propagation_authorized": False,
        "full_fine_generic_anchor_required": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "raw_inner_face_rejection_preserved": True,
    }
    combined = {
        "field_scales": field_scales,
        "extraction_partition_scales": extraction_scales,
        "spatial_differences": np.asarray(
            [item["spatial_difference"] for item in analysis["observables"].values()]
        ),
        "temporal_uncertainties": np.asarray(
            [
                item["combined_temporal_uncertainty"]
                for item in analysis["observables"].values()
            ]
        ),
        "temporal_to_spatial_fractions": np.asarray(
            [
                item["temporal_to_spatial_fraction"]
                for item in analysis["observables"].values()
            ]
        ),
    }
    for stage, arrays in stages.items():
        combined.update({f"{stage}__{key}": value for key, value in arrays.items()})
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "start_microseconds": c4e5.START_MICROSECONDS,
            "stop_microseconds": c4e5.STOP_MICROSECONDS,
            "strict_target_microseconds": c4e5.STRICT_TARGET_MICROSECONDS,
            "manifest_sha256": _sha256(c4e5.MANIFEST_PATH),
        },
    )
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "temporal_manifest": _sha256(c4e5.MANIFEST_PATH),
                "c4e4_summary": _sha256(c4e4.SUMMARY_PATH),
                "coarse_10ms_arrays": _sha256(c4b2.DECISIVE_ARRAYS),
                "coarse_20ms_arrays": _sha256(c4c1.DECISIVE_ARRAYS),
                "middle_20ms_arrays": _sha256(c4e3.DECISIVE_ARRAYS),
                "middle_base_checkpoint": _sha256(MIDDLE_BASE_PATH),
                "middle_anchor_checkpoint": _sha256(MIDDLE_ANCHOR_PATH),
                "middle_extraction_checkpoint": _sha256(MIDDLE_EXTRACTION_PATH),
            },
            "implementation_source_hashes": _source_identity(),
            "stage_hashes": _stage_hashes(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Middle 20 ms temporal-reference shadow WP10c9d6c7c3b5c4e6",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        "This package runs only the frozen response-specific `16.0 -> 16.4 ms` interior shadows. It changes no operator and executes no fine propagation.",
        "",
        "## Temporal-to-spatial ratios",
        "",
    ]
    for name, item in analysis["observables"].items():
        lines.append(
            f"- `{name}`: `{item['temporal_to_spatial_fraction']:.6e}` "
            f"(gate `<= {manifest['temporal_uncertainty']['maximum_fraction_of_spatial_difference']:.2f}`)."
        )
    lines.extend(
        (
            "",
            f"Temporal reference hardened: `{hardened}`.",
            "",
            f"Authorized next: `{authorized_next}`.",
            "",
            "A pass authorizes only a cost-bounded fine definitions manifest. Fine propagation, 50 ms evolution, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGE_ORDER, default=STAGE_ORDER[-1])
    arguments = parser.parse_args(argv)
    started = time.perf_counter()
    parent, manifest = _validate_parent()
    progress = _progress()
    middle_arrays = _load_npz(c4e3.DECISIVE_ARRAYS)
    coarse_arrays = _load_npz(c4c1.DECISIVE_ARRAYS)
    field_scales = np.asarray(middle_arrays["tangent__field_scales"], dtype=float)
    coarse_export_scales = np.asarray(coarse_arrays["field_scales"], dtype=float)
    del coarse_export_scales
    coarse_pilot = _load_npz(c4b2.c4b.DECISIVE_ARRAYS)
    coarse_raw_scales = np.asarray(coarse_pilot["export_scales"], dtype=float)
    middle_raw_scales = np.asarray(
        middle_arrays["tangent__export_scales"], dtype=float
    )
    coarse_manifest = _read_json(c4c1.c4c.MANIFEST_PATH)
    coarse_configuration = None
    coarse_tangent = None
    middle_configuration = None
    middle_tangent = None
    stages = {}
    for stage in STAGE_ORDER:
        if stage in progress["completed_stages"]:
            stages[stage] = _load_npz(_stage_path(stage))
        else:
            if stage.startswith("coarse"):
                if coarse_configuration is None:
                    print("c4e6: build coarse configuration and tangent", flush=True)
                    coarse_configuration = _coarse_configuration()
                    coarse_tangent = _coarse_tangent(coarse_configuration)
                state, history = _coarse_seed(stage, coarse_configuration["context"])
                strict = stage.endswith("strict")
                arrays, report = _run_stage(
                    stage,
                    coarse_configuration,
                    coarse_tangent,
                    state,
                    history,
                    field_scales,
                    coarse_raw_scales,
                    c4b2.c2.COUPLING_FACE,
                    coarse_manifest[
                        "strict_controller" if strict else "main_controller"
                    ],
                    manifest,
                )
            else:
                if middle_configuration is None:
                    print("c4e6: build middle configuration and tangent", flush=True)
                    middle_configuration = _middle_configuration()
                    middle_tangent = _middle_tangent(middle_configuration)
                state, history = _middle_seed(stage)
                arrays, report = _run_stage(
                    stage,
                    middle_configuration,
                    middle_tangent,
                    state,
                    history,
                    field_scales,
                    middle_raw_scales,
                    c4e3.COUPLING_FACE,
                    coarse_manifest["strict_controller"],
                    manifest,
                )
            _save_stage(stage, arrays, report, progress)
            stages[stage] = arrays
            print(
                f"c4e6: completed {stage} in {report['wall_seconds']:.1f}s",
                flush=True,
            )
        if stage == arguments.through:
            if stage != STAGE_ORDER[-1]:
                print(
                    json.dumps(
                        {
                            "work_package": WORK_PACKAGE,
                            "completed_through": stage,
                            "checkpoint_directory": str(CHECKPOINT_DIRECTORY),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            break
    for stage in STAGE_ORDER:
        if stage not in stages:
            stages[stage] = _load_npz(_stage_path(stage))
    return _finalize(parent, manifest, progress, stages, started)


if __name__ == "__main__":
    raise SystemExit(main())
