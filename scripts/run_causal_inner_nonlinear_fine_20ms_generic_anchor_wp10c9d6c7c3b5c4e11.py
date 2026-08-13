#!/usr/bin/env python3
"""Run the targeted fine-layout generic nonlinear anchor through 20 ms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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

import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as fine5  # noqa: E402
import run_causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_wp10c9d6c7c3b5c4e8 as c4e8  # noqa: E402
import run_causal_inner_nonlinear_fine_20ms_generic_anchor_manifest_wp10c9d6c7c3b5c4e10 as c4e10  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e11"
ANALYZED_BASE_COMMIT = c4e10.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e10.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e10.ANALYZED_BASE_TREE

FINE_LAYOUT = c4e10.FINE_LAYOUT
GENERIC_INDEX = c4e8.GENERIC_INDEX
GENERIC_PROFILE = c4e10.GENERIC_PROFILE
COUPLING_FACE = c4e10.COUPLING_FACE
EXTRACTION_FACE = c4e10.EXTRACTION_FACE
TARGET_MICROSECONDS = c4e10.TARGET_MICROSECONDS
TEMPORAL_AUDIT_TARGET_MICROSECONDS = c4e10.TEMPORAL_AUDIT_TARGET_MICROSECONDS
MAXIMUM_TIMESTEP_SECONDS = c4e10.MAXIMUM_TIMESTEP_SECONDS

ARTIFACT = (
    "causal_inner_nonlinear_fine_20ms_generic_anchor_"
    "wp10c9d6c7c3b5c4e11"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_fine_20ms_generic_anchor_"
    "wp10c9d6c7c3b5c4e11.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_fine_20ms_generic_anchor_"
    "wp10c9d6c7c3b5c4e11.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_FINE_20MS_GENERIC_"
    "ANCHOR_WP10C9D6C7C3B5C4E11_2026-08-12.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
LATEST_PATH = CHECKPOINT_DIRECTORY / "LATEST"


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
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    h2b1 = c4e8.h2b1
    dependencies = (
        THIS_RUNNER,
        THIS_TEST,
        c4e10.THIS_RUNNER,
        c4e8.THIS_RUNNER,
        c4e8.c4e3.THIS_RUNNER,
        h2b1.CONTROLLER_RELATIVE,
        h2b1.MODULE_RELATIVE,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
        "scripts/run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_audit_wp10c9d6c7c3b5c3h2j1.py",
    )
    return {
        path: _sha256(ROOT / path) for path in dependencies if (ROOT / path).exists()
    }


def _input_identity() -> dict[str, str]:
    return {
        "anchor_manifest": _sha256(c4e10.MANIFEST_PATH),
        "fine_base_tangent_arrays": _sha256(c4e8.DECISIVE_ARRAYS),
        "fine_base_tangent_summary": _sha256(c4e8.SUMMARY_PATH),
        "fine_5ms_arrays": _sha256(fine5.DECISIVE_ARRAYS),
        "fine_5ms_summary": _sha256(fine5.SUMMARY_PATH),
    }


def _validate_parent() -> None:
    manifest = _read_json(c4e10.SUMMARY_PATH)
    if (
        not manifest["passed"]
        or not manifest["fine_generic_anchor_propagation_authorized"]
        or manifest["fine_base_rerun_authorized"]
        or manifest["other_nonlinear_profiles_authorized"]
        or manifest["fine_twenty_ms_spatial_certificate_issued"]
        or manifest["fifty_ms_propagation_authorized"]
        or manifest["physical_failure_detected"]
    ):
        raise RuntimeError("c4e11 manifest authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e11 analyzed identity changed")


def _parent_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    parent = _load_npz(c4e8.DECISIVE_ARRAYS)
    fine_start = _load_npz(fine5.DECISIVE_ARRAYS)
    return parent, fine_start


def _base(parent: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    marker = "base__"
    return {
        key.removeprefix(marker): value
        for key, value in parent.items()
        if key.startswith(marker)
    }


def _initial_arrays(
    parent: dict[str, np.ndarray], fine_start: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    base = _base(parent)
    state = np.asarray(fine_start["anchor__anchor_states"][-1], dtype=float)
    if not np.array_equal(
        np.asarray(fine_start["base__accepted_states"][-1]),
        np.asarray(base["accepted_states"][0]),
    ):
        raise RuntimeError("c4e11 fine base 5 ms restart changed")
    tangent_state = np.asarray(
        parent["tangent__state_directions"][:, GENERIC_INDEX], dtype=float
    )
    predicted = np.asarray(base["accepted_states"][0]) + tangent_state[0]
    if np.max(np.abs(predicted - state)) > 1.0e-6:
        raise RuntimeError("c4e11 fine generic 5 ms anchor no longer calibrated")
    return {
        "anchor_states": state[None, ...],
        "anchor_primitive_histories": fine_start[
            "anchor__anchor_primitive_histories"
        ][-1:],
        "anchor_mapped_histories": fine_start["anchor__anchor_mapped_histories"][-1:],
        "anchor_height_histories": fine_start["anchor__anchor_height_histories"][-1:],
        "anchor_previous_timesteps": fine_start[
            "anchor__anchor_previous_timesteps"
        ][-1:],
        "anchor_predictors": np.empty((0, *state.shape), dtype=float),
        "anchor_step_wall_seconds": np.empty(0, dtype=float),
        "sampled_flags": np.empty(0, dtype=bool),
        "sampled_state_error_estimates": np.empty(0, dtype=float),
        "sampled_extraction_error_estimates": np.empty(0, dtype=float),
        "step_scaled_residuals": np.empty(0, dtype=float),
        "step_discrete_ledger_defects": np.empty(0, dtype=float),
        "step_mapped_closure_defects": np.empty(0, dtype=float),
        "step_reconstruction_factors": np.empty(0, dtype=float),
        "step_incoming_characteristics": np.empty(0, dtype=np.int64),
        "anchor_extraction_values": np.empty((0, 13), dtype=float),
        "extraction_identity_defects": np.empty(0, dtype=float),
        "extraction_ledger_audits": np.empty((0, 4), dtype=float),
    }


def _append(existing: np.ndarray, new) -> np.ndarray:
    addition = np.asarray(new)
    if addition.size == 0:
        return np.asarray(existing)
    return np.concatenate((np.asarray(existing), addition), axis=0)


def _time_us(value: float) -> int:
    return int(np.rint(float(value) * 1.0e6))


def _audit_indices(base: dict[str, np.ndarray]) -> set[int]:
    targets = set(TEMPORAL_AUDIT_TARGET_MICROSECONDS)
    return {
        index
        for index, value in enumerate(base["accepted_times"][1:])
        if _time_us(value) in targets
    }


def _checkpoint_metadata(completed: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "completed_steps": completed,
        "source_identity": _source_identity(),
        "input_identity": _input_identity(),
    }


def _validate_generation(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata_path = path / "metadata.json"
    arrays_path = path / "anchor.npz"
    hashes = _read_json(path / "hashes.json")
    if hashes != {
        "anchor.npz": _sha256(arrays_path),
        "metadata.json": _sha256(metadata_path),
    }:
        raise RuntimeError(f"c4e11 checkpoint hashes changed: {path.name}")
    metadata = _read_json(metadata_path)
    if metadata["source_identity"] != _source_identity():
        raise RuntimeError("c4e11 checkpoint source identity changed")
    if metadata["input_identity"] != _input_identity():
        raise RuntimeError("c4e11 checkpoint input identity changed")
    arrays = _load_npz(arrays_path)
    if int(metadata["completed_steps"]) != arrays["anchor_step_wall_seconds"].size:
        raise RuntimeError("c4e11 checkpoint step count changed")
    return metadata, arrays


def _load_latest_checkpoint() -> tuple[dict | None, dict[str, np.ndarray] | None]:
    if not CHECKPOINT_DIRECTORY.exists():
        return None, None
    candidates = sorted(
        path
        for path in CHECKPOINT_DIRECTORY.glob("generation_[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    if not candidates:
        return None, None
    metadata, arrays = _validate_generation(candidates[-1])
    return metadata, arrays


def _save_generation(arrays: dict[str, np.ndarray]) -> None:
    completed = int(arrays["anchor_step_wall_seconds"].size)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    name = f"generation_{completed:04d}"
    final = CHECKPOINT_DIRECTORY / name
    if final.exists():
        _validate_generation(final)
    else:
        temporary = CHECKPOINT_DIRECTORY / f"{name}.tmp-{os.getpid()}"
        temporary.mkdir(parents=False, exist_ok=False)
        arrays_path = temporary / "anchor.npz"
        metadata_path = temporary / "metadata.json"
        np.savez_compressed(arrays_path, **arrays)
        _write_json(metadata_path, _checkpoint_metadata(completed))
        _write_json(
            temporary / "hashes.json",
            {
                "anchor.npz": _sha256(arrays_path),
                "metadata.json": _sha256(metadata_path),
            },
        )
        os.replace(temporary, final)
    latest_tmp = CHECKPOINT_DIRECTORY / f"LATEST.tmp-{os.getpid()}"
    latest_tmp.write_text(name + "\n", encoding="utf-8")
    os.replace(latest_tmp, LATEST_PATH)


def _patch_configuration():
    c4e8._patch_modules()
    configuration = c4e8.h2b1._configuration()
    began = time.perf_counter()
    frozen_tangent, _ = c4e8.h2b1._build_frozen_tangent(configuration)
    return configuration, frozen_tangent, time.perf_counter() - began


def _step_record(result) -> dict:
    return c4e8.h2b1.controller._step_record(result)


def _run_anchor(
    configuration,
    frozen_tangent,
    parent: dict[str, np.ndarray],
    fine_start: dict[str, np.ndarray],
    *,
    through_steps: int | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    h2b1 = c4e8.h2b1
    controller = h2b1.controller
    base = _base(parent)
    tangent_state = np.asarray(
        parent["tangent__state_directions"][:, GENERIC_INDEX], dtype=float
    )
    field_scales = np.asarray(parent["tangent__field_scales"], dtype=float)
    extraction_scales = _load_npz(
        c4e8.c4e3.extraction5.DECISIVE_ARRAYS
    )["export_scales"]
    metadata, arrays = _load_latest_checkpoint()
    if arrays is None:
        arrays = _initial_arrays(parent, fine_start)
        anchor_value, identity, audit = c4e8.c4e3._extraction_value(
            configuration["context"], arrays["anchor_states"][0]
        )
        arrays["anchor_extraction_values"] = anchor_value[None, :]
        arrays["extraction_identity_defects"] = np.asarray([identity])
        arrays["extraction_ledger_audits"] = audit[None, :]
        _save_generation(arrays)
    start = int(arrays["anchor_step_wall_seconds"].size)
    stop = base["accepted_timesteps"].size
    if through_steps is not None:
        stop = min(stop, start + max(int(through_steps), 0))
    audit_indices = _audit_indices(base)
    main_contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    main_contract = json.loads(json.dumps(main_contract))
    main_contract["maximum_timestep_seconds"] = MAXIMUM_TIMESTEP_SECONDS
    for index in range(start, stop):
        state = np.asarray(arrays["anchor_states"][-1], dtype=float)
        history = h2b1.h2a2._history(
            arrays["anchor_primitive_histories"][-1],
            arrays["anchor_mapped_histories"][-1],
            arrays["anchor_height_histories"][-1],
            arrays["anchor_previous_timesteps"][-1],
        )
        predictor = base["accepted_states"][index + 1] + tangent_state[index + 1] - state
        dt = float(base["accepted_timesteps"][index])
        began = time.perf_counter()
        full = h2b1.advance_causal_five_field_monolithic_bdf(
            configuration["context"],
            state,
            dt,
            frozen_tangent,
            order=2,
            history=history,
            initial_primitive_increment=predictor,
            residual_tolerance=1.0e-10,
            ledger_tolerance=1.0e-12,
            maximum_scaled_primitive_change=5.0e-3,
        )
        if full.history is None or not controller._step_passed(full, main_contract):
            raise RuntimeError(f"c4e11 full anchor step {index} failed")
        records = [_step_record(full)]
        sampled = index in audit_indices
        state_error = 0.0
        extraction_error = 0.0
        if sampled:
            half_first = h2b1.advance_causal_five_field_monolithic_bdf(
                configuration["context"],
                state,
                0.5 * dt,
                frozen_tangent,
                order=2,
                history=history,
                residual_tolerance=1.0e-10,
                ledger_tolerance=1.0e-12,
                maximum_scaled_primitive_change=5.0e-3,
            )
            if half_first.history is None:
                raise RuntimeError(f"c4e11 first half step {index} has no history")
            half_second = h2b1.advance_causal_five_field_monolithic_bdf(
                configuration["context"],
                half_first.primitive_charts,
                0.5 * dt,
                frozen_tangent,
                order=2,
                history=half_first.history,
                residual_tolerance=1.0e-10,
                ledger_tolerance=1.0e-12,
                maximum_scaled_primitive_change=5.0e-3,
            )
            if half_second.history is None or not (
                controller._step_passed(half_first, main_contract)
                and controller._step_passed(half_second, main_contract)
            ):
                raise RuntimeError(f"c4e11 sampled half steps {index} failed")
            records.extend((_step_record(half_first), _step_record(half_second)))
            state_error = controller._state_estimate(
                full.primitive_charts, half_second.primitive_charts, field_scales
            )
            full_extraction, _, _ = c4e8.c4e3._extraction_value(
                configuration["context"], full.primitive_charts
            )
            half_extraction, _, _ = c4e8.c4e3._extraction_value(
                configuration["context"], half_second.primitive_charts
            )
            extraction_error = float(
                np.max(np.abs(full_extraction - half_extraction) / extraction_scales)
            )
            if max(state_error, extraction_error) > 2.5e-4:
                raise RuntimeError(f"c4e11 sampled temporal error {index} failed")
        anchor_value, identity, audit = c4e8.c4e3._extraction_value(
            configuration["context"], full.primitive_charts
        )
        wall = time.perf_counter() - began
        arrays["anchor_states"] = _append(
            arrays["anchor_states"], full.primitive_charts[None, ...]
        )
        arrays["anchor_primitive_histories"] = _append(
            arrays["anchor_primitive_histories"],
            full.history.previous_primitive_increment[None, ...],
        )
        arrays["anchor_mapped_histories"] = _append(
            arrays["anchor_mapped_histories"],
            full.history.previous_mapped_storage_increment[None, ...],
        )
        arrays["anchor_height_histories"] = _append(
            arrays["anchor_height_histories"],
            full.history.previous_responsive_height_storage_increment[None, ...],
        )
        arrays["anchor_previous_timesteps"] = _append(
            arrays["anchor_previous_timesteps"],
            [full.history.previous_timestep_seconds],
        )
        arrays["anchor_predictors"] = _append(
            arrays["anchor_predictors"], predictor[None, ...]
        )
        arrays["anchor_step_wall_seconds"] = _append(
            arrays["anchor_step_wall_seconds"], [wall]
        )
        arrays["sampled_flags"] = _append(arrays["sampled_flags"], [sampled])
        arrays["sampled_state_error_estimates"] = _append(
            arrays["sampled_state_error_estimates"], [state_error]
        )
        arrays["sampled_extraction_error_estimates"] = _append(
            arrays["sampled_extraction_error_estimates"], [extraction_error]
        )
        arrays["step_scaled_residuals"] = _append(
            arrays["step_scaled_residuals"],
            [max(item["maximum_scaled_residual"] for item in records)],
        )
        arrays["step_discrete_ledger_defects"] = _append(
            arrays["step_discrete_ledger_defects"],
            [max(item["maximum_discrete_ledger_defect"] for item in records)],
        )
        arrays["step_mapped_closure_defects"] = _append(
            arrays["step_mapped_closure_defects"],
            [max(item["maximum_mapped_endpoint_path_closure_defect"] for item in records)],
        )
        arrays["step_reconstruction_factors"] = _append(
            arrays["step_reconstruction_factors"],
            [min(item["minimum_path_reconstruction_factor"] for item in records)],
        )
        arrays["step_incoming_characteristics"] = _append(
            arrays["step_incoming_characteristics"],
            [max(item["incoming_excision_characteristics"] for item in records)],
        )
        arrays["anchor_extraction_values"] = _append(
            arrays["anchor_extraction_values"], anchor_value[None, :]
        )
        arrays["extraction_identity_defects"] = _append(
            arrays["extraction_identity_defects"], [identity]
        )
        arrays["extraction_ledger_audits"] = _append(
            arrays["extraction_ledger_audits"], audit[None, :]
        )
        _save_generation(arrays)
        print(
            f"c4e11-anchor: {index + 1}/{base['accepted_timesteps'].size} "
            f"t={base['accepted_times'][index + 1]:.8e} wall={wall:.1f}s "
            f"sampled={sampled}",
            flush=True,
        )
    report = _report(configuration, frozen_tangent, parent, arrays)
    return report, arrays


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    for index in range(1, times.size):
        dt = float(times[index] - times[index - 1])
        result[index] = result[index - 1] + 0.5 * dt * (
            values[index - 1] + values[index]
        )
    return result


def _response_metrics(predicted, actual, scales) -> dict:
    metrics = c4e8.h2b1.h2a3.h2a1.h1._response_metrics(
        predicted, actual, scales
    )
    metrics["discrepancy_fraction_of_observable_response"] = float(
        metrics["maximum_scaled_discrepancy"]
        / max(metrics["maximum_scaled_actual_response"], np.finfo(float).tiny)
    )
    return metrics


def _report(configuration, frozen_tangent, parent, arrays) -> dict:
    base = _base(parent)
    completed = int(arrays["anchor_step_wall_seconds"].size)
    complete = completed == base["accepted_timesteps"].size
    if not complete:
        return {"passed": True, "complete": False, "completed_steps": completed}
    field_scales = np.asarray(parent["tangent__field_scales"], dtype=float)
    extraction_scales = _load_npz(
        c4e8.c4e3.extraction5.DECISIVE_ARRAYS
    )["export_scales"]
    actual_state = arrays["anchor_states"] - base["accepted_states"]
    predicted_state = parent["tangent__state_directions"][:, GENERIC_INDEX]
    base_extraction = parent["extraction__base_values"]
    actual_extraction = arrays["anchor_extraction_values"] - base_extraction
    predicted_extraction = parent["extraction__tangent_directions"][:, GENERIC_INDEX]
    actual_cumulative = _cumulative(actual_extraction, base["accepted_times"])
    predicted_cumulative = _cumulative(predicted_extraction, base["accepted_times"])
    state_metrics = _response_metrics(predicted_state, actual_state, field_scales)
    extraction_metrics = _response_metrics(
        predicted_extraction, actual_extraction, extraction_scales
    )
    cumulative_metrics = _response_metrics(
        predicted_cumulative, actual_cumulative, extraction_scales
    )
    readiness = c4e8.h2b1.h2a2.h2.h1.b1a._state_audit(
        configuration["context"], arrays["anchor_states"][-1]
    )
    replay = c4e8.h2b1._serialized_last_step_replay(
        "fine_generic_anchor",
        configuration,
        frozen_tangent,
        arrays["anchor_states"],
        arrays["anchor_primitive_histories"],
        arrays["anchor_mapped_histories"],
        arrays["anchor_height_histories"],
        arrays["anchor_previous_timesteps"],
        base["accepted_timesteps"],
        base["accepted_times"],
        arrays["anchor_predictors"][-1],
    )
    routine = arrays["anchor_step_wall_seconds"][~arrays["sampled_flags"]]
    sampled = arrays["anchor_step_wall_seconds"][arrays["sampled_flags"]]
    passed = bool(
        np.max(arrays["step_scaled_residuals"]) <= 1.0e-10
        and np.max(arrays["step_discrete_ledger_defects"]) <= 1.0e-12
        and np.max(arrays["step_mapped_closure_defects"]) <= 1.0e-9
        and np.min(arrays["step_reconstruction_factors"]) >= 1.0
        and np.max(arrays["step_incoming_characteristics"]) == 0
        and np.max(arrays["sampled_state_error_estimates"]) <= 2.5e-4
        and np.max(arrays["sampled_extraction_error_estimates"]) <= 2.5e-4
        and np.max(arrays["extraction_identity_defects"]) <= 1.0e-12
        and np.max(arrays["extraction_ledger_audits"]) <= 1.0e-9
        and readiness["maximum_h_over_r"] <= 0.12
        and readiness["minimum_scattering_optical_depth"] > 1.0
        and readiness["minimum_reconstruction_factor"] >= 1.0
        and replay["checkpoint_roundtrip_bitwise"]
        and replay["last_step_replay_bitwise"]
        and replay["maximum_scaled_residual"] <= 1.0e-10
    )
    arrays["actual_state_response"] = actual_state
    arrays["predicted_state_response"] = predicted_state
    arrays["base_extraction_values"] = base_extraction
    arrays["actual_extraction_response"] = actual_extraction
    arrays["predicted_extraction_response"] = predicted_extraction
    arrays["actual_cumulative_extraction_response"] = actual_cumulative
    arrays["predicted_cumulative_extraction_response"] = predicted_cumulative
    return {
        "passed": passed,
        "complete": True,
        "completed_steps": completed,
        "sampled_temporal_audits": int(np.sum(arrays["sampled_flags"])),
        "maximum_scaled_residual": float(np.max(arrays["step_scaled_residuals"])),
        "maximum_discrete_ledger_defect": float(
            np.max(arrays["step_discrete_ledger_defects"])
        ),
        "maximum_mapped_endpoint_path_closure_defect": float(
            np.max(arrays["step_mapped_closure_defects"])
        ),
        "minimum_reconstruction_factor": float(
            np.min(arrays["step_reconstruction_factors"])
        ),
        "maximum_incoming_excision_characteristics": int(
            np.max(arrays["step_incoming_characteristics"])
        ),
        "maximum_sampled_state_error_estimate": float(
            np.max(arrays["sampled_state_error_estimates"])
        ),
        "maximum_sampled_extraction_error_estimate": float(
            np.max(arrays["sampled_extraction_error_estimates"])
        ),
        "maximum_extraction_identity_defect": float(
            np.max(arrays["extraction_identity_defects"])
        ),
        "maximum_extraction_ledger_audit": float(
            np.max(arrays["extraction_ledger_audits"])
        ),
        "state_tangent_comparison": state_metrics,
        "instantaneous_extraction_tangent_comparison": extraction_metrics,
        "cumulative_extraction_tangent_comparison": cumulative_metrics,
        "final_state_audit": readiness,
        "serialized_last_step_replay": replay,
        "median_routine_step_wall_seconds": float(np.median(routine)),
        "median_sampled_step_wall_seconds": float(np.median(sampled)),
        "total_anchor_step_wall_seconds": float(
            np.sum(arrays["anchor_step_wall_seconds"])
        ),
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
                    "sha256": _sha256(path),
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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


def _finalize(setup_wall: float, report: dict, arrays: dict[str, np.ndarray]) -> int:
    passed = bool(report["complete"] and report["passed"])
    classification = (
        "fine_20ms_generic_nonlinear_anchor_passed_final_spatial_reanalysis_authorized"
        if passed
        else "fine_20ms_generic_nonlinear_anchor_failed_spatial_certificate_blocked"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "setup_wall_seconds": setup_wall,
        "anchor": report,
        "fine_generic_anchor_completed": passed,
        "fine_twenty_ms_spatial_certificate_issued": False,
        "final_three_grid_spatial_reanalysis_authorized": passed,
        "fifty_ms_manifest_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4e12_final_three_grid_20ms_spatial_reanalysis"
            if passed
            else "fine_generic_anchor_failure_localization_only"
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "layout": FINE_LAYOUT,
            "profile": GENERIC_PROFILE,
            "target_microseconds": TARGET_MICROSECONDS,
            "temporal_audit_target_microseconds": (
                TEMPORAL_AUDIT_TARGET_MICROSECONDS
            ),
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "maximum_timestep_seconds": MAXIMUM_TIMESTEP_SECONDS,
        },
    )
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
            "input_hashes": _input_identity(),
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Fine 20 ms generic nonlinear anchor WP10c9d6c7c3b5c4e11",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"The targeted fine generic anchor completed `{report['completed_steps']}` accepted steps with `{report['sampled_temporal_audits']}` sampled full-versus-two-half temporal audits.",
                "",
                f"The maximum scaled nonlinear residual was `{report['maximum_scaled_residual']:.6e}`. The tangent discrepancy was `{report['state_tangent_comparison']['discrepancy_fraction_of_observable_response']:.6e}` of the state response and `{report['instantaneous_extraction_tangent_comparison']['discrepancy_fraction_of_observable_response']:.6e}` of the instantaneous extraction response.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
                "The 20 ms spatial certificate, 50 ms campaign, fixed-Q experiments, and reduced slow evolution remain blocked pending the final analysis-only three-grid decision.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--through-steps",
        type=int,
        default=None,
        help="Run at most this many additional accepted anchor steps.",
    )
    arguments = parser.parse_args(argv)
    _validate_parent()
    parent, fine_start = _parent_arrays()
    configuration, frozen_tangent, setup_wall = _patch_configuration()
    report, arrays = _run_anchor(
        configuration,
        frozen_tangent,
        parent,
        fine_start,
        through_steps=arguments.through_steps,
    )
    if not report["complete"]:
        print(json.dumps(_plain(report), indent=2, sort_keys=True))
        return 0
    return _finalize(setup_wall, report, arrays)


if __name__ == "__main__":
    raise SystemExit(main())
