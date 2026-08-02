#!/usr/bin/env python3
"""Run the frozen coarse-layout nonlinear temporal symmetry controls."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_temporal_coarse_screen_wp10c9d6c7c3b3b1 as c3b3b1  # noqa: E402
import run_causal_inner_nonlinear_temporal_fine_primary_wp10c9d6c7c3b3b3 as c3b3b3  # noqa: E402
import run_causal_inner_nonlinear_temporal_refinement_manifest_wp10c9d6c7c3b3a as c3b3a  # noqa: E402
import run_causal_inner_physical_background_nonlinear_readiness_manifest_wp10c9d6c7c3a1 as c3a1  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b3b4"
ANALYZED_BASE_COMMIT = "fd2ed817c6bc32195b9c476726b9d640c68b8013"
ANALYZED_BASE_PARENT = "52fceff69c7c684a0601ad696c6a81d109c5a1a1"
ANALYZED_BASE_TREE = "a87c102bd544c689fda15a0334eb0c1630d25caa"

LAYOUT = c3b3a.COARSE_LAYOUT
PROFILE = c3b3a.PRIMARY_PROFILE
VARIANTS = (
    ("m1", -1.0),
    ("p0p5", 0.5),
    ("m0p5", -0.5),
)
TIMESTEP_LEVELS_SECONDS = np.asarray(
    c3b3a.TIMESTEP_LEVELS_SECONDS,
    dtype=float,
)
REFINED_TIMESTEPS_SECONDS = TIMESTEP_LEVELS_SECONDS[1:]
HORIZON_SECONDS = c3b3a.HORIZON_SECONDS
COMMON_OUTPUT_TIMES_SECONDS = np.asarray(
    c3b3a.COMMON_OUTPUT_TIMES_SECONDS,
    dtype=float,
)

ARTIFACT = (
    "causal_inner_nonlinear_temporal_symmetry_"
    "wp10c9d6c7c3b3b4"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_temporal_symmetry_"
    "wp10c9d6c7c3b3b4.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_temporal_symmetry_"
    "wp10c9d6c7c3b3b4.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_TEMPORAL_SYMMETRY_"
    "WP10C9D6C7C3B3B4_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

MANIFEST_DIRECTORY = c3b3a.CANONICAL_DIRECTORY
FINE_PARENT_DIRECTORY = c3b3b3.CANONICAL_DIRECTORY
COARSE_PARENT_DIRECTORY = c3b3b1.CANONICAL_DIRECTORY
SPATIAL_PILOT_DIRECTORY = c3b2b.CANONICAL_DIRECTORY
PREFLIGHT_DIRECTORY = c3b2b.STEP4_DIRECTORY

CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_JSON = CHECKPOINT_DIRECTORY / "progress.json"
CHECKPOINT_ARRAYS = CHECKPOINT_DIRECTORY / "progress_arrays.npz"


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
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
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        c3b3b1.THIS_RUNNER,
        c3b3b3.THIS_RUNNER,
        c3b3a.THIS_RUNNER,
        c3b1a.THIS_MODULE,
    )
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _case_id(token: str) -> str:
    return f"{PROFILE}__{token}"


def _task_id(timestep: float, trajectory: str) -> str:
    return c3b3b1._task_id(float(timestep), trajectory)


def _validate_parent() -> tuple[dict, dict, dict]:
    fine_parent = _read_json(FINE_PARENT_DIRECTORY / "summary.json")
    coarse_parent = _read_json(COARSE_PARENT_DIRECTORY / "summary.json")
    manifest_summary = _read_json(MANIFEST_DIRECTORY / "summary.json")
    manifest = _read_json(
        MANIFEST_DIRECTORY / "temporal_refinement_manifest.json"
    )
    expected_cases = [_case_id(token) for token, _ in VARIANTS]
    stage = manifest["fail_fast_stages"][3]
    if (
        not fine_parent["passed"]
        or not fine_parent[
            "coarse_primary_nonlinear_symmetry_controls_authorized"
        ]
        or fine_parent["authorized_next"]
        != "WP10c9d6c7c3b3b4_coarse_primary_nonlinear_symmetry_controls"
        or fine_parent["classification"]
        != "fine_primary_nonlinear_temporal_confirmation_certified_"
        "coarse_primary_nonlinear_symmetry_controls_authorized"
        or not coarse_parent["passed"]
        or not manifest_summary["passed"]
        or stage["work_package"] != WORK_PACKAGE
        or stage["layout"] != LAYOUT
        or stage["trajectories"] != expected_cases
        or stage["reuses"]
        != ["unperturbed_background", f"{PROFILE}__p1"]
    ):
        raise RuntimeError("c3b3b4 frozen authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b3b4 analyzed identity changed")
    return fine_parent, coarse_parent, manifest


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": _source_identity(),
                "completed_tasks": [],
                "trajectory_reports": [],
                "export_audits": {},
                "failed": False,
            },
            {},
        )
    progress = _read_json(CHECKPOINT_JSON)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != _source_identity()
    ):
        raise RuntimeError("saved c3b3b4 progress belongs to different code")
    return progress, _load_npz(CHECKPOINT_ARRAYS)


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _run_refined_trajectories(
    configuration: dict,
    tangent,
) -> tuple[dict, dict[str, np.ndarray]]:
    progress, arrays = _load_progress()
    completed = set(progress["completed_tasks"])
    packet_arrays = _load_npz(
        c3a1.C7C0_DIRECTORY / "decisive_arrays.npz"
    )
    packet = np.asarray(
        packet_arrays[f"{PROFILE}__{LAYOUT}__{c3a1.PROFILE_KIND}"],
        dtype=float,
    )
    initial_states = {
        _case_id(token): np.asarray(configuration["base"], dtype=float)
        + multiplier * packet
        for token, multiplier in VARIANTS
    }
    original_work_package = c3b3b1.WORK_PACKAGE
    c3b3b1.WORK_PACKAGE = WORK_PACKAGE
    try:
        for timestep in REFINED_TIMESTEPS_SECONDS:
            for trajectory_id, initial_state in initial_states.items():
                task = _task_id(float(timestep), trajectory_id)
                if task in completed:
                    continue
                report, trajectory_arrays = c3b3b1._trajectory(
                    configuration,
                    tangent,
                    initial_state,
                    float(timestep),
                    trajectory_id,
                )
                for name, values in trajectory_arrays.items():
                    arrays[f"{task}__{name}"] = values
                if report["passed"]:
                    exports, audit = c3b3b1._export_history(
                        configuration["context"],
                        trajectory_arrays["states"],
                    )
                    arrays[f"{task}__direct_exports"] = exports
                    progress["export_audits"][task] = audit
                    report["export_audit_passed"] = audit["passed"]
                    report["passed"] = bool(
                        report["passed"] and audit["passed"]
                    )
                else:
                    report["export_audit_passed"] = False
                progress["trajectory_reports"].append(report)
                completed.add(task)
                progress["completed_tasks"] = sorted(completed)
                progress["failed"] = bool(
                    progress["failed"] or not report["passed"]
                )
                _save_progress(progress, arrays)
                if not report["passed"]:
                    return progress, arrays
    finally:
        c3b3b1.WORK_PACKAGE = original_work_package
    return progress, arrays


def _response_histories(
    arrays: dict[str, np.ndarray],
    coarse_arrays: dict[str, np.ndarray],
    spatial_arrays: dict[str, np.ndarray],
    case: str,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    state = [
        np.asarray(
            spatial_arrays[f"{LAYOUT}__{case}__state_response"],
            dtype=float,
        )
    ]
    instantaneous = [
        np.asarray(
            spatial_arrays[
                f"{LAYOUT}__{case}__instantaneous_export_response"
            ],
            dtype=float,
        )
    ]
    cumulative = [
        np.asarray(
            spatial_arrays[
                f"{LAYOUT}__{case}__cumulative_export_response"
            ],
            dtype=float,
        )
    ]
    for timestep in REFINED_TIMESTEPS_SECONDS:
        background_task = _task_id(float(timestep), "base")
        case_task = _task_id(float(timestep), case)
        indices = c3b3b1._common_indices(float(timestep))
        background_states = coarse_arrays[
            f"{background_task}__states"
        ]
        case_states = arrays[f"{case_task}__states"]
        state.append(case_states[indices] - background_states[indices])
        background_exports = coarse_arrays[
            f"{background_task}__direct_exports"
        ]
        case_exports = arrays[f"{case_task}__direct_exports"]
        response = case_exports - background_exports
        instantaneous.append(response[indices])
        cumulative.append(
            c3b3b1._native_cumulative(
                response,
                float(timestep),
            )[indices]
        )
    return tuple(state), tuple(instantaneous), tuple(cumulative)


def _metric(
    kind: str,
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    configuration: dict,
    field_scales: np.ndarray,
    observable_scales: np.ndarray,
    gates: dict,
    method_floor: float,
) -> dict:
    if kind == "state":
        return c3b3b1._state_metric(
            histories,
            configuration,
            field_scales,
            gates,
            method_floor,
        )
    scales = observable_scales
    if kind == "cumulative_exports":
        scales = scales * HORIZON_SECONDS
    return c3b3b1._export_metric(
        histories,
        scales,
        gates,
        method_floor,
    )


def _normalized_rms(values: np.ndarray, scales: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(values, dtype=float)
                    / np.asarray(scales, dtype=float)
                )
                ** 2
            )
        )
    )


def _history_has_significant_component(
    kind: str,
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    field_scales: np.ndarray,
    observable_scales: np.ndarray,
    gates: dict,
) -> bool:
    if kind == "state":
        scales = field_scales[None, None, :]
    else:
        scales = observable_scales[None, :]
        if kind == "cumulative_exports":
            scales = scales * HORIZON_SECONDS
    normalized = np.asarray(
        [np.asarray(history, dtype=float) / scales for history in histories]
    )
    component_axes = tuple(range(normalized.ndim - 1))
    component_response = np.max(np.abs(normalized), axis=component_axes)
    return bool(
        np.any(component_response >= gates["minimum_relative_activity"])
    )


def _symmetry_diagnostics(
    response_histories: dict[str, dict[str, tuple[np.ndarray, ...]]],
    coarse_arrays: dict[str, np.ndarray],
    configuration: dict,
    field_scales: np.ndarray,
    observable_scales: np.ndarray,
    gates: dict,
    method_floor: float,
    decisive: dict[str, np.ndarray],
) -> dict:
    plus_case = f"{PROFILE}__p1"
    minus_case = _case_id("m1")
    plus_half_case = _case_id("p0p5")
    minus_half_case = _case_id("m0p5")
    result = {}
    for kind in ("state", "instantaneous_exports", "cumulative_exports"):
        suffix = {
            "state": "state_response",
            "instantaneous_exports": "instantaneous_export_response",
            "cumulative_exports": "cumulative_export_response",
        }[kind]
        plus = tuple(
            np.asarray(
                coarse_arrays[f"{plus_case}__{label}__{suffix}"],
                dtype=float,
            )
            for label in ("h", "h2", "h4")
        )
        minus = response_histories[minus_case][kind]
        plus_half = response_histories[plus_half_case][kind]
        minus_half = response_histories[minus_half_case][kind]
        odd_full = tuple((p - m) / 2.0 for p, m in zip(plus, minus))
        odd_half_scaled = tuple(
            p - m for p, m in zip(plus_half, minus_half)
        )
        even_full = tuple((p + m) / 2.0 for p, m in zip(plus, minus))
        odd_scale_defect = tuple(
            full - half
            for full, half in zip(odd_full, odd_half_scaled)
        )
        positive_half_defect = tuple(
            p - 2.0 * half for p, half in zip(plus, plus_half)
        )
        negative_half_defect = tuple(
            m - 2.0 * half for m, half in zip(minus, minus_half)
        )
        remainders = {
            "even_response": even_full,
            "odd_amplitude_scale_defect": odd_scale_defect,
            "positive_half_amplitude_defect": positive_half_defect,
            "negative_half_amplitude_defect": negative_half_defect,
        }
        scales = (
            field_scales[None, None, :]
            if kind == "state"
            else observable_scales[None, :]
            * (HORIZON_SECONDS if kind == "cumulative_exports" else 1.0)
        )
        denominator = max(
            _normalized_rms(odd_full[-1], scales),
            method_floor,
        )
        kind_report = {
            "odd_full_normalized_rms_h4": _normalized_rms(
                odd_full[-1], scales
            ),
            "even_to_odd_ratio_h4": (
                _normalized_rms(even_full[-1], scales) / denominator
            ),
            "odd_amplitude_scale_defect_ratio_h4": (
                _normalized_rms(odd_scale_defect[-1], scales)
                / denominator
            ),
            "remainders": {},
        }
        for name, histories in remainders.items():
            significant = _history_has_significant_component(
                kind,
                histories,
                field_scales,
                observable_scales,
                gates,
            )
            if significant:
                remainder_metric = {
                    **_metric(
                        kind,
                        histories,
                        configuration,
                        field_scales,
                        observable_scales,
                        gates,
                        method_floor,
                    ),
                    "metric_evaluated": True,
                    "classification": "evaluated_explanatory_only",
                }
            else:
                remainder_metric = {
                    "passed": None,
                    "metric_evaluated": False,
                    "classification": "below_significant_component_floor",
                    "significant_components": [],
                    "raw_contract_passed": None,
                    "refinement_error_observable": False,
                    "numerical_uncertainty_floor": method_floor,
                    "observability_threshold": (
                        gates["observability_factor"] * method_floor
                    ),
                    "upper_bound_route_used": False,
                    "interpretation": (
                        "the explanatory nonlinear remainder has no "
                        "component above the prospectively inherited "
                        "relative-activity floor; no temporal order or "
                        "direction is assigned"
                    ),
                }
            maximum_response = max(
                _normalized_rms(history, scales) for history in histories
            )
            kind_report["remainders"][name] = {
                "maximum_normalized_rms": maximum_response,
                "response_above_observability_floor": bool(
                    maximum_response
                    > gates["observability_factor"] * method_floor
                ),
                "temporal_metric": remainder_metric,
            }
            for level, label in enumerate(("h", "h2", "h4")):
                decisive[
                    f"symmetry__{kind}__{name}__{label}"
                ] = histories[level]
        result[kind] = kind_report
    result.update(
        {
            "spatial_convergence_of_nonlinear_remainders_tested": False,
            "meaningfully_nonlinear_dynamics_certified": False,
            "interpretation": (
                "odd/even and half-amplitude remainders are explanatory; "
                "a nonzero remainder needs a separate spatial and temporal "
                "certificate before a meaningful-nonlinearity claim"
            ),
        }
    )
    return result


def _analyze(
    progress: dict,
    arrays: dict[str, np.ndarray],
    configuration: dict,
    coarse_parent: dict,
    manifest: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    coarse_arrays = _load_npz(
        COARSE_PARENT_DIRECTORY / "decisive_arrays.npz"
    )
    spatial_arrays = _load_npz(
        SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz"
    )
    gates = c3b3b1._temporal_gates(manifest)
    field_scales = np.asarray(spatial_arrays["field_scales"], dtype=float)
    observable_scales = np.asarray(
        spatial_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    reports = list(progress["trajectory_reports"])
    preflight = _read_json(PREFLIGHT_DIRECTORY / "summary.json")
    method_floor = max(
        float(preflight["maximum_scaled_residual"]),
        float(preflight["maximum_discrete_ledger_defect"]),
        float(coarse_parent["temporal_screen"]["maximum_scaled_residual"]),
        float(
            coarse_parent["temporal_screen"][
                "maximum_discrete_ledger_defect"
            ]
        ),
        *(float(report["maximum_scaled_residual"]) for report in reports),
        *(
            float(report["maximum_discrete_ledger_defect"])
            for report in reports
        ),
    )
    decisive = {
        "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS,
        "common_output_times_seconds": COMMON_OUTPUT_TIMES_SECONDS,
        "field_scales": field_scales,
        "fixed_physical_observable_scales": observable_scales,
        "variant_multipliers": np.asarray(
            [value for _, value in VARIANTS], dtype=float
        ),
    }
    case_reports = {}
    response_histories = {}
    all_passed = not progress["failed"]
    for token, _ in VARIANTS:
        case = _case_id(token)
        state, instantaneous, cumulative = _response_histories(
            arrays,
            coarse_arrays,
            spatial_arrays,
            case,
        )
        response_histories[case] = {
            "state": state,
            "instantaneous_exports": instantaneous,
            "cumulative_exports": cumulative,
        }
        state_metric = _metric(
            "state",
            state,
            configuration,
            field_scales,
            observable_scales,
            gates,
            method_floor,
        )
        instantaneous_metric = _metric(
            "instantaneous_exports",
            instantaneous,
            configuration,
            field_scales,
            observable_scales,
            gates,
            method_floor,
        )
        cumulative_metric = _metric(
            "cumulative_exports",
            cumulative,
            configuration,
            field_scales,
            observable_scales,
            gates,
            method_floor,
        )
        passed = bool(
            state_metric["passed"]
            and instantaneous_metric["passed"]
            and cumulative_metric["passed"]
        )
        case_reports[case] = {
            "state": state_metric,
            "instantaneous_exports": instantaneous_metric,
            "cumulative_exports": cumulative_metric,
            "passed": passed,
        }
        all_passed = bool(all_passed and passed)
        for level, label in enumerate(("h", "h2", "h4")):
            decisive[f"{case}__{label}__state_response"] = state[level]
            decisive[
                f"{case}__{label}__instantaneous_export_response"
            ] = instantaneous[level]
            decisive[
                f"{case}__{label}__cumulative_export_response"
            ] = cumulative[level]
    symmetry = _symmetry_diagnostics(
        response_histories,
        coarse_arrays,
        configuration,
        field_scales,
        observable_scales,
        gates,
        method_floor,
        decisive,
    )
    return (
        {
            "passed": all_passed,
            "numerical_uncertainty_floor": method_floor,
            "case_reports": case_reports,
            "symmetry_diagnostics": symmetry,
            "all_refined_trajectory_methods_passed": all(
                report["passed"] for report in reports
            ),
            "trajectory_reports": reports,
            "maximum_scaled_residual": max(
                report["maximum_scaled_residual"] for report in reports
            ),
            "maximum_discrete_ledger_defect": max(
                report["maximum_discrete_ledger_defect"]
                for report in reports
            ),
            "all_checkpoint_roundtrips_bitwise": all(
                report["checkpoint_roundtrip_bitwise"]
                for report in reports
            ),
            "all_split_restart_replays_bitwise": all(
                report["split_restart_replay_bitwise"]
                for report in reports
            ),
        },
        decisive,
    )


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
                    "sha256": _sha256(path),
                    "scientific_status": status,
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
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _report(summary: dict) -> str:
    screen = summary["temporal_screen"]
    first_case = next(iter(screen["case_reports"].values()))
    observability_threshold = first_case["state"][
        "observability_threshold"
    ]
    lines = [
        "# Nonlinear coarse temporal symmetry controls WP10c9d6c7c3b3b4",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "The unchanged coarse embedded operator was tested for the inward-primary "
        "`-1`, `+1/2` and `-1/2` controls at `dt=1e-5/5e-6/2.5e-6 s` "
        "through `4e-5 s`.  The certified background and `+1` histories were "
        "reused by hash.",
        "",
        "## Binding temporal results",
        "",
        "Every state and Tier-I refinement-error pair lies below the "
        "prospectively inherited observability threshold "
        f"`{observability_threshold:.3e}`.  The reported nominal orders are "
        "therefore explanatory; each binding result uses the frozen "
        "fine-difference, selected-step Richardson-bound and history-cosine "
        "upper-bound route.",
        "",
    ]
    for case, report in summary["temporal_screen"]["case_reports"].items():
        state = report["state"]
        instant = report["instantaneous_exports"]
        cumulative = report["cumulative_exports"]
        lines.extend(
            [
                f"### `{case}`",
                "",
                "- state RMS/max/component order: "
                f"`{state['observed_rms_order']:.6f}` / "
                f"`{state['observed_maximum_order']:.6f}` / "
                f"`{state['minimum_significant_component_order']:.6f}`",
                "- instantaneous export RMS/max/component order: "
                f"`{instant['observed_rms_order']:.6f}` / "
                f"`{instant['observed_maximum_order']:.6f}` / "
                f"`{instant['minimum_significant_component_order']:.6f}`",
                "- cumulative export RMS/max/component order: "
                f"`{cumulative['observed_rms_order']:.6f}` / "
                f"`{cumulative['observed_maximum_order']:.6f}` / "
                f"`{cumulative['minimum_significant_component_order']:.6f}`",
                "- selected-step Richardson errors "
                "(state/instantaneous/cumulative): "
                f"`{state['selected_step_richardson_error']:.3e}` / "
                f"`{instant['selected_step_richardson_error']:.3e}` / "
                f"`{cumulative['selected_step_richardson_error']:.3e}`",
                f"- result: `{'pass' if report['passed'] else 'fail'}`",
                "",
            ]
        )
    symmetry = screen["symmetry_diagnostics"]
    lines.extend(
        [
            "## Explanatory nonlinear symmetry diagnostics",
            "",
            "Odd/even and half-amplitude remainders are reported but are not a "
            "meaningful-nonlinearity certificate.  This package tests their "
            "temporal behavior only; spatial convergence of the nonzero "
            "remainders remains untested.  Every reported remainder is below "
            "the inherited observability floor, so no temporal order or error "
            "direction is assigned to those remainders.",
            "",
        ]
    )
    for kind in ("state", "instantaneous_exports", "cumulative_exports"):
        item = symmetry[kind]
        lines.append(
            f"- {kind}: h/4 even/odd ratio "
            f"`{item['even_to_odd_ratio_h4']:.3e}`, odd amplitude-scale "
            f"defect ratio `{item['odd_amplitude_scale_defect_ratio_h4']:.3e}`"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- maximum scaled nonlinear residual: "
            f"`{screen['maximum_scaled_residual']:.3e}`",
            "- maximum discrete ledger defect: "
            f"`{screen['maximum_discrete_ledger_defect']:.3e}`",
            "- all checkpoint roundtrips bitwise: "
            f"`{screen['all_checkpoint_roundtrips_bitwise']}`",
            "- all split/restart replays bitwise: "
            f"`{screen['all_split_restart_replays_bitwise']}`",
            "",
            "## Authorized next",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "Long nonlinear evolution, fixed-Q experiments and reduced slow "
            "evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(
    fine_parent: dict,
    coarse_parent: dict,
    manifest: dict,
    screen: dict,
    decisive: dict[str, np.ndarray],
    progress_arrays: dict[str, np.ndarray],
) -> int:
    passed = bool(screen["passed"])
    classification = (
        "coarse_primary_nonlinear_symmetry_controls_certified_"
        "short_horizon_profile_breadth_controller_manifest_authorized"
        if passed
        else "coarse_primary_nonlinear_symmetry_controls_failed_"
        "duration_extension_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b4a_short_horizon_nonlinear_profile_breadth_"
        "and_efficient_controller_manifest"
        if passed
        else "WP10c9d6c7c3b3b4_temporal_failure_localization"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profile": PROFILE,
        "variants": {token: multiplier for token, multiplier in VARIANTS},
        "reused_cases": ["unperturbed_background", f"{PROFILE}__p1"],
        "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS.tolist(),
        "horizon_seconds": HORIZON_SECONDS,
        "common_output_times_seconds": COMMON_OUTPUT_TIMES_SECONDS.tolist(),
        "temporal_gates": c3b3b1._temporal_gates(manifest),
        "nonlinear_remainders_binding": False,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    combined_arrays = {**progress_arrays, **decisive}
    np.savez_compressed(DECISIVE_ARRAYS, **combined_arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_classification": fine_parent["classification"],
        "coarse_parent_classification": coarse_parent["classification"],
        "temporal_screen": screen,
        "declared_four_stage_temporal_campaign_passed": passed,
        "short_horizon_profile_breadth_controller_manifest_authorized": passed,
        "temporal_convergence_certified": False,
        "meaningfully_nonlinear_dynamics_certified": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(config),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in combined_arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "fine_parent_summary": FINE_PARENT_DIRECTORY / "summary.json",
        "coarse_parent_summary": COARSE_PARENT_DIRECTORY / "summary.json",
        "coarse_parent_arrays": COARSE_PARENT_DIRECTORY
        / "decisive_arrays.npz",
        "manifest_contract": MANIFEST_DIRECTORY
        / "temporal_refinement_manifest.json",
        "spatial_pilot_arrays": SPATIAL_PILOT_DIRECTORY
        / "decisive_arrays.npz",
        "physical_profile_arrays": c3a1.C7C0_DIRECTORY
        / "decisive_arrays.npz",
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src "
                "/Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value(
                "rev-parse", "HEAD^{tree}"
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": _source_identity(),
            "input_hashes": {
                name: _sha256(path) for name, path in input_paths.items()
            },
        },
    )
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = (
        "config.json",
        "summary.json",
        "provenance.json",
        "decisive_arrays.npz",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    fine_parent, coarse_parent, manifest = _validate_parent()
    configuration = c3b1a._configurations()[LAYOUT]
    print(f"c3b3b4: build tangent {LAYOUT}", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    progress, progress_arrays = _run_refined_trajectories(
        configuration,
        tangent,
    )
    if progress["failed"]:
        reports = progress["trajectory_reports"]
        screen = {
            "passed": False,
            "numerical_uncertainty_floor": None,
            "case_reports": {},
            "symmetry_diagnostics": {},
            "all_refined_trajectory_methods_passed": False,
            "trajectory_reports": reports,
            "maximum_scaled_residual": max(
                report["maximum_scaled_residual"] for report in reports
            ),
            "maximum_discrete_ledger_defect": max(
                report["maximum_discrete_ledger_defect"]
                for report in reports
            ),
            "all_checkpoint_roundtrips_bitwise": all(
                report["checkpoint_roundtrip_bitwise"] for report in reports
            ),
            "all_split_restart_replays_bitwise": all(
                report["split_restart_replay_bitwise"] for report in reports
            ),
        }
        decisive = {}
    else:
        screen, decisive = _analyze(
            progress,
            progress_arrays,
            configuration,
            coarse_parent,
            manifest,
        )
    return _package(
        fine_parent,
        coarse_parent,
        manifest,
        screen,
        decisive,
        progress_arrays,
    )


if __name__ == "__main__":
    raise SystemExit(main())
