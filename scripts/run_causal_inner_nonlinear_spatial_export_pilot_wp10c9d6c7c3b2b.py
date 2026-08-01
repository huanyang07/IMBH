#!/usr/bin/env python3
"""Analyze the certified four-step nonlinear state and Tier-I exports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_embedded_manifest_wp10c9d6c7a as c7a  # noqa: E402
import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_manifest_wp10c9d6c7c3b2a as c3b2a  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_validation import (  # noqa: E402
    causal_embedded_active_direct_observables,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_field_history_inner_product,
    causal_field_history_norm,
    causal_trapezoid_weights,
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b2b"
ANALYZED_BASE_COMMIT = "67b3e70e3eadb43d12229f89daecec5b04d0e7fb"
ANALYZED_BASE_PARENT = "bb7eac431f4f12fd03d27f2937a515e5f5993eb1"
ANALYZED_BASE_TREE = "c0dfa27cedccd46ccc6f90cc12de844ef7ab95e2"

LAYOUTS = c3b2a.LAYOUTS
INNER_REFINEMENT_RATIOS = c3b2a.INNER_REFINEMENT_RATIOS
PROFILES = c3b2a.PROFILES
VARIANT_MULTIPLIERS = c3b2a.VARIANT_MULTIPLIERS
TIMES = c3b2a.OUTPUT_TIMES_SECONDS
TIMESTEP_SECONDS = c3b2a.TIMESTEP_SECONDS
OBSERVABLE_NAMES = c3b2a.OBSERVABLE_NAMES
COUPLING_PARENT_FACE = c3b2a.COUPLING_PARENT_FACE

ARTIFACT = (
    "causal_inner_nonlinear_spatial_export_pilot_"
    "wp10c9d6c7c3b2b"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_spatial_export_pilot_"
    "wp10c9d6c7c3b2b.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_spatial_export_pilot_"
    "wp10c9d6c7c3b2b.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_SPATIAL_EXPORT_PILOT_"
    "WP10C9D6C7C3B2B_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

MANIFEST_DIRECTORY = c3b2a.CANONICAL_DIRECTORY
BASE_DIRECTORY = c3b2a.BASE_DIRECTORY
STEP1_DIRECTORY = c3b2a.STEP1_DIRECTORY
STEP2_DIRECTORY = c3b2a.STEP2_DIRECTORY
STEP3_DIRECTORY = c3b2a.STEP3_DIRECTORY
STEP4_DIRECTORY = c3b2a.STEP4_DIRECTORY
C7A_DIRECTORY = c3b2a.C7A_DIRECTORY

CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
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
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(MANIFEST_DIRECTORY / "summary.json")
    manifest = _read_json(
        MANIFEST_DIRECTORY / "nonlinear_spatial_export_manifest.json"
    )
    if (
        not summary["passed"]
        or not summary["short_horizon_spatial_export_pilot_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b2b_nonlinear_short_horizon_"
        "spatial_export_pilot"
        or manifest["classification"]
        != "nonlinear_short_horizon_spatial_export_manifest_frozen_"
        "canonical_response_pilot_authorized"
    ):
        raise RuntimeError("c3b2a nonlinear pilot authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b2b analyzed identity changed")
    return summary, manifest


def _input_arrays() -> dict[str, dict[str, np.ndarray]]:
    return {
        "base": _load_npz(BASE_DIRECTORY / "decisive_arrays.npz"),
        "step1": _load_npz(STEP1_DIRECTORY / "decisive_arrays.npz"),
        "step2": _load_npz(STEP2_DIRECTORY / "decisive_arrays.npz"),
        "step3": _load_npz(STEP3_DIRECTORY / "decisive_arrays.npz"),
        "step4": _load_npz(STEP4_DIRECTORY / "decisive_arrays.npz"),
        "c7a": _load_npz(C7A_DIRECTORY / "decisive_arrays.npz"),
    }


def _case_id(profile: str, multiplier: float) -> str:
    return c3b2a._case_prefix("", profile, multiplier).removeprefix("__")


def _perturbed_history(
    arrays: dict[str, dict[str, np.ndarray]],
    layout: str,
    profile: str,
    multiplier: float,
) -> np.ndarray:
    prefix = c3b2a._case_prefix(layout, profile, multiplier)
    return np.asarray(
        (
            arrays["step1"][f"{prefix}__old_state"],
            arrays["step1"][f"{prefix}__final_state"],
            arrays["step2"][f"{prefix}__step2_final_state"],
            arrays["step3"][f"{prefix}__step3_final_state"],
            arrays["step4"][f"{prefix}__step4_final_state"],
        ),
        dtype=float,
    )


def _layouts_and_contexts(
    arrays: dict[str, dict[str, np.ndarray]],
):
    payloads = _read_json(c7a.C0E_CONTEXTS)
    gravitational_radius = float(
        payloads["contexts"][LAYOUTS[0]]["grid_gravitational_radius"]
    )
    parent_grid = make_kerr_schild_column_grid_from_edges(
        arrays["c7a"]["parent_grid_edges"],
        gravitational_radius,
    )
    layouts = {
        label: make_causal_embedded_patch_layout(
            parent_grid,
            COUPLING_PARENT_FACE,
            ratio,
        )
        for ratio, label in zip(
            INNER_REFINEMENT_RATIOS,
            LAYOUTS,
            strict=True,
        )
    }
    configurations = c3b1a._configurations()
    for label in LAYOUTS:
        if not np.array_equal(
            configurations[label]["context"].grid.edges,
            layouts[label].grid.edges,
        ):
            raise RuntimeError("embedded layout/context identity changed")
    return parent_grid, layouts, configurations


def _state_metrics(
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    parent_grid,
    field_scales: np.ndarray,
    gates: dict,
) -> dict:
    richardson = causal_windowed_richardson_reference(
        *histories,
        times=TIMES,
        coarse_cell_measures=parent_grid.cell_measures,
        field_scales=field_scales,
        relative_activity=gates["minimum_relative_activity"],
    )
    normalized = tuple(
        values / field_scales[None, None, :] for values in histories
    )
    differences = (
        normalized[1] - normalized[0],
        normalized[2] - normalized[1],
    )
    tiny = np.finfo(float).tiny
    coarse_maximum = float(np.max(np.abs(differences[0])))
    fine_maximum = float(np.max(np.abs(differences[1])))
    maximum_order = float(
        np.log2(
            max(coarse_maximum, tiny) / max(fine_maximum, tiny)
        )
    )
    weights = causal_trapezoid_weights(TIMES)
    medium_norm = causal_field_history_norm(
        histories[1],
        cell_measures=parent_grid.cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    )
    fine_norm = causal_field_history_norm(
        histories[2],
        cell_measures=parent_grid.cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    )
    history_cosine = float(
        causal_field_history_inner_product(
            histories[1],
            histories[2],
            cell_measures=parent_grid.cell_measures,
            field_scales=field_scales,
            time_weights=weights,
        )
        / max(medium_norm * fine_norm, tiny)
    )
    passed = bool(
        richardson.observed_order >= gates["minimum_rms_order"]
        and maximum_order >= gates["minimum_maximum_order"]
        and richardson.minimum_significant_component_order
        >= gates["minimum_significant_component_order"]
        and fine_maximum <= gates["maximum_fine_normalized_difference"]
        and history_cosine >= gates["minimum_history_cosine"]
        and richardson.refinement_error_cosine
        >= gates["minimum_refinement_error_cosine"]
    )
    return {
        "passed": passed,
        "observed_rms_order": richardson.observed_order,
        "observed_maximum_order": maximum_order,
        "minimum_significant_component_order": (
            richardson.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": fine_maximum,
        "history_cosine": history_cosine,
        "refinement_error_cosine": richardson.refinement_error_cosine,
        "coarse_medium_history_norm": (
            richardson.coarse_medium_history_norm
        ),
        "medium_fine_history_norm": richardson.medium_fine_history_norm,
        "maximum_coarse_reference_relative_error": (
            richardson.maximum_coarse_reference_relative_error
        ),
        "reference_choice_to_fine_difference_ratio": (
            richardson.reference_choice_to_fine_difference_ratio
        ),
    }


def _metric_payload(metrics) -> dict:
    indices = np.asarray(metrics.significant_components, dtype=int)
    return {
        "passed": metrics.passed,
        "significant_components": [
            OBSERVABLE_NAMES[index] for index in indices
        ],
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "component_orders": {
            OBSERVABLE_NAMES[index]: float(
                metrics.component_orders[position]
            )
            for position, index in enumerate(indices)
        },
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "coarse_medium_rms_difference": (
            metrics.coarse_medium_rms_difference
        ),
        "medium_fine_rms_difference": (
            metrics.medium_fine_rms_difference
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _packet_metrics(
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    scales: np.ndarray,
    gates: dict,
):
    return causal_packet_history_metrics(
        *histories,
        physical_scales=scales,
        relative_activity=gates["minimum_relative_activity"],
        minimum_rms_order=gates["minimum_rms_order"],
        minimum_maximum_order=gates["minimum_maximum_order"],
        minimum_significant_component_order=gates[
            "minimum_significant_component_order"
        ],
        maximum_fine_normalized_difference=gates[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=gates["minimum_history_cosine"],
        minimum_refinement_error_cosine=gates[
            "minimum_refinement_error_cosine"
        ],
    )


def _state_gate(
    arrays: dict[str, dict[str, np.ndarray]],
    layouts: dict,
    parent_grid,
    gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    field_scales = np.asarray(arrays["c7a"]["field_scales"], dtype=float)
    reports = {}
    decisive = {
        "times_seconds": np.asarray(TIMES),
        "field_scales": field_scales,
    }
    all_passed = True
    for profile in PROFILES:
        for multiplier in VARIANT_MULTIPLIERS:
            case = _case_id(profile, multiplier)
            histories = []
            for layout_label in LAYOUTS:
                response = (
                    _perturbed_history(
                        arrays,
                        layout_label,
                        profile,
                        multiplier,
                    )
                    - arrays["base"][f"{layout_label}__states"]
                )
                restricted = np.asarray(
                    [
                        restrict_causal_embedded_patch_cell_averages(
                            state,
                            layouts[layout_label],
                        )
                        for state in response
                    ]
                )
                histories.append(restricted)
                decisive[f"{layout_label}__{case}__state_response"] = (
                    restricted
                )
            metrics = _state_metrics(
                tuple(histories),
                parent_grid,
                field_scales,
                gates,
            )
            reports[case] = metrics
            all_passed = all_passed and metrics["passed"]
    return (
        {
            "passed": bool(all_passed),
            "case_count": len(reports),
            "reports": reports,
            "worst_observed_rms_order": min(
                item["observed_rms_order"] for item in reports.values()
            ),
            "worst_observed_maximum_order": min(
                item["observed_maximum_order"]
                for item in reports.values()
            ),
            "worst_component_order": min(
                item["minimum_significant_component_order"]
                for item in reports.values()
            ),
            "largest_fine_normalized_difference": max(
                item["maximum_fine_normalized_difference"]
                for item in reports.values()
            ),
            "minimum_history_cosine": min(
                item["history_cosine"] for item in reports.values()
            ),
            "minimum_refinement_error_cosine": min(
                item["refinement_error_cosine"]
                for item in reports.values()
            ),
        },
        decisive,
    )


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": _source_identity(),
                "completed": [],
                "maximum_local_block_ledger_defect": 0.0,
                "maximum_source_double_count_defect": 0.0,
                "maximum_shared_conservative_face_defect": 0.0,
                "maximum_split_closure_defect": 0.0,
                "maximum_incoming_excision_characteristics": 0,
            },
            {},
        )
    progress = _read_json(CHECKPOINT_JSON)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != _source_identity()
    ):
        raise RuntimeError("saved c3b2b progress belongs to different code")
    arrays = (
        _load_npz(CHECKPOINT_ARRAYS)
        if CHECKPOINT_ARRAYS.exists()
        else {}
    )
    return progress, arrays


def _save_progress(
    progress: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _direct_observable(context, state, coupling_face: int):
    ledger = causal_five_field_radial_candidate_ledger(context, state)
    evaluation = SimpleNamespace(
        stationary_ledger=ledger,
        residual_rows=ledger.residual_rows,
        cooling_rows=ledger.cooling_rows,
        lower_height_work_rows=ledger.lower_height_work_rows,
    )
    values = causal_embedded_active_direct_observables(
        evaluation,
        coupling_face,
    )
    audit = {
        "local_block_ledger_defect": ledger.local_block_ledger_defect,
        "source_double_count_defect": ledger.source_double_count_defect,
        "shared_conservative_face_defect": (
            ledger.interfaces.shared_conservative_face_defect
        ),
        "split_closure_defect": (
            ledger.interfaces.maximum_split_closure_defect
        ),
        "incoming_excision_characteristics": (
            ledger.interfaces.incoming_excision_characteristics
        ),
    }
    return values, audit


def _export_histories(
    input_arrays: dict[str, dict[str, np.ndarray]],
    layouts: dict,
    configurations: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    progress, arrays = _load_progress()
    completed = set(progress["completed"])
    tasks = []
    for layout_label in LAYOUTS:
        for time_index in range(TIMES.size):
            tasks.append(
                (
                    f"{layout_label}__base__t{time_index}",
                    layout_label,
                    input_arrays["base"][
                        f"{layout_label}__states"
                    ][time_index],
                )
            )
        for profile in PROFILES:
            for multiplier in VARIANT_MULTIPLIERS:
                case = _case_id(profile, multiplier)
                history = _perturbed_history(
                    input_arrays,
                    layout_label,
                    profile,
                    multiplier,
                )
                for time_index, state in enumerate(history):
                    tasks.append(
                        (
                            f"{layout_label}__{case}__t{time_index}",
                            layout_label,
                            state,
                        )
                    )
    total = len(tasks)
    for index, (task_id, layout_label, state) in enumerate(tasks, start=1):
        if task_id in completed:
            continue
        started = time.perf_counter()
        coupling_face = layouts[layout_label].coupling_face_index
        values, audit = _direct_observable(
            configurations[layout_label]["context"],
            state,
            coupling_face,
        )
        arrays[task_id] = np.asarray(values, dtype=float)
        completed.add(task_id)
        progress["completed"] = sorted(completed)
        for key in (
            "maximum_local_block_ledger_defect",
            "maximum_source_double_count_defect",
            "maximum_shared_conservative_face_defect",
            "maximum_split_closure_defect",
        ):
            source = key.removeprefix("maximum_")
            progress[key] = max(progress[key], float(audit[source]))
        progress["maximum_incoming_excision_characteristics"] = max(
            progress["maximum_incoming_excision_characteristics"],
            int(audit["incoming_excision_characteristics"]),
        )
        _save_progress(progress, arrays)
        print(
            f"c3b2b: {index}/{total} {task_id} "
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )
    if len(completed) != total:
        raise RuntimeError("direct observable checkpoint is incomplete")
    return arrays, progress


def _cumulative(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    result = np.zeros_like(values)
    increments = 0.5 * np.diff(TIMES)[:, None] * (
        values[:-1] + values[1:]
    )
    result[1:] = np.cumsum(increments, axis=0)
    return result


def _export_gate(
    cached: dict[str, np.ndarray],
    observable_scales: np.ndarray,
    gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    decisive = {
        "fixed_physical_observable_scales": observable_scales,
    }
    all_passed = True
    for profile in PROFILES:
        for multiplier in VARIANT_MULTIPLIERS:
            case = _case_id(profile, multiplier)
            histories = []
            cumulative = []
            for layout_label in LAYOUTS:
                base = np.asarray(
                    [
                        cached[f"{layout_label}__base__t{index}"]
                        for index in range(TIMES.size)
                    ]
                )
                perturbed = np.asarray(
                    [
                        cached[
                            f"{layout_label}__{case}__t{index}"
                        ]
                        for index in range(TIMES.size)
                    ]
                )
                response = perturbed - base
                integrated = _cumulative(response)
                histories.append(response)
                cumulative.append(integrated)
                decisive[
                    f"{layout_label}__{case}__instantaneous_export_response"
                ] = response
                decisive[
                    f"{layout_label}__{case}__cumulative_export_response"
                ] = integrated
            instantaneous_metric = _packet_metrics(
                tuple(histories),
                observable_scales,
                gates,
            )
            cumulative_metric = _packet_metrics(
                tuple(cumulative),
                observable_scales * float(TIMES[-1]),
                gates,
            )
            report = {
                "instantaneous": _metric_payload(instantaneous_metric),
                "cumulative": _metric_payload(cumulative_metric),
                "passed": bool(
                    instantaneous_metric.passed
                    and cumulative_metric.passed
                ),
            }
            reports[case] = report
            all_passed = all_passed and report["passed"]

    def values(channel: str, metric: str):
        return [
            report[channel][metric] for report in reports.values()
        ]

    return (
        {
            "passed": bool(all_passed),
            "case_count": len(reports),
            "reports": reports,
            "worst_instantaneous_rms_order": min(
                values("instantaneous", "observed_rms_order")
            ),
            "worst_instantaneous_maximum_order": min(
                values("instantaneous", "observed_maximum_order")
            ),
            "worst_instantaneous_component_order": min(
                values(
                    "instantaneous",
                    "minimum_significant_component_order",
                )
            ),
            "largest_instantaneous_fine_normalized_difference": max(
                values(
                    "instantaneous",
                    "maximum_fine_normalized_difference",
                )
            ),
            "minimum_instantaneous_history_cosine": min(
                values("instantaneous", "history_cosine")
            ),
            "minimum_instantaneous_error_cosine": min(
                values("instantaneous", "refinement_error_cosine")
            ),
            "worst_cumulative_rms_order": min(
                values("cumulative", "observed_rms_order")
            ),
            "worst_cumulative_maximum_order": min(
                values("cumulative", "observed_maximum_order")
            ),
            "worst_cumulative_component_order": min(
                values(
                    "cumulative",
                    "minimum_significant_component_order",
                )
            ),
            "largest_cumulative_fine_normalized_difference": max(
                values(
                    "cumulative",
                    "maximum_fine_normalized_difference",
                )
            ),
            "minimum_cumulative_history_cosine": min(
                values("cumulative", "history_cosine")
            ),
            "minimum_cumulative_error_cosine": min(
                values("cumulative", "refinement_error_cosine")
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
    global_summary = _read_json(CANONICAL_SUMMARY)
    global_summary.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    global_summary["latest_work_package"] = WORK_PACKAGE
    global_summary["latest_source_parent_commit"] = ANALYZED_BASE_COMMIT
    global_summary["case_count"] = len(global_summary["artifacts"])
    _write_json(CANONICAL_SUMMARY, global_summary)


def _report(summary: dict) -> str:
    state = summary["state_response"]
    exports = summary.get("tier_I_exports")
    lines = [
        "# Nonlinear short-horizon spatial/export pilot "
        "WP10c9d6c7c3b2b",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "This package reuses the certified BDF1 plus three-BDF2 "
        "histories. It changes no operator and launches no new nonlinear "
        "trajectory.",
        "",
        "## Restricted state response",
        "",
        f"- all 16 controls pass: `{state['passed']}`",
        "- worst RMS/maximum/field order: "
        f"`{state['worst_observed_rms_order']:.6f}` / "
        f"`{state['worst_observed_maximum_order']:.6f}` / "
        f"`{state['worst_component_order']:.6f}`",
        "- largest fine normalized difference: "
        f"`{state['largest_fine_normalized_difference']:.6e}`",
        "- minimum history/error cosine: "
        f"`{state['minimum_history_cosine']:.9f}` / "
        f"`{state['minimum_refinement_error_cosine']:.9f}`",
        "",
    ]
    if exports is not None:
        lines.extend(
            [
                "## Tier-I physical exports",
                "",
                f"- all 16 controls pass: `{exports['passed']}`",
                "- worst instantaneous RMS/maximum/component order: "
                f"`{exports['worst_instantaneous_rms_order']:.6f}` / "
                f"`{exports['worst_instantaneous_maximum_order']:.6f}` / "
                f"`{exports['worst_instantaneous_component_order']:.6f}`",
                "- minimum instantaneous history/error cosine: "
                f"`{exports['minimum_instantaneous_history_cosine']:.9f}` / "
                f"`{exports['minimum_instantaneous_error_cosine']:.9f}`",
                "- worst cumulative RMS/maximum/component order: "
                f"`{exports['worst_cumulative_rms_order']:.6f}` / "
                f"`{exports['worst_cumulative_maximum_order']:.6f}` / "
                f"`{exports['worst_cumulative_component_order']:.6f}`",
                "- minimum cumulative history/error cosine: "
                f"`{exports['minimum_cumulative_history_cosine']:.9f}` / "
                f"`{exports['minimum_cumulative_error_cosine']:.9f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "The result covers only five saved times through `4e-5 s` at "
            "one timestep. It does not certify temporal convergence, "
            "long-time nonlinear physics, Tier-II interface scattering, "
            "fixed-Q averaging, or reduced slow evolution.",
            "",
            "## Authorized next",
            "",
            (
                f"`{summary['authorized_next']}`"
                if summary["authorized_next"] is not None
                else "None."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parent_summary, parent_manifest = _validate_parent()
    input_arrays = _input_arrays()
    parent_grid, layouts, configurations = _layouts_and_contexts(
        input_arrays
    )
    gates = parent_manifest["tier_I_binding_contract"]["gates"]
    state_report, decisive = _state_gate(
        input_arrays,
        layouts,
        parent_grid,
        gates,
    )

    export_report = None
    export_progress = None
    if state_report["passed"]:
        cached, export_progress = _export_histories(
            input_arrays,
            layouts,
            configurations,
        )
        export_report, export_arrays = _export_gate(
            cached,
            np.asarray(
                input_arrays["c7a"][
                    "fixed_physical_observable_scales"
                ],
                dtype=float,
            ),
            gates,
        )
        decisive.update(export_arrays)

    passed = bool(
        state_report["passed"]
        and export_report is not None
        and export_report["passed"]
    )
    if passed:
        classification = (
            "nonlinear_short_horizon_state_and_tier_I_export_spatial_"
            "pilot_certified_temporal_refinement_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b3a_nonlinear_temporal_refinement_"
            "pilot_manifest"
        )
    elif not state_report["passed"]:
        classification = (
            "nonlinear_short_horizon_state_spatial_pilot_failed_"
            "export_evaluation_stopped"
        )
        authorized_next = None
    else:
        classification = (
            "nonlinear_short_horizon_state_passed_tier_I_export_"
            "spatial_pilot_failed"
        )
        authorized_next = None

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "profiles": list(PROFILES),
        "variant_multipliers": list(VARIANT_MULTIPLIERS),
        "times_seconds": TIMES.tolist(),
        "observable_names": list(OBSERVABLE_NAMES),
        "tier_I_gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    input_paths = {
        "manifest_summary": MANIFEST_DIRECTORY / "summary.json",
        "manifest": (
            MANIFEST_DIRECTORY / "nonlinear_spatial_export_manifest.json"
        ),
        "base_arrays": BASE_DIRECTORY / "decisive_arrays.npz",
        "step1_arrays": STEP1_DIRECTORY / "decisive_arrays.npz",
        "step2_arrays": STEP2_DIRECTORY / "decisive_arrays.npz",
        "step3_arrays": STEP3_DIRECTORY / "decisive_arrays.npz",
        "step4_arrays": STEP4_DIRECTORY / "decisive_arrays.npz",
        "c7a_arrays": C7A_DIRECTORY / "decisive_arrays.npz",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
        "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
        "passed": passed,
        "classification": classification,
        "authorized_next": authorized_next,
        "operator_changed": False,
        "new_propagation_executed": False,
        "canonical_saved_history_reused": True,
        "state_response": state_report,
        "tier_I_exports": export_report,
        "export_ledger_audit": (
            None
            if export_progress is None
            else {
                "completed_count": len(export_progress["completed"]),
                "maximum_local_block_ledger_defect": export_progress[
                    "maximum_local_block_ledger_defect"
                ],
                "maximum_source_double_count_defect": export_progress[
                    "maximum_source_double_count_defect"
                ],
                "maximum_shared_conservative_face_defect": export_progress[
                    "maximum_shared_conservative_face_defect"
                ],
                "maximum_split_closure_defect": export_progress[
                    "maximum_split_closure_defect"
                ],
                "maximum_incoming_excision_characteristics": (
                    export_progress[
                        "maximum_incoming_excision_characteristics"
                    ]
                ),
            }
        ),
        "tier_II_status": "diagnostic_only_nonpromoted",
        "temporal_convergence_certified": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "parent_classification": parent_summary["classification"],
        "config_sha256": _sha256(CONFIG_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "implementation_source_hashes": _source_identity(),
        "input_hashes": {
            name: _sha256(path) for name, path in input_paths.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src:scripts "
                "/Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "implementation_parent_commit": _git_value(
                "rev-parse", "HEAD"
            ),
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
            "implementation_source_hashes": summary[
                "implementation_source_hashes"
            ],
            "input_hashes": summary["input_hashes"],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = (
        "config.json",
        "decisive_arrays.npz",
        "summary.json",
        "provenance.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}"
            for name in names
        )
        + "\n",
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
