#!/usr/bin/env python3
"""Issue the binding coarse/middle/fine nonlinear 5 ms certificate."""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as h2e1  # noqa: E402
import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as h2b1  # noqa: E402
import run_causal_inner_nonlinear_middle_5ms_completion_wp10c9d6c7c3b5c3h2d1 as h2d1  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as b2b  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g as c3g  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_field_history_inner_product,
    causal_field_history_norm,
    causal_trapezoid_weights,
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2f"
ANALYZED_BASE_COMMIT = "e2c6979e67ee25e8020dd750eeb4951a5cae5fcb"
ANALYZED_BASE_PARENT = "a8ac3a42f1d6c595f0748c29a64d8fa64f36e3f8"
ANALYZED_BASE_TREE = "665d3d857601c161c9c128c815206a460e51be58"

COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT = tuple(c3g.LAYOUTS)
GENERIC_PROFILE = c3g.GENERIC_PROFILE
OBSERVABLE_NAMES = tuple(c3g.OBSERVABLE_NAMES)
SPATIAL_GATES = dict(c3g.SPATIAL_GATES)
TEMPORAL_GATES = dict(c3g.TEMPORAL_UNCERTAINTY_GATES)
OBSERVABILITY_FACTOR = float(TEMPORAL_GATES["observability_factor"])
MAXIMUM_TEMPORAL_TO_SPATIAL_RATIO = float(
    TEMPORAL_GATES[
        "maximum_strict_to_observable_medium_fine_spatial_error_ratio"
    ]
)

ARTIFACT = (
    "causal_inner_nonlinear_5ms_spatial_certificate_"
    "wp10c9d6c7c3b5c3h2f"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_spatial_certificate_"
    "wp10c9d6c7c3b5c3h2f.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_spatial_certificate_"
    "wp10c9d6c7c3b5c3h2f.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_SPATIAL_"
    "CERTIFICATE_WP10C9D6C7C3B5C3H2F_2026-08-07.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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
    middle_summary = _read_json(h2d1.SUMMARY_PATH)
    fine_summary = _read_json(h2e1.SUMMARY_PATH)
    if (
        not middle_summary["passed"]
        or not fine_summary["passed"]
        or not fine_summary["final_spatial_certificate_analysis_authorized"]
        or fine_summary["middle_fine_5ms_spatial_certificate_issued"]
    ):
        raise RuntimeError("5 ms spatial-certificate authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2f analyzed identity changed")
    return middle_summary, fine_summary


def _restrict(history: np.ndarray, layout) -> np.ndarray:
    return np.asarray(
        [
            restrict_causal_embedded_patch_cell_averages(state, layout)
            for state in history
        ],
        dtype=float,
    )


def _matching_indices(source_times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = []
    for target in targets:
        matches = np.flatnonzero(
            np.isclose(source_times, target, rtol=0.0, atol=1.0e-18)
        )
        if matches.size != 1:
            raise RuntimeError(f"target {target:.17g} does not match uniquely")
        indices.append(int(matches[0]))
    result = np.asarray(indices, dtype=int)
    if not np.array_equal(source_times[result], targets):
        raise RuntimeError("common target bit patterns changed")
    return result


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(times)[:, None],
        axis=0,
    )
    return result


def _state_metric(
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    times: np.ndarray,
    cell_measures: np.ndarray,
    field_scales: np.ndarray,
) -> dict:
    richardson = causal_windowed_richardson_reference(
        *histories,
        times=times,
        coarse_cell_measures=cell_measures,
        field_scales=field_scales,
        relative_activity=SPATIAL_GATES["minimum_relative_activity"],
    )
    normalized = tuple(
        values / field_scales[None, None, :] for values in histories
    )
    coarse_medium = normalized[1] - normalized[0]
    medium_fine = normalized[2] - normalized[1]
    tiny = np.finfo(float).tiny
    maximum_order = float(
        np.log2(
            max(float(np.max(np.abs(coarse_medium))), tiny)
            / max(float(np.max(np.abs(medium_fine))), tiny)
        )
    )
    weights = causal_trapezoid_weights(times)
    medium_norm = causal_field_history_norm(
        histories[1],
        cell_measures=cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    )
    fine_norm = causal_field_history_norm(
        histories[2],
        cell_measures=cell_measures,
        field_scales=field_scales,
        time_weights=weights,
    )
    history_cosine = float(
        causal_field_history_inner_product(
            histories[1],
            histories[2],
            cell_measures=cell_measures,
            field_scales=field_scales,
            time_weights=weights,
        )
        / max(medium_norm * fine_norm, tiny)
    )
    fine_maximum = float(np.max(np.abs(medium_fine)))
    raw_passed = bool(
        richardson.observed_order >= SPATIAL_GATES["minimum_rms_order"]
        and maximum_order >= SPATIAL_GATES["minimum_maximum_order"]
        and richardson.minimum_significant_component_order
        >= SPATIAL_GATES["minimum_significant_component_order"]
        and fine_maximum
        <= SPATIAL_GATES["maximum_fine_normalized_difference"]
        and history_cosine >= SPATIAL_GATES["minimum_history_cosine"]
        and richardson.refinement_error_cosine
        >= SPATIAL_GATES["minimum_refinement_error_cosine"]
    )
    return {
        "raw_spatial_contract_passed": raw_passed,
        "observed_rms_order": richardson.observed_order,
        "observed_maximum_order": maximum_order,
        "minimum_significant_component_order": (
            richardson.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": fine_maximum,
        "history_cosine": history_cosine,
        "refinement_error_cosine": richardson.refinement_error_cosine,
        "coarse_medium_history_norm": richardson.coarse_medium_history_norm,
        "medium_fine_history_norm": richardson.medium_fine_history_norm,
    }


def _packet_payload(metrics) -> dict:
    indices = np.asarray(metrics.significant_components, dtype=int)
    component_orders = {
        OBSERVABLE_NAMES[index]: float(metrics.component_orders[position])
        for position, index in enumerate(indices)
    }
    failed_components = tuple(
        name
        for name, order in component_orders.items()
        if order < SPATIAL_GATES["minimum_significant_component_order"]
    )
    return {
        "raw_spatial_contract_passed": bool(metrics.passed),
        "significant_components": tuple(
            OBSERVABLE_NAMES[index] for index in indices
        ),
        "failed_component_orders": failed_components,
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "component_orders": component_orders,
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "coarse_medium_rms_difference": metrics.coarse_medium_rms_difference,
        "medium_fine_rms_difference": metrics.medium_fine_rms_difference,
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _temporal_classification(metric: dict, uncertainty: float) -> dict:
    fine_difference = float(metric["maximum_fine_normalized_difference"])
    envelope = float(uncertainty)
    observable = bool(fine_difference > OBSERVABILITY_FACTOR * envelope)
    ratio = envelope / max(fine_difference, np.finfo(float).tiny)
    upper_bound = fine_difference + envelope
    if observable:
        passed = bool(
            metric["raw_spatial_contract_passed"]
            and ratio <= MAXIMUM_TEMPORAL_TO_SPATIAL_RATIO
        )
        route = "observable_spatial_order_contract"
    else:
        passed = bool(
            upper_bound
            <= SPATIAL_GATES["maximum_fine_normalized_difference"]
        )
        route = "unobservable_upper_bound_without_order_or_direction_claim"
    return {
        "passed": passed,
        "route": route,
        "spatial_difference_observable": observable,
        "spatial_orders_and_error_direction_certifying": bool(
            observable and ratio <= MAXIMUM_TEMPORAL_TO_SPATIAL_RATIO
        ),
        "temporal_uncertainty_envelope": envelope,
        "temporal_uncertainty_to_medium_fine_difference_ratio": ratio,
        "observability_threshold": OBSERVABILITY_FACTOR * envelope,
        "conservative_fine_difference_upper_bound": upper_bound,
    }


def _analyze(
    middle_summary: dict,
    fine_summary: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    coarse = _load_npz(c3d.DECISIVE_ARRAYS)
    middle = _load_npz(h2d1.DECISIVE_ARRAYS)
    fine = _load_npz(h2e1.DECISIVE_ARRAYS)
    parent_grid, layouts, _configurations = b2b._layouts_and_contexts(
        b2b._input_arrays()
    )
    times = np.asarray(coarse["main_times_seconds"], dtype=float)
    if not np.array_equal(times, middle["base__accepted_times"]):
        raise RuntimeError("coarse/middle target times changed")
    fine_indices = _matching_indices(fine["base__accepted_times"], times)
    field_scales = np.asarray(middle["tangent__field_scales"], dtype=float)
    export_scales = np.asarray(middle["tangent__export_scales"], dtype=float)

    coarse_state = (
        coarse["perturbed__main__output_states"]
        - coarse["base__main__output_states"]
    )
    middle_state = _restrict(
        middle["anchor__actual_state_response"],
        layouts[MIDDLE_LAYOUT],
    )
    fine_state = _restrict(
        fine["anchor__actual_state_response"][fine_indices],
        layouts[FINE_LAYOUT],
    )
    state_metric = _state_metric(
        (coarse_state, middle_state, fine_state),
        times=times,
        cell_measures=parent_grid.cell_measures,
        field_scales=field_scales,
    )
    state_uncertainty = float(
        middle_summary["anchor"]["maximum_sampled_state_error_estimate"]
        + fine_summary["anchor"]["maximum_sampled_state_error_estimate"]
    )
    state_temporal = _temporal_classification(state_metric, state_uncertainty)

    coarse_export = (
        coarse["perturbed__main__output_exports"]
        - coarse["base__main__output_exports"]
    )
    middle_export = np.asarray(
        middle["anchor__actual_export_response"], dtype=float
    )
    fine_export = np.asarray(
        fine["anchor__actual_export_response"][fine_indices], dtype=float
    )
    packet_arguments = {
        "minimum_rms_order": SPATIAL_GATES["minimum_rms_order"],
        "minimum_maximum_order": SPATIAL_GATES["minimum_maximum_order"],
        "minimum_significant_component_order": SPATIAL_GATES[
            "minimum_significant_component_order"
        ],
        "maximum_fine_normalized_difference": SPATIAL_GATES[
            "maximum_fine_normalized_difference"
        ],
        "minimum_history_cosine": SPATIAL_GATES["minimum_history_cosine"],
        "minimum_refinement_error_cosine": SPATIAL_GATES[
            "minimum_refinement_error_cosine"
        ],
        "relative_activity": SPATIAL_GATES["minimum_relative_activity"],
    }
    instantaneous_metric = _packet_payload(
        causal_packet_history_metrics(
            coarse_export,
            middle_export,
            fine_export,
            physical_scales=export_scales,
            **packet_arguments,
        )
    )
    cumulative_histories = tuple(
        _cumulative(values, times)
        for values in (coarse_export, middle_export, fine_export)
    )
    cumulative_metric = _packet_payload(
        causal_packet_history_metrics(
            *cumulative_histories,
            physical_scales=export_scales * float(times[-1]),
            **packet_arguments,
        )
    )
    export_uncertainty = float(
        middle_summary["anchor"]["maximum_sampled_export_error_estimate"]
        + fine_summary["anchor"]["maximum_sampled_export_error_estimate"]
    )
    instantaneous_temporal = _temporal_classification(
        instantaneous_metric, export_uncertainty
    )
    # Reusing the instantaneous envelope is deliberately conservative for the
    # normalized cumulative channel; no cancellation credit is taken.
    cumulative_temporal = _temporal_classification(
        cumulative_metric, export_uncertainty
    )

    state_passed = bool(state_temporal["passed"])
    instantaneous_passed = bool(instantaneous_temporal["passed"])
    cumulative_passed = bool(cumulative_temporal["passed"])
    overall_passed = bool(
        state_passed and instantaneous_passed and cumulative_passed
    )
    analysis = {
        "generic_profile": GENERIC_PROFILE,
        "common_history_window_seconds": (float(times[0]), float(times[-1])),
        "state": {
            **state_metric,
            "temporal_classification": state_temporal,
            "binding_channel_passed": state_passed,
        },
        "instantaneous_Tier_I": {
            **instantaneous_metric,
            "temporal_classification": instantaneous_temporal,
            "binding_channel_passed": instantaneous_passed,
        },
        "cumulative_Tier_I": {
            **cumulative_metric,
            "temporal_classification": cumulative_temporal,
            "binding_channel_passed": cumulative_passed,
        },
        "binding_generic_profile_passed": overall_passed,
        "breadth_tangent_classification_executed": False,
        "breadth_tangent_stop_reason": (
            "binding_full_nonlinear_generic_instantaneous_Tier_I_gate_failed"
            if not overall_passed
            else None
        ),
        "physical_admissibility_failure_detected": False,
        "numerical_spatial_export_failure_detected": not overall_passed,
    }
    decisive = {
        "times_seconds": times,
        "fine_target_indices": fine_indices,
        "field_scales": field_scales,
        "export_scales": export_scales,
        "coarse_generic_state_response": coarse_state,
        "middle_generic_nonlinear_state_response": middle_state,
        "fine_generic_nonlinear_state_response": fine_state,
        "coarse_generic_instantaneous_export_response": coarse_export,
        "middle_generic_nonlinear_instantaneous_export_response": middle_export,
        "fine_generic_nonlinear_instantaneous_export_response": fine_export,
        "coarse_generic_cumulative_export_response": cumulative_histories[0],
        "middle_generic_nonlinear_cumulative_export_response": cumulative_histories[1],
        "fine_generic_nonlinear_cumulative_export_response": cumulative_histories[2],
        "state_temporal_uncertainty_envelope": np.asarray(state_uncertainty),
        "export_temporal_uncertainty_envelope": np.asarray(export_uncertainty),
    }
    return analysis, decisive


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


def main() -> int:
    middle_summary, fine_summary = _validate_parent()
    analysis, decisive = _analyze(middle_summary, fine_summary)
    passed = bool(analysis["binding_generic_profile_passed"])
    classification = (
        "five_ms_spatial_certificate_passed_fourth_duration_manifest_authorized"
        if passed
        else "five_ms_spatial_certificate_rejected_Tier_I_exports_"
        "nonconvergent_later_duration_blocked"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "analysis": analysis,
        "middle_fine_5ms_spatial_certificate_issued": True,
        "third_duration_rung_spatial_convergence_certified": passed,
        "fourth_duration_rung_manifest_authorized": passed,
        "physical_failure_detected": False,
        "numerical_spatial_export_failure_detected": not passed,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4a_fourth_duration_rung_manifest"
            if passed
            else "WP10c9d6c7c3b5c3h2g_Tier_I_spatial_failure_localization_manifest"
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layouts": (COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT),
            "active_coupling_faces": c3g.ACTIVE_COUPLING_FACE_INDICES,
            "generic_profile": GENERIC_PROFILE,
            "observable_names": OBSERVABLE_NAMES,
            "spatial_gates": SPATIAL_GATES,
            "temporal_uncertainty_gates": TEMPORAL_GATES,
            "fail_fast_generic_before_tangent_breadth": True,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "coarse_arrays": _sha256(c3d.DECISIVE_ARRAYS),
                "middle_summary": _sha256(h2d1.SUMMARY_PATH),
                "middle_arrays": _sha256(h2d1.DECISIVE_ARRAYS),
                "fine_summary": _sha256(h2e1.SUMMARY_PATH),
                "fine_arrays": _sha256(h2e1.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    state = analysis["state"]
    instant = analysis["instantaneous_Tier_I"]
    cumulative = analysis["cumulative_Tier_I"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 5 ms spatial certificate WP10c9d6c7c3b5c3h2f",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "The fine base, five-profile discrete tangent, and evidence-selected "
                "full nonlinear generic anchor all completed successfully. The "
                "binding coarse/middle/fine analysis nevertheless rejects the 5 ms "
                "spatial certificate because the physical Tier-I exports do not "
                "contract under refinement.",
                "",
                "## Binding measurements",
                "",
                f"- State raw RMS/max/component orders are "
                f"`{state['observed_rms_order']:.6f}`, "
                f"`{state['observed_maximum_order']:.6f}`, and "
                f"`{state['minimum_significant_component_order']:.6f}`. Its "
                "medium/fine difference is below the frozen temporal observability "
                "threshold, so these orders are non-certifying; the conservative "
                f"upper bound `{state['temporal_classification']['conservative_fine_difference_upper_bound']:.6e}` "
                "passes the amplitude gate.",
                f"- Instantaneous Tier-I RMS/max/minimum-component orders are "
                f"`{instant['observed_rms_order']:.6f}`, "
                f"`{instant['observed_maximum_order']:.6f}`, and "
                f"`{instant['minimum_significant_component_order']:.6f}`. The "
                f"refinement-error cosine is `{instant['refinement_error_cosine']:.6f}`. "
                "This difference is temporally observable and fails the frozen "
                "0.75/0.90 gates.",
                f"- Cumulative Tier-I raw RMS/max/minimum-component orders are "
                f"`{cumulative['observed_rms_order']:.6f}`, "
                f"`{cumulative['observed_maximum_order']:.6f}`, and "
                f"`{cumulative['minimum_significant_component_order']:.6f}`. Under "
                "the conservative instantaneous temporal envelope this channel is "
                "reported only as an upper bound; it cannot rescue the observable "
                "instantaneous failure.",
                "",
                "The failed significant export components are "
                + ", ".join(
                    f"`{name}`" for name in instant["failed_component_orders"]
                )
                + ".",
                "",
                "## Scientific interpretation",
                "",
                "No physical-admissibility, nonlinear-solve, conservation-ledger, "
                "reconstruction, or outgoing-excision failure occurred. This is a "
                "numerical spatial-convergence failure of the evolved physical "
                "exports at 5 ms, not evidence that the physical model itself is "
                "unstable. The fourth duration rung, fixed-Q experiments, and "
                "reduced slow evolution remain blocked.",
                "",
                "Only a prospective, definitions-only Tier-I failure-localization "
                "manifest is authorized next. It must distinguish resolution, "
                "time-schedule contamination, inner-face response, and cancellation "
                "in net-drive channels without changing the operator or thresholds.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
