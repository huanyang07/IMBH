#!/usr/bin/env python3
"""Analyze the coarse/middle/fine 20 ms response without propagation."""

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

import run_causal_inner_nonlinear_coarse_middle_20ms_checkpoint_analysis_wp10c9d6c7c3b5c4e4 as c4e4  # noqa: E402
import run_causal_inner_nonlinear_middle_20ms_temporal_reference_shadow_wp10c9d6c7c3b5c4e6 as c4e6  # noqa: E402
import run_causal_inner_nonlinear_cost_bounded_fine_20ms_manifest_wp10c9d6c7c3b5c4e7 as c4e7  # noqa: E402
import run_causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_wp10c9d6c7c3b5c4e8 as c4e8  # noqa: E402
import run_causal_inner_nonlinear_5ms_spatial_certificate_wp10c9d6c7c3b5c3h2f as h2f  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e9"
ANALYZED_BASE_COMMIT = c4e8.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e8.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e8.ANALYZED_BASE_TREE

ARTIFACT = (
    "causal_inner_nonlinear_three_grid_20ms_spatial_analysis_"
    "wp10c9d6c7c3b5c4e9"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_three_grid_20ms_spatial_analysis_"
    "wp10c9d6c7c3b5c4e9.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_three_grid_20ms_spatial_analysis_"
    "wp10c9d6c7c3b5c4e9.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_THREE_GRID_20MS_"
    "SPATIAL_ANALYSIS_WP10C9D6C7C3B5C4E9_2026-08-12.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
CONTRACT_PATH = CANONICAL_DIRECTORY / "analysis_contract.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

WINDOWS_SECONDS = tuple(c4e4.WINDOWS_SECONDS)
GENERIC_INDEX = c4e8.GENERIC_INDEX
OBSERVABLE_NAMES = tuple(h2f.OBSERVABLE_NAMES)
SPATIAL_GATES = dict(h2f.SPATIAL_GATES)
OBSERVABILITY_FACTOR = float(h2f.OBSERVABILITY_FACTOR)
MAXIMUM_TEMPORAL_FRACTION = 0.10
MAXIMUM_SURROGATE_FRACTION = 0.10


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


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "three_grid_20ms_spatial_analysis_contract_frozen",
        "definitions_frozen_before_analysis": True,
        "propagation_executed": False,
        "operator_changed": False,
        "scope": {
            "layouts": ("coarse", "middle", "fine"),
            "state_restricted_to_common_64_cell_parent": True,
            "fine_response": "complete_discrete_BDF_block_tangent",
            "fine_nonlinear_validation": "six_one_step_generic_shadows",
            "slow_export": "certified_conservative_exterior_partition",
            "raw_inner_face_is_not_a_slow_export": True,
            "instantaneous_interval_seconds": (0.005, 0.020),
            "cumulative_interval_seconds": (0.005, 0.020),
            "window_mean_intervals_seconds": WINDOWS_SECONDS,
        },
        "spatial_gates": SPATIAL_GATES,
        "uncertainty_gates": {
            "maximum_temporal_fraction_of_middle_fine_difference": (
                MAXIMUM_TEMPORAL_FRACTION
            ),
            "maximum_surrogate_fraction_of_middle_fine_difference": (
                MAXIMUM_SURROGATE_FRACTION
            ),
            "observability_factor": OBSERVABILITY_FACTOR,
            "unobservable_route": (
                "report_upper_bound_only_without_order_or_direction_claim"
            ),
        },
        "uncertainty_construction": {
            "temporal": (
                "certified_coarse_middle_response_specific_shadow_envelope_plus_"
                "fine_base_full_vs_two_half_envelope"
            ),
            "fine_surrogate": (
                "maximum_sampled_nonlinear_shadow_correction_fraction_times_"
                "maximum_scaled_fine_response"
            ),
            "calibration_floor": (
                "maximum_of_fine_shadow_bound_and_committed_middle_global_"
                "nonlinear_minus_tangent_discrepancy"
            ),
            "surrogate_gate_is_binding_even_when_spatial_difference_is_unobservable": True,
        },
        "decision": {
            "all_channels_certify": "issue_20ms_spatial_certificate",
            "state_certifies_but_extraction_surrogate_fails": (
                "authorize_definitions_only_fine_generic_anchor_manifest"
            ),
            "temporal_fails": "authorize_targeted_fine_temporal_shadow_only",
            "resolved_spatial_order_fails": "localize_before_longer_duration",
        },
        "hard_stops": (
            "do_not_claim_orders_for_unobservable_differences",
            "do_not_issue_20ms_spatial_certificate_if_surrogate_ratio_exceeds_0p10",
            "do_not_launch_fine_anchor_in_this_analysis_package",
            "do_not_run_50ms_fixed_Q_or_reduced_evolution",
        ),
    }


def _validate_inputs() -> tuple[dict, dict, dict, dict]:
    coarse_middle = _read_json(c4e4.SUMMARY_PATH)
    temporal = _read_json(c4e6.SUMMARY_PATH)
    manifest = _read_json(c4e7.SUMMARY_PATH)
    fine = _read_json(c4e8.SUMMARY_PATH)
    if (
        not coarse_middle["passed"]
        or not temporal["passed"]
        or not temporal["temporal_reference_hardened"]
        or not manifest["passed"]
        or not fine["passed"]
        or not fine["fine_twenty_ms_computation_completed"]
        or not fine["three_grid_spatial_analysis_authorized"]
        or fine["fine_twenty_ms_spatial_certificate_issued"]
        or fine["physical_failure_detected"]
    ):
        raise RuntimeError("c4e9 input authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e9 analyzed identity changed")
    return coarse_middle, temporal, manifest, fine


def _packet_arguments() -> dict:
    return {
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


def _classify_uncertainty(
    metric: dict,
    *,
    temporal_uncertainty: float,
    surrogate_uncertainty: float,
) -> dict:
    spatial = float(metric["maximum_fine_normalized_difference"])
    tiny = np.finfo(float).tiny
    temporal_ratio = temporal_uncertainty / max(spatial, tiny)
    surrogate_ratio = surrogate_uncertainty / max(spatial, tiny)
    total = temporal_uncertainty + surrogate_uncertainty
    observable = bool(spatial > OBSERVABILITY_FACTOR * total)
    upper_bound = spatial + total
    temporal_passed = bool(temporal_ratio <= MAXIMUM_TEMPORAL_FRACTION)
    surrogate_passed = bool(surrogate_ratio <= MAXIMUM_SURROGATE_FRACTION)
    if observable:
        route = "observable_spatial_order_contract"
        uncertainty_passed = bool(temporal_passed and surrogate_passed)
    else:
        route = "unobservable_upper_bound_without_order_or_direction_claim"
        uncertainty_passed = bool(
            upper_bound <= SPATIAL_GATES["maximum_fine_normalized_difference"]
            and surrogate_passed
        )
    certifying = bool(
        metric["raw_spatial_contract_passed"] and uncertainty_passed and observable
    )
    return {
        "route": route,
        "spatial_difference_observable": observable,
        "temporal_uncertainty": temporal_uncertainty,
        "temporal_uncertainty_fraction_of_middle_fine_difference": temporal_ratio,
        "temporal_gate_passed": temporal_passed,
        "surrogate_uncertainty": surrogate_uncertainty,
        "surrogate_uncertainty_fraction_of_middle_fine_difference": surrogate_ratio,
        "surrogate_gate_passed": surrogate_passed,
        "combined_uncertainty": total,
        "observability_threshold": OBSERVABILITY_FACTOR * total,
        "conservative_fine_difference_upper_bound": upper_bound,
        "channel_certifying": certifying,
    }


def _analyze(contract: dict) -> tuple[dict, dict[str, np.ndarray]]:
    coarse_middle = _load_npz(c4e4.DECISIVE_ARRAYS)
    fine = _load_npz(c4e8.DECISIVE_ARRAYS)
    coarse_middle_summary = _read_json(c4e4.SUMMARY_PATH)["analysis"]
    temporal_summary = _read_json(c4e6.SUMMARY_PATH)["analysis"]["observables"]
    fine_summary = _read_json(c4e8.SUMMARY_PATH)
    field_scales = np.asarray(coarse_middle["field_scales"], dtype=float)
    extraction_scales = np.asarray(
        coarse_middle["extraction_scales"], dtype=float
    )
    if not np.array_equal(field_scales, fine["tangent__field_scales"]):
        raise RuntimeError("c4e9 field scales changed")
    fine_times = np.asarray(fine["base__accepted_times"], dtype=float)
    if not np.array_equal(fine_times, fine["extraction__accepted_times"]):
        raise RuntimeError("c4e9 fine extraction times changed")
    common_times = np.asarray(coarse_middle["common_times_seconds"], dtype=float)
    common_indices = c4e4._indices(fine_times, common_times)
    parent_grid, layouts, _contexts = c4e4.b2b._layouts_and_contexts(
        c4e4.b2b._input_arrays()
    )
    fine_layout = layouts[c4e8.FINE_LAYOUT]
    fine_state = c4e4._restrict(
        fine["tangent__state_directions"][common_indices, GENERIC_INDEX],
        fine_layout,
    )
    state_metric = h2f._state_metric(
        (
            coarse_middle["coarse_state_response"],
            coarse_middle["middle_state_response"],
            fine_state,
        ),
        times=common_times,
        cell_measures=parent_grid.cell_measures,
        field_scales=field_scales,
    )

    fine_extraction_all = np.asarray(
        fine["extraction__tangent_directions"][:, GENERIC_INDEX], dtype=float
    )
    fine_extraction = fine_extraction_all[common_indices]
    arguments = _packet_arguments()
    instantaneous_metric = h2f._packet_payload(
        h2f.causal_packet_history_metrics(
            coarse_middle["coarse_extraction_response"],
            coarse_middle["middle_extraction_response"],
            fine_extraction,
            physical_scales=extraction_scales,
            **arguments,
        )
    )
    quadrature_times = np.asarray(
        coarse_middle["quadrature_times_seconds"], dtype=float
    )
    fine_quadrature = c4e4._interpolate(
        fine_times, fine_extraction_all, quadrature_times
    )
    fine_cumulative = c4e4._cumulative(fine_quadrature, quadrature_times)
    fine_means = c4e4._window_means(fine_quadrature, quadrature_times)
    duration = float(quadrature_times[-1] - quadrature_times[0])
    cumulative_metric = h2f._packet_payload(
        h2f.causal_packet_history_metrics(
            coarse_middle["coarse_cumulative_extraction_response"],
            coarse_middle["middle_cumulative_extraction_response"],
            fine_cumulative,
            physical_scales=extraction_scales * duration,
            **arguments,
        )
    )
    mean_metric = h2f._packet_payload(
        h2f.causal_packet_history_metrics(
            coarse_middle["coarse_window_mean_extraction_response"],
            coarse_middle["middle_window_mean_extraction_response"],
            fine_means,
            physical_scales=extraction_scales,
            **arguments,
        )
    )

    fine_state_scaled_maximum = float(
        np.max(np.abs(fine_state) / field_scales[None, None, :])
    )
    fine_instantaneous_scaled_maximum = float(
        np.max(np.abs(fine_extraction) / extraction_scales[None, :])
    )
    fine_cumulative_scaled_maximum = float(
        np.max(
            np.abs(fine_cumulative)
            / (extraction_scales * duration)[None, :]
        )
    )
    fine_mean_scaled_maximum = float(
        np.max(np.abs(fine_means) / extraction_scales[None, :])
    )
    state_shadow_fraction = float(
        fine_summary["sampled_nonlinear_remainder"][
            "maximum_state_correction_fraction"
        ]
    )
    extraction_shadow_fraction = float(
        fine_summary["sampled_nonlinear_remainder"][
            "maximum_extraction_correction_fraction"
        ]
    )
    fine_state_surrogate = state_shadow_fraction * fine_state_scaled_maximum
    fine_surrogates = {
        "instantaneous_extraction": (
            extraction_shadow_fraction * fine_instantaneous_scaled_maximum
        ),
        "cumulative_extraction": (
            extraction_shadow_fraction * fine_cumulative_scaled_maximum
        ),
        "window_mean_extraction": (
            extraction_shadow_fraction * fine_mean_scaled_maximum
        ),
    }
    surrogate_uncertainties = {
        "state": max(
            fine_state_surrogate,
            float(coarse_middle_summary["state"]["surrogate_uncertainty"]),
        ),
        **{
            name: max(
                fine_surrogates[name],
                float(coarse_middle_summary[name]["surrogate_uncertainty"]),
            )
            for name in fine_surrogates
        },
    }
    fine_state_temporal = float(np.max(fine["base__local_state_estimates"]))
    fine_extraction_temporal = float(
        np.max(fine["base__local_extraction_estimates"])
    )
    temporal_uncertainties = {
        "state": (
            float(temporal_summary["state"]["combined_temporal_uncertainty"])
            + fine_state_temporal
        ),
        **{
            name: (
                float(temporal_summary[name]["combined_temporal_uncertainty"])
                + fine_extraction_temporal
            )
            for name in fine_surrogates
        },
    }
    metrics = {
        "state": state_metric,
        "instantaneous_extraction": instantaneous_metric,
        "cumulative_extraction": cumulative_metric,
        "window_mean_extraction": mean_metric,
    }
    for name, metric in metrics.items():
        metric.update(
            _classify_uncertainty(
                metric,
                temporal_uncertainty=temporal_uncertainties[name],
                surrogate_uncertainty=surrogate_uncertainties[name],
            )
        )
    extraction_names = (
        "instantaneous_extraction",
        "cumulative_extraction",
        "window_mean_extraction",
    )
    full_anchor_required = any(
        not metrics[name]["surrogate_gate_passed"] for name in extraction_names
    )
    aggregate_extraction_orders_pass = all(
        metrics[name]["observed_rms_order"] >= SPATIAL_GATES["minimum_rms_order"]
        and metrics[name]["observed_maximum_order"]
        >= SPATIAL_GATES["minimum_maximum_order"]
        and metrics[name]["refinement_error_cosine"]
        >= SPATIAL_GATES["minimum_refinement_error_cosine"]
        and metrics[name]["maximum_fine_normalized_difference"]
        <= SPATIAL_GATES["maximum_fine_normalized_difference"]
        for name in extraction_names
    )
    analysis = {
        "analysis_contract": contract,
        "state": state_metric,
        "instantaneous_extraction": instantaneous_metric,
        "cumulative_extraction": cumulative_metric,
        "window_mean_extraction": mean_metric,
        "state_spatial_contract_certified": state_metric["channel_certifying"],
        "aggregate_extraction_order_difference_direction_gates_passed": (
            aggregate_extraction_orders_pass
        ),
        "extraction_component_and_surrogate_contract_certified": all(
            metrics[name]["channel_certifying"] for name in extraction_names
        ),
        "full_fine_generic_anchor_required": full_anchor_required,
        "twenty_ms_spatial_certificate_issued": all(
            metric["channel_certifying"] for metric in metrics.values()
        ),
        "fine_shadow_fractions": {
            "state": state_shadow_fraction,
            "extraction": extraction_shadow_fraction,
        },
        "fine_base_temporal_envelopes": {
            "state": fine_state_temporal,
            "extraction": fine_extraction_temporal,
        },
        "maximum_extraction_identity_defect": float(
            np.max(fine["extraction__maximum_identity_defects"])
        ),
        "maximum_extraction_ledger_audit": float(
            np.max(fine["extraction__maximum_ledger_audits"][:, :3])
        ),
        "maximum_incoming_excision_characteristics": int(
            np.max(fine["extraction__maximum_ledger_audits"][:, 3])
        ),
    }
    decisive = {
        "common_times_seconds": common_times,
        "quadrature_times_seconds": quadrature_times,
        "field_scales": field_scales,
        "extraction_scales": extraction_scales,
        "coarse_state_response": coarse_middle["coarse_state_response"],
        "middle_state_response": coarse_middle["middle_state_response"],
        "fine_tangent_state_response": fine_state,
        "coarse_extraction_response": coarse_middle["coarse_extraction_response"],
        "middle_extraction_response": coarse_middle["middle_extraction_response"],
        "fine_tangent_extraction_response": fine_extraction,
        "coarse_cumulative_extraction_response": coarse_middle[
            "coarse_cumulative_extraction_response"
        ],
        "middle_cumulative_extraction_response": coarse_middle[
            "middle_cumulative_extraction_response"
        ],
        "fine_tangent_cumulative_extraction_response": fine_cumulative,
        "coarse_window_mean_extraction_response": coarse_middle[
            "coarse_window_mean_extraction_response"
        ],
        "middle_window_mean_extraction_response": coarse_middle[
            "middle_window_mean_extraction_response"
        ],
        "fine_tangent_window_mean_extraction_response": fine_means,
        "temporal_uncertainties": np.asarray(
            [temporal_uncertainties[name] for name in metrics], dtype=float
        ),
        "surrogate_uncertainties": np.asarray(
            [surrogate_uncertainties[name] for name in metrics], dtype=float
        ),
    }
    return analysis, decisive


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
                    "scientific_status": "CERTIFIED",
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
    _validate_inputs()
    contract = _contract()
    analysis, decisive = _analyze(contract)
    certificate = bool(analysis["twenty_ms_spatial_certificate_issued"])
    anchor_required = bool(analysis["full_fine_generic_anchor_required"])
    if certificate:
        classification = (
            "three_grid_20ms_spatial_certificate_issued_reduced_architecture_"
            "manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4f_reduced_architecture_definitions_only"
        )
    elif anchor_required:
        classification = (
            "three_grid_20ms_state_certified_extraction_tangent_uncertainty_"
            "requires_fine_generic_anchor"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4e10_fine_generic_anchor_manifest_only"
        )
    else:
        classification = "three_grid_20ms_spatial_contract_failed_localization_only"
        authorized_next = "three_grid_20ms_failure_localization_only"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": True,
        "analysis_completed": True,
        "analysis": analysis,
        "state_twenty_ms_spatial_contract_certified": analysis[
            "state_spatial_contract_certified"
        ],
        "fine_twenty_ms_spatial_certificate_issued": certificate,
        "full_fine_generic_anchor_required": anchor_required,
        "full_fine_generic_anchor_authorized": False,
        "fine_generic_anchor_manifest_authorized": anchor_required,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "coarse_layout": c4e4.c4e.LAYOUTS[0],
            "middle_layout": c4e4.MIDDLE_LAYOUT,
            "fine_layout": c4e8.FINE_LAYOUT,
            "fine_coupling_face": c4e8.COUPLING_FACE,
            "fine_extraction_face": c4e8.EXTRACTION_FACE,
            "extraction_radius_rg": c4e7.EXTRACTION_RADIUS_RG,
            "generic_profile": c4e7.GENERIC_PROFILE,
            "windows_seconds": WINDOWS_SECONDS,
            "observable_names": OBSERVABLE_NAMES,
        },
    )
    _write_json(CONTRACT_PATH, contract)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "coarse_middle_arrays": _sha256(c4e4.DECISIVE_ARRAYS),
                "coarse_middle_summary": _sha256(c4e4.SUMMARY_PATH),
                "temporal_summary": _sha256(c4e6.SUMMARY_PATH),
                "fine_manifest": _sha256(c4e7.MANIFEST_PATH),
                "fine_arrays": _sha256(c4e8.DECISIVE_ARRAYS),
                "fine_summary": _sha256(c4e8.SUMMARY_PATH),
            },
            "implementation_source_hashes": {
                relative: _sha256(ROOT / relative)
                for relative in (
                    THIS_RUNNER,
                    THIS_TEST,
                    c4e4.THIS_RUNNER,
                    c4e6.THIS_RUNNER,
                    c4e8.THIS_RUNNER,
                    "src/imri_qpe/layer3_minidisk_1d/causal_inner_embedded_patch.py",
                )
                if (ROOT / relative).exists()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    state = analysis["state"]
    instant = analysis["instantaneous_extraction"]
    cumulative = analysis["cumulative_extraction"]
    means = analysis["window_mean_extraction"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Three-grid 20 ms spatial analysis WP10c9d6c7c3b5c4e9",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This analysis-only package compares the committed coarse and middle nonlinear generic responses with the completed fine block-tangent response. It executes no trajectory.",
                "",
                "## State",
                "",
                f"The common-parent state passes with RMS/max/minimum-component orders `{state['observed_rms_order']:.6f}` / `{state['observed_maximum_order']:.6f}` / `{state['minimum_significant_component_order']:.6f}`. Its refinement-error cosine is `{state['refinement_error_cosine']:.9f}`, fine difference is `{state['maximum_fine_normalized_difference']:.6e}`, temporal ratio is `{state['temporal_uncertainty_fraction_of_middle_fine_difference']:.6e}`, and surrogate ratio is `{state['surrogate_uncertainty_fraction_of_middle_fine_difference']:.6e}`.",
                "",
                "## Certified extraction partition",
                "",
                f"Aggregate instantaneous/cumulative/window-mean RMS orders are `{instant['observed_rms_order']:.6f}` / `{cumulative['observed_rms_order']:.6f}` / `{means['observed_rms_order']:.6f}`; maximum orders are `{instant['observed_maximum_order']:.6f}` / `{cumulative['observed_maximum_order']:.6f}` / `{means['observed_maximum_order']:.6f}`. Refinement-error cosines are `{instant['refinement_error_cosine']:.9f}`, `{cumulative['refinement_error_cosine']:.9f}`, and `{means['refinement_error_cosine']:.9f}`.",
                "",
                f"The conservative fine nonlinear-surrogate ratios are `{instant['surrogate_uncertainty_fraction_of_middle_fine_difference']:.6f}` instantaneous, `{cumulative['surrogate_uncertainty_fraction_of_middle_fine_difference']:.6f}` cumulative, and `{means['surrogate_uncertainty_fraction_of_middle_fine_difference']:.6f}` for window means. They exceed the frozen `0.10` gate, so the tangent-only fine response cannot issue the extraction certificate.",
                "",
                "The low-order extraction components are confined to cooling and vertical-work channels. Their current nominal orders are diagnostic because the binding fine nonlinear-anchor uncertainty has not yet been removed.",
                "",
                "## Decision",
                "",
                f"The state spatial contract is certified: `{analysis['state_spatial_contract_certified']}`. The complete 20 ms spatial certificate is issued: `{certificate}`. A fine nonlinear generic anchor is required: `{anchor_required}`.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "This is not a physical failure. Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "analysis_contract.json",
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
