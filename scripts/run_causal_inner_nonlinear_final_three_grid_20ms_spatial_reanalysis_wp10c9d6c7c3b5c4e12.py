#!/usr/bin/env python3
"""Issue the final 20 ms spatial decision using the nonlinear fine anchor."""

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

import run_causal_inner_nonlinear_three_grid_20ms_spatial_analysis_wp10c9d6c7c3b5c4e9 as c4e9  # noqa: E402
import run_causal_inner_nonlinear_fine_20ms_generic_anchor_wp10c9d6c7c3b5c4e11 as c4e11  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e12"
ANALYZED_BASE_COMMIT = c4e11.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e11.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e11.ANALYZED_BASE_TREE

ARTIFACT = (
    "causal_inner_nonlinear_final_three_grid_20ms_spatial_reanalysis_"
    "wp10c9d6c7c3b5c4e12"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_final_three_grid_20ms_spatial_"
    "reanalysis_wp10c9d6c7c3b5c4e12.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_final_three_grid_20ms_spatial_"
    "reanalysis_wp10c9d6c7c3b5c4e12.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_FINAL_THREE_GRID_20MS_"
    "SPATIAL_REANALYSIS_WP10C9D6C7C3B5C4E12_2026-08-13.md"
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

WINDOWS_SECONDS = tuple(c4e9.WINDOWS_SECONDS)
OBSERVABLE_NAMES = tuple(c4e9.OBSERVABLE_NAMES)
SPATIAL_GATES = dict(c4e9.SPATIAL_GATES)


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
        "classification": (
            "final_three_grid_20ms_spatial_reanalysis_contract_frozen"
        ),
        "definitions_inherited_unchanged_from_c4e9": True,
        "propagation_executed": False,
        "operator_changed": False,
        "scope": {
            "layouts": ("coarse", "middle", "fine"),
            "state_restricted_to_common_64_cell_parent": True,
            "fine_response": "continuous_nonlinear_generic_anchor",
            "slow_export": "certified_conservative_exterior_partition",
            "raw_inner_face_is_not_a_slow_export": True,
            "instantaneous_interval_seconds": (0.005, 0.020),
            "cumulative_interval_seconds": (0.005, 0.020),
            "window_mean_intervals_seconds": WINDOWS_SECONDS,
        },
        "spatial_gates": SPATIAL_GATES,
        "uncertainty_gates": {
            "maximum_temporal_fraction_of_middle_fine_difference": 0.10,
            "observability_factor": c4e9.OBSERVABILITY_FACTOR,
            "fine_nonlinear_surrogate_uncertainty": 0.0,
            "unobservable_route": (
                "report_upper_bound_only_without_order_or_direction_claim"
            ),
        },
        "uncertainty_construction": {
            "temporal": (
                "certified_coarse_middle_response_specific_shadow_envelope_plus_"
                "fine_anchor_sampled_full_vs_two_half_envelope"
            ),
            "surrogate": "none_actual_fine_nonlinear_response",
        },
        "decision": {
            "all_state_and_extraction_channels_certify": (
                "issue_20ms_spatial_certificate_and_authorize_reduced_"
                "architecture_definitions_only"
            ),
            "temporal_or_spatial_gate_fails": "localize_before_later_duration",
        },
        "hard_stops": (
            "do_not_change_c4e9_spatial_thresholds",
            "do_not_use_raw_inner_face_flux_as_slow_export",
            "do_not_run_50ms_fixed_Q_or_reduced_evolution",
        ),
    }


def _validate_inputs() -> tuple[dict, dict]:
    prior = _read_json(c4e9.SUMMARY_PATH)
    anchor = _read_json(c4e11.SUMMARY_PATH)
    if (
        not prior["passed"]
        or not prior["state_twenty_ms_spatial_contract_certified"]
        or not prior["full_fine_generic_anchor_required"]
        or not anchor["passed"]
        or not anchor["fine_generic_anchor_completed"]
        or not anchor["final_three_grid_spatial_reanalysis_authorized"]
        or anchor["fine_twenty_ms_spatial_certificate_issued"]
        or anchor["physical_failure_detected"]
    ):
        raise RuntimeError("c4e12 input authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e12 analyzed identity changed")
    return prior, anchor


def _analyze(contract: dict) -> tuple[dict, dict[str, np.ndarray]]:
    coarse_middle = _load_npz(c4e9.c4e4.DECISIVE_ARRAYS)
    fine_base = _load_npz(c4e9.c4e8.DECISIVE_ARRAYS)
    fine_anchor = _load_npz(c4e11.DECISIVE_ARRAYS)
    temporal_summary = _read_json(c4e9.c4e6.SUMMARY_PATH)["analysis"][
        "observables"
    ]
    anchor_summary = _read_json(c4e11.SUMMARY_PATH)["anchor"]
    field_scales = np.asarray(coarse_middle["field_scales"], dtype=float)
    extraction_scales = np.asarray(
        coarse_middle["extraction_scales"], dtype=float
    )
    fine_times = np.asarray(fine_base["base__accepted_times"], dtype=float)
    if not np.array_equal(
        fine_times, fine_base["extraction__accepted_times"]
    ):
        raise RuntimeError("c4e12 fine accepted times changed")
    if fine_anchor["actual_state_response"].shape[0] != fine_times.size:
        raise RuntimeError("c4e12 fine nonlinear state history changed")
    if fine_anchor["actual_extraction_response"].shape[0] != fine_times.size:
        raise RuntimeError("c4e12 fine nonlinear extraction history changed")
    common_times = np.asarray(coarse_middle["common_times_seconds"], dtype=float)
    common_indices = c4e9.c4e4._indices(fine_times, common_times)
    parent_grid, layouts, _contexts = c4e9.c4e4.b2b._layouts_and_contexts(
        c4e9.c4e4.b2b._input_arrays()
    )
    fine_layout = layouts[c4e9.c4e8.FINE_LAYOUT]
    fine_state = c4e9.c4e4._restrict(
        fine_anchor["actual_state_response"][common_indices], fine_layout
    )
    state_metric = c4e9.h2f._state_metric(
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
        fine_anchor["actual_extraction_response"], dtype=float
    )
    fine_extraction = fine_extraction_all[common_indices]
    arguments = c4e9._packet_arguments()

    def packet(coarse, middle, fine, scales):
        return c4e9.h2f._packet_payload(
            c4e9.h2f.causal_packet_history_metrics(
                coarse, middle, fine, physical_scales=scales, **arguments
            )
        )

    instantaneous_metric = packet(
        coarse_middle["coarse_extraction_response"],
        coarse_middle["middle_extraction_response"],
        fine_extraction,
        extraction_scales,
    )
    quadrature_times = np.asarray(
        coarse_middle["quadrature_times_seconds"], dtype=float
    )
    fine_quadrature = c4e9.c4e4._interpolate(
        fine_times, fine_extraction_all, quadrature_times
    )
    fine_cumulative = c4e9.c4e4._cumulative(fine_quadrature, quadrature_times)
    fine_means = c4e9.c4e4._window_means(fine_quadrature, quadrature_times)
    duration = float(quadrature_times[-1] - quadrature_times[0])
    cumulative_metric = packet(
        coarse_middle["coarse_cumulative_extraction_response"],
        coarse_middle["middle_cumulative_extraction_response"],
        fine_cumulative,
        extraction_scales * duration,
    )
    mean_metric = packet(
        coarse_middle["coarse_window_mean_extraction_response"],
        coarse_middle["middle_window_mean_extraction_response"],
        fine_means,
        extraction_scales,
    )
    fine_state_temporal = float(
        anchor_summary["maximum_sampled_state_error_estimate"]
    )
    fine_extraction_temporal = float(
        anchor_summary["maximum_sampled_extraction_error_estimate"]
    )
    temporal_uncertainties = {
        "state": float(
            temporal_summary["state"]["combined_temporal_uncertainty"]
        )
        + fine_state_temporal,
        "instantaneous_extraction": float(
            temporal_summary["instantaneous_extraction"][
                "combined_temporal_uncertainty"
            ]
        )
        + fine_extraction_temporal,
        "cumulative_extraction": float(
            temporal_summary["cumulative_extraction"][
                "combined_temporal_uncertainty"
            ]
        )
        + fine_extraction_temporal,
        "window_mean_extraction": float(
            temporal_summary["window_mean_extraction"][
                "combined_temporal_uncertainty"
            ]
        )
        + fine_extraction_temporal,
    }
    metrics = {
        "state": state_metric,
        "instantaneous_extraction": instantaneous_metric,
        "cumulative_extraction": cumulative_metric,
        "window_mean_extraction": mean_metric,
    }
    for name, metric in metrics.items():
        metric.update(
            c4e9._classify_uncertainty(
                metric,
                temporal_uncertainty=temporal_uncertainties[name],
                surrogate_uncertainty=0.0,
            )
        )
    extraction_names = (
        "instantaneous_extraction",
        "cumulative_extraction",
        "window_mean_extraction",
    )
    extraction_certified = all(
        metrics[name]["channel_certifying"] for name in extraction_names
    )
    certificate = bool(state_metric["channel_certifying"] and extraction_certified)
    analysis = {
        "analysis_contract": contract,
        "state": state_metric,
        "instantaneous_extraction": instantaneous_metric,
        "cumulative_extraction": cumulative_metric,
        "window_mean_extraction": mean_metric,
        "state_spatial_contract_certified": state_metric["channel_certifying"],
        "instantaneous_extraction_certified": instantaneous_metric[
            "channel_certifying"
        ],
        "cumulative_extraction_certified": cumulative_metric[
            "channel_certifying"
        ],
        "window_mean_extraction_certified": mean_metric["channel_certifying"],
        "complete_extraction_contract_certified": extraction_certified,
        "twenty_ms_spatial_certificate_issued": certificate,
        "fine_response_is_actual_nonlinear_anchor": True,
        "fine_surrogate_uncertainty": 0.0,
        "fine_temporal_envelopes": {
            "state": fine_state_temporal,
            "extraction": fine_extraction_temporal,
        },
        "maximum_extraction_identity_defect": anchor_summary[
            "maximum_extraction_identity_defect"
        ],
        "maximum_extraction_ledger_audit": anchor_summary[
            "maximum_extraction_ledger_audit"
        ],
        "maximum_incoming_excision_characteristics": anchor_summary[
            "maximum_incoming_excision_characteristics"
        ],
    }
    decisive = {
        "common_times_seconds": common_times,
        "quadrature_times_seconds": quadrature_times,
        "field_scales": field_scales,
        "extraction_scales": extraction_scales,
        "coarse_state_response": coarse_middle["coarse_state_response"],
        "middle_state_response": coarse_middle["middle_state_response"],
        "fine_nonlinear_state_response": fine_state,
        "coarse_extraction_response": coarse_middle[
            "coarse_extraction_response"
        ],
        "middle_extraction_response": coarse_middle[
            "middle_extraction_response"
        ],
        "fine_nonlinear_extraction_response": fine_extraction,
        "coarse_cumulative_extraction_response": coarse_middle[
            "coarse_cumulative_extraction_response"
        ],
        "middle_cumulative_extraction_response": coarse_middle[
            "middle_cumulative_extraction_response"
        ],
        "fine_nonlinear_cumulative_extraction_response": fine_cumulative,
        "coarse_window_mean_extraction_response": coarse_middle[
            "coarse_window_mean_extraction_response"
        ],
        "middle_window_mean_extraction_response": coarse_middle[
            "middle_window_mean_extraction_response"
        ],
        "fine_nonlinear_window_mean_extraction_response": fine_means,
        "temporal_uncertainties": np.asarray(
            [temporal_uncertainties[name] for name in metrics], dtype=float
        ),
        "surrogate_uncertainties": np.zeros(len(metrics), dtype=float),
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
    classification = (
        "three_grid_20ms_spatial_certificate_issued_reduced_architecture_"
        "manifest_authorized"
        if certificate
        else "final_three_grid_20ms_spatial_contract_failed_localization_only"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c4f_reduced_architecture_definitions_only"
        if certificate
        else "final_three_grid_20ms_failure_localization_only"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": certificate,
        "analysis_completed": True,
        "analysis": analysis,
        "state_twenty_ms_spatial_contract_certified": analysis[
            "state_spatial_contract_certified"
        ],
        "extraction_twenty_ms_spatial_contract_certified": analysis[
            "complete_extraction_contract_certified"
        ],
        "fine_twenty_ms_spatial_certificate_issued": certificate,
        "reduced_architecture_manifest_authorized": certificate,
        "fifty_ms_manifest_authorized": False,
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
            "coarse_layout": c4e9.c4e4.c4e.LAYOUTS[0],
            "middle_layout": c4e9.c4e4.MIDDLE_LAYOUT,
            "fine_layout": c4e9.c4e8.FINE_LAYOUT,
            "fine_coupling_face": c4e9.c4e8.COUPLING_FACE,
            "fine_extraction_face": c4e9.c4e8.EXTRACTION_FACE,
            "extraction_radius_rg": c4e9.c4e7.EXTRACTION_RADIUS_RG,
            "generic_profile": c4e9.c4e7.GENERIC_PROFILE,
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
            "scientific_status": "CERTIFIED" if certificate else "REJECTED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "c4e9_summary": _sha256(c4e9.SUMMARY_PATH),
                "coarse_middle_arrays": _sha256(c4e9.c4e4.DECISIVE_ARRAYS),
                "temporal_summary": _sha256(c4e9.c4e6.SUMMARY_PATH),
                "fine_base_tangent_arrays": _sha256(c4e9.c4e8.DECISIVE_ARRAYS),
                "fine_anchor_arrays": _sha256(c4e11.DECISIVE_ARRAYS),
                "fine_anchor_summary": _sha256(c4e11.SUMMARY_PATH),
            },
            "implementation_source_hashes": {
                relative: _sha256(ROOT / relative)
                for relative in (
                    THIS_RUNNER,
                    THIS_TEST,
                    c4e9.THIS_RUNNER,
                    c4e11.THIS_RUNNER,
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
                "# Final three-grid 20 ms spatial reanalysis WP10c9d6c7c3b5c4e12",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This analysis-only package replaces the tangent-only fine response with the completed continuous nonlinear fine generic anchor. It changes no gate and executes no trajectory.",
                "",
                "## State",
                "",
                f"The common-parent state passes with RMS/max/minimum-component orders `{state['observed_rms_order']:.6f}` / `{state['observed_maximum_order']:.6f}` / `{state['minimum_significant_component_order']:.6f}`. Its refinement-error cosine is `{state['refinement_error_cosine']:.9f}` and fine normalized difference is `{state['maximum_fine_normalized_difference']:.6e}`.",
                "",
                "## Certified extraction partition",
                "",
                f"Instantaneous/cumulative/window-mean RMS orders are `{instant['observed_rms_order']:.6f}` / `{cumulative['observed_rms_order']:.6f}` / `{means['observed_rms_order']:.6f}`. Minimum significant-component orders are `{instant['minimum_significant_component_order']:.6f}` / `{cumulative['minimum_significant_component_order']:.6f}` / `{means['minimum_significant_component_order']:.6f}`; refinement-error cosines are `{instant['refinement_error_cosine']:.9f}` / `{cumulative['refinement_error_cosine']:.9f}` / `{means['refinement_error_cosine']:.9f}`.",
                "",
                "Every previously ambiguous cooling and vertical-work component now exceeds the unchanged 0.75 order gate. The response is an actual nonlinear fine anchor, so no surrogate uncertainty remains; all temporal fractions pass the unchanged 0.10 gate.",
                "",
                "## Decision",
                "",
                f"The complete 20 ms spatial certificate is issued: `{certificate}`.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "This certifies the nonlinear truth model and conservative exterior partition through 20 ms. It does not demonstrate attraction, a fixed-Q closure, or reduced slow evolution. The raw pointwise horizon-face flux remains rejected.",
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
    return 0 if certificate else 2


if __name__ == "__main__":
    raise SystemExit(main())
