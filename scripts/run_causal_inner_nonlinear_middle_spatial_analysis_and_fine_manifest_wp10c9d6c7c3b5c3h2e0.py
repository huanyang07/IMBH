#!/usr/bin/env python3
"""Analyze coarse/middle 5 ms evidence and freeze the fine campaign."""

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

import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as h2b1  # noqa: E402
import run_causal_inner_nonlinear_middle_5ms_completion_wp10c9d6c7c3b5c3h2d1 as h2d1  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as b2b  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_cost_bounded_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g1 as g1  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_coarse_heldout_third_duration_rung_screen_wp10c9d6c7c3b5c3f as c3f  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2e0"
ANALYZED_BASE_COMMIT = "e3728c4c6fe7148aeca9ca80e10a7c8d97095c24"
ANALYZED_BASE_PARENT = "92627724ccc70511d663f4aeeab59954c3014007"
ANALYZED_BASE_TREE = "70048fc70240c1a750a3d5627ac3650f78dee9d5"

PROFILES = tuple(h2b1.PROFILES)
GENERIC_PROFILE = h2b1.GENERIC_PROFILE
MIDDLE_LAYOUT = h2b1.MIDDLE_LAYOUT
FINE_LAYOUT = g1.FINE_LAYOUT
SURROGATE_TO_SPATIAL_GATE = 0.10
FINE_TARGET_MICROSECONDS = (
    40,
    200,
    400,
    600,
    800,
    1000,
    1400,
    1800,
    2000,
    2400,
    2800,
    3200,
    3600,
    4000,
    4400,
    4800,
    5000,
)
FINE_AUDIT_TARGET_MICROSECONDS = (200, 2000, 5000)

ARTIFACT = (
    "causal_inner_nonlinear_middle_spatial_analysis_and_fine_manifest_"
    "wp10c9d6c7c3b5c3h2e0"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_spatial_analysis_and_fine_"
    "manifest_wp10c9d6c7c3b5c3h2e0.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_spatial_analysis_and_fine_"
    "manifest_wp10c9d6c7c3b5c3h2e0.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_SPATIAL_"
    "ANALYSIS_AND_FINE_MANIFEST_WP10C9D6C7C3B5C3H2E0_2026-08-06.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "fine_manifest.json"
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _validate_parent() -> None:
    parent = _read_json(h2d1.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["middle_5ms_spatial_analysis_authorized"]
        or not parent["fine_manifest_authorized"]
        or parent["middle_fine_5ms_spatial_certificate_issued"]
    ):
        raise RuntimeError("h2e0 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2e0 analyzed identity changed")


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(times)[:, None],
        axis=0,
    )
    return result


def _maximum_scaled_difference(
    left: np.ndarray,
    right: np.ndarray,
    scales: np.ndarray,
) -> float:
    shape = (1,) * (left.ndim - 1) + (scales.size,)
    return float(np.max(np.abs(left - right) / scales.reshape(shape)))


def _restrict(history: np.ndarray, layout) -> np.ndarray:
    return np.asarray(
        [
            restrict_causal_embedded_patch_cell_averages(state, layout)
            for state in history
        ],
        dtype=float,
    )


def _analyze() -> tuple[dict, dict[str, np.ndarray]]:
    coarse_generic = _load_npz(c3d.DECISIVE_ARRAYS)
    coarse_heldout = _load_npz(c3f.DECISIVE_ARRAYS)
    middle = _load_npz(h2d1.DECISIVE_ARRAYS)
    times = np.asarray(coarse_generic["main_times_seconds"], dtype=float)
    if not np.array_equal(times, middle["base__accepted_times"]):
        raise RuntimeError("coarse/middle 5 ms target times changed")
    if middle["tangent__state_directions"].shape[1] != len(PROFILES):
        raise RuntimeError("middle tangent profile count changed")
    field_scales = np.asarray(middle["tangent__field_scales"], dtype=float)
    export_scales = np.asarray(middle["tangent__export_scales"], dtype=float)
    _parent_grid, layouts, _configurations = b2b._layouts_and_contexts(
        b2b._input_arrays()
    )
    middle_layout = layouts[MIDDLE_LAYOUT]
    coarse_base_state = coarse_generic["base__main__output_states"]
    coarse_base_export = coarse_generic["base__main__output_exports"]

    reports = {}
    decisive = {
        "times_seconds": times,
        "field_scales": field_scales,
        "export_scales": export_scales,
    }
    generic_ratios = None
    for index, profile in enumerate(PROFILES):
        if profile == GENERIC_PROFILE:
            coarse_state = (
                coarse_generic["perturbed__main__output_states"]
                - coarse_base_state
            )
            coarse_export = (
                coarse_generic["perturbed__main__output_exports"]
                - coarse_base_export
            )
        else:
            coarse_state = (
                coarse_heldout[f"{profile}__main__output_states"]
                - coarse_base_state
            )
            coarse_export = (
                coarse_heldout[f"{profile}__main__output_exports"]
                - coarse_base_export
            )
        middle_state = _restrict(
            middle["tangent__state_directions"][:, index],
            middle_layout,
        )
        middle_export = middle["tangent__export_directions"][:, index]
        coarse_cumulative = _cumulative(coarse_export, times)
        middle_cumulative = _cumulative(middle_export, times)
        state_spatial = _maximum_scaled_difference(
            middle_state, coarse_state, field_scales
        )
        instant_spatial = _maximum_scaled_difference(
            middle_export, coarse_export, export_scales
        )
        cumulative_spatial = _maximum_scaled_difference(
            middle_cumulative,
            coarse_cumulative,
            export_scales * times[-1],
        )
        report = {
            "coarse_middle_state_maximum_scaled_difference": state_spatial,
            "coarse_middle_instantaneous_Tier_I_maximum_scaled_difference": (
                instant_spatial
            ),
            "coarse_middle_cumulative_Tier_I_maximum_scaled_difference": (
                cumulative_spatial
            ),
        }
        decisive[f"{profile}__coarse_state_response"] = coarse_state
        decisive[f"{profile}__middle_tangent_state_response"] = middle_state
        decisive[f"{profile}__coarse_export_response"] = coarse_export
        decisive[f"{profile}__middle_tangent_export_response"] = middle_export
        if profile == GENERIC_PROFILE:
            actual_state = _restrict(
                middle["anchor__actual_state_response"], middle_layout
            )
            actual_export = middle["anchor__actual_export_response"]
            actual_cumulative = _cumulative(actual_export, times)
            state_surrogate = _maximum_scaled_difference(
                actual_state, middle_state, field_scales
            )
            instant_surrogate = _maximum_scaled_difference(
                actual_export, middle_export, export_scales
            )
            cumulative_surrogate = _maximum_scaled_difference(
                actual_cumulative,
                middle_cumulative,
                export_scales * times[-1],
            )
            generic_ratios = {
                "state": state_surrogate / max(state_spatial, np.finfo(float).tiny),
                "instantaneous_Tier_I": instant_surrogate
                / max(instant_spatial, np.finfo(float).tiny),
                "cumulative_Tier_I": cumulative_surrogate
                / max(cumulative_spatial, np.finfo(float).tiny),
            }
            report.update(
                {
                    "middle_tangent_nonlinear_state_maximum_scaled_difference": (
                        state_surrogate
                    ),
                    "middle_tangent_nonlinear_instantaneous_Tier_I_maximum_scaled_difference": (
                        instant_surrogate
                    ),
                    "middle_tangent_nonlinear_cumulative_Tier_I_maximum_scaled_difference": (
                        cumulative_surrogate
                    ),
                    "surrogate_to_coarse_middle_spatial_difference_ratios": (
                        generic_ratios
                    ),
                }
            )
            decisive["generic_middle_nonlinear_state_response"] = actual_state
            decisive["generic_middle_nonlinear_export_response"] = actual_export
        reports[profile] = report
    if generic_ratios is None:
        raise RuntimeError("generic profile missing from analysis")
    full_anchor_required = any(
        value > SURROGATE_TO_SPATIAL_GATE for value in generic_ratios.values()
    )
    return (
        {
            "profiles": reports,
            "surrogate_to_spatial_gate": SURROGATE_TO_SPATIAL_GATE,
            "generic_surrogate_to_spatial_difference_ratios": generic_ratios,
            "generic_state_surrogate_small_relative_to_spatial_difference": (
                generic_ratios["state"] <= SURROGATE_TO_SPATIAL_GATE
            ),
            "generic_Tier_I_surrogate_small_relative_to_spatial_difference": (
                generic_ratios["instantaneous_Tier_I"]
                <= SURROGATE_TO_SPATIAL_GATE
                and generic_ratios["cumulative_Tier_I"]
                <= SURROGATE_TO_SPATIAL_GATE
            ),
            "full_fine_generic_nonlinear_anchor_required": full_anchor_required,
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
        "passed": True,
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
    _validate_parent()
    analysis, decisive = _analyze()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "middle_spatial_analysis_complete_fine_base_tangent_and_full_"
            "generic_anchor_authorized"
        ),
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "layout": FINE_LAYOUT,
        "active_coupling_face": int(
            g1.c3g.ACTIVE_COUPLING_FACE_INDICES[FINE_LAYOUT]
        ),
        "target_microseconds": FINE_TARGET_MICROSECONDS,
        "audit_target_microseconds": FINE_AUDIT_TARGET_MICROSECONDS,
        "required_trajectories": (
            "fine_nonlinear_base",
            "fine_five_profile_complete_discrete_BDF_tangent",
            "fine_full_nonlinear_generic_anchor",
        ),
        "full_fine_generic_anchor_selected_by_evidence": True,
        "selection_reason": (
            "middle_generic_Tier_I_tangent_uncertainty_exceeds_0p10_of_"
            "coarse_middle_spatial_difference"
        ),
        "execution_contract": {
            "fine_layout_owns_adaptive_base_schedule": True,
            "generic_anchor_replays_exact_accepted_fine_base_schedule": True,
            "tangent_uses_exact_accepted_fine_base_schedule": True,
            "tangent_predictor_initializes_generic_anchor_Newton_solves": True,
            "durable_checkpoint_after_every_target": True,
            "base_full_step_doubling": True,
            "generic_anchor_sampled_step_doubling_targets_microseconds": (
                FINE_AUDIT_TARGET_MICROSECONDS
            ),
            "complete_residual_JVP_audit_targets_microseconds": (
                FINE_AUDIT_TARGET_MICROSECONDS
            ),
            "base_and_anchor_last_step_bitwise_replay": True,
            "no_coarse_or_middle_schedule_as_acceptance_authority": True,
        },
        "method_gates": {
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure_defect": 1.0e-9,
            "minimum_path_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
            "maximum_export_ledger_defect": 1.0e-9,
            "maximum_step_matrix_JVP_relative_defect": 1.0e-6,
            "bitwise_last_step_replay": True,
        },
        "spatial_gates_unchanged": dict(g1.SPATIAL_GATES),
        "temporal_uncertainty_gate": {
            "maximum_uncertainty_to_observable_medium_fine_difference": 0.10,
            "unobservable_route": (
                "report_upper_bound_only_without_order_or_direction_claim"
            ),
        },
        "resource_policy": {
            "24_hours_is_soft_planning_target_not_a_scientific_gate": True,
            "checkpointed_unattended_segments": True,
            "stop_before_anchor_only_for_scientific_method_or_cost_reassessment": True,
        },
        "hard_stops": (
            "do_not_issue_spatial_certificate_before_fine_completion",
            "do_not_drop_full_fine_generic_anchor",
            "do_not_relax_spatial_temporal_method_or_replay_gates",
            "do_not_begin_fourth_duration_rung_fixed_Q_or_reduced_evolution",
            "do_not_use_N1024_as_rescue",
        ),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "analysis": analysis,
        "fine_manifest_frozen": True,
        "fine_cost_bounded_propagation_authorized": True,
        "full_fine_generic_nonlinear_anchor_required": True,
        "middle_fine_5ms_spatial_certificate_issued": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c3h2e1_fine_5ms_completion",
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "profiles": PROFILES,
            "middle_layout": MIDDLE_LAYOUT,
            "fine_layout": FINE_LAYOUT,
            "surrogate_to_spatial_gate": SURROGATE_TO_SPATIAL_GATE,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "middle_summary": _sha256(h2d1.SUMMARY_PATH),
                "middle_arrays": _sha256(h2d1.DECISIVE_ARRAYS),
                "coarse_generic_arrays": _sha256(c3d.DECISIVE_ARRAYS),
                "coarse_heldout_arrays": _sha256(c3f.DECISIVE_ARRAYS),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    ratios = analysis["generic_surrogate_to_spatial_difference_ratios"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Middle spatial analysis and fine manifest WP10c9d6c7c3b5c3h2e0",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "The middle tangent/nonlinear discrepancy is "
                f"`{ratios['state']:.6e}` of the coarse-middle state difference, "
                f"`{ratios['instantaneous_Tier_I']:.6e}` of the instantaneous "
                "Tier-I difference, and "
                f"`{ratios['cumulative_Tier_I']:.6e}` of the cumulative Tier-I "
                "difference.",
                "",
                "The state surrogate is comfortably below the prospective 0.10 "
                "spatial-uncertainty fraction. The Tier-I surrogate is not. The "
                "fine campaign therefore requires the nonlinear base, all five "
                "block tangents, and a full nonlinear generic anchor. This is an "
                "evidence-selected cost increase, not a relaxation of the spatial "
                "certificate.",
                "",
                "Fine propagation is authorized next. The 5 ms spatial certificate, "
                "fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "config.json",
        "decisive_arrays.npz",
        "fine_manifest.json",
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
