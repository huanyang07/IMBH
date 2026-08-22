#!/usr/bin/env python3
"""Freeze the final contract immediately preceding complete-cycle execution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_reaction_free_field_architecture_diagnosis_wp10c9d6c7c3b5c4f25f6 as reaction_diagnosis  # noqa: E402
import run_causal_inner_truth_free_hot_mode_engine_wp10c9d6c7c3b5c4f25fc as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fd"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fe_complete_cycle_execution"
CLASSIFICATION = (
    "conservative_event_driven_free_field_cycle_atlas_architecture_verified_"
    "complete_cycle_execution_manifest_frozen"
)
ARTIFACT = "causal_inner_adaptive_complete_cycle_manifest_wp10c9d6c7c3b5c4f25fd"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_COMPLETE_CYCLE_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FD_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_complete_cycle_manifest_"
    "wp10c9d6c7c3b5c4f25fd.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_complete_cycle_manifest_"
    "wp10c9d6c7c3b5c4f25fd.py"
)

COLD_ENGINE_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_hybrid_phase_engine_"
    "wp10c9d6c7c3b5c4f25e3"
)
PRIOR_CYCLE_ARCHITECTURE_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_cycle_map_architecture_decision_"
    "wp10c9d6c7c3b5c4f25ec_v2"
)
HOT_AXIS_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_hot_free_field_rom_preflight_"
    "wp10c9d6c7c3b5c4f25f8"
)
OFF_AXIS_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_hot_mode_off_axis_preflight_"
    "wp10c9d6c7c3b5c4f25fa"
)

MACRO_DIMENSION = 82
HIDDEN_RANK_CANDIDATES = (2, 4, 8, 16)
MAXIMUM_HIDDEN_RANK = 16
MACRO_STEP_SECONDS = 2.5e-4
MAXIMUM_MACROSTEPS = 100_000
MAXIMUM_PATCHES = 64
MAXIMUM_EXACT_FREE_FIELD_WITNESSES = 192
MAXIMUM_WITNESSES_PER_PATCH = 3
MODE_SWITCH_MARGIN = 0.1
MODE_SWITCH_PERSISTENCE = 2
MAXIMUM_PATCH_COORDINATE = 1.25
MAXIMUM_EMBEDDED_ERROR = 5.0e-2
MAXIMUM_BLIND_RATE_DEFECT = 5.0e-2
MAXIMUM_STEP_HALVING_CYCLE_MAP_DEFECT = 5.0e-2
MAXIMUM_STEP_HALVING_CYCLE_DURATION_DEFECT = 2.0e-2
MAXIMUM_HIDDEN_SECTION_RETURN_DEFECT = 5.0e-2
MAXIMUM_OFFLINE_ACQUISITION_WALL_HOURS = 24.0
MAXIMUM_COMPLETE_EXECUTION_WALL_HOURS = 48.0
COST_RESERVE_FACTOR = 1.25


def _helper():
    return parent._helper()


def _decisive_directories() -> dict[str, Path]:
    return {
        "reaction_free_architecture": reaction_diagnosis.CANONICAL_DIRECTORY,
        "truth_free_hot_engine": parent.CANONICAL_DIRECTORY,
        "cold_observed_engine": COLD_ENGINE_DIRECTORY,
        "prior_cycle_architecture": PRIOR_CYCLE_ARCHITECTURE_DIRECTORY,
        "hot_axis_preflight": HOT_AXIS_DIRECTORY,
        "hot_off_axis_preflight": OFF_AXIS_DIRECTORY,
    }


def _source_paths() -> tuple[Path, ...]:
    return (
        ROOT / THIS_RUNNER,
        ROOT / THIS_TEST,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
        ROOT / parent.THIS_RUNNER,
        ROOT / reaction_diagnosis.THIS_RUNNER,
    )


def _validate_inputs(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = {
        name: helper._validate_checksums(directory)
        for name, directory in _decisive_directories().items()
    }
    hot_engine = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    reaction = helper._read(
        reaction_diagnosis.CANONICAL_DIRECTORY / "summary.json"
    )
    cold_summary = helper._read(COLD_ENGINE_DIRECTORY / "summary.json")
    cold_metrics = helper._read(COLD_ENGINE_DIRECTORY / "engine_metrics.json")
    prior_cycle = helper._read(
        PRIOR_CYCLE_ARCHITECTURE_DIRECTORY / "summary.json"
    )
    cold_required = (
        "cold_capture",
        "cold_holdout_local",
        "cold_holdout_path",
        "cold_macro_ledger",
        "cold_rank",
        "macro_closure",
        "online_cost",
        "restart_bitwise",
        "truth_free",
    )
    if (
        not hot_engine["passed"]
        or hot_engine["classification"] != parent.PASS_CLASSIFICATION
        or not hot_engine["adaptive_complete_cycle_acquisition_manifest_authorized"]
        or hot_engine["authorized_next"] != WORK_PACKAGE
        or not reaction["passed"]
        or not reaction["conservative_free_field_hidden_amplitude_rom_selected"]
        or reaction["fixed_Q_physical_phase_authorized"]
        or cold_summary["classification"] != "truth_free_hybrid_phase_engine_failed"
        or not all(cold_metrics["gates"][name] for name in cold_required)
        or not prior_cycle["passed"]
        or not prior_cycle["working_mathematical_architecture_selected"]
        or not prior_cycle["complete_cycle_calibration_missing"]
    ):
        raise RuntimeError("complete-cycle architecture input changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("complete-cycle manifest requires a clean tracked tree")
    return {
        "directory_hashes": hashes,
        "classifications": {
            "hot_engine": hot_engine["classification"],
            "reaction_diagnosis": reaction["classification"],
            "cold_engine": cold_summary["classification"],
            "prior_cycle_architecture": prior_cycle["classification"],
        },
        "cold_retained_gates": {
            name: cold_metrics["gates"][name] for name in cold_required
        },
    }


def _cost_projection() -> dict:
    helper = _helper()
    hot = helper._read(HOT_AXIS_DIRECTORY / "hot_free_field_metrics.json")
    off = helper._read(OFF_AXIS_DIRECTORY / "hot_mode_off_axis_metrics.json")
    engine = helper._read(parent.CANONICAL_DIRECTORY / "hot_engine_metrics.json")
    hot_seconds = (
        float(hot["gate_values"]["median_direct_free_evaluation_wall_seconds"])
        + float(hot["gate_values"]["median_historical_retraction_wall_seconds"])
    )
    off_totals = np.asarray([
        float(record["total_free_evaluation_wall_seconds"])
        + float(record["retraction_wall_seconds"])
        for record in off["records"]
    ])
    witness_seconds = max(hot_seconds, float(np.median(off_totals)))
    anchor_seconds = float(
        off["gate_values"]["anchor_exact_assembly_wall_seconds"]
    )
    raw_seconds = (
        MAXIMUM_EXACT_FREE_FIELD_WITNESSES * witness_seconds
        + MAXIMUM_PATCHES * anchor_seconds
    )
    reserved_hours = COST_RESERVE_FACTOR * raw_seconds / 3600.0
    online_100k = float(engine["gate_values"]["benchmark_wall_seconds"])
    return {
        "conservative_seconds_per_exact_witness": witness_seconds,
        "exact_anchor_seconds_per_patch": anchor_seconds,
        "maximum_exact_free_field_witnesses": MAXIMUM_EXACT_FREE_FIELD_WITNESSES,
        "maximum_patches": MAXIMUM_PATCHES,
        "raw_offline_acquisition_wall_hours": raw_seconds / 3600.0,
        "reserve_factor": COST_RESERVE_FACTOR,
        "reserved_offline_acquisition_wall_hours": reserved_hours,
        "maximum_offline_acquisition_wall_hours": MAXIMUM_OFFLINE_ACQUISITION_WALL_HOURS,
        "measured_100k_online_update_decode_wall_seconds": online_100k,
        "maximum_complete_execution_wall_hours": MAXIMUM_COMPLETE_EXECUTION_WALL_HOURS,
        "offline_cost_gate_passed": reserved_hours
        <= MAXIMUM_OFFLINE_ACQUISITION_WALL_HOURS,
        "online_cost_gate_passed": online_100k <= 60.0,
    }


def _architecture() -> dict:
    return {
        "name": "conservative_event_driven_original_free_field_cycle_atlas",
        "state": {
            "q": "82 retained macro coordinates",
            "a_sigma": "mode-local hidden amplitudes with rank selected from 2,4,8,16",
            "theta": "physical driver phase on S1; the 20 ms value was replay-only and is not imposed as a universal period",
            "sigma": "discrete physical mode",
            "j": "active local atlas patch identifier",
        },
        "decoder": "y=L*q+Z*(h_sigma_j+V_sigma_j*a_sigma)",
        "dynamics": {
            "truth_field": "original unconstrained reaction-free monolithic field only",
            "macro": "dq/dt=R*f_free(y,theta)",
            "hidden": "da/dt=V_sigma_j^T*Q*f_free(y,theta)",
            "local_surrogate": "anchor reduced rate plus certified physical-direction terms; add cross/quadratic terms only after blind validation",
            "integrator": "explicit Heun with Euler embedded defect and exact event localization",
        },
        "mode_policy": {
            "known_retained_mode": "accepted full-model cold segment only",
            "post_cold_modes": "discovered from original-free-field trajectory; no fixed-Q time labels are inherited",
            "switching": "nearest validated atlas with 10 percent margin and two accepted-step persistence",
            "no_chatter": True,
        },
        "fixed_Q_scope": {
            "residual_and_jacobian_certificates_preserved": True,
            "arclength_use": "offline geometry and witness placement only",
            "physical_clock": "forbidden",
            "reaction_in_cycle_dynamics": "forbidden",
            "old_fixed_Q_transition_and_post_transition_timing": "superseded",
        },
        "conservation": {
            "macro_projection": "forbidden",
            "macro_ledger": "all 82 rates and event resets recorded exactly",
            "cumulative_cycle_output": "Delta_q and cycle duration define the later slow cycle map",
        },
    }


def _execution_contract(cost: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "authorized_execution": AUTHORIZED_NEXT,
        "initialization": {
            "retain": "hash-validated accepted cold full-model phase and its terminal primitive state",
            "discard_as_physical_time": "all fixed-Q transition/post-transition trajectories",
            "hot_fixed_Q_samples": "geometry seeds only; they may not glue a physical trajectory",
        },
        "adaptive_acquisition": {
            "maximum_patches": MAXIMUM_PATCHES,
            "maximum_exact_free_field_witnesses": MAXIMUM_EXACT_FREE_FIELD_WITNESSES,
            "maximum_witnesses_per_patch": MAXIMUM_WITNESSES_PER_PATCH,
            "hidden_rank_candidates": list(HIDDEN_RANK_CANDIDATES),
            "macro_step_seconds_initial": MACRO_STEP_SECONDS,
            "maximum_macrosteps": MAXIMUM_MACROSTEPS,
            "patch_coordinate_maximum": MAXIMUM_PATCH_COORDINATE,
            "embedded_error_maximum": MAXIMUM_EMBEDDED_ERROR,
            "blind_rate_defect_maximum": MAXIMUM_BLIND_RATE_DEFECT,
            "rule": (
                "reject before propagation on trust/error/event/guard failure; then "
                "retract exactly, audit physically, evaluate f_free, and recenter or enrich"
            ),
            "witness_pattern": "anchor, half/full physical-axis witness, and periodic diagonal blind witness",
            "rank_rule": "smallest candidate rank passing all training and blind defects",
        },
        "physical_gates": {
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "exact_chart_coordinate_residual_maximum": 5.0e-10,
            "all_original_storage_reaction_free_ledgers": "binding at exact witnesses",
        },
        "cycle_section": {
            "definition": "return to the cold/recovery Poincare section after one positive physical-driver phase winding",
            "slow_macro_drift_allowed": True,
            "hidden_section_return_defect_maximum": MAXIMUM_HIDDEN_SECTION_RETURN_DEFECT,
            "minimum_persistent_mode_switches": 2,
            "crossing_orientation": "same positive orientation as departure",
            "output": "cycle duration, Delta_q82, event sequence, averaged physical fluxes, and uncertainty ledger",
        },
        "independent_validation": {
            "blind_witness_policy": "exclude every fourth exact witness from fitting",
            "maximum_blind_rate_defect": MAXIMUM_BLIND_RATE_DEFECT,
            "replay_at_half_macro_step": True,
            "maximum_cycle_map_step_halving_defect": MAXIMUM_STEP_HALVING_CYCLE_MAP_DEFECT,
            "maximum_cycle_duration_step_halving_defect": MAXIMUM_STEP_HALVING_CYCLE_DURATION_DEFECT,
            "restart_and_suffix_replay_bitwise": True,
        },
        "cost": cost,
        "outcomes": {
            "pass": "complete_cycle_atlas_acquired_and_independently_replayed",
            "budget": "complete_cycle_inconclusive_acquisition_budget_exhausted",
            "no_return": "complete_cycle_not_observed_within_frozen_horizon",
            "physical": "complete_cycle_original_free_field_physical_gate_failed",
            "validation": "complete_cycle_atlas_blind_or_step_halving_validation_failed",
        },
        "post_pass_authorization": (
            "definitions-only multi-anchor cycle-map and slow-closure calibration; "
            "reduced slow evolution remains separately gated"
        ),
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent.manifest.parent.arclength._source()._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "DEFINITIONS_ONLY",
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "case", "path", "bytes", "sha256", "scientific_status"
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("adaptive complete-cycle manifest already exists")
    locked = _validate_inputs(require_clean=True)
    cost = _cost_projection()
    if not cost["offline_cost_gate_passed"] or not cost["online_cost_gate_passed"]:
        raise RuntimeError("complete-cycle architecture cost projection failed")
    architecture = _architecture()
    contract = _execution_contract(cost)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    helper._write_json(CANONICAL_DIRECTORY / "mathematical_architecture.json", architecture)
    helper._write_json(CANONICAL_DIRECTORY / "complete_cycle_execution_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "cost_projection.json", cost)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            str(path.relative_to(ROOT)): helper._sha(path) for path in _source_paths()
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "mathematical_architecture_verified": True,
        "cold_full_model_prefix_retained": True,
        "fixed_Q_transition_and_post_timing_superseded": True,
        "fixed_Q_physical_phase_authorized": False,
        "complete_cycle_execution_manifest_frozen": True,
        "complete_cycle_execution_authorized": True,
        "complete_cycle_executed": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Adaptive complete-cycle execution manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "## Verified mathematical architecture",
            "",
            "The online state is `(q82, a_sigma, theta, sigma, patch_id)`. The decoder is `y=L q + Z(h_sigma,j + V_sigma,j a_sigma)`, with `dq/dt=R f_free` and `da/dt=V^T Q f_free`. The original reaction-free field is the only physical field. Fixed-Q arclength is retained solely for offline geometry and witness placement.",
            "",
            "The accepted full-model cold segment is retained. Earlier fixed-Q transition and post-transition timing is superseded; all post-cold modes must be acquired prospectively from the original free field with local trust, embedded-error rejection, physical audits, blind witnesses, and hysteretic events.",
            "",
            "## Cost and execution boundary",
            "",
            f"At most `{MAXIMUM_EXACT_FREE_FIELD_WITNESSES}` exact witnesses and `{MAXIMUM_PATCHES}` patches project to `{cost['reserved_offline_acquisition_wall_hours']:.3f}` wall hours including a `{COST_RESERVE_FACTOR:.2f}x` reserve. The measured 100,000-step online update-plus-decode cost is `{cost['measured_100k_online_update_decode_wall_seconds']:.3f}` s.",
            "",
            "This is the final definitions-only package before complete-cycle execution. It authorizes that execution next but does not execute it. A cycle pass may authorize only multi-anchor cycle-map/slow-closure calibration; reduced slow evolution remains separately gated.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
