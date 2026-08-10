#!/usr/bin/env python3
"""Run the middle-layout 5-to-6 ms spatial cost pilot."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as h2b1  # noqa: E402
import run_causal_inner_nonlinear_middle_2ms_continuation_wp10c9d6c7c3b5c3h2c1 as h2c1  # noqa: E402
import run_causal_inner_nonlinear_middle_5ms_completion_wp10c9d6c7c3b5c3h2d1 as middle5  # noqa: E402
import run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_audit_wp10c9d6c7c3b5c3h2j1 as extraction5  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_spatial_checkpoint_manifest_wp10c9d6c7c3b5c4e as c4e  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e1"
ANALYZED_BASE_COMMIT = "bc159c719a16670890e11b62f9b2db521ce8764f"
ANALYZED_BASE_PARENT = "a4ead99e2265bee5d6af463f823c470a8cf6319e"
ANALYZED_BASE_TREE = "8cf2f4ec4cde4a2b16cd4c68d45e1febb09476dc"

TARGET_MICROSECONDS = (5000, *c4e.PILOT_TARGET_MICROSECONDS)
SAMPLED_INDICES = {0, 1, 2}
EXTRACTION_FACE = c4e.EXTRACTION_FACE_INDICES[1]
COUPLING_FACE = c4e.COUPLING_FACE_INDICES[1]
PRE_PROJECTION_CORRECTION_RUNNER_SHA256 = (
    "f800286a28dbd2575aeea7f6fae0f21f5ff5767bb5b9fb84b701382a85029047"
)

ARTIFACT = (
    "causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_"
    "wp10c9d6c7c3b5c4e1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_"
    "wp10c9d6c7c3b5c4e1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_"
    "wp10c9d6c7c3b5c4e1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_6MS_SPATIAL_"
    "COST_PILOT_WP10C9D6C7C3B5C4E1_2026-08-10.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "progress.json"
BASE_PATH = CHECKPOINT_DIRECTORY / "base.npz"
TANGENT_PATH = CHECKPOINT_DIRECTORY / "tangent.npz"
ANCHOR_PATH = CHECKPOINT_DIRECTORY / "anchor.npz"


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


def _source_identity() -> dict[str, str]:
    return {
        path: h2c1._sha256(ROOT / path)
        for path in (
            THIS_RUNNER,
            THIS_TEST,
            h2b1.CONTROLLER_RELATIVE,
            h2b1.MODULE_RELATIVE,
        )
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(c4e.SUMMARY_PATH)
    manifest = _read_json(c4e.MANIFEST_PATH)
    if (
        not summary["passed"]
        or not summary["middle_six_ms_cost_pilot_authorized"]
        or summary["middle_twenty_ms_completion_authorized"]
        or summary["fine_twenty_ms_propagation_authorized"]
        or summary["fifty_ms_propagation_authorized"]
        or summary["fixed_q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["authorized_next"]
        != f"{WORK_PACKAGE}_middle_5_to_6ms_spatial_cost_pilot"
    ):
        raise RuntimeError("c4e1 authorization changed")
    if tuple(manifest["execution_stages"][0]["target_microseconds"]) != tuple(
        c4e.PILOT_TARGET_MICROSECONDS
    ):
        raise RuntimeError("c4e1 target contract changed")
    if (
        h2c1._git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or h2c1._git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or h2c1._git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e1 analyzed identity changed")
    return summary, manifest


def _load_parent_arrays() -> dict[str, np.ndarray]:
    with np.load(middle5.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _patch_shared_modules() -> None:
    for module in (h2b1, h2c1):
        module.WORK_PACKAGE = WORK_PACKAGE
        module.ARTIFACT = ARTIFACT
        module.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
        module.PROGRESS_PATH = PROGRESS_PATH
        module.BASE_PATH = BASE_PATH
        module.TANGENT_PATH = TANGENT_PATH
        module.ANCHOR_PATH = ANCHOR_PATH
        module._source_identity = _source_identity
    h2b1.TARGET_MICROSECONDS = TARGET_MICROSECONDS
    h2b1.START_SECONDS = 5.0e-3
    h2b1.STOP_SECONDS = 20.0e-3
    h2b1._anchor_sample_indices = lambda _base: set(SAMPLED_INDICES)
    h2b1._tangent_audit_indices = lambda _base: set(SAMPLED_INDICES)
    h2c1._load_parent_arrays = _load_parent_arrays
    h2c1.CANONICAL_DIRECTORY = CANONICAL_DIRECTORY
    h2c1.CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
    h2c1.CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
    h2c1.ANALYZED_BASE_COMMIT = ANALYZED_BASE_COMMIT


def _migrate_projection_only_checkpoint_identity() -> None:
    if not PROGRESS_PATH.exists():
        return
    progress = _read_json(PROGRESS_PATH)
    prior = dict(progress.get("source_identity", {}))
    current = _source_identity()
    runner_key = THIS_RUNNER
    if prior == current:
        return
    if prior.get(runner_key) != PRE_PROJECTION_CORRECTION_RUNNER_SHA256:
        raise RuntimeError("c4e1 checkpoint is not the pre-projection-correction run")
    prior_without_runner = {key: value for key, value in prior.items() if key != runner_key}
    current_without_runner = {
        key: value for key, value in current.items() if key != runner_key
    }
    if prior_without_runner != current_without_runner:
        raise RuntimeError("c4e1 scientific checkpoint source identity changed")
    progress["source_identity"] = current
    progress["projection_only_identity_migration"] = {
        "prior_runner_sha256": PRE_PROJECTION_CORRECTION_RUNNER_SHA256,
        "state_arrays_recomputed": False,
        "scientific_results_changed": False,
    }
    _write_json(PROGRESS_PATH, progress)


def _extraction_histories(configuration, base: dict, anchor: dict):
    context = configuration["context"]
    accepted_times = np.asarray(base["accepted_times"], dtype=float)
    base_values = []
    anchor_values = []
    identity_defects = []
    ledger_audits = []
    for base_state, anchor_state in zip(
        base["accepted_states"], anchor["anchor_states"], strict=True
    ):
        pair = []
        for state in (base_state, anchor_state):
            ledger = causal_five_field_radial_candidate_ledger(context, state)
            value, identity = extraction5._observable_from_ledger(
                ledger, EXTRACTION_FACE, COUPLING_FACE
            )
            pair.append(value)
            identity_defects.append(identity)
            ledger_audits.append(
                (
                    ledger.interfaces.shared_conservative_face_defect,
                    ledger.local_block_ledger_defect,
                    ledger.source_double_count_defect,
                    ledger.interfaces.incoming_excision_characteristics,
                )
            )
        base_values.append(pair[0])
        anchor_values.append(pair[1])
    base_values = np.asarray(base_values)
    anchor_values = np.asarray(anchor_values)
    return {
        "accepted_times": accepted_times,
        "base": base_values,
        "anchor": anchor_values,
        "response": anchor_values - base_values,
        "identity_defects": np.asarray(identity_defects),
        "ledger_audits": np.asarray(ledger_audits),
    }


def _simulate_remaining_steps(base: dict, contract: dict) -> int:
    elapsed = float(base["accepted_times"][-1])
    previous_timestep = float(base["accepted_timesteps"][-1])
    candidate_timestep = float(base["next_candidate_timestep"][0])
    local_error = float(base["local_error_estimates"][-1])
    targets = np.asarray(c4e.COMPLETION_OUTPUT_MICROSECONDS, dtype=float) * 1.0e-6
    target_index = int(np.searchsorted(targets, elapsed + 1.0e-15, side="right"))
    count = 0
    while elapsed < 20.0e-3 - 1.0e-15:
        target = float(targets[target_index])
        dt = min(
            candidate_timestep,
            target - elapsed,
            contract["maximum_BDF2_step_ratio"] * previous_timestep,
            contract["maximum_timestep_seconds"],
        )
        elapsed = target if abs(elapsed + dt - target) <= 1.0e-15 else elapsed + dt
        previous_timestep = dt
        candidate_timestep = h2b1.controller._next_timestep(dt, local_error, contract)
        count += 1
        if elapsed >= target - 1.0e-15:
            target_index += 1
    return count


def _projection(base_report, base, tangent_report, tangent, anchor_report, replays, setup, contract, manifest):
    remaining = _simulate_remaining_steps(base, contract)
    middle5_summary = _read_json(middle5.SUMMARY_PATH)
    routine_anchor = float(
        middle5_summary["anchor"]["median_routine_step_wall_seconds"]
    )
    sampled_anchor = float(anchor_report["median_sampled_step_wall_seconds"])
    routine_tangent = float(
        tangent_report["median_matrix_assembly_wall_seconds"]
        + middle5_summary["tangent"]["median_routine_block_step_wall_seconds"]
    )
    audited = tangent["block_step_wall_seconds"][tangent["audit_flags"]]
    audit_extra = max(
        float(np.median(audited))
        - middle5_summary["tangent"]["median_routine_block_step_wall_seconds"],
        0.0,
    )
    replay_median = float(np.median([item["wall_seconds"] for item in replays.values()]))
    raw = (
        setup
        + remaining * base_report["median_accepted_step_wall_seconds"]
        + remaining * routine_anchor
        + 3.0 * max(sampled_anchor - routine_anchor, 0.0)
        + remaining * routine_tangent
        + 3.0 * audit_extra
        + 2.0 * replay_median
        + 3600.0
    )
    safety_factor = float(manifest["pilot_contract"]["projection_safety_factor"])
    projected = safety_factor * raw
    hours = projected / 3600.0
    return {
        "remaining_steps_to_20ms": remaining,
        "safety_factor": safety_factor,
        "projected_remaining_wall_seconds": projected,
        "projected_remaining_wall_hours": hours,
        "resource_tier": (
            "automatic_continuation"
            if hours <= 24.0
            else "optimization_review"
            if hours <= 48.0
            else "explicit_cost_benefit_decision"
        ),
        "cost_projection_is_not_a_scientific_gate": True,
    }


def main() -> int:
    _parent, manifest = _validate_parent()
    _patch_shared_modules()
    _migrate_projection_only_checkpoint_identity()
    progress = h2c1._seed_checkpoints()
    configuration = h2b1._configuration()
    print("c4e1: build middle frozen nonlinear tangent", flush=True)
    frozen_tangent, setup_seconds = h2b1._build_frozen_tangent(configuration)
    with np.load(middle5.DECISIVE_ARRAYS, allow_pickle=False) as parent:
        field_scales = np.asarray(parent["tangent__field_scales"], dtype=float)
        export_scales = np.asarray(parent["tangent__export_scales"], dtype=float)
    contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    base_report, base = h2b1._run_base_targets(
        progress, configuration, frozen_tangent, field_scales, export_scales, contract
    )
    tangent_report, tangent = h2b1._run_tangent(progress, configuration, base)
    anchor_report, anchor = h2b1._run_anchor(
        progress,
        configuration,
        frozen_tangent,
        base,
        tangent,
        field_scales,
        export_scales,
        contract,
    )
    replays = {
        "base": h2b1._serialized_last_step_replay(
            "base",
            configuration,
            frozen_tangent,
            base["accepted_states"],
            base["accepted_primitive_histories"],
            base["accepted_mapped_histories"],
            base["accepted_height_histories"],
            base["accepted_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            None,
        ),
        "anchor": h2b1._serialized_last_step_replay(
            "anchor",
            configuration,
            frozen_tangent,
            anchor["anchor_states"],
            anchor["anchor_primitive_histories"],
            anchor["anchor_mapped_histories"],
            anchor["anchor_height_histories"],
            anchor["anchor_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            anchor["anchor_predictors"][-1],
        ),
    }
    replay_passed = all(
        item["checkpoint_roundtrip_bitwise"]
        and item["last_step_replay_bitwise"]
        and item["maximum_scaled_residual"] <= 1.0e-10
        for item in replays.values()
    )
    extraction = _extraction_histories(configuration, base, anchor)
    maximum_identity = float(np.max(extraction["identity_defects"]))
    maximum_ledger = np.max(extraction["ledger_audits"], axis=0)
    extraction_passed = bool(
        maximum_identity
        <= manifest["method_gates"]["maximum_extraction_identity_defect"]
        and maximum_ledger[0] <= 1.0e-12
        and maximum_ledger[1] <= 1.0e-11
        and maximum_ledger[2] <= 1.0e-12
        and int(maximum_ledger[3]) == 0
    )
    surrogate_passed = bool(
        anchor_report["state"]["discrepancy_fraction_of_observable_response"]
        <= 0.01
        and anchor_report["instantaneous_Tier_I"][
            "discrepancy_fraction_of_observable_response"
        ]
        <= 0.01
        and anchor_report["cumulative_Tier_I"][
            "discrepancy_fraction_of_observable_response"
        ]
        <= 0.01
    )
    projection = _projection(
        base_report,
        base,
        tangent_report,
        tangent,
        anchor_report,
        replays,
        setup_seconds,
        contract,
        manifest,
    )
    scientific_passed = bool(
        base_report["passed"]
        and tangent_report["passed"]
        and anchor_report["passed"]
        and replay_passed
        and surrogate_passed
        and extraction_passed
    )
    cost_review_passed = bool(
        projection["projected_remaining_wall_hours"]
        <= manifest["resource_policy"][
            "stop_for_optimization_review_if_single_stage_projects_above_hours"
        ]
    )
    passed = bool(scientific_passed and cost_review_passed)
    if scientific_passed and cost_review_passed:
        classification = (
            "middle_6ms_spatial_cost_pilot_passed_middle_twenty_ms_"
            "completion_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4e2_middle_6_to_20ms_completion_manifest"
        )
    elif scientific_passed:
        classification = (
            "middle_6ms_spatial_cost_pilot_scientifically_passed_"
            "optimization_review_required"
        )
        authorized_next = "middle_twenty_ms_cost_optimization_only"
    else:
        classification = "middle_6ms_spatial_cost_pilot_failed_fine_blocked"
        authorized_next = "middle_pilot_failure_localization_only"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "scientific_gates_passed": scientific_passed,
        "cost_review_passed": cost_review_passed,
        "setup_wall_seconds": setup_seconds,
        "base": base_report,
        "tangent": tangent_report,
        "anchor": anchor_report,
        "extraction_partition": {
            "passed": extraction_passed,
            "maximum_direct_identity_defect": maximum_identity,
            "maximum_shared_conservative_face_defect": maximum_ledger[0],
            "maximum_local_block_ledger_defect": maximum_ledger[1],
            "maximum_source_double_count_defect": maximum_ledger[2],
            "maximum_incoming_excision_characteristics": int(maximum_ledger[3]),
            "maximum_scaled_response": float(
                np.max(
                    np.abs(extraction["response"])
                    / export_scales[None, :]
                )
            ),
        },
        "serialized_replays": replays,
        "surrogate_relative_gate_passed": surrogate_passed,
        "remaining_cost_projection": projection,
        "middle_twenty_ms_completion_manifest_authorized": passed,
        "middle_twenty_ms_propagation_authorized": False,
        "fine_twenty_ms_propagation_authorized": False,
        "twenty_ms_spatial_checkpoint_certified": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    combined = {
        **{f"base__{key}": value for key, value in base.items()},
        **{f"tangent__{key}": value for key, value in tangent.items()},
        **{f"anchor__{key}": value for key, value in anchor.items()},
        **{f"extraction__{key}": value for key, value in extraction.items()},
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layout": h2b1.MIDDLE_LAYOUT,
            "profiles": h2b1.PROFILES,
            "generic_profile": h2b1.GENERIC_PROFILE,
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "extraction_radius_rg": c4e.EXTRACTION_RADIUS_RG,
            "target_microseconds": TARGET_MICROSECONDS,
            "sampled_indices": sorted(SAMPLED_INDICES),
            "controller": contract,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "working_head": h2c1._git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "spatial_checkpoint_manifest": h2c1._sha256(c4e.MANIFEST_PATH),
                "middle_5ms_summary": h2c1._sha256(middle5.SUMMARY_PATH),
                "middle_5ms_arrays": h2c1._sha256(middle5.DECISIVE_ARRAYS),
                "five_ms_extraction_certificate": h2c1._sha256(
                    extraction5.SUMMARY_PATH
                ),
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Middle 6 ms spatial cost pilot WP10c9d6c7c3b5c4e1",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"The middle base, generic nonlinear anchor, and five-profile tangent advanced from 5 to 6 ms in `{base_report['accepted_steps']}` accepted steps with `{base_report['rejected_attempts']}` retries.",
                "",
                f"The generic tangent discrepancy is `{anchor_report['state']['discrepancy_fraction_of_observable_response']:.6e}` of the state response and `{anchor_report['instantaneous_Tier_I']['discrepancy_fraction_of_observable_response']:.6e}` of the instantaneous Tier-I response. The selected extraction partition closes with a maximum direct identity defect of `{maximum_identity:.6e}`.",
                "",
                f"The projected remaining middle cost to 20 ms is `{projection['projected_remaining_wall_hours']:.2f}` hours with the frozen safety factor. Cost classification is scheduling-only and does not weaken scientific gates.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "Fine propagation, 50 ms propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{h2c1._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    h2c1._update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
