#!/usr/bin/env python3
"""Freeze the remaining nonlinear third-duration-rung breadth campaign."""

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

import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as breadth  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_temporal_wp10c9d6c7c3b4b2 as temporal  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_completion_manifest_wp10c9d6c7c3b5c3c as c3c  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3e"
ANALYZED_BASE_COMMIT = "bafc167cfc0871c892982dc13bbb74e2fae5db22"
ANALYZED_BASE_PARENT = "762b49394d6b568b837ac3198212e4803f863ace"
ANALYZED_BASE_TREE = "f9037565d579422643ab2a1d02f264949f81f43d"

GENERIC_PROFILE = c3c.GENERIC_PROFILE
COARSE_HELDOUT_PROFILES = tuple(
    profile for profile in breadth.PROFILE_NAMES if profile != GENERIC_PROFILE
)
COARSE_EXECUTION_ORDER = COARSE_HELDOUT_PROFILES
COARSE_LAYOUT = breadth.LAYOUTS[0]
SPATIAL_LAYOUTS = tuple(breadth.LAYOUTS)
ACTIVE_COUPLING_FACE_INDICES = {
    SPATIAL_LAYOUTS[0]: 48,
    SPATIAL_LAYOUTS[1]: 96,
    SPATIAL_LAYOUTS[2]: 192,
}

INITIAL_HELDOUT_HISTORY_SECONDS = np.asarray((37.5e-6, 40.0e-6))
INITIAL_HELDOUT_PREVIOUS_TIMESTEP_SECONDS = 2.5e-6
HORIZON_SECONDS = c3c.RUNG_HORIZON_SECONDS
MAIN_TARGET_MICROSECONDS = np.asarray(c3c.MAIN_TARGET_MICROSECONDS, dtype=int)
REPLAY_TARGET_MICROSECONDS = np.asarray(c3c.REPLAY_TARGET_MICROSECONDS, dtype=int)
STRICT_TARGET_MICROSECONDS = np.asarray(c3c.STRICT_TARGET_MICROSECONDS, dtype=int)
MAIN_TARGETS_SECONDS = MAIN_TARGET_MICROSECONDS.astype(float) * 1.0e-6
REPLAY_TARGETS_SECONDS = REPLAY_TARGET_MICROSECONDS.astype(float) * 1.0e-6
STRICT_TARGETS_SECONDS = STRICT_TARGET_MICROSECONDS.astype(float) * 1.0e-6

ARTIFACT = (
    "causal_inner_nonlinear_third_duration_rung_breadth_manifest_"
    "wp10c9d6c7c3b5c3e"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_third_duration_rung_breadth_"
    "manifest_wp10c9d6c7c3b5c3e.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_third_duration_rung_breadth_"
    "manifest_wp10c9d6c7c3b5c3e.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_THIRD_DURATION_RUNG_BREADTH_MANIFEST_"
    "WP10C9D6C7C3B5C3E_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "breadth_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n")


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


def _validate_parent() -> tuple[dict, dict, dict]:
    parent = _read_json(c3d.SUMMARY_PATH)
    breadth_summary = _read_json(breadth.SUMMARY_PATH)
    temporal_summary = _read_json(temporal.SUMMARY_PATH)
    if (
        not parent["passed"]
        or parent["classification"]
        != "coarse_third_rung_completion_certified_remaining_third_rung_breadth_manifest_authorized"
        or not parent["third_duration_rung_breadth_manifest_authorized"]
        or parent["third_duration_rung_breadth_propagation_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3e_third_duration_rung_breadth_manifest"
        or breadth_summary["classification"]
        != "short_horizon_nonlinear_profile_breadth_and_controller_manifest_frozen_coarse_breadth_screen_authorized"
        or not temporal_summary["passed"]
        or temporal_summary["classification"]
        != "coarse_heldout_profile_temporal_refinement_certified_middle_fine_spatial_confirmation_authorized"
    ):
        raise RuntimeError("c3e authorization or inherited breadth certificate changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3e analyzed identity changed")
    return parent, breadth_summary, temporal_summary


def _main_controller() -> dict:
    controller = json.loads(json.dumps(c3c._controller_contracts()[0]))
    controller.update(
        {
            "initial_timestep_seconds": 5.0e-6,
            "minimum_timestep_seconds": 1.25e-6,
            "maximum_timestep_seconds": 4.0e-4,
            "maximum_BDF2_step_ratio": 2.0,
        }
    )
    return controller


def _manifest() -> dict:
    spatial_gates = {
        "minimum_rms_order": 0.75,
        "minimum_maximum_order": 0.75,
        "minimum_significant_component_order": 0.75,
        "maximum_fine_normalized_difference": 0.05,
        "minimum_history_cosine": 0.90,
        "minimum_refinement_error_cosine": 0.90,
        "minimum_relative_activity": 1.0e-8,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "third_duration_rung_breadth_manifest_frozen_coarse_heldout_"
            "duration_screen_authorized"
        ),
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_c2d_failure_preserved": True,
        "common_contract": {
            "horizon_seconds": HORIZON_SECONDS,
            "fraction_of_N128_cell_crossing": HORIZON_SECONDS / 5.54e-3,
            "canonical_target_source": "c3c_single_integer_100_microsecond_source",
            "main_target_microseconds": MAIN_TARGET_MICROSECONDS,
            "replay_target_microseconds": REPLAY_TARGET_MICROSECONDS,
            "strict_target_microseconds": STRICT_TARGET_MICROSECONDS,
            "independent_target_construction_forbidden": True,
            "main_controller": _main_controller(),
            "strict_controller": c3c._controller_contracts()[1],
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "minimum_reconstruction_factor": 1.0,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_h_over_r": 0.12,
            "maximum_incoming_excision_characteristics": 0,
        },
        "coarse_heldout_duration_stage": {
            "authorized_now": True,
            "layout": COARSE_LAYOUT,
            "active_coupling_face": ACTIVE_COUPLING_FACE_INDICES[COARSE_LAYOUT],
            "profiles": COARSE_HELDOUT_PROFILES,
            "fail_fast_execution_order": COARSE_EXECUTION_ORDER,
            "binding_multiplier": breadth.BINDING_PROPAGATION_MULTIPLIER,
            "initial_history_source": (
                "committed_b4b2_dt_2p5em06_last_two_primitive_states_at_"
                "37p5_and_40_microseconds"
            ),
            "initial_history_seconds": INITIAL_HELDOUT_HISTORY_SECONDS,
            "initial_previous_timestep_seconds": (
                INITIAL_HELDOUT_PREVIOUS_TIMESTEP_SECONDS
            ),
            "complete_mapped_and_height_histories_reconstructed_from_"
            "the_committed_primitive_states": True,
            "certified_c3d_base_main_replay_strict_reused_by_hash": True,
            "new_profile_histories_must_not_be_recombined_from_linear_bases": True,
            "one_tangent_and_one_process_per_profile": True,
            "durable_cache_after_each_complete_profile": True,
            "serialized_replay_start_seconds": float(REPLAY_TARGETS_SECONDS[0]),
            "strict_shadow_start_seconds": float(STRICT_TARGETS_SECONDS[0]),
            "same_target_replay_states_exports_histories_and_restart_bitwise": True,
            "strict_response_maximum_scaled_state_difference": 5.0e-3,
            "strict_response_maximum_scaled_Tier_I_difference": 5.0e-3,
            "strict_response_history_cosine_minimum": 0.90,
            "stop_on_first_profile_failure": True,
        },
        "generic_spatial_confirmation_scope": {
            "scope_frozen_but_propagation_not_authorized": True,
            "profile": GENERIC_PROFILE,
            "layouts": SPATIAL_LAYOUTS,
            "active_coupling_face_indices": ACTIVE_COUPLING_FACE_INDICES,
            "coarse_c3d_base_and_perturbed_evidence_reused_by_hash": True,
            "middle_and_fine_short_horizon_history_source": (
                "committed_b4b3_layout_native_base_and_generic_histories"
            ),
            "compare_conservatively_restricted_state_response_on_common_parent": True,
            "compare_13_Tier_I_exports_at_correct_active_faces": True,
            "spatial_gates": spatial_gates,
            "replay_and_strict_shadow_required_on_middle_and_fine": True,
            "maximum_strict_to_observable_medium_fine_spatial_error_ratio": 0.10,
            "fresh_definitions_only_manifest_required_after_coarse_breadth_passes": True,
        },
        "positive_branch": {
            "classification": (
                "coarse_heldout_third_rung_duration_breadth_certified_"
                "generic_spatial_confirmation_manifest_authorized"
            ),
            "authorized_next": (
                "WP10c9d6c7c3b5c3g_third_duration_rung_spatial_"
                "confirmation_manifest"
            ),
            "fourth_duration_rung_still_blocked": True,
        },
        "negative_branch": {
            "classification": (
                "coarse_heldout_third_rung_duration_breadth_failed_"
                "later_duration_blocked"
            ),
            "authorized_next": "none",
            "localize_the_first_failed_profile_before_redesign": True,
        },
        "hard_stops": [
            "do not amend the c2d formal failure",
            "do not tune a heldout profile after this manifest",
            "do not generate targets independently",
            "do not relax nonlinear residual, ledger, replay or response gates",
            "do not begin middle/fine propagation before a fresh spatial manifest",
            "do not begin the 2e-2 s fourth rung before breadth and spatial pass",
            "do not begin fixed-Q experiments or reduced slow evolution",
            "do not add tide, wind, hot-state, S-curve or QPE-cycle physics",
            "do not use N1024 as a rescue",
        ],
        "authorized_next": (
            "WP10c9d6c7c3b5c3f_coarse_heldout_third_duration_rung_screen"
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
    parent, breadth_summary, temporal_summary = _validate_parent()
    manifest = _manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "horizon_seconds": HORIZON_SECONDS,
        "coarse_layout": COARSE_LAYOUT,
        "coarse_heldout_profiles": COARSE_HELDOUT_PROFILES,
        "coarse_execution_order": COARSE_EXECUTION_ORDER,
        "generic_profile": GENERIC_PROFILE,
        "spatial_layouts": SPATIAL_LAYOUTS,
        "active_coupling_face_indices": ACTIVE_COUPLING_FACE_INDICES,
        "main_targets_seconds": MAIN_TARGETS_SECONDS,
        "replay_targets_seconds": REPLAY_TARGETS_SECONDS,
        "strict_targets_seconds": STRICT_TARGETS_SECONDS,
        "propagation_executed": False,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_c2d_classification_preserved": (
            "second_rung_perturbed_completion_failed_later_duration_blocked"
        ),
        "parent_classification_preserved": parent["classification"],
        "inherited_breadth_classification": breadth_summary["classification"],
        "inherited_temporal_classification": temporal_summary["classification"],
        "coarse_heldout_duration_screen_authorized": True,
        "coarse_heldout_duration_propagation_executed": False,
        "third_duration_rung_spatial_confirmation_manifest_authorized": False,
        "third_duration_rung_spatial_confirmation_propagation_authorized": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_summary": c3d.SUMMARY_PATH,
        "parent_arrays": c3d.DECISIVE_ARRAYS,
        "breadth_manifest": breadth.MANIFEST_PATH,
        "temporal_summary": temporal.SUMMARY_PATH,
        "temporal_arrays": temporal.DECISIVE_ARRAYS,
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent": ANALYZED_BASE_PARENT,
            "analyzed_base_tree": ANALYZED_BASE_TREE,
            "implementation_commit_before_manifest": _git_value("rev-parse", "HEAD"),
            "implementation_tree_before_manifest": _git_value(
                "rev-parse", "HEAD^{tree}"
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "test": THIS_TEST,
            "input_hashes": {
                name: _sha256(path) for name, path in input_paths.items()
            },
            "config_sha256": causal_canonical_json_sha256(_plain(config)),
            "manifest_sha256": causal_canonical_json_sha256(_plain(manifest)),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Third nonlinear duration-rung breadth manifest "
        "WP10c9d6c7c3b5c3e\n\n"
        "## Classification\n\n"
        f"`{manifest['classification']}`\n\n"
        "This definitions-only package freezes four coarse held-out duration "
        "trajectories through `5e-3 s` and the later generic middle/fine "
        "spatial-confirmation scope. It propagates no state.\n\n"
        f"Held-outs: `{', '.join(COARSE_HELDOUT_PROFILES)}`.\n\n"
        f"Authorized next: `{manifest['authorized_next']}`.\n\n"
        "Middle/fine propagation, the `2e-2 s` rung, fixed-Q experiments, "
        "and reduced slow evolution remain blocked.\n",
        encoding="utf-8",
    )
    checksum_paths = (CONFIG_PATH, MANIFEST_PATH, PROVENANCE_PATH, SUMMARY_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
