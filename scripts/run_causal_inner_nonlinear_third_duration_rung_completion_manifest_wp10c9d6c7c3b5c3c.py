#!/usr/bin/env python3
"""Freeze the staged completion contract for the 5e-3 s nonlinear rung."""

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
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_manifest_wp10c9d6c7c3b5c3a as c3a  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_screen_wp10c9d6c7c3b5c3b as c3b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3c"
ANALYZED_BASE_COMMIT = "79e5aef347028cef396a933897d5f00e250fb49a"
ANALYZED_BASE_PARENT = "baee3ba4b3b3c42e9d0ea3b444b0157e443e8275"
ANALYZED_BASE_TREE = "20cebde87e3f3e1755da572b0e81a9a514a3cd1d"

RUNG_START_SECONDS = 2.0e-3
RUNG_HORIZON_SECONDS = 5.0e-3
MASTER_TARGET_MICROSECONDS = np.arange(2000, 5001, 100, dtype=int)
MAIN_TARGET_INDICES = np.asarray((0, 4, 8, 12, 16, 20, 24, 28, 30), dtype=int)
REPLAY_TARGET_INDICES = np.asarray((24, 28, 30), dtype=int)
STRICT_TARGET_INDICES = np.asarray((28, 29, 30), dtype=int)
MAIN_TARGET_MICROSECONDS = MASTER_TARGET_MICROSECONDS[MAIN_TARGET_INDICES]
REPLAY_TARGET_MICROSECONDS = MASTER_TARGET_MICROSECONDS[REPLAY_TARGET_INDICES]
STRICT_TARGET_MICROSECONDS = MASTER_TARGET_MICROSECONDS[STRICT_TARGET_INDICES]
MAIN_TARGETS_SECONDS = MAIN_TARGET_MICROSECONDS.astype(float) * 1.0e-6
REPLAY_TARGETS_SECONDS = REPLAY_TARGET_MICROSECONDS.astype(float) * 1.0e-6
STRICT_TARGETS_SECONDS = STRICT_TARGET_MICROSECONDS.astype(float) * 1.0e-6

GENERIC_PROFILE = "p3_buffer45__generic_five_field"
HELDOUT_COARSE_PROFILES = tuple(
    profile for profile in breadth.PROFILE_NAMES if profile != GENERIC_PROFILE
)
LAYOUTS = tuple(breadth.LAYOUTS)

ARTIFACT = (
    "causal_inner_nonlinear_third_duration_rung_completion_manifest_"
    "wp10c9d6c7c3b5c3c"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_third_duration_rung_completion_manifest_"
    "wp10c9d6c7c3b5c3c.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_third_duration_rung_completion_manifest_"
    "wp10c9d6c7c3b5c3c.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_THIRD_DURATION_RUNG_COMPLETION_MANIFEST_"
    "WP10C9D6C7C3B5C3C_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "completion_manifest.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c3b.CANONICAL_DIRECTORY


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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    if (
        not parent["passed"]
        or not parent["third_duration_rung_completion_manifest_authorized"]
        or parent["third_duration_rung_completion_propagation_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3c_third_duration_rung_completion_manifest"
        or parent["classification"]
        != (
            "third_rung_screen_certified_five_e_minus_three_"
            "completion_manifest_authorized"
        )
    ):
        raise RuntimeError("c3c parent authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3c analyzed identity changed")
    return parent


def _controller_contracts() -> tuple[dict, dict]:
    main = json.loads(json.dumps(c3a._manifest()["main_controller"]))
    main.update(
        {
            "initial_timestep_seconds": 4.0e-4,
            "minimum_timestep_seconds": 2.5e-5,
            "maximum_timestep_seconds": 4.0e-4,
            "maximum_BDF2_step_ratio": 2.0,
        }
    )
    strict = json.loads(json.dumps(main))
    strict.update(
        {
            "initial_timestep_seconds": 1.0e-4,
            "maximum_timestep_seconds": 1.0e-4,
        }
    )
    strict["error_estimator"]["local_tolerance"] = 3.125e-5
    return main, strict


def _manifest() -> dict:
    main_controller, strict_controller = _controller_contracts()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "third_duration_rung_completion_manifest_frozen_coarse_"
            "five_e_minus_three_second_completion_authorized"
        ),
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_c2d_failure_preserved": True,
        "rung": {
            "start_seconds": RUNG_START_SECONDS,
            "horizon_seconds": RUNG_HORIZON_SECONDS,
            "fraction_of_N128_cell_crossing": RUNG_HORIZON_SECONDS / 5.54e-3,
            "trajectories": ("base", "perturbed"),
            "coarse_layout": c2.LAYOUT,
            "generic_profile": GENERIC_PROFILE,
            "coupling_face": c2.COUPLING_FACE,
            "initial_restart_source": (
                "committed_c3b_final_state_and_complete_BDF2_history_at_2e-3_s"
            ),
            "no_new_BDF1_startup": True,
        },
        "canonical_targets": {
            "construction": "single_integer_100_microsecond_master_source",
            "master_microseconds": MASTER_TARGET_MICROSECONDS,
            "main_indices": MAIN_TARGET_INDICES,
            "main_microseconds": MAIN_TARGET_MICROSECONDS,
            "main_seconds": MAIN_TARGETS_SECONDS,
            "replay_indices": REPLAY_TARGET_INDICES,
            "replay_microseconds": REPLAY_TARGET_MICROSECONDS,
            "replay_seconds": REPLAY_TARGETS_SECONDS,
            "strict_indices": STRICT_TARGET_INDICES,
            "strict_microseconds": STRICT_TARGET_MICROSECONDS,
            "strict_seconds": STRICT_TARGETS_SECONDS,
            "independent_target_construction_forbidden": True,
        },
        "main_controller": main_controller,
        "strict_controller": strict_controller,
        "coarse_completion_execution": {
            "run_base_stage_first": True,
            "run_perturbed_stage_only_after_base_passes": True,
            "serialized_replay_start_seconds": float(REPLAY_TARGETS_SECONDS[0]),
            "strict_shadow_start_seconds": float(STRICT_TARGETS_SECONDS[0]),
            "main_expected_comparisons_per_trajectory": 8,
            "replay_expected_comparisons_per_trajectory": 2,
            "strict_expected_comparisons_per_trajectory": 2,
            "estimated_implicit_solves_per_trajectory": 36,
            "durable_cache_after_each_complete_trajectory_stage": True,
            "exact_output_landing": True,
            "one_tangent_per_execution_process": True,
        },
        "binding_gates": {
            "all_main_replay_and_strict_steps_pass_inherited_method_gates": True,
            "main_local_error_maximum": 2.5e-4,
            "main_local_error_sum_maximum": 5.0e-3,
            "strict_local_error_maximum": 3.125e-5,
            "main_and_serialized_replay_target_labels_bitwise": True,
            "main_and_serialized_replay_states_bitwise": True,
            "main_and_serialized_replay_Tier_I_exports_bitwise": True,
            "main_and_serialized_replay_complete_BDF_histories_bitwise": True,
            "strict_response_maximum_scaled_state_difference": 5.0e-3,
            "strict_response_maximum_scaled_Tier_I_difference": 5.0e-3,
            "strict_response_history_cosine_minimum": 0.90,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_h_over_r": 0.12,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
        },
        "remaining_third_rung_scope": {
            "stage_1_now_authorized": (
                "coarse_generic_base_and_perturbed_completion_to_5e-3_s"
            ),
            "stage_2_conditional_definitions_only": {
                "scope": "coarse_heldout_profile_duration_breadth",
                "profiles": HELDOUT_COARSE_PROFILES,
            },
            "stage_3_conditional_definitions_only": {
                "scope": "middle_and_fine_generic_spatial_confirmation",
                "layouts": LAYOUTS[1:],
            },
            "all_three_stages_required_before_c4_manifest": True,
        },
        "positive_branch": {
            "classification": (
                "coarse_third_rung_completion_certified_remaining_"
                "third_rung_breadth_manifest_authorized"
            ),
            "authorized_next": (
                "WP10c9d6c7c3b5c3e_third_duration_rung_breadth_manifest"
            ),
            "middle_duration_rung_c4_still_blocked": True,
        },
        "negative_branch": {
            "classification": (
                "coarse_third_rung_completion_failed_later_duration_blocked"
            ),
            "authorized_next": "none",
            "localize_time_storage_or_response_failure_before_redesign": True,
        },
        "hard_stops": [
            "do not amend the c2d formal failure",
            "do not use independently generated target grids",
            "do not relax nonlinear residual, ledger or replay gates",
            "do not claim the complete c3 rung from the coarse stage alone",
            "do not begin the 2e-2 s c4 rung before c3 breadth and spatial gates",
            "do not begin fixed-Q or reduced slow evolution",
            "do not add tide, wind, hot-state, S-curve or QPE-cycle physics",
            "do not use N1024 as a rescue",
        ],
        "authorized_next": (
            "WP10c9d6c7c3b5c3d_coarse_third_duration_rung_completion"
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
    parent = _validate_parent()
    manifest = _manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "rung_start_seconds": RUNG_START_SECONDS,
        "rung_horizon_seconds": RUNG_HORIZON_SECONDS,
        "master_target_microseconds": MASTER_TARGET_MICROSECONDS,
        "main_targets_seconds": MAIN_TARGETS_SECONDS,
        "replay_targets_seconds": REPLAY_TARGETS_SECONDS,
        "strict_targets_seconds": STRICT_TARGETS_SECONDS,
        "coarse_layout": c2.LAYOUT,
        "generic_profile": GENERIC_PROFILE,
        "heldout_coarse_profiles": HELDOUT_COARSE_PROFILES,
        "spatial_confirmation_layouts": LAYOUTS,
        "propagation_executed": False,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "historical_c2d_classification_preserved": (
            "second_rung_perturbed_completion_failed_later_duration_blocked"
        ),
        "parent_classification_preserved": parent["classification"],
        "coarse_third_duration_rung_completion_authorized": True,
        "coarse_third_duration_rung_completion_propagation_authorized": True,
        "third_duration_rung_breadth_manifest_authorized": False,
        "third_duration_rung_spatial_confirmation_authorized": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
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
            "input_summary": str((PARENT_DIRECTORY / "summary.json").relative_to(ROOT)),
            "input_summary_sha256": _sha256(PARENT_DIRECTORY / "summary.json"),
            "input_decisive_arrays": str(
                (PARENT_DIRECTORY / "decisive_arrays.npz").relative_to(ROOT)
            ),
            "input_decisive_arrays_sha256": _sha256(
                PARENT_DIRECTORY / "decisive_arrays.npz"
            ),
            "config_sha256": causal_canonical_json_sha256(_plain(config)),
            "manifest_sha256": causal_canonical_json_sha256(_plain(manifest)),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Third nonlinear duration-rung completion manifest "
        "WP10c9d6c7c3b5c3c\n\n"
        "## Classification\n\n"
        f"`{manifest['classification']}`\n\n"
        "This definitions-only package freezes coarse generic continuation "
        "from `2e-3` through `5e-3 s`. Main, replay and strict targets are "
        "slices of one integer-`100 us` source.\n\n"
        "A coarse pass authorizes only a fresh third-rung breadth manifest. "
        "Coarse held-outs and middle/fine generic spatial confirmation remain "
        "required before the `2e-2 s` rung.\n\n"
        f"Authorized next: `{manifest['authorized_next']}`.\n\n"
        "Fixed-Q experiments and reduced slow evolution remain blocked.\n",
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
