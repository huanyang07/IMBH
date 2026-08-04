#!/usr/bin/env python3
"""Freeze the fail-fast screen for the 5e-3 s nonlinear duration rung."""

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

import run_causal_inner_nonlinear_canonical_time_replay_audit_wp10c9d6c7c3b5c2e1 as c2e1  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3a"
ANALYZED_BASE_COMMIT = "7620f5819a3a3bdd629a3a7f7c2fe75044f47fcf"
ANALYZED_BASE_PARENT = "d18fe34d01f5f62f8229b109353fc8c846fc685c"
ANALYZED_BASE_TREE = "7778636477f9e1b75680ed756b9d82258e4b75d1"

RUNG_START_SECONDS = 1.0e-3
SCREEN_HORIZON_SECONDS = 2.0e-3
FULL_RUNG_HORIZON_SECONDS = 5.0e-3
MAIN_TARGET_MICROSECONDS = np.arange(1000, 2001, 200, dtype=int)
MAIN_TARGETS_SECONDS = MAIN_TARGET_MICROSECONDS.astype(float) * 1.0e-6
REPLAY_TARGET_MICROSECONDS = np.asarray((1600, 1800, 2000), dtype=int)
REPLAY_TARGETS_SECONDS = REPLAY_TARGET_MICROSECONDS.astype(float) * 1.0e-6
STRICT_TARGET_MICROSECONDS = np.asarray((1800, 1900, 2000), dtype=int)
STRICT_TARGETS_SECONDS = STRICT_TARGET_MICROSECONDS.astype(float) * 1.0e-6

ARTIFACT = "causal_inner_nonlinear_third_duration_rung_manifest_wp10c9d6c7c3b5c3a"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_third_duration_rung_manifest_"
    "wp10c9d6c7c3b5c3a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_third_duration_rung_manifest_"
    "wp10c9d6c7c3b5c3a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_THIRD_DURATION_RUNG_MANIFEST_"
    "WP10C9D6C7C3B5C3A_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "third_duration_rung_manifest.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c2e1.CANONICAL_DIRECTORY


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
        or not parent["third_duration_rung_manifest_authorized"]
        or parent["third_duration_rung_propagation_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3a_third_duration_rung_manifest"
        or parent["classification"]
        != "canonical_target_replay_bitwise_certified_third_rung_manifest_authorized"
    ):
        raise RuntimeError("c3a authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3a analyzed identity changed")
    return parent


def _controller_contract() -> dict:
    prior = _read_json(c2.CONFIG_PATH)["main_controller"]
    contract = json.loads(json.dumps(prior))
    contract.update(
        {
            "initial_timestep_seconds": 2.0e-4,
            "minimum_timestep_seconds": 2.5e-5,
            "maximum_timestep_seconds": 2.0e-4,
            "maximum_BDF2_step_ratio": 2.0,
        }
    )
    return contract


def _manifest() -> dict:
    main_controller = _controller_contract()
    strict_controller = json.loads(json.dumps(main_controller))
    strict_controller.update(
        {
            "initial_timestep_seconds": 1.0e-4,
            "maximum_timestep_seconds": 1.0e-4,
        }
    )
    strict_controller["error_estimator"]["local_tolerance"] = 3.125e-5
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "third_nonlinear_duration_rung_manifest_frozen_"
            "two_e_minus_three_second_screen_authorized"
        ),
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_c2d_failure_preserved": True,
        "canonical_time_replay_certificate_required": True,
        "rung": {
            "start_seconds": RUNG_START_SECONDS,
            "screen_horizon_seconds": SCREEN_HORIZON_SECONDS,
            "full_horizon_seconds": FULL_RUNG_HORIZON_SECONDS,
            "screen_fraction_of_N128_cell_crossing": SCREEN_HORIZON_SECONDS
            / 5.54e-3,
            "full_fraction_of_N128_cell_crossing": FULL_RUNG_HORIZON_SECONDS
            / 5.54e-3,
            "trajectories": ("base", "perturbed"),
            "layout": c2.LAYOUT,
            "profile": c2.PROFILE,
            "coupling_face": c2.COUPLING_FACE,
            "initial_restart_source": (
                "committed_c2e1_canonical_final_state_and_complete_BDF_history"
            ),
        },
        "canonical_targets": {
            "construction": "integer_microseconds_times_1e_minus_6",
            "main_microseconds": MAIN_TARGET_MICROSECONDS,
            "main_seconds": MAIN_TARGETS_SECONDS,
            "replay_microseconds": REPLAY_TARGET_MICROSECONDS,
            "replay_seconds": REPLAY_TARGETS_SECONDS,
            "strict_microseconds": STRICT_TARGET_MICROSECONDS,
            "strict_seconds": STRICT_TARGETS_SECONDS,
            "single_source_slices_required": True,
            "independent_linspace_construction_forbidden": True,
        },
        "main_controller": main_controller,
        "strict_controller": strict_controller,
        "screen_execution": {
            "run_base_main_first": True,
            "run_perturbed_main_only_after_base_passes": True,
            "serialized_replay_start_seconds": 1.6e-3,
            "strict_shadow_start_seconds": 1.8e-3,
            "exact_output_landing": True,
            "no_new_BDF1_startup": True,
            "main_expected_comparisons_per_trajectory": 5,
            "replay_expected_comparisons_per_trajectory": 2,
            "strict_expected_comparisons_per_trajectory": 2,
            "durable_cache_after_each_trajectory_stage": True,
        },
        "binding_gates": {
            "all_main_replay_and_strict_steps_pass_inherited_method_gates": True,
            "main_local_error_maximum": 2.5e-4,
            "main_local_error_sum_maximum": 5.0e-3,
            "strict_local_error_maximum": 3.125e-5,
            "canonical_main_and_serialized_replay_states_bitwise": True,
            "canonical_main_and_serialized_replay_Tier_I_exports_bitwise": True,
            "canonical_main_and_serialized_replay_complete_histories_bitwise": True,
            "strict_response_maximum_scaled_state_difference": 5.0e-3,
            "strict_response_maximum_scaled_Tier_I_difference": 5.0e-3,
            "strict_response_history_cosine_minimum": 0.90,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_h_over_r": 0.12,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
        },
        "positive_branch": {
            "classification": (
                "third_rung_screen_certified_five_e_minus_three_"
                "completion_manifest_authorized"
            ),
            "authorized_next": (
                "WP10c9d6c7c3b5c3c_third_duration_rung_completion_manifest"
            ),
            "full_rung_propagation_still_requires_fresh_manifest": True,
        },
        "negative_branch": {
            "classification": "third_rung_screen_failed_later_duration_blocked",
            "authorized_next": "none",
            "localize_time_storage_or_response_failure_before_redesign": True,
        },
        "hard_stops": [
            "do not amend the c2d formal failure",
            "do not use independently generated target grids",
            "do not relax nonlinear residual, ledger or replay gates",
            "do not propagate directly to 5e-3 s before the screen passes",
            "do not begin fixed-Q or reduced slow evolution",
            "do not add tide, wind, hot-state, S-curve or QPE-cycle physics",
            "do not use N1024 as a rescue",
        ],
        "authorized_next": "WP10c9d6c7c3b5c3b_third_duration_rung_screen",
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
        "screen_horizon_seconds": SCREEN_HORIZON_SECONDS,
        "full_rung_horizon_seconds": FULL_RUNG_HORIZON_SECONDS,
        "main_target_microseconds": MAIN_TARGET_MICROSECONDS,
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
        "historical_c2d_classification_preserved": (
            "second_rung_perturbed_completion_failed_later_duration_blocked"
        ),
        "parent_classification_preserved": parent["classification"],
        "third_duration_rung_screen_authorized": True,
        "third_duration_rung_completion_manifest_authorized": False,
        "third_duration_rung_completion_propagation_authorized": False,
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
        "# Third nonlinear duration-rung manifest WP10c9d6c7c3b5c3a\n\n"
        "## Classification\n\n"
        f"`{manifest['classification']}`\n\n"
        "This definitions-only package freezes a fail-fast `2e-3 s` screen "
        "for the eventual `5e-3 s` third rung. It uses one integer-microsecond "
        "target source for main, replay and strict branches.\n\n"
        f"Authorized next: `{manifest['authorized_next']}`.\n\n"
        "The full third rung, fixed-Q experiments, and reduced slow evolution "
        "remain blocked.\n",
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
