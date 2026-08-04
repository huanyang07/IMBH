#!/usr/bin/env python3
"""Freeze the second nonlinear duration rung without propagation."""

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
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_nonlinear_first_duration_rung_manifest_wp10c9d6c7c3b5c1a as c1a  # noqa: E402
import run_causal_inner_nonlinear_first_duration_rung_wp10c9d6c7c3b5c1 as c1  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c2a"
ANALYZED_BASE_COMMIT = "9bac242a727b5c7dc254735b2c37b88057a60f45"
ANALYZED_BASE_PARENT = "b127bad624d2bc1f2c90ce0dbcde11f2f4767991"
ANALYZED_BASE_TREE = "add64e7a567be5061d8e9a3acfd637aa1c176c28"

HORIZON_SECONDS = 1.0e-3
CONTINUATION_START_SECONDS = 2.0e-4
PREVIOUS_HISTORY_TIME_SECONDS = 1.8e-4
OUTPUT_TIMES_SECONDS = np.linspace(0.0, HORIZON_SECONDS, 11)
CONTINUATION_OUTPUT_TIMES_SECONDS = np.linspace(
    CONTINUATION_START_SECONDS, HORIZON_SECONDS, 9
)
RESTART_TIME_SECONDS = 6.0e-4
STRICT_SHADOW_START_SECONDS = 8.0e-4
LAYOUT = c1a.LAYOUT
PROFILE = c1a.PROFILE
COUPLING_FACE = 48

ARTIFACT = (
    "causal_inner_nonlinear_second_duration_rung_manifest_"
    "wp10c9d6c7c3b5c2a"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_second_duration_rung_manifest_"
    "wp10c9d6c7c3b5c2a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_second_duration_rung_manifest_"
    "wp10c9d6c7c3b5c2a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_SECOND_DURATION_RUNG_MANIFEST_"
    "WP10C9D6C7C3B5C2A_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "second_duration_rung_manifest.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c1.CANONICAL_DIRECTORY


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
        for path in (THIS_RUNNER, THIS_TEST, c1.THIS_RUNNER, c1a.THIS_RUNNER)
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    first_manifest = _read_json(c1a.MANIFEST_PATH)
    with np.load(PARENT_DIRECTORY / "decisive_arrays.npz") as arrays:
        for trajectory in ("base", "perturbed"):
            times = arrays[f"{trajectory}__times_seconds"]
            accepted_dt = arrays[
                f"{trajectory}__main_accepted_timesteps_seconds"
            ]
            if not (
                np.isclose(
                    times[-2], PREVIOUS_HISTORY_TIME_SECONDS, rtol=0.0, atol=1e-18
                )
                and np.isclose(
                    times[-1], CONTINUATION_START_SECONDS, rtol=0.0, atol=1e-18
                )
                and np.isclose(accepted_dt[-1], 2.0e-5, rtol=0.0, atol=1e-18)
            ):
                raise RuntimeError("c2a canonical continuation history changed")
    if (
        parent["classification"]
        != "first_nonlinear_duration_rung_certified_"
        "second_rung_manifest_authorized"
        or not parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c2a_second_duration_rung_manifest"
        or parent["second_duration_rung_propagation_authorized"]
        or not parent["strict_shadow_comparison"]["passed"]
        or first_manifest["propagation_executed"]
    ):
        raise RuntimeError("c2a authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2a analyzed identity changed")
    return parent, first_manifest, _read_json(PARENT_DIRECTORY / "provenance.json")


def _manifest(first_manifest: dict) -> dict:
    main_controller = json.loads(json.dumps(first_manifest["main_controller"]))
    main_controller["initial_timestep_seconds"] = 4.0e-5
    main_controller["maximum_timestep_seconds"] = 1.0e-4
    strict_controller = json.loads(json.dumps(main_controller))
    strict_controller["initial_timestep_seconds"] = 5.0e-5
    strict_controller["maximum_timestep_seconds"] = 5.0e-5
    strict_controller["error_estimator"]["local_tolerance"] = 3.125e-5
    strict_controller["error_estimator"][
        "rung_sum_of_accepted_error_estimates"
    ] = 6.25e-4
    reference_clocks = first_manifest["reference_clocks_seconds"]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "second_nonlinear_duration_rung_manifest_frozen_"
            "one_e_minus_three_second_propagation_authorized"
        ),
        "passed": True,
        "operator_changed": False,
        "production_defaults_changed": False,
        "propagation_executed": False,
        "layout": LAYOUT,
        "profile": PROFILE,
        "profile_kind": "primary_physical",
        "coupling_face": COUPLING_FACE,
        "background_and_perturbed_trajectories_required": True,
        "horizon_seconds": HORIZON_SECONDS,
        "output_times_seconds": OUTPUT_TIMES_SECONDS,
        "continuation_output_times_seconds": CONTINUATION_OUTPUT_TIMES_SECONDS,
        "restart_time_seconds": RESTART_TIME_SECONDS,
        "continuation_contract": {
            "canonical_parent_artifact": c1.ARTIFACT,
            "previous_history_time_seconds": PREVIOUS_HISTORY_TIME_SECONDS,
            "continuation_start_seconds": CONTINUATION_START_SECONDS,
            "previous_timestep_seconds": 2.0e-5,
            "continue_BDF2_without_new_BDF1_startup": True,
            "reconstruct_temporal_maps_from_committed_primitive_states": True,
            "verify_parent_state_and_input_hashes_before_propagation": True,
            "stitch_parent_outputs_at_0_1e-4_2e-4": True,
        },
        "main_controller": main_controller,
        "main_rung_error_budget": {
            "maximum_local_error_estimate": 2.5e-4,
            "maximum_sum_of_accepted_error_estimates": 5.0e-3,
            "fixed_state_and_Tier_I_scales": True,
        },
        "strict_shadow": {
            "required": True,
            "start_time_seconds": STRICT_SHADOW_START_SECONDS,
            "stop_time_seconds": HORIZON_SECONDS,
            "controller": strict_controller,
            "same_saved_state_and_BDF_history_as_main": True,
            "maximum_scaled_state_response_difference": 5.0e-3,
            "maximum_scaled_Tier_I_response_difference": 5.0e-3,
            "minimum_state_and_Tier_I_history_cosine": 0.90,
        },
        "method_gates": main_controller["step_method_gates"],
        "physical_readiness_gates": first_manifest["physical_readiness_gates"],
        "reference_clocks_seconds": reference_clocks,
        "clock_fractions": {
            "N128_cell_crossing": HORIZON_SECONDS
            / reference_clocks["minimum_N128_cell_characteristic_crossing"],
            "stress_relaxation": HORIZON_SECONDS
            / reference_clocks["minimum_stress_relaxation"],
        },
        "estimated_execution": {
            "main_step_comparisons_per_trajectory": 9,
            "restart_replay_comparisons_per_trajectory": 4,
            "strict_shadow_comparisons_per_trajectory": 4,
            "trajectory_count": 2,
            "estimated_wall_hours": 7.0,
            "checkpoint_after_each_complete_trajectory": True,
        },
        "binding_decision": {
            "pass": "WP10c9d6c7c3b5c3a_third_duration_rung_manifest",
            "fail_method_or_accuracy": (
                "WP10c9d6c7c3b5c2b_second_duration_rung_localization"
            ),
            "no_automatic_gate_relaxation": True,
        },
        "stage_authorization": {
            "authorized_now": (
                "WP10c9d6c7c3b5c2_second_duration_rung_propagation"
            ),
            "later_duration_rungs_authorized_now": False,
            "fixed_q_authorized_now": False,
            "reduced_evolution_authorized_now": False,
        },
        "hard_stops": [
            "no face-index alias; use active coarse coupling face 48",
            "no tolerance, output, profile, continuation or shadow tuning",
            "no third duration rung before the second rung passes",
            "no spatial-operator or interface redesign",
            "no fixed-Q experiment or reduced evolution",
            "no tide, wind, hot-state, S-curve or QPE-cycle physics",
            "no N1024 rescue",
        ],
        "authorized_next": (
            "WP10c9d6c7c3b5c2_second_duration_rung_propagation"
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


def _report(manifest: dict) -> str:
    return "\n".join(
        [
            "# Second nonlinear duration-rung manifest WP10c9d6c7c3b5c2a",
            "",
            "## Classification",
            "",
            f"`{manifest['classification']}`",
            "",
            "This definitions-only package freezes continuation from the "
            "certified first-rung BDF2 history. It changes no operator or "
            "production default and propagates no state.",
            "",
            "## Frozen experiment",
            "",
            f"- committed `{PROFILE}` base/response history through `2e-4 s`",
            "- continuation to `1e-3 s`; common outputs every `1e-4 s`",
            "- no new BDF1 startup; committed `1.8e-4/2e-4 s` history",
            "- maximum main step `1e-4 s`, reached only through ratio `<=2`",
            "- restart/replay from `6e-4 s`",
            "- strict `dt<=5e-5 s` shadow over `8e-4-1e-3 s`",
            "- main local/summed error budgets remain `2.5e-4` / `5e-3`",
            "- correct active coupling face: `48`",
            "",
            "## Scope",
            "",
            "The horizon is about "
            f"`{manifest['clock_fractions']['N128_cell_crossing']:.4f}` of "
            "one N128 cell-crossing time. A pass authorizes only the "
            "definitions-only `5e-3 s` third-rung manifest. Fixed-Q and "
            "reduced evolution remain blocked.",
            "",
        ]
    )


def main() -> int:
    parent, first_manifest, parent_provenance = _validate_parent()
    manifest = _manifest(first_manifest)
    if (
        not np.array_equal(
            np.asarray(manifest["output_times_seconds"]), OUTPUT_TIMES_SECONDS
        )
        or not np.array_equal(
            np.asarray(manifest["continuation_output_times_seconds"]),
            CONTINUATION_OUTPUT_TIMES_SECONDS,
        )
        or not np.any(
            np.isclose(
                OUTPUT_TIMES_SECONDS,
                RESTART_TIME_SECONDS,
                rtol=0.0,
                atol=1e-18,
            )
        )
        or not np.any(
            np.isclose(
                OUTPUT_TIMES_SECONDS,
                STRICT_SHADOW_START_SECONDS,
                rtol=0.0,
                atol=1e-18,
            )
        )
        or manifest["main_controller"]["coupling_face_contract"][LAYOUT]
        != COUPLING_FACE
    ):
        raise RuntimeError("c2a frozen contract is internally inconsistent")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profile": PROFILE,
        "horizon_seconds": HORIZON_SECONDS,
        "continuation_start_seconds": CONTINUATION_START_SECONDS,
        "output_times_seconds": OUTPUT_TIMES_SECONDS,
        "restart_time_seconds": RESTART_TIME_SECONDS,
        "strict_shadow_start_seconds": STRICT_SHADOW_START_SECONDS,
        "coupling_face": COUPLING_FACE,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "authorized_next": manifest["authorized_next"],
        "parent_classification_preserved": parent["classification"],
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "second_duration_rung_propagation_authorized": True,
        "later_duration_rungs_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "manifest_sha256": causal_canonical_json_sha256(_plain(manifest)),
    }
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "parent_arrays": PARENT_DIRECTORY / "decisive_arrays.npz",
        "parent_provenance": PARENT_DIRECTORY / "provenance.json",
        "first_rung_manifest": c1a.MANIFEST_PATH,
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DIAGNOSTIC ONLY",
            "command": (
                "PYTHONPATH=src:scripts /Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value(
                "rev-parse", "HEAD^{tree}"
            ),
            "parent_implementation_source_hashes": parent_provenance.get(
                "implementation_source_hashes", {}
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": _source_identity(),
            "input_hashes": {
                name: _sha256(path) for name, path in input_paths.items()
            },
        },
    )
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")
    names = (
        "config.json",
        "second_duration_rung_manifest.json",
        "summary.json",
        "provenance.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
