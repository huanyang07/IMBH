#!/usr/bin/env python3
"""Freeze the first nonlinear duration-rung experiment without propagation."""

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

import run_causal_inner_nonlinear_duration_controller_manifest_wp10c9d6c7c3b5a as c3b5a  # noqa: E402
import run_causal_inner_nonlinear_duration_controller_validation_wp10c9d6c7c3b5b as c3b5b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c1a"
ANALYZED_BASE_COMMIT = "f509d8231ef4837e249b79adad4ad53d8dab2885"
ANALYZED_BASE_PARENT = "d2ece4ec850905e6e5ae7a673dde48fa6b414c99"
ANALYZED_BASE_TREE = "d7802fe44204d05bb00d6fa552164a40c02cdad4"

HORIZON_SECONDS = 2.0e-4
OUTPUT_TIMES_SECONDS = np.linspace(0.0, HORIZON_SECONDS, 11)
RESTART_TIME_SECONDS = 1.0e-4
STRICT_SHADOW_START_SECONDS = 1.6e-4
LAYOUT = c3b5a.CONTROLLER_LAYOUT
PROFILE = c3b5a.CONTROLLER_PROFILE
COUPLING_FACE = 48

ARTIFACT = (
    "causal_inner_nonlinear_first_duration_rung_manifest_"
    "wp10c9d6c7c3b5c1a"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_first_duration_rung_manifest_"
    "wp10c9d6c7c3b5c1a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_first_duration_rung_manifest_"
    "wp10c9d6c7c3b5c1a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_FIRST_DURATION_RUNG_MANIFEST_"
    "WP10C9D6C7C3B5C1A_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "first_duration_rung_manifest.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c3b5b.CANONICAL_DIRECTORY


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
        for path in (THIS_RUNNER, THIS_TEST, c3b5b.THIS_RUNNER, c3b5a.THIS_RUNNER)
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    controller_manifest = _read_json(c3b5a.MANIFEST_PATH)
    if (
        parent["classification"]
        != "short_horizon_variable_step_controller_certified_"
        "first_duration_rung_manifest_authorized"
        or not parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c1a_first_duration_rung_manifest"
        or parent["first_duration_rung_propagation_authorized"]
        or controller_manifest["propagation_executed"]
    ):
        raise RuntimeError("c1a authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c1a analyzed identity changed")
    return parent, controller_manifest


def _manifest(controller_manifest: dict) -> dict:
    main_controller = controller_manifest["controller_contract"]
    strict_controller = json.loads(json.dumps(main_controller))
    strict_controller["maximum_timestep_seconds"] = 1.0e-5
    strict_controller["error_estimator"]["local_tolerance"] = 3.125e-5
    strict_controller["error_estimator"][
        "rung_sum_of_accepted_error_estimates"
    ] = 6.25e-4
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "first_nonlinear_duration_rung_manifest_frozen_"
            "two_e_minus_four_second_propagation_authorized"
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
        "restart_time_seconds": RESTART_TIME_SECONDS,
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
        "physical_readiness_gates": {
            "maximum_h_over_r": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
        },
        "reference_clocks_seconds": controller_manifest[
            "reference_clocks_seconds"
        ],
        "clock_fractions": {
            "N128_cell_crossing": HORIZON_SECONDS
            / controller_manifest["reference_clocks_seconds"][
                "minimum_N128_cell_characteristic_crossing"
            ],
            "stress_relaxation": HORIZON_SECONDS
            / controller_manifest["reference_clocks_seconds"][
                "minimum_stress_relaxation"
            ],
        },
        "estimated_execution": {
            "main_step_comparisons_per_trajectory": 12,
            "restart_replay_comparisons_per_trajectory": 5,
            "strict_shadow_comparisons_per_trajectory": 4,
            "trajectory_count": 2,
            "estimated_wall_hours": 6.5,
            "checkpoint_after_each_complete_trajectory": True,
        },
        "binding_decision": {
            "pass": "WP10c9d6c7c3b5c2a_second_duration_rung_manifest",
            "fail_method_or_accuracy": (
                "WP10c9d6c7c3b5c1b_first_duration_rung_localization"
            ),
            "no_automatic_gate_relaxation": True,
        },
        "stage_authorization": {
            "authorized_now": (
                "WP10c9d6c7c3b5c1_first_duration_rung_propagation"
            ),
            "later_duration_rungs_authorized_now": False,
            "fixed_q_authorized_now": False,
            "reduced_evolution_authorized_now": False,
        },
        "hard_stops": [
            "no face-index alias; use active coarse coupling face 48",
            "no tolerance, output, profile or shadow tuning after propagation",
            "no second duration rung before the first rung passes",
            "no spatial-operator or interface redesign",
            "no fixed-Q experiment or reduced evolution",
            "no tide, wind, hot-state, S-curve or QPE-cycle physics",
            "no N1024 rescue",
        ],
        "authorized_next": (
            "WP10c9d6c7c3b5c1_first_duration_rung_propagation"
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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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
            "# First nonlinear duration-rung manifest WP10c9d6c7c3b5c1a",
            "",
            "## Classification",
            "",
            f"`{manifest['classification']}`",
            "",
            "This definitions-only package freezes the first geometric duration "
            "rung. It changes no operator or production default and propagates "
            "no state.",
            "",
            "## Frozen experiment",
            "",
            f"- coarse background plus `{PROFILE}` response",
            f"- horizon `{HORIZON_SECONDS:.1e} s`; outputs every `2e-5 s`",
            f"- restart/replay from `{RESTART_TIME_SECONDS:.1e} s`",
            f"- strict shadow over `{STRICT_SHADOW_START_SECONDS:.1e}-"
            f"{HORIZON_SECONDS:.1e} s` with maximum step `1e-5 s`",
            "- main local/summed error budgets: `2.5e-4` / `5e-3`",
            "- correct active coupling face: `48`",
            "",
            "## Scope",
            "",
            "The horizon is only a controller-depth/restart rung: about "
            f"`{manifest['clock_fractions']['N128_cell_crossing']:.4f}` of one "
            "N128 cell-crossing time. A pass authorizes only the definitions-only "
            "`1e-3 s` second-rung manifest. Fixed-Q and reduced evolution remain "
            "blocked.",
            "",
        ]
    )


def main() -> int:
    parent, controller_manifest = _validate_parent()
    manifest = _manifest(controller_manifest)
    if (
        not np.array_equal(
            np.asarray(manifest["output_times_seconds"]), OUTPUT_TIMES_SECONDS
        )
        or RESTART_TIME_SECONDS not in OUTPUT_TIMES_SECONDS
        or STRICT_SHADOW_START_SECONDS not in OUTPUT_TIMES_SECONDS
        or manifest["main_controller"]["coupling_face_contract"][LAYOUT]
        != COUPLING_FACE
    ):
        raise RuntimeError("c1a frozen contract is internally inconsistent")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profile": PROFILE,
        "horizon_seconds": HORIZON_SECONDS,
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
        "first_duration_rung_propagation_authorized": True,
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
        "controller_manifest": c3b5a.MANIFEST_PATH,
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
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
        "first_duration_rung_manifest.json",
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
