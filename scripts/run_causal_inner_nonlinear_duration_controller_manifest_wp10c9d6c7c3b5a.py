#!/usr/bin/env python3
"""Freeze the fail-fast variable-step monolithic duration controller."""

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

import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as c3b4a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_export_face_audit_wp10c9d6c7c3b4d as c3b4d  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_bdf import (  # noqa: E402
    CAUSAL_BDF2_MAXIMUM_STEP_RATIO,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5a"
ANALYZED_BASE_COMMIT = "e3c4550ed5588db93ca0784a5c7827f2d07c590f"
ANALYZED_BASE_PARENT = "ca39da8fc5e54f0745c8261743b277ebfcad05fc"
ANALYZED_BASE_TREE = "65f71fd80c673881756c7ce49b83fe403e74554c"

ARTIFACT = (
    "causal_inner_nonlinear_duration_controller_manifest_"
    "wp10c9d6c7c3b5a"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_duration_controller_manifest_"
    "wp10c9d6c7c3b5a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_duration_controller_manifest_"
    "wp10c9d6c7c3b5a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_DURATION_CONTROLLER_MANIFEST_"
    "WP10C9D6C7C3B5A_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c3b4d.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "duration_controller_manifest.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LAYOUTS = tuple(c3b4a.LAYOUTS)
CONTROLLER_LAYOUT = LAYOUTS[0]
CONTROLLER_PROFILE = "p3_buffer45__generic_five_field"
VALIDATION_HORIZON_SECONDS = 4.0e-5
INITIAL_TIMESTEP_SECONDS = 2.5e-6
MINIMUM_TIMESTEP_SECONDS = 1.25e-6
MAXIMUM_TIMESTEP_SECONDS = 2.0e-5
DURATION_RUNGS_SECONDS = (2.0e-4, 1.0e-3, 5.0e-3, 2.0e-2, 5.0e-2, 1.25e-1)
REFERENCE_CLOCKS_SECONDS = {
    "minimum_N128_cell_characteristic_crossing": 5.5433e-3,
    "minimum_radial_advection": 1.4029e-1,
    "minimum_stress_relaxation": 1.4705e-1,
    "minimum_luminosity_response": 1.1384,
    "minimum_thermal_response": 4.5538,
    "global_loading": 8.48e5,
}


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
        for path in (THIS_RUNNER, THIS_TEST, c3b4d.THIS_RUNNER)
        if (ROOT / path).exists()
    }


def _validate_parent() -> dict:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    if (
        parent["classification"]
        != "heldout_spatial_export_failure_caused_by_active_face_alias_"
        "corrected_physical_face_contract_passes"
        or not parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5a_variable_step_duration_controller_manifest"
        or not parent["heldout_spatial_convergence_certified"]
        or not parent["variable_step_duration_controller_manifest_authorized"]
        or parent["long_nonlinear_physical_ladder_authorized"]
    ):
        raise RuntimeError("b5a authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("b5a analyzed identity changed")
    return parent


def _controller_contract() -> dict:
    return {
        "method": "monolithic_increment_primary_variable_step_BDF1_BDF2",
        "initial_timestep_seconds": INITIAL_TIMESTEP_SECONDS,
        "minimum_timestep_seconds": MINIMUM_TIMESTEP_SECONDS,
        "maximum_timestep_seconds": MAXIMUM_TIMESTEP_SECONDS,
        "maximum_BDF2_step_ratio": 2.0,
        "analytic_stability_bound": float(CAUSAL_BDF2_MAXIMUM_STEP_RATIO),
        "error_estimator": {
            "kind": "one_full_step_versus_two_independent_half_steps",
            "formal_order": 2,
            "accepted_branch": "full_step",
            "accepted_branch_error_multiplier": 4.0 / 3.0,
            "state_norm": "maximum_of_scaled_RMS_and_scaled_fieldwise_maximum",
            "export_norm": "maximum_fixed_scale_Tier_I_endpoint_difference",
            "local_error": "maximum_of_state_and_Tier_I_export_estimates",
            "local_tolerance": 2.5e-4,
            "short_horizon_sum_of_accepted_error_estimates": 5.0e-3,
        },
        "proposal": {
            "safety_factor": 0.8,
            "error_exponent": 1.0 / 3.0,
            "minimum_factor": 0.5,
            "maximum_factor": 2.0,
            "maximum_retries": 8,
            "reject_on_any_full_or_half_method_failure": True,
            "exact_landing_at_every_declared_output": True,
        },
        "step_method_gates": {
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure": 1.0e-9,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
            "checkpoint_roundtrip": "bitwise",
            "split_restart_replay": "bitwise_at_declared_outputs",
        },
        "coupling_face_contract": {
            layout: index
            for layout, index in zip(LAYOUTS, (48, 96, 192), strict=True)
        },
    }


def _validation_contract() -> dict:
    spatial_manifest = _read_json(
        c3b4d.c3b4b3.SPATIAL_MANIFEST_DIRECTORY
        / "nonlinear_spatial_export_manifest.json"
    )
    return {
        "layout": CONTROLLER_LAYOUT,
        "profile": CONTROLLER_PROFILE,
        "background_and_perturbed_trajectories_required": True,
        "horizon_seconds": VALIDATION_HORIZON_SECONDS,
        "output_times_seconds": c3b4a.COMMON_OUTPUT_TIMES_SECONDS.tolist(),
        "independent_reference": {
            "kind": "committed_fixed_step_BDF1_BDF2",
            "timestep_seconds": 2.5e-6,
            "artifact": (
                "causal_inner_nonlinear_profile_breadth_temporal_"
                "wp10c9d6c7c3b4b2"
            ),
        },
        "state_and_Tier_I_gates": spatial_manifest[
            "tier_I_binding_contract"
        ]["gates"],
        "maximum_controller_to_reference_scaled_state_difference": 5.0e-3,
        "maximum_controller_to_reference_scaled_Tier_I_difference": 5.0e-3,
        "minimum_history_cosine": 0.90,
        "fixed_physical_export_scales": True,
        "correct_active_coupling_face_required": True,
        "no_profile_or_gate_tuning_after_manifest": True,
    }


def _duration_ladder() -> list[dict]:
    return [
        {
            "work_package": "WP10c9d6c7c3b5c1",
            "horizon_seconds": 2.0e-4,
            "scope": "coarse background plus generic five-field response",
            "binding_clock": "controller_depth_and_restart",
        },
        {
            "work_package": "WP10c9d6c7c3b5c2",
            "horizon_seconds": 1.0e-3,
            "scope": "coarse generic response plus strict-controller shadow",
            "binding_clock": "sub_cell_crossing",
        },
        {
            "work_package": "WP10c9d6c7c3b5c3",
            "horizon_seconds": 5.0e-3,
            "scope": "coarse/middle/fine generic response and held-out coarse controls",
            "binding_clock": "approximately_one_N128_cell_crossing",
        },
        {
            "work_package": "WP10c9d6c7c3b5c4",
            "horizon_seconds": 2.0e-2,
            "scope": "middle fail-fast physical duration screen",
            "binding_clock": "multiple_cell_crossings",
        },
        {
            "work_package": "WP10c9d6c7c3b5c5",
            "horizon_seconds": 5.0e-2,
            "scope": "spatial/temporal Tier-I breadth certification",
            "binding_clock": "one_third_stress_relaxation",
        },
        {
            "work_package": "WP10c9d6c7c3b5c6",
            "horizon_seconds": 1.25e-1,
            "scope": "conditional truth-model fast-horizon certification",
            "binding_clock": "order_one_stress_relaxation",
        },
    ]


def _manifest() -> dict:
    controller = _controller_contract()
    validation = _validation_contract()
    ladder = _duration_ladder()
    passed = bool(
        controller["maximum_BDF2_step_ratio"]
        <= controller["analytic_stability_bound"]
        and CONTROLLER_PROFILE in c3b4a.PROFILE_NAMES
        and tuple(item["horizon_seconds"] for item in ladder)
        == DURATION_RUNGS_SECONDS
        and validation["correct_active_coupling_face_required"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "variable_step_monolithic_duration_controller_manifest_frozen_"
            "short_horizon_controller_validation_authorized"
            if passed
            else "duration_controller_manifest_invalid_duration_work_blocked"
        ),
        "passed": passed,
        "operator_changed": False,
        "production_defaults_changed": False,
        "propagation_executed": False,
        "controller_contract": controller,
        "short_horizon_validation_contract": validation,
        "duration_ladder": ladder,
        "reference_clocks_seconds": REFERENCE_CLOCKS_SECONDS,
        "stage_authorization": {
            "authorized_now": "WP10c9d6c7c3b5b_short_horizon_variable_step_controller_validation",
            "duration_rungs_authorized_now": False,
            "each_later_rung_requires_previous_binding_pass": True,
            "each_rung_freezes_its_outputs_profiles_and_strict_shadow_before_propagation": True,
        },
        "hard_stops": [
            "no duration rung before the short-horizon controller validates",
            "no face-index alias: every export evaluation requires the active coupling face explicitly",
            "no step larger than the frozen maximum or BDF2 stability ratio",
            "no tolerance or profile tuning after observing propagation",
            "no spatial-operator or interface redesign",
            "no fixed-Q experiment or reduced evolution",
            "no tide, wind, hot-state, S-curve or QPE-cycle physics",
            "no N1024 rescue",
        ],
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5b_short_horizon_variable_step_controller_validation"
            if passed
            else "none"
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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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


def _report(manifest: dict) -> str:
    controller = manifest["controller_contract"]
    lines = [
        "# Nonlinear variable-step duration-controller manifest WP10c9d6c7c3b5a",
        "",
        "## Classification",
        "",
        f"`{manifest['classification']}`",
        "",
        "This is a definitions-only package. It changes no operator or production "
        "default and propagates no trajectory.",
        "",
        "## Frozen controller",
        "",
        f"- initial/minimum/maximum step: `{controller['initial_timestep_seconds']:.3e}` / "
        f"`{controller['minimum_timestep_seconds']:.3e}` / "
        f"`{controller['maximum_timestep_seconds']:.3e} s`",
        "- estimator: one full BDF step versus two independently executed half steps",
        "- accepted branch: full step; error multiplier: `4/3`",
        "- local state/Tier-I tolerance: `2.5e-4`",
        "- maximum proposed growth: `2`; analytic BDF2 limit: "
        f"`{controller['analytic_stability_bound']:.9f}`",
        "- every export call must receive the active-grid coupling face explicitly",
        "",
        "## First authorized propagation",
        "",
        f"Layout `{CONTROLLER_LAYOUT}`, profile `{CONTROLLER_PROFILE}`, background "
        f"plus perturbed trajectories, through `{VALIDATION_HORIZON_SECONDS:.1e} s`.",
        "The adaptive response is compared at frozen common times with the already "
        "committed `dt=2.5e-6 s` fixed-step reference.",
        "",
        "## Conditional duration ladder",
        "",
    ]
    for item in manifest["duration_ladder"]:
        lines.append(
            f"- `{item['work_package']}`: `{item['horizon_seconds']:.3e} s` — "
            f"{item['scope']}"
        )
    lines.extend(
        [
            "",
            "No duration rung is authorized until the short-horizon controller "
            "matches the independent fixed-step reference. Every later rung requires "
            "a fresh definitions-only manifest and the previous rung's binding pass.",
            "",
            "## Authorized next",
            "",
            f"`{manifest['authorized_next']}`",
            "",
            "Fixed-Q experiments and reduced slow evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(parent: dict, manifest: dict) -> int:
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "controller_layout": CONTROLLER_LAYOUT,
        "controller_profile": CONTROLLER_PROFILE,
        "validation_horizon_seconds": VALIDATION_HORIZON_SECONDS,
        "duration_rungs_seconds": list(DURATION_RUNGS_SECONDS),
        "propagation_executed": False,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    summary = dict(manifest)
    summary.update(
        {
            "parent_classification_preserved": parent["classification"],
            "manifest_sha256": causal_canonical_json_sha256(_plain(manifest)),
            "config_sha256": causal_canonical_json_sha256(_plain(config)),
        }
    )
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src /Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value("rev-parse", "HEAD^{tree}"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": _source_identity(),
            "input_hashes": {
                "parent_summary": _sha256(PARENT_DIRECTORY / "summary.json"),
            },
        },
    )
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")
    names = (
        "config.json",
        "duration_controller_manifest.json",
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
    return 0 if summary["passed"] else 1


def main() -> int:
    parent = _validate_parent()
    return _package(parent, _manifest())


if __name__ == "__main__":
    raise SystemExit(main())
