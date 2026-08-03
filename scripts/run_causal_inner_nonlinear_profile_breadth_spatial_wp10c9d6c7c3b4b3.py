#!/usr/bin/env python3
"""Run the frozen middle/fine held-out nonlinear spatial confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_coarse_screen_wp10c9d6c7c3b4b1 as c3b4b1  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as c3b4a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_temporal_wp10c9d6c7c3b4b2 as c3b4b2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_manifest_wp10c9d6c7c3b2a as c3b2a  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_temporal_coarse_screen_wp10c9d6c7c3b3b1 as c3b3b1  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b4b3"
ANALYZED_BASE_COMMIT = "29ad409d915a9602b3ef550630ba881fb01046a0"
ANALYZED_BASE_PARENT = "abb848d08c181f77b477575baba116c823569d6f"
ANALYZED_BASE_TREE = "90db2e2f8a2c0a5d86a65d5396efd9318346e3cb"

LAYOUTS = tuple(c3b4a.LAYOUTS)
COARSE_LAYOUT = LAYOUTS[0]
NEW_LAYOUTS = LAYOUTS[1:]
PROFILES = tuple(c3b4a.PROFILE_NAMES)
TIMESTEP_SECONDS = 1.0e-5
HORIZON_SECONDS = c3b4a.HORIZON_SECONDS
OUTPUT_TIMES_SECONDS = np.asarray(c3b4a.COMMON_OUTPUT_TIMES_SECONDS)
PROFILE_KIND = "primary_physical"

ARTIFACT = (
    "causal_inner_nonlinear_profile_breadth_spatial_"
    "wp10c9d6c7c3b4b3"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_profile_breadth_spatial_"
    "wp10c9d6c7c3b4b3.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_profile_breadth_spatial_"
    "wp10c9d6c7c3b4b3.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_PROFILE_BREADTH_SPATIAL_"
    "WP10C9D6C7C3B4B3_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c3b4b2.CANONICAL_DIRECTORY
COARSE_DIRECTORY = c3b4b1.CANONICAL_DIRECTORY
MANIFEST_DIRECTORY = c3b4a.CANONICAL_DIRECTORY
BASE_DIRECTORY = c3b2b.BASE_DIRECTORY
SPATIAL_MANIFEST_DIRECTORY = c3b2a.CANONICAL_DIRECTORY
SPATIAL_PILOT_DIRECTORY = c3b2b.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_JSON = CHECKPOINT_DIRECTORY / "progress.json"
CHECKPOINT_ARRAYS = CHECKPOINT_DIRECTORY / "progress_arrays.npz"


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


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
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        c3b4b1.THIS_RUNNER,
        c3b4b2.THIS_RUNNER,
        c3b2b.THIS_RUNNER,
        c3b3b1.THIS_RUNNER,
    )
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    breadth_manifest = _read_json(c3b4a.MANIFEST_PATH)
    spatial_manifest = _read_json(
        SPATIAL_MANIFEST_DIRECTORY / "nonlinear_spatial_export_manifest.json"
    )
    if (
        not parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b4b3_middle_fine_heldout_profile_spatial_confirmation"
        or not parent["middle_fine_heldout_spatial_confirmation_authorized"]
        or parent["heldout_spatial_convergence_certified"]
        or parent["long_nonlinear_physical_ladder_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("b4b2 spatial authorization changed")
    stage = breadth_manifest["campaign_controller"]["stages"][2]
    if (
        stage["work_package"] != WORK_PACKAGE
        or tuple(stage["layouts"]) != NEW_LAYOUTS
        or tuple(stage["profiles"]) != PROFILES
        or stage["timestep_seconds"] != TIMESTEP_SECONDS
        or not stage["stop_on_any_state_or_Tier_I_spatial_failure"]
    ):
        raise RuntimeError("b4a middle/fine spatial stage changed")
    if spatial_manifest["classification"] != (
        "nonlinear_short_horizon_spatial_export_manifest_frozen_"
        "canonical_response_pilot_authorized"
    ):
        raise RuntimeError("spatial response contract changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("b4b3 analyzed identity changed")
    return parent, breadth_manifest, spatial_manifest


def _task_id(layout: str, profile: str) -> str:
    return f"{layout}__{profile}__p1__dt_1e-5"


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": _source_identity(),
                "completed_tasks": [],
                "trajectory_reports": {},
                "export_audits": {},
                "failed": False,
            },
            {},
        )
    progress = _read_json(CHECKPOINT_JSON)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != _source_identity()
    ):
        raise RuntimeError("saved b4b3 progress belongs to different code")
    return progress, _load_npz(CHECKPOINT_ARRAYS)


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _run_middle_fine(
    configurations: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    progress, arrays = _load_progress()
    completed = set(progress["completed_tasks"])
    packets = _load_npz(c3b4a.DECISIVE_ARRAYS)
    original_work_package = c3b3b1.WORK_PACKAGE
    c3b3b1.WORK_PACKAGE = WORK_PACKAGE
    try:
        for layout in NEW_LAYOUTS:
            configuration = configurations[layout]
            pending = [
                profile
                for profile in PROFILES
                if _task_id(layout, profile) not in completed
            ]
            if not pending:
                continue
            print(f"b4b3: build tangent {layout}", flush=True)
            started = time.perf_counter()
            tangent = causal_five_field_monolithic_frozen_tangent(
                configuration["context"],
                configuration["base"],
                primitive_column_scales=configuration["columns"],
                conservation_row_scales=configuration["rows"],
            )
            print(
                f"b4b3: tangent {layout} built in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )
            for profile in pending:
                task = _task_id(layout, profile)
                packet = np.asarray(
                    packets[f"{profile}__{layout}__{PROFILE_KIND}"],
                    dtype=float,
                )
                initial = np.asarray(configuration["base"], dtype=float) + packet
                report, trajectory = c3b3b1._trajectory(
                    configuration,
                    tangent,
                    initial,
                    TIMESTEP_SECONDS,
                    f"{profile}__p1",
                )
                for name, values in trajectory.items():
                    arrays[f"{task}__{name}"] = np.asarray(values)
                if report["passed"]:
                    exports, audit = c3b3b1._export_history(
                        configuration["context"], trajectory["states"]
                    )
                    arrays[f"{task}__direct_exports"] = exports
                    progress["export_audits"][task] = audit
                    report["export_audit_passed"] = audit["passed"]
                    report["passed"] = bool(report["passed"] and audit["passed"])
                else:
                    report["export_audit_passed"] = False
                progress["trajectory_reports"][task] = report
                completed.add(task)
                progress["completed_tasks"] = sorted(completed)
                progress["failed"] = bool(
                    progress["failed"] or not report["passed"]
                )
                _save_progress(progress, arrays)
                if not report["passed"]:
                    return progress, arrays
    finally:
        c3b3b1.WORK_PACKAGE = original_work_package
    return progress, arrays


def _spatial_geometry():
    old_inputs = c3b2b._input_arrays()
    parent_grid, layouts, _ = c3b2b._layouts_and_contexts(old_inputs)
    return parent_grid, layouts


def _base_exports(configurations: dict) -> dict[str, np.ndarray]:
    base = _load_npz(BASE_DIRECTORY / "decisive_arrays.npz")
    result = {}
    for layout in LAYOUTS:
        values, audit = c3b3b1._export_history(
            configurations[layout]["context"],
            np.asarray(base[f"{layout}__states"], dtype=float),
        )
        if not audit["passed"]:
            raise RuntimeError(f"committed base export audit failed for {layout}")
        result[layout] = values
    return result


def _histories(
    profile: str,
    progress_arrays: dict[str, np.ndarray],
    base_exports: dict[str, np.ndarray],
    layouts: dict,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    coarse = _load_npz(COARSE_DIRECTORY / "decisive_arrays.npz")
    base = _load_npz(BASE_DIRECTORY / "decisive_arrays.npz")
    states = []
    instantaneous = []
    cumulative = []
    for layout in LAYOUTS:
        if layout == COARSE_LAYOUT:
            task = c3b4b1._task_id(profile)
            perturbed_states = coarse[f"{task}__states"]
            perturbed_exports = coarse[f"{task}__direct_exports"]
        else:
            task = _task_id(layout, profile)
            perturbed_states = progress_arrays[f"{task}__states"]
            perturbed_exports = progress_arrays[f"{task}__direct_exports"]
        response = perturbed_states - base[f"{layout}__states"]
        restricted = np.asarray(
            [
                restrict_causal_embedded_patch_cell_averages(
                    state, layouts[layout]
                )
                for state in response
            ],
            dtype=float,
        )
        export_response = perturbed_exports - base_exports[layout]
        states.append(restricted)
        instantaneous.append(export_response)
        cumulative.append(c3b2b._cumulative(export_response))
    return tuple(states), tuple(instantaneous), tuple(cumulative)


def _analyze(
    progress: dict,
    progress_arrays: dict[str, np.ndarray],
    configurations: dict,
    spatial_manifest: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    parent_grid, layouts = _spatial_geometry()
    pilot = _load_npz(SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz")
    field_scales = np.asarray(pilot["field_scales"], dtype=float)
    observable_scales = np.asarray(
        pilot["fixed_physical_observable_scales"], dtype=float
    )
    gates = spatial_manifest["tier_I_binding_contract"]["gates"]
    base_exports = _base_exports(configurations)
    case_reports = {}
    decisive = {
        "times_seconds": OUTPUT_TIMES_SECONDS,
        "field_scales": field_scales,
        "fixed_physical_observable_scales": observable_scales,
    }
    all_passed = not progress["failed"]
    for profile in PROFILES:
        state, instantaneous, cumulative = _histories(
            profile, progress_arrays, base_exports, layouts
        )
        state_metric = c3b2b._state_metrics(
            state, parent_grid, field_scales, gates
        )
        instant_raw = c3b2b._packet_metrics(
            instantaneous, observable_scales, gates
        )
        cumulative_raw = c3b2b._packet_metrics(
            cumulative, observable_scales * HORIZON_SECONDS, gates
        )
        instant_metric = c3b2b._metric_payload(instant_raw)
        cumulative_metric = c3b2b._metric_payload(cumulative_raw)
        passed = bool(
            state_metric["passed"]
            and instant_metric["passed"]
            and cumulative_metric["passed"]
        )
        case_reports[profile] = {
            "state": state_metric,
            "instantaneous_exports": instant_metric,
            "cumulative_exports": cumulative_metric,
            "passed": passed,
        }
        all_passed = bool(all_passed and passed)
        for layout, state_values, instant_values, cumulative_values in zip(
            LAYOUTS, state, instantaneous, cumulative, strict=True
        ):
            decisive[f"{layout}__{profile}__state_response"] = state_values
            decisive[
                f"{layout}__{profile}__instantaneous_export_response"
            ] = instant_values
            decisive[
                f"{layout}__{profile}__cumulative_export_response"
            ] = cumulative_values
    reports = list(progress["trajectory_reports"].values())
    return (
        {
            "passed": all_passed,
            "case_reports": case_reports,
            "all_new_trajectory_methods_passed": all(
                report["passed"] for report in reports
            ),
            "trajectory_reports": progress["trajectory_reports"],
            "export_audits": progress["export_audits"],
            "maximum_scaled_residual": max(
                report["maximum_scaled_residual"] for report in reports
            ),
            "maximum_discrete_ledger_defect": max(
                report["maximum_discrete_ledger_defect"] for report in reports
            ),
            "all_checkpoint_roundtrips_bitwise": all(
                report["checkpoint_roundtrip_bitwise"] for report in reports
            ),
            "all_split_restart_replays_bitwise": all(
                report["split_restart_replay_bitwise"] for report in reports
            ),
        },
        decisive,
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": status,
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


def _report(summary: dict) -> str:
    screen = summary["spatial_confirmation"]
    lines = [
        "# Nonlinear held-out profile spatial confirmation WP10c9d6c7c3b4b3",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "The five prospectively frozen held-outs were evolved independently "
        "on the middle and fine embedded layouts at `dt=1e-5 s` through "
        "the common `4e-5 s` horizon. The committed coarse and unperturbed "
        "histories were reused by hash.",
        "",
        "## Binding results",
        "",
    ]
    for profile, result in screen["case_reports"].items():
        state = result["state"]
        instant = result["instantaneous_exports"]
        cumulative = result["cumulative_exports"]
        lines.extend(
            [
                f"### `{profile}`",
                "",
                "- state RMS/max/component order: "
                f"`{state['observed_rms_order']:.6f}` / "
                f"`{state['observed_maximum_order']:.6f}` / "
                f"`{state['minimum_significant_component_order']:.6f}`",
                "- state fine difference / history / error cosine: "
                f"`{state['maximum_fine_normalized_difference']:.3e}` / "
                f"`{state['history_cosine']:.9f}` / "
                f"`{state['refinement_error_cosine']:.9f}`",
                "- instantaneous export RMS/max/component order: "
                f"`{instant['observed_rms_order']:.6f}` / "
                f"`{instant['observed_maximum_order']:.6f}` / "
                f"`{instant['minimum_significant_component_order']:.6f}`",
                "- cumulative export RMS/max/component order: "
                f"`{cumulative['observed_rms_order']:.6f}` / "
                f"`{cumulative['observed_maximum_order']:.6f}` / "
                f"`{cumulative['minimum_significant_component_order']:.6f}`",
                f"- result: `{'pass' if result['passed'] else 'fail'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Method",
            "",
            f"- maximum scaled residual: `{screen['maximum_scaled_residual']:.3e}`",
            f"- maximum discrete ledger defect: `{screen['maximum_discrete_ledger_defect']:.3e}`",
            "- all checkpoint roundtrips bitwise: "
            f"`{screen['all_checkpoint_roundtrips_bitwise']}`",
            "- all split/restart replays bitwise: "
            f"`{screen['all_split_restart_replays_bitwise']}`",
            "",
            "## Authorized next",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "This completes the short-horizon temporal/spatial breadth "
            "campaign only. Meaningful-duration evolution, fixed-Q "
            "experiments and reduced slow evolution remain blocked until "
            "a prospective duration controller is frozen and passes.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(
    parent: dict,
    breadth_manifest: dict,
    spatial_manifest: dict,
    progress_arrays: dict[str, np.ndarray],
    screen: dict,
    decisive: dict[str, np.ndarray],
) -> int:
    passed = screen["passed"]
    classification = (
        "heldout_profile_temporal_and_spatial_breadth_certified_variable_step_duration_controller_manifest_authorized"
        if passed
        else "heldout_profile_spatial_confirmation_failed_duration_extension_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b5a_variable_step_duration_controller_manifest"
        if passed
        else "WP10c9d6c7c3b4b3_spatial_failure_localization"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "new_layouts": list(NEW_LAYOUTS),
        "profiles": list(PROFILES),
        "timestep_seconds": TIMESTEP_SECONDS,
        "horizon_seconds": HORIZON_SECONDS,
        "spatial_gates": spatial_manifest["tier_I_binding_contract"]["gates"],
    }
    combined = {**progress_arrays, **decisive}
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_classification": parent["classification"],
        "spatial_confirmation": screen,
        "coarse_heldout_temporal_convergence_certified": True,
        "heldout_spatial_convergence_certified": passed,
        "short_horizon_profile_breadth_certified": passed,
        "meaningfully_nonlinear_dynamics_certified": False,
        "variable_step_duration_controller_manifest_authorized": passed,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(config),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values) for name, values in combined.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "parent_arrays": c3b4b2.DECISIVE_ARRAYS,
        "coarse_summary": COARSE_DIRECTORY / "summary.json",
        "coarse_arrays": c3b4b1.DECISIVE_ARRAYS,
        "breadth_manifest": c3b4a.MANIFEST_PATH,
        "background_arrays": BASE_DIRECTORY / "decisive_arrays.npz",
        "spatial_manifest": SPATIAL_MANIFEST_DIRECTORY
        / "nonlinear_spatial_export_manifest.json",
        "spatial_pilot_arrays": SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz",
    }
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
                name: _sha256(path) for name, path in input_paths.items()
            },
        },
    )
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parent, breadth_manifest, spatial_manifest = _validate_parent()
    configurations = c3b1a._configurations()
    progress, progress_arrays = _run_middle_fine(configurations)
    if progress["failed"]:
        reports = list(progress["trajectory_reports"].values())
        screen = {
            "passed": False,
            "case_reports": {},
            "trajectory_reports": progress["trajectory_reports"],
            "export_audits": progress["export_audits"],
            "maximum_scaled_residual": max(
                report["maximum_scaled_residual"] for report in reports
            ),
            "maximum_discrete_ledger_defect": max(
                report["maximum_discrete_ledger_defect"] for report in reports
            ),
            "all_checkpoint_roundtrips_bitwise": all(
                report["checkpoint_roundtrip_bitwise"] for report in reports
            ),
            "all_split_restart_replays_bitwise": all(
                report["split_restart_replay_bitwise"] for report in reports
            ),
        }
        decisive = {}
    else:
        screen, decisive = _analyze(
            progress,
            progress_arrays,
            configurations,
            spatial_manifest,
        )
    return _package(
        parent,
        breadth_manifest,
        spatial_manifest,
        progress_arrays,
        screen,
        decisive,
    )


if __name__ == "__main__":
    raise SystemExit(main())
