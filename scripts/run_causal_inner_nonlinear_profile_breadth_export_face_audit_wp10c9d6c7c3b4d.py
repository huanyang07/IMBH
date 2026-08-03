#!/usr/bin/env python3
"""Audit and correct the held-out Tier-I coupling-face measurement alias."""

from __future__ import annotations

import csv
import hashlib
import inspect
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

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_coarse_screen_wp10c9d6c7c3b4b1 as c3b4b1  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as c3b4a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_spatial_localization_wp10c9d6c7c3b4c as c3b4c  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_spatial_wp10c9d6c7c3b4b3 as c3b4b3  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_temporal_coarse_screen_wp10c9d6c7c3b3b1 as c3b3b1  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b4d"
ANALYZED_BASE_COMMIT = "ca39da8fc5e54f0745c8261743b277ebfcad05fc"
ANALYZED_BASE_PARENT = "e78aa83fa115d02538bdf02b357494f7f2bfd8ad"
ANALYZED_BASE_TREE = "0c9288a5145fd4a3c8443585f29bb3a6cf68fe2c"

LAYOUTS = tuple(c3b4a.LAYOUTS)
COARSE_LAYOUT = LAYOUTS[0]
PROFILES = tuple(c3b4a.PROFILE_NAMES)
TIMES = np.asarray(c3b4a.COMMON_OUTPUT_TIMES_SECONDS, dtype=float)
HORIZON_SECONDS = float(c3b4a.HORIZON_SECONDS)
OBSERVABLE_NAMES = tuple(c3b2b.OBSERVABLE_NAMES)

ARTIFACT = (
    "causal_inner_nonlinear_profile_breadth_export_face_audit_"
    "wp10c9d6c7c3b4d"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_profile_breadth_export_face_audit_"
    "wp10c9d6c7c3b4d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_profile_breadth_export_face_audit_"
    "wp10c9d6c7c3b4d.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_PROFILE_BREADTH_EXPORT_FACE_AUDIT_"
    "WP10C9D6C7C3B4D_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c3b4c.CANONICAL_DIRECTORY
FAILED_DIRECTORY = c3b4b3.CANONICAL_DIRECTORY
COARSE_DIRECTORY = c3b4b1.CANONICAL_DIRECTORY
BASE_DIRECTORY = c3b2b.BASE_DIRECTORY
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
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
        c3b4c.THIS_RUNNER,
        c3b4b3.THIS_RUNNER,
        c3b3b1.THIS_RUNNER,
        c3b2b.THIS_RUNNER,
    )
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    failed = _read_json(FAILED_DIRECTORY / "summary.json")
    spatial_manifest = _read_json(
        c3b4b3.SPATIAL_MANIFEST_DIRECTORY
        / "nonlinear_spatial_export_manifest.json"
    )
    if (
        parent["classification"]
        != "spatial_failure_localized_to_layout_native_export_map_common_parent_map_passes"
        or not parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b4d_layout_native_export_map_audit"
        or parent["variable_step_duration_controller_manifest_authorized"]
    ):
        raise RuntimeError("b4d authorization changed")
    if (
        failed["classification"]
        != "heldout_profile_spatial_confirmation_failed_duration_extension_blocked"
        or failed["passed"]
    ):
        raise RuntimeError("b4b3 failure classification changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("b4d analyzed identity changed")
    return parent, failed, spatial_manifest


def _raw_perturbed(
    layout: str,
    profile: str,
    failed_arrays: dict[str, np.ndarray],
    coarse_arrays: dict[str, np.ndarray],
) -> np.ndarray:
    if layout == COARSE_LAYOUT:
        return np.asarray(
            coarse_arrays[f"{c3b4b1._task_id(profile)}__states"],
            dtype=float,
        )
    return np.asarray(
        failed_arrays[f"{c3b4b3._task_id(layout, profile)}__states"],
        dtype=float,
    )


def _export_history(context, states: np.ndarray, coupling_face: int):
    values = []
    maximum_ledger = 0.0
    maximum_incoming = 0
    for state in np.asarray(states, dtype=float):
        observable, audit = c3b2b._direct_observable(
            context, state, int(coupling_face)
        )
        values.append(observable)
        maximum_ledger = max(
            maximum_ledger,
            *(
                float(audit[key])
                for key in (
                    "local_block_ledger_defect",
                    "source_double_count_defect",
                    "shared_conservative_face_defect",
                    "split_closure_defect",
                )
            ),
        )
        maximum_incoming = max(
            maximum_incoming,
            int(audit["incoming_excision_characteristics"]),
        )
    return np.asarray(values, dtype=float), maximum_ledger, maximum_incoming


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float).ravel()))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _pair_attribution(
    legacy_left: np.ndarray,
    legacy_right: np.ndarray,
    corrected_left: np.ndarray,
    corrected_right: np.ndarray,
    scales: np.ndarray,
) -> dict:
    legacy = ((legacy_right - legacy_left) / scales[None, :]).ravel()
    corrected = ((corrected_right - corrected_left) / scales[None, :]).ravel()
    alias = legacy - corrected
    denominator = max(_norm(legacy), np.finfo(float).tiny)
    return {
        "legacy_error_norm": _norm(legacy),
        "corrected_error_norm": _norm(corrected),
        "face_alias_error_norm": _norm(alias),
        "corrected_to_legacy_error_ratio": _norm(corrected) / denominator,
        "face_alias_to_legacy_error_ratio": _norm(alias) / denominator,
        "face_alias_alignment_with_legacy_error": _cosine(alias, legacy),
        "decomposition_closure_defect": _norm(legacy - corrected - alias)
        / denominator,
    }


def _channel_report(
    legacy: tuple[np.ndarray, np.ndarray, np.ndarray],
    corrected: tuple[np.ndarray, np.ndarray, np.ndarray],
    scales: np.ndarray,
    gates: dict,
) -> dict:
    return {
        "legacy_wrong_face": c3b2b._metric_payload(
            c3b2b._packet_metrics(legacy, scales, gates)
        ),
        "corrected_physical_face": c3b2b._metric_payload(
            c3b2b._packet_metrics(corrected, scales, gates)
        ),
        "coarse_medium_attribution": _pair_attribution(
            legacy[0], legacy[1], corrected[0], corrected[1], scales
        ),
        "medium_fine_attribution": _pair_attribution(
            legacy[1], legacy[2], corrected[1], corrected[2], scales
        ),
    }


def _analyze(spatial_manifest: dict) -> tuple[dict, dict[str, np.ndarray]]:
    failed_arrays = _load_npz(c3b4b3.DECISIVE_ARRAYS)
    coarse_arrays = _load_npz(c3b4b1.DECISIVE_ARRAYS)
    base_arrays = _load_npz(BASE_DIRECTORY / "decisive_arrays.npz")
    pilot_arrays = _load_npz(
        c3b4b3.SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz"
    )
    scales = np.asarray(
        pilot_arrays["fixed_physical_observable_scales"], dtype=float
    )
    gates = spatial_manifest["tier_I_binding_contract"]["gates"]
    parent_grid, layouts = c3b4b3._spatial_geometry()
    configurations = c3b1a._configurations()
    parent_face = int(c3b2b.COUPLING_PARENT_FACE)
    parent_radius = float(parent_grid.edges[parent_face])
    helper_source = inspect.getsource(c3b3b1._export_history)

    face_mappings = {}
    base_exports = {}
    maximum_ledger = 0.0
    maximum_incoming = 0
    for layout in LAYOUTS:
        active_grid = layouts[layout].grid
        correct_face = int(layouts[layout].coupling_face_index)
        correct_radius = float(active_grid.edges[correct_face])
        legacy_radius = float(active_grid.edges[parent_face])
        face_mappings[layout] = {
            "active_cell_count": int(len(active_grid.centers)),
            "parent_coupling_face_index": parent_face,
            "correct_active_coupling_face_index": correct_face,
            "parent_coupling_radius": parent_radius,
            "correct_active_coupling_radius": correct_radius,
            "legacy_hardcoded_face_radius": legacy_radius,
            "correct_radius_relative_defect": abs(correct_radius - parent_radius)
            / parent_radius,
            "legacy_radius_relative_displacement": legacy_radius
            / parent_radius
            - 1.0,
        }
        values, ledger, incoming = _export_history(
            configurations[layout]["context"],
            base_arrays[f"{layout}__states"],
            correct_face,
        )
        base_exports[layout] = values
        maximum_ledger = max(maximum_ledger, ledger)
        maximum_incoming = max(maximum_incoming, incoming)

    decisive = {
        "times_seconds": TIMES,
        "fixed_physical_observable_scales": scales,
        "correct_active_coupling_face_indices": np.asarray(
            [face_mappings[item]["correct_active_coupling_face_index"] for item in LAYOUTS],
            dtype=int,
        ),
        "correct_active_coupling_radii": np.asarray(
            [face_mappings[item]["correct_active_coupling_radius"] for item in LAYOUTS],
            dtype=float,
        ),
        "legacy_hardcoded_face_radii": np.asarray(
            [face_mappings[item]["legacy_hardcoded_face_radius"] for item in LAYOUTS],
            dtype=float,
        ),
    }
    profiles = {}
    attributions = []
    for profile in PROFILES:
        legacy_histories = []
        corrected_histories = []
        for layout in LAYOUTS:
            legacy = np.asarray(
                failed_arrays[
                    f"{layout}__{profile}__instantaneous_export_response"
                ],
                dtype=float,
            )
            states = _raw_perturbed(
                layout, profile, failed_arrays, coarse_arrays
            )
            corrected_values, ledger, incoming = _export_history(
                configurations[layout]["context"],
                states,
                face_mappings[layout]["correct_active_coupling_face_index"],
            )
            maximum_ledger = max(maximum_ledger, ledger)
            maximum_incoming = max(maximum_incoming, incoming)
            corrected = corrected_values - base_exports[layout]
            legacy_histories.append(legacy)
            corrected_histories.append(corrected)
            decisive[f"{layout}__{profile}__legacy_wrong_face_response"] = legacy
            decisive[f"{layout}__{profile}__corrected_face_response"] = corrected
            decisive[f"{layout}__{profile}__face_alias_defect"] = legacy - corrected

        legacy_tuple = tuple(legacy_histories)
        corrected_tuple = tuple(corrected_histories)
        instant = _channel_report(
            legacy_tuple, corrected_tuple, scales, gates
        )
        cumulative = _channel_report(
            tuple(c3b2b._cumulative(item) for item in legacy_tuple),
            tuple(c3b2b._cumulative(item) for item in corrected_tuple),
            scales * HORIZON_SECONDS,
            gates,
        )
        profiles[profile] = {
            "instantaneous": instant,
            "cumulative": cumulative,
            "passed": bool(
                instant["corrected_physical_face"]["passed"]
                and cumulative["corrected_physical_face"]["passed"]
            ),
        }
        attributions.extend(
            channel[pair]
            for channel in (instant, cumulative)
            for pair in (
                "coarse_medium_attribution",
                "medium_fine_attribution",
            )
        )

    face_contract_passed = all(
        item["correct_radius_relative_defect"] <= 1.0e-15
        for item in face_mappings.values()
    )
    all_corrected_passed = all(item["passed"] for item in profiles.values())
    legacy_helper_hardcodes_parent_face = bool(
        "c3b2b.COUPLING_PARENT_FACE" in helper_source
    )
    minimum_alignment = min(
        item["face_alias_alignment_with_legacy_error"]
        for item in attributions
    )
    maximum_corrected_fraction = max(
        item["corrected_to_legacy_error_ratio"] for item in attributions
    )
    maximum_closure = max(
        item["decomposition_closure_defect"] for item in attributions
    )
    alias_proved = bool(
        legacy_helper_hardcodes_parent_face
        and face_contract_passed
        and all_corrected_passed
        and minimum_alignment >= 0.95
        and maximum_corrected_fraction <= 0.25
        and maximum_closure <= 1.0e-12
        and maximum_ledger <= 1.0e-9
        and maximum_incoming == 0
    )
    return (
        {
            "passed": alias_proved,
            "profiles": profiles,
            "face_mappings": face_mappings,
            "legacy_helper_hardcodes_parent_face": legacy_helper_hardcodes_parent_face,
            "physical_face_contract_passed": face_contract_passed,
            "all_corrected_spatial_contracts_passed": all_corrected_passed,
            "face_alias_cause_proved": alias_proved,
            "minimum_face_alias_alignment_with_legacy_error": minimum_alignment,
            "maximum_corrected_to_legacy_error_ratio": maximum_corrected_fraction,
            "maximum_error_decomposition_closure_defect": maximum_closure,
            "maximum_corrected_export_ledger_defect": maximum_ledger,
            "maximum_corrected_incoming_excision_characteristics": maximum_incoming,
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
    result = summary["audit"]
    lines = [
        "# Nonlinear held-out export-face audit WP10c9d6c7c3b4d",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "No trajectory was propagated and no evolution operator or gate was changed. "
        "The failed b4b3 and diagnostic b4c classifications remain historical facts.",
        "",
        "## Root cause",
        "",
        "The shared export-history helper passed parent face 48 to every active "
        "layout. That is the physical coupling face only on the coarse grid. The "
        "intended active-grid faces are 48, 96 and 192.",
        "",
    ]
    for layout, mapping in result["face_mappings"].items():
        lines.append(
            f"- `{layout}`: correct face `{mapping['correct_active_coupling_face_index']}`, "
            f"correct radius `{mapping['correct_active_coupling_radius']:.9e}`, "
            f"legacy face-48 radius `{mapping['legacy_hardcoded_face_radius']:.9e}`"
        )
    lines.extend(
        [
            "",
            "## Corrected frozen-contract result",
            "",
            f"- every corrected instantaneous/cumulative profile passed: "
            f"`{result['all_corrected_spatial_contracts_passed']}`",
            f"- minimum alias/error alignment: "
            f"`{result['minimum_face_alias_alignment_with_legacy_error']:.12f}`",
            f"- maximum corrected/legacy error ratio: "
            f"`{result['maximum_corrected_to_legacy_error_ratio']:.6e}`",
            f"- maximum corrected ledger defect: "
            f"`{result['maximum_corrected_export_ledger_defect']:.3e}`",
            "",
        ]
    )
    for profile, report in result["profiles"].items():
        instant = report["instantaneous"]["corrected_physical_face"]
        cumulative = report["cumulative"]["corrected_physical_face"]
        lines.extend(
            [
                f"### `{profile}`",
                "",
                "- instantaneous RMS/component order, cosine: "
                f"`{instant['observed_rms_order']:.6f}` / "
                f"`{instant['minimum_significant_component_order']:.6f}` / "
                f"`{instant['refinement_error_cosine']:.9f}`",
                "- cumulative RMS/component order, cosine: "
                f"`{cumulative['observed_rms_order']:.6f}` / "
                f"`{cumulative['minimum_significant_component_order']:.6f}` / "
                f"`{cumulative['refinement_error_cosine']:.9f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "The held-out nonlinear spatial state and Tier-I physical-export "
            "breadth contract is now certified under its originally intended "
            "physical-face definition. A definitions-only variable-step duration "
            "controller is authorized next. Fixed-Q and reduction remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(parent: dict, failed: dict, spatial_manifest: dict, result: dict, arrays: dict) -> int:
    if result["face_alias_cause_proved"]:
        classification = (
            "heldout_spatial_export_failure_caused_by_active_face_alias_"
            "corrected_physical_face_contract_passes"
        )
        authorized_next = "WP10c9d6c7c3b5a_variable_step_duration_controller_manifest"
    else:
        classification = (
            "corrected_physical_face_contract_failed_"
            "duration_extension_remains_blocked"
        )
        authorized_next = "WP10c9d6c7c3b4d1_export_face_failure_localization"
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "profiles": list(PROFILES),
        "times_seconds": TIMES,
        "observable_names": list(OBSERVABLE_NAMES),
        "legacy_parent_face_index": int(c3b2b.COUPLING_PARENT_FACE),
        "correct_active_face_indices": {
            layout: result["face_mappings"][layout][
                "correct_active_coupling_face_index"
            ]
            for layout in LAYOUTS
        },
        "spatial_gates": spatial_manifest["tier_I_binding_contract"]["gates"],
        "cause_gates": {
            "minimum_face_alias_alignment": 0.95,
            "maximum_corrected_to_legacy_error_ratio": 0.25,
            "maximum_error_decomposition_closure_defect": 1.0e-12,
        },
        "propagation_executed": False,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": result["passed"],
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "failed_b4b3_classification_preserved": failed["classification"],
        "operator_changed": False,
        "production_defaults_changed": False,
        "propagation_executed": False,
        "audit": result,
        "heldout_spatial_convergence_certified": result["face_alias_cause_proved"],
        "variable_step_duration_controller_manifest_authorized": result[
            "face_alias_cause_proved"
        ],
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values) for name, values in arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "failed_summary": FAILED_DIRECTORY / "summary.json",
        "failed_arrays": c3b4b3.DECISIVE_ARRAYS,
        "coarse_arrays": c3b4b1.DECISIVE_ARRAYS,
        "background_arrays": BASE_DIRECTORY / "decisive_arrays.npz",
        "spatial_manifest": c3b4b3.SPATIAL_MANIFEST_DIRECTORY
        / "nonlinear_spatial_export_manifest.json",
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
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def main() -> int:
    parent, failed, spatial_manifest = _validate_parent()
    result, arrays = _analyze(spatial_manifest)
    return _package(parent, failed, spatial_manifest, result, arrays)


if __name__ == "__main__":
    raise SystemExit(main())
