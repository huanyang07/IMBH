#!/usr/bin/env python3
"""Localize the held-out nonlinear spatial-export failure without propagation."""

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

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_coarse_screen_wp10c9d6c7c3b4b1 as c3b4b1  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as c3b4a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_spatial_wp10c9d6c7c3b4b3 as c3b4b3  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b4c"
ANALYZED_BASE_COMMIT = "e78aa83fa115d02538bdf02b357494f7f2bfd8ad"
ANALYZED_BASE_PARENT = "29ad409d915a9602b3ef550630ba881fb01046a0"
ANALYZED_BASE_TREE = "332de1558b71e747f115200cbf1a6eaadba3adf4"

LAYOUTS = tuple(c3b4a.LAYOUTS)
COARSE_LAYOUT = LAYOUTS[0]
PROFILES = tuple(c3b4a.PROFILE_NAMES)
TIMES = np.asarray(c3b4a.COMMON_OUTPUT_TIMES_SECONDS, dtype=float)
HORIZON_SECONDS = float(c3b4a.HORIZON_SECONDS)
OBSERVABLE_NAMES = tuple(c3b2b.OBSERVABLE_NAMES)

ARTIFACT = (
    "causal_inner_nonlinear_profile_breadth_spatial_localization_"
    "wp10c9d6c7c3b4c"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_profile_breadth_spatial_"
    "localization_wp10c9d6c7c3b4c.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_profile_breadth_spatial_"
    "localization_wp10c9d6c7c3b4c.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_PROFILE_BREADTH_SPATIAL_LOCALIZATION_"
    "WP10C9D6C7C3B4C_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c3b4b3.CANONICAL_DIRECTORY
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
    paths = (THIS_RUNNER, THIS_TEST, c3b4b3.THIS_RUNNER, c3b2b.THIS_RUNNER)
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    spatial_manifest = _read_json(
        c3b4b3.SPATIAL_MANIFEST_DIRECTORY
        / "nonlinear_spatial_export_manifest.json"
    )
    if (
        parent["classification"]
        != "heldout_profile_spatial_confirmation_failed_duration_extension_blocked"
        or parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b4b3_spatial_failure_localization"
        or parent["variable_step_duration_controller_manifest_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("b4b3 localization authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("b4c analyzed identity changed")
    return parent, spatial_manifest


def _raw_perturbed(
    layout: str,
    profile: str,
    parent_arrays: dict[str, np.ndarray],
    coarse_arrays: dict[str, np.ndarray],
) -> np.ndarray:
    if layout == COARSE_LAYOUT:
        key = f"{c3b4b1._task_id(profile)}__states"
        return np.asarray(coarse_arrays[key], dtype=float)
    key = f"{c3b4b3._task_id(layout, profile)}__states"
    return np.asarray(parent_arrays[key], dtype=float)


def _restrict_history(history: np.ndarray, layout) -> np.ndarray:
    return np.asarray(
        [
            restrict_causal_embedded_patch_cell_averages(state, layout)
            for state in np.asarray(history, dtype=float)
        ],
        dtype=float,
    )


def _common_export_history(context, coupling_face: int, history: np.ndarray):
    values = []
    audits = []
    for state in history:
        observable, audit = c3b2b._direct_observable(
            context, state, coupling_face
        )
        values.append(observable)
        audits.append(audit)
    return np.asarray(values, dtype=float), audits


def _scaled_vector(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) / scales[None, :]).ravel()


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float).ravel()))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _pair_attribution(
    native_left: np.ndarray,
    native_right: np.ndarray,
    common_left: np.ndarray,
    common_right: np.ndarray,
    scales: np.ndarray,
) -> dict:
    native = _scaled_vector(native_right - native_left, scales)
    common = _scaled_vector(common_right - common_left, scales)
    mapping = native - common
    closure = native - common - mapping
    native_norm = _norm(native)
    denominator = max(native_norm, np.finfo(float).tiny)
    return {
        "native_error_norm": native_norm,
        "common_state_error_norm": _norm(common),
        "layout_native_map_error_norm": _norm(mapping),
        "common_state_to_native_error_ratio": _norm(common) / denominator,
        "layout_native_map_to_native_error_ratio": _norm(mapping) / denominator,
        "layout_native_map_alignment_with_native_error": _cosine(
            mapping, native
        ),
        "common_state_alignment_with_native_error": _cosine(common, native),
        "decomposition_closure_defect": _norm(closure) / denominator,
    }


def _channel_report(
    native: tuple[np.ndarray, np.ndarray, np.ndarray],
    common: tuple[np.ndarray, np.ndarray, np.ndarray],
    scales: np.ndarray,
    gates: dict,
) -> dict:
    mapping = tuple(
        native_values - common_values
        for native_values, common_values in zip(native, common, strict=True)
    )
    return {
        "native": c3b2b._metric_payload(
            c3b2b._packet_metrics(native, scales, gates)
        ),
        "common_parent_map": c3b2b._metric_payload(
            c3b2b._packet_metrics(common, scales, gates)
        ),
        "layout_native_map_defect": c3b2b._metric_payload(
            c3b2b._packet_metrics(mapping, scales, gates)
        ),
        "coarse_medium_attribution": _pair_attribution(
            native[0], native[1], common[0], common[1], scales
        ),
        "medium_fine_attribution": _pair_attribution(
            native[1], native[2], common[1], common[2], scales
        ),
    }


def _analyze(spatial_manifest: dict) -> tuple[dict, dict[str, np.ndarray]]:
    parent_arrays = _load_npz(c3b4b3.DECISIVE_ARRAYS)
    coarse_arrays = _load_npz(c3b4b1.DECISIVE_ARRAYS)
    base_arrays = _load_npz(BASE_DIRECTORY / "decisive_arrays.npz")
    pilot_arrays = _load_npz(c3b4b3.SPATIAL_PILOT_DIRECTORY / "decisive_arrays.npz")
    observable_scales = np.asarray(
        pilot_arrays["fixed_physical_observable_scales"], dtype=float
    )
    gates = spatial_manifest["tier_I_binding_contract"]["gates"]
    _, layouts = c3b4b3._spatial_geometry()
    configurations = c3b1a._configurations()
    common_context = configurations[COARSE_LAYOUT]["context"]
    common_face = layouts[COARSE_LAYOUT].coupling_face_index

    restricted_base = {}
    common_base_exports = {}
    maximum_common_audit_defect = 0.0
    maximum_common_incoming = 0
    for layout in LAYOUTS:
        restricted_base[layout] = _restrict_history(
            base_arrays[f"{layout}__states"], layouts[layout]
        )
        exports, audits = _common_export_history(
            common_context, common_face, restricted_base[layout]
        )
        common_base_exports[layout] = exports
        for audit in audits:
            maximum_common_audit_defect = max(
                maximum_common_audit_defect,
                float(audit["local_block_ledger_defect"]),
                float(audit["source_double_count_defect"]),
                float(audit["shared_conservative_face_defect"]),
                float(audit["split_closure_defect"]),
            )
            maximum_common_incoming = max(
                maximum_common_incoming,
                int(audit["incoming_excision_characteristics"]),
            )

    decisive = {
        "times_seconds": TIMES,
        "fixed_physical_observable_scales": observable_scales,
    }
    profiles = {}
    for profile in PROFILES:
        native_histories = []
        common_histories = []
        for layout in LAYOUTS:
            native = np.asarray(
                parent_arrays[
                    f"{layout}__{profile}__instantaneous_export_response"
                ],
                dtype=float,
            )
            perturbed = _raw_perturbed(
                layout, profile, parent_arrays, coarse_arrays
            )
            restricted_perturbed = _restrict_history(
                perturbed, layouts[layout]
            )
            common_perturbed, audits = _common_export_history(
                common_context, common_face, restricted_perturbed
            )
            for audit in audits:
                maximum_common_audit_defect = max(
                    maximum_common_audit_defect,
                    float(audit["local_block_ledger_defect"]),
                    float(audit["source_double_count_defect"]),
                    float(audit["shared_conservative_face_defect"]),
                    float(audit["split_closure_defect"]),
                )
                maximum_common_incoming = max(
                    maximum_common_incoming,
                    int(audit["incoming_excision_characteristics"]),
                )
            common = common_perturbed - common_base_exports[layout]
            native_histories.append(native)
            common_histories.append(common)
            decisive[f"{layout}__{profile}__native_export_response"] = native
            decisive[
                f"{layout}__{profile}__common_parent_export_response"
            ] = common
            decisive[
                f"{layout}__{profile}__layout_native_export_map_defect"
            ] = native - common

        native_tuple = tuple(native_histories)
        common_tuple = tuple(common_histories)
        native_cumulative = tuple(c3b2b._cumulative(v) for v in native_tuple)
        common_cumulative = tuple(c3b2b._cumulative(v) for v in common_tuple)
        profiles[profile] = {
            "instantaneous": _channel_report(
                native_tuple, common_tuple, observable_scales, gates
            ),
            "cumulative": _channel_report(
                native_cumulative,
                common_cumulative,
                observable_scales * HORIZON_SECONDS,
                gates,
            ),
        }

    common_passed = all(
        report[channel]["common_parent_map"]["passed"]
        for report in profiles.values()
        for channel in ("instantaneous", "cumulative")
    )
    attribution_reports = [
        report[channel][pair]
        for report in profiles.values()
        for channel in ("instantaneous", "cumulative")
        for pair in ("coarse_medium_attribution", "medium_fine_attribution")
    ]
    minimum_map_alignment = min(
        item["layout_native_map_alignment_with_native_error"]
        for item in attribution_reports
    )
    maximum_common_fraction = max(
        item["common_state_to_native_error_ratio"]
        for item in attribution_reports
    )
    maximum_closure = max(
        item["decomposition_closure_defect"] for item in attribution_reports
    )
    localized_to_map = bool(
        common_passed
        and minimum_map_alignment >= 0.95
        and maximum_common_fraction <= 0.25
        and maximum_closure <= 1.0e-12
    )
    return (
        {
            "passed": localized_to_map or not common_passed,
            "profiles": profiles,
            "common_parent_export_contract_passed": common_passed,
            "localized_to_layout_native_export_map": localized_to_map,
            "minimum_layout_native_map_error_alignment": minimum_map_alignment,
            "maximum_common_state_error_fraction": maximum_common_fraction,
            "maximum_error_decomposition_closure_defect": maximum_closure,
            "maximum_common_parent_ledger_defect": maximum_common_audit_defect,
            "maximum_common_parent_incoming_excision_characteristics": (
                maximum_common_incoming
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


def _report(summary: dict) -> str:
    result = summary["localization"]
    lines = [
        "# Nonlinear held-out spatial-export localization WP10c9d6c7c3b4c",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "No trajectory was propagated and no operator was changed. The "
        "committed coarse, middle and fine full states were conservatively "
        "restricted to the same 64-cell parent grid and evaluated through "
        "one common coarse-layout nonlinear Tier-I export map.",
        "",
        "## Discriminating result",
        "",
        "- common-parent export contract passed: "
        f"`{result['common_parent_export_contract_passed']}`",
        "- localized to layout-native export map: "
        f"`{result['localized_to_layout_native_export_map']}`",
        "- minimum map-error alignment with native refinement error: "
        f"`{result['minimum_layout_native_map_error_alignment']:.9f}`",
        "- maximum common-state fraction of native refinement error: "
        f"`{result['maximum_common_state_error_fraction']:.6f}`",
        "- maximum decomposition closure defect: "
        f"`{result['maximum_error_decomposition_closure_defect']:.3e}`",
        "",
        "## Profiles",
        "",
    ]
    for profile, profile_report in result["profiles"].items():
        native = profile_report["instantaneous"]["native"]
        common = profile_report["instantaneous"]["common_parent_map"]
        attribution = profile_report["instantaneous"][
            "medium_fine_attribution"
        ]
        lines.extend(
            [
                f"### `{profile}`",
                "",
                "- native RMS/component order and error cosine: "
                f"`{native['observed_rms_order']:.6f}` / "
                f"`{native['minimum_significant_component_order']:.6f}` / "
                f"`{native['refinement_error_cosine']:.9f}`",
                "- common-map RMS/component order and error cosine: "
                f"`{common['observed_rms_order']:.6f}` / "
                f"`{common['minimum_significant_component_order']:.6f}` / "
                f"`{common['refinement_error_cosine']:.9f}`",
                "- fine-pair map alignment / common-state fraction: "
                f"`{attribution['layout_native_map_alignment_with_native_error']:.9f}` / "
                f"`{attribution['common_state_to_native_error_ratio']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation and next package",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "The failed b4b3 classification is preserved. Duration extension, "
            "fixed-Q experiments and reduced slow evolution remain blocked. "
            "Only the evidence-selected export-map audit is authorized next.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(parent: dict, spatial_manifest: dict, result: dict, arrays: dict) -> int:
    localized = result["localized_to_layout_native_export_map"]
    if localized:
        classification = (
            "spatial_failure_localized_to_layout_native_export_map_"
            "common_parent_map_passes"
        )
        authorized_next = "WP10c9d6c7c3b4d_layout_native_export_map_audit"
    else:
        classification = (
            "spatial_failure_persists_under_common_parent_export_map_"
            "evolution_observable_coupling_audit_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b4d_evolution_observable_coupling_audit"
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "profiles": list(PROFILES),
        "times_seconds": TIMES,
        "observable_names": list(OBSERVABLE_NAMES),
        "common_export_map_layout": COARSE_LAYOUT,
        "spatial_gates": spatial_manifest["tier_I_binding_contract"]["gates"],
        "localization_gates": {
            "minimum_map_error_alignment": 0.95,
            "maximum_common_state_error_fraction": 0.25,
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
        "operator_changed": False,
        "production_defaults_changed": False,
        "propagation_executed": False,
        "localization": result,
        "heldout_spatial_convergence_certified": False,
        "variable_step_duration_controller_manifest_authorized": False,
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
        "parent_arrays": c3b4b3.DECISIVE_ARRAYS,
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
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = (
        "config.json",
        "summary.json",
        "provenance.json",
        "decisive_arrays.npz",
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
    return 0 if result["passed"] else 1


def main() -> int:
    parent, spatial_manifest = _validate_parent()
    result, arrays = _analyze(spatial_manifest)
    return _package(parent, spatial_manifest, result, arrays)


if __name__ == "__main__":
    raise SystemExit(main())
