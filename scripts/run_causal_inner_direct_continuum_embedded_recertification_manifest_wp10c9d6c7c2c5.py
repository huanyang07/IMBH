#!/usr/bin/env python3
"""Freeze the prospective two-route embedded recertification contract."""

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

import run_causal_inner_direct_continuum_embedded_manifest_wp10c9d6c7c2c1 as c2c1  # noqa: E402
import run_causal_inner_embedded_cumulative_flux_diagnostic_wp10c9d6c7c2c4 as c2c4  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2c5"
ANALYZED_BASE_COMMIT = "f2e90efbf3c5a2293d7ed7455609fc1bdbba4c31"
ANALYZED_BASE_PARENT = "b2a09dd319099b8194b99e2a5df1393e9aecbcec"
ANALYZED_BASE_TREE = "29a80aa2cb603b3423d65cc5784c604edbddc7d0"

ANGLES_DEGREES = (
    11.25,
    33.75,
    56.25,
    78.75,
    101.25,
    123.75,
    146.25,
    168.75,
)
PROFILES = tuple(
    f"unseen_angle_{str(angle).replace('.', 'p')}_acoustic_shear"
    for angle in ANGLES_DEGREES
)
MULTIPLIERS = (1.0, -1.0, 0.5, -0.5)
LABELS = c2c1.LAYOUT_LABELS
REFINEMENT_RATIOS = (1, 2, 4)
BOUNDARY_FLUX_COMPONENTS = (
    "inner_mass_flux",
    "inner_angular_momentum_flux",
    "inner_killing_energy_flux",
    "coupling_mass_flux",
    "coupling_angular_momentum_flux",
    "coupling_killing_energy_flux",
)
MAXIMUM_RESTRICTION_DEFECT = 1.0e-12
MAXIMUM_INITIAL_INNER_ACTIVITY = 1.0e-15
MAXIMUM_COEFFICIENT_NORM_DEFECT = 1.0e-15

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_direct_continuum_embedded_"
    "recertification_manifest_wp10c9d6c7c2c5.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_direct_continuum_embedded_"
    "recertification_manifest_wp10c9d6c7c2c5.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_DIRECT_CONTINUUM_EMBEDDED_RECERTIFICATION_"
    "MANIFEST_WP10C9D6C7C2C5_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
PARENT_DIRECTORY = c2c4.CANONICAL_DIRECTORY
SOURCE_DIRECTORY = c2c1.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_embedded_recertification_manifest_"
    "wp10c9d6c7c2c5"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
MANIFEST_PATH = CANONICAL_DIRECTORY / "recertification_manifest.json"
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


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _validate_parent() -> tuple[dict, dict, dict]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    source_summary = _read_json(SOURCE_DIRECTORY / "summary.json")
    source_manifest = _read_json(
        SOURCE_DIRECTORY / "embedded_manifest.json"
    )
    if (
        not summary["passed"]
        or summary["classification"]
        != "cumulative_boundary_flux_absolute_envelope_supported_"
        "strict_order_unresolved_manifest_authorized"
        or summary["authorized_next"]
        != "WP10c9d6c7c2c5_direct_continuum_embedded_"
        "recertification_manifest"
        or source_summary["manifest_sha256"]
        != source_manifest["manifest_sha256"]
    ):
        raise RuntimeError("c2c4/c2c1 manifest authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2c5 analyzed identity changed")
    return summary, source_summary, source_manifest


def _coefficients() -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            (
                np.cos(np.deg2rad(angle)),
                np.sin(np.deg2rad(angle)),
            ),
            dtype=float,
        )
        for name, angle in zip(PROFILES, ANGLES_DEGREES, strict=True)
    }


def _profile_audit(
    arrays: dict[str, np.ndarray],
    coefficients: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        _parent_base,
        _field_scales,
    ) = c2a2._load_inputs()
    parent_grid = make_kerr_schild_column_grid_from_edges(
        np.asarray(arrays["parent_patch_edges"], dtype=float),
        parent_context.grid.gravitational_radius,
    )
    layouts = {
        ratio: make_causal_embedded_patch_layout(
            parent_grid, c2c1.PARENT_COUPLING_FACE, ratio
        )
        for ratio in REFINEMENT_RATIOS
    }
    packets = {}
    restricted = {}
    initial_inner = {}
    for ratio in REFINEMENT_RATIOS:
        label = LABELS[ratio]
        acoustic = np.asarray(
            arrays[f"acoustic__{label}__packet"], dtype=float
        )
        shear = np.asarray(
            arrays[f"shear__{label}__packet"], dtype=float
        )
        packets[label] = np.stack(
            [
                pair[0] * acoustic + pair[1] * shear
                for pair in coefficients.values()
            ],
            axis=0,
        )
        restricted[label] = (
            restrict_causal_embedded_patch_cell_averages(
                packets[label], layouts[ratio]
            )
        )
        initial_inner[label] = float(
            np.max(
                np.abs(
                    packets[label][
                        :, : layouts[ratio].n_refined_cells
                    ]
                )
            )
        )
    reference = restricted[LABELS[1]]
    restriction_defects = {
        label: _relative_defect(values, reference)
        for label, values in restricted.items()
    }
    coefficient_norm_defect = max(
        abs(float(np.linalg.norm(pair)) - 1.0)
        for pair in coefficients.values()
    )
    report = {
        "coefficient_norm_defect": coefficient_norm_defect,
        "restriction_defects": restriction_defects,
        "maximum_restriction_defect": max(
            restriction_defects.values()
        ),
        "initial_inner_activity_by_layout": initial_inner,
        "maximum_initial_inner_activity": max(initial_inner.values()),
        "passed": bool(
            coefficient_norm_defect <= MAXIMUM_COEFFICIENT_NORM_DEFECT
            and max(restriction_defects.values())
            <= MAXIMUM_RESTRICTION_DEFECT
            and max(initial_inner.values())
            <= MAXIMUM_INITIAL_INNER_ACTIVITY
        ),
    }
    decisive = {
        "angles_degrees": np.asarray(ANGLES_DEGREES, dtype=float),
        "acoustic_shear_coefficients": np.stack(
            tuple(coefficients.values())
        ),
        "common_parent_packets": reference,
        **{
            f"{label}__packets": packets[label] for label in packets
        },
        **{
            f"{label}__restricted_parent_packets": restricted[label]
            for label in restricted
        },
    }
    return report, decisive


def _manifest(
    source_manifest: dict,
    coefficients: dict[str, np.ndarray],
) -> dict:
    tier_i = dict(source_manifest["tier_I_contract"])
    tolerance = float(tier_i["maximum_fine_normalized_difference"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "historical_c2c3_rejection_preserved": True,
        "layouts": [
            {
                "refinement_ratio": ratio,
                "label": LABELS[ratio],
                "outer_parent_resolution": 98,
                "coupling_parent_face": c2c1.PARENT_COUPLING_FACE,
            }
            for ratio in REFINEMENT_RATIOS
        ],
        "binding_profiles": [
            {
                "name": name,
                "role": "prospective_unseen_heldout",
                "angle_degrees": angle,
                "acoustic_shear_coefficients": (
                    coefficients[name].tolist()
                ),
            }
            for name, angle in zip(
                PROFILES, ANGLES_DEGREES, strict=True
            )
        ],
        "variant_multipliers": list(MULTIPLIERS),
        "variant_count": len(PROFILES) * len(MULTIPLIERS),
        "tier_I_global_contract": tier_i,
        "component_order_routes": {
            "primary_route": {
                "name": "unchanged_pairwise_component_order",
                "minimum_order": tier_i[
                    "minimum_significant_component_order"
                ],
            },
            "alternate_route": {
                "name": "fixed_exterior_direct_continuum_absolute_envelope",
                "eligible_components": list(BOUNDARY_FLUX_COMPONENTS),
                "requires_all_three_embedded_levels_inside_envelope": True,
                "maximum_fixed_scale_RMS_error": tolerance,
                "maximum_fixed_scale_maximum_error": tolerance,
                "maximum_response_relative_RMS_error": tolerance,
                "maximum_response_relative_maximum_error": tolerance,
                "minimum_continuum_history_cosine": tier_i[
                    "minimum_history_cosine"
                ],
                "reference_nodes": [513, 769],
                "maximum_reference_uncertainty_fixed_scale": (
                    0.10 * tolerance
                ),
                "maximum_reference_uncertainty_response_relative": (
                    0.10 * tolerance
                ),
                "maximum_exact_integral_solve_residual": 1.0e-10,
                "strict_direct_error_order_required": False,
            },
        },
        "alternate_route_limits": {
            "may_replace_only_failed_significant_component_order": True,
            "global_RMS_order_still_binding": True,
            "global_maximum_order_still_binding": True,
            "global_fine_difference_still_binding": True,
            "global_history_cosine_still_binding": True,
            "global_refinement_error_cosine_still_binding": True,
            "non_boundary_component_failure_is_binding_failure": True,
        },
        "decision": {
            "all_profiles_pass": (
                "certify_declared_linear_embedded_class_and_authorize_"
                "definitions_only_bounded_nonlinear_manifest"
            ),
            "absolute_envelope_failure": (
                "freeze_exact_failure_and_localize_against_continuum"
            ),
            "non_boundary_component_failure": (
                "reject_recertification_without_metric_tuning"
            ),
        },
        "hard_stops": [
            "do_not_amend_c2c3",
            "do_not_change_operator_or_interface",
            "do_not_change_thresholds_after_propagation",
            "do_not_run_N1024",
            "do_not_start_nonlinear_before_complete_recertification",
            "do_not_start_fixed_Q_or_reduced_evolution",
        ],
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(
        manifest
    )
    return manifest


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        SOURCE_DIRECTORY / "embedded_manifest.json",
        SOURCE_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path) for path in paths
    }


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def _refresh_catalog() -> None:
    rows = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        provenance_path = case / "provenance.json"
        if not case.is_dir() or not provenance_path.is_file():
            continue
        provenance = _read_json(provenance_path)
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "case": case.name,
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "scientific_status": status,
                    }
                )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _write_report(summary: dict) -> None:
    audit = summary["profile_audit"]
    lines = [
        "# Direct-continuum embedded recertification manifest "
        "WP10c9d6c7c2c5",
        "",
        "## Result",
        "",
        "The definitions-only two-route contract is frozen. No state was "
        "propagated and no operator, interface, profile, or historical "
        "classification changed.",
        "",
        f"Eight unseen coefficient directions define 32 sign/amplitude "
        f"variants. Their maximum common-parent restriction defect is "
        f"`{audit['maximum_restriction_defect']:.3e}` and initial inner "
        f"activity is `{audit['maximum_initial_inner_activity']:.3e}`.",
        "",
        "The unchanged pairwise component-order gate remains primary. Only "
        "an M/J/E boundary-flux component may use the prospectively frozen "
        "absolute direct-continuum envelope, and only while every aggregate "
        "Tier-I gate still passes.",
        "",
        "## Decision",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        f"Authorized next: `{summary['authorized_next']}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    parent_summary, _source_summary, source_manifest = _validate_parent()
    arrays = _load_npz(SOURCE_DIRECTORY / "decisive_arrays.npz")
    coefficients = _coefficients()
    profile_audit, decisive = _profile_audit(arrays, coefficients)
    manifest = _manifest(source_manifest, coefficients)
    passed = profile_audit["passed"]
    classification = (
        "direct_continuum_embedded_two_route_contract_frozen_"
        "recertification_authorized"
        if passed
        else "direct_continuum_embedded_recertification_manifest_failed"
    )
    authorized_next = (
        "WP10c9d6c7c2c6_direct_continuum_embedded_recertification"
        if passed
        else "diagnose_embedded_recertification_manifest"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "angles_degrees": list(ANGLES_DEGREES),
        "variant_multipliers": list(MULTIPLIERS),
        "gates": {
            "maximum_restriction_defect": MAXIMUM_RESTRICTION_DEFECT,
            "maximum_initial_inner_activity": (
                MAXIMUM_INITIAL_INNER_ACTIVITY
            ),
            "maximum_coefficient_norm_defect": (
                MAXIMUM_COEFFICIENT_NORM_DEFECT
            ),
        },
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    decisive.update(
        {
            "profile_names_utf8": np.frombuffer(
                "\n".join(PROFILES).encode("utf-8"), dtype=np.uint8
            ),
            "variant_multipliers": np.asarray(MULTIPLIERS),
        }
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).is_file()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "historical_c2c3_classification_preserved": (
            parent_summary["historical_c2c3_classification_preserved"]
        ),
        "profile_audit": profile_audit,
        "profile_count": len(PROFILES),
        "variant_count": len(PROFILES) * len(MULTIPLIERS),
        "manifest_sha256": manifest["manifest_sha256"],
        "binding_decision": {
            "two_route_contract_frozen": passed,
            "embedded_recertification_authorized": passed,
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_propagation_authorized": False,
            "fixed_Q_or_reduced_evolution_authorized": False,
        },
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": passed,
        "config_sha256": _sha256(CONFIG_PATH),
        "recertification_manifest_file_sha256": _sha256(MANIFEST_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_hashes)
        ),
        "input_hashes": _input_hashes(),
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
            "classification": classification,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree": ANALYZED_BASE_TREE,
            "implementation_worktree_head": _git_value(
                "rev-parse", "HEAD"
            ),
            "implementation_source_hashes": source_hashes,
            "input_hashes": _input_hashes(),
            "command": (
                "PYTHONPATH=src python "
                "scripts/"
                "run_causal_inner_direct_continuum_embedded_"
                "recertification_manifest_wp10c9d6c7c2c5.py"
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
    )
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    return summary


def main() -> None:
    summary = run()
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "classification": summary["classification"],
                "passed": summary["passed"],
                "profile_count": summary["profile_count"],
                "variant_count": summary["variant_count"],
                "binding_decision": summary["binding_decision"],
                "authorized_next": summary["authorized_next"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
