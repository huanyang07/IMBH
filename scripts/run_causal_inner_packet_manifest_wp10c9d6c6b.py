#!/usr/bin/env python3
"""Freeze a prospective uniform packet manifest without propagation."""

from __future__ import annotations

import argparse
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

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3
import run_causal_inner_local_truncation_wp10c9d6c5 as c5
import run_causal_inner_packet_resolution_wp10c9d6c6a as c6a
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
    causal_characteristic_purity,
    causal_scaled_variant_defect,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6b"
ANALYZED_BASE_COMMIT = "4fd671c10809fb015476549a7afb5fc56f0e3d0a"
ANALYZED_BASE_PARENT = "8d7f4ebcf5ab3fe97dfdc54abf2eb82c5ffb0858"
ANALYZED_BASE_TREE = "09cc420fb3b5780b77417dddd9744ed04fbbdbb2"
THIS_RUNNER = (
    "scripts/run_causal_inner_packet_manifest_wp10c9d6c6b.py"
)

REFERENCE_LABEL = c6a2.REFERENCE_LABEL
AMPLITUDE_FACTORS = (0.5, 1.0)
SIGNS = (-1, 1)
MINIMUM_GLOBAL_FAMILY_PURITY = 0.995
MINIMUM_ACTIVE_CELL_FAMILY_PURITY = 0.98
MINIMUM_MIXED_COEFFICIENT_COSINE = 0.99
MAXIMUM_REPLAY_DEFECT = 2.0e-12
MAXIMUM_SCALING_DEFECT = 1.0e-15

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_windowed_contract_wp10c9d6c6a2"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
C6A_SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_resolution_wp10c9d6c6a/summary.json"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_manifest_wp10c9d6c6b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
MANIFEST_PATH = CANONICAL_DIRECTORY / "packet_manifest.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_windowed_contract_wp10c9d6c6a2.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_manifest.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_windowed_contract.py",
    "tests/test_causal_inner_packet_manifest.py",
    "tests/test_causal_inner_packet_manifest_wp10c9d6c6b.py",
)


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return causal_array_sha256(values)


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_canonical_catalog() -> None:
    rows: list[dict[str, str | int]] = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "case": case.name,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CANONICAL_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
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
    summary = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
    summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c6b analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).exists()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def _mixed_coefficient_report(
    values: np.ndarray,
    bases: np.ndarray,
    field_scales: np.ndarray,
    cell_measures: np.ndarray,
) -> dict:
    normalized_values = values / field_scales[None, :]
    normalized_bases = bases / field_scales[None, :, None]
    coefficients = np.asarray(
        [
            np.linalg.solve(basis, value)
            for basis, value in zip(
                normalized_bases,
                normalized_values,
                strict=True,
            )
        ],
        dtype=float,
    )
    target = np.asarray(c6a2.MIXED_COEFFICIENTS, dtype=float)
    target /= np.linalg.norm(target)
    activity = np.linalg.norm(coefficients, axis=1)
    active = activity >= 1.0e-6 * float(np.max(activity))
    normalized = coefficients[active] / activity[active, None]
    cosines = np.abs(normalized @ target)
    energy = np.einsum(
        "ci,c->i",
        np.abs(coefficients) ** 2,
        cell_measures,
    )
    return {
        "minimum_active_cell_coefficient_cosine": float(
            np.min(cosines)
        ),
        "energy_fractions": energy / float(np.sum(energy)),
        "passed": bool(
            np.min(cosines) >= MINIMUM_MIXED_COEFFICIENT_COSINE
        ),
    }


def _build_manifest() -> tuple[dict, dict, dict[str, np.ndarray]]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != (
            "variable_coefficient_windowed_contract_certified_"
            "packet_manifest_authorized"
        )
        or parent["authorized_next"]
        != "WP10c9d6c6b_packet_definition_manifest_only"
        or not parent["passed"]
    ):
        raise RuntimeError("c6a2 manifest authorization changed")
    configurations, construction_arrays, construction = (
        c3._build_continuum_configurations()
    )
    interpolator, characteristic_report, characteristic_arrays = (
        c6a2._build_characteristic_interpolator(
            configurations,
            construction_arrays,
        )
    )
    directions, probe_report, replay_arrays = c6a2._build_probes(
        configurations,
        construction_arrays,
        interpolator,
    )
    if not construction["passed"] or not characteristic_report["passed"]:
        raise RuntimeError("manifest construction method failed")
    with np.load(PARENT_ARRAYS, allow_pickle=False) as source:
        parent_arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    reference = configurations[REFERENCE_LABEL]
    grid = reference["context"].grid
    centers = np.asarray(grid.centers, dtype=float)
    measures = np.asarray(grid.cell_measures, dtype=float)
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    center_bases = interpolator.evaluate(centers)
    maximum_replay = 0.0
    base_entries = []
    variants = []
    arrays: dict[str, np.ndarray] = {
        "N128_centers": centers,
        "N128_cell_measures": measures,
        "N128_characteristic_bases": center_bases,
        **characteristic_arrays,
    }
    purity_reports = {}
    maximum_scaling_defect = 0.0
    for base_name in c6a2.PROBE_NAMES:
        values = directions[REFERENCE_LABEL][base_name][
            "primary_physical"
        ]
        frozen = parent_arrays[f"{base_name}__N128_primary_physical"]
        replay = _relative_difference(values, frozen)
        maximum_replay = max(maximum_replay, replay)
        definition = c6a2.PROBE_DEFINITIONS[base_name]
        spectrum = parent["probe_construction_report"][
            "spectrum_reports"
        ][base_name]
        family = definition["family"]
        if family == "mixed":
            purity = _mixed_coefficient_report(
                values,
                center_bases,
                field_scales,
                measures,
            )
            purity_reports[base_name] = purity
            purity_passed = purity["passed"]
        else:
            family_index = interpolator.family_labels.index(family)
            result = causal_characteristic_purity(
                values,
                center_bases,
                field_scales,
                measures,
                selected_family=family_index,
            )
            global_fraction = float(
                result.family_energy_fractions[family_index]
            )
            purity = {
                "selected_family": family,
                "selected_global_energy_fraction": global_fraction,
                "minimum_active_cell_selected_fraction": (
                    result.minimum_active_cell_selected_fraction
                ),
                "family_energy_fractions": (
                    result.family_energy_fractions
                ),
                "maximum_reconstruction_defect": (
                    result.maximum_reconstruction_defect
                ),
                "passed": bool(
                    global_fraction >= MINIMUM_GLOBAL_FAMILY_PURITY
                    and result.minimum_active_cell_selected_fraction
                    >= MINIMUM_ACTIVE_CELL_FAMILY_PURITY
                ),
            }
            purity_reports[base_name] = purity
            purity_passed = purity["passed"]
        base_id = f"base::{base_name}"
        base_entries.append(
            {
                "base_id": base_id,
                "source_probe": base_name,
                "role": definition["role"],
                "family": family,
                "window": {
                    "kind": "full_domain_sine_power",
                    "power": int(definition["window_power"]),
                    "support_inner_over_rg": float(
                        grid.edges[0] / grid.gravitational_radius
                    ),
                    "support_outer_over_rg": float(
                        grid.edges[-1] / grid.gravitational_radius
                    ),
                },
                "base_amplitude": c6a2.PROBE_AMPLITUDE,
                "theta_99": spectrum["theta_99"],
                "nyquist_alias_fraction": (
                    spectrum["nyquist_alias_fraction"]
                ),
                "endpoint_cell_fraction_bound": (
                    parent["probe_construction_report"][
                        "maximum_endpoint_cell_fraction"
                    ]
                ),
                "projection_sha256": causal_array_sha256(values),
                "definition_sha256": causal_canonical_json_sha256(
                    {
                        "definition": definition,
                        "base_amplitude": c6a2.PROBE_AMPLITUDE,
                    }
                ),
                "purity_passed": purity_passed,
                "eligible_for_prospective_propagation": bool(
                    spectrum["spectral_passed"] and purity_passed
                ),
            }
        )
        arrays[f"{base_name}__N128_physical"] = values
        for factor in AMPLITUDE_FACTORS:
            for sign in SIGNS:
                multiplier = float(factor) * int(sign)
                variant_values = multiplier * values
                scaling_defect = causal_scaled_variant_defect(
                    values,
                    variant_values,
                    expected_factor=multiplier,
                )
                maximum_scaling_defect = max(
                    maximum_scaling_defect,
                    scaling_defect,
                )
                variants.append(
                    {
                        "packet_id": (
                            f"{base_name}::"
                            f"a{factor:.2f}::"
                            f"{'plus' if sign > 0 else 'minus'}"
                        ),
                        "base_id": base_id,
                        "role": definition["role"],
                        "family": family,
                        "sign": int(sign),
                        "amplitude_factor": float(factor),
                        "physical_amplitude": (
                            c6a2.PROBE_AMPLITUDE * float(factor)
                        ),
                        "projection_sha256": causal_array_sha256(
                            variant_values
                        ),
                        "propagate_in_prospective_uniform_suite": True,
                    }
                )

    c6a_summary = json.loads(
        C6A_SUMMARY.read_text(encoding="utf-8")
    )
    stress_controls = []
    for name, definition in c5.BOUNDARY_PROFILE_DEFINITIONS.items():
        spectral = c6a_summary["packet_resolution_report"][
            "profiles"
        ][name]
        stress_controls.append(
            {
                "packet_id": f"stress::{name}",
                "definition": definition,
                "source_work_package": "WP10c9d6c5",
                "theta_99": spectral["theta_quantile"],
                "nyquist_alias_fraction": (
                    spectral["nyquist_alias_fraction"]
                ),
                "spectrally_eligible": False,
                "binding": False,
                "propagate_in_prospective_uniform_suite": False,
                "preserved_as_historical_stress_control": True,
            }
        )

    propagation_contract = {
        "binding_grids": ("uniform_N128", "uniform_N256", "uniform_N512"),
        "time_horizon_s": c6a2.TIME_HORIZON_S,
        "instantaneous_and_cumulative_gates": {
            "minimum_rms_order": 0.75,
            "minimum_maximum_order": 0.75,
            "minimum_significant_component_order": 0.75,
            "maximum_fine_normalized_difference": 0.05,
            "minimum_history_cosine": 0.90,
            "minimum_refinement_error_cosine": 0.90,
        },
        "state_reference_gates": {
            "maximum_N128_Richardson_error": (
                c6a2.MAXIMUM_COMPLETE_SEMIGROUP_ERROR
            ),
            "maximum_reference_uncertainty_to_fine_difference": 0.10,
            "maximum_projection_uncertainty_to_fine_difference": 0.10,
            "maximum_restart_uncertainty_to_fine_difference": 0.10,
            "maximum_boundary_integral_uncertainty_to_fine_difference": (
                0.10
            ),
        },
        "exact_boundary_semigroup_integral_required": True,
        "physical_export_vector_required": True,
        "all_manifest_variants_binding": True,
        "stress_controls_nonbinding_and_not_propagated": True,
        "no_threshold_changes_after_propagation": True,
    }
    manifest_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "base_profiles": base_entries,
        "packet_variants": variants,
        "nonbinding_stress_controls": stress_controls,
        "prospective_propagation_contract": propagation_contract,
    }
    manifest_hash = causal_canonical_json_sha256(manifest_without_hash)
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": manifest_hash,
    }
    all_base_eligible = all(
        entry["eligible_for_prospective_propagation"]
        for entry in base_entries
    )
    report = {
        "continuum_construction_passed": construction["passed"],
        "characteristic_report": characteristic_report,
        "probe_replay_report": probe_report,
        "maximum_parent_projection_replay_defect": maximum_replay,
        "purity_reports": purity_reports,
        "maximum_scaling_defect": maximum_scaling_defect,
        "base_profile_count": len(base_entries),
        "packet_variant_count": len(variants),
        "stress_control_count": len(stress_controls),
        "all_base_profiles_eligible": all_base_eligible,
        "manifest_sha256": manifest_hash,
        "passed": bool(
            maximum_replay <= MAXIMUM_REPLAY_DEFECT
            and maximum_scaling_defect <= MAXIMUM_SCALING_DEFECT
            and all_base_eligible
            and characteristic_report["passed"]
            and probe_report["passed"]
        ),
    }
    arrays["family_purity_matrix"] = np.asarray(
        [
            purity_reports[name].get(
                "family_energy_fractions",
                purity_reports[name].get("energy_fractions"),
            )
            for name in c6a2.PROBE_NAMES
        ],
        dtype=float,
    )
    return manifest, report, arrays


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "propagation_executed": False,
        "amplitude_factors": AMPLITUDE_FACTORS,
        "signs": SIGNS,
        "gates": {
            "minimum_global_family_purity": (
                MINIMUM_GLOBAL_FAMILY_PURITY
            ),
            "minimum_active_cell_family_purity": (
                MINIMUM_ACTIVE_CELL_FAMILY_PURITY
            ),
            "minimum_mixed_coefficient_cosine": (
                MINIMUM_MIXED_COEFFICIENT_COSINE
            ),
            "maximum_replay_defect": MAXIMUM_REPLAY_DEFECT,
            "maximum_scaling_defect": MAXIMUM_SCALING_DEFECT,
        },
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    manifest, report, arrays = _build_manifest()
    if report["passed"]:
        classification = (
            "packet_definition_manifest_frozen_"
            "uniform_propagation_authorized"
        )
        authorized_next = (
            "WP10c9d6c6c_prospective_uniform_packet_propagation"
        )
    else:
        classification = "packet_definition_manifest_failed"
        authorized_next = "none"
    passed = bool(authorized_next != "none")
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    _write_json(MANIFEST_PATH, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "audit_executed": True,
        "operator_changed": False,
        "propagation_executed": False,
        "parent_classification": (
            "variable_coefficient_windowed_contract_certified_"
            "packet_manifest_authorized"
        ),
        "parent_classification_preserved": True,
        "manifest_report": report,
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_file_sha256": _sha256(MANIFEST_PATH),
        "prospective_uniform_packet_propagation_authorized": passed,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in sorted(arrays.items())
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "DIAGNOSTIC ONLY" if passed else "REJECTED",
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse",
            "HEAD^{tree}",
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python "
            "scripts/run_causal_inner_packet_manifest_wp10c9d6c6b.py"
        ),
        "environment": _environment(),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                PARENT_CONFIG,
                PARENT_SUMMARY,
                PARENT_ARRAYS,
                PARENT_PROVENANCE,
            )
        },
    }
    _write_json(CONFIG_PATH, _config())
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        json.dumps(
            _plain(
                {
                    "classification": classification,
                    "authorized_next": authorized_next,
                    "base_profile_count": report[
                        "base_profile_count"
                    ],
                    "packet_variant_count": report[
                        "packet_variant_count"
                    ],
                    "manifest_sha256": report["manifest_sha256"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def refresh_metadata_only() -> dict:
    if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
        raise RuntimeError("c6b canonical evidence is unavailable")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    source_hashes, source_manifest = _source_manifest()
    summary["implementation_source_hashes"] = source_hashes
    summary["implementation_source_manifest_sha256"] = source_manifest
    summary["manifest_file_sha256"] = _sha256(MANIFEST_PATH)
    summary["decisive_arrays_sha256"] = _sha256(DECISIVE_ARRAYS)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    provenance = json.loads(
        PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "working_tree_status": _git_value("status", "--short"),
            "environment": _environment(),
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
        }
    )
    _write_json(CONFIG_PATH, _config())
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-metadata-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    else:
        run()


if __name__ == "__main__":
    main()
