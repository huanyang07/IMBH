#!/usr/bin/env python3
"""Evaluate the frozen c6e0 integral-conditioning manifest."""

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
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3
import run_causal_inner_height_localization_wp10c9d6c6d as c6d
import run_causal_inner_integral_conditioning_manifest_wp10c9d6c6e0 as c6e0
import run_causal_inner_packet_validation_wp10c9d6c6c as c6c
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
    linearize_causal_five_field_continuum_reference,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_height_localization import (  # noqa: E402
    causal_partition_cell_integrals,
    causal_restrict_cell_integrals,
    causal_signed_band_gram_matrix,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_integral_conditioning import (  # noqa: E402
    causal_absolute_band_error_envelope,
    causal_cancellation_ratio,
    causal_integral_conditioning_decision,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_characteristic_purity,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_resolution import (  # noqa: E402
    causal_packet_spectrum,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_exact_semigroup_integral_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_trapezoid_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6e1"
ANALYZED_BASE_COMMIT = "8e7b567d5f64b28db8405726586e1bf78fe9da67"
ANALYZED_BASE_PARENT = "c3acf82390a6f4fca1efd891bc4823d3b5ee318b"
ANALYZED_BASE_TREE = "ccaa4959f7606675019011fb8eae58310db8bc79"
FROZEN_MANIFEST_SHA256 = (
    "7eee9c710df8ee48418e0e54007d2f5a02360c07f42af2a750df5d15b3cc9f92"
)
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1.py"
)

LABELS = ("uniform_N128", "uniform_N256", "uniform_N512")
REFERENCE_LABEL = LABELS[0]
TIME_HORIZON_S = 0.125
TIME_SAMPLE_COUNT = 65
PRIMARY_PROJECTION_ORDER = 24
SECONDARY_PROJECTION_ORDER = 12
PRIMARY_CONTINUUM_NODES = 769
SECONDARY_CONTINUUM_NODES = 513
ANGULAR_FIELD = 2
ANGULAR_OBSERVABLE_INDEX = 11
TARGET_OBSERVABLE_NAME = "vertical_work_angular_momentum"
MAXIMUM_PROPAGATION_SCALING_DEFECT = 1.0e-10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_manifest_wp10c9d6c6e0"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_MANIFEST = PARENT_DIRECTORY / "conditioning_manifest.json"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
C3_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3/decisive_arrays.npz"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/"
    "run_causal_inner_integral_conditioning_manifest_wp10c9d6c6e0.py",
    "scripts/run_causal_inner_packet_validation_wp10c9d6c6c.py",
    "scripts/run_causal_inner_height_localization_wp10c9d6c6d.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_integral_conditioning.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_height_localization.py",
    "tests/test_causal_inner_integral_conditioning.py",
    "tests/"
    "test_causal_inner_integral_conditioning_validation_wp10c9d6c6e1.py",
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
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


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
        raise RuntimeError("WP10c9d6c6e1 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


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


def _load_manifest() -> tuple[dict, dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != (
            "integral_conditioning_contract_and_profiles_frozen_"
            "eligibility_audit_authorized"
        )
        or parent["authorized_next"]
        != "WP10c9d6c6e1_profile_eligibility_and_propagation"
        or not parent["passed"]
    ):
        raise RuntimeError("c6e1 parent authorization changed")
    stored_hash = manifest.pop("manifest_sha256")
    computed_hash = causal_canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = stored_hash
    if (
        stored_hash != FROZEN_MANIFEST_SHA256
        or computed_hash != FROZEN_MANIFEST_SHA256
        or parent["manifest_sha256"] != FROZEN_MANIFEST_SHA256
    ):
        raise RuntimeError("c6e0 manifest hash changed")
    return parent, manifest


def _continuum_action(
    background,
    evaluator,
    edges: np.ndarray,
):
    reference = linearize_causal_five_field_continuum_reference(
        background,
        evaluator,
    )
    rows = reference.integrate_blocks(edges)[
        "candidate_lower_height_work"
    ]
    return reference, float(np.sum(rows[:, ANGULAR_FIELD]))


def _build_inputs(manifest: dict):
    configurations, construction_arrays, construction = (
        c3._build_continuum_configurations()
    )
    interpolator, characteristic_report, characteristic_arrays = (
        c6a2._build_characteristic_interpolator(
            configurations,
            construction_arrays,
        )
    )
    reference_grid = configurations[REFERENCE_LABEL]["context"].grid
    finest = configurations[LABELS[-1]]
    background_profile = c3.SmoothCellAverageProfile(
        knots=np.asarray(
            construction_arrays["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            construction_arrays["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(reference_grid.gravitational_radius),
    )
    print("WP10c9d6c6e1: build continuum balance references", flush=True)
    primary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=PRIMARY_CONTINUUM_NODES,
    )
    secondary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=SECONDARY_CONTINUUM_NODES,
    )
    lower_radius = float(reference_grid.edges[0])
    upper_radius = float(reference_grid.edges[-1])
    base_definitions = manifest["base_profile_definitions"]
    evaluators = {}
    continuum_references = {}
    balance_reports = {}
    for name, definition in base_definitions.items():
        envelope = definition["envelope"]
        family = definition["family"]
        if envelope["kind"] == "full_domain_sine_power":
            evaluator = c6a2._probe_evaluator(
                {
                    "window_power": int(envelope["power"]),
                    "family": family,
                    "mixed_coefficients": None,
                },
                interpolator,
                lower_radius=lower_radius,
                upper_radius=upper_radius,
            )
            primary_reference, primary_action = _continuum_action(
                primary_background,
                evaluator,
                reference_grid.edges,
            )
            secondary_reference, secondary_action = _continuum_action(
                secondary_background,
                evaluator,
                reference_grid.edges,
            )
            balance_reports[name] = {
                "kind": "ordinary",
                "primary_initial_global_action": primary_action,
                "secondary_initial_global_action": secondary_action,
                "passed": True,
            }
        else:
            first_power = int(envelope["first_power"])
            second_power = int(envelope["second_power"])
            first = c6a2._probe_evaluator(
                {
                    "window_power": first_power,
                    "family": family,
                    "mixed_coefficients": None,
                },
                interpolator,
                lower_radius=lower_radius,
                upper_radius=upper_radius,
            )
            second = c6a2._probe_evaluator(
                {
                    "window_power": second_power,
                    "family": family,
                    "mixed_coefficients": None,
                },
                interpolator,
                lower_radius=lower_radius,
                upper_radius=upper_radius,
            )
            first_primary, first_primary_action = _continuum_action(
                primary_background,
                first,
                reference_grid.edges,
            )
            second_primary, second_primary_action = _continuum_action(
                primary_background,
                second,
                reference_grid.edges,
            )
            first_secondary, first_secondary_action = _continuum_action(
                secondary_background,
                first,
                reference_grid.edges,
            )
            second_secondary, second_secondary_action = _continuum_action(
                secondary_background,
                second,
                reference_grid.edges,
            )
            coefficient = -first_primary_action / second_primary_action
            secondary_coefficient = (
                -first_secondary_action / second_secondary_action
            )

            def evaluator(radii, first=first, second=second, coefficient=coefficient):
                return first(radii) + coefficient * second(radii)

            primary_reference, primary_action = _continuum_action(
                primary_background,
                evaluator,
                reference_grid.edges,
            )
            secondary_reference, secondary_action = _continuum_action(
                secondary_background,
                evaluator,
                reference_grid.edges,
            )
            coefficient_defect = abs(
                coefficient - secondary_coefficient
            ) / max(
                abs(coefficient),
                abs(secondary_coefficient),
                np.finfo(float).tiny,
            )
            cancellation_scale = max(
                abs(first_secondary_action)
                + abs(coefficient * second_secondary_action),
                np.finfo(float).tiny,
            )
            secondary_cancellation = (
                abs(secondary_action) / cancellation_scale
            )
            eligibility = manifest["eligibility_contract"]
            balance_passed = bool(
                coefficient_defect
                <= eligibility[
                    "maximum_balance_coefficient_relative_769_513_difference"
                ]
                and secondary_cancellation
                <= eligibility[
                    "maximum_secondary_continuum_initial_cancellation_ratio"
                ]
            )
            balance_reports[name] = {
                "kind": "continuum_balanced",
                "primary_coefficient": coefficient,
                "secondary_coefficient": secondary_coefficient,
                "coefficient_relative_769_513_difference": (
                    coefficient_defect
                ),
                "primary_initial_global_action": primary_action,
                "secondary_initial_global_action": secondary_action,
                "secondary_initial_cancellation_ratio": (
                    secondary_cancellation
                ),
                "passed": balance_passed,
            }
            del first_primary, second_primary, first_secondary, second_secondary
        evaluators[name] = evaluator
        continuum_references[name] = {
            "primary": primary_reference,
            "secondary": secondary_reference,
        }

    profile_names = tuple(base_definitions)
    directions = {label: {} for label in LABELS}
    arrays = dict(characteristic_arrays)
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    maximum_projection_defect = 0.0
    maximum_endpoint_fraction = 0.0
    for label in LABELS:
        grid = configurations[label]["context"].grid
        columns = np.asarray(
            configurations[label]["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, 5)
        for name in profile_names:
            primary = c3._project_callable_to_cells(
                grid,
                evaluators[name],
                quadrature_order=PRIMARY_PROJECTION_ORDER,
            )
            secondary = c3._project_callable_to_cells(
                grid,
                evaluators[name],
                quadrature_order=SECONDARY_PROJECTION_ORDER,
            )
            defect = _relative_defect(primary, secondary)
            maximum_projection_defect = max(
                maximum_projection_defect,
                defect,
            )
            normalized = primary / field_scales[None, :]
            norms = np.linalg.norm(normalized, axis=1)
            endpoint_fraction = max(
                float(norms[0]),
                float(norms[-1]),
            ) / max(float(np.max(norms)), np.finfo(float).tiny)
            maximum_endpoint_fraction = max(
                maximum_endpoint_fraction,
                endpoint_fraction,
            )
            directions[label][name] = {
                "primary_physical": primary,
                "secondary_physical": secondary,
                "primary_scaled": (primary / columns).ravel(),
                "secondary_scaled": (secondary / columns).ravel(),
                "projection_defect": defect,
                "endpoint_fraction": endpoint_fraction,
            }
            arrays[f"{name}__{label}__primary_physical"] = primary
            arrays[f"{name}__{label}__secondary_physical"] = secondary

    n128_grid = configurations[REFERENCE_LABEL]["context"].grid
    centers = np.asarray(n128_grid.centers, dtype=float)
    measures = np.asarray(n128_grid.cell_measures, dtype=float)
    bases = interpolator.evaluate(centers)
    spacing = float(np.mean(np.diff(np.log(n128_grid.edges))))
    eligibility_contract = manifest["eligibility_contract"]
    profile_reports = {}
    for name in profile_names:
        values = directions[REFERENCE_LABEL][name]["primary_physical"]
        spectrum = causal_packet_spectrum(
            values / field_scales[None, :],
            spacing,
            quantile=eligibility_contract["spectral_energy_quantile"],
        )
        theta = spectrum.quantile_angular_wavenumber * spacing
        family = base_definitions[name]["family"]
        family_index = interpolator.family_labels.index(family)
        purity = causal_characteristic_purity(
            values,
            bases,
            field_scales,
            measures,
            selected_family=family_index,
        )
        selected_fraction = float(
            purity.family_energy_fractions[family_index]
        )
        spectral_passed = bool(
            eligibility_contract["minimum_theta_99"]
            <= theta
            <= eligibility_contract["maximum_theta_99"]
            and spectrum.nyquist_alias_fraction
            <= eligibility_contract["maximum_nyquist_alias_fraction"]
        )
        purity_passed = bool(
            selected_fraction
            >= eligibility_contract["minimum_global_family_purity"]
            and purity.minimum_active_cell_selected_fraction
            >= eligibility_contract[
                "minimum_active_cell_family_purity"
            ]
        )
        projection_passed = bool(
            max(
                directions[label][name]["projection_defect"]
                for label in LABELS
            )
            <= eligibility_contract["maximum_projection_replay_defect"]
        )
        endpoint_passed = bool(
            max(
                directions[label][name]["endpoint_fraction"]
                for label in LABELS
            )
            <= eligibility_contract["maximum_endpoint_cell_fraction"]
        )
        profile_reports[name] = {
            "role": base_definitions[name]["role"],
            "family": family,
            "theta_99": theta,
            "nyquist_alias_fraction": spectrum.nyquist_alias_fraction,
            "selected_global_family_fraction": selected_fraction,
            "minimum_active_cell_family_fraction": (
                purity.minimum_active_cell_selected_fraction
            ),
            "maximum_projection_defect": max(
                directions[label][name]["projection_defect"]
                for label in LABELS
            ),
            "maximum_endpoint_cell_fraction": max(
                directions[label][name]["endpoint_fraction"]
                for label in LABELS
            ),
            "spectral_passed": spectral_passed,
            "purity_passed": purity_passed,
            "projection_passed": projection_passed,
            "endpoint_passed": endpoint_passed,
            "balance_passed": balance_reports[name]["passed"],
            "passed": bool(
                spectral_passed
                and purity_passed
                and projection_passed
                and endpoint_passed
                and balance_reports[name]["passed"]
            ),
        }
        arrays[f"{name}__spectrum_theta"] = (
            spectrum.angular_wavenumbers * spacing
        )
        arrays[f"{name}__spectrum_energy"] = spectrum.spectral_energy
        arrays[f"{name}__spectrum_cumulative"] = (
            spectrum.cumulative_energy_fraction
        )
    report = {
        "continuum_construction_passed": construction["passed"],
        "characteristic_field_passed": characteristic_report["passed"],
        "profile_reports": profile_reports,
        "balance_reports": balance_reports,
        "maximum_projection_defect": maximum_projection_defect,
        "maximum_endpoint_cell_fraction": maximum_endpoint_fraction,
        "all_profiles_eligible": all(
            item["passed"] for item in profile_reports.values()
        ),
        "passed": bool(
            construction["passed"]
            and characteristic_report["passed"]
            and all(item["passed"] for item in profile_reports.values())
        ),
    }
    return (
        configurations,
        construction_arrays,
        directions,
        evaluators,
        continuum_references,
        report,
        arrays,
    )


def _variant_directions(manifest: dict, base_directions: dict):
    base_names = tuple(manifest["base_profile_definitions"])
    base_lookup = {
        name: index for index, name in enumerate(base_names)
    }
    variants = manifest["profile_variants"]
    packet_ids = tuple(item["profile_id"] for item in variants)
    base_indices = np.asarray(
        [base_lookup[item["base_profile"]] for item in variants],
        dtype=int,
    )
    multipliers = np.asarray(
        [
            float(item["amplitude_factor"]) * int(item["sign"])
            for item in variants
        ],
        dtype=float,
    )
    result = {}
    for label in LABELS:
        base_primary = np.column_stack(
            [
                base_directions[label][name]["primary_scaled"]
                for name in base_names
            ]
        )
        base_secondary = np.column_stack(
            [
                base_directions[label][name]["secondary_scaled"]
                for name in base_names
            ]
        )
        result[label] = {
            "primary_scaled": (
                base_primary[:, base_indices] * multipliers[None, :]
            ),
            "secondary_base_scaled": base_secondary,
        }
    metadata = {
        "packet_ids": packet_ids,
        "base_names": base_names,
        "base_indices": base_indices,
        "multipliers": multipliers,
    }
    return result, metadata


def _propagate(
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    directions: dict,
    metadata: dict,
):
    times = np.linspace(0.0, TIME_HORIZON_S, TIME_SAMPLE_COUNT)
    propagated = {}
    report = {}
    base_variant_indices = []
    for base_index in range(len(metadata["base_names"])):
        matches = np.flatnonzero(
            (metadata["base_indices"] == base_index)
            & (metadata["multipliers"] == 1.0)
        )
        if matches.size != 1:
            raise RuntimeError("missing positive unit base variant")
        base_variant_indices.append(int(matches[0]))
    base_variant_indices = np.asarray(base_variant_indices, dtype=int)
    for label in LABELS:
        print(f"WP10c9d6c6e1: propagate frozen profiles on {label}", flush=True)
        configuration = configurations[label]
        generator = np.asarray(
            tangents[label].scaled_generator_per_s,
            dtype=float,
        )
        primary = np.asarray(
            directions[label]["primary_scaled"],
            dtype=float,
        )
        secondary = np.asarray(
            directions[label]["secondary_base_scaled"],
            dtype=float,
        )
        initial = np.column_stack((primary, secondary))
        trace = float(np.trace(generator))
        scaled = np.asarray(
            expm_multiply(
                generator,
                initial,
                start=0.0,
                stop=TIME_HORIZON_S,
                num=TIME_SAMPLE_COUNT,
                endpoint=True,
                traceA=trace,
            ),
            dtype=float,
        )
        primary_scaled = scaled[:, :, : primary.shape[1]]
        secondary_scaled = scaled[:, :, primary.shape[1] :]
        half = np.asarray(
            expm_multiply(
                0.5 * TIME_HORIZON_S * generator,
                primary,
                traceA=0.5 * TIME_HORIZON_S * trace,
            ),
            dtype=float,
        )
        restarted = np.asarray(
            expm_multiply(
                0.5 * TIME_HORIZON_S * generator,
                half,
                traceA=0.5 * TIME_HORIZON_S * trace,
            ),
            dtype=float,
        )
        exact = causal_exact_semigroup_integral_history(
            generator,
            primary_scaled,
            primary,
        )
        observable = np.asarray(observable_maps[label], dtype=float)
        signals = np.einsum("tnp,on->pto", primary_scaled, observable)
        cumulative = np.einsum(
            "tnp,on->pto",
            exact.integrated_states,
            observable,
        )
        cumulative_corrections = np.einsum(
            "tnp,on->pto",
            exact.correction_states,
            observable,
        )
        columns = np.asarray(
            configuration["primitive_column_scales"],
            dtype=float,
        ).ravel()
        cells = configuration["context"].grid.centers.size
        primary_physical = np.transpose(
            primary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(primary.shape[1], TIME_SAMPLE_COUNT, cells, 5)
        secondary_physical = np.transpose(
            secondary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(secondary.shape[1], TIME_SAMPLE_COUNT, cells, 5)
        restart_physical = np.transpose(
            restarted * columns[:, None],
            (1, 0),
        ).reshape(primary.shape[1], cells, 5)
        cell_map = c6d._lower_height_cell_map(tangents[label])
        base_states = primary_scaled[:, :, base_variant_indices]
        base_integrals = exact.integrated_states[
            :, :, base_variant_indices
        ]
        cell_actions = np.transpose(
            np.einsum("cfn,tnp->tpcf", cell_map, base_states),
            (1, 0, 2, 3),
        )
        cumulative_cell_actions = np.transpose(
            np.einsum("cfn,tnp->tpcf", cell_map, base_integrals),
            (1, 0, 2, 3),
        )
        propagated[label] = {
            "times": times,
            "primary_physical": primary_physical,
            "secondary_base_physical": secondary_physical,
            "restart_physical": restart_physical,
            "signals": signals,
            "cumulative_signals": cumulative,
            "cumulative_corrections": cumulative_corrections,
            "integral_relative_solve_residuals": (
                exact.relative_solve_residuals.T
            ),
            "base_cell_actions": cell_actions,
            "base_cumulative_cell_actions": cumulative_cell_actions,
        }
        report[label] = {
            "cell_count": int(cells),
            "variant_count": int(primary.shape[1]),
            "base_count": int(len(base_variant_indices)),
            "maximum_exact_integral_relative_solve_residual": (
                exact.maximum_relative_solve_residual
            ),
        }
    return propagated, report


def _parent_contract_manifest(manifest: dict) -> dict:
    propagation = manifest["prospective_propagation_contract"]
    return {
        "packet_variants": manifest["profile_variants"],
        "prospective_propagation_contract": {
            "instantaneous_and_cumulative_gates": {
                "minimum_rms_order": propagation["minimum_rms_order"],
                "minimum_maximum_order": propagation[
                    "minimum_maximum_order"
                ],
                "minimum_significant_component_order": 0.75,
                "maximum_fine_normalized_difference": propagation[
                    "maximum_fine_normalized_difference"
                ],
                "minimum_history_cosine": propagation[
                    "minimum_history_cosine"
                ],
                "minimum_refinement_error_cosine": propagation[
                    "minimum_refinement_error_cosine"
                ],
            },
            "state_reference_gates": {
                "maximum_N128_Richardson_error": (
                    c6a2.MAXIMUM_COMPLETE_SEMIGROUP_ERROR
                ),
                "maximum_reference_uncertainty_to_fine_difference": (
                    propagation[
                        "maximum_state_reference_uncertainty_to_fine_difference"
                    ]
                ),
                "maximum_projection_uncertainty_to_fine_difference": 0.10,
                "maximum_restart_uncertainty_to_fine_difference": 0.10,
                "maximum_boundary_integral_uncertainty_to_fine_difference": (
                    0.10
                ),
            },
        },
    }


def _continuum_ratios(
    configurations: dict,
    tangents: dict,
    propagated: dict,
    references: dict,
    metadata: dict,
    observable_scales: np.ndarray,
):
    reports = {}
    arrays = {}
    scale = float(observable_scales[ANGULAR_OBSERVABLE_INDEX])
    for profile_index, name in enumerate(metadata["base_names"]):
        discrete = []
        primary = []
        secondary = []
        for label in LABELS:
            grid = configurations[label]["context"].grid
            discrete.append(
                -float(
                    np.sum(
                        propagated[label]["base_cell_actions"][
                            profile_index, 0, :, ANGULAR_FIELD
                        ]
                    )
                )
            )
            primary_rows = references[name]["primary"].integrate_blocks(
                grid.edges
            )["candidate_lower_height_work"]
            secondary_rows = references[name][
                "secondary"
            ].integrate_blocks(grid.edges)[
                "candidate_lower_height_work"
            ]
            primary.append(float(np.sum(primary_rows[:, ANGULAR_FIELD])))
            secondary.append(
                float(np.sum(secondary_rows[:, ANGULAR_FIELD]))
            )
        discrete = np.asarray(discrete)
        primary = np.asarray(primary)
        secondary = np.asarray(secondary)
        fine_difference = max(
            abs(discrete[2] - discrete[1]) / scale,
            np.finfo(float).tiny,
        )
        uncertainty = abs(primary[2] - secondary[2]) / scale
        reports[name] = {
            "discrete_initial_global_residual": discrete,
            "primary_continuum_initial_global_residual": primary,
            "secondary_continuum_initial_global_residual": secondary,
            "uncertainty_to_fine_difference": (
                uncertainty / fine_difference
            ),
        }
        arrays[f"{name}__initial_discrete_global_residual"] = discrete
        arrays[f"{name}__initial_primary_continuum_residual"] = primary
        arrays[f"{name}__initial_secondary_continuum_residual"] = (
            secondary
        )
    return reports, arrays


def _conditioning_report(
    manifest: dict,
    configurations: dict,
    propagated: dict,
    metadata: dict,
    continuum_reports: dict,
    observable_scales: np.ndarray,
):
    grid = configurations[REFERENCE_LABEL]["context"].grid
    band_indices, band_edges = c6d._band_edges(grid)
    times = np.asarray(propagated[REFERENCE_LABEL]["times"], dtype=float)
    weights = causal_trapezoid_weights(times)
    scale = float(observable_scales[ANGULAR_OBSERVABLE_INDEX])
    contract = manifest["integral_conditioning_contract"]
    reports = {}
    arrays = {
        "conditioning_band_indices": band_indices,
        "conditioning_band_edges_over_rg": band_edges,
    }
    for profile_index, name in enumerate(metadata["base_names"]):
        reports[name] = {}
        for history_name, source_key in (
            ("instantaneous", "base_cell_actions"),
            ("cumulative", "base_cumulative_cell_actions"),
        ):
            coarse = propagated[LABELS[0]][source_key][
                profile_index, :, :, ANGULAR_FIELD
            ]
            medium = causal_restrict_cell_integrals(
                propagated[LABELS[1]][source_key][
                    profile_index, :, :, ANGULAR_FIELD
                ],
                refinement_factor=2,
            )
            fine = causal_restrict_cell_integrals(
                propagated[LABELS[2]][source_key][
                    profile_index, :, :, ANGULAR_FIELD
                ],
                refinement_factor=4,
            )
            bands = tuple(
                causal_partition_cell_integrals(values, band_indices)
                for values in (coarse, medium, fine)
            )
            global_metrics = c6d._scalar_metrics(
                np.sum(coarse, axis=-1),
                np.sum(medium, axis=-1),
                np.sum(fine, axis=-1),
                physical_scale=scale,
            )
            cell_metrics = [
                c6d._scalar_metrics(
                    coarse[:, index],
                    medium[:, index],
                    fine[:, index],
                    physical_scale=scale,
                )
                for index in range(coarse.shape[1])
            ]
            band_metrics = [
                c6d._scalar_metrics(
                    bands[0][:, index],
                    bands[1][:, index],
                    bands[2][:, index],
                    physical_scale=scale,
                )
                for index in range(bands[0].shape[1])
            ]
            cell_response = np.maximum.reduce(
                (
                    np.max(np.abs(coarse), axis=0),
                    np.max(np.abs(medium), axis=0),
                    np.max(np.abs(fine), axis=0),
                )
            ) / scale
            band_response = np.maximum.reduce(
                (
                    np.max(np.abs(bands[0]), axis=0),
                    np.max(np.abs(bands[1]), axis=0),
                    np.max(np.abs(bands[2]), axis=0),
                )
            ) / scale
            active_cells = (
                cell_response
                >= contract["historical_minimum_relative_activity"]
            )
            active_bands = (
                band_response
                >= contract["historical_minimum_relative_activity"]
            )
            first_band_errors = bands[1] - bands[0]
            second_band_errors = bands[2] - bands[1]
            gram = causal_signed_band_gram_matrix(
                second_band_errors,
                physical_scale=scale,
                time_weights=weights,
            )
            signed_norm_squared = float(
                np.sum(
                    weights
                    * (
                        np.sum(second_band_errors, axis=1)
                        / scale
                    )
                    ** 2
                )
            )
            gram_closure = abs(
                float(np.sum(gram)) - signed_norm_squared
            ) / max(
                abs(float(np.sum(gram))),
                abs(signed_norm_squared),
                np.finfo(float).tiny,
            )
            signal_key = (
                "signals"
                if history_name == "instantaneous"
                else "cumulative_signals"
            )
            plus_variant = int(
                np.flatnonzero(
                    (metadata["base_indices"] == profile_index)
                    & (metadata["multipliers"] == 1.0)
                )[0]
            )
            parity = max(
                _relative_defect(
                    np.sum(values, axis=-1),
                    propagated[label][signal_key][
                        plus_variant, :, ANGULAR_OBSERVABLE_INDEX
                    ],
                )
                for label, values in zip(
                    LABELS,
                    (
                        propagated[LABELS[0]][source_key][
                            profile_index, :, :, ANGULAR_FIELD
                        ],
                        propagated[LABELS[1]][source_key][
                            profile_index, :, :, ANGULAR_FIELD
                        ],
                        propagated[LABELS[2]][source_key][
                            profile_index, :, :, ANGULAR_FIELD
                        ],
                    ),
                    strict=True,
                )
            )
            coarse_cancellation = causal_cancellation_ratio(
                first_band_errors,
                time_weights=weights,
            )
            fine_cancellation = causal_cancellation_ratio(
                second_band_errors,
                time_weights=weights,
            )
            envelope = causal_absolute_band_error_envelope(
                second_band_errors,
                physical_scale=scale,
            )
            if np.any(active_cells) and np.any(active_bands):
                decision = causal_integral_conditioning_decision(
                    global_rms_order=global_metrics["observed_rms_order"],
                    global_maximum_order=global_metrics[
                        "observed_maximum_order"
                    ],
                    global_fine_maximum=global_metrics[
                        "maximum_fine_normalized_difference"
                    ],
                    cell_rms_orders=np.asarray(
                        [
                            item["observed_rms_order"]
                            for item in cell_metrics
                        ]
                    ),
                    active_cells=active_cells,
                    band_rms_orders=np.asarray(
                        [
                            item["observed_rms_order"]
                            for item in band_metrics
                        ]
                    ),
                    band_maximum_orders=np.asarray(
                        [
                            item["observed_maximum_order"]
                            for item in band_metrics
                        ]
                    ),
                    band_error_cosines=np.asarray(
                        [
                            item["refinement_error_cosine"]
                            for item in band_metrics
                        ]
                    ),
                    active_bands=active_bands,
                    absolute_band_error_envelope=envelope,
                    coarse_medium_cancellation_ratio=(
                        coarse_cancellation
                    ),
                    medium_fine_cancellation_ratio=fine_cancellation,
                    direct_sum_defect=parity,
                    gram_closure_defect=gram_closure,
                    continuum_uncertainty_to_fine=continuum_reports[name][
                        "uncertainty_to_fine_difference"
                    ],
                    minimum_order=contract[
                        "minimum_direct_or_band_rms_order"
                    ],
                    minimum_error_cosine=contract[
                        "minimum_active_band_refinement_error_cosine"
                    ],
                    maximum_fine_difference=contract[
                        "maximum_global_fine_normalized_difference"
                    ],
                    maximum_cancellation_ratio=contract[
                        "maximum_cancellation_ratio_each_grid_pair"
                    ],
                    maximum_ledger_defect=contract[
                        "maximum_direct_sum_defect"
                    ],
                    maximum_continuum_ratio=contract[
                        "maximum_continuum_uncertainty_to_fine_difference"
                    ],
                )
                decision_payload = {
                    "passed": decision.passed,
                    "route": decision.route,
                    "active_band_count": decision.active_band_count,
                    "maximum_cancellation_ratio": (
                        decision.maximum_cancellation_ratio
                    ),
                    "absolute_band_error_envelope": (
                        decision.absolute_band_error_envelope
                    ),
                }
            else:
                decision_payload = {
                    "passed": True,
                    "route": "inactive_component",
                    "active_band_count": int(
                        np.count_nonzero(active_bands)
                    ),
                    "maximum_cancellation_ratio": max(
                        coarse_cancellation,
                        fine_cancellation,
                    ),
                    "absolute_band_error_envelope": envelope,
                }
            reports[name][history_name] = {
                "decision": decision_payload,
                "global_metrics": global_metrics,
                "minimum_active_cell_rms_order": (
                    float(
                        np.min(
                            np.asarray(
                                [
                                    item["observed_rms_order"]
                                    for item in cell_metrics
                                ]
                            )[active_cells]
                        )
                    )
                    if np.any(active_cells)
                    else None
                ),
                "minimum_active_band_rms_order": (
                    float(
                        np.min(
                            np.asarray(
                                [
                                    item["observed_rms_order"]
                                    for item in band_metrics
                                ]
                            )[active_bands]
                        )
                    )
                    if np.any(active_bands)
                    else None
                ),
                "minimum_active_band_maximum_order": (
                    float(
                        np.min(
                            np.asarray(
                                [
                                    item["observed_maximum_order"]
                                    for item in band_metrics
                                ]
                            )[active_bands]
                        )
                    )
                    if np.any(active_bands)
                    else None
                ),
                "minimum_active_band_error_cosine": (
                    float(
                        np.min(
                            np.asarray(
                                [
                                    item["refinement_error_cosine"]
                                    for item in band_metrics
                                ]
                            )[active_bands]
                        )
                    )
                    if np.any(active_bands)
                    else None
                ),
                "direct_sum_defect": parity,
                "signed_gram_closure_defect": gram_closure,
                "continuum_uncertainty_to_fine_difference": (
                    continuum_reports[name][
                        "uncertainty_to_fine_difference"
                    ]
                ),
            }
            prefix = f"{name}__{history_name}__conditioning_"
            arrays[prefix + "coarse_cells"] = coarse
            arrays[prefix + "medium_on_coarse_cells"] = medium
            arrays[prefix + "fine_on_coarse_cells"] = fine
            arrays[prefix + "medium_fine_band_gram"] = gram
            arrays[prefix + "active_cells"] = active_cells.astype(np.int8)
            arrays[prefix + "active_bands"] = active_bands.astype(np.int8)
    return reports, arrays


def _other_export_gates_pass(metric: dict, propagation: dict) -> bool:
    return bool(
        metric["observed_rms_order"]
        >= propagation["minimum_rms_order"]
        and metric["observed_maximum_order"]
        >= propagation["minimum_maximum_order"]
        and metric["maximum_fine_normalized_difference"]
        <= propagation["maximum_fine_normalized_difference"]
        and metric["history_cosine"]
        >= propagation["minimum_history_cosine"]
        and metric["refinement_error_cosine"]
        >= propagation["minimum_refinement_error_cosine"]
    )


def _prospective_decision(
    manifest: dict,
    direct: dict,
    conditioning: dict,
    metadata: dict,
):
    profiles = manifest["base_profile_definitions"]
    propagation = manifest["prospective_propagation_contract"]
    reports = {}
    base_alternate = {name: False for name in metadata["base_names"]}
    for packet_id, base_index, multiplier in zip(
        metadata["packet_ids"],
        metadata["base_indices"],
        metadata["multipliers"],
        strict=True,
    ):
        name = metadata["base_names"][int(base_index)]
        role = profiles[name]["role"]
        parent = direct["packet_reports"][packet_id]
        if parent["passed"]:
            passed = True
            route = "historical_direct_contract"
        elif role != "unseen_cancellation_stress":
            passed = False
            route = "ordinary_profile_failed_historical_contract"
        else:
            histories_pass = True
            used_alternate = False
            for history_name, metric_name in (
                ("instantaneous", "instantaneous_exports"),
                ("cumulative", "cumulative_exports"),
            ):
                metric = parent[metric_name]
                low = {
                    component
                    for component, order in metric[
                        "component_orders"
                    ].items()
                    if order < 0.75
                }
                if low and low != {TARGET_OBSERVABLE_NAME}:
                    histories_pass = False
                decision = conditioning[name][history_name]["decision"]
                if TARGET_OBSERVABLE_NAME in low:
                    histories_pass = bool(
                        histories_pass
                        and decision["passed"]
                        and decision["route"]
                        == "cancellation_conditioned_band_envelope"
                    )
                    used_alternate = True
                else:
                    histories_pass = bool(
                        histories_pass
                        and metric["minimum_significant_component_order"]
                        >= 0.75
                    )
                histories_pass = bool(
                    histories_pass
                    and _other_export_gates_pass(metric, propagation)
                )
            passed = bool(
                histories_pass
                and parent["state_reference"]["passed"]
                and max(parent["propagation_scaling_defects"].values())
                <= MAXIMUM_PROPAGATION_SCALING_DEFECT
            )
            route = (
                "prospective_cancellation_conditioned_contract"
                if passed and used_alternate
                else "failed"
            )
            base_alternate[name] = bool(
                base_alternate[name] or (passed and used_alternate)
            )
        reports[packet_id] = {
            "base_profile": name,
            "role": role,
            "multiplier": float(multiplier),
            "historical_direct_passed": parent["passed"],
            "route": route,
            "passed": passed,
        }
    stress_names = {
        name
        for name, definition in profiles.items()
        if definition["role"] == "unseen_cancellation_stress"
    }
    alternate_stress = {
        name for name in stress_names if base_alternate[name]
    }
    minimum_alternate = manifest["integral_conditioning_contract"][
        "minimum_unseen_cancellation_stress_profiles_using_alternate_route"
    ]
    all_variants = all(item["passed"] for item in reports.values())
    exercised = len(alternate_stress) >= minimum_alternate
    return {
        "variant_reports": reports,
        "all_variants_passed": all_variants,
        "alternate_stress_profiles": sorted(alternate_stress),
        "alternate_stress_profile_count": len(alternate_stress),
        "minimum_required_alternate_stress_profile_count": (
            minimum_alternate
        ),
        "contract_exercised": exercised,
        "passed": bool(all_variants and exercised),
    }


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "labels": LABELS,
        "time_horizon_s": TIME_HORIZON_S,
        "time_sample_count": TIME_SAMPLE_COUNT,
        "eligibility_contract": manifest["eligibility_contract"],
        "integral_conditioning_contract": (
            manifest["integral_conditioning_contract"]
        ),
        "prospective_propagation_contract": (
            manifest["prospective_propagation_contract"]
        ),
    }


def _finalize(
    *,
    identity: dict,
    parent: dict,
    manifest: dict,
    summary: dict,
    arrays: dict[str, np.ndarray],
) -> dict:
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        **summary,
        "operator_changed": False,
        "parent_classification": parent["classification"],
        "parent_classification_preserved": True,
        "c6c_rejection_preserved": True,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_path": str(
            DECISIVE_ARRAYS.relative_to(ROOT)
        ),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in sorted(arrays.items())
        },
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "PROSPECTIVE VALIDATION"
            if result.get("passed")
            else "REJECTED OR STOPPED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse",
            "HEAD^{tree}",
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python "
            "scripts/"
            "run_causal_inner_integral_conditioning_validation_"
            "wp10c9d6c6e1.py"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                PARENT_CONFIG,
                PARENT_SUMMARY,
                PARENT_MANIFEST,
                PARENT_PROVENANCE,
            )
        },
    }
    _write_json(CONFIG_PATH, _config(manifest))
    _write_json(SUMMARY_PATH, result)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return result


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, manifest = _load_manifest()
    (
        configurations,
        construction_arrays,
        base_directions,
        _evaluators,
        continuum_references,
        eligibility,
        arrays,
    ) = _build_inputs(manifest)
    if not eligibility["passed"]:
        result = _finalize(
            identity=identity,
            parent=parent,
            manifest=manifest,
            arrays=arrays,
            summary={
                "classification": "frozen_integral_profiles_ineligible",
                "authorized_next": "none",
                "passed": False,
                "eligibility_report": eligibility,
                "propagation_executed": False,
                "embedded_export_discrimination_authorized": False,
                "runtime_seconds": float(
                    time.perf_counter() - started
                ),
            },
        )
        print(
            json.dumps(
                {
                    "classification": result["classification"],
                    "propagation_executed": False,
                    "failed_profiles": [
                        name
                        for name, report in eligibility[
                            "profile_reports"
                        ].items()
                        if not report["passed"]
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return result

    variant_directions, metadata = _variant_directions(
        manifest,
        base_directions,
    )
    print("WP10c9d6c6e1: build unchanged monolithic tangents", flush=True)
    tangents, observable_maps, method_reports, _baselines = (
        c3._build_tangents(configurations, construction_arrays)
    )
    method_passed = all(
        method_reports[label]["passed"] for label in LABELS
    )
    propagated, propagation_report = _propagate(
        configurations,
        tangents,
        observable_maps,
        variant_directions,
        metadata,
    )
    pseudo_manifest = _parent_contract_manifest(manifest)
    direct, direct_arrays = c6c._comparison_report(
        pseudo_manifest,
        configurations,
        construction_arrays,
        metadata,
        propagated,
    )
    with np.load(C3_ARRAYS, allow_pickle=False) as source:
        observable_scales = np.asarray(
            source["fixed_physical_observable_scales"],
            dtype=float,
        )
    continuum, continuum_arrays = _continuum_ratios(
        configurations,
        tangents,
        propagated,
        continuum_references,
        metadata,
        observable_scales,
    )
    conditioning, conditioning_arrays = _conditioning_report(
        manifest,
        configurations,
        propagated,
        metadata,
        continuum,
        observable_scales,
    )
    prospective = _prospective_decision(
        manifest,
        direct,
        conditioning,
        metadata,
    )
    maximum_integral_residual = max(
        report["maximum_exact_integral_relative_solve_residual"]
        for report in propagation_report.values()
    )
    method_and_integration_passed = bool(
        method_passed and maximum_integral_residual <= 1.0e-12
    )
    passed = bool(
        eligibility["passed"]
        and method_and_integration_passed
        and prospective["passed"]
    )
    if passed:
        classification = (
            "prospective_integral_conditioning_uniform_validation_"
            "certified"
        )
        authorized_next = "WP10c9d6c7_embedded_discrimination"
    elif (
        prospective["all_variants_passed"]
        and not prospective["contract_exercised"]
    ):
        classification = "integral_conditioning_contract_not_exercised"
        authorized_next = "none"
    else:
        classification = (
            "prospective_integral_conditioning_uniform_validation_failed"
        )
        authorized_next = "none"
    arrays.update(direct_arrays)
    arrays.update(continuum_arrays)
    arrays.update(conditioning_arrays)
    arrays["fixed_physical_observable_scales"] = observable_scales
    for label in LABELS:
        arrays[f"{label}__base_cell_actions"] = propagated[label][
            "base_cell_actions"
        ]
        arrays[f"{label}__base_cumulative_cell_actions"] = propagated[
            label
        ]["base_cumulative_cell_actions"]
    result = _finalize(
        identity=identity,
        parent=parent,
        manifest=manifest,
        arrays=arrays,
        summary={
            "classification": classification,
            "authorized_next": authorized_next,
            "passed": passed,
            "eligibility_report": eligibility,
            "propagation_executed": True,
            "method_reports": {
                label: method_reports[label] for label in LABELS
            },
            "method_passed": method_passed,
            "propagation_report": propagation_report,
            "maximum_exact_integral_relative_solve_residual": (
                maximum_integral_residual
            ),
            "historical_direct_contract_report": direct,
            "continuum_reference_report": continuum,
            "integral_conditioning_report": conditioning,
            "prospective_decision": prospective,
            "embedded_export_discrimination_authorized": bool(passed),
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "authorized_next": authorized_next,
                "historical_direct_failed_count": len(
                    direct["failed_packets"]
                ),
                "prospective_failed_count": sum(
                    not item["passed"]
                    for item in prospective[
                        "variant_reports"
                    ].values()
                ),
                "alternate_stress_profiles": prospective[
                    "alternate_stress_profiles"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
