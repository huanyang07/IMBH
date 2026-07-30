#!/usr/bin/env python3
"""Validate the frozen positive one-way arrival-energy contract.

This package reuses the unchanged N98/N196/N392 monolithic tangents and the
exact WP10c9d6c7c2b2 semidiscrete descriptor/block maps.  It propagates the
frozen c2a3 packet family and applies the c2b3 positive fixed-band
arrival-energy contract.  It performs no embedded or nonlinear evolution and
changes no physical or numerical operator.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import json
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

import run_causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2 as c2b2  # noqa: E402
import run_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1 as c2b1  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_energy_transfer import (  # noqa: E402
    causal_normalized_arrival_energy,
    causal_positive_band_energy_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_one_way_scattering import (  # noqa: E402
    causal_amplitude_scaling_defect,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b4"
ANALYZED_BASE_COMMIT = "f9fbdb866d8f692f482c2fbf7455d4a496a11867"
LEVELS = c2b1.LEVELS
PRIMARY_FAMILIES = c2b1.PRIMARY_FAMILIES
TARGET_FAMILIES = c2b1.TARGET_FAMILIES
MAXIMUM_LEDGER_DEFECT = 1.0e-10
MAXIMUM_REFERENCE_UNCERTAINTY_RATIO = 0.10
OBSERVABILITY_FACTOR = c2a3.OBSERVABILITY_FACTOR

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_uniform_arrival_energy_wp10c9d6c7c2b4.py"
)
THIS_HELPER = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_energy_transfer.py"
)
THIS_HELPER_TEST = "tests/test_causal_inner_energy_transfer.py"
THIS_CANONICAL_TEST = (
    "tests/"
    "test_causal_inner_uniform_arrival_energy_wp10c9d6c7c2b4.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_UNIFORM_ARRIVAL_ENERGY_"
    "WP10C9D6C7C2B4_RESULTS_2026-07-30.md"
)

C2B3_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_semidiscrete_energy_transfer_contract_wp10c9d6c7c2b3"
)
SCOPE_DIRECTORY = c2b1.SCOPE_DIRECTORY
C2A2_DIRECTORY = c2b1.C2A2_DIRECTORY
C7A_DIRECTORY = c2b1.C7A_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_arrival_energy_wp10c9d6c7c2b4"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    THIS_HELPER,
    THIS_HELPER_TEST,
    THIS_CANONICAL_TEST,
)


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first - second)) / scale)


def _validate_parent() -> tuple[dict, dict, dict, dict[str, np.ndarray]]:
    summary = _read_json(C2B3_DIRECTORY / "summary.json")
    manifest = _read_json(C2B3_DIRECTORY / "transfer_manifest.json")
    if (
        summary["classification"]
        != "positive_fixed_band_arrival_energy_contract_frozen_"
        "uniform_validation_authorized"
        or not summary["passed"]
        or summary["propagation_executed"]
        or summary["operator_changed"]
        or summary["authorized_next"]
        != "WP10c9d6c7c2b4_one_way_uniform_arrival_energy_validation"
        or summary["manifest_sha256"] != manifest["manifest_sha256"]
        or not summary["binding_decision"]["uniform_c2b4_authorized"]
        or summary["binding_decision"]["embedded_c2c2_authorized"]
        or summary["binding_decision"][
            "operator_or_interface_redesign_authorized"
        ]
    ):
        raise RuntimeError("WP10c9d6c7c2b3 binding status changed")
    if _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT:
        raise RuntimeError("analyzed base commit changed")
    arrays = _load_npz(C2B3_DIRECTORY / "decisive_arrays.npz")
    scope = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")
    return summary, manifest, scope, arrays


def _input_hashes() -> dict[str, str]:
    paths = (
        C2B3_DIRECTORY / "config.json",
        C2B3_DIRECTORY / "transfer_manifest.json",
        C2B3_DIRECTORY / "summary.json",
        C2B3_DIRECTORY / "decisive_arrays.npz",
        c2b2.C2B1_DIRECTORY / "summary.json",
        c2b2.C2B1_DIRECTORY / "decisive_arrays.npz",
        c2b2.CANONICAL_DIRECTORY / "summary.json",
        c2b2.CANONICAL_DIRECTORY / "decisive_arrays.npz",
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "decisive_arrays.npz",
        C7A_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _binding_case_indices(cases: list[dict]) -> list[int]:
    return [
        index
        for index, case in enumerate(cases)
        if bool(case["binding"])
    ]


def _representative_case_indices(cases: list[dict]) -> dict[str, int]:
    return {
        family: next(
            index
            for index, case in enumerate(cases)
            if case["family"] == family
            and case["sign"] == 1
            and case["amplitude"] == 1.0
        )
        for family in PRIMARY_FAMILIES
    }


def _level_energy_data(
    level: dict,
    propagated: dict,
    cases: list[dict],
    *,
    arrival_windows: dict[str, tuple[float, float]],
    nuisance_windows: dict[str, list[tuple[float, float]]],
    nuisance_bands_N98: np.ndarray,
) -> dict:
    cells = int(level["cells"])
    factor = cells // LEVELS[0]
    physical = np.asarray(propagated["physical"], dtype=float)
    times = np.asarray(propagated["times"], dtype=float)
    log_edges = np.log(np.asarray(level["grid"].edges, dtype=float))
    source = (
        c2a3.PACKET_SUPPORT[0] * factor,
        c2a3.PACKET_SUPPORT[1] * factor,
    )
    primary_band = (
        c2a3.DOWNSTREAM_MEASUREMENT_FACE * factor,
        c2a3.PATCH_INTERFACE_FACE * factor,
    )
    source_energy = causal_positive_band_energy_history(
        physical[:1],
        log_edges=log_edges,
        energy_metrics=level["energy"],
        projectors=level["projectors"],
        lower_face=source[0],
        upper_face=source[1],
    )
    initial = np.asarray(source_energy.total_energy[0], dtype=float)
    binding = _binding_case_indices(cases)
    if np.any(initial[binding] <= 0.0):
        raise RuntimeError(f"N{cells} binding source energy is not positive")
    binding_lookup = {
        case_index: local_index
        for local_index, case_index in enumerate(binding)
    }

    bands = [primary_band]
    for lower, upper in np.asarray(nuisance_bands_N98, dtype=int):
        band = (int(lower) * factor, int(upper) * factor)
        if band not in bands:
            bands.append(band)
    band_histories = {
        band: causal_positive_band_energy_history(
            physical,
            log_edges=log_edges,
            energy_metrics=level["energy"],
            projectors=level["projectors"],
            lower_face=band[0],
            upper_face=band[1],
        )
        for band in bands
    }
    primary_history = band_histories[primary_band]
    binding_primary_history = replace(
        primary_history,
        total_energy=primary_history.total_energy[:, binding],
        family_energy=primary_history.family_energy[:, binding],
    )
    family_report = {}
    cases_report = {}
    variant_values: dict[str, dict[str, list[float]]] = {}
    primary_histories: dict[str, dict[str, np.ndarray]] = {}
    raw_averages: dict[str, float] = {}

    for index in binding:
        case = cases[index]
        local_index = binding_lookup[index]
        family = str(case["family"])
        target = tuple(TARGET_FAMILIES[family])
        window = arrival_windows[family]
        measured = causal_normalized_arrival_energy(
            times,
            binding_primary_history,
            initial_source_energy=initial[binding],
            window_seconds=window,
        )
        total = float(measured.total_time_average[local_index])
        target_value = float(
            np.sum(
                measured.family_time_average[
                    local_index,
                    list(target),
                ]
            )
        )
        leakage_value = float(
            np.sum(measured.family_time_average[local_index])
            - target_value
        )
        normalized_total_history = (
            primary_history.total_energy[:, index] / initial[index]
        )
        normalized_family_history = (
            primary_history.family_energy[:, index] / initial[index]
        )
        target_history = np.sum(
            normalized_family_history[:, list(target)],
            axis=1,
        )
        leakage_history = (
            np.sum(normalized_family_history, axis=1) - target_history
        )
        primary_histories[case["name"]] = {
            "total": normalized_total_history,
            "target": target_history,
            "leakage": leakage_history,
        }
        raw_averages[case["name"]] = total * initial[index]

        variants = {"band": [], "window": [], "time": []}
        for band, history in band_histories.items():
            evaluated = causal_normalized_arrival_energy(
                times,
                replace(
                    history,
                    total_energy=history.total_energy[:, binding],
                    family_energy=history.family_energy[:, binding],
                ),
                initial_source_energy=initial[binding],
                window_seconds=window,
            )
            values = (
                float(evaluated.total_time_average[local_index]),
                float(
                    np.sum(
                        evaluated.family_time_average[
                            local_index,
                            list(target),
                        ]
                    )
                ),
            )
            variants["band"].append(
                (
                    values[0],
                    values[1],
                    values[0] - values[1],
                    float(evaluated.peak_total[local_index]),
                )
            )
        for varied_window in nuisance_windows[family]:
            evaluated = causal_normalized_arrival_energy(
                times,
                binding_primary_history,
                initial_source_energy=initial[binding],
                window_seconds=varied_window,
            )
            values = (
                float(evaluated.total_time_average[local_index]),
                float(
                    np.sum(
                        evaluated.family_time_average[
                            local_index,
                            list(target),
                        ]
                    )
                ),
            )
            variants["window"].append(
                (
                    values[0],
                    values[1],
                    values[0] - values[1],
                    float(evaluated.peak_total[local_index]),
                )
            )
        for stride in (1, 2, 4):
            evaluated = causal_normalized_arrival_energy(
                times[::stride],
                replace(
                    binding_primary_history,
                    total_energy=(
                        binding_primary_history.total_energy[::stride]
                    ),
                    family_energy=(
                        binding_primary_history.family_energy[::stride]
                    ),
                ),
                initial_source_energy=initial[binding],
                window_seconds=window,
            )
            values = (
                float(evaluated.total_time_average[local_index]),
                float(
                    np.sum(
                        evaluated.family_time_average[
                            local_index,
                            list(target),
                        ]
                    )
                ),
            )
            variants["time"].append(
                (
                    values[0],
                    values[1],
                    values[0] - values[1],
                    float(evaluated.peak_total[local_index]),
                )
            )
        variant_values[case["name"]] = {
            name: [list(item) for item in values]
            for name, values in variants.items()
        }
        cases_report[case["name"]] = {
            "family": family,
            "sign": int(case["sign"]),
            "amplitude": float(case["amplitude"]),
            "initial_source_energy": float(initial[index]),
            "total_arrival": total,
            "target_arrival": target_value,
            "opposite_family_leakage": leakage_value,
            "peak_total_arrival": float(
                measured.peak_total[local_index]
            ),
            "integrated_family_partition_relative_defect": (
                measured.maximum_integrated_partition_relative_defect
            ),
        }
        family_report.setdefault(family, []).append(case["name"])

    material_index = next(
        index
        for index, case in enumerate(cases)
        if case["family"] == "material_null"
    )
    zero_index = next(
        index
        for index, case in enumerate(cases)
        if case["family"] == "zero_null"
    )
    return {
        "initial_source_energy": initial,
        "primary_history": primary_history,
        "primary_normalized_histories": primary_histories,
        "variant_values": variant_values,
        "raw_averages": raw_averages,
        "cases": cases_report,
        "family_case_names": family_report,
        "maximum_band_partition_relative_defect": max(
            history.maximum_family_partition_relative_defect
            for history in band_histories.values()
        ),
        "minimum_band_total_energy": min(
            history.minimum_total_energy
            for history in band_histories.values()
        ),
        "minimum_band_family_energy": min(
            history.minimum_family_energy
            for history in band_histories.values()
        ),
        "zero_null_maximum_energy": float(
            np.max(primary_history.total_energy[:, zero_index])
        ),
        "material_null_peak_total_energy": float(
            np.max(primary_history.total_energy[:, material_index])
        ),
        "restart_defect": float(propagated["restart_defect"]),
    }


def _nuisance_error_bounds(
    nominal: np.ndarray,
    per_level_variants: list[dict[str, list[list[float]]]],
    component: int,
    restart_defects: np.ndarray,
) -> dict:
    nominal = np.asarray(nominal, dtype=float)
    errors = np.diff(nominal)
    bounds_cm = {}
    bounds_mf = {}
    for category in ("band", "window", "time"):
        arrays = [
            np.asarray(item[category], dtype=float)[:, component]
            for item in per_level_variants
        ]
        count = min(array.size for array in arrays)
        coarse = arrays[0][:count]
        medium = arrays[1][:count]
        fine = arrays[2][:count]
        bounds_cm[category] = float(
            np.max(np.abs((medium - coarse) - errors[0]))
        )
        bounds_mf[category] = float(
            np.max(np.abs((fine - medium) - errors[1]))
        )
    scale = max(float(np.max(np.abs(nominal))), np.finfo(float).tiny)
    bounds_cm["restart_roundoff"] = float(
        scale * (restart_defects[0] + restart_defects[1])
    )
    bounds_mf["restart_roundoff"] = float(
        scale * (restart_defects[1] + restart_defects[2])
    )
    bounds_cm["projection_and_subspace"] = 0.0
    bounds_mf["projection_and_subspace"] = 0.0
    total_cm = float(sum(bounds_cm.values()))
    total_mf = float(sum(bounds_mf.values()))
    return {
        "coarse_medium_components": bounds_cm,
        "medium_fine_components": bounds_mf,
        "coarse_medium_conservative_sum": total_cm,
        "medium_fine_conservative_sum": total_mf,
        "RSS_used": False,
    }


def _scalar_contract(
    values: np.ndarray,
    uncertainty: dict,
    gates: dict,
) -> dict:
    data = np.asarray(values, dtype=float)
    tiny = np.finfo(float).tiny
    first = float(data[1] - data[0])
    second = float(data[2] - data[1])
    order = float(np.log2(max(abs(first), tiny) / max(abs(second), tiny)))
    scale = max(float(np.max(np.abs(data))), tiny)
    fine = abs(second) / scale
    ucm = float(uncertainty["coarse_medium_conservative_sum"])
    umf = float(uncertainty["medium_fine_conservative_sum"])
    observable = bool(
        abs(first) >= OBSERVABILITY_FACTOR * ucm
        and abs(second) >= OBSERVABILITY_FACTOR * umf
    )
    cosine = 1.0 if first * second >= 0.0 else -1.0
    direction_passed = bool(
        not observable
        or cosine >= gates["minimum_observable_error_cosine"]
    )
    reference_ratio = umf / max(abs(second), tiny)
    passed = bool(
        order >= gates["minimum_RMS_order"]
        and fine <= gates["maximum_fine_normalized_difference"]
        and direction_passed
        and reference_ratio <= MAXIMUM_REFERENCE_UNCERTAINTY_RATIO
    )
    return {
        "values": data.tolist(),
        "observed_order": order,
        "maximum_fine_normalized_difference": fine,
        "refinement_error_cosine": cosine,
        "error_direction_observable": observable,
        "direction_classification": (
            "binding_pass"
            if observable and direction_passed
            else "binding_fail"
            if observable
            else "direction_not_certifying_because_error_is_below_"
            "observability"
        ),
        "reference_uncertainty_to_medium_fine_difference": (
            reference_ratio
        ),
        "uncertainty": uncertainty,
        "passed": passed,
    }


def _history_contract(
    histories: list[np.ndarray],
    *,
    gates: dict,
    restart_defects: np.ndarray,
) -> dict:
    arrays = [np.asarray(item, dtype=float)[:, None] for item in histories]
    metrics = causal_packet_history_metrics(
        *arrays,
        physical_scales=np.ones(1),
        minimum_rms_order=gates["minimum_RMS_order"],
        minimum_maximum_order=gates["minimum_maximum_order"],
        minimum_significant_component_order=gates[
            "minimum_component_order"
        ],
        maximum_fine_normalized_difference=gates[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=gates["minimum_history_cosine"],
        minimum_refinement_error_cosine=gates[
            "minimum_observable_error_cosine"
        ],
    )
    coarse, medium, fine = (
        np.asarray(item, dtype=float) for item in histories
    )
    cm = medium - coarse
    mf = fine - medium
    cm_norm = float(np.linalg.norm(cm))
    mf_norm = float(np.linalg.norm(mf))
    scale = max(
        float(np.linalg.norm(coarse)),
        float(np.linalg.norm(medium)),
        float(np.linalg.norm(fine)),
        np.finfo(float).tiny,
    )
    ucm = scale * float(restart_defects[0] + restart_defects[1])
    umf = scale * float(restart_defects[1] + restart_defects[2])
    reference_ratio = umf / max(mf_norm, np.finfo(float).tiny)
    observable = bool(
        cm_norm >= OBSERVABILITY_FACTOR * ucm
        and mf_norm >= OBSERVABILITY_FACTOR * umf
    )
    direction_passed = bool(
        not observable
        or metrics.refinement_error_cosine
        >= gates["minimum_observable_error_cosine"]
    )
    passed = bool(
        metrics.observed_rms_order >= gates["minimum_RMS_order"]
        and metrics.observed_maximum_order
        >= gates["minimum_maximum_order"]
        and metrics.minimum_significant_component_order
        >= gates["minimum_component_order"]
        and metrics.maximum_fine_normalized_difference
        <= gates["maximum_fine_normalized_difference"]
        and metrics.history_cosine >= gates["minimum_history_cosine"]
        and direction_passed
        and reference_ratio <= MAXIMUM_REFERENCE_UNCERTAINTY_RATIO
    )
    return {
        "observed_RMS_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "minimum_component_order": (
            metrics.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
        "error_direction_observable": observable,
        "reference_uncertainty_to_medium_fine_difference": (
            reference_ratio
        ),
        "passed": passed,
    }


def _evaluate(
    levels: dict[int, dict],
    propagated: dict[int, dict],
    cases: list[dict],
    level_energy: dict[int, dict],
    exact_ledgers: dict[int, dict],
    b1_result: dict,
    gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    binding = _binding_case_indices(cases)
    restart = np.asarray(
        [level_energy[cells]["restart_defect"] for cells in LEVELS],
        dtype=float,
    )
    decisive: dict[str, np.ndarray] = {
        "reference_levels": np.asarray(LEVELS, dtype=np.int64),
        "primary_times_seconds": np.asarray(
            propagated[LEVELS[0]]["times"][::2],
            dtype=float,
        ),
        "restart_defects": restart,
    }
    case_results = {}
    for index in binding:
        case = cases[index]
        name = case["name"]
        scalar_values = {
            key: np.asarray(
                [
                    level_energy[cells]["cases"][name][key]
                    for cells in LEVELS
                ],
                dtype=float,
            )
            for key in (
                "total_arrival",
                "target_arrival",
                "opposite_family_leakage",
                "peak_total_arrival",
            )
        }
        per_level_variants = [
            level_energy[cells]["variant_values"][name]
            for cells in LEVELS
        ]
        contracts = {}
        for component, key in enumerate(
            (
                "total_arrival",
                "target_arrival",
                "opposite_family_leakage",
            )
        ):
            uncertainty = _nuisance_error_bounds(
                scalar_values[key],
                per_level_variants,
                component,
                restart,
            )
            contracts[key] = _scalar_contract(
                scalar_values[key],
                uncertainty,
                gates,
            )
        peak_uncertainty = _nuisance_error_bounds(
            scalar_values["peak_total_arrival"],
            per_level_variants,
            3,
            restart,
        )
        contracts["peak_total_arrival"] = _scalar_contract(
            scalar_values["peak_total_arrival"],
            peak_uncertainty,
            gates,
        )
        histories = {}
        for key in ("total", "target", "leakage"):
            values = [
                level_energy[cells]["primary_normalized_histories"][name][
                    key
                ][::2]
                for cells in LEVELS
            ]
            histories[key] = _history_contract(
                values,
                gates=gates,
                restart_defects=restart,
            )
            if (
                case["sign"] == 1
                and case["amplitude"] == 1.0
            ):
                for cells, history in zip(LEVELS, values, strict=True):
                    decisive[
                        f"N{cells}__{case['family']}__{key}_history"
                    ] = np.asarray(history)
        partition = max(
            level_energy[cells]["cases"][name][
                "integrated_family_partition_relative_defect"
            ]
            for cells in LEVELS
        )
        passed = bool(
            all(item["passed"] for item in contracts.values())
            and all(item["passed"] for item in histories.values())
            and partition <= MAXIMUM_LEDGER_DEFECT
        )
        case_results[name] = {
            "family": case["family"],
            "sign": case["sign"],
            "amplitude": case["amplitude"],
            "scalar_contracts": contracts,
            "history_contracts": histories,
            "maximum_family_partition_relative_defect": partition,
            "passed": passed,
        }
        for key, values in scalar_values.items():
            decisive[f"{name}__{key}"] = values

    scaling_defects = []
    for family in PRIMARY_FAMILIES:
        indices = {
            (case["sign"], case["amplitude"]): case["name"]
            for case in cases
            if case["family"] == family
        }
        for cells in LEVELS:
            raw = level_energy[cells]["raw_averages"]
            full = raw[indices[(1, 1.0)]]
            half = raw[indices[(1, 0.5)]]
            negative = raw[indices[(-1, 1.0)]]
            scaling_defects.extend(
                (
                    abs(half - 0.25 * full)
                    / max(abs(full), np.finfo(float).tiny),
                    abs(negative - full)
                    / max(abs(full), np.finfo(float).tiny),
                )
            )
            initial = level_energy[cells]["initial_source_energy"]
            case_lookup = {
                case["name"]: index
                for index, case in enumerate(cases)
            }
            scaling_defects.extend(
                (
                    abs(
                        initial[case_lookup[indices[(1, 0.5)]]]
                        - 0.25 * initial[
                            case_lookup[indices[(1, 1.0)]]
                        ]
                    )
                    / max(
                        abs(
                            initial[
                                case_lookup[indices[(1, 1.0)]]
                            ]
                        ),
                        np.finfo(float).tiny,
                    ),
                    abs(
                        initial[case_lookup[indices[(-1, 1.0)]]]
                        - initial[case_lookup[indices[(1, 1.0)]]]
                    )
                    / max(
                        abs(
                            initial[
                                case_lookup[indices[(1, 1.0)]]
                            ]
                        ),
                        np.finfo(float).tiny,
                    ),
                )
            )
    maximum_scaling = float(max(scaling_defects))
    maximum_zero = max(
        level_energy[cells]["zero_null_maximum_energy"]
        for cells in LEVELS
    )
    maximum_partition = max(
        level_energy[cells]["maximum_band_partition_relative_defect"]
        for cells in LEVELS
    )
    minimum_total = min(
        level_energy[cells]["minimum_band_total_energy"]
        for cells in LEVELS
    )
    minimum_family = min(
        level_energy[cells]["minimum_band_family_energy"]
        for cells in LEVELS
    )
    positive_contract_passed = bool(
        maximum_scaling <= 1.0e-12
        and maximum_zero == 0.0
        and maximum_partition <= MAXIMUM_LEDGER_DEFECT
        and minimum_total >= -1.0e-12
        and minimum_family >= -1.0e-12
    )

    exact_method = {
        f"N{cells}": exact_ledgers[cells]["method"]
        for cells in LEVELS
    }
    representative = _representative_case_indices(cases)
    stored_parity = []
    for cells in LEVELS:
        numerical = np.asarray(
            exact_ledgers[cells]["arrays"]["stored_energy"],
            dtype=float,
        )
        positive = level_energy[cells]["primary_history"].total_energy[
            :,
            [representative[family] for family in PRIMARY_FAMILIES],
        ]
        stored_parity.append(_relative_defect(numerical, positive))
    maximum_stored_parity = float(max(stored_parity))
    exact_ledger_passed = bool(
        maximum_stored_parity <= MAXIMUM_LEDGER_DEFECT
        and max(
            exact_method[f"N{cells}"]["maximum_block_power_defect"]
            for cells in LEVELS
        )
        <= MAXIMUM_LEDGER_DEFECT
        and max(
            exact_method[f"N{cells}"]["maximum_face_power_defect"]
            for cells in LEVELS
        )
        <= MAXIMUM_LEDGER_DEFECT
        and max(
            exact_method[f"N{cells}"][
                "maximum_time_integrated_energy_defect"
            ]
            for cells in LEVELS
        )
        <= c2b2.MAXIMUM_ENERGY_INTEGRATION_DEFECT
    )
    method_passed = bool(
        all(levels[cells]["method_report"]["passed"] for cells in LEVELS)
        and b1_result["binding_decision"]["method_passed"]
        and exact_ledger_passed
    )
    tier_I_passed = bool(
        b1_result["binding_decision"]["tier_I_passed"]
    )
    arrival_passed = all(item["passed"] for item in case_results.values())
    passed = bool(
        method_passed
        and tier_I_passed
        and positive_contract_passed
        and arrival_passed
    )
    classification = (
        "one_way_uniform_arrival_energy_certified_"
        "embedded_discrimination_authorized"
        if passed
        else "one_way_uniform_arrival_energy_validation_failed_"
        "embedded_discrimination_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c2c2_one_way_embedded_arrival_energy_discrimination"
        if passed
        else "WP10c9d6c7c2b5_frozen_arrival_energy_failure_audit"
    )
    report = {
        "method": {
            "c2b1_tangent_method": b1_result["method"],
            "exact_semidiscrete_energy": exact_method,
            "maximum_positive_to_semidiscrete_stored_energy_defect": (
                maximum_stored_parity
            ),
            "passed": method_passed,
        },
        "tier_I": b1_result["tier_I"],
        "arrival_energy": case_results,
        "amplitude_and_positive_controls": {
            "maximum_quadratic_or_sign_scaling_defect": maximum_scaling,
            "maximum_zero_state_energy": maximum_zero,
            "maximum_family_partition_relative_defect": maximum_partition,
            "minimum_total_energy_before_roundoff_clamp": minimum_total,
            "minimum_family_energy_before_roundoff_clamp": minimum_family,
            "passed": positive_contract_passed,
        },
        "binding_decision": {
            "c2b1_rejection_preserved": True,
            "c2b2_interpretation_preserved": True,
            "method_passed": method_passed,
            "tier_I_passed": tier_I_passed,
            "all_arrival_energy_cases_passed": arrival_passed,
            "amplitude_positive_null_controls_passed": (
                positive_contract_passed
            ),
            "uniform_c2b4_passed": passed,
            "one_way_embedded_c2c2_authorized": passed,
            "old_embedded_c2c1_authorized": False,
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": passed,
    }
    return report, decisive


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "embedded_or_nonlinear_propagation_executed": False,
        "reference_levels": list(LEVELS),
        "positive_transfer_contract": manifest["prospective_observable"],
        "family_transfer_contract": manifest["family_transfer"],
        "geometry_and_windows": manifest["geometry_and_windows"],
        "uncertainty_contract": manifest["uncertainty_contract"],
        "gates": manifest["future_uniform_gates"],
        "historical_local_face_ratio_is_nonbinding": True,
    }


def _write_report(summary: dict) -> None:
    decision = summary["binding_decision"]
    lines = [
        "# WP10c9d6c7c2b4 — Uniform positive arrival-energy validation",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Passed: `{summary['passed']}`",
        "- Operator changed: `False`",
        "- Embedded, nonlinear, fixed-Q, and reduced evolution were not run.",
        "",
        "## Binding result",
        "",
        (
            "Method / Tier I / arrival / positivity controls: "
            f"`{decision['method_passed']}` / "
            f"`{decision['tier_I_passed']}` / "
            f"`{decision['all_arrival_energy_cases_passed']}` / "
            f"`{decision['amplitude_positive_null_controls_passed']}`."
        ),
        "",
        "| Family | total A(N98) | A(N196) | A(N392) | order | fine diff. | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for family, index in _representative_case_indices_from_summary(
        summary
    ).items():
        item = summary["arrival_energy"][index][
            "scalar_contracts"
        ]["total_arrival"]
        values = item["values"]
        lines.append(
            f"| {family} | {values[0]:.8e} | {values[1]:.8e} | "
            f"{values[2]:.8e} | {item['observed_order']:.4f} | "
            f"{item['maximum_fine_normalized_difference']:.4e} | "
            f"{item['passed']} |"
        )
    failed = [
        name
        for name, item in summary["arrival_energy"].items()
        if not item["passed"]
    ]
    lines.extend(
        (
            "",
            "The c2b1 local face-transmission rejection and c2b2 exact "
            "semidiscrete interpretation remain unchanged. The new "
            "observable uses positive stored energy in the frozen receiving "
            "band and never divides two local descriptor-dual face powers.",
            "",
            "## Failed binding cases",
            "",
            ", ".join(f"`{name}`" for name in failed) if failed else "None.",
            "",
            "## Next step",
            "",
            f"`{summary['authorized_next']}`",
            "",
        )
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _representative_case_indices_from_summary(summary: dict) -> dict[str, str]:
    return {
        family: next(
            name
            for name, item in summary["arrival_energy"].items()
            if item["family"] == family
            and item["sign"] == 1
            and item["amplitude"] == 1.0
        )
        for family in PRIMARY_FAMILIES
    }


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{c2a._sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_catalog() -> None:
    rows = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
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
                        "sha256": c2a._sha256(path),
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
    canonical_summary = _read_json(CANONICAL_SUMMARY)
    canonical_summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    c2a._write_json(CANONICAL_SUMMARY, canonical_summary)


def main() -> None:
    started = time.perf_counter()
    parent_summary, manifest, scope, contract_arrays = _validate_parent()
    (
        _geometry_summary,
        _geometry_manifest,
        _geometry_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    c2a2_arrays = _load_npz(C2A2_DIRECTORY / "decisive_arrays.npz")
    c7a_arrays = _load_npz(C7A_DIRECTORY / "decisive_arrays.npz")
    scope_arrays = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    base_edges = np.asarray(c2a2_arrays["patch_edges"], dtype=float)
    support_log_bounds = (
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[0]])),
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[1]])),
    )
    travel = np.asarray(scope_arrays["travel_windows_seconds"], dtype=float)
    old_windows = {
        "interface": {
            family: tuple(travel[index, :2])
            for index, family in enumerate(PRIMARY_FAMILIES)
        },
        "downstream": {
            family: tuple(travel[index, 2:])
            for index, family in enumerate(PRIMARY_FAMILIES)
        },
    }
    arrival_windows_array = np.asarray(
        contract_arrays["primary_arrival_windows_seconds"],
        dtype=float,
    )
    nuisance_windows_array = np.asarray(
        contract_arrays["arrival_window_nuisance_seconds"],
        dtype=float,
    )
    horizon = float(
        scope["packet_and_window_contract"]["experiment_end_seconds"]
    )
    arrival_windows = {
        family: tuple(arrival_windows_array[index])
        for index, family in enumerate(PRIMARY_FAMILIES)
    }
    nuisance_windows = {
        family: [
            (
                float(nuisance_windows_array[factor, index, 0]),
                min(
                    float(nuisance_windows_array[factor, index, 1]),
                    horizon,
                ),
            )
            for factor in range(nuisance_windows_array.shape[0])
        ]
        for index, family in enumerate(PRIMARY_FAMILIES)
    }

    levels = c2b2._build_levels(
        base_edges,
        parent_context,
        parent_base,
        field_scales,
    )
    initials = {}
    cases = None
    for cells, level in levels.items():
        initial, current_cases, packets = c2b1._packet_matrix(
            level,
            scope_arrays,
            support_log_bounds,
        )
        if cases is None:
            cases = current_cases
        elif cases != current_cases:
            raise RuntimeError("packet ordering changed across levels")
        if cells == LEVELS[0]:
            for family in (
                "acoustic",
                "shear",
                "mixed_shear_acoustic",
                "material_null",
                "zero_null",
            ):
                if (
                    _relative_defect(
                        packets[family],
                        scope_arrays[f"packet__{family}"],
                    )
                    > 1.0e-12
                ):
                    raise RuntimeError(f"N98 packet {family} replay changed")
        initials[cells] = initial
    assert cases is not None
    if not all(level["method_report"]["passed"] for level in levels.values()):
        raise RuntimeError("method preflight failed; propagation forbidden")

    common_log_centers = np.log(np.asarray(base_edges[:-1])) + 0.5 * np.diff(
        np.log(base_edges)
    )
    propagated = {
        cells: c2b1._propagate_level(
            levels[cells],
            initials[cells],
            cases,
            old_windows,
            horizon,
            common_log_centers,
        )
        for cells in LEVELS
    }
    level_energy = {
        cells: _level_energy_data(
            levels[cells],
            propagated[cells],
            cases,
            arrival_windows=arrival_windows,
            nuisance_windows=nuisance_windows,
            nuisance_bands_N98=np.asarray(
                contract_arrays["receiving_band_nuisance_faces_N98"],
                dtype=int,
            ),
        )
        for cells in LEVELS
    }
    exact_ledgers = {}
    for cells in LEVELS:
        report, arrays = c2b2._audit_level(
            levels[cells],
            propagated[cells],
            cases,
            old_windows,
        )
        exact_ledgers[cells] = {"method": report["method"], "arrays": arrays}
    b1_result, _b1_decisive = c2b1._evaluate_results(
        levels,
        propagated,
        cases,
        np.asarray(
            c7a_arrays["fixed_physical_observable_scales"],
            dtype=float,
        ),
    )
    result, decisive = _evaluate(
        levels,
        propagated,
        cases,
        level_energy,
        exact_ledgers,
        b1_result,
        manifest["future_uniform_gates"],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": True,
        "embedded_or_nonlinear_propagation_executed": False,
        "historical_classifications_preserved": True,
        "parent_classification": parent_summary["classification"],
        "old_local_face_transmission_contract_binding": False,
        **result,
        "runtime_seconds": time.perf_counter() - started,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = _config(manifest)
    c2a._write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_manifest = {
        relative: c2a._sha256(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
        if (ROOT / relative).is_file()
    }
    summary["decisive_array_hashes"] = {
        name: causal_array_sha256(value)
        for name, value in decisive.items()
    }
    summary["decisive_arrays_sha256"] = c2a._sha256(DECISIVE_ARRAYS)
    summary["config_sha256"] = c2a._sha256(CONFIG_PATH)
    summary["implementation_source_hashes"] = source_manifest
    summary["implementation_source_manifest_sha256"] = (
        causal_canonical_json_sha256(source_manifest)
    )
    summary["input_hashes"] = _input_hashes()
    c2a._write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": _git_value(
            "rev-parse", f"{ANALYZED_BASE_COMMIT}^"
        ),
        "analyzed_base_tree": _git_value(
            "rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}"
        ),
        "implementation_head_before_commit": _git_value(
            "rev-parse", "HEAD"
        ),
        "current_branch": _git_value("branch", "--show-current"),
        "input_hashes": _input_hashes(),
        "implementation_source_hashes": source_manifest,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "command": f"{sys.executable} {THIS_RUNNER}",
        "scientific_status": (
            "CERTIFIED" if summary["passed"] else "REJECTED"
        ),
    }
    c2a._write_json(PROVENANCE_PATH, provenance)
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    print(json.dumps(summary["binding_decision"], indent=2), flush=True)
    print(f"classification={summary['classification']}", flush=True)


if __name__ == "__main__":
    main()
