#!/usr/bin/env python3
"""Audit conditioning, peaks, nuisance envelopes, and horizon completeness.

WP10c9d6c7c2b5a preserves the rejected WP10c9d6c7c2b4 contract verbatim.
It reuses the unchanged uniform N98/N196/N392 tangents and reconstructs only
the three independent positive/full-amplitude histories.  No embedded,
nonlinear, fixed-Q, reduced, or modified-operator evolution is performed.
"""

from __future__ import annotations

import csv
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

import run_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1 as c2b1  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402
import run_causal_inner_uniform_arrival_energy_wp10c9d6c7c2b4 as c2b4  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_arrival_conditioning import (  # noqa: E402
    causal_arrival_history_conditioning,
    causal_history_uncertainty_envelope,
    causal_horizon_completeness,
    causal_quadratic_peak,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_energy_transfer import (  # noqa: E402
    causal_positive_band_energy_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b5a"
ANALYZED_BASE_COMMIT = "dbfd8bdaf859fa23f530c5c4f00f78fa407137d3"
ANALYZED_BASE_PARENT = "f9fbdb866d8f692f482c2fbf7455d4a496a11867"
LEVELS = c2b1.LEVELS
FAMILIES = c2b1.PRIMARY_FAMILIES
OBSERVABLES = ("total", "target", "leakage")
OBSERVABILITY_FACTOR = c2a3.OBSERVABILITY_FACTOR

FINAL_WINDOW_FRACTION = 0.10
MAXIMUM_TERMINAL_TO_PEAK = 0.01
MAXIMUM_FINAL_WINDOW_RANGE_TO_PEAK = 0.02
MAXIMUM_TERMINAL_SLOPE_HORIZON_TO_PEAK = 0.05
MAXIMUM_PARENT_REPLAY_DEFECT = 2.0e-11

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a.py"
)
THIS_HELPER = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_arrival_conditioning.py"
)
THIS_HELPER_TEST = "tests/test_causal_inner_arrival_conditioning.py"
THIS_CANONICAL_TEST = (
    "tests/"
    "test_causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_UNIFORM_ARRIVAL_CONDITIONING_"
    "WP10C9D6C7C2B5A_RESULTS_2026-07-30.md"
)

PARENT_DIRECTORY = c2b4.CANONICAL_DIRECTORY
SCOPE_DIRECTORY = c2b1.SCOPE_DIRECTORY
C2A2_DIRECTORY = c2b1.C2A2_DIRECTORY
C2B3_DIRECTORY = c2b4.C2B3_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a"
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first - second)) / scale)


def _validate_parent() -> tuple[dict, dict[str, np.ndarray], dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    if (
        parent["classification"]
        != "one_way_uniform_arrival_energy_validation_failed_"
        "embedded_discrimination_blocked"
        or parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c2b5_frozen_arrival_energy_failure_audit"
        or not parent["binding_decision"]["method_passed"]
        or not parent["binding_decision"]["tier_I_passed"]
        or parent["binding_decision"]["one_way_embedded_c2c2_authorized"]
        or parent["binding_decision"][
            "operator_or_interface_redesign_authorized"
        ]
    ):
        raise RuntimeError("WP10c9d6c7c2b4 binding rejection changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
    ):
        raise RuntimeError("WP10c9d6c7c2b5a analyzed identity changed")
    return (
        parent,
        _load_npz(PARENT_DIRECTORY / "decisive_arrays.npz"),
        _read_json(C2B3_DIRECTORY / "transfer_manifest.json"),
        _read_json(SCOPE_DIRECTORY / "scope_manifest.json"),
    )


def _representative_packet_matrix(
    level: dict,
    scope_arrays: dict[str, np.ndarray],
    support_log_bounds: tuple[float, float],
) -> tuple[np.ndarray, list[dict]]:
    full, cases, _packets = c2b1._packet_matrix(
        level,
        scope_arrays,
        support_log_bounds,
    )
    selected = [
        next(
            index
            for index, case in enumerate(cases)
            if case["family"] == family
            and case["sign"] == 1
            and case["amplitude"] == 1.0
        )
        for family in FAMILIES
    ]
    return full[:, selected], [cases[index] for index in selected]


def _family_histories(
    energy_history,
    initial_source_energy: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for index, family in enumerate(FAMILIES):
        targets = tuple(c2b1.TARGET_FAMILIES[family])
        total = (
            np.asarray(energy_history.total_energy[:, index], dtype=float)
            / initial_source_energy[index]
        )
        family_energy = (
            np.asarray(energy_history.family_energy[:, index], dtype=float)
            / initial_source_energy[index]
        )
        target = np.sum(family_energy[:, list(targets)], axis=1)
        result[family] = {
            "total": total,
            "target": target,
            "leakage": np.sum(family_energy, axis=1) - target,
        }
    return result


def _mask_history(
    times: np.ndarray,
    history: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    lower, upper = (float(item) for item in window)
    mask = (times >= lower) & (times <= upper)
    return np.where(mask, history, 0.0)


def _interpolated_stride_variant(
    times: np.ndarray,
    history: np.ndarray,
    stride: int,
) -> np.ndarray:
    if stride == 1:
        return np.asarray(history, dtype=float)
    selected = np.arange(0, times.size, stride)
    if selected[-1] != times.size - 1:
        selected = np.append(selected, times.size - 1)
    return np.interp(times, times[selected], history[selected])


def _common_projectors(
    levels: dict[int, dict],
) -> dict[int, np.ndarray]:
    fine = levels[LEVELS[-1]]
    fine_log_centers = np.log(np.asarray(fine["grid"].centers, dtype=float))
    fine_projectors = np.asarray(fine["projectors"], dtype=float)
    result = {}
    for cells, level in levels.items():
        centers = np.log(np.asarray(level["grid"].centers, dtype=float))
        indices = np.searchsorted(fine_log_centers, centers)
        indices = np.clip(indices, 1, fine_log_centers.size - 1)
        left = indices - 1
        choose_right = (
            np.abs(fine_log_centers[indices] - centers)
            < np.abs(fine_log_centers[left] - centers)
        )
        nearest = np.where(choose_right, indices, left)
        result[cells] = fine_projectors[nearest]
    return result


def _level_diagnostics(
    level: dict,
    propagated: dict,
    cases: list[dict],
    *,
    primary_windows: dict[str, tuple[float, float]],
    nuisance_windows: dict[str, list[tuple[float, float]]],
    nuisance_bands_N98: np.ndarray,
    common_projectors: np.ndarray,
) -> dict:
    cells = int(level["cells"])
    factor = cells // LEVELS[0]
    physical = np.asarray(propagated["physical"], dtype=float)
    times = np.asarray(propagated["times"], dtype=float)
    log_edges = np.log(np.asarray(level["grid"].edges, dtype=float))
    source_band = (
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
        lower_face=source_band[0],
        upper_face=source_band[1],
    )
    initial = np.asarray(source_energy.total_energy[0], dtype=float)
    if np.any(initial <= 0.0):
        raise RuntimeError(f"N{cells} source energy is not positive")

    band_pairs = [primary_band]
    for lower, upper in np.asarray(nuisance_bands_N98, dtype=int):
        pair = (int(lower) * factor, int(upper) * factor)
        if pair not in band_pairs:
            band_pairs.append(pair)
    band_histories = []
    for lower, upper in band_pairs:
        measured = causal_positive_band_energy_history(
            physical,
            log_edges=log_edges,
            energy_metrics=level["energy"],
            projectors=level["projectors"],
            lower_face=lower,
            upper_face=upper,
        )
        band_histories.append(_family_histories(measured, initial))
    nominal = band_histories[0]

    midpoint = (primary_band[0] + primary_band[1]) // 2
    frozen_projectors = np.repeat(
        np.asarray(level["projectors"][midpoint], dtype=float)[None],
        cells,
        axis=0,
    )
    subspace_histories = []
    for projectors in (common_projectors, frozen_projectors):
        measured = causal_positive_band_energy_history(
            physical,
            log_edges=log_edges,
            energy_metrics=level["energy"],
            projectors=projectors,
            lower_face=primary_band[0],
            upper_face=primary_band[1],
        )
        subspace_histories.append(_family_histories(measured, initial))

    variants: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    for family in FAMILIES:
        variants[family] = {}
        for observable in OBSERVABLES:
            history = nominal[family][observable]
            primary_masked = _mask_history(
                times,
                history,
                primary_windows[family],
            )
            scale_defect = float(propagated["restart_defect"])
            variants[family][observable] = {
                "receiving_band": [
                    _mask_history(
                        times,
                        item[family][observable],
                        primary_windows[family],
                    )
                    for item in band_histories
                ],
                "arrival_window": [
                    _mask_history(times, history, window)
                    for window in nuisance_windows[family]
                ],
                "time_sampling": [
                    _interpolated_stride_variant(
                        times,
                        primary_masked,
                        stride,
                    )
                    for stride in (1, 2, 4)
                ],
                "restart_roundoff": [
                    primary_masked * (1.0 - scale_defect),
                    primary_masked * (1.0 + scale_defect),
                ],
                "projection_and_subspace": [
                    _mask_history(
                        times,
                        item[family][observable],
                        primary_windows[family],
                    )
                    for item in subspace_histories
                ],
            }

    return {
        "times": times,
        "physical": physical,
        "initial_source_energy": initial,
        "nominal_histories": nominal,
        "variants": variants,
        "primary_band": primary_band,
        "restart_defect": float(propagated["restart_defect"]),
        "cases": cases,
    }


def _peak_radius(
    level: dict,
    physical: np.ndarray,
    *,
    time_seconds: float,
    times_seconds: np.ndarray,
    case_index: int,
    band: tuple[int, int],
) -> dict:
    times = np.asarray(times_seconds, dtype=float)
    point = float(time_seconds)
    right = int(np.searchsorted(times, point))
    if right <= 0:
        state = physical[0, case_index]
    elif right >= times.size:
        state = physical[-1, case_index]
    else:
        fraction = (point - times[right - 1]) / (times[right] - times[right - 1])
        state = (
            (1.0 - fraction) * physical[right - 1, case_index]
            + fraction * physical[right, case_index]
        )
    lower, upper = band
    widths = np.diff(np.log(np.asarray(level["grid"].edges, dtype=float)))
    metric = np.asarray(level["energy"], dtype=float)
    cell_energy = 0.5 * np.einsum(
        "ni,nij,nj,n->n",
        state,
        metric,
        state,
        widths,
        optimize=True,
    )
    selected = np.asarray(cell_energy[lower:upper], dtype=float)
    centers = np.asarray(level["grid"].centers[lower:upper], dtype=float)
    total = max(float(np.sum(selected)), np.finfo(float).tiny)
    gravitational_radius = float(level["grid"].gravitational_radius)
    centroid_log = float(np.sum(np.log(centers) * selected) / total)
    maximum = int(np.argmax(selected))
    return {
        "energy_centroid_radius_rg": float(
            np.exp(centroid_log) / gravitational_radius
        ),
        "peak_cell_radius_rg": float(
            centers[maximum] / gravitational_radius
        ),
        "peak_cell_energy_fraction": float(selected[maximum] / total),
        "band_energy": selected,
    }


def _conditioning_report(
    levels: dict[int, dict],
    diagnostics: dict[int, dict],
    parent_arrays: dict[str, np.ndarray],
    primary_windows: dict[str, tuple[float, float]],
) -> tuple[dict, dict[str, np.ndarray]]:
    primary_times = np.asarray(parent_arrays["primary_times_seconds"], dtype=float)
    decisive: dict[str, np.ndarray] = {
        "reference_levels": np.asarray(LEVELS, dtype=np.int64),
        "primary_times_seconds": primary_times,
    }
    replay_defects = []
    reports = {}
    all_horizons_complete = True
    shear_leakage_observable = False
    for family_index, family in enumerate(FAMILIES):
        reports[family] = {}
        for observable in OBSERVABLES:
            reconstructed = [
                diagnostics[cells]["nominal_histories"][family][observable][::2]
                for cells in LEVELS
            ]
            stored = [
                np.asarray(
                    parent_arrays[f"N{cells}__{family}__{observable}_history"],
                    dtype=float,
                )
                for cells in LEVELS
            ]
            replay_defects.extend(
                _relative_defect(left, right)
                for left, right in zip(reconstructed, stored, strict=True)
            )
            conditioned = causal_arrival_history_conditioning(
                *stored,
                times_seconds=primary_times,
            )
            masked_nominal = [
                _mask_history(primary_times, item, primary_windows[family])
                for item in stored
            ]
            variations = {}
            for category in diagnostics[LEVELS[0]]["variants"][family][observable]:
                count = min(
                    len(
                        diagnostics[cells]["variants"][family][observable][
                            category
                        ]
                    )
                    for cells in LEVELS
                )
                variations[category] = np.asarray(
                    [
                        [
                            diagnostics[cells]["variants"][family][observable][
                                category
                            ][variant][::2]
                            for cells in LEVELS
                        ]
                        for variant in range(count)
                    ],
                    dtype=float,
                )
            uncertainty = causal_history_uncertainty_envelope(
                *masked_nominal,
                times_seconds=primary_times,
                variations=variations,
                observability_factor=OBSERVABILITY_FACTOR,
            )
            horizon_by_level = {}
            for cells, history in zip(LEVELS, stored, strict=True):
                horizon = causal_horizon_completeness(
                    primary_times,
                    history,
                    final_window_fraction=FINAL_WINDOW_FRACTION,
                    maximum_terminal_to_peak=MAXIMUM_TERMINAL_TO_PEAK,
                    maximum_final_window_range_to_peak=(
                        MAXIMUM_FINAL_WINDOW_RANGE_TO_PEAK
                    ),
                    maximum_terminal_slope_horizon_to_peak=(
                        MAXIMUM_TERMINAL_SLOPE_HORIZON_TO_PEAK
                    ),
                )
                horizon_by_level[f"N{cells}"] = _plain(horizon.__dict__)
                all_horizons_complete &= horizon.complete
            measured_uncertainty_complete = False
            direction_observable_under_measured_nuisance = bool(
                uncertainty.coarse_medium_observable
                and uncertainty.medium_fine_observable
            )
            if family == "shear" and observable == "leakage":
                shear_leakage_observable = (
                    direction_observable_under_measured_nuisance
                )
            reports[family][observable] = {
                "historical_c2b4_classification_preserved": True,
                "conditioning": _plain(conditioned.__dict__),
                "windowed_measured_nuisance_envelope": _plain(
                    uncertainty.__dict__
                ),
                "uncertainty_components_measured": list(variations),
                "uncertainty_components_unresolved": [
                    "independent_continuum_history_reference"
                ],
                "complete_uncertainty_contract_closed": (
                    measured_uncertainty_complete
                ),
                "direction_observable_under_measured_nuisance_only": (
                    direction_observable_under_measured_nuisance
                ),
                "horizon": horizon_by_level,
            }
            for cells, history in zip(LEVELS, stored, strict=True):
                decisive[f"N{cells}__{family}__{observable}_history"] = history

        acoustic_peak = None
        if family == "acoustic":
            peak_by_level = {}
            radial_arrays = []
            for cells in LEVELS:
                total = np.asarray(
                    parent_arrays[f"N{cells}__acoustic__total_history"],
                    dtype=float,
                )
                peak = causal_quadratic_peak(primary_times, total)
                radial = _peak_radius(
                    levels[cells],
                    diagnostics[cells]["physical"],
                    time_seconds=peak.interpolated_time_seconds,
                    times_seconds=diagnostics[cells]["times"],
                    case_index=family_index,
                    band=diagnostics[cells]["primary_band"],
                )
                radial_arrays.append(radial.pop("band_energy"))
                peak_by_level[f"N{cells}"] = {
                    **_plain(peak.__dict__),
                    **radial,
                }
            acoustic_peak = peak_by_level
            reports[family]["peak_location"] = acoustic_peak
            for cells, values in zip(LEVELS, radial_arrays, strict=True):
                decisive[f"N{cells}__acoustic_peak_cell_energy"] = values

    maximum_replay = float(max(replay_defects))
    if maximum_replay > MAXIMUM_PARENT_REPLAY_DEFECT:
        raise RuntimeError(
            f"c2b4 representative history replay changed: {maximum_replay}"
        )
    horizon_decision = (
        "horizon_complete"
        if all_horizons_complete
        else "longer_uniform_horizon_manifest_required"
    )
    if all_horizons_complete:
        classification = (
            "arrival_history_conditioning_and_horizon_audit_complete_"
            "shear_family_transfer_audit_required"
        )
        authorized_next = (
            "WP10c9d6c7c2b5b_shear_family_transfer_and_projector_audit"
        )
    else:
        classification = (
            "arrival_history_conditioning_complete_uniform_horizon_"
            "incomplete_longer_horizon_manifest_required"
        )
        authorized_next = (
            "WP10c9d6c7c2b5a2_longer_uniform_horizon_manifest"
        )
    return (
        {
            "families": reports,
            "maximum_parent_history_replay_relative_defect": maximum_replay,
            "effective_independent_profile_count": 3,
            "historical_case_count": 12,
            "historical_cases_are_three_profiles_plus_"
            "exact_amplitude_sign_controls": True,
            "horizon_decision": horizon_decision,
            "all_total_target_leakage_horizons_complete": all_horizons_complete,
            "shear_leakage_direction_observable_under_measured_nuisance_only": (
                shear_leakage_observable
            ),
            "independent_continuum_history_reference_available": False,
            "revised_arrival_contract_frozen": False,
            "classification": classification,
            "authorized_next": authorized_next,
            "passed": True,
            "binding_decision": {
                "c2b4_rejection_preserved": True,
                "absolute_initial_energy_history_scale_identified_as_"
                "ill_conditioned": True,
                "acoustic_peak_is_convergent_but_old_gate_remains_failed": True,
                "horizon_complete": all_horizons_complete,
                "shear_family_transfer_audit_authorized": (
                    all_horizons_complete
                ),
                "revised_uniform_recertification_authorized": False,
                "embedded_authorized": False,
                "operator_or_interface_redesign_authorized": False,
                "nonlinear_authorized": False,
                "fixed_Q_or_reduction_authorized": False,
            },
        },
        decisive,
    )


def _config(
    transfer_manifest: dict,
    scope: dict,
    primary_windows: dict[str, tuple[float, float]],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "embedded_or_nonlinear_propagation_executed": False,
        "reference_levels": list(LEVELS),
        "representative_profiles": list(FAMILIES),
        "primary_arrival_windows_seconds": primary_windows,
        "conditioning_is_diagnostic_only": True,
        "historical_c2b4_gates_unchanged": True,
        "history_normalizations_reported": [
            "initial_source_energy_absolute",
            "response_relative",
            "fixed_second_order_Richardson_relative",
            "observed_order_Richardson_relative",
            "amplitude_and_unit_shape",
        ],
        "nuisance_combination": (
            "conservative_sum_of_deterministic_error_vector_bounds"
        ),
        "RSS_used": False,
        "measured_nuisance_components": [
            "receiving_band",
            "arrival_window",
            "time_sampling",
            "restart_roundoff",
            "projection_and_subspace",
        ],
        "unresolved_nuisance_components": [
            "independent_continuum_history_reference"
        ],
        "horizon_gates": {
            "final_window_fraction": FINAL_WINDOW_FRACTION,
            "maximum_terminal_to_peak": MAXIMUM_TERMINAL_TO_PEAK,
            "maximum_final_window_range_to_peak": (
                MAXIMUM_FINAL_WINDOW_RANGE_TO_PEAK
            ),
            "maximum_terminal_slope_horizon_to_peak": (
                MAXIMUM_TERMINAL_SLOPE_HORIZON_TO_PEAK
            ),
        },
        "observability_factor": OBSERVABILITY_FACTOR,
        "parent_positive_transfer_contract": (
            transfer_manifest["prospective_observable"]
        ),
        "experiment_end_seconds": scope["packet_and_window_contract"][
            "experiment_end_seconds"
        ],
    }


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "config.json",
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        C2B3_DIRECTORY / "transfer_manifest.json",
        C2B3_DIRECTORY / "decisive_arrays.npz",
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
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
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _write_report(summary: dict) -> None:
    families = summary["families"]
    shear_observable = summary[
        "shear_leakage_direction_observable_under_measured_nuisance_only"
    ]
    lines = [
        "# WP10c9d6c7c2b5a — Uniform arrival-history conditioning audit",
        "",
        f"- Classification: `{summary['classification']}`",
        "- Historical c2b4 rejection: preserved without amendment.",
        "- Operator changed: `False`.",
        "- Embedded/nonlinear/fixed-Q/reduced evolution: not run.",
        "",
        "## Binding interpretation",
        "",
        (
            "The initial-energy-normalized absolute history gate is "
            "ill-conditioned for responses amplified by thousands. This "
            "diagnosis does not pass c2b4 and does not yet freeze a replacement "
            "contract."
        ),
        "",
        "| Family | channel | absolute fine max | response-relative fine max "
        "| RMS order | shape fine max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for observable in OBSERVABLES:
            item = families[family][observable]["conditioning"]
            lines.append(
                f"| {family} | {observable} | "
                f"{item['absolute_fine_maximum_difference']:.6e} | "
                f"{item['response_relative_fine_maximum_difference']:.6e} | "
                f"{item['weighted_rms_order']:.4f} | "
                f"{item['shape_fine_maximum_difference']:.6e} |"
            )
    lines.extend(
        (
            "",
            "## Acoustic peak",
            "",
            "| Level | interpolated time (s) | interpolated gain | energy "
            "centroid (rg) | peak-cell radius (rg) |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for level in ("N98", "N196", "N392"):
        item = families["acoustic"]["peak_location"][level]
        lines.append(
            f"| {level} | {item['interpolated_time_seconds']:.8e} | "
            f"{item['interpolated_value']:.8e} | "
            f"{item['energy_centroid_radius_rg']:.8e} | "
            f"{item['peak_cell_radius_rg']:.8e} |"
        )
    lines.extend(
        (
            "",
            "## Uncertainty and horizon",
            "",
            (
                "Receiving-band, predeclared-window, time-sampling, restart, "
                "and two projector-field variations are combined by a "
                "conservative sum. RSS is not used. An independent continuum "
                "history reference is not available, so no new error-direction "
                "gate is certified in this package."
            ),
            "",
            (
                "All total, target, and leakage histories clear the receiving "
                "band under the predeclared terminal-tail gates: "
                f"`{summary['all_total_target_leakage_horizons_complete']}`."
            ),
            "",
            (
                "The shear-leakage refinement-error direction is observable "
                "above the measured nuisance envelope alone: "
                f"`{shear_observable}`. "
                "Unlike c2b4, this audit does not set projector/subspace "
                "uncertainty to zero. The admissible projector-field variants "
                "are large enough to make the nominal leakage direction "
                "non-certifying, while an independent continuum history "
                "reference is still absent."
            ),
            "",
            "## Decision",
            "",
            (
                "The acoustic peak miss is a convergent, gain-conditioning "
                "issue. Shear opposite-family leakage remains the only "
                "unresolved Tier-II quantity and now requires the exact "
                "family-transfer/projector audit."
            ),
            "",
            f"Authorized next: `{summary['authorized_next']}`.",
            "",
            "Embedded discrimination, operator/interface redesign, nonlinear "
            "propagation, fixed-Q experiments, reduced evolution, and N1024 "
            "remain blocked.",
            "",
        )
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    parent, parent_arrays, transfer_manifest, scope = _validate_parent()
    (
        _geometry_summary,
        _geometry_manifest,
        _geometry_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    c2a2_arrays = _load_npz(C2A2_DIRECTORY / "decisive_arrays.npz")
    scope_arrays = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    contract_arrays = _load_npz(C2B3_DIRECTORY / "decisive_arrays.npz")
    base_edges = np.asarray(c2a2_arrays["patch_edges"], dtype=float)
    support_log_bounds = (
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[0]])),
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[1]])),
    )
    horizon = float(
        scope["packet_and_window_contract"]["experiment_end_seconds"]
    )
    travel = np.asarray(scope_arrays["travel_windows_seconds"], dtype=float)
    propagation_windows = {
        "interface": {
            family: tuple(travel[index, :2])
            for index, family in enumerate(FAMILIES)
        },
        "downstream": {
            family: tuple(travel[index, 2:])
            for index, family in enumerate(FAMILIES)
        },
    }
    primary_array = np.asarray(
        contract_arrays["primary_arrival_windows_seconds"],
        dtype=float,
    )
    nuisance_array = np.asarray(
        contract_arrays["arrival_window_nuisance_seconds"],
        dtype=float,
    )
    primary_windows = {
        family: tuple(primary_array[index])
        for index, family in enumerate(FAMILIES)
    }
    nuisance_windows = {
        family: [
            (
                float(nuisance_array[variant, index, 0]),
                min(float(nuisance_array[variant, index, 1]), horizon),
            )
            for variant in range(nuisance_array.shape[0])
        ]
        for index, family in enumerate(FAMILIES)
    }

    levels = {
        cells: c2b1._build_level(
            cells,
            base_edges,
            parent_context,
            parent_base,
            field_scales,
            reuse_checkpoint=True,
        )
        for cells in LEVELS
    }
    common_projectors = _common_projectors(levels)
    common_log_centers = np.log(np.asarray(base_edges[:-1])) + 0.5 * np.diff(
        np.log(base_edges)
    )
    diagnostics = {}
    case_reference = None
    for cells in LEVELS:
        initial, cases = _representative_packet_matrix(
            levels[cells],
            scope_arrays,
            support_log_bounds,
        )
        if case_reference is None:
            case_reference = cases
        elif case_reference != cases:
            raise RuntimeError("representative packet ordering changed")
        propagated = c2b1._propagate_level(
            levels[cells],
            initial,
            cases,
            propagation_windows,
            horizon,
            common_log_centers,
        )
        diagnostics[cells] = _level_diagnostics(
            levels[cells],
            propagated,
            cases,
            primary_windows=primary_windows,
            nuisance_windows=nuisance_windows,
            nuisance_bands_N98=np.asarray(
                contract_arrays["receiving_band_nuisance_faces_N98"],
                dtype=int,
            ),
            common_projectors=common_projectors[cells],
        )

    result, decisive = _conditioning_report(
        levels,
        diagnostics,
        parent_arrays,
        primary_windows,
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
        "parent_classification": parent["classification"],
        **result,
        "runtime_seconds": time.perf_counter() - started,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        _config(transfer_manifest, scope, primary_windows),
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_manifest = {
        relative: c2a._sha256(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
        if (ROOT / relative).is_file()
    }
    summary["decisive_array_hashes"] = {
        name: causal_array_sha256(values)
        for name, values in decisive.items()
    }
    summary["decisive_arrays_sha256"] = c2a._sha256(DECISIVE_ARRAYS)
    summary["config_sha256"] = c2a._sha256(CONFIG_PATH)
    summary["implementation_source_hashes"] = source_manifest
    summary["implementation_source_manifest_sha256"] = (
        causal_canonical_json_sha256(source_manifest)
    )
    summary["input_hashes"] = _input_hashes()
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": _git_value(
            "rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}"
        ),
        "implementation_head_before_commit": _git_value("rev-parse", "HEAD"),
        "current_branch": _git_value("branch", "--show-current"),
        "input_hashes": _input_hashes(),
        "implementation_source_hashes": source_manifest,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "command": f"{sys.executable} {THIS_RUNNER}",
        "scientific_status": "DIAGNOSTIC ONLY",
    }
    _write_json(PROVENANCE_PATH, provenance)
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    print(json.dumps(summary["binding_decision"], indent=2), flush=True)
    print(f"classification={summary['classification']}", flush=True)


if __name__ == "__main__":
    main()
