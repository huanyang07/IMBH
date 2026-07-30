#!/usr/bin/env python3
"""Audit the interpretation of the rejected one-way transmission metric.

WP10c9d6c7c2b1 passed every method and Tier-I contract but rejected the
continuum-symmetrizer shear transmission ratio.  This operator-neutral audit
constructs the exact semidiscrete control-volume energy identity of the same
N98/N196/N392 tangents.  It resolves the energy action by descriptor/storage
and stationary-residual block, derives face power from the actual shared-face
linear operator, and compares that numerical transfer with the preceding
continuum symmetrizer flux.  It performs no embedded or nonlinear evolution.
"""

from __future__ import annotations

import argparse
import csv
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

import run_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1 as c2b1  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_one_way_scattering import (  # noqa: E402
    causal_integrate_frozen_window,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_semidiscrete_energy import (  # noqa: E402
    causal_scaled_control_energy_metric,
    causal_semidiscrete_control_energy_history,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b2"
ANALYZED_BASE_COMMIT = "51a32ff686cea3b91d7f5056c464004399318172"
LEVELS = c2b1.LEVELS
PRIMARY_FAMILIES = c2b1.PRIMARY_FAMILIES
FIELDS = c2b1.FIELDS
MAXIMUM_ALGEBRAIC_DEFECT = 1.0e-10
MAXIMUM_ENERGY_INTEGRATION_DEFECT = 5.0e-5

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_one_way_transmission_interpretation_"
    "wp10c9d6c7c2b2.py"
)
THIS_HELPER = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_semidiscrete_energy.py"
)
THIS_HELPER_TEST = "tests/test_causal_inner_semidiscrete_energy.py"
THIS_CANONICAL_TEST = (
    "tests/"
    "test_causal_inner_one_way_transmission_interpretation_"
    "wp10c9d6c7c2b2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_ONE_WAY_TRANSMISSION_INTERPRETATION_"
    "WP10C9D6C7C2B2_RESULTS_2026-07-30.md"
)

C2B1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1"
)
SCOPE_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_scope_wp10c9d6c7c2a3"
)
C2A2_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_energy_wp10c9d6c7c2a2"
)
C7A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2"
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


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float).ravel()
    second = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def _validate_parent() -> tuple[dict, dict, dict[str, np.ndarray]]:
    summary = _read_json(C2B1_DIRECTORY / "summary.json")
    if (
        summary["classification"]
        != "one_way_uniform_scattering_validation_failed_"
        "embedded_discrimination_blocked"
        or summary["passed"]
        or not summary["binding_decision"]["method_passed"]
        or not summary["binding_decision"]["tier_I_passed"]
        or summary["binding_decision"]["tier_II_passed"]
        or summary["authorized_next"]
        != "WP10c9d6c7c2b2_one_way_uniform_transmission_"
        "interpretation_audit"
    ):
        raise RuntimeError("WP10c9d6c7c2b1 binding status changed")
    if _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT:
        raise RuntimeError("analyzed base commit changed")
    scope = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")
    arrays = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    return summary, scope, arrays


def _input_hashes() -> dict[str, str]:
    paths = (
        C2B1_DIRECTORY / "config.json",
        C2B1_DIRECTORY / "summary.json",
        C2B1_DIRECTORY / "decisive_arrays.npz",
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "decisive_arrays.npz",
        C7A_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _energy_operator_checkpoint(level: dict, cells: int) -> dict:
    path = CHECKPOINT_DIRECTORY / f"N{cells}.npz"
    report_path = CHECKPOINT_DIRECTORY / f"N{cells}.json"
    if path.is_file() and report_path.is_file():
        stored = _load_npz(path)
        report = _read_json(report_path)
        if (
            report["analyzed_base_commit"] != ANALYZED_BASE_COMMIT
            or report["maximum_generator_replay_defect"]
            > MAXIMUM_ALGEBRAIC_DEFECT
        ):
            raise RuntimeError(f"invalid c2b2 N{cells} checkpoint")
        return {
            "descriptor": stored["descriptor"],
            "mapped_storage_rate": stored["mapped_storage_rate"],
            "height_storage_rate": stored["height_storage_rate"],
            "row_scales": stored["row_scales"],
            "face_maps": stored["face_maps"],
            "blocks": {
                name.removeprefix("block__"): value
                for name, value in stored.items()
                if name.startswith("block__")
            },
            "report": report,
        }

    charts = np.asarray(level["extension"].primitive_charts, dtype=float)
    columns = np.asarray(level["columns"], dtype=float)
    rows = c2b1._conservation_row_scales(level["context"], charts)
    print(
        f"{WORK_PACKAGE}: reconstruct exact N{cells} descriptor/block maps",
        flush=True,
    )
    started = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        level["context"],
        charts,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        path_quadrature_order=c2b1.PATH_QUADRATURE_ORDER,
        centered_storage_action_scaled_step=c2b1.STORAGE_ACTION_STEP,
    )
    replay = _relative_defect(
        tangent.scaled_generator_per_s,
        level["generator"],
    )
    report = {
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "maximum_generator_replay_defect": replay,
        "maximum_generator_factorization_defect": float(
            tangent.maximum_generator_factorization_defect
        ),
        "maximum_stationary_block_ledger_defect": float(
            tangent.spatial_tangent.maximum_block_ledger_relative_defect
        ),
        "runtime_seconds": time.perf_counter() - started,
        "passed": bool(
            replay <= MAXIMUM_ALGEBRAIC_DEFECT
            and tangent.maximum_generator_factorization_defect
            <= MAXIMUM_ALGEBRAIC_DEFECT
            and tangent.spatial_tangent.maximum_block_ledger_relative_defect
            <= MAXIMUM_ALGEBRAIC_DEFECT
        ),
    }
    if not report["passed"]:
        raise RuntimeError(f"N{cells} semidiscrete reconstruction failed")
    arrays = {
        "descriptor": np.asarray(tangent.descriptor_scaled_matrix),
        "mapped_storage_rate": np.asarray(
            tangent.mapped_storage_rate_derivative_scaled_matrix
        ),
        "height_storage_rate": np.asarray(
            tangent.responsive_height_storage_rate_derivative_scaled_matrix
        ),
        "row_scales": rows,
        "face_maps": np.asarray(
            tangent.spatial_tangent.shared_face_flux_scaled_jacobians
        ),
        **{
            f"block__{name}": np.asarray(matrix)
            for name, matrix in (
                tangent.spatial_tangent.block_scaled_jacobians.items()
            )
        },
    }
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "descriptor": arrays["descriptor"],
        "mapped_storage_rate": arrays["mapped_storage_rate"],
        "height_storage_rate": arrays["height_storage_rate"],
        "row_scales": arrays["row_scales"],
        "face_maps": arrays["face_maps"],
        "blocks": {
            name.removeprefix("block__"): value
            for name, value in arrays.items()
            if name.startswith("block__")
        },
        "report": report,
    }


def _build_levels(
    base_edges: np.ndarray,
    parent_context,
    parent_base: np.ndarray,
    field_scales: np.ndarray,
) -> dict[int, dict]:
    result = {}
    for cells in LEVELS:
        level = c2b1._build_level(
            cells,
            base_edges,
            parent_context,
            parent_base,
            field_scales,
            reuse_checkpoint=True,
        )
        level["energy_operator"] = _energy_operator_checkpoint(level, cells)
        result[cells] = level
    return result


def _representative_cases(cases: list[dict]) -> tuple[list[int], list[str]]:
    indices = []
    names = []
    for family in PRIMARY_FAMILIES:
        selected = next(
            index
            for index, case in enumerate(cases)
            if case["family"] == family
            and case["sign"] == 1
            and case["amplitude"] == 1.0
        )
        indices.append(selected)
        names.append(family)
    return indices, names


def _integrated_face_energy(
    times: np.ndarray,
    values: np.ndarray,
    window: tuple[float, float],
) -> float:
    return float(
        causal_integrate_frozen_window(
            times,
            np.asarray(values, dtype=float)[:, None],
            window,
        )[0]
    )


def _history_comparison(
    numerical: np.ndarray,
    continuum: np.ndarray,
) -> dict:
    discrete = np.asarray(numerical, dtype=float)
    reference = np.asarray(continuum, dtype=float)
    denominator = max(
        float(np.dot(reference, reference)),
        np.finfo(float).tiny,
    )
    fitted_scale = float(np.dot(discrete, reference) / denominator)
    fitted = fitted_scale * reference
    return {
        "history_cosine": _cosine(discrete, reference),
        "least_squares_scale": fitted_scale,
        "scaled_shape_relative_defect": _relative_defect(discrete, fitted),
        "direct_relative_difference": _relative_defect(discrete, reference),
    }


def _audit_level(
    level: dict,
    propagated: dict,
    cases: list[dict],
    windows: dict[str, dict[str, tuple[float, float]]],
) -> tuple[dict, dict[str, np.ndarray]]:
    cells = int(level["cells"])
    indices, families = _representative_cases(cases)
    times = np.asarray(propagated["times"], dtype=float)
    scaled = np.transpose(
        propagated["scaled"][:, :, indices],
        (0, 2, 1),
    )
    lower_face = c2b1._face_index(c2a3.DOWNSTREAM_MEASUREMENT_FACE, cells)
    upper_face = c2b1._face_index(c2a3.PATCH_INTERFACE_FACE, cells)
    metric = causal_scaled_control_energy_metric(
        level["energy"],
        np.log(np.asarray(level["grid"].edges, dtype=float)),
        level["columns"],
        lower_face,
        upper_face,
    )
    operator = level["energy_operator"]
    audit = causal_semidiscrete_control_energy_history(
        scaled,
        scaled_energy_metric=metric,
        descriptor_scaled_matrix=operator["descriptor"],
        scaled_generator_per_s=level["generator"],
        stationary_scaled_blocks=operator["blocks"],
        mapped_storage_rate_scaled_matrix=operator["mapped_storage_rate"],
        responsive_height_storage_rate_scaled_matrix=(
            operator["height_storage_rate"]
        ),
        conservation_row_scales=operator["row_scales"],
        shared_face_flux_scaled_jacobians=operator["face_maps"],
    )
    stored_change = audit.stored_energy[-1] - audit.stored_energy[0]
    integrated_total_power = np.trapezoid(
        audit.direct_generator_power,
        times,
        axis=0,
    )
    integration_scale = np.maximum.reduce(
        (
            np.abs(stored_change),
            np.abs(integrated_total_power),
            np.finfo(float).tiny * np.ones_like(stored_change),
        )
    )
    cancellation_relative_integration_defect = np.abs(
        stored_change - integrated_total_power
    ) / integration_scale
    integrated_absolute_power = np.trapezoid(
        np.abs(audit.direct_generator_power),
        times,
        axis=0,
    )
    action_scale = np.maximum.reduce(
        (
            integrated_absolute_power,
            np.abs(stored_change),
            np.abs(integrated_total_power),
            np.finfo(float).tiny * np.ones_like(stored_change),
        )
    )
    integration_defect = np.abs(
        stored_change - integrated_total_power
    ) / action_scale
    block_integrals = {
        name: np.trapezoid(values, times, axis=0)
        for name, values in audit.block_powers.items()
    }
    face_integrals = np.trapezoid(
        audit.conservative_face_powers,
        times,
        axis=0,
    )

    result = {
        "method": {
            **operator["report"],
            "maximum_generator_power_defect": (
                audit.maximum_generator_power_defect
            ),
            "maximum_block_power_defect": audit.maximum_block_power_defect,
            "maximum_face_power_defect": audit.maximum_face_power_defect,
            "maximum_time_integrated_energy_defect": float(
                np.max(integration_defect)
            ),
        },
        "families": {},
    }
    decisive = {
        "stored_energy": audit.stored_energy,
        "direct_generator_power": audit.direct_generator_power,
        "conservative_face_power": audit.conservative_face_powers,
        "integrated_face_power": face_integrals,
    }
    for name, values in block_integrals.items():
        decisive[f"integrated_block__{name}"] = values

    energy_history = propagated["energy_history"]
    for local_index, (case_index, family) in enumerate(
        zip(indices, families, strict=True)
    ):
        numerical_incident_history = audit.conservative_face_powers[
            :, local_index, upper_face
        ]
        numerical_transmitted_history = -audit.conservative_face_powers[
            :, local_index, lower_face
        ]
        continuum_incident_history = (
            energy_history.incident_total_flux[:, case_index]
        )
        continuum_transmitted_history = (
            energy_history.transmitted_total_flux[:, case_index]
        )
        numerical_incident = _integrated_face_energy(
            times,
            numerical_incident_history,
            windows["interface"][family],
        )
        numerical_transmitted = _integrated_face_energy(
            times,
            numerical_transmitted_history,
            windows["downstream"][family],
        )
        continuum = propagated["ledgers"][case_index]
        transmission_variants = []
        for stride in (1, 2, 4):
            sampled_times = times[::stride]
            for factor in c2a3.WINDOW_PADDING_NUISANCE_FACTORS:
                varied_incident = _integrated_face_energy(
                    sampled_times,
                    numerical_incident_history[::stride],
                    c2b1._window_with_padding_factor(
                        windows["interface"][family],
                        factor,
                        c2a3.WINDOW_PADDING_FRACTION,
                        times[-1],
                    ),
                )
                varied_transmitted = _integrated_face_energy(
                    sampled_times,
                    numerical_transmitted_history[::stride],
                    c2b1._window_with_padding_factor(
                        windows["downstream"][family],
                        factor,
                        c2a3.WINDOW_PADDING_FRACTION,
                        times[-1],
                    ),
                )
                transmission_variants.append(
                    varied_transmitted
                    / max(abs(varied_incident), np.finfo(float).tiny)
                )
        nominal_transmission = (
            numerical_transmitted
            / max(abs(numerical_incident), np.finfo(float).tiny)
        )
        stability_scale = max(
            abs(nominal_transmission),
            np.finfo(float).tiny,
        )
        stability = float(
            np.max(
                np.abs(
                    np.asarray(transmission_variants)
                    - nominal_transmission
                )
            )
            / stability_scale
        )
        total_face_absolute = float(
            np.sum(np.abs(face_integrals[local_index]))
        )
        selected_face_absolute = float(
            abs(face_integrals[local_index, upper_face])
            + abs(face_integrals[local_index, lower_face])
        )
        result["families"][family] = {
            "numerical_incident_energy": numerical_incident,
            "numerical_transmitted_energy": numerical_transmitted,
            "numerical_transmission": nominal_transmission,
            "continuum_incident_energy": continuum["incident_energy"],
            "continuum_transmitted_energy": continuum["transmitted_energy"],
            "continuum_transmission": continuum["transmission"],
            "incident_flux_comparison": _history_comparison(
                numerical_incident_history,
                continuum_incident_history,
            ),
            "transmitted_flux_comparison": _history_comparison(
                numerical_transmitted_history,
                continuum_transmitted_history,
            ),
            "stored_energy_change": float(stored_change[local_index]),
            "integrated_total_power": float(
                integrated_total_power[local_index]
            ),
            "time_integrated_energy_defect": float(
                integration_defect[local_index]
            ),
            "cancellation_relative_time_integration_defect": float(
                cancellation_relative_integration_defect[local_index]
            ),
            "integrated_absolute_generator_power": float(
                integrated_absolute_power[local_index]
            ),
            "integrated_block_powers": {
                block: float(values[local_index])
                for block, values in block_integrals.items()
            },
            "selected_boundary_face_absolute_fraction": (
                selected_face_absolute
                / max(total_face_absolute, np.finfo(float).tiny)
            ),
            "window_time_stability_defect": stability,
        }
        decisive[f"{family}__numerical_incident_power"] = (
            numerical_incident_history
        )
        decisive[f"{family}__numerical_transmitted_power"] = (
            numerical_transmitted_history
        )
        decisive[f"{family}__continuum_incident_power"] = (
            continuum_incident_history
        )
        decisive[f"{family}__continuum_transmitted_power"] = (
            continuum_transmitted_history
        )
    return result, decisive


def _evaluate(
    levels: dict[int, dict],
    propagated: dict[int, dict],
    cases: list[dict],
    windows: dict[str, dict[str, tuple[float, float]]],
    parent_summary: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    per_level = {}
    decisive = {}
    for cells in LEVELS:
        report, arrays = _audit_level(
            levels[cells],
            propagated[cells],
            cases,
            windows,
        )
        per_level[f"N{cells}"] = report
        for name, value in arrays.items():
            decisive[f"N{cells}__{name}"] = value

    transmission = {}
    for family in PRIMARY_FAMILIES:
        numerical = np.asarray(
            [
                per_level[f"N{cells}"]["families"][family][
                    "numerical_transmission"
                ]
                for cells in LEVELS
            ]
        )
        continuum = np.asarray(
            [
                per_level[f"N{cells}"]["families"][family][
                    "continuum_transmission"
                ]
                for cells in LEVELS
            ]
        )
        maximum_algebraic = max(
            per_level[f"N{cells}"]["method"][
                "maximum_block_power_defect"
            ]
            for cells in LEVELS
        )
        maximum_stability = max(
            per_level[f"N{cells}"]["families"][family][
                "window_time_stability_defect"
            ]
            for cells in LEVELS
        )
        uncertainty = maximum_algebraic + maximum_stability
        numerical_convergence = c2b1._scalar_convergence(
            numerical,
            uncertainty,
        )
        transmission[family] = {
            "numerical": numerical_convergence,
            "continuum": parent_summary["tier_II"][family]["transmission"],
            "numerical_to_continuum_values": (
                numerical
                / np.maximum(np.abs(continuum), np.finfo(float).tiny)
            ).tolist(),
            "maximum_window_time_stability_defect": maximum_stability,
            "maximum_selected_boundary_face_absolute_fraction": max(
                per_level[f"N{cells}"]["families"][family][
                    "selected_boundary_face_absolute_fraction"
                ]
                for cells in LEVELS
            ),
            "minimum_selected_boundary_face_absolute_fraction": min(
                per_level[f"N{cells}"]["families"][family][
                    "selected_boundary_face_absolute_fraction"
                ]
                for cells in LEVELS
            ),
            "numerical_incident_sign_stable_and_positive": bool(
                all(
                    per_level[f"N{cells}"]["families"][family][
                        "numerical_incident_energy"
                    ]
                    > 0.0
                    for cells in LEVELS
                )
            ),
        }
        decisive[f"{family}__numerical_transmission"] = numerical
        decisive[f"{family}__continuum_transmission"] = continuum

    method_passed = bool(
        max(
            max(
                per_level[f"N{cells}"]["method"][name]
                for cells in LEVELS
            )
            for name in (
                "maximum_generator_replay_defect",
                "maximum_generator_factorization_defect",
                "maximum_stationary_block_ledger_defect",
                "maximum_generator_power_defect",
                "maximum_block_power_defect",
                "maximum_face_power_defect",
            )
        )
        <= MAXIMUM_ALGEBRAIC_DEFECT
        and max(
            per_level[f"N{cells}"]["method"][
                "maximum_time_integrated_energy_defect"
            ]
            for cells in LEVELS
        )
        <= MAXIMUM_ENERGY_INTEGRATION_DEFECT
    )
    numerical_passed = all(
        item["numerical"]["passed"] for item in transmission.values()
    )
    all_incident_signs_stable = all(
        item["numerical_incident_sign_stable_and_positive"]
        for item in transmission.values()
    )
    shear_continuum_failed = not parent_summary["tier_II"]["shear"][
        "passed"
    ]
    shear_numerical_passed = transmission["shear"]["numerical"]["passed"]
    if not method_passed:
        classification = (
            "semidiscrete_energy_interpretation_audit_failed_"
            "no_downstream_work_authorized"
        )
        authorized_next = (
            "repair_WP10c9d6c7c2b2_energy_accounting_before_interpretation"
        )
    elif numerical_passed and all_incident_signs_stable:
        classification = (
            "continuum_symmetrizer_face_observable_mismatch_identified_"
            "uniform_redefinition_required"
        )
        authorized_next = (
            "WP10c9d6c7c2b3_freeze_semidiscrete_transmission_contract"
        )
    else:
        classification = (
            "exact_semidiscrete_energy_identity_certified_"
            "local_face_transmission_not_certifiable"
        )
        authorized_next = (
            "WP10c9d6c7c2b3_definitions_only_semidiscrete_energy_"
            "transfer_contract"
        )
    return {
        "method": {
            "passed": method_passed,
            "maximum_algebraic_defect_gate": MAXIMUM_ALGEBRAIC_DEFECT,
            "maximum_time_integration_defect_gate": (
                MAXIMUM_ENERGY_INTEGRATION_DEFECT
            ),
        },
        "per_level": per_level,
        "transmission_comparison": transmission,
        "binding_decision": {
            "c2b1_rejection_preserved": True,
            "semidiscrete_energy_identity_passed": method_passed,
            "all_numerical_transmission_channels_passed": numerical_passed,
            "all_numerical_incident_signs_stable_and_positive": (
                all_incident_signs_stable
            ),
            "continuum_shear_transmission_failed": shear_continuum_failed,
            "semidiscrete_shear_transmission_passed": (
                shear_numerical_passed
            ),
            "embedded_c2c1_authorized": False,
            "operator_or_interface_redesign_authorized": False,
            "genuine_uniform_transport_error_selected": False,
            "local_face_transmission_contract_certified": bool(
                method_passed
                and numerical_passed
                and all_incident_signs_stable
            ),
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": method_passed,
    }, decisive


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_scope": (
            "reuse frozen c2b1 uniform tangents, packets, windows, and "
            "histories; no embedded or nonlinear work"
        ),
        "reference_levels": list(LEVELS),
        "control_volume_faces_on_N98": [
            c2a3.DOWNSTREAM_MEASUREMENT_FACE,
            c2a3.PATCH_INTERFACE_FACE,
        ],
        "energy_definition": (
            "exact scaled quadratic control metric with descriptor-dual "
            "residual and actual shared-face operator decomposition"
        ),
        "maximum_algebraic_defect": MAXIMUM_ALGEBRAIC_DEFECT,
        "maximum_time_integrated_energy_defect": (
            MAXIMUM_ENERGY_INTEGRATION_DEFECT
        ),
        "transmission_gates": _read_json(
            SCOPE_DIRECTORY / "scope_manifest.json"
        )["uniform_c2b1_contract"],
        "c2b1_values_and_thresholds_preserved": True,
        "root_sum_square_used": False,
        "slow_impact_threshold_used": False,
    }


def _write_report(summary: dict) -> None:
    lines = [
        "# WP10c9d6c7c2b2 — One-way transmission interpretation audit",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Passed: `{summary['passed']}`",
        "- Production/operator change: `False`",
        "- Embedded, nonlinear, fixed-Q, and reduced evolution were not run.",
        "",
        "## Exact semidiscrete identity",
        "",
        "The audit evaluates `d(1/2 z^T W z)/dt = z^T W G z` directly "
        "from the c2b1 descriptor and generator. The descriptor dual "
        "`D^{-T}Wz` resolves every stationary/storage block and every "
        "actual shared conservative face.",
        "",
        "| Family | numerical T(N98) | T(N196) | T(N392) | order | fine diff. | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for family, item in summary["transmission_comparison"].items():
        values = item["numerical"]["values"]
        lines.append(
            f"| {family} | {values[0]:.8g} | {values[1]:.8g} | "
            f"{values[2]:.8g} | "
            f"{item['numerical']['observed_order']:.4f} | "
            f"{item['numerical']['maximum_fine_normalized_difference']:.5f} | "
            f"{item['numerical']['passed']} |"
        )
    lines.extend(
        (
            "",
            "## Binding interpretation",
            "",
            "The c2b1 rejection remains unchanged. This package distinguishes "
            "the continuum symmetrizer face observable from the exact "
            "semidiscrete descriptor/face energy transfer. It does not "
            "authorize embedded propagation or an operator redesign.",
            "",
            "## Next step",
            "",
            f"`{summary['authorized_next']}`",
            "",
        )
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


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
    CANONICAL_SUMMARY.write_text(
        json.dumps(canonical_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-energy-checkpoints",
        action="store_true",
        help="retained for explicit provenance; checkpoints are reused when valid",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    parent_summary, _scope, scope_arrays = _validate_parent()
    (
        _geometry_summary,
        _geometry_manifest,
        _geometry_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    c2a2_arrays = _load_npz(C2A2_DIRECTORY / "decisive_arrays.npz")
    base_edges = np.asarray(c2a2_arrays["patch_edges"], dtype=float)
    support_log_bounds = (
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[0]])),
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[1]])),
    )
    travel = np.asarray(scope_arrays["travel_windows_seconds"], dtype=float)
    windows = {
        "interface": {
            family: tuple(travel[index, :2])
            for index, family in enumerate(PRIMARY_FAMILIES)
        },
        "downstream": {
            family: tuple(travel[index, 2:])
            for index, family in enumerate(PRIMARY_FAMILIES)
        },
    }
    horizon = float(
        _read_json(SCOPE_DIRECTORY / "scope_manifest.json")[
            "packet_and_window_contract"
        ]["experiment_end_seconds"]
    )
    levels = _build_levels(
        base_edges,
        parent_context,
        parent_base,
        field_scales,
    )
    initials = {}
    cases = None
    for cells, level in levels.items():
        initial, current_cases, _packets = c2b1._packet_matrix(
            level,
            scope_arrays,
            support_log_bounds,
        )
        if cases is None:
            cases = current_cases
        elif cases != current_cases:
            raise RuntimeError("c2b1 packet ordering changed")
        initials[cells] = initial
    assert cases is not None
    common_log_centers = np.log(np.asarray(base_edges[:-1])) + 0.5 * np.diff(
        np.log(base_edges)
    )
    propagated = {
        cells: c2b1._propagate_level(
            level,
            initials[cells],
            cases,
            windows,
            horizon,
            common_log_centers,
        )
        for cells, level in levels.items()
    }
    result, decisive = _evaluate(
        levels,
        propagated,
        cases,
        windows,
        parent_summary,
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
        **result,
        "runtime_seconds": time.perf_counter() - started,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        DECISIVE_ARRAYS,
        reference_levels=np.asarray(LEVELS, dtype=np.int64),
        times_seconds=propagated[LEVELS[0]]["times"],
        **decisive,
    )
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
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        "implementation_head_before_commit": _git_value("rev-parse", "HEAD"),
        "current_branch": _git_value("branch", "--show-current"),
        "input_hashes": _input_hashes(),
        "implementation_source_hashes": source_manifest,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "command": (
            f"{sys.executable} {THIS_RUNNER}"
            + (
                " --reuse-energy-checkpoints"
                if arguments.reuse_energy_checkpoints
                else ""
            )
        ),
        "scientific_status": (
            "SUPPORTED BUT NOT FULLY CERTIFIED"
            if summary["passed"]
            else "DIAGNOSTIC ONLY"
        ),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    print(json.dumps(summary["binding_decision"], indent=2), flush=True)
    print(f"classification={summary['classification']}", flush=True)


if __name__ == "__main__":
    main()
