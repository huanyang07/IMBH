#!/usr/bin/env python3
"""Audit shear family transfer and projector dependence on uniform grids.

WP10c9d6c7c2b5b preserves every c2b4/c2b5a classification.  It changes no
operator and runs no embedded, nonlinear, fixed-Q, or reduced evolution.
The exact frozen semidiscrete generator is decomposed by residual block,
source family, receiver family, and radius.  Acoustic and mixed packets are
retained as controls.  A polynomial projector construction and a 769/513
node continuum-action reference provide independent algorithmic checks.
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
from scipy.interpolate import make_interp_spline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3  # noqa: E402
import run_causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2 as c2b2  # noqa: E402
import run_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1 as c2b1  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402
import run_causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a as b5a  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    CONTINUUM_DAE_BLOCK_NAMES,
    build_causal_five_field_continuum_background,
    linearize_causal_five_field_continuum_reference,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_energy_transfer import (  # noqa: E402
    causal_positive_band_energy_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_family_energy_transfer import (  # noqa: E402
    causal_physical_family_transfer_ledger,
    causal_polynomial_spectral_projectors,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_scattering_energy import (  # noqa: E402
    causal_c4_manufactured_primitive_state,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b5b"
ANALYZED_BASE_COMMIT = "dc27efe1b414143cec67a050f8dc5c9ccff69ee4"
ANALYZED_BASE_PARENT = "dbfd8bdaf859fa23f530c5c4f00f78fa407137d3"
LEVELS = c2b1.LEVELS
FAMILIES = c2b1.PRIMARY_FAMILIES
PRIMARY_CONTINUUM_NODES = 769
SECONDARY_CONTINUUM_NODES = 513
CONTINUUM_COMPARISON_SAMPLES = 257
PROJECTION_ORDER = 24

MAXIMUM_PROJECTOR_ALGEBRA_DEFECT = 2.0e-9
MAXIMUM_EQUIVALENT_PROJECTOR_DEFECT = 2.0e-8
MAXIMUM_TRANSFER_CLOSURE_DEFECT = 2.0e-9
MAXIMUM_CONTINUUM_REFERENCE_DIFFERENCE = 2.0e-5
MINIMUM_TRUNCATION_ORDER = 0.75
MAXIMUM_COMMON_PROJECTOR_TO_FINE_SPATIAL_RATIO = 0.10
MINIMUM_DOMINANT_BLOCK_FRACTION = 0.50

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b.py"
)
THIS_HELPER = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_family_energy_transfer.py"
)
THIS_HELPER_TEST = "tests/test_causal_inner_family_energy_transfer.py"
THIS_CANONICAL_TEST = (
    "tests/"
    "test_causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_UNIFORM_FAMILY_TRANSFER_"
    "WP10C9D6C7C2B5B_RESULTS_2026-07-30.md"
)

PARENT_DIRECTORY = b5a.CANONICAL_DIRECTORY
SCOPE_DIRECTORY = c2b1.SCOPE_DIRECTORY
C2A2_DIRECTORY = c2b1.C2A2_DIRECTORY
C2B3_DIRECTORY = b5a.C2B3_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b"
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


def _validate_parent() -> dict:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    decision = parent["binding_decision"]
    if (
        parent["classification"]
        != "arrival_history_conditioning_and_horizon_audit_complete_"
        "shear_family_transfer_audit_required"
        or parent["authorized_next"]
        != "WP10c9d6c7c2b5b_shear_family_transfer_and_projector_audit"
        or not parent["passed"]
        or not decision["c2b4_rejection_preserved"]
        or not decision["shear_family_transfer_audit_authorized"]
        or decision["revised_uniform_recertification_authorized"]
        or decision["embedded_authorized"]
        or decision["operator_or_interface_redesign_authorized"]
    ):
        raise RuntimeError("WP10c9d6c7c2b5a binding status changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
    ):
        raise RuntimeError("WP10c9d6c7c2b5b analyzed identity changed")
    return parent


def _three_level_scalar(values: np.ndarray) -> dict:
    data = np.asarray(values, dtype=float)
    first = float(data[1] - data[0])
    second = float(data[2] - data[1])
    tiny = np.finfo(float).tiny
    order = float(np.log2(max(abs(first), tiny) / max(abs(second), tiny)))
    scale = max(float(np.max(np.abs(data))), tiny)
    return {
        "values": data,
        "observed_order": order,
        "fine_normalized_difference": abs(second) / scale,
        "refinement_error_cosine": (
            1.0 if first * second >= 0.0 else -1.0
        ),
    }


def _projector_audit(level: dict) -> tuple[np.ndarray, dict]:
    polynomial = []
    reports = []
    existing = np.asarray(level["projectors"], dtype=float)
    field_scales = np.asarray(level["field_scales"], dtype=float)
    for temporal, spatial in zip(
        level["temporal"],
        level["spatial"],
        strict=True,
    ):
        audit = causal_polynomial_spectral_projectors(
            temporal,
            spatial,
            field_scales,
        )
        polynomial.append(audit.primitive_projectors)
        reports.append(audit)
    projectors = np.asarray(polynomial)
    maximum_algebra = max(
        max(
            item.maximum_identity_defect,
            item.maximum_idempotence_defect,
            item.maximum_cross_projector_defect,
            item.maximum_eigenpair_defect,
            item.maximum_energy_orthogonality_defect,
            item.maximum_symmetrizer_defect,
            item.maximum_imaginary_part,
        )
        for item in reports
    )
    return projectors, {
        "maximum_polynomial_algebra_defect": maximum_algebra,
        "maximum_eigenvector_polynomial_projector_defect": (
            _relative_defect(existing, projectors)
        ),
        "minimum_spectral_gap": min(
            item.minimum_spectral_gap for item in reports
        ),
        "passed": bool(
            maximum_algebra <= MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
            and _relative_defect(existing, projectors)
            <= MAXIMUM_EQUIVALENT_PROJECTOR_DEFECT
        ),
    }


def _family_histories_for_projectors(
    level: dict,
    physical: np.ndarray,
    projectors: dict[str, np.ndarray],
) -> tuple[dict, np.ndarray]:
    cells = int(level["cells"])
    factor = cells // LEVELS[0]
    log_edges = np.log(np.asarray(level["grid"].edges, dtype=float))
    source_band = (
        c2a3.PACKET_SUPPORT[0] * factor,
        c2a3.PACKET_SUPPORT[1] * factor,
    )
    receiving_band = (
        c2a3.DOWNSTREAM_MEASUREMENT_FACE * factor,
        c2a3.PATCH_INTERFACE_FACE * factor,
    )
    source = causal_positive_band_energy_history(
        physical[:1],
        log_edges=log_edges,
        energy_metrics=level["energy"],
        projectors=level["projectors"],
        lower_face=source_band[0],
        upper_face=source_band[1],
    )
    initial = np.asarray(source.total_energy[0], dtype=float)
    if np.any(initial <= 0.0):
        raise RuntimeError("initial source energy is not positive")
    result = {}
    for label, matrices in projectors.items():
        measured = causal_positive_band_energy_history(
            physical,
            log_edges=log_edges,
            energy_metrics=level["energy"],
            projectors=matrices,
            lower_face=receiving_band[0],
            upper_face=receiving_band[1],
        )
        by_family = np.asarray(measured.family_energy, dtype=float)
        result[label] = {}
        for case_index, family in enumerate(FAMILIES):
            targets = tuple(c2b1.TARGET_FAMILIES[family])
            normalized = by_family[:, case_index] / initial[case_index]
            target = np.sum(normalized[:, list(targets)], axis=1)
            result[label][family] = {
                "target": target,
                "leakage": np.sum(normalized, axis=1) - target,
                "partition_defect": float(
                    measured.maximum_family_partition_relative_defect
                ),
            }
    return result, initial


def _window_average(
    times: np.ndarray,
    values: np.ndarray,
    window: tuple[float, float],
) -> float:
    lower, upper = (float(item) for item in window)
    mask = (times >= lower) & (times <= upper)
    selected_times = np.asarray(times[mask], dtype=float)
    selected_values = np.asarray(values[mask], dtype=float)
    if selected_times.size < 2:
        raise RuntimeError("arrival window has fewer than two samples")
    return float(
        np.trapezoid(selected_values, selected_times) / (upper - lower)
    )


def _residual_blocks(operator: dict) -> dict[str, np.ndarray]:
    return {
        **operator["blocks"],
        "mapped_storage_rate_derivative": operator["mapped_storage_rate"],
        "responsive_height_storage_rate_derivative": (
            operator["height_storage_rate"]
        ),
    }


def _transfer_audit(
    level: dict,
    physical: np.ndarray,
    times: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    cells = int(level["cells"])
    factor = cells // LEVELS[0]
    lower = c2a3.DOWNSTREAM_MEASUREMENT_FACE * factor
    upper = c2a3.PATCH_INTERFACE_FACE * factor
    operator = level["energy_operator"]
    result = {}
    arrays = {}
    for case_index, family in enumerate(FAMILIES):
        ledger = causal_physical_family_transfer_ledger(
            physical[:, case_index],
            times,
            log_edges=np.log(np.asarray(level["grid"].edges, dtype=float)),
            primitive_energy_metrics=level["energy"],
            primitive_projectors=level["projectors"],
            scaled_generator_per_s=level["generator"],
            descriptor_scaled_matrix=operator["descriptor"],
            scaled_residual_blocks=_residual_blocks(operator),
            primitive_column_scales=level["columns"],
            lower_face=lower,
            upper_face=upper,
        )
        targets = set(c2b1.TARGET_FAMILIES[family])
        opposite = [
            index for index in range(5) if index not in targets
        ]
        tensor = np.asarray(
            ledger.integrated_block_source_receiver_work,
            dtype=float,
        )
        block_scores = np.sum(
            np.abs(tensor[:, opposite, :]),
            axis=(1, 2),
        )
        total_score = max(float(np.sum(block_scores)), np.finfo(float).tiny)
        dominant = int(np.argmax(block_scores))
        result[family] = {
            "block_names": list(ledger.block_names),
            "maximum_family_partition_defect": (
                ledger.maximum_family_partition_defect
            ),
            "maximum_power_closure_defect": (
                ledger.maximum_power_closure_defect
            ),
            "maximum_block_matrix_closure_defect": (
                ledger.maximum_block_matrix_closure_defect
            ),
            "maximum_integrated_energy_defect": (
                ledger.maximum_integrated_energy_defect
            ),
            "dominant_opposite_receiver_block": (
                ledger.block_names[dominant]
            ),
            "dominant_opposite_receiver_absolute_fraction": (
                float(block_scores[dominant] / total_score)
            ),
            "net_opposite_receiver_work_by_block": np.sum(
                tensor[:, opposite, :],
                axis=(1, 2),
            ),
            "absolute_opposite_receiver_work_by_block": block_scores,
        }
        prefix = f"N{cells}__{family}__"
        arrays[prefix + "family_energy"] = ledger.family_energy
        arrays[prefix + "family_power"] = ledger.family_power_per_s
        arrays[prefix + "integrated_block_source_receiver_work"] = tensor
        if family == "shear":
            arrays[
                prefix + "integrated_block_source_receiver_cell_work"
            ] = ledger.integrated_block_source_receiver_cell_work
    return result, arrays


def _background_evaluator(parent_context, parent_base, field_scales):
    parent_log_spacing = float(
        np.mean(np.diff(np.log(parent_context.grid.edges)))
    )

    def evaluate(radii: np.ndarray) -> np.ndarray:
        return causal_c4_manufactured_primitive_state(
            np.log(np.asarray(radii, dtype=float)),
            np.log(
                parent_context.grid.centers[c2a2.PARENT_CORE_CELLS]
            ),
            parent_base[c2a2.PARENT_CORE_CELLS],
            parent_base[0],
            parent_base[-1],
            transition_log_width=(
                c2a2.TRANSITION_PARENT_CELLS * parent_log_spacing
            ),
            field_scales=field_scales,
        ).primitive_charts

    return evaluate


def _smooth_state_evaluator(level: dict, state: np.ndarray):
    log_centers = np.log(np.asarray(level["grid"].centers, dtype=float))
    spline = make_interp_spline(
        log_centers,
        np.asarray(state, dtype=float),
        k=5,
        axis=0,
    )

    def evaluate(radii: np.ndarray) -> np.ndarray:
        return np.asarray(spline(np.log(np.asarray(radii))), dtype=float)

    return evaluate


def _scaled_rms(values: np.ndarray, row_scales: np.ndarray) -> float:
    scaled = np.asarray(values, dtype=float).ravel() / np.asarray(
        row_scales,
        dtype=float,
    ).ravel()
    return float(np.sqrt(np.mean(scaled**2)))


def _continuum_action_audit(
    levels: dict[int, dict],
    initial_physical: dict[int, np.ndarray],
    parent_context,
    parent_base: np.ndarray,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    evaluator = _background_evaluator(
        parent_context,
        parent_base,
        field_scales,
    )
    print(f"{WORK_PACKAGE}: build 769-node continuum background", flush=True)
    primary_background = build_causal_five_field_continuum_background(
        levels[LEVELS[-1]]["context"],
        evaluator,
        node_count=PRIMARY_CONTINUUM_NODES,
    )
    print(f"{WORK_PACKAGE}: build 513-node continuum background", flush=True)
    secondary_background = build_causal_five_field_continuum_background(
        levels[LEVELS[-1]]["context"],
        evaluator,
        node_count=SECONDARY_CONTINUUM_NODES,
    )
    result = {}
    arrays = {}
    fine = levels[LEVELS[-1]]
    comparison_radii = np.geomspace(
        float(fine["grid"].edges[0]),
        float(fine["grid"].edges[-1]),
        CONTINUUM_COMPARISON_SAMPLES,
    )
    for case_index, family in enumerate(FAMILIES):
        perturbation_evaluator = _smooth_state_evaluator(
            fine,
            initial_physical[LEVELS[-1]][case_index],
        )
        primary = linearize_causal_five_field_continuum_reference(
            primary_background,
            perturbation_evaluator,
        )
        secondary = linearize_causal_five_field_continuum_reference(
            secondary_background,
            perturbation_evaluator,
        )
        reference_difference = _relative_defect(
            primary.evaluate_rate(comparison_radii),
            secondary.evaluate_rate(comparison_radii),
        )
        total_norms = []
        rate_norms = []
        block_norms = {name: [] for name in CONTINUUM_DAE_BLOCK_NAMES}
        for cells in LEVELS:
            level = levels[cells]
            operator = level["energy_operator"]
            grid = level["grid"]
            direction = c3._project_callable_to_cells(
                grid,
                perturbation_evaluator,
                quadrature_order=PROJECTION_ORDER,
            )
            continuum_rate = c3._project_callable_to_cells(
                grid,
                primary.evaluate_rate,
                quadrature_order=PROJECTION_ORDER,
            )
            continuum_rows = primary.integrate_blocks(grid.edges)
            scaled_direction = direction.ravel() / level["columns"]
            scaled_rate = continuum_rate.ravel() / level["columns"]
            discrete_scaled = (
                operator["descriptor"] @ scaled_rate
                + operator["mapped_storage_rate"] @ scaled_direction
                + operator["height_storage_rate"] @ scaled_direction
                + sum(
                    (
                        matrix @ scaled_direction
                        for matrix in operator["blocks"].values()
                    ),
                    start=np.zeros_like(scaled_direction),
                )
            )
            discrete_rows = (
                discrete_scaled * operator["row_scales"]
            ).reshape(cells, 5)
            continuum_total = sum(
                continuum_rows.values(),
                start=np.zeros((cells, 5), dtype=float),
            )
            truncation = discrete_rows - continuum_total
            total_norms.append(
                _scaled_rms(truncation, operator["row_scales"])
            )
            rate_error = np.linalg.solve(
                operator["descriptor"],
                truncation.ravel() / operator["row_scales"],
            )
            rate_norms.append(
                float(np.sqrt(np.mean(rate_error**2)))
            )

            discrete_blocks = {
                "mapped_temporal": (
                    operator["descriptor"] @ scaled_rate
                )
                * operator["row_scales"],
                "responsive_height_temporal": np.zeros_like(scaled_rate),
                "mapped_storage_rate": (
                    operator["mapped_storage_rate"] @ scaled_direction
                )
                * operator["row_scales"],
                "responsive_height_storage_rate": (
                    operator["height_storage_rate"] @ scaled_direction
                )
                * operator["row_scales"],
                **{
                    name: (matrix @ scaled_direction)
                    * operator["row_scales"]
                    for name, matrix in operator["blocks"].items()
                },
            }
            # The checkpoint stores only the complete descriptor.  Compare
            # its action with the sum of the two continuum temporal rows.
            temporal_continuum = (
                continuum_rows["mapped_temporal"]
                + continuum_rows["responsive_height_temporal"]
            )
            temporal_discrete = discrete_blocks["mapped_temporal"].reshape(
                cells,
                5,
            )
            temporal_error = temporal_discrete - temporal_continuum
            temporal_norm = _scaled_rms(
                temporal_error,
                operator["row_scales"],
            )
            block_norms["mapped_temporal"].append(temporal_norm)
            block_norms["responsive_height_temporal"].append(temporal_norm)
            for name in CONTINUUM_DAE_BLOCK_NAMES[2:]:
                values = discrete_blocks[name].reshape(cells, 5)
                block_norms[name].append(
                    _scaled_rms(
                        values - continuum_rows[name],
                        operator["row_scales"],
                    )
                )
            prefix = f"N{cells}__{family}__continuum_"
            arrays[prefix + "total_truncation_rows"] = truncation
            arrays[prefix + "mass_solved_scaled_rate_error"] = rate_error

        total_metric = _three_level_scalar(np.asarray(total_norms))
        rate_metric = _three_level_scalar(np.asarray(rate_norms))
        block_metrics = {
            name: {
                **_three_level_scalar(np.asarray(values)),
                "active": bool(np.max(np.abs(values)) > 0.0),
            }
            for name, values in block_norms.items()
        }
        active_block_orders = [
            item["observed_order"]
            for item in block_metrics.values()
            if item["active"]
        ]
        if not active_block_orders:
            raise RuntimeError("continuum action has no active blocks")
        result[family] = {
            "primary_secondary_rate_relative_difference": (
                reference_difference
            ),
            "maximum_primary_secondary_pointwise_ledger_defect": max(
                primary.maximum_pointwise_ledger_relative_defect,
                secondary.maximum_pointwise_ledger_relative_defect,
            ),
            "unsolved_DAE_truncation": total_metric,
            "mass_solved_rate_error": rate_metric,
            "block_truncation": block_metrics,
            "minimum_block_truncation_order": min(
                active_block_orders
            ),
        }
    return result, arrays


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "config.json",
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        C2B3_DIRECTORY / "transfer_manifest.json",
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "embedded_or_nonlinear_propagation_executed": False,
        "reference_levels": list(LEVELS),
        "representative_profiles": list(FAMILIES),
        "projector_definitions": [
            "local_eigenvector",
            "local_polynomial",
            "common_N392_field",
            "frozen_receiving_band_midpoint_diagnostic",
        ],
        "equivalent_projector_gate": (
            MAXIMUM_EQUIVALENT_PROJECTOR_DEFECT
        ),
        "common_projector_robustness_ratio": (
            MAXIMUM_COMMON_PROJECTOR_TO_FINE_SPATIAL_RATIO
        ),
        "continuum_nodes": [
            PRIMARY_CONTINUUM_NODES,
            SECONDARY_CONTINUUM_NODES,
        ],
        "minimum_truncation_order": MINIMUM_TRUNCATION_ORDER,
        "minimum_dominant_block_fraction": (
            MINIMUM_DOMINANT_BLOCK_FRACTION
        ),
        "frozen_projector_is_rotation_diagnostic_not_equivalent_"
        "uncertainty": True,
        "historical_c2b4_and_c2b5a_classifications_unchanged": True,
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
    lines = [
        "# WP10c9d6c7c2b5b — Uniform shear family-transfer audit",
        "",
        f"- Classification: `{summary['classification']}`",
        "- c2b4 and c2b5a classifications: preserved.",
        "- Operator changed: `False`.",
        "- Embedded/nonlinear/fixed-Q/reduced evolution: not run.",
        "",
        "## Projector audit",
        "",
        "| Level | polynomial algebra | eig/poly difference | minimum gap |",
        "|---|---:|---:|---:|",
    ]
    for level in LEVELS:
        item = summary["projector_audit"][f"N{level}"]
        lines.append(
            f"| N{level} | "
            f"{item['maximum_polynomial_algebra_defect']:.3e} | "
            f"{item['maximum_eigenvector_polynomial_projector_defect']:.3e} "
            f"| {item['minimum_spectral_gap']:.3e} |"
        )
    lines.extend(
        (
            "",
            "## Exact shear transfer",
            "",
            "| Level | dominant opposite-family block | absolute fraction | "
            "partition defect | power defect |",
            "|---|---|---:|---:|---:|",
        )
    )
    for level in LEVELS:
        item = summary["transfer_audit"][f"N{level}"]["shear"]
        lines.append(
            f"| N{level} | "
            f"{item['dominant_opposite_receiver_block']} | "
            f"{item['dominant_opposite_receiver_absolute_fraction']:.4f} | "
            f"{item['maximum_family_partition_defect']:.3e} | "
            f"{item['maximum_power_closure_defect']:.3e} |"
        )
    lines.extend(
        (
            "",
            "The transfer tensor is the exact transfer of the implemented "
            "frozen DAE. A large block contribution is not automatically a "
            "numerical defect.",
            "",
            "## Shear-leakage projector comparison",
            "",
            "| Projector definition | N98 | N196 | N392 | order |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for definition in (
        "local_eigenvector",
        "local_polynomial",
        "common_N392_field",
        "frozen_receiving_band_midpoint_diagnostic",
    ):
        item = summary["projector_observable"][definition]["shear"][
            "leakage"
        ]
        values = item["values"]
        lines.append(
            f"| {definition} | {values[0]:.6e} | {values[1]:.6e} | "
            f"{values[2]:.6e} | {item['observed_order']:.4f} |"
        )
    lines.extend(
        (
            "",
            "## Independent continuum action",
            "",
            "| Profile | reference difference | unsolved order | solved "
            "order | minimum block order |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for family in FAMILIES:
        item = summary["continuum_action_audit"][family]
        lines.append(
            f"| {family} | "
            f"{item['primary_secondary_rate_relative_difference']:.3e} | "
            f"{item['unsolved_DAE_truncation']['observed_order']:.4f} | "
            f"{item['mass_solved_rate_error']['observed_order']:.4f} | "
            f"{item['minimum_block_truncation_order']:.4f} |"
        )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            summary["decision_explanation"],
            "",
            f"Authorized next: `{summary['authorized_next']}`.",
            "",
            "No historical rejection is relabeled. Embedded discrimination, "
            "operator/interface redesign, nonlinear propagation, fixed-Q "
            "experiments, reduced evolution, and N1024 remain blocked.",
            "",
        )
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    parent = _validate_parent()
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
    scope = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")
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
    primary_windows = {
        family: tuple(primary_array[index])
        for index, family in enumerate(FAMILIES)
    }

    levels = c2b2._build_levels(
        base_edges,
        parent_context,
        parent_base,
        field_scales,
    )
    common_projectors = b5a._common_projectors(levels)
    projector_reports = {}
    polynomial_projectors = {}
    for cells in LEVELS:
        polynomial_projectors[cells], projector_reports[f"N{cells}"] = (
            _projector_audit(levels[cells])
        )
    if not all(item["passed"] for item in projector_reports.values()):
        raise RuntimeError("polynomial projector audit failed")

    common_log_centers = np.log(np.asarray(base_edges[:-1])) + 0.5 * np.diff(
        np.log(base_edges)
    )
    diagnostics = {}
    initial_physical = {}
    decisive = {
        "reference_levels": np.asarray(LEVELS, dtype=int),
    }
    for cells in LEVELS:
        initial, cases = b5a._representative_packet_matrix(
            levels[cells],
            scope_arrays,
            support_log_bounds,
        )
        propagated = c2b1._propagate_level(
            levels[cells],
            initial,
            cases,
            propagation_windows,
            horizon,
            common_log_centers,
        )
        times = np.asarray(propagated["times"], dtype=float)[::2]
        physical = np.asarray(propagated["physical"], dtype=float)[::2]
        initial_physical[cells] = physical[0]
        receiving_lower = (
            c2a3.DOWNSTREAM_MEASUREMENT_FACE * (cells // LEVELS[0])
        )
        receiving_upper = (
            c2a3.PATCH_INTERFACE_FACE * (cells // LEVELS[0])
        )
        midpoint = (receiving_lower + receiving_upper) // 2
        frozen = np.repeat(
            np.asarray(levels[cells]["projectors"][midpoint])[None],
            cells,
            axis=0,
        )
        histories, initial_energy = _family_histories_for_projectors(
            levels[cells],
            physical,
            {
                "local_eigenvector": levels[cells]["projectors"],
                "local_polynomial": polynomial_projectors[cells],
                "common_N392_field": common_projectors[cells],
                "frozen_receiving_band_midpoint_diagnostic": frozen,
            },
        )
        transfer, transfer_arrays = _transfer_audit(
            levels[cells],
            physical,
            times,
        )
        diagnostics[cells] = {
            "times": times,
            "histories": histories,
            "initial_energy": initial_energy,
            "transfer": transfer,
        }
        decisive[f"N{cells}__times_seconds"] = times
        decisive[f"N{cells}__initial_source_energy"] = initial_energy
        decisive.update(transfer_arrays)
        for definition, by_family in histories.items():
            for family, values in by_family.items():
                for observable in ("target", "leakage"):
                    decisive[
                        f"N{cells}__{definition}__{family}__{observable}"
                    ] = values[observable]

    projector_observable = {}
    common_ratios = []
    for definition in (
        "local_eigenvector",
        "local_polynomial",
        "common_N392_field",
        "frozen_receiving_band_midpoint_diagnostic",
    ):
        projector_observable[definition] = {}
        for family in FAMILIES:
            projector_observable[definition][family] = {}
            for observable in ("target", "leakage"):
                averages = np.asarray(
                    [
                        _window_average(
                            diagnostics[cells]["times"],
                            diagnostics[cells]["histories"][definition][
                                family
                            ][observable],
                            primary_windows[family],
                        )
                        for cells in LEVELS
                    ]
                )
                projector_observable[definition][family][observable] = (
                    _three_level_scalar(averages)
                )
    for family in FAMILIES:
        for observable in ("target", "leakage"):
            local = np.asarray(
                projector_observable["local_eigenvector"][family][observable][
                    "values"
                ]
            )
            common = np.asarray(
                projector_observable["common_N392_field"][family][observable][
                    "values"
                ]
            )
            common_ratios.append(
                float(
                    np.max(np.abs(common - local))
                    / max(
                        abs(float(local[-1] - local[-2])),
                        np.finfo(float).tiny,
                    )
                )
            )
    maximum_common_ratio = max(common_ratios)
    shear_leakage_local = projector_observable[
        "local_eigenvector"
    ]["shear"]["leakage"]
    shear_leakage_common = projector_observable[
        "common_N392_field"
    ]["shear"]["leakage"]
    shear_leakage_frozen = projector_observable[
        "frozen_receiving_band_midpoint_diagnostic"
    ]["shear"]["leakage"]
    shear_local_values = np.asarray(shear_leakage_local["values"])
    shear_common_values = np.asarray(shear_leakage_common["values"])
    shear_common_ratio = float(
        np.max(np.abs(shear_common_values - shear_local_values))
        / max(
            abs(float(shear_local_values[-1] - shear_local_values[-2])),
            np.finfo(float).tiny,
        )
    )

    transfer_reports = {
        f"N{cells}": diagnostics[cells]["transfer"]
        for cells in LEVELS
    }
    maximum_transfer_closure = max(
        max(
            item["maximum_family_partition_defect"],
            item["maximum_power_closure_defect"],
            item["maximum_block_matrix_closure_defect"],
        )
        for level in transfer_reports.values()
        for item in level.values()
    )
    shear_items = [
        transfer_reports[f"N{cells}"]["shear"] for cells in LEVELS
    ]
    dominant_names = [
        item["dominant_opposite_receiver_block"] for item in shear_items
    ]
    dominant_fractions = [
        item["dominant_opposite_receiver_absolute_fraction"]
        for item in shear_items
    ]
    stable_dominant = bool(
        len(set(dominant_names)) == 1
        and min(dominant_fractions) >= MINIMUM_DOMINANT_BLOCK_FRACTION
    )
    dominant_index = (
        shear_items[0]["block_names"].index(dominant_names[0])
        if stable_dominant
        else None
    )
    dominant_metric = (
        _three_level_scalar(
            np.asarray(
                [
                    item["net_opposite_receiver_work_by_block"][
                        dominant_index
                    ]
                    for item in shear_items
                ]
            )
        )
        if dominant_index is not None
        else None
    )

    continuum_report, continuum_arrays = _continuum_action_audit(
        levels,
        initial_physical,
        parent_context,
        parent_base,
        field_scales,
    )
    decisive.update(continuum_arrays)
    continuum_passed = bool(
        max(
            item["primary_secondary_rate_relative_difference"]
            for item in continuum_report.values()
        )
        <= MAXIMUM_CONTINUUM_REFERENCE_DIFFERENCE
        and min(
            min(
                item["unsolved_DAE_truncation"]["observed_order"],
                item["mass_solved_rate_error"]["observed_order"],
                item["minimum_block_truncation_order"],
            )
            for item in continuum_report.values()
        )
        >= MINIMUM_TRUNCATION_ORDER
    )
    common_projector_robust = bool(
        shear_common_ratio
        <= MAXIMUM_COMMON_PROJECTOR_TO_FINE_SPATIAL_RATIO
    )
    stable_noncontracting_block = bool(
        stable_dominant
        and dominant_metric is not None
        and dominant_metric["observed_order"] < MINIMUM_TRUNCATION_ORDER
        and continuum_report["shear"]["block_truncation"][
            dominant_names[0]
        ]["observed_order"]
        < MINIMUM_TRUNCATION_ORDER
    )
    rotation_sensitive_leakage = bool(
        shear_leakage_local["observed_order"] < MINIMUM_TRUNCATION_ORDER
        and shear_leakage_frozen["observed_order"]
        >= MINIMUM_TRUNCATION_ORDER
        and common_projector_robust
        and continuum_passed
        and not stable_noncontracting_block
    )
    if maximum_transfer_closure > MAXIMUM_TRANSFER_CLOSURE_DEFECT:
        raise RuntimeError("exact family-transfer ledger failed")

    if stable_noncontracting_block:
        classification = (
            "stable_noncontracting_shear_transfer_block_selected_"
            "local_audit_required"
        )
        authorized_next = (
            "WP10c9d6c7c2b5c_selected_block_local_truncation_audit"
        )
        explanation = (
            "The same dominant shear-transfer block is noncontracting in "
            "both the exact transfer and independent continuum-action "
            "audits. Only a local audit of that block is authorized."
        )
        revised_manifest_authorized = False
        local_block_audit_authorized = True
    elif rotation_sensitive_leakage:
        classification = (
            "raw_local_family_leakage_projector_rotation_sensitive_"
            "revised_transfer_observable_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c2b6a_revised_uniform_arrival_contract_manifest"
        )
        explanation = (
            "Equivalent local eigensolver and polynomial projectors agree, "
            "the common N392 local-projector field is robust for shear "
            "leakage, all independent continuum-action truncations contract, "
            "and no stable noncontracting transfer block is selected. Raw "
            "local opposite-family stored energy remains nonconvergent while "
            "the deliberately frozen-subspace diagnostic converges. The raw "
            "quantity therefore mixes transfer with spatial projector "
            "rotation and is non-certifying by itself. A definitions-only "
            "uniform manifest may retain total positive energy, target "
            "arrival, the exact covariant transfer balance, and explicitly "
            "projector-qualified quantities."
        )
        revised_manifest_authorized = True
        local_block_audit_authorized = False
    elif (
        continuum_passed
        and common_projector_robust
        and shear_leakage_local["observed_order"]
        >= MINIMUM_TRUNCATION_ORDER
    ):
        classification = (
            "family_transfer_physically_resolved_revised_observable_"
            "manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c2b6a_revised_uniform_arrival_contract_manifest"
        )
        explanation = (
            "Projector definitions and independent continuum action are "
            "consistent, with no stable numerical transfer defect. A "
            "definitions-only revised Tier-II observable manifest is "
            "authorized before any propagation."
        )
        revised_manifest_authorized = True
        local_block_audit_authorized = False
    else:
        classification = (
            "shear_family_transfer_mechanism_unresolved_embedded_blocked"
        )
        authorized_next = "none"
        explanation = (
            "The audit does not select a stable physical, projector, or "
            "numerical explanation. No recertification or redesign is "
            "authorized."
        )
        revised_manifest_authorized = False
        local_block_audit_authorized = False

    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "embedded_or_nonlinear_propagation_executed": False,
        "historical_classifications_preserved": True,
        "parent_classification": parent["classification"],
        "projector_audit": projector_reports,
        "projector_observable": projector_observable,
        "maximum_common_projector_difference_to_fine_spatial_ratio": (
            maximum_common_ratio
        ),
        "shear_leakage_common_projector_difference_to_fine_spatial_ratio": (
            shear_common_ratio
        ),
        "common_cross_grid_projector_robust": common_projector_robust,
        "raw_shear_leakage_projector_rotation_sensitive": (
            rotation_sensitive_leakage
        ),
        "transfer_audit": transfer_reports,
        "maximum_exact_transfer_closure_defect": maximum_transfer_closure,
        "stable_dominant_shear_transfer_block": stable_dominant,
        "dominant_shear_transfer_block_metric": dominant_metric,
        "stable_noncontracting_numerical_block_selected": (
            stable_noncontracting_block
        ),
        "continuum_action_audit": continuum_report,
        "continuum_action_audit_passed": continuum_passed,
        "independent_continuum_history_reference_available": False,
        "independent_continuum_action_reference_available": True,
        "classification": classification,
        "authorized_next": authorized_next,
        "decision_explanation": explanation,
        "passed": True,
        "binding_decision": {
            "c2b4_and_c2b5a_classifications_preserved": True,
            "exact_family_transfer_ledger_passed": True,
            "equivalent_local_projectors_passed": True,
            "independent_continuum_action_passed": continuum_passed,
            "family_resolved_leakage_certifying": False,
            "revised_uniform_manifest_authorized": (
                revised_manifest_authorized
            ),
            "selected_block_local_audit_authorized": (
                local_block_audit_authorized
            ),
            "uniform_recertification_propagation_authorized": False,
            "embedded_authorized": False,
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, _config())
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
    print(f"classification={classification}", flush=True)


if __name__ == "__main__":
    main()
