#!/usr/bin/env python3
"""Audit common anchors and the rotating monolithic inner-export error.

WP10c9d6c certified the self-consistent monolithic frozen tangent on three
uniform grids, but its common-mode physical-export gate failed because the
coarse/medium and medium/fine inner M/J/E errors were not aligned.  This
package preserves that rejection and asks two bounded questions:

1. are the inherited grid-native base states restrictions of one smooth
   continuum background; and
2. does a declared common-continuum lift remove the export-error rotation?

The direct inner-face flux is the attribution target.  It is excluded from
all explanatory first-cell groups.  No embedded or nonlinear work is
authorized by this runner.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_monolithic_uniform_exports_wp10c9d6c as wp10c9d6c

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_reconstruct_face_charts,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c1"
ANALYZED_BASE_COMMIT = "db6625f397141083d359505d84d072d4381ce92a"
ANALYZED_BASE_PARENT = "5884b307a3245e6f1c948d5147b5c2a1c70a509a"
ANALYZED_BASE_TREE = "099b8d308e9f612b21129b6b61206df5497b54ea"
THIS_RUNNER = (
    "scripts/run_causal_inner_monolithic_anchor_audit_wp10c9d6c1.py"
)

LABELS = tuple(wp10c9d6c.LABELS)
REFERENCE_LABEL = "uniform_N128"
N_FIELDS = 5
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
FIELD_NAMES = ("mass", "angular_momentum", "killing_energy")
STRIDE_AUDITS = (1, 2, 4)
PRIMARY_STRIDE = 2

MAXIMUM_PARENT_REPLAY_DEFECT = 1.0e-12
MAXIMUM_REFERENCE_ANCHOR_DEFECT = 1.0e-14
MAXIMUM_FIRST_CELL_LEDGER_DEFECT = 1.0e-10
MAXIMUM_CONSERVATIVE_TRANSPORT_DEFECT = 1.0e-12
MAXIMUM_STRIDE_DEFECT = 5.0e-3
MINIMUM_PROFILE_ORDER = 0.75
MINIMUM_PROFILE_ERROR_COSINE = 0.90
MAXIMUM_PROFILE_FINE_DIFFERENCE = 0.05
MINIMUM_ERROR_COSINE_IMPROVEMENT = 0.50
MINIMUM_GROUP_TARGET_ALIGNED_FRACTION = 0.80
MAXIMUM_GROUP_FIXED_COEFFICIENT_RESIDUAL = 0.45
MINIMUM_GROUP_TARGET_COSINE = 0.90

SPATIAL_BLOCKS = (
    "shear_principal",
    "height_principal",
    "local_stress_relaxation",
    "geometry",
    "cooling",
    "stream",
    "lower_height_work",
)
EXPLANATORY_TERMS = (
    "outer_face",
    "mapped_descriptor_rate",
    "responsive_height_descriptor_rate",
    "mapped_storage_rate_derivative",
    "responsive_height_storage_rate_derivative",
    *SPATIAL_BLOCKS,
)
GROUPS = {
    "outer_first_cell_transport": ("outer_face",),
    "mapped_storage": (
        "mapped_descriptor_rate",
        "mapped_storage_rate_derivative",
    ),
    "height_space_storage": (
        "responsive_height_descriptor_rate",
        "responsive_height_storage_rate_derivative",
        "height_principal",
        "lower_height_work",
    ),
    "stress_principal_relaxation": (
        "shear_principal",
        "local_stress_relaxation",
    ),
    "lower_sources": ("geometry", "cooling", "stream"),
}

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_uniform_exports_wp10c9d6c"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_REPLAY_CONTEXTS = PARENT_DIRECTORY / "replay_contexts.json"
PARENT_REPLAY_INPUTS = PARENT_DIRECTORY / "replay_inputs.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_anchor_audit_wp10c9d6c1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "tests/test_causal_inner_monolithic_anchor_audit_wp10c9d6c1.py",
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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        raise RuntimeError("WP10c9d6c1 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _load_parent() -> tuple[
    dict,
    dict[str, np.ndarray],
    dict,
    dict,
]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if not (
        summary["work_package"] == "WP10c9d6c"
        and summary["method_passed"]
        and not summary["passed"]
        and summary["classification"]
        == "monolithic_uniform_physical_exports_rejected"
        and not summary["embedded_export_discrimination_authorized"]
    ):
        raise RuntimeError("WP10c9d6c parent classification changed")
    parent_arrays = _load_npz(PARENT_ARRAYS)
    replay_payload, replay_arrays = wp10c9d6c._load_replay_inputs()
    configurations = wp10c9d6c._configurations(
        replay_payload,
        replay_arrays,
    )
    return summary, parent_arrays, replay_payload, configurations


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _selected_indices(count: int, stride: int) -> np.ndarray:
    indices = np.arange(0, int(count), int(stride), dtype=int)
    if indices[-1] != int(count) - 1:
        indices = np.append(indices, int(count) - 1)
    return indices


def _pchip_values(
    source_radii: np.ndarray,
    source_values: np.ndarray,
    target_radii: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_radii, dtype=float)
    target = np.asarray(target_radii, dtype=float)
    values = np.asarray(source_values, dtype=float)
    if (
        source.ndim != 1
        or target.ndim != 1
        or values.shape[0] != source.size
        or source.size < 4
        or np.any(source <= 0.0)
        or np.any(target <= 0.0)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("common-profile interpolation inputs are invalid")
    flat = values.reshape(source.size, -1)
    result = np.empty((target.size, flat.shape[1]), dtype=float)
    source_x = np.log(source)
    target_x = np.log(target)
    for column in range(flat.shape[1]):
        result[:, column] = PchipInterpolator(
            source_x,
            flat[:, column],
            extrapolate=True,
        )(target_x)
    return result.reshape((target.size,) + values.shape[1:])


def _profile_boundary_values(configuration: dict) -> tuple[np.ndarray, np.ndarray]:
    reconstruction = causal_five_field_reconstruct_face_charts(
        configuration["context"],
        configuration["base_primitives"],
        purpose="flux",
    )
    return (
        np.asarray(reconstruction.right_face_charts[0], dtype=float),
        np.asarray(
            configuration["context"].outer_boundary_frozen_exterior_chart,
            dtype=float,
        ),
    )


def _common_continuum_configurations(
    native: dict,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    reference = native[REFERENCE_LABEL]
    context = reference["context"]
    inner_anchor, outer_anchor = _profile_boundary_values(reference)
    source_radii = np.concatenate(
        (
            [float(context.grid.edges[0])],
            np.asarray(context.grid.centers, dtype=float),
            [float(context.grid.edges[-1])],
        )
    )
    source_values = np.vstack(
        (
            inner_anchor,
            np.asarray(reference["base_primitives"], dtype=float),
            outer_anchor,
        )
    )
    common = {}
    decisive = {
        "common_continuum_source_radii": source_radii,
        "common_continuum_source_values": source_values,
        "common_inner_boundary_anchor": inner_anchor,
        "common_outer_boundary_anchor": outer_anchor,
    }
    maximum_reference_defect = 0.0
    maximum_reconstruction_factor_change = 0.0
    for label in LABELS:
        configuration = native[label]
        base = _pchip_values(
            source_radii,
            source_values,
            configuration["context"].grid.centers,
        )
        common_context = replace(
            configuration["context"],
            outer_boundary_frozen_exterior_chart=np.array(
                outer_anchor,
                copy=True,
            ),
        ).validated()
        reconstruction = causal_five_field_reconstruct_face_charts(
            common_context,
            base,
            purpose="flux",
        )
        factor_change = float(
            np.max(np.abs(reconstruction.admissibility_factors - 1.0))
        )
        maximum_reconstruction_factor_change = max(
            maximum_reconstruction_factor_change,
            factor_change,
        )
        if label == REFERENCE_LABEL:
            maximum_reference_defect = _relative_difference(
                base,
                configuration["base_primitives"],
            )
        common[label] = {
            **configuration,
            "context": common_context,
            "base_primitives": base,
        }
        decisive[f"{label}__common_base_primitives"] = base
        decisive[f"{label}__native_minus_common_base"] = (
            np.asarray(configuration["base_primitives"], dtype=float) - base
        )
    report = {
        "reference_label": REFERENCE_LABEL,
        "source_definition": (
            "PCHIP in log radius through the native N128 inner trace, "
            "N128 cell centers, and N128 frozen outer chart"
        ),
        "maximum_reference_anchor_defect": maximum_reference_defect,
        "maximum_reconstruction_factor_change": (
            maximum_reconstruction_factor_change
        ),
        "passed": bool(
            maximum_reference_defect <= MAXIMUM_REFERENCE_ANCHOR_DEFECT
            and maximum_reconstruction_factor_change == 0.0
        ),
    }
    return common, decisive, report


def _field_scales(configurations: dict) -> np.ndarray:
    values = [
        np.asarray(
            configurations[label]["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, N_FIELDS)
        for label in LABELS
    ]
    return np.maximum(
        np.max(np.abs(np.concatenate(values, axis=0)), axis=0),
        np.finfo(float).tiny,
    )


def _profile_metrics(
    profiles: dict[str, np.ndarray],
    configurations: dict,
    scales: np.ndarray,
) -> dict:
    coarse_label, medium_label, fine_label = LABELS
    coarse_radii = configurations[coarse_label]["context"].grid.centers
    medium_radii = configurations[medium_label]["context"].grid.centers
    fine_radii = configurations[fine_label]["context"].grid.centers
    coarse = np.asarray(profiles[coarse_label], dtype=float)
    medium = np.asarray(profiles[medium_label], dtype=float)
    fine = np.asarray(profiles[fine_label], dtype=float)
    medium_on_coarse = _pchip_values(
        medium_radii,
        medium,
        coarse_radii,
    )
    fine_on_medium = _pchip_values(
        fine_radii,
        fine,
        medium_radii,
    )
    first = (medium_on_coarse - coarse) / scales
    second_medium = (fine_on_medium - medium) / scales
    second = _pchip_values(
        medium_radii,
        second_medium,
        coarse_radii,
    )
    weights = np.asarray(
        configurations[coarse_label]["context"].grid.cell_measures,
        dtype=float,
    )
    weights = weights / np.sum(weights)

    def norm(values: np.ndarray) -> float:
        return float(
            np.sqrt(np.sum(weights[:, None] * np.asarray(values) ** 2))
        )

    first_norm = norm(first)
    second_norm = norm(second)
    order = float(np.log2(first_norm / second_norm))
    error_cosine = _cosine(
        first * np.sqrt(weights[:, None]),
        second * np.sqrt(weights[:, None]),
    )
    fine_maximum = float(np.max(np.abs(second_medium)))
    component_first = np.sqrt(np.sum(weights[:, None] * first**2, axis=0))
    component_second = np.sqrt(
        np.sum(weights[:, None] * second**2, axis=0)
    )
    active = np.maximum(component_first, component_second) >= 1.0e-12
    component_orders = np.full(N_FIELDS, np.nan, dtype=float)
    component_cosines = np.full(N_FIELDS, np.nan, dtype=float)
    for field in np.flatnonzero(active):
        component_orders[field] = float(
            np.log2(component_first[field] / component_second[field])
        )
        component_cosines[field] = _cosine(
            first[:, field] * np.sqrt(weights),
            second[:, field] * np.sqrt(weights),
        )
    passed = bool(
        order >= MINIMUM_PROFILE_ORDER
        and error_cosine >= MINIMUM_PROFILE_ERROR_COSINE
        and fine_maximum <= MAXIMUM_PROFILE_FINE_DIFFERENCE
        and np.all(
            component_orders[active] >= MINIMUM_PROFILE_ORDER
        )
        and np.all(
            component_cosines[active]
            >= MINIMUM_PROFILE_ERROR_COSINE
        )
    )
    return {
        "passed": passed,
        "observed_order": order,
        "error_cosine": error_cosine,
        "fine_maximum_scaled_difference": fine_maximum,
        "component_orders": component_orders.tolist(),
        "component_error_cosines": component_cosines.tolist(),
        "active_fields": np.flatnonzero(active).tolist(),
        "coarse_medium_norm": first_norm,
        "medium_fine_restricted_norm": second_norm,
    }


def _build_tangents(
    configurations: dict,
    *,
    anchor_name: str,
) -> dict:
    result = {}
    for label in LABELS:
        configuration = configurations[label]
        print(
            f"WP10c9d6c1: build {anchor_name} tangent {label}",
            flush=True,
        )
        result[label] = causal_five_field_monolithic_frozen_tangent(
            configuration["context"],
            configuration["base_primitives"],
            primitive_column_scales=(
                configuration["primitive_column_scales"]
            ),
            conservation_row_scales=(
                configuration["conservation_row_scales"]
            ),
            path_quadrature_order=wp10c9d6c.PATH_QUADRATURE_ORDER,
        )
    return result


def _native_replay_report(
    tangents: dict,
    parent_arrays: dict[str, np.ndarray],
) -> dict:
    configurations = {}
    maximum = 0.0
    for label in LABELS:
        tangent = tangents[label]
        defects = {
            "descriptor": _relative_difference(
                tangent.descriptor_scaled_matrix,
                parent_arrays[f"{label}__descriptor"],
            ),
            "storage_rate_derivative": _relative_difference(
                tangent.storage_rate_derivative_scaled_matrix,
                parent_arrays[f"{label}__storage_rate_derivative"],
            ),
            "stationary_jacobian": _relative_difference(
                tangent.stationary_scaled_jacobian,
                parent_arrays[f"{label}__stationary_jacobian"],
            ),
            "generator": _relative_difference(
                tangent.scaled_generator_per_s,
                parent_arrays[f"{label}__generator"],
            ),
            "observable_map": _relative_difference(
                wp10c9d6c._observable_map(tangent),
                parent_arrays[f"{label}__observable_map"],
            ),
        }
        local_maximum = max(defects.values())
        maximum = max(maximum, local_maximum)
        configurations[label] = {
            "defects": defects,
            "maximum_defect": local_maximum,
            "passed": bool(
                local_maximum <= MAXIMUM_PARENT_REPLAY_DEFECT
            ),
        }
    return {
        "configurations": configurations,
        "maximum_defect": maximum,
        "passed": bool(maximum <= MAXIMUM_PARENT_REPLAY_DEFECT),
    }


def _physical_rows(
    matrix: np.ndarray,
    vectors: np.ndarray,
    row_scales: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    scaled = np.asarray(vectors, dtype=float) @ np.asarray(
        matrix,
        dtype=float,
    ).T
    return scaled.reshape(-1, n_cells, N_FIELDS) * np.asarray(
        row_scales,
        dtype=float,
    ).reshape(1, n_cells, N_FIELDS)


def _first_cell_ledger(
    tangent,
    scaled_state: np.ndarray,
    scaled_rate: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    n_cells = int(tangent.base_primitives.shape[0])
    rows = tangent.conservation_row_scales
    spatial = tangent.spatial_tangent
    face = np.einsum(
        "fkd,td->tfk",
        spatial.shared_face_flux_scaled_jacobians[
            :,
            CONSERVATIVE_FIELDS,
            :,
        ],
        scaled_state,
        optimize=True,
    )
    conservative_rows = _physical_rows(
        spatial.block_scaled_jacobians[
            "candidate_conservative_transport"
        ],
        scaled_state,
        rows,
        n_cells,
    )[:, 0, CONSERVATIVE_FIELDS]
    face_difference = face[:, 1] - face[:, 0]
    transport_defect = _relative_difference(
        conservative_rows,
        face_difference,
    )
    terms = {
        "target_inner_face": face[:, 0],
        "outer_face": face[:, 1],
        "mapped_descriptor_rate": _physical_rows(
            tangent.mapped_descriptor_scaled_matrix,
            scaled_rate,
            rows,
            n_cells,
        )[:, 0, CONSERVATIVE_FIELDS],
        "responsive_height_descriptor_rate": _physical_rows(
            tangent.responsive_height_descriptor_scaled_matrix,
            scaled_rate,
            rows,
            n_cells,
        )[:, 0, CONSERVATIVE_FIELDS],
        "mapped_storage_rate_derivative": _physical_rows(
            tangent.mapped_storage_rate_derivative_scaled_matrix,
            scaled_state,
            rows,
            n_cells,
        )[:, 0, CONSERVATIVE_FIELDS],
        "responsive_height_storage_rate_derivative": _physical_rows(
            tangent.responsive_height_storage_rate_derivative_scaled_matrix,
            scaled_state,
            rows,
            n_cells,
        )[:, 0, CONSERVATIVE_FIELDS],
    }
    for name in SPATIAL_BLOCKS:
        terms[name] = _physical_rows(
            spatial.block_scaled_jacobians[f"candidate_{name}"],
            scaled_state,
            rows,
            n_cells,
        )[:, 0, CONSERVATIVE_FIELDS]
    explanatory = sum(
        (terms[name] for name in EXPLANATORY_TERMS),
        start=np.zeros_like(terms["target_inner_face"]),
    )
    ledger_defect = _relative_difference(
        terms["target_inner_face"],
        explanatory,
    )
    return terms, {
        "conservative_transport_defect": transport_defect,
        "first_cell_ledger_defect": ledger_defect,
        "passed": bool(
            transport_defect <= MAXIMUM_CONSERVATIVE_TRANSPORT_DEFECT
            and ledger_defect <= MAXIMUM_FIRST_CELL_LEDGER_DEFECT
        ),
    }


def _history(
    configuration: dict,
    tangent,
) -> tuple[dict, dict]:
    times = np.asarray(configuration["times"], dtype=float)
    state, restart = wp10c9d6c._propagate(
        tangent.scaled_generator_per_s,
        configuration["initial_directions"]["common_mode"],
        times,
    )
    rate = state @ tangent.scaled_generator_per_s.T
    observable_map = wp10c9d6c._observable_map(tangent)
    signals = state @ observable_map.T
    ledger, ledger_report = _first_cell_ledger(
        tangent,
        state,
        rate,
    )
    return {
        "times": times,
        "scaled_state": state,
        "scaled_rate": rate,
        "signals": signals,
        "observable_map": observable_map,
        "first_cell_ledger": ledger,
    }, {
        "restart_defect": restart,
        **ledger_report,
        "passed": bool(
            ledger_report["passed"]
            and restart <= wp10c9d6c.MAXIMUM_RESTART_DEFECT
        ),
    }


def _stride_report(
    histories: dict,
    physical_scales: np.ndarray,
) -> dict:
    reports = {}
    reference_endpoints = {}
    for label in LABELS:
        history = histories[label]
        reference_endpoints[label] = wp10c9d6c._cumulative(
            history["times"],
            history["signals"],
        )[-1]
    maximum_endpoint_defect = 0.0
    duration = max(
        float(histories[LABELS[0]]["times"][-1]),
        np.finfo(float).tiny,
    )
    for stride in STRIDE_AUDITS:
        signals = {}
        cumulative = {}
        for label in LABELS:
            history = histories[label]
            indices = _selected_indices(
                history["times"].size,
                stride,
            )
            selected_times = history["times"][indices]
            selected_signals = history["signals"][indices]
            signals[label] = selected_signals
            cumulative[label] = wp10c9d6c._cumulative(
                selected_times,
                selected_signals,
            )
            endpoint_defect = float(
                np.max(
                    np.abs(
                        cumulative[label][-1]
                        - reference_endpoints[label]
                    )
                    / (physical_scales * duration)
                )
            )
            maximum_endpoint_defect = max(
                maximum_endpoint_defect,
                endpoint_defect,
            )
        reports[str(stride)] = {
            "instantaneous": wp10c9d6c._history_metrics(
                signals,
                physical_scales,
            ),
            "cumulative": wp10c9d6c._history_metrics(
                cumulative,
                physical_scales * duration,
            ),
        }
    primary = reports[str(PRIMARY_STRIDE)]
    restart_passed = all(
        histories[label]["restart_defect"]
        <= wp10c9d6c.MAXIMUM_RESTART_DEFECT
        if "restart_defect" in histories[label]
        else True
        for label in LABELS
    )
    return {
        "strides": reports,
        "primary_stride": PRIMARY_STRIDE,
        "primary_instantaneous": primary["instantaneous"],
        "primary_cumulative": primary["cumulative"],
        "maximum_cumulative_endpoint_defect": maximum_endpoint_defect,
        "stride_passed": bool(
            maximum_endpoint_defect <= MAXIMUM_STRIDE_DEFECT
        ),
        "restart_passed": restart_passed,
        "passed": bool(
            primary["instantaneous"]["passed"]
            and primary["cumulative"]["passed"]
            and maximum_endpoint_defect <= MAXIMUM_STRIDE_DEFECT
        ),
    }


def _parent_history_replay(
    histories: dict,
    parent_arrays: dict[str, np.ndarray],
) -> dict:
    configurations = {}
    maximum = 0.0
    for label in LABELS:
        history = histories[label]
        indices = _selected_indices(
            history["times"].size,
            wp10c9d6c.TIME_SAMPLE_STRIDE,
        )
        times = history["times"][indices]
        signals = history["signals"][indices]
        cumulative = wp10c9d6c._cumulative(times, signals)
        defects = {
            "times": _relative_difference(
                times,
                parent_arrays[f"common_mode__{label}__times"],
            ),
            "signals": _relative_difference(
                signals,
                parent_arrays[f"common_mode__{label}__signals"],
            ),
            "cumulative": _relative_difference(
                cumulative,
                parent_arrays[f"common_mode__{label}__cumulative"],
            ),
            "final_state": _relative_difference(
                history["scaled_state"][-1],
                parent_arrays[
                    f"common_mode__{label}__final_scaled_state"
                ],
            ),
        }
        local_maximum = max(defects.values())
        maximum = max(maximum, local_maximum)
        configurations[label] = {
            "defects": defects,
            "maximum_defect": local_maximum,
            "passed": bool(
                local_maximum <= MAXIMUM_PARENT_REPLAY_DEFECT
            ),
        }
    return {
        "configurations": configurations,
        "maximum_defect": maximum,
        "passed": bool(maximum <= MAXIMUM_PARENT_REPLAY_DEFECT),
    }


def _cumulative_terms(
    times: np.ndarray,
    terms: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: wp10c9d6c._cumulative(times, values)
        for name, values in terms.items()
    }


def _group_metrics(
    target: np.ndarray,
    group: np.ndarray,
) -> dict:
    target_vector = np.asarray(target, dtype=float).ravel()
    group_vector = np.asarray(group, dtype=float).ravel()
    denominator = max(
        float(np.dot(target_vector, target_vector)),
        np.finfo(float).tiny,
    )
    aligned_fraction = float(
        np.dot(target_vector, group_vector) / denominator
    )
    fixed_residual = float(
        np.linalg.norm(target_vector - group_vector)
        / max(np.linalg.norm(target_vector), np.finfo(float).tiny)
    )
    cosine = _cosine(target_vector, group_vector)
    return {
        "target_aligned_fraction": aligned_fraction,
        "fixed_coefficient_residual": fixed_residual,
        "target_cosine": cosine,
        "passed": bool(
            aligned_fraction >= MINIMUM_GROUP_TARGET_ALIGNED_FRACTION
            and fixed_residual
            <= MAXIMUM_GROUP_FIXED_COEFFICIENT_RESIDUAL
            and cosine >= MINIMUM_GROUP_TARGET_COSINE
        ),
    }


def _attribution_for_kind(
    histories: dict,
    physical_scales: np.ndarray,
    *,
    cumulative: bool,
) -> tuple[dict, float]:
    scaled_terms = {}
    for label in LABELS:
        history = histories[label]
        terms = history["first_cell_ledger"]
        if cumulative:
            terms = _cumulative_terms(history["times"], terms)
        scaled_terms[label] = {
            name: np.asarray(values, dtype=float) / physical_scales
            for name, values in terms.items()
        }
    pairs = (
        ("coarse_medium", LABELS[0], LABELS[1]),
        ("medium_fine", LABELS[1], LABELS[2]),
    )
    reports = {}
    maximum_closure = 0.0
    for pair_name, coarse, fine in pairs:
        target = (
            scaled_terms[fine]["target_inner_face"]
            - scaled_terms[coarse]["target_inner_face"]
        )
        differences = {
            name: scaled_terms[fine][name] - scaled_terms[coarse][name]
            for name in EXPLANATORY_TERMS
        }
        complete = sum(
            differences.values(),
            start=np.zeros_like(target),
        )
        closure = _relative_difference(target, complete)
        maximum_closure = max(maximum_closure, closure)
        groups = {}
        for group_name, names in GROUPS.items():
            group = sum(
                (differences[name] for name in names),
                start=np.zeros_like(target),
            )
            groups[group_name] = _group_metrics(target, group)
        reports[pair_name] = {
            "complete_explanatory_closure_defect": closure,
            "groups": groups,
        }
    return reports, maximum_closure


def _attribution_report(
    histories: dict,
    physical_scales: np.ndarray,
) -> dict:
    instantaneous, instant_closure = _attribution_for_kind(
        histories,
        physical_scales,
        cumulative=False,
    )
    cumulative, cumulative_closure = _attribution_for_kind(
        histories,
        physical_scales,
        cumulative=True,
    )
    stable_groups = []
    for group_name in GROUPS:
        if all(
            reports[pair]["groups"][group_name]["passed"]
            for reports in (instantaneous, cumulative)
            for pair in ("coarse_medium", "medium_fine")
        ):
            stable_groups.append(group_name)
    maximum_closure = max(instant_closure, cumulative_closure)
    return {
        "target": "direct inner-face M/J/E refinement difference",
        "target_excluded_from_explanatory_groups": True,
        "instantaneous": instantaneous,
        "cumulative": cumulative,
        "stable_groups": stable_groups,
        "maximum_complete_explanatory_closure_defect": maximum_closure,
        "passed": bool(
            maximum_closure <= MAXIMUM_FIRST_CELL_LEDGER_DEFECT
        ),
    }


def _store_history(
    decisive: dict[str, np.ndarray],
    anchor_name: str,
    label: str,
    history: dict,
) -> None:
    prefix = f"{anchor_name}__{label}__"
    decisive[prefix + "times"] = history["times"]
    decisive[prefix + "signals"] = history["signals"]
    decisive[prefix + "final_scaled_state"] = history["scaled_state"][-1]
    for name, values in history["first_cell_ledger"].items():
        decisive[prefix + f"first_cell__{name}"] = values


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, parent_arrays, _replay_payload, native = _load_parent()
    common, common_decisive, common_lift_report = (
        _common_continuum_configurations(native)
    )
    decisive: dict[str, np.ndarray] = dict(common_decisive)
    physical_scales = np.asarray(
        parent["fixed_physical_observable_scales"],
        dtype=float,
    )
    decisive["fixed_physical_observable_scales"] = physical_scales
    primitive_scales = _field_scales(native)
    decisive["fixed_primitive_field_scales"] = primitive_scales

    native_tangents = _build_tangents(native, anchor_name="native")
    common_tangents = _build_tangents(common, anchor_name="common")
    native_replay = _native_replay_report(
        native_tangents,
        parent_arrays,
    )
    reference_invariance = {
        "generator_defect": _relative_difference(
            native_tangents[REFERENCE_LABEL].scaled_generator_per_s,
            common_tangents[REFERENCE_LABEL].scaled_generator_per_s,
        ),
        "descriptor_defect": _relative_difference(
            native_tangents[REFERENCE_LABEL].descriptor_scaled_matrix,
            common_tangents[REFERENCE_LABEL].descriptor_scaled_matrix,
        ),
    }
    reference_invariance["passed"] = bool(
        max(
            reference_invariance["generator_defect"],
            reference_invariance["descriptor_defect"],
        )
        <= MAXIMUM_PARENT_REPLAY_DEFECT
    )
    common_method_reports = {
        label: wp10c9d6c._method_report(
            common[label],
            common_tangents[label],
        )
        for label in LABELS
    }
    common_method_passed = bool(
        all(report["passed"] for report in common_method_reports.values())
    )

    native_histories = {}
    common_histories = {}
    history_reports = {"native": {}, "common": {}}
    for anchor_name, configurations, tangents, destination in (
        ("native", native, native_tangents, native_histories),
        ("common", common, common_tangents, common_histories),
    ):
        for label in LABELS:
            print(
                f"WP10c9d6c1: propagate {anchor_name} {label}",
                flush=True,
            )
            history, report = _history(
                configurations[label],
                tangents[label],
            )
            history["restart_defect"] = report["restart_defect"]
            destination[label] = history
            history_reports[anchor_name][label] = report
            _store_history(decisive, anchor_name, label, history)
            if anchor_name == "common":
                decisive[f"common__{label}__generator"] = (
                    tangents[label].scaled_generator_per_s
                )
                decisive[f"common__{label}__descriptor"] = (
                    tangents[label].descriptor_scaled_matrix
                )
                decisive[f"common__{label}__scaled_base_rate"] = (
                    tangents[label].scaled_base_rate_per_s
                )

    native_history_replay = _parent_history_replay(
        native_histories,
        parent_arrays,
    )
    native_ladder = _stride_report(
        native_histories,
        physical_scales,
    )
    common_ladder = _stride_report(
        common_histories,
        physical_scales,
    )
    native_attribution = _attribution_report(
        native_histories,
        physical_scales[:3],
    )
    common_attribution = _attribution_report(
        common_histories,
        physical_scales[:3],
    )

    native_base_profiles = {
        label: native[label]["base_primitives"] for label in LABELS
    }
    common_base_profiles = {
        label: common[label]["base_primitives"] for label in LABELS
    }
    native_rate_profiles = {
        label: native_tangents[label].physical_base_rate_per_s.reshape(
            -1,
            N_FIELDS,
        )
        for label in LABELS
    }
    common_rate_profiles = {
        label: common_tangents[label].physical_base_rate_per_s.reshape(
            -1,
            N_FIELDS,
        )
        for label in LABELS
    }
    rate_scale = np.maximum(
        np.max(
            np.abs(
                np.concatenate(
                    (
                        *native_rate_profiles.values(),
                        *common_rate_profiles.values(),
                    ),
                    axis=0,
                )
            ),
            axis=0,
        ),
        np.finfo(float).tiny,
    )
    decisive["fixed_base_rate_field_scales"] = rate_scale
    profile_reports = {
        "native_base": _profile_metrics(
            native_base_profiles,
            native,
            primitive_scales,
        ),
        "common_base": _profile_metrics(
            common_base_profiles,
            common,
            primitive_scales,
        ),
        "native_base_rate": _profile_metrics(
            native_rate_profiles,
            native,
            rate_scale,
        ),
        "common_base_rate": _profile_metrics(
            common_rate_profiles,
            common,
            rate_scale,
        ),
    }

    native_error_floor = min(
        float(
            native_ladder["primary_instantaneous"][
                "refinement_error_cosine"
            ]
        ),
        float(
            native_ladder["primary_cumulative"][
                "refinement_error_cosine"
            ]
        ),
    )
    common_error_floor = min(
        float(
            common_ladder["primary_instantaneous"][
                "refinement_error_cosine"
            ]
        ),
        float(
            common_ladder["primary_cumulative"][
                "refinement_error_cosine"
            ]
        ),
    )
    error_cosine_improvement = common_error_floor - native_error_floor
    native_profile_gate_failed = bool(
        not profile_reports["native_base"]["passed"]
        or not profile_reports["native_base_rate"]["passed"]
    )
    common_profile_gate_failed = bool(
        not profile_reports["common_base"]["passed"]
        or not profile_reports["common_base_rate"]["passed"]
    )
    profile_mapping_is_discriminating = bool(
        not common_profile_gate_failed
    )
    native_anchor_inconsistency_established = bool(
        native_profile_gate_failed and profile_mapping_is_discriminating
    )
    anchor_hypothesis_supported = bool(
        common_ladder["passed"]
        or error_cosine_improvement
        >= MINIMUM_ERROR_COSINE_IMPROVEMENT
    )
    stable_groups = list(common_attribution["stable_groups"])

    maximum_ledger_defect = max(
        report["first_cell_ledger_defect"]
        for anchor_reports in history_reports.values()
        for report in anchor_reports.values()
    )
    maximum_transport_defect = max(
        report["conservative_transport_defect"]
        for anchor_reports in history_reports.values()
        for report in anchor_reports.values()
    )
    history_method_passed = bool(
        all(
            report["passed"]
            for anchor_reports in history_reports.values()
            for report in anchor_reports.values()
        )
    )
    method_passed = bool(
        common_lift_report["passed"]
        and native_replay["passed"]
        and reference_invariance["passed"]
        and common_method_passed
        and native_history_replay["passed"]
        and history_method_passed
        and native_attribution["passed"]
        and common_attribution["passed"]
    )

    if not method_passed:
        classification = "monolithic_anchor_audit_method_gate_failed"
        authorized_next = "none"
    elif anchor_hypothesis_supported and common_ladder["passed"]:
        classification = (
            "common_continuum_anchor_removes_uniform_error_rotation"
        )
        authorized_next = (
            "monolithic_equilibrium_background_construction"
        )
    elif anchor_hypothesis_supported:
        classification = (
            "grid_native_anchor_inconsistency_supported_"
            "uniform_gate_still_failed"
        )
        authorized_next = (
            "monolithic_equilibrium_background_construction"
        )
    elif stable_groups:
        classification = (
            "stable_first_cell_group_selected_uniform_gate_still_failed"
        )
        authorized_next = (
            "targeted_inner_first_cell_consistency_audit"
        )
    else:
        classification = (
            "uniform_inner_export_error_direction_unresolved"
        )
        authorized_next = "none"

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    gates = {
        "maximum_parent_replay_defect": MAXIMUM_PARENT_REPLAY_DEFECT,
        "maximum_reference_anchor_defect": (
            MAXIMUM_REFERENCE_ANCHOR_DEFECT
        ),
        "maximum_first_cell_ledger_defect": (
            MAXIMUM_FIRST_CELL_LEDGER_DEFECT
        ),
        "maximum_conservative_transport_defect": (
            MAXIMUM_CONSERVATIVE_TRANSPORT_DEFECT
        ),
        "maximum_stride_defect": MAXIMUM_STRIDE_DEFECT,
        "minimum_profile_order": MINIMUM_PROFILE_ORDER,
        "minimum_profile_error_cosine": (
            MINIMUM_PROFILE_ERROR_COSINE
        ),
        "maximum_profile_fine_difference": (
            MAXIMUM_PROFILE_FINE_DIFFERENCE
        ),
        "minimum_error_cosine_improvement": (
            MINIMUM_ERROR_COSINE_IMPROVEMENT
        ),
        "minimum_group_target_aligned_fraction": (
            MINIMUM_GROUP_TARGET_ALIGNED_FRACTION
        ),
        "maximum_group_fixed_coefficient_residual": (
            MAXIMUM_GROUP_FIXED_COEFFICIENT_RESIDUAL
        ),
        "minimum_group_target_cosine": MINIMUM_GROUP_TARGET_COSINE,
        "inherited_uniform_export_gates": (
            json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))["gates"]
        ),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "reference_label": REFERENCE_LABEL,
        "primary_stride": PRIMARY_STRIDE,
        "stride_audits": STRIDE_AUDITS,
        "explanatory_terms": EXPLANATORY_TERMS,
        "groups": GROUPS,
        "target_definition": (
            "direct monolithic inner-face M/J/E refinement difference"
        ),
        "target_excluded_from_explanatory_groups": True,
        "gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "passed": bool(
            method_passed
            and (
                anchor_hypothesis_supported
                or bool(stable_groups)
            )
        ),
        "method_passed": method_passed,
        "parent_wp10c9d6c_classification_preserved": True,
        "parent_summary_path": _relative(PARENT_SUMMARY),
        "parent_summary_sha256": _sha256(PARENT_SUMMARY),
        "parent_decisive_arrays_path": _relative(PARENT_ARRAYS),
        "parent_decisive_arrays_sha256": _sha256(PARENT_ARRAYS),
        "parent_replay_contexts_sha256": _sha256(
            PARENT_REPLAY_CONTEXTS
        ),
        "parent_replay_inputs_sha256": _sha256(PARENT_REPLAY_INPUTS),
        "common_lift_report": common_lift_report,
        "native_replay_report": native_replay,
        "reference_invariance_report": reference_invariance,
        "common_method_reports": common_method_reports,
        "history_method_reports": history_reports,
        "maximum_first_cell_ledger_defect": maximum_ledger_defect,
        "maximum_conservative_transport_defect": (
            maximum_transport_defect
        ),
        "native_history_replay": native_history_replay,
        "profile_reports": profile_reports,
        "native_ladder": native_ladder,
        "common_ladder": common_ladder,
        "native_attribution": native_attribution,
        "common_attribution": common_attribution,
        "native_error_cosine_floor": native_error_floor,
        "common_error_cosine_floor": common_error_floor,
        "error_cosine_improvement": error_cosine_improvement,
        "native_profile_gate_failed": native_profile_gate_failed,
        "common_profile_gate_failed": common_profile_gate_failed,
        "profile_mapping_is_discriminating": (
            profile_mapping_is_discriminating
        ),
        "native_anchor_inconsistency_established": (
            native_anchor_inconsistency_established
        ),
        "anchor_hypothesis_supported": anchor_hypothesis_supported,
        "stable_common_first_cell_groups": stable_groups,
        "authorized_next": authorized_next,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "uses_production_generator": False,
        "uses_production_anchor_storage_derivative": False,
        "decisive_arrays_path": _relative(DECISIVE_ARRAYS),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "REJECTED" if not method_passed else "DIAGNOSTIC ONLY"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "generation_command": (
            "PYTHONPATH=src:scripts python3 "
            "scripts/run_causal_inner_monolithic_anchor_audit_"
            "wp10c9d6c1.py"
        ),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            _relative(PARENT_SUMMARY): _sha256(PARENT_SUMMARY),
            _relative(PARENT_ARRAYS): _sha256(PARENT_ARRAYS),
            _relative(PARENT_REPLAY_CONTEXTS): _sha256(
                PARENT_REPLAY_CONTEXTS
            ),
            _relative(PARENT_REPLAY_INPUTS): _sha256(
                PARENT_REPLAY_INPUTS
            ),
        },
        "establishes": (
            "Whether the grid-native anchors are continuum-consistent, "
            "whether one declared common lift changes the uniform physical "
            "export error direction, and whether a proper non-target "
            "first-cell group explains the direct inner-face error."
        ),
        "does_not_establish": (
            "A monolithic equilibrium background, embedded convergence, "
            "a nonlinear trajectory, production readiness, fixed-Q "
            "closure, or reduced slow evolution."
        ),
        "authorization_status": authorized_next,
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
