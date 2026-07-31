#!/usr/bin/env python3
"""Freeze the fixed-exterior direct-continuum embedded contract.

WP10c9d6c7c2c1 is definitions-only.  It preserves the certified b6e
uniform class, changes no operator, and propagates no state.  The package
maps the nine frozen arrival packets onto the unchanged one-way embedded
geometry and requires an independent fixed-N98-exterior/N513-N769-inner
reference before any embedded propagation.
"""

from __future__ import annotations

from dataclasses import replace
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

import run_causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d as b6d  # noqa: E402
import run_causal_inner_direct_continuum_uniform_recertification_wp10c9d6c7c2b6e as b6e  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_analytic_local_maps,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _cell_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_scattering_energy import (  # noqa: E402
    causal_c4_manufactured_primitive_state,
    causal_normalization_invariant_scattering_energy,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2c1"
ANALYZED_BASE_COMMIT = "d29cba5d5bfe5950cba4e458a6ff8b458e1364de"
ANALYZED_BASE_PARENT = "f6936282ab216e5e3db4ae04b981c544854dff87"
ANALYZED_BASE_TREE = "cd28b8cf456bf0326c8a75d68486e8deb2beb646"

PARENT_CELLS = 98
PARENT_COUPLING_FACE = 49
REFINEMENT_RATIOS = (1, 2, 4)
LAYOUT_LABELS = {
    1: "N98_outer_N98_inner_f49",
    2: "N98_outer_N196_inner_f49",
    4: "N98_outer_N392_inner_f49",
}
UNIFORM_MATCHES = {
    1: "uniform_N98",
    2: "uniform_N196",
    4: "uniform_N392",
}
COMMON_PARENT_FACES = (0, 6, 49, 52, 92, 95, 98)
SOURCE_BAND_PARENT_FACES = (52, 95)
RECEIVING_BAND_PARENT_FACES = (6, 49)
UPSTREAM_DIAGNOSTIC_PARENT_FACE = 92
BASES = b6d.BINDING_BASES
AMPLITUDE_FACTORS = b6d.AMPLITUDE_FACTORS
SIGNS = b6d.SIGNS
REFERENCE_LEVELS = b6d.LEVELS
CONTINUUM_NODES = b6d.CONTINUUM_NODES

MAXIMUM_GRID_REPLAY_DEFECT = 0.0
MAXIMUM_EXTERIOR_REPLAY_DEFECT = 2.0e-12
MAXIMUM_PARENT_PACKET_RESTRICTION_DEFECT = 2.0e-12
MAXIMUM_INITIAL_INNER_PACKET_NORM = 0.0
MAXIMUM_PROJECTOR_ALGEBRA_DEFECT = 2.0e-9
MAXIMUM_EQUIVALENT_PROJECTOR_DEFECT = 2.0e-8
MINIMUM_CHARACTERISTIC_GAP = 1.0e-6
MAXIMUM_EIGENVECTOR_CONDITION = 1.0e10
MINIMUM_ENERGY_EIGENVALUE = 0.0

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_direct_continuum_embedded_manifest_"
    "wp10c9d6c7c2c1.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_direct_continuum_embedded_manifest_"
    "wp10c9d6c7c2c1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_DIRECT_CONTINUUM_EMBEDDED_MANIFEST_"
    "WP10C9D6C7C2C1_RESULTS_2026-07-30.md"
)

B6E_DIRECTORY = b6e.CANONICAL_DIRECTORY
B6D_DIRECTORY = b6d.CANONICAL_DIRECTORY
C2A2_DIRECTORY = c2a2.CANONICAL_DIRECTORY
C2A3_DIRECTORY = c2a3.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_embedded_manifest_wp10c9d6c7c2c1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "embedded_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (THIS_RUNNER, THIS_TEST)


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


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return c2a._sha256(path)


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _mapped_face(parent_face: int, ratio: int) -> int:
    face = int(parent_face)
    if face <= PARENT_COUPLING_FACE:
        return face * ratio
    return PARENT_COUPLING_FACE * ratio + (
        face - PARENT_COUPLING_FACE
    )


def _validate_inputs() -> tuple[
    dict,
    dict,
    dict,
    dict,
    dict[str, np.ndarray],
]:
    b6e_summary = _read_json(B6E_DIRECTORY / "summary.json")
    b6d_manifest = _read_json(B6D_DIRECTORY / "contract_manifest.json")
    c2a2_summary = _read_json(C2A2_DIRECTORY / "summary.json")
    c2a3_manifest = _read_json(C2A3_DIRECTORY / "scope_manifest.json")
    b6d_arrays = _load_npz(B6D_DIRECTORY / "decisive_arrays.npz")
    if (
        b6e_summary["classification"]
        != "direct_continuum_uniform_arrival_class_certified_"
        "embedded_manifest_authorized"
        or not b6e_summary["passed"]
        or not b6e_summary["binding_decision"][
            "uniform_direct_continuum_class_certified"
        ]
        or not b6e_summary["binding_decision"][
            "definitions_only_embedded_manifest_authorized"
        ]
        or b6e_summary["binding_decision"][
            "embedded_propagation_authorized"
        ]
        or b6e_summary["authorized_next"]
        != "WP10c9d6c7c2c1_direct_continuum_embedded_manifest"
    ):
        raise RuntimeError("WP10c9d6c7c2b6e binding status changed")
    if (
        c2a2_summary["classification"]
        != "manufactured_interface_patch_rejected_"
        "unidirectional_characteristic_core"
        or c2a3_manifest["classification"]
        != "one_way_physical_core_scattering_scope_frozen_"
        "uniform_validation_authorized"
        or not c2a3_manifest["packet_and_window_contract"][
            "all_characteristics_inward_over_patch"
        ]
        or c2a3_manifest["packet_and_window_contract"][
            "reflection_coefficient_defined"
        ]
        or b6d_manifest["profile_manifest"]["binding_base_count"] != 9
        or b6d_manifest["profile_manifest"]["binding_variant_count"] != 36
        or tuple(b6d_arrays["reference_levels"]) != REFERENCE_LEVELS
        or tuple(b6d_arrays["continuum_nodes"]) != CONTINUUM_NODES
    ):
        raise RuntimeError("embedded geometry or frozen profile scope changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c2c1 analyzed identity changed")
    return (
        b6e_summary,
        b6d_manifest,
        c2a2_summary,
        c2a3_manifest,
        b6d_arrays,
    )


def _local_energy_audit(
    context,
    charts: np.ndarray,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    speeds = []
    energy_metrics = []
    projectors = []
    minimum_energy = math.inf
    maximum_condition = 0.0
    maximum_algebra = 0.0
    maximum_imaginary = 0.0
    minimum_gap = math.inf
    for radius, chart in zip(
        context.grid.centers, charts, strict=True
    ):
        _cell_state(context, float(radius), chart)
        maps = causal_five_field_analytic_local_maps(
            context, float(radius), chart
        )
        temporal = np.asarray(
            maps.temporal_storage_matrix, dtype=float
        )
        spatial = np.asarray(
            maps.physical_flux_jacobian
            - maps.shear_principal_source_matrix
            - maps.vertical_principal_source_matrix,
            dtype=float,
        )
        basis = causal_normalization_invariant_scattering_energy(
            temporal, spatial, field_scales
        )
        local_speeds = np.asarray(
            basis.characteristic_speeds, dtype=float
        )
        speeds.append(local_speeds)
        energy_metrics.append(basis.primitive_energy_metric)
        projectors.append(basis.primitive_projectors)
        minimum_energy = min(
            minimum_energy, basis.minimum_energy_eigenvalue
        )
        maximum_condition = max(
            maximum_condition, basis.eigenvector_condition_number
        )
        maximum_algebra = max(
            maximum_algebra,
            basis.maximum_projector_identity_defect,
            basis.maximum_projector_idempotence_defect,
            basis.maximum_cross_projector_defect,
            basis.maximum_energy_orthogonality_defect,
            basis.maximum_symmetrizer_defect,
            basis.maximum_eigenpair_defect,
            basis.maximum_rescaling_invariance_defect,
        )
        maximum_imaginary = max(
            maximum_imaginary, basis.maximum_imaginary_part
        )
        minimum_gap = min(
            minimum_gap,
            float(np.min(np.abs(np.diff(local_speeds)))),
        )
    speed_array = np.asarray(speeds)
    energy_array = np.asarray(energy_metrics)
    projector_array = np.asarray(projectors)
    all_inward = bool(np.max(speed_array) < 0.0)
    passed = bool(
        all_inward
        and minimum_energy > MINIMUM_ENERGY_EIGENVALUE
        and minimum_gap >= MINIMUM_CHARACTERISTIC_GAP
        and maximum_condition <= MAXIMUM_EIGENVECTOR_CONDITION
        and maximum_algebra <= MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
        and maximum_imaginary <= 1.0e-12
    )
    return {
        "passed": passed,
        "all_characteristics_inward": all_inward,
        "minimum_characteristic_speed_over_c": float(
            np.min(speed_array)
        ),
        "maximum_characteristic_speed_over_c": float(
            np.max(speed_array)
        ),
        "minimum_characteristic_gap": minimum_gap,
        "minimum_energy_eigenvalue": minimum_energy,
        "maximum_eigenvector_condition_number": maximum_condition,
        "maximum_projector_or_energy_algebra_defect": maximum_algebra,
        "maximum_imaginary_part": maximum_imaginary,
    }, {
        "characteristic_speeds_over_c": speed_array,
        "primitive_energy_metrics": energy_array,
        "normalization_invariant_projectors": projector_array,
    }


def _band_energy(
    packet: np.ndarray,
    metric: np.ndarray,
    widths: np.ndarray,
) -> float:
    return float(
        0.5
        * np.einsum(
            "ni,nij,nj,n->",
            packet,
            metric,
            packet,
            widths,
            optimize=True,
        )
    )


def _family_energy(
    packet: np.ndarray,
    projectors: np.ndarray,
    metric: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    values = []
    for family in range(projectors.shape[1]):
        projected = np.einsum(
            "nij,nj->ni",
            projectors[:, family],
            packet,
            optimize=True,
        )
        values.append(_band_energy(projected, metric, widths))
    return np.asarray(values)


def _build_layouts_and_profiles(
    b6d_manifest: dict,
    c2a3_manifest: dict,
    b6d_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    (
        _geometry_summary,
        _geometry_manifest,
        _geometry_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    parent_arrays = _load_npz(C2A2_DIRECTORY / "decisive_arrays.npz")
    parent_edges = np.asarray(parent_arrays["patch_edges"], dtype=float)
    committed_parent_charts = np.asarray(
        parent_arrays["manufactured_primitive_charts"], dtype=float
    )
    if (
        parent_edges.shape != (PARENT_CELLS + 1,)
        or committed_parent_charts.shape != (PARENT_CELLS, 5)
        or not np.array_equal(
            parent_edges, np.asarray(b6d_arrays["patch_edges_N98"])
        )
    ):
        raise RuntimeError("parent manufactured patch changed")

    parent_patch = c2a2._build_level(
        cells=PARENT_CELLS,
        base_edges=parent_edges,
        parent_context=parent_context,
        parent_base=parent_base,
        field_scales=field_scales,
    )
    parent_grid = parent_patch["grid"]
    parent_charts = np.asarray(
        parent_patch["extension"].primitive_charts, dtype=float
    )
    if _relative_defect(
        parent_charts, committed_parent_charts
    ) > 2.0e-12:
        raise RuntimeError("manufactured parent state replay changed")
    parent_log_spacing = float(
        np.mean(np.diff(np.log(parent_grid.edges)))
    )
    core_slice = c2a2.PARENT_CORE_CELLS
    profile_reports: dict[str, dict] = {
        name: {
            "role": b6d_manifest["profile_manifest"]["per_profile"][name][
                "role"
            ],
            "target_family_indices": b6d_manifest["profile_manifest"][
                "per_profile"
            ][name]["target_family_indices"],
            "layouts": {},
        }
        for name in BASES
    }
    layout_reports = {}
    decisive: dict[str, np.ndarray] = {
        "parent_patch_edges": parent_edges,
        "field_scales": np.asarray(field_scales, dtype=float),
        "refinement_ratios": np.asarray(
            REFINEMENT_RATIOS, dtype=np.int64
        ),
        "reference_levels": np.asarray(
            REFERENCE_LEVELS, dtype=np.int64
        ),
        "continuum_nodes": np.asarray(
            CONTINUUM_NODES, dtype=np.int64
        ),
        "common_parent_faces": np.asarray(
            COMMON_PARENT_FACES, dtype=np.int64
        ),
        "common_physical_face_radii": parent_edges[
            list(COMMON_PARENT_FACES)
        ],
        "source_band_parent_faces": np.asarray(
            SOURCE_BAND_PARENT_FACES, dtype=np.int64
        ),
        "receiving_band_parent_faces": np.asarray(
            RECEIVING_BAND_PARENT_FACES, dtype=np.int64
        ),
        "primary_time_samples_seconds": np.asarray(
            b6d_arrays.get(
                "primary_time_samples_seconds",
                _load_npz(
                    C2A3_DIRECTORY / "decisive_arrays.npz"
                )["primary_time_samples_seconds"],
            ),
            dtype=float,
        ),
    }
    minimum_target_fraction = math.inf
    maximum_partition = 0.0
    maximum_restriction = 0.0
    maximum_inner_norm = 0.0
    all_passed = True

    for ratio in REFINEMENT_RATIOS:
        label = LAYOUT_LABELS[ratio]
        layout = make_causal_embedded_patch_layout(
            parent_grid, PARENT_COUPLING_FACE, ratio
        )
        context = replace(
            parent_context,
            grid=layout.grid,
            stream_sources=None,
        ).validated()
        extension = causal_c4_manufactured_primitive_state(
            np.log(layout.grid.centers),
            np.log(parent_context.grid.centers[core_slice]),
            parent_base[core_slice],
            parent_base[0],
            parent_base[-1],
            transition_log_width=(
                c2a2.TRANSITION_PARENT_CELLS * parent_log_spacing
            ),
            field_scales=field_scales,
        )
        charts = np.asarray(extension.primitive_charts, dtype=float)
        local_report, local_arrays = _local_energy_audit(
            context, charts, field_scales
        )
        coupling = int(layout.coupling_face_index)
        grid_replay = float(
            np.max(
                np.abs(
                    layout.grid.edges[coupling:]
                    - parent_edges[PARENT_COUPLING_FACE:]
                )
            )
        )
        exterior_replay = _relative_defect(
            charts[coupling:],
            parent_charts[PARENT_COUPLING_FACE:],
        )
        parent_replay = (
            _relative_defect(charts, parent_charts)
            if ratio == 1
            else None
        )
        common_map = {
            str(face): _mapped_face(face, ratio)
            for face in COMMON_PARENT_FACES
        }
        common_radius_defect = max(
            abs(
                float(layout.grid.edges[index])
                - float(parent_edges[int(face)])
            )
            for face, index in (
                (face, common_map[str(face)])
                for face in COMMON_PARENT_FACES
            )
        )
        layout_passed = bool(
            grid_replay <= MAXIMUM_GRID_REPLAY_DEFECT
            and exterior_replay <= MAXIMUM_EXTERIOR_REPLAY_DEFECT
            and (parent_replay is None or parent_replay <= 2.0e-12)
            and common_radius_defect == 0.0
            and local_report["passed"]
        )
        layout_reports[label] = {
            "passed": layout_passed,
            "refinement_ratio": ratio,
            "total_cells": layout.n_cells,
            "refined_inner_cells": layout.n_refined_cells,
            "fixed_outer_cells": (
                layout.n_cells - layout.n_refined_cells
            ),
            "coupling_face_index": coupling,
            "parent_coupling_face": PARENT_COUPLING_FACE,
            "matched_uniform_reference": UNIFORM_MATCHES[ratio],
            "grid_exterior_replay_defect": grid_replay,
            "primitive_exterior_replay_defect": exterior_replay,
            "ratio_one_parent_replay_defect": parent_replay,
            "common_face_radius_replay_defect": common_radius_defect,
            "common_parent_to_embedded_faces": common_map,
            "local_energy_and_causality": local_report,
        }
        decisive[f"{label}__grid_edges"] = np.asarray(
            layout.grid.edges, dtype=float
        )
        decisive[f"{label}__parent_cell_indices"] = np.asarray(
            layout.parent_cell_indices, dtype=np.int64
        )
        decisive[f"{label}__subcell_indices"] = np.asarray(
            layout.subcell_indices, dtype=np.int64
        )
        decisive[f"{label}__base_primitive_charts"] = charts
        decisive[
            f"{label}__characteristic_speeds_over_c"
        ] = local_arrays["characteristic_speeds_over_c"]

        widths = np.diff(np.log(layout.grid.edges))
        metric = local_arrays["primitive_energy_metrics"]
        projectors = local_arrays[
            "normalization_invariant_projectors"
        ]
        for name in BASES:
            parent_packet = np.asarray(
                b6d_arrays[f"packet__{name}"], dtype=float
            )
            packet = parent_packet[layout.parent_cell_indices]
            restricted = restrict_causal_embedded_patch_cell_averages(
                packet, layout
            )
            restriction = _relative_defect(restricted, parent_packet)
            inner_norm = float(np.max(np.abs(packet[:coupling])))
            total = _band_energy(packet, metric, widths)
            families = _family_energy(
                packet, projectors, metric, widths
            )
            target_indices = tuple(
                profile_reports[name]["target_family_indices"]
            )
            target_fraction = float(
                np.sum(families[list(target_indices)]) / total
            )
            partition = abs(float(np.sum(families)) - total) / total
            profile_passed = bool(
                restriction
                <= MAXIMUM_PARENT_PACKET_RESTRICTION_DEFECT
                and inner_norm <= MAXIMUM_INITIAL_INNER_PACKET_NORM
                and target_fraction >= 1.0 - 1.0e-9
                and partition <= MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
            )
            profile_reports[name]["layouts"][label] = {
                "passed": profile_passed,
                "parent_restriction_defect": restriction,
                "initial_inner_packet_norm": inner_norm,
                "initial_total_energy": total,
                "initial_target_family_fraction": target_fraction,
                "family_partition_relative_defect": partition,
                "packet_sha256": causal_array_sha256(packet),
            }
            decisive[f"{name}__{label}__packet"] = packet
            decisive[f"{name}__{label}__initial_family_energy"] = (
                families
            )
            minimum_target_fraction = min(
                minimum_target_fraction, target_fraction
            )
            maximum_partition = max(maximum_partition, partition)
            maximum_restriction = max(
                maximum_restriction, restriction
            )
            maximum_inner_norm = max(maximum_inner_norm, inner_norm)
            all_passed = bool(all_passed and profile_passed)
        all_passed = bool(all_passed and layout_passed)

    variants = np.asarray(
        [
            (BASES.index(name), amplitude, sign)
            for name in BASES
            for amplitude in AMPLITUDE_FACTORS
            for sign in SIGNS
        ],
        dtype=float,
    )
    decisive["binding_variant_table"] = variants
    profile_summary = {
        "passed": all(
            item["passed"]
            for report in profile_reports.values()
            for item in report["layouts"].values()
        ),
        "binding_base_count": len(BASES),
        "binding_variant_count": int(variants.shape[0]),
        "maximum_parent_restriction_defect": maximum_restriction,
        "maximum_initial_inner_packet_norm": maximum_inner_norm,
        "minimum_initial_target_family_fraction": (
            minimum_target_fraction
        ),
        "maximum_family_partition_relative_defect": maximum_partition,
        "per_profile": profile_reports,
    }
    decisive["travel_windows_seconds"] = np.asarray(
        [
            c2a3_manifest["packet_and_window_contract"][
                "downstream_windows_seconds"
            ][name]
            for name in (
                "acoustic",
                "shear",
                "mixed_shear_acoustic",
            )
        ],
        dtype=float,
    )
    return (
        {
            "passed": bool(
                all_passed
                and profile_summary["passed"]
                and variants.shape == (36, 3)
            ),
            "layouts": layout_reports,
        },
        profile_summary,
        decisive,
    )


def _manifest(
    b6e_summary: dict,
    b6d_manifest: dict,
    c2a2_summary: dict,
    c2a3_manifest: dict,
    layouts: dict,
    profiles: dict,
) -> dict:
    direct = b6d_manifest["direct_continuum_contract"]
    tier_i = c2a3_manifest["certification_tiers"][
        "tier_I_primary_physics"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "historical_classifications_preserved": {
            "WP10c9d6c7c2a2": c2a2_summary["classification"],
            "WP10c9d6c7c2a3": c2a3_manifest["classification"],
            "WP10c9d6c7c2b6d": b6d_manifest["classification"],
            "WP10c9d6c7c2b6e": b6e_summary["classification"],
        },
        "scope": {
            "physical_core": "exact_c2a2_one_way_interface_core",
            "route": "fixed_N98_outer_coarse_to_refined_inner",
            "all_characteristics_inward": True,
            "reflection_coefficient_defined": False,
            "upstream_contamination_reported_instead": True,
            "manufactured_extension_is_physical_background": False,
            "claim": (
                "operator-neutral linear embedded arrival and transfer "
                "method certification on the manufactured patch"
            ),
        },
        "layout_contract": {
            "parent_cells": PARENT_CELLS,
            "parent_coupling_face": PARENT_COUPLING_FACE,
            "refinement_ratios": list(REFINEMENT_RATIOS),
            "layout_labels": [
                LAYOUT_LABELS[ratio]
                for ratio in REFINEMENT_RATIOS
            ],
            "fixed_outer_parent_cells": 49,
            "refined_inner_cells": {
                LAYOUT_LABELS[ratio]: 49 * ratio
                for ratio in REFINEMENT_RATIOS
            },
            "total_cells": {
                LAYOUT_LABELS[ratio]: 49 * ratio + 49
                for ratio in REFINEMENT_RATIOS
            },
            "same_parent_faces_retained_exactly_outside_interface": True,
            "same_C4_manufactured_state_formula_at_every_center": True,
            "common_parent_faces": list(COMMON_PARENT_FACES),
            "source_band_parent_faces": list(
                SOURCE_BAND_PARENT_FACES
            ),
            "receiving_band_parent_faces": list(
                RECEIVING_BAND_PARENT_FACES
            ),
            "upstream_diagnostic_parent_face": (
                UPSTREAM_DIAGNOSTIC_PARENT_FACE
            ),
            "layout_preflight": layouts,
        },
        "profile_contract": {
            "base_profiles": list(BASES),
            "amplitude_factors": list(AMPLITUDE_FACTORS),
            "signs": list(SIGNS),
            "binding_variant_count": 36,
            "same_frozen_packet_formula_and_parent_array": True,
            "initial_support_is_entirely_in_fixed_outer_region": True,
            "inner_initial_state_is_exactly_zero": True,
            "no_shift_taper_or_reoptimization": True,
            "profile_preflight": profiles,
        },
        "matched_reference_contract": {
            "required_before_embedded_propagation": True,
            "reference_name": (
                "fixed_N98_exterior_driven_N769_inner_continuum"
            ),
            "secondary_reference": (
                "fixed_N98_exterior_driven_N513_inner_continuum"
            ),
            "fixed_exterior": {
                "cells": "parent_cells_49_through_97",
                "operator": (
                    "unchanged_complete_N98_semidiscrete_DAE_tangent"
                ),
                "initial_packet": "exact_frozen_outer_packet",
                "outer_boundary": "unchanged_uniform_N98_boundary",
                "interface_drive": (
                    "actual_five_component_shared_face_trace_and_flux "
                    "on the frozen travel_time_samples"
                ),
            },
            "continuum_inner": {
                "domain": "parent_faces_0_through_49",
                "primary_nodes": 769,
                "secondary_nodes": 513,
                "operator": (
                    "independent_complete_DAE_collocation_with_all "
                    "principal_descriptor_and_lower_source_blocks"
                ),
                "outer_interface_condition": (
                    "all_five_incoming_characteristic_data_from_the "
                    "fixed_N98_exterior_drive"
                ),
                "inner_boundary_condition": (
                    "causal_outflow_with_zero_incoming_characteristics"
                ),
                "energy_identity": (
                    "positive_total_and_target_energy_plus_exact "
                    "covariant_background_height_relaxation_and_other_work"
                ),
            },
            "mandatory_preflight_gates": {
                "maximum_N769_N513_action_difference": 2.0e-5,
                "maximum_reference_uncertainty_to_N392_embedded_error": 0.10,
                "maximum_ratio_one_outer_trace_replay_defect": 1.0e-10,
                "maximum_characteristic_boundary_closure_defect": 1.0e-10,
                "maximum_energy_and_covariant_work_ledger_defect": 1.0e-10,
                "incoming_interface_characteristic_count": 5,
                "incoming_inner_boundary_characteristic_count": 0,
                "restart_replay_required": True,
            },
            "uniform_controls": {
                "finite": [
                    "uniform_N98",
                    "uniform_N196",
                    "uniform_N392",
                ],
                "continuum": [
                    "uniform_N513",
                    "uniform_N769",
                ],
                "embedded_minus_matched_uniform_reported": True,
                "full_uniform_continuum_is_not_the_binding_"
                "fixed_exterior_reference": True,
            },
            "forbidden_shortcuts": [
                "do_not_call_N392_embedded_exact",
                "do_not_use_embedded_Richardson_extrapolate_as_truth",
                "do_not_use_full_uniform_N769_as_the_only_reference",
                "do_not_move_windows_from_observed_histories",
            ],
        },
        "tier_I_contract": {
            **tier_i,
            "state_comparison": (
                "conservative_restriction_to_the_common_N98_parent_grid"
            ),
            "active_prefix": "cells_strictly_inside_coupling_face",
            "minimum_RMS_order": 0.75,
            "minimum_maximum_order": 0.75,
            "minimum_significant_component_order": 0.75,
            "maximum_fine_normalized_difference": 0.05,
            "minimum_history_cosine": 0.90,
            "minimum_refinement_error_cosine": 0.90,
        },
        "tier_II_direct_continuum_contract": {
            **direct,
            "primary_reference": (
                "fixed_N98_exterior_driven_N769_inner_continuum"
            ),
            "secondary_reference": (
                "fixed_N98_exterior_driven_N513_inner_continuum"
            ),
            "finite_layouts": [
                LAYOUT_LABELS[ratio]
                for ratio in REFINEMENT_RATIOS
            ],
            "binding_observables": [
                "positive_total_receiving_band_energy_gain",
                "positive_target_receiving_band_energy_gain",
                "total_unit_shape_history",
                "target_unit_shape_history",
                "total_covariant_receiver_work",
                "target_covariant_receiver_work",
                "stored_energy_and_physical_work_balance",
            ],
            "raw_local_opposite_family_energy_binding_alone": False,
            "pairwise_embedded_error_direction_binding": False,
            "pairwise_embedded_error_direction_reported": True,
        },
        "interface_and_ledger_contract": {
            "one_shared_MJE_coupling_flux": True,
            "exact_coupling_face_telescoping": True,
            "direct_face_JVP_and_prefix_ledger_parity": True,
            "prefix_MJE_balance_at_every_common_face": True,
            "maximum_conservative_ledger_defect": 1.0e-12,
            "maximum_active_export_JVP_defect": 5.0e-6,
            "report_interface_state_and_traction": True,
            "pointwise_direction_binding_only_above_frozen_"
            "observability_floor": True,
            "report_upstream_contamination": True,
            "report_embedded_minus_matched_uniform_architecture_excess": True,
            "no_absolute_reflection_gate": True,
        },
        "uncertainty_contract": {
            **c2a3_manifest["uncertainty_and_observability"],
            "matched_reference_uncertainty_is_mandatory": True,
            "complete_history_nuisance_envelopes_required": True,
            "conservative_deterministic_sum_default": True,
        },
        "execution_order": [
            "c2c2_build_and_certify_fixed_exterior_continuum_reference",
            "c2c3_propagate_uniform_certified_profiles_on_embedded_layouts",
            "stop_on_first_failed_binding_profile",
        ],
        "decision_table": {
            "reference_preflight_fails": (
                "repair_reference_or_boundary_definition_before_embedded"
            ),
            "tier_I_and_direct_tier_II_all_pass": (
                "certify_declared_linear_embedded_arrival_class_and_"
                "authorize_definitions_only_nonlinear_manifest"
            ),
            "tier_I_passes_direct_tier_II_fails_with_contracting_errors": (
                "freeze_failure_and_run_direct_fixed_exterior_localization"
            ),
            "stable_interface_local_noncontracting_mechanism": (
                "authorize_interface_local_truncation_audit_only"
            ),
            "integrated_quantities_pass_pointwise_direction_fails": (
                "revise_future_pointwise_metric_not_operator"
            ),
            "projector_definition_not_robust": (
                "retain_total_energy_and_covariant_balance_and_keep_"
                "family_channel_noncertifying"
            ),
        },
        "hard_stops": [
            "do_not_relabel_any_c2b_or_b6_result",
            "do_not_change_the_operator_or_interface",
            "do_not_run_embedded_before_c2c2_reference_passes",
            "do_not_tune_packet_coefficients_windows_or_thresholds",
            "do_not_use_N1024_as_rescue",
            "do_not_start_nonlinear_fixed_Q_or_reduced_evolution",
        ],
        "binding_decision": {
            "uniform_direct_continuum_class_preserved": True,
            "layout_and_profile_preflight_passed": bool(
                layouts["passed"] and profiles["passed"]
            ),
            "matched_fixed_exterior_reference_required": True,
            "matched_fixed_exterior_reference_available_now": False,
            "fixed_exterior_reference_preflight_authorized": bool(
                layouts["passed"] and profiles["passed"]
            ),
            "embedded_propagation_authorized": False,
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": (
            "direct_continuum_embedded_contract_frozen_fixed_exterior_"
            "reference_preflight_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c7c2c2_fixed_exterior_continuum_reference_preflight"
        ),
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(manifest)
    return manifest


def _input_hashes() -> dict[str, str]:
    paths = (
        B6E_DIRECTORY / "summary.json",
        B6E_DIRECTORY / "config.json",
        B6E_DIRECTORY / "decisive_arrays.npz",
        B6D_DIRECTORY / "contract_manifest.json",
        B6D_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "method_manifest.json",
        C2A2_DIRECTORY / "decisive_arrays.npz",
        C2A3_DIRECTORY / "scope_manifest.json",
        C2A3_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
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
        if not case.is_dir():
            continue
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
    summary = _read_json(CANONICAL_SUMMARY)
    summary.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _write_report(summary: dict) -> None:
    layouts = summary["layout_preflight"]["layouts"]
    lines = [
        "# Causal inner direct-continuum embedded manifest "
        "WP10c9d6c7c2c1",
        "",
        "## Result",
        "",
        "This definitions-only package passes. It changes no operator and "
        "propagates no state.",
        "",
        "The certified b6e uniform class is preserved. The fixed N98 "
        "incident exterior is not refined by the embedded ladder, so the "
        "full-domain N769 history is not used as the sole binding reference.",
        "",
        "## Frozen embedded layouts",
        "",
        "| Layout | Inner cells | Fixed outer cells | Total |",
        "|---|---:|---:|---:|",
    ]
    for ratio in REFINEMENT_RATIOS:
        label = LAYOUT_LABELS[ratio]
        item = layouts[label]
        lines.append(
            f"| `{label}` | `{item['refined_inner_cells']}` | "
            f"`{item['fixed_outer_cells']}` | `{item['total_cells']}` |"
        )
    lines.extend(
        [
            "",
            "All nine frozen packet bases remain exactly zero in the inner "
            "region initially, replay the fixed outer cells, and restrict "
            "back to the same N98 parent packet. All local characteristics "
            "remain inward on every layout.",
            "",
            "## Required matched reference",
            "",
            "Before embedded propagation, c2c2 must couple the unchanged "
            "N98 exterior semidiscrete drive to independent N769/N513 "
            "complete-DAE inner continuum solves. This isolates inner "
            "refinement while preserving the incident discretization.",
            "",
            "The uniform N98/N196/N392 and N513/N769 results remain matched "
            "controls. An embedded level, an embedded Richardson "
            "extrapolate, or the full-domain N769 history may not be called "
            "the fixed-exterior truth.",
            "",
            "## Decision",
            "",
            "Only the fixed-exterior reference preflight is authorized. "
            "Embedded propagation, numerical redesign, nonlinear evolution, "
            "fixed-Q experiments, and reduced evolution remain blocked.",
            "",
            "Classification:",
            "",
            f"`{summary['classification']}`",
            "",
            "Authorized next:",
            "",
            f"`{summary['authorized_next']}`",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    (
        b6e_summary,
        b6d_manifest,
        c2a2_summary,
        c2a3_manifest,
        b6d_arrays,
    ) = _validate_inputs()
    layouts, profiles, decisive = _build_layouts_and_profiles(
        b6d_manifest, c2a3_manifest, b6d_arrays
    )
    if not layouts["passed"] or not profiles["passed"]:
        raise RuntimeError("embedded layout/profile preflight failed")
    manifest = _manifest(
        b6e_summary,
        b6d_manifest,
        c2a2_summary,
        c2a3_manifest,
        layouts,
        profiles,
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "parent_cells": PARENT_CELLS,
        "parent_coupling_face": PARENT_COUPLING_FACE,
        "refinement_ratios": list(REFINEMENT_RATIOS),
        "layout_labels": [
            LAYOUT_LABELS[ratio] for ratio in REFINEMENT_RATIOS
        ],
        "common_parent_faces": list(COMMON_PARENT_FACES),
        "binding_bases": list(BASES),
        "binding_variant_count": 36,
        "reference_levels": list(REFERENCE_LEVELS),
        "continuum_nodes": list(CONTINUUM_NODES),
        "authorized_next": manifest["authorized_next"],
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).is_file()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "embedded_or_nonlinear_propagation_executed": False,
        "uniform_parent_classification": b6e_summary["classification"],
        "historical_classifications_preserved": manifest[
            "historical_classifications_preserved"
        ],
        "layout_preflight": layouts,
        "profile_preflight": profiles,
        "binding_decision": manifest["binding_decision"],
        "manifest_sha256": manifest["manifest_sha256"],
        "classification": manifest["classification"],
        "authorized_next": manifest["authorized_next"],
        "passed": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "embedded_manifest_file_sha256": _sha256(MANIFEST_PATH),
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
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "DIAGNOSTIC ONLY",
        "classification": summary["classification"],
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "source_parent_tree": ANALYZED_BASE_TREE,
        "implementation_worktree_head": _git_value("rev-parse", "HEAD"),
        "implementation_source_hashes": source_hashes,
        "input_hashes": _input_hashes(),
        "command": (
            "PYTHONPATH=src python "
            "scripts/"
            "run_causal_inner_direct_continuum_embedded_manifest_"
            "wp10c9d6c7c2c1.py"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    _write_json(PROVENANCE_PATH, provenance)
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
                "binding_decision": summary["binding_decision"],
                "authorized_next": summary["authorized_next"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
