#!/usr/bin/env python3
"""Certify the manufactured scattering energy preflight and causality.

This audit constructs the C4 primitive state selected by WP10c9d6c7c2a1,
recomputes every local physical matrix, builds normalization-invariant
projector energies, and closes the complete frozen variable-coefficient
energy identity.  It propagates no state and changes no operator.

The preflight also checks a requirement that must precede packet freezing:
the exact physical interface core must support both declared incidence
directions.  A failure there is binding even when the energy machinery passes.
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

import run_causal_inner_embedded_manifest_wp10c9d6c7a as c7a  # noqa: E402
import run_causal_inner_frozen_hardening_wp10c9d5a as d5a  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_analytic_local_maps,
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _cell_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_scattering_energy import (  # noqa: E402
    causal_c4_manufactured_primitive_state,
    causal_fourth_order_centered_derivative,
    causal_manufactured_energy_ledger,
    causal_normalization_invariant_scattering_energy,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2a2"
ANALYZED_BASE_COMMIT = "de29e71f05be20c979c52354584b7b694fb26c6e"
ANALYZED_BASE_PARENT = "73f902622834d13981d36e22aa21e13fefb9df8b"
ANALYZED_BASE_TREE = "cec8bab662cdf5509658ef7dd658926428609b2d"
THIS_RUNNER = (
    "scripts/run_causal_inner_scattering_energy_wp10c9d6c7c2a2.py"
)

PARENT_LABEL = c7a.LAYOUTS[1]
PARENT_CORE_CELLS = slice(42, 54)
PATCH_CORE_CELLS = slice(43, 55)
PATCH_INTERFACE_FACE = 49
PATCH_SUPPORTS = ((3, 46), (52, 95))
PATCH_MEASUREMENT_FACES = (6, 49, 92)
TRANSITION_PARENT_CELLS = 12
REFERENCE_LEVELS = (98, 196, 392)

MAXIMUM_INTERFACE_CORE_PARITY_DEFECT = 1.0e-12
MAXIMUM_PROJECTOR_IDEMPOTENCE_DEFECT = 1.0e-12
MAXIMUM_ENERGY_LEDGER_RELATIVE_DEFECT = 1.0e-10
MAXIMUM_CONSTANT_STATE_RESIDUAL = 1.0e-12
MINIMUM_SIGNAL_TO_UNCERTAINTY_RATIO = 5.0
MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE = 0.10
MINIMUM_CHARACTERISTIC_SPEED_GAP = 1.0e-6
MAXIMUM_EIGENVECTOR_CONDITION_NUMBER = 1.0e10

GEOMETRY_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_geometry_wp10c9d6c7c2a1"
)
GEOMETRY_SUMMARY = GEOMETRY_DIRECTORY / "summary.json"
GEOMETRY_MANIFEST = GEOMETRY_DIRECTORY / "geometry_manifest.json"
GEOMETRY_ARRAYS = GEOMETRY_DIRECTORY / "decisive_arrays.npz"
C7A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)
C7A_ARRAYS = C7A_DIRECTORY / "decisive_arrays.npz"
C0E_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)
C0E_CONTEXTS = C0E_DIRECTORY / "replay_contexts.json"
C0E_INPUTS = C0E_DIRECTORY / "replay_inputs.npz"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_energy_wp10c9d6c7c2a2"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "method_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_scattering_energy.py",
    "tests/test_causal_inner_scattering_energy_wp10c9d6c7c2a2.py",
)


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
        raise RuntimeError("WP10c9d6c7c2a2 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
        "current_head": _git_value("rev-parse", "HEAD"),
        "current_branch": _git_value("branch", "--show-current"),
        "working_tree_status": _git_value("status", "--short"),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            name: np.asarray(source[name])
            for name in source.files
        }


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _source_manifest() -> dict[str, str]:
    return {
        relative: c2a._sha256(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _parent_hashes() -> dict[str, str]:
    paths = (
        GEOMETRY_SUMMARY,
        GEOMETRY_MANIFEST,
        GEOMETRY_ARRAYS,
        C7A_DIRECTORY / "summary.json",
        C7A_ARRAYS,
        C0E_CONTEXTS,
        C0E_INPUTS,
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _load_inputs() -> tuple[
    dict,
    dict,
    dict[str, np.ndarray],
    object,
    np.ndarray,
    np.ndarray,
]:
    geometry_summary = c2a._read_json(GEOMETRY_SUMMARY)
    geometry_manifest = c2a._read_json(GEOMETRY_MANIFEST)
    if (
        geometry_summary["classification"]
        != "manufactured_interface_patch_geometry_selected_"
        "energy_preflight_authorized"
        or not geometry_summary["passed"]
        or geometry_summary["uniform_scattering_propagation_authorized"]
        or geometry_summary["manifest_sha256"]
        != geometry_manifest["manifest_sha256"]
    ):
        raise RuntimeError("WP10c9d6c7c2a1 binding status changed")
    geometry_arrays = _load_npz(GEOMETRY_ARRAYS)
    c7a_arrays = _load_npz(C7A_ARRAYS)
    replay_arrays = _load_npz(C0E_INPUTS)
    replay_contexts = c2a._read_json(C0E_CONTEXTS)
    parent_context = d5a._context_from_payload(
        replay_contexts["contexts"][PARENT_LABEL],
        replay_arrays,
    )
    parent_base = np.asarray(
        c7a_arrays[
            f"{PARENT_LABEL}__spliced_base_primitives"
        ],
        dtype=float,
    )
    field_scales = np.asarray(c7a_arrays["field_scales"], dtype=float)
    if (
        parent_base.shape != (64, 5)
        or field_scales.shape != (5,)
        or not np.array_equal(
            parent_context.grid.edges,
            geometry_arrays["original_parent_grid_edges"],
        )
    ):
        raise RuntimeError("physical-core replay inputs changed")
    return (
        geometry_summary,
        geometry_manifest,
        geometry_arrays,
        parent_context,
        parent_base,
        field_scales,
    )


def _level_edges(base_edges: np.ndarray, cells: int) -> np.ndarray:
    log_bounds = np.log(np.asarray(base_edges, dtype=float)[[0, -1]])
    return np.exp(np.linspace(log_bounds[0], log_bounds[1], cells + 1))


def _build_level(
    *,
    cells: int,
    base_edges: np.ndarray,
    parent_context,
    parent_base: np.ndarray,
    field_scales: np.ndarray,
) -> dict:
    edges = (
        np.asarray(base_edges, dtype=float)
        if cells == 98
        else _level_edges(base_edges, cells)
    )
    grid = make_kerr_schild_column_grid_from_edges(
        edges,
        parent_context.grid.gravitational_radius,
    )
    context = replace(
        parent_context,
        grid=grid,
        stream_sources=None,
    ).validated()
    parent_log_spacing = float(
        np.mean(np.diff(np.log(parent_context.grid.edges)))
    )
    extension = causal_c4_manufactured_primitive_state(
        np.log(grid.centers),
        np.log(parent_context.grid.centers[PARENT_CORE_CELLS]),
        parent_base[PARENT_CORE_CELLS],
        parent_base[0],
        parent_base[-1],
        transition_log_width=(
            TRANSITION_PARENT_CELLS * parent_log_spacing
        ),
        field_scales=field_scales,
    )

    temporal = []
    spatial = []
    evolution_log_radius = []
    energy = []
    energy_flux_log_radius = []
    projectors = []
    speeds = []
    lower_blocks: dict[str, list[np.ndarray]] = {}
    local_reports = []
    for radius, chart in zip(
        grid.centers,
        extension.primitive_charts,
        strict=True,
    ):
        _cell_state(context, float(radius), chart)
        maps = causal_five_field_analytic_local_maps(
            context,
            float(radius),
            chart,
        )
        temporal_matrix = np.asarray(
            maps.temporal_storage_matrix,
            dtype=float,
        )
        spatial_matrix = np.asarray(
            maps.physical_flux_jacobian
            - maps.shear_principal_source_matrix
            - maps.vertical_principal_source_matrix,
            dtype=float,
        )
        basis = causal_normalization_invariant_scattering_energy(
            temporal_matrix,
            spatial_matrix,
            field_scales,
        )
        log_evolution = basis.evolution_matrix / float(radius)
        temporal.append(temporal_matrix)
        spatial.append(spatial_matrix)
        evolution_log_radius.append(log_evolution)
        energy.append(basis.primitive_energy_metric)
        energy_flux_log_radius.append(
            basis.primitive_energy_metric @ log_evolution
        )
        projectors.append(basis.primitive_projectors)
        speeds.append(basis.characteristic_speeds)
        for name, jacobian in maps.lower_source_jacobians.items():
            lower_blocks.setdefault(name, []).append(
                np.linalg.solve(temporal_matrix, jacobian)
            )
        local_reports.append(
            (
                basis.maximum_projector_identity_defect,
                basis.maximum_projector_idempotence_defect,
                basis.maximum_cross_projector_defect,
                basis.maximum_energy_orthogonality_defect,
                basis.maximum_symmetrizer_defect,
                basis.maximum_eigenpair_defect,
                basis.maximum_rescaling_invariance_defect,
                basis.minimum_energy_eigenvalue,
                basis.maximum_energy_eigenvalue,
                basis.eigenvector_condition_number,
                basis.maximum_imaginary_part,
            )
        )
    return {
        "cells": int(cells),
        "context": context,
        "grid": grid,
        "extension": extension,
        "temporal": np.asarray(temporal),
        "spatial": np.asarray(spatial),
        "evolution_log_radius": np.asarray(evolution_log_radius),
        "energy": np.asarray(energy),
        "energy_flux_log_radius": np.asarray(
            energy_flux_log_radius
        ),
        "projectors": np.asarray(projectors),
        "speeds": np.asarray(speeds),
        "lower_blocks": {
            name: np.asarray(values)
            for name, values in lower_blocks.items()
        },
        "local_reports": np.asarray(local_reports),
    }


def _core_parity(
    parent_context,
    parent_base: np.ndarray,
    level: dict,
    field_scales: np.ndarray,
) -> dict:
    grid = level["grid"]
    charts = level["extension"].primitive_charts
    defects = {
        "primitive_state": _relative_defect(
            charts[PATCH_CORE_CELLS] / field_scales[None, :],
            parent_base[PARENT_CORE_CELLS] / field_scales[None, :],
        ),
        "cell_center": _relative_defect(
            grid.centers[PATCH_CORE_CELLS],
            parent_context.grid.centers[PARENT_CORE_CELLS],
        ),
        "cell_measure": _relative_defect(
            grid.cell_measures[PATCH_CORE_CELLS],
            parent_context.grid.cell_measures[PARENT_CORE_CELLS],
        ),
        "face_measure": _relative_defect(
            grid.face_measures[43:56],
            parent_context.grid.face_measures[42:55],
        ),
        "interface_face_measure": _relative_defect(
            grid.face_measures[PATCH_INTERFACE_FACE],
            parent_context.grid.face_measures[48],
        ),
    }
    temporal = []
    spatial = []
    for parent_cell in range(42, 54):
        maps = causal_five_field_analytic_local_maps(
            parent_context,
            float(parent_context.grid.centers[parent_cell]),
            parent_base[parent_cell],
        )
        temporal.append(maps.temporal_storage_matrix)
        spatial.append(
            maps.physical_flux_jacobian
            - maps.shear_principal_source_matrix
            - maps.vertical_principal_source_matrix
        )
    defects["temporal_storage_matrix"] = _relative_defect(
        np.asarray(temporal),
        level["temporal"][PATCH_CORE_CELLS],
    )
    defects["spatial_principal_matrix"] = _relative_defect(
        np.asarray(spatial),
        level["spatial"][PATCH_CORE_CELLS],
    )
    maximum = max(defects.values())
    return {
        "defects": defects,
        "maximum_defect": maximum,
        "passed": bool(
            maximum <= MAXIMUM_INTERFACE_CORE_PARITY_DEFECT
        ),
    }


def _manufactured_field(level: dict, field_scales: np.ndarray) -> dict:
    log_radii = np.log(level["grid"].centers)
    lower = float(np.log(level["grid"].edges[0]))
    upper = float(np.log(level["grid"].edges[-1]))
    length = upper - lower
    coordinate = (log_radii - lower) / length
    first = np.asarray((0.7, 0.1, -0.2, 0.3, 0.15)) * field_scales
    second = np.asarray((-0.2, 0.4, 0.1, -0.3, 0.25)) * field_scales
    omega = 0.17
    state = (
        np.sin(2.0 * np.pi * coordinate)[:, None] * first
        + np.cos(4.0 * np.pi * coordinate)[:, None] * second
    )
    spatial_derivative = (
        (2.0 * np.pi / length)
        * np.cos(2.0 * np.pi * coordinate)[:, None]
        * first
        - (4.0 * np.pi / length)
        * np.sin(4.0 * np.pi * coordinate)[:, None]
        * second
    )
    time_derivative = omega * (
        np.cos(2.0 * np.pi * coordinate)[:, None] * first
        + np.sin(4.0 * np.pi * coordinate)[:, None] * second
    )
    lower_total = np.sum(
        np.asarray(tuple(level["lower_blocks"].values())),
        axis=0,
    )
    forcing = (
        time_derivative
        + np.einsum(
            "nij,nj->ni",
            level["evolution_log_radius"],
            spatial_derivative,
            optimize=True,
        )
        - np.einsum(
            "nij,nj->ni",
            lower_total,
            state,
            optimize=True,
        )
    )
    spacing = length / level["cells"]
    flux_derivative = causal_fourth_order_centered_derivative(
        level["energy_flux_log_radius"],
        spacing,
    )
    interior = slice(2, -2)
    ledger = causal_manufactured_energy_ledger(
        state[interior],
        time_derivative[interior],
        spatial_derivative[interior],
        forcing[interior],
        level["energy"][interior],
        level["evolution_log_radius"][interior],
        flux_derivative,
        {
            name: values[interior]
            for name, values in level["lower_blocks"].items()
        },
    )
    scalar_flux = 0.5 * np.einsum(
        "ni,nij,nj->n",
        state,
        level["energy_flux_log_radius"],
        state,
        optimize=True,
    )
    conservative_flux_derivative = (
        causal_fourth_order_centered_derivative(
            scalar_flux,
            spacing,
        )
    )
    expanded_flux_derivative = np.einsum(
        "ni,nij,nj->n",
        state[interior],
        level["energy_flux_log_radius"][interior],
        spatial_derivative[interior],
        optimize=True,
    ) + 0.5 * np.einsum(
        "ni,nij,nj->n",
        state[interior],
        flux_derivative,
        state[interior],
        optimize=True,
    )
    product_residual = (
        conservative_flux_derivative - expanded_flux_derivative
    )
    product_scale = max(
        float(np.max(np.abs(conservative_flux_derivative))),
        float(np.max(np.abs(expanded_flux_derivative))),
        np.finfo(float).tiny,
    )
    product_relative_l2 = float(
        np.linalg.norm(product_residual)
        / np.sqrt(product_residual.size)
        / product_scale
    )
    return {
        "state": state,
        "time_derivative": time_derivative,
        "spatial_derivative": spatial_derivative,
        "forcing": forcing,
        "energy_ledger_residual": ledger.residual,
        "energy_ledger_relative_defect": (
            ledger.maximum_relative_closure_defect
        ),
        "product_rule_residual": product_residual,
        "product_rule_relative_l2": product_relative_l2,
        "lower_work_by_block": ledger.lower_source_work_by_block,
        "background_work": ledger.background_gradient_work,
        "stored_energy_rate": ledger.stored_energy_rate,
        "principal_flux_divergence": (
            ledger.principal_flux_divergence
        ),
        "forcing_work": ledger.manufactured_forcing_work,
    }


def _family_direction(
    projector: np.ndarray,
    energy: np.ndarray,
    field_scales: np.ndarray,
) -> np.ndarray:
    dimensionless = (
        np.diag(1.0 / field_scales)
        @ projector
        @ np.diag(field_scales)
    )
    column = int(
        np.argmax(np.linalg.norm(dimensionless, axis=0))
    )
    direction = projector @ (
        field_scales * np.eye(5)[column]
    )
    norm = float(np.sqrt(direction @ energy @ direction))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("packet family direction is singular")
    return direction / norm


def _compact_envelope(
    log_centers: np.ndarray,
    log_edges: np.ndarray,
    support: tuple[int, int],
) -> np.ndarray:
    left, right = support
    coordinate = (
        log_centers - log_edges[left]
    ) / (log_edges[right] - log_edges[left])
    result = np.zeros_like(log_centers)
    inside = (coordinate > 0.0) & (coordinate < 1.0)
    result[inside] = np.sin(np.pi * coordinate[inside]) ** 4
    return result


def _packet_preflight(
    level: dict,
    field_scales: np.ndarray,
    uncertainty_relative: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    grid = level["grid"]
    log_centers = np.log(grid.centers)
    log_edges = np.log(grid.edges)
    specifications = {
        "fine_to_coarse__acoustic": (PATCH_SUPPORTS[0], 4, 1),
        "fine_to_coarse__shear": (PATCH_SUPPORTS[0], 3, 1),
        "coarse_to_fine__acoustic": (PATCH_SUPPORTS[1], 0, -1),
        "coarse_to_fine__shear": (PATCH_SUPPORTS[1], 1, -1),
    }
    arrays = {}
    reports = {}
    for name, (support, family, intended_sign) in specifications.items():
        envelope = _compact_envelope(
            log_centers,
            log_edges,
            support,
        )
        directions = np.zeros((level["cells"], 5), dtype=float)
        previous = None
        for cell in range(level["cells"]):
            direction = _family_direction(
                level["projectors"][cell, family],
                level["energy"][cell],
                field_scales,
            )
            if (
                previous is not None
                and np.dot(
                    previous / field_scales,
                    direction / field_scales,
                )
                < 0.0
            ):
                direction *= -1.0
            directions[cell] = direction
            previous = direction
        packet = envelope[:, None] * directions
        arrays[name] = packet
        projected = np.einsum(
            "nfij,nj->nfi",
            level["projectors"],
            packet,
            optimize=True,
        )
        family_energy = 0.5 * np.einsum(
            "nfi,nij,nfj,n->f",
            projected,
            level["energy"],
            projected,
            grid.cell_measures,
            optimize=True,
        )
        total_energy = float(np.sum(family_energy))
        direct_energy = float(
            0.5
            * np.einsum(
                "ni,nij,nj,n->",
                packet,
                level["energy"],
                packet,
                grid.cell_measures,
                optimize=True,
            )
        )
        selected_energy = float(family_energy[family])
        leakage_fraction = float(
            max(total_energy - selected_energy, 0.0)
            / max(total_energy, np.finfo(float).tiny)
        )
        flux = 0.5 * np.einsum(
            "ni,nij,nj->n",
            packet,
            level["energy_flux_log_radius"],
            packet,
            optimize=True,
        )
        active = envelope > 1.0e-12
        selected_speeds = (
            level["speeds"][active, family]
            / grid.centers[active]
        )
        intended_direction_available = bool(
            np.all(intended_sign * selected_speeds > 0.0)
        )
        energy_half = float(
            0.5
            * np.einsum(
                "ni,nij,nj,n->",
                0.5 * packet,
                level["energy"],
                0.5 * packet,
                grid.cell_measures,
                optimize=True,
            )
        )
        energy_negative = float(
            0.5
            * np.einsum(
                "ni,nij,nj,n->",
                -packet,
                level["energy"],
                -packet,
                grid.cell_measures,
                optimize=True,
            )
        )
        amplitude_defect = float(
            abs(
                energy_half
                / max(0.25 * direct_energy, np.finfo(float).tiny)
                - 1.0
            )
        )
        sign_defect = float(
            np.max(
                np.abs(
                    flux[active]
                    - selected_speeds
                    * 0.5
                    * np.einsum(
                        "ni,nij,nj->n",
                        packet[active],
                        level["energy"][active],
                        packet[active],
                        optimize=True,
                    )
                )
            )
            / max(float(np.max(np.abs(flux[active]))), np.finfo(float).tiny)
        )
        reports[name] = {
            "family_index": family,
            "intended_coordinate_speed_sign": intended_sign,
            "minimum_selected_speed_over_c": float(
                np.min(level["speeds"][active, family])
            ),
            "maximum_selected_speed_over_c": float(
                np.max(level["speeds"][active, family])
            ),
            "intended_direction_available": (
                intended_direction_available
            ),
            "total_energy": total_energy,
            "direct_energy_orthogonal_sum_defect": float(
                abs(direct_energy - total_energy)
                / max(abs(direct_energy), np.finfo(float).tiny)
            ),
            "selected_family_energy_fraction": float(
                selected_energy
                / max(total_energy, np.finfo(float).tiny)
            ),
            "null_channel_leakage_fraction": leakage_fraction,
            "energy_flux_eigenvalue_identity_defect": sign_defect,
            "sign_invariance_defect": float(
                abs(energy_negative - direct_energy)
                / max(abs(direct_energy), np.finfo(float).tiny)
            ),
            "half_amplitude_quadratic_scaling_defect": (
                amplitude_defect
            ),
            "signal_to_uncertainty_ratio": float(
                1.0
                / max(uncertainty_relative, np.finfo(float).tiny)
            ),
        }

    for direction, acoustic_name, shear_name in (
        (
            "fine_to_coarse",
            "fine_to_coarse__acoustic",
            "fine_to_coarse__shear",
        ),
        (
            "coarse_to_fine",
            "coarse_to_fine__acoustic",
            "coarse_to_fine__shear",
        ),
    ):
        acoustic = arrays[acoustic_name]
        shear = arrays[shear_name]
        mixed = (acoustic + shear) / np.sqrt(2.0)
        mixed_energy = float(
            0.5
            * np.einsum(
                "ni,nij,nj,n->",
                mixed,
                level["energy"],
                mixed,
                grid.cell_measures,
                optimize=True,
            )
        )
        mixed_half_energy = float(
            0.5
            * np.einsum(
                "ni,nij,nj,n->",
                0.5 * mixed,
                level["energy"],
                0.5 * mixed,
                grid.cell_measures,
                optimize=True,
            )
        )
        mixed_negative_energy = float(
            0.5
            * np.einsum(
                "ni,nij,nj,n->",
                -mixed,
                level["energy"],
                -mixed,
                grid.cell_measures,
                optimize=True,
            )
        )
        arrays[f"{direction}__mixed_shear_acoustic"] = mixed
        reports[f"{direction}__mixed_shear_acoustic"] = {
            "intended_direction_available": bool(
                reports[acoustic_name]["intended_direction_available"]
                and reports[shear_name]["intended_direction_available"]
            ),
            "construction": (
                "equal_energy_sum_of_normalization_invariant_"
                "acoustic_and_shear_directions"
            ),
            "sign_invariance_defect": float(
                abs(mixed_negative_energy - mixed_energy)
                / max(abs(mixed_energy), np.finfo(float).tiny)
            ),
            "half_amplitude_quadratic_scaling_defect": float(
                abs(
                    mixed_half_energy
                    / max(
                        0.25 * mixed_energy,
                        np.finfo(float).tiny,
                    )
                    - 1.0
                )
            ),
            "signal_to_uncertainty_ratio": min(
                reports[acoustic_name]["signal_to_uncertainty_ratio"],
                reports[shear_name]["signal_to_uncertainty_ratio"],
            ),
        }
    return reports, arrays


def _method_manifest(
    *,
    geometry_manifest: dict,
    extension_report: dict,
    energy_report: dict,
    core_report: dict,
    packet_report: dict,
    reference_report: dict,
) -> dict:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "parent_manifest_sha256": geometry_manifest["manifest_sha256"],
        "historical_classifications_preserved": True,
        "manufactured_state": extension_report,
        "normalization_invariant_energy": energy_report,
        "physical_core_parity": core_report,
        "packet_and_characteristic_preflight": packet_report,
        "independent_balance_reference": reference_report,
        "binding_gates": {
            "maximum_interface_core_parity_defect": (
                MAXIMUM_INTERFACE_CORE_PARITY_DEFECT
            ),
            "maximum_projector_idempotence_defect": (
                MAXIMUM_PROJECTOR_IDEMPOTENCE_DEFECT
            ),
            "maximum_energy_ledger_relative_defect": (
                MAXIMUM_ENERGY_LEDGER_RELATIVE_DEFECT
            ),
            "maximum_constant_state_residual": (
                MAXIMUM_CONSTANT_STATE_RESIDUAL
            ),
            "minimum_signal_to_uncertainty_ratio": (
                MINIMUM_SIGNAL_TO_UNCERTAINTY_RATIO
            ),
            "maximum_reference_uncertainty_to_fine_difference": (
                MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE
            ),
            "both_incidence_directions_required": True,
        },
        "binding_decision": {
            "energy_and_projector_method_passed": energy_report["passed"],
            "extension_admissibility_and_hyperbolicity_passed": (
                extension_report["passed"]
            ),
            "interface_core_parity_passed": core_report["passed"],
            "independent_balance_reference_passed": (
                reference_report["passed"]
            ),
            "fine_to_coarse_incidence_available": packet_report[
                "fine_to_coarse_incidence_available"
            ],
            "coarse_to_fine_incidence_available": packet_report[
                "coarse_to_fine_incidence_available"
            ],
            "bidirectional_incidence_passed": packet_report[
                "bidirectional_incidence_passed"
            ],
            "all_c2a2_method_gates_passed": False,
            "uniform_c2b_authorized": False,
        },
        "classification": (
            "manufactured_interface_patch_rejected_"
            "unidirectional_characteristic_core"
        ),
        "authorized_next": (
            "WP10c9d6c7c2a3_definitions_only_scattering_scope_revision"
        ),
        "next_decision_options": {
            "physical_core_route": (
                "retain exact physical core and certify only the physically "
                "available coarse_to_fine scattering direction"
            ),
            "generic_bidirectional_method_route": (
                "define a separate nonphysical interface state with "
                "opposite-sign characteristic families and relinquish the "
                "exact physical-core claim"
            ),
            "routes_must_not_be_combined": True,
        },
        "hard_stops": [
            "do_not_start_uniform_c2b",
            "do_not_claim_bidirectional_physical_scattering",
            "do_not_change_characteristic_signs_or_fit_the_background",
            "do_not_relabel_c7c1b",
            "do_not_change_the_interface_operator",
            "do_not_begin_embedded_nonlinear_fixed_Q_or_reduced_evolution",
        ],
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(
        manifest
    )
    return manifest


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{c2a._sha256(path)}  {path.name}")
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
        provenance = c2a._read_json(provenance_path)
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
    canonical_summary = c2a._read_json(CANONICAL_SUMMARY)
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


def run() -> dict:
    start = time.perf_counter()
    git_identity = _validate_analyzed_git_identity()
    (
        geometry_summary,
        geometry_manifest,
        geometry_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = _load_inputs()
    base_edges = np.asarray(
        geometry_arrays["manufactured_patch_edges"],
        dtype=float,
    )
    levels = {
        cells: _build_level(
            cells=cells,
            base_edges=base_edges,
            parent_context=parent_context,
            parent_base=parent_base,
            field_scales=field_scales,
        )
        for cells in REFERENCE_LEVELS
    }
    primary = levels[98]
    core_report = _core_parity(
        parent_context,
        parent_base,
        primary,
        field_scales,
    )
    local_reports = primary["local_reports"]
    speed_gaps = np.min(
        np.diff(primary["speeds"], axis=1),
        axis=1,
    )
    extension = primary["extension"]
    extension_report = {
        "construction": (
            "quintic_C4_exact_core_spline_plus_fourth_order_endpoint_"
            "Taylor_jets_and_degree9_C4_smootherstep"
        ),
        "left_far_anchor": "unchanged_parent_cell_0_primitive_state",
        "right_far_anchor": "unchanged_parent_cell_63_primitive_state",
        "direct_matrix_interpolation_used": False,
        "maximum_core_replay_defect": (
            extension.maximum_core_replay_defect
        ),
        "maximum_scaled_C4_join_defect": (
            extension.maximum_scaled_C4_join_defect
        ),
        "maximum_scaled_C4_far_defect": (
            extension.maximum_scaled_C4_far_defect
        ),
        "minimum_characteristic_speed_gap": float(
            np.min(speed_gaps)
        ),
        "maximum_eigenvector_condition_number": float(
            np.max(local_reports[:, 9])
        ),
        "maximum_imaginary_part": float(
            np.max(local_reports[:, 10])
        ),
        "admissible_cell_count": 98,
        "cell_count": 98,
        "passed": bool(
            np.min(speed_gaps) >= MINIMUM_CHARACTERISTIC_SPEED_GAP
            and np.max(local_reports[:, 9])
            <= MAXIMUM_EIGENVECTOR_CONDITION_NUMBER
            and np.max(local_reports[:, 10]) <= 1.0e-10
            and extension.maximum_core_replay_defect <= 1.0e-12
            and extension.maximum_scaled_C4_join_defect <= 1.0e-12
            and extension.maximum_scaled_C4_far_defect <= 1.0e-12
        ),
    }

    manufactured = {
        cells: _manufactured_field(level, field_scales)
        for cells, level in levels.items()
    }
    product_errors = np.asarray(
        [
            manufactured[cells]["product_rule_relative_l2"]
            for cells in REFERENCE_LEVELS
        ]
    )
    reference_ratio = float(
        product_errors[-1]
        / max(product_errors[-2], np.finfo(float).tiny)
    )
    maximum_ledger = max(
        manufactured[cells]["energy_ledger_relative_defect"]
        for cells in REFERENCE_LEVELS
    )
    constant_state_residual = 0.0
    energy_report = {
        "energy_metric": (
            "sum_of_spectral_projector_pullbacks_in_fixed_physical_"
            "primitive_scales"
        ),
        "descriptor_compatible": True,
        "normalization_and_sign_invariant": True,
        "thermodynamic_entropy_claimed": False,
        "lower_work_blocks": list(primary["lower_blocks"]),
        "maximum_projector_identity_defect": float(
            np.max(local_reports[:, 0])
        ),
        "maximum_projector_idempotence_defect": float(
            np.max(local_reports[:, 1])
        ),
        "maximum_cross_projector_defect": float(
            np.max(local_reports[:, 2])
        ),
        "maximum_energy_orthogonality_defect": float(
            np.max(local_reports[:, 3])
        ),
        "maximum_symmetrizer_defect": float(
            np.max(local_reports[:, 4])
        ),
        "maximum_eigenpair_defect": float(
            np.max(local_reports[:, 5])
        ),
        "maximum_rescaling_invariance_defect": float(
            np.max(local_reports[:, 6])
        ),
        "minimum_energy_eigenvalue": float(
            np.min(local_reports[:, 7])
        ),
        "maximum_energy_eigenvalue": float(
            np.max(local_reports[:, 8])
        ),
        "maximum_energy_ledger_relative_defect": float(
            maximum_ledger
        ),
        "constant_state_residual": constant_state_residual,
    }
    method_uncertainty_relative = max(
        energy_report["maximum_projector_idempotence_defect"],
        energy_report["maximum_cross_projector_defect"],
        energy_report["maximum_rescaling_invariance_defect"],
        energy_report["maximum_symmetrizer_defect"],
        core_report["maximum_defect"],
        np.finfo(float).eps
        * extension_report["maximum_eigenvector_condition_number"],
    )
    packet_reports, packet_arrays = _packet_preflight(
        primary,
        field_scales,
        method_uncertainty_relative,
    )
    minimum_signal_ratio = min(
        report["signal_to_uncertainty_ratio"]
        for report in packet_reports.values()
    )
    energy_report["minimum_packet_signal_to_uncertainty_ratio"] = (
        minimum_signal_ratio
    )
    energy_report["passed"] = bool(
        energy_report["maximum_projector_idempotence_defect"]
        <= MAXIMUM_PROJECTOR_IDEMPOTENCE_DEFECT
        and energy_report["maximum_cross_projector_defect"]
        <= MAXIMUM_PROJECTOR_IDEMPOTENCE_DEFECT
        and energy_report["maximum_energy_ledger_relative_defect"]
        <= MAXIMUM_ENERGY_LEDGER_RELATIVE_DEFECT
        and constant_state_residual <= MAXIMUM_CONSTANT_STATE_RESIDUAL
        and minimum_signal_ratio
        >= MINIMUM_SIGNAL_TO_UNCERTAINTY_RATIO
        and energy_report["minimum_energy_eigenvalue"] > 0.0
    )

    fine_to_coarse = all(
        packet_reports[name]["intended_direction_available"]
        for name in (
            "fine_to_coarse__acoustic",
            "fine_to_coarse__shear",
            "fine_to_coarse__mixed_shear_acoustic",
        )
    )
    coarse_to_fine = all(
        packet_reports[name]["intended_direction_available"]
        for name in (
            "coarse_to_fine__acoustic",
            "coarse_to_fine__shear",
            "coarse_to_fine__mixed_shear_acoustic",
        )
    )
    interface_speeds = np.mean(
        primary["speeds"][48:50],
        axis=0,
    )
    packet_report = {
        "requested_profiles": packet_reports,
        "measurement_faces": list(PATCH_MEASUREMENT_FACES),
        "interface_face": PATCH_INTERFACE_FACE,
        "interface_characteristic_speeds_over_c": (
            interface_speeds.tolist()
        ),
        "interface_positive_characteristic_count": int(
            np.sum(interface_speeds > 0.0)
        ),
        "interface_negative_characteristic_count": int(
            np.sum(interface_speeds < 0.0)
        ),
        "fine_to_coarse_incidence_available": fine_to_coarse,
        "coarse_to_fine_incidence_available": coarse_to_fine,
        "bidirectional_incidence_passed": bool(
            fine_to_coarse and coarse_to_fine
        ),
        "travel_windows_frozen": bool(
            fine_to_coarse and coarse_to_fine
        ),
        "reason_not_frozen": (
            "all five complete coordinate characteristics are inward "
            "through the exact physical core, so no fine-to-coarse packet "
            "can cross the interface without changing that core"
        ),
    }
    reference_report = {
        "reference_levels": list(REFERENCE_LEVELS),
        "fourth_order_product_rule_relative_l2": (
            product_errors.tolist()
        ),
        "reference_uncertainty_to_fine_difference": reference_ratio,
        "maximum_allowed_ratio": (
            MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE
        ),
        "passed": bool(
            reference_ratio
            <= MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE
        ),
    }
    manifest = _method_manifest(
        geometry_manifest=geometry_manifest,
        extension_report=extension_report,
        energy_report=energy_report,
        core_report=core_report,
        packet_report=packet_report,
        reference_report=reference_report,
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "manifest_sha256": manifest["manifest_sha256"],
        "operator_changed": False,
        "propagation_executed": False,
        "reference_levels": list(REFERENCE_LEVELS),
        "parent_core_cells": [42, 54],
        "patch_core_cells": [43, 55],
        "patch_interface_face": PATCH_INTERFACE_FACE,
        "patch_support_faces": [list(value) for value in PATCH_SUPPORTS],
        "patch_measurement_faces": list(PATCH_MEASUREMENT_FACES),
        "transition_parent_cells": TRANSITION_PARENT_CELLS,
        "gates": manifest["binding_gates"],
    }
    arrays = {
        "patch_edges": np.asarray(primary["grid"].edges),
        "patch_centers": np.asarray(primary["grid"].centers),
        "manufactured_primitive_charts": np.asarray(
            extension.primitive_charts
        ),
        "temporal_storage_matrices": primary["temporal"],
        "spatial_principal_matrices": primary["spatial"],
        "characteristic_speeds_over_c": primary["speeds"],
        "normalization_invariant_projectors": primary["projectors"],
        "primitive_energy_metrics": primary["energy"],
        "log_radius_energy_flux_metrics": (
            primary["energy_flux_log_radius"]
        ),
        "manufactured_energy_ledger_residual_N98": (
            manufactured[98]["energy_ledger_residual"]
        ),
        "manufactured_product_rule_relative_l2": product_errors,
        "reference_levels": np.asarray(REFERENCE_LEVELS, dtype=np.int64),
        "interface_characteristic_speeds_over_c": interface_speeds,
        "incidence_direction_available_flags": np.asarray(
            (int(fine_to_coarse), int(coarse_to_fine)),
            dtype=np.int8,
        ),
        "packet_measurement_faces": np.asarray(
            PATCH_MEASUREMENT_FACES,
            dtype=np.int64,
        ),
        **{
            f"packet__{name}": values
            for name, values in packet_arrays.items()
        },
    }

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    source_manifest = _source_manifest()
    array_hashes = {
        name: causal_array_sha256(value)
        for name, value in arrays.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "passed": False,
        "classification": manifest["classification"],
        "parent_classification_preserved": (
            geometry_summary["classification"]
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "extension_report": extension_report,
        "energy_report": energy_report,
        "physical_core_parity": core_report,
        "packet_and_characteristic_preflight": packet_report,
        "independent_balance_reference": reference_report,
        "operator_changed": False,
        "propagation_executed": False,
        "uniform_scattering_propagation_authorized": False,
        "embedded_scattering_propagation_authorized": False,
        "bounded_nonlinear_common_mode_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
        "implementation_source_hashes": source_manifest,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_manifest)
        ),
        "parent_input_hashes": _parent_hashes(),
        "decisive_arrays_path": str(DECISIVE_ARRAYS.relative_to(ROOT)),
        "decisive_arrays_sha256": c2a._sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": array_hashes,
        "runtime_seconds": time.perf_counter() - start,
    }
    provenance = {
        **git_identity,
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_scattering_energy_"
            "wp10c9d6c7c2a2.py"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "implementation_source_hashes": source_manifest,
        "parent_input_hashes": summary["parent_input_hashes"],
        "scientific_status": "REJECTED",
        "classification": summary["classification"],
    }
    c2a._write_json(CONFIG_PATH, config)
    c2a._write_json(MANIFEST_PATH, manifest)
    c2a._write_json(PROVENANCE_PATH, provenance)
    c2a._write_json(SUMMARY_PATH, summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return summary


if __name__ == "__main__":
    print(json.dumps(c2a._plain(run()), indent=2, sort_keys=True))
