#!/usr/bin/env python3
"""Freeze the prospective embedded-grid discrimination contract.

This definitions-only package changes no operator and propagates no state.
It maps the uniformly certified WP10c9d6c6f1 profiles onto the existing
fixed-N128-exterior embedded layouts, verifies their exact conservative
restriction, and freezes the physical exports and coupling diagnostics that
the later propagation package must use.
"""

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

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_frozen_hardening_wp10c9d5a as wp10c9d5a  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_reconstruct_face_charts,
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7a"
ANALYZED_BASE_COMMIT = "c852575f9f41ccc7d9a8c25b7265a2491c3738aa"
ANALYZED_BASE_PARENT = "7005aa11bb22f40862abf886bc3f7fee26ec68b8"
ANALYZED_BASE_TREE = "8dda44e48b2f92e790a455a03fb4573065141d08"
THIS_RUNNER = (
    "scripts/run_causal_inner_embedded_manifest_wp10c9d6c7a.py"
)

PARENT_COUPLING_FACE = 48
REFINEMENT_RATIOS = (1, 2, 4)
LAYOUTS = {
    1: "N128_exterior_N128_inner_c48",
    2: "N128_exterior_N256_inner_c48",
    4: "N128_exterior_N512_inner_c48",
}
UNIFORM_LABELS = {
    1: "uniform_N128",
    2: "uniform_N256",
    4: "uniform_N512",
}
BASE_PROFILES = (
    "p3__inward_shear",
    "p3__outward_shear",
    "p5__inward_shear",
    "p5__outward_shear",
    "p3__material",
)
PROFILE_KINDS = ("primary_physical", "secondary_physical")
COMMON_PARENT_FACE_INDICES = (0, 13, 25, 37, 43, 45, 48, 51)
RECONSTRUCTION_HALO_CELLS = 3

MAXIMUM_GRID_REPLAY_DEFECT = 0.0
MAXIMUM_EXTERIOR_REPLAY_DEFECT = 2.0e-12
MAXIMUM_BACKGROUND_RESTRICTION_DEFECT = 2.0e-12
MAXIMUM_PROFILE_RESTRICTION_DEFECT = 2.0e-12
MAXIMUM_PROFILE_EXTERIOR_NORM = 0.0
MAXIMUM_COUPLING_TRACE_JUMP = 1.0e-4
MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE = 0.0

F0_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_manifest_wp10c9d6c6f0"
)
F0_MANIFEST = F0_DIRECTORY / "band_envelope_manifest.json"
F0_SUMMARY = F0_DIRECTORY / "summary.json"

F1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_validation_wp10c9d6c6f1"
)
F1_CONFIG = F1_DIRECTORY / "config.json"
F1_SUMMARY = F1_DIRECTORY / "summary.json"
F1_ARRAYS = F1_DIRECTORY / "decisive_arrays.npz"
F1_PROVENANCE = F1_DIRECTORY / "provenance.json"

C3_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3"
)
C3_ARRAYS = C3_DIRECTORY / "decisive_arrays.npz"
C3_SUMMARY = C3_DIRECTORY / "summary.json"

C0E_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)
C0E_CONTEXTS = C0E_DIRECTORY / "replay_contexts.json"
C0E_INPUTS = C0E_DIRECTORY / "replay_inputs.npz"
C0E_SUMMARY = C0E_DIRECTORY / "summary.json"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "embedded_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_frozen_hardening_wp10c9d5a.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_embedded_patch.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py",
    "tests/test_causal_inner_embedded_manifest_wp10c9d6c7a.py",
)

OBSERVABLE_CONTRACT = {
    "observable_names": (
        "inner_flux_mass",
        "inner_flux_angular_momentum",
        "inner_flux_killing_energy",
        "interface_flux_mass",
        "interface_flux_angular_momentum",
        "interface_flux_killing_energy",
        "net_drive_mass",
        "net_drive_angular_momentum",
        "net_drive_killing_energy",
        "cooling_angular_momentum",
        "cooling_killing_energy",
        "vertical_work_angular_momentum",
        "vertical_work_killing_energy",
    ),
    "inner_flux_face": 0,
    "interface_flux_face": "layout.coupling_face_index",
    "volume_sum": "cells_strictly_inside_coupling_face",
    "net_drive_identity": (
        "inner_shared_flux_minus_interface_shared_flux_plus_all_active_"
        "stationary_sources"
    ),
    "one_shared_interface_flux_required": True,
    "maximum_shared_flux_telescoping_defect": 1.0e-12,
    "direct_face_jvp_and_prefix_ledger_parity_required": True,
}

COUPLING_DIAGNOSTIC_CONTRACT = {
    "preinterface_parent_face": 45,
    "coupling_parent_face": 48,
    "postinterface_parent_face": 51,
    "inner_energy_window_parent_faces": (42, 48),
    "outer_energy_window_parent_faces": (48, 54),
    "physical_energy": (
        "descriptor_compatible_characteristic_energy_with_fixed_physical_"
        "field_scales"
    ),
    "report_selected_opposite_and_other_family_energy": True,
    "report_incident_reflected_and_transmitted_history": True,
    "report_interface_flux_and_state_histories": True,
    "report_reflection_and_transmission_convergence": True,
    "absolute_reflection_threshold": None,
    "reason_no_absolute_reflection_threshold": (
        "the unchanged coarse_fine interface is being discriminated; "
        "convergence and physical ledger closure are binding"
    ),
}

PROPAGATION_CONTRACT = {
    "binding_layouts": tuple(LAYOUTS.values()),
    "fixed_parent_exterior_resolution": "N128",
    "inner_refinement_ratios": REFINEMENT_RATIOS,
    "time_horizon_s": 0.125,
    "time_sample_count": 65,
    "profile_variants_inherited_without_change": True,
    "profile_extension_rule": (
        "the certified full_inner_domain_sine_power continuum profile is "
        "extended by exact zero into the coarse exterior"
    ),
    "no_shift_taper_or_reoptimization": True,
    "exact_semigroup_boundary_integrals_required": True,
    "minimum_rms_order": 0.75,
    "minimum_maximum_order": 0.75,
    "minimum_significant_component_order": 0.75,
    "maximum_fine_normalized_difference": 0.05,
    "minimum_history_cosine": 0.90,
    "minimum_refinement_error_cosine": 0.90,
    "maximum_reference_uncertainty_to_fine_difference": 0.10,
    "minimum_relative_activity": 1.0e-8,
    "state_reference_gates_unchanged_from_wp10c9d6c6f1": True,
    "component_route_contract_inherited_from_wp10c9d6c6f0": True,
    "method_gate_must_pass_on_every_layout_before_propagation": True,
    "sign_and_amplitude_scaling_must_replay": True,
    "all_profile_variants_binding": True,
    "fail_fast_on_first_profile_failure": True,
    "no_definition_or_threshold_changes_after_this_manifest": True,
    "classification_if_all_variants_pass": (
        "embedded_operator_certified_for_declared_resolved_profile_class"
    ),
    "classification_if_any_variant_fails": (
        "prospective_embedded_profile_validation_failed"
    ),
}


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


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


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
        raise RuntimeError("WP10c9d6c7a analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {path: _sha256(ROOT / path) for path in IMPLEMENTATION_SOURCES}
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
    CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CANONICAL_MANIFEST.open(
        "w", newline="", encoding="utf-8"
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _parent_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _face_index(parent_face: int, ratio: int) -> int:
    if parent_face <= PARENT_COUPLING_FACE:
        return int(parent_face * ratio)
    return int(
        PARENT_COUPLING_FACE * ratio
        + parent_face
        - PARENT_COUPLING_FACE
    )


def _build() -> tuple[dict, dict[str, np.ndarray], dict]:
    f0_manifest = json.loads(F0_MANIFEST.read_text(encoding="utf-8"))
    f0_summary = json.loads(F0_SUMMARY.read_text(encoding="utf-8"))
    f1_summary = json.loads(F1_SUMMARY.read_text(encoding="utf-8"))
    c3_summary = json.loads(C3_SUMMARY.read_text(encoding="utf-8"))
    c0e_summary = json.loads(C0E_SUMMARY.read_text(encoding="utf-8"))
    replay_contexts = json.loads(
        C0E_CONTEXTS.read_text(encoding="utf-8")
    )
    f1_arrays = _load_npz(F1_ARRAYS)
    c3_arrays = _load_npz(C3_ARRAYS)
    replay_arrays = _load_npz(C0E_INPUTS)

    if f0_manifest["manifest_sha256"] != f0_summary["manifest_sha256"]:
        raise RuntimeError("WP10c9d6c6f0 manifest identity changed")
    if f1_summary["classification"] != (
        "prospective_band_envelope_uniform_validation_certified"
    ):
        raise RuntimeError("WP10c9d6c6f1 certification changed")
    if not f1_summary["passed"]:
        raise RuntimeError("WP10c9d6c6f1 no longer passes")
    if c3_summary["classification"] != (
        "smooth_continuum_four_level_export_direction_certified"
    ):
        raise RuntimeError("WP10c9d6c3 continuum background changed")
    if not c0e_summary["passed"]:
        raise RuntimeError("WP10c9d5c0e replay inputs are not certified")

    variants = f0_manifest["profile_variants"]
    if (
        len(variants) != 20
        or {item["base_profile"] for item in variants}
        != set(BASE_PROFILES)
        or not all(item["binding"] for item in variants)
    ):
        raise RuntimeError("frozen profile variants changed")

    first_label = LAYOUTS[1]
    parent_edges = np.asarray(
        replay_arrays[f"{first_label}__grid_edges"],
        dtype=float,
    )
    gravitational_radius = float(
        replay_contexts["contexts"][first_label][
            "grid_gravitational_radius"
        ]
    )
    parent_grid = make_kerr_schild_column_grid_from_edges(
        parent_edges,
        gravitational_radius,
    )
    if (
        parent_grid.centers.size != 64
        or PARENT_COUPLING_FACE >= parent_grid.centers.size
    ):
        raise RuntimeError("embedded parent layout changed")
    coupling_radius = float(parent_edges[PARENT_COUPLING_FACE])
    coupling_radius_over_rg = coupling_radius / gravitational_radius
    common_exterior_base = np.asarray(
        replay_arrays[f"{first_label}__base_primitives"],
        dtype=float,
    )[PARENT_COUPLING_FACE:]

    arrays: dict[str, np.ndarray] = {
        "parent_grid_edges": parent_edges,
        "common_parent_face_indices": np.asarray(
            COMMON_PARENT_FACE_INDICES,
            dtype=np.int64,
        ),
        "common_face_radii_over_rg": (
            parent_edges[
                np.asarray(COMMON_PARENT_FACE_INDICES, dtype=int)
            ]
            / gravitational_radius
        ),
        "times": np.asarray(f1_arrays["times"], dtype=float),
        "fixed_physical_observable_scales": np.asarray(
            f1_arrays["fixed_physical_observable_scales"],
            dtype=float,
        ),
        "field_scales": np.asarray(
            f1_arrays["field_scales"],
            dtype=float,
        ),
        "common_exterior_base_primitives": common_exterior_base,
    }
    reference_embedded_profiles: dict[
        tuple[str, str], np.ndarray
    ] = {}
    reference_spliced_base = None
    layout_reports = {}
    embedded_projection_hashes = {}
    maximum_grid_replay = 0.0
    maximum_exterior_replay = 0.0
    maximum_background_restriction = 0.0
    maximum_profile_restriction = 0.0
    maximum_profile_exterior_norm = 0.0
    maximum_coupling_trace_jump = 0.0
    maximum_factor_change = 0.0

    exterior_suffixes = (
        "base_primitives",
        "stream_rest_mass",
        "stream_radial_momentum_over_c",
        "stream_angular_momentum_over_c",
        "stream_killing_energy_over_c2",
    )
    for ratio in REFINEMENT_RATIOS:
        label = LAYOUTS[ratio]
        uniform_label = UNIFORM_LABELS[ratio]
        layout = make_causal_embedded_patch_layout(
            parent_grid,
            PARENT_COUPLING_FACE,
            ratio,
        )
        active_cells = int(layout.coupling_face_index)
        replay_edges = np.asarray(
            replay_arrays[f"{label}__grid_edges"],
            dtype=float,
        )
        grid_defect = float(
            np.max(
                np.abs(layout.grid.edges - replay_edges)
                / np.maximum(np.abs(replay_edges), 1.0)
            )
        )
        maximum_grid_replay = max(maximum_grid_replay, grid_defect)

        context = wp10c9d5a._context_from_payload(
            replay_contexts["contexts"][label],
            replay_arrays,
        )
        if not np.array_equal(context.grid.edges, layout.grid.edges):
            raise RuntimeError(f"{label} context/layout grid mismatch")

        inner_base = np.asarray(
            c3_arrays[f"{uniform_label}__base_primitives"],
            dtype=float,
        )
        spliced_base = np.concatenate(
            (inner_base, common_exterior_base),
            axis=0,
        )
        if spliced_base.shape != (layout.n_cells, 5):
            raise RuntimeError(f"{label} spliced background shape changed")
        arrays[f"{label}__grid_edges"] = np.asarray(layout.grid.edges)
        arrays[f"{label}__parent_cell_indices"] = np.asarray(
            layout.parent_cell_indices,
            dtype=np.int64,
        )
        arrays[f"{label}__subcell_indices"] = np.asarray(
            layout.subcell_indices,
            dtype=np.int64,
        )
        arrays[f"{label}__spliced_base_primitives"] = spliced_base

        exterior_defects = {}
        for suffix in exterior_suffixes:
            replay = np.asarray(
                replay_arrays[f"{label}__{suffix}"],
                dtype=float,
            )
            selected = (
                replay[active_cells:]
                if suffix != "grid_edges"
                else replay[active_cells:]
            )
            reference = (
                common_exterior_base
                if suffix == "base_primitives"
                else np.asarray(
                    replay_arrays[f"{first_label}__{suffix}"],
                    dtype=float,
                )[PARENT_COUPLING_FACE:]
            )
            defect = _relative_defect(selected, reference)
            exterior_defects[suffix] = defect
            maximum_exterior_replay = max(
                maximum_exterior_replay,
                defect,
            )

        restricted_base = restrict_causal_embedded_patch_cell_averages(
            spliced_base,
            layout,
        )
        if reference_spliced_base is None:
            reference_spliced_base = np.array(restricted_base, copy=True)
        background_restriction = _relative_defect(
            restricted_base,
            reference_spliced_base,
        )
        maximum_background_restriction = max(
            maximum_background_restriction,
            background_restriction,
        )

        reconstruction = causal_five_field_reconstruct_face_charts(
            context,
            spliced_base,
            purpose="flux",
        )
        factor_change = float(
            np.max(
                np.abs(
                    np.asarray(
                        reconstruction.admissibility_factors,
                        dtype=float,
                    )
                    - 1.0
                )
            )
        )
        base_scales = np.maximum(
            np.max(np.abs(spliced_base), axis=0),
            1.0,
        )
        coupling_trace_jump = float(
            np.max(
                np.abs(
                    np.asarray(
                        reconstruction.right_face_charts[active_cells],
                        dtype=float,
                    )
                    - np.asarray(
                        reconstruction.left_face_charts[active_cells],
                        dtype=float,
                    )
                )
                / base_scales
            )
        )
        maximum_factor_change = max(
            maximum_factor_change,
            factor_change,
        )
        maximum_coupling_trace_jump = max(
            maximum_coupling_trace_jump,
            coupling_trace_jump,
        )

        profile_reports = {}
        for profile in BASE_PROFILES:
            kind_reports = {}
            for kind in PROFILE_KINDS:
                uniform_key = f"{profile}__{uniform_label}__{kind}"
                inherited = np.asarray(f1_arrays[uniform_key], dtype=float)
                expected_hash = f1_summary["decisive_array_hashes"][
                    uniform_key
                ]
                if _array_sha256(inherited) != expected_hash:
                    raise RuntimeError(
                        f"{uniform_key} projection hash changed"
                    )
                embedded = np.concatenate(
                    (
                        inherited,
                        np.zeros(
                            (layout.n_cells - active_cells, 5),
                            dtype=float,
                        ),
                    ),
                    axis=0,
                )
                array_key = f"{profile}__{label}__{kind}"
                arrays[array_key] = embedded
                embedded_projection_hashes[array_key] = (
                    _array_sha256(embedded)
                )
                exterior_norm = float(
                    np.linalg.norm(embedded[active_cells:])
                )
                maximum_profile_exterior_norm = max(
                    maximum_profile_exterior_norm,
                    exterior_norm,
                )
                restricted = (
                    restrict_causal_embedded_patch_cell_averages(
                        embedded,
                        layout,
                    )
                )
                reference_key = (profile, kind)
                if reference_key not in reference_embedded_profiles:
                    reference_embedded_profiles[reference_key] = np.array(
                        restricted,
                        copy=True,
                    )
                restriction_defect = _relative_defect(
                    restricted,
                    reference_embedded_profiles[reference_key],
                )
                maximum_profile_restriction = max(
                    maximum_profile_restriction,
                    restriction_defect,
                )
                kind_reports[kind] = {
                    "embedded_array_sha256": (
                        embedded_projection_hashes[array_key]
                    ),
                    "inherited_uniform_array_sha256": expected_hash,
                    "restriction_to_parent_defect": restriction_defect,
                    "exterior_norm": exterior_norm,
                }
            profile_reports[profile] = kind_reports

        face_map = {
            str(parent_face): _face_index(parent_face, ratio)
            for parent_face in COMMON_PARENT_FACE_INDICES
        }
        layout_reports[label] = {
            "refinement_ratio": ratio,
            "n_cells": int(layout.n_cells),
            "n_refined_cells": int(layout.n_refined_cells),
            "n_coarse_exterior_cells": int(
                layout.n_cells - layout.n_refined_cells
            ),
            "parent_coupling_face_index": int(
                layout.parent_coupling_face_index
            ),
            "coupling_face_index": int(layout.coupling_face_index),
            "coupling_radius_over_rg": coupling_radius_over_rg,
            "grid_replay_defect": grid_defect,
            "exterior_replay_defects": exterior_defects,
            "background_restriction_defect": background_restriction,
            "maximum_reconstruction_factor_change": factor_change,
            "normalized_coupling_trace_jump": coupling_trace_jump,
            "common_parent_to_embedded_face_indices": face_map,
            "profile_reports": profile_reports,
        }

    layout_passed = bool(
        maximum_grid_replay <= MAXIMUM_GRID_REPLAY_DEFECT
        and maximum_exterior_replay <= MAXIMUM_EXTERIOR_REPLAY_DEFECT
        and maximum_background_restriction
        <= MAXIMUM_BACKGROUND_RESTRICTION_DEFECT
        and maximum_profile_restriction
        <= MAXIMUM_PROFILE_RESTRICTION_DEFECT
        and maximum_profile_exterior_norm
        <= MAXIMUM_PROFILE_EXTERIOR_NORM
        and maximum_coupling_trace_jump
        <= MAXIMUM_COUPLING_TRACE_JUMP
        and maximum_factor_change
        <= MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
    )

    layout_contract = {
        "parent_grid": "fixed_N128_exterior_64_cells",
        "parent_cell_count": 64,
        "parent_inner_radius_over_rg": float(
            parent_edges[0] / gravitational_radius
        ),
        "parent_outer_radius_over_rg": float(
            parent_edges[-1] / gravitational_radius
        ),
        "parent_coupling_face_index": PARENT_COUPLING_FACE,
        "coupling_radius_over_rg": coupling_radius_over_rg,
        "refinement_ratios": REFINEMENT_RATIOS,
        "layout_labels": tuple(LAYOUTS.values()),
        "coarse_exterior_cell_count": 16,
        "grid_rule": (
            "subdivide_each_parent_cell_inside_parent_face_48_uniformly_"
            "in_log_radius_and_retain_parent_exterior_exactly"
        ),
        "background_rule": (
            "certified_c3_smooth_uniform_inner_projection_spliced_to_one_"
            "common_c0e_N128_exterior_base"
        ),
        "stream_and_geometry_rule": (
            "replay_the_committed_c0e_context_for_each_exact_layout"
        ),
        "maximum_grid_replay_defect": MAXIMUM_GRID_REPLAY_DEFECT,
        "maximum_exterior_replay_defect": (
            MAXIMUM_EXTERIOR_REPLAY_DEFECT
        ),
        "maximum_background_restriction_defect": (
            MAXIMUM_BACKGROUND_RESTRICTION_DEFECT
        ),
        "maximum_coupling_trace_jump": MAXIMUM_COUPLING_TRACE_JUMP,
        "maximum_reconstruction_factor_change": (
            MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
        ),
    }
    profile_contract = {
        "base_profiles": BASE_PROFILES,
        "profile_variants": variants,
        "definitions_inherited_from_manifest_sha256": (
            f0_manifest["manifest_sha256"]
        ),
        "uniform_validation_decisive_arrays_sha256": (
            f1_summary["decisive_arrays_sha256"]
        ),
        "extension_outside_coupling": "exact_zero",
        "continuum_endpoint_behavior": (
            "sin_power_p3_or_p5_vanishes_at_the_coupling_surface; "
            "the_zero_extension_is_at_least_C2"
        ),
        "no_shift_taper_or_reoptimization": True,
        "maximum_profile_restriction_defect": (
            MAXIMUM_PROFILE_RESTRICTION_DEFECT
        ),
        "maximum_profile_exterior_norm": (
            MAXIMUM_PROFILE_EXTERIOR_NORM
        ),
        "embedded_projection_hashes": embedded_projection_hashes,
    }
    common_surface_contract = {
        "selection_grid": first_label,
        "parent_face_indices": COMMON_PARENT_FACE_INDICES,
        "radii_over_rg": tuple(
            float(parent_edges[index] / gravitational_radius)
            for index in COMMON_PARENT_FACE_INDICES
        ),
        "mapping_rule_inside_coupling": "embedded_face=parent_face*ratio",
        "mapping_rule_outside_coupling": (
            "embedded_face=48*ratio+(parent_face-48)"
        ),
        "reconstruction_halo_cells": RECONSTRUCTION_HALO_CELLS,
        "last_preinterface_parent_face_with_inner_halo": 45,
        "first_postinterface_parent_face_with_outer_halo": 51,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "uniform_certification_preserved": True,
        "historical_c6c_rejection_preserved": True,
        "layout_contract": layout_contract,
        "layout_reports": layout_reports,
        "profile_contract": profile_contract,
        "common_surface_contract": common_surface_contract,
        "observable_contract": OBSERVABLE_CONTRACT,
        "coupling_diagnostic_contract": COUPLING_DIAGNOSTIC_CONTRACT,
        "component_route_contract": (
            f0_manifest["component_route_contract"]
        ),
        "prospective_propagation_contract": PROPAGATION_CONTRACT,
        "layout_and_profile_eligibility_passed": layout_passed,
        "measured_extrema": {
            "maximum_grid_replay_defect": maximum_grid_replay,
            "maximum_exterior_replay_defect": maximum_exterior_replay,
            "maximum_background_restriction_defect": (
                maximum_background_restriction
            ),
            "maximum_profile_restriction_defect": (
                maximum_profile_restriction
            ),
            "maximum_profile_exterior_norm": (
                maximum_profile_exterior_norm
            ),
            "maximum_normalized_coupling_trace_jump": (
                maximum_coupling_trace_jump
            ),
            "maximum_reconstruction_factor_change": (
                maximum_factor_change
            ),
        },
    }
    manifest = {
        **payload,
        "manifest_sha256": causal_canonical_json_sha256(payload),
    }
    inherited = {
        "f0": f0_summary,
        "f1": f1_summary,
        "c3": c3_summary,
        "c0e": c0e_summary,
    }
    return manifest, arrays, inherited


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "propagation_executed": False,
        "layout_contract": manifest["layout_contract"],
        "profile_contract": {
            key: value
            for key, value in manifest["profile_contract"].items()
            if key != "embedded_projection_hashes"
        },
        "common_surface_contract": manifest[
            "common_surface_contract"
        ],
        "observable_contract": OBSERVABLE_CONTRACT,
        "coupling_diagnostic_contract": COUPLING_DIAGNOSTIC_CONTRACT,
        "prospective_propagation_contract": PROPAGATION_CONTRACT,
    }


def run() -> dict:
    identity = _validate_analyzed_git_identity()
    manifest, arrays, inherited = _build()
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, _config(manifest))
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    array_hashes = {
        name: _array_sha256(values) for name, values in arrays.items()
    }
    passed = bool(manifest["layout_and_profile_eligibility_passed"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": (
            "embedded_layout_and_profile_manifest_frozen_"
            "propagation_authorized"
            if passed
            else "embedded_layout_or_profile_manifest_ineligible"
        ),
        "authorized_next": (
            "WP10c9d6c7b_prospective_embedded_propagation"
            if passed
            else None
        ),
        "passed": passed,
        "operator_changed": False,
        "propagation_executed": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "decisive_arrays_path": str(DECISIVE_ARRAYS.relative_to(ROOT)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": array_hashes,
        "layout_reports": manifest["layout_reports"],
        "measured_extrema": manifest["measured_extrema"],
        "base_profile_count": len(BASE_PROFILES),
        "profile_variant_count": len(
            manifest["profile_contract"]["profile_variants"]
        ),
        "common_surface_count": len(COMMON_PARENT_FACE_INDICES),
        "uniform_parent_classification": inherited["f1"][
            "classification"
        ],
        "uniform_certification_preserved": True,
        "historical_c6c_rejection_preserved": True,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "embedded_propagation_authorized": passed,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "EMBEDDED MANIFEST ONLY",
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse", "HEAD^{tree}"
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_embedded_manifest_wp10c9d6c7a.py"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": _parent_hashes(
            (
                F0_MANIFEST,
                F0_SUMMARY,
                F1_CONFIG,
                F1_SUMMARY,
                F1_ARRAYS,
                F1_PROVENANCE,
                C3_ARRAYS,
                C3_SUMMARY,
                C0E_CONTEXTS,
                C0E_INPUTS,
                C0E_SUMMARY,
            )
        ),
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "authorized_next": summary["authorized_next"],
                "manifest_sha256": summary["manifest_sha256"],
                "profile_variant_count": summary[
                    "profile_variant_count"
                ],
                "measured_extrema": summary["measured_extrema"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
