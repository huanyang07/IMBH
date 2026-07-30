#!/usr/bin/env python3
"""Propagate the frozen WP10c9d6c7a embedded-grid manifest.

The exact twenty uniformly certified profile variants are propagated on the
three nonoverlapping embedded layouts with the unchanged monolithic tangent.
The fixed N128 exterior, coupling face, active-domain exports, common faces,
and all gates come from the immutable c7a manifest.
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

import run_causal_inner_embedded_manifest_wp10c9d6c7a as c7a
import run_causal_inner_frozen_hardening_wp10c9d5a as wp10c9d5a
import run_causal_inner_height_localization_wp10c9d6c6d as c6d
import run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1 as c6e1
import run_causal_inner_monolithic_four_level_wp10c9d6c2 as c6c2
import run_causal_inner_monolithic_uniform_exports_wp10c9d6c as c6base
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
    evaluate_causal_five_field_monolithic_backward_euler,
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_validation import (  # noqa: E402
    causal_characteristic_energy_history,
    causal_dimensionless_characteristic_inverse,
    causal_embedded_active_direct_observables,
    causal_embedded_active_observable_audit,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_height_localization import (  # noqa: E402
    causal_partition_cell_integrals,
    causal_signed_band_gram_matrix,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_exact_semigroup_integral_history,
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_linear_tangent import (
    _analytic_coordinate_principal_basis,
)  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_field_history_norm,
    causal_trapezoid_weights,
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7b"
ANALYZED_BASE_COMMIT = "82c3a9e5a326fedeccfafd8e8a4a9704935c64a3"
ANALYZED_BASE_PARENT = "c852575f9f41ccc7d9a8c25b7265a2491c3738aa"
ANALYZED_BASE_TREE = "476342f0cdeb8f2222c9495aed89cddaf04ae965"
FROZEN_MANIFEST_SHA256 = (
    "c465f284dd2991fa0241b2bb268fc723a89bc111bedd59c3cf5a5830346e554a"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_embedded_validation_wp10c9d6c7b.py"
)

LABELS = tuple(c7a.LAYOUTS[ratio] for ratio in c7a.REFINEMENT_RATIOS)
BASE_PROFILES = tuple(c7a.BASE_PROFILES)
OBSERVABLE_NAMES = tuple(c7a.OBSERVABLE_CONTRACT["observable_names"])
CONSERVATIVE_FIELDS = np.asarray((0, 2, 4), dtype=int)
TARGET_OBSERVABLE_NAME = "vertical_work_angular_momentum"
TARGET_OBSERVABLE_INDEX = OBSERVABLE_NAMES.index(TARGET_OBSERVABLE_NAME)
ANGULAR_FIELD = 2
MAXIMUM_PROPAGATION_SCALING_DEFECT = 1.0e-10
MAXIMUM_EXACT_INTEGRAL_RESIDUAL = 1.0e-12

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_MANIFEST = PARENT_DIRECTORY / "embedded_manifest.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"

C0E_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)
C0E_CONTEXTS = C0E_DIRECTORY / "replay_contexts.json"
C0E_INPUTS = C0E_DIRECTORY / "replay_inputs.npz"

F1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_validation_wp10c9d6c6f1"
)
F1_SUMMARY = F1_DIRECTORY / "summary.json"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_validation_wp10c9d6c7b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints"
PROPAGATION_CHECKPOINT = (
    CHECKPOINT_DIRECTORY
    / "causal_inner_embedded_validation_wp10c9d6c7b_propagated.npz"
)
PROPAGATION_CHECKPOINT_REPORT = (
    CHECKPOINT_DIRECTORY
    / "causal_inner_embedded_validation_wp10c9d6c7b_report.json"
)

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_embedded_manifest_wp10c9d6c7a.py",
    "scripts/run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py",
    "scripts/run_causal_inner_packet_validation_wp10c9d6c6c.py",
    "scripts/"
    "run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_embedded_validation.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_embedded_patch.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_validation.py",
    "tests/test_causal_inner_embedded_validation.py",
    "tests/test_causal_inner_embedded_validation_wp10c9d6c7b.py",
)

FAMILY_INDEX = {
    "inward_acoustic": 0,
    "inward_shear": 1,
    "material": 2,
    "outward_shear": 3,
    "outward_acoustic": 4,
}
OPPOSITE_INDEX = {
    "inward_shear": FAMILY_INDEX["outward_shear"],
    "outward_shear": FAMILY_INDEX["inward_shear"],
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
    return causal_array_sha256(np.asarray(values))


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
        raise RuntimeError("WP10c9d6c7b analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).is_file()
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
        return {
            name: np.array(source[name], copy=True)
            for name in source.files
        }


def _load_frozen_inputs() -> tuple[dict, dict, dict[str, np.ndarray], dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    arrays = _load_npz(PARENT_ARRAYS)
    f1 = json.loads(F1_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != (
            "embedded_layout_and_profile_manifest_frozen_"
            "propagation_authorized"
        )
        or parent["authorized_next"]
        != "WP10c9d6c7b_prospective_embedded_propagation"
        or not parent["passed"]
        or not parent["embedded_propagation_authorized"]
        or f1["classification"]
        != "prospective_band_envelope_uniform_validation_certified"
    ):
        raise RuntimeError("WP10c9d6c7b authorization changed")
    stored = manifest.pop("manifest_sha256")
    calculated = causal_canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = stored
    if (
        stored != FROZEN_MANIFEST_SHA256
        or calculated != FROZEN_MANIFEST_SHA256
        or parent["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or not manifest["layout_and_profile_eligibility_passed"]
        or manifest["propagation_executed"]
    ):
        raise RuntimeError("frozen WP10c9d6c7a manifest changed")
    if set(arrays) != set(parent["decisive_array_hashes"]):
        raise RuntimeError("WP10c9d6c7a array set changed")
    for name, expected in parent["decisive_array_hashes"].items():
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"WP10c9d6c7a array changed: {name}")
    return parent, manifest, arrays, f1


def _build_inputs(
    manifest: dict,
    arrays: dict[str, np.ndarray],
) -> tuple[dict, dict, dict]:
    replay_contexts = json.loads(
        C0E_CONTEXTS.read_text(encoding="utf-8")
    )
    replay_arrays = _load_npz(C0E_INPUTS)
    parent_edges = np.asarray(arrays["parent_grid_edges"], dtype=float)
    first_payload = replay_contexts["contexts"][LABELS[0]]
    gravitational_radius = float(
        first_payload["grid_gravitational_radius"]
    )
    parent_grid = make_kerr_schild_column_grid_from_edges(
        parent_edges,
        gravitational_radius,
    )
    configurations = {}
    layouts = {}
    for ratio, label in zip(
        c7a.REFINEMENT_RATIOS,
        LABELS,
        strict=True,
    ):
        context = wp10c9d5a._context_from_payload(
            replay_contexts["contexts"][label],
            replay_arrays,
        )
        layout = make_causal_embedded_patch_layout(
            parent_grid,
            c7a.PARENT_COUPLING_FACE,
            ratio,
        )
        base = np.asarray(
            arrays[f"{label}__spliced_base_primitives"],
            dtype=float,
        )
        if (
            not np.array_equal(context.grid.edges, layout.grid.edges)
            or not np.array_equal(context.grid.edges, arrays[f"{label}__grid_edges"])
            or base.shape != (layout.n_cells, 5)
        ):
            raise RuntimeError(f"{label} embedded replay changed")
        columns, rows = c6c2._scales_for(context, base)
        initial_directions = {
            "common_mode": (
                np.asarray(
                    arrays[
                        f"p3__inward_shear__{label}__primary_physical"
                    ],
                    dtype=float,
                ).ravel()
                / columns
            ),
            "heldout_near_excision": (
                np.asarray(
                    arrays[f"p3__material__{label}__primary_physical"],
                    dtype=float,
                ).ravel()
                / columns
            ),
        }
        configurations[label] = {
            "context": context,
            "base_primitives": base,
            "primitive_column_scales": columns,
            "conservation_row_scales": rows,
            "initial_directions": initial_directions,
            "times": np.asarray(arrays["times"], dtype=float),
            "active_cells": int(layout.coupling_face_index),
            "ratio": int(ratio),
        }
        layouts[label] = layout
    metadata = _variant_metadata(manifest)
    return configurations, layouts, metadata


def _variant_metadata(manifest: dict) -> dict:
    variants = manifest["profile_contract"]["profile_variants"]
    base_names = list(BASE_PROFILES)
    packet_ids = [item["profile_id"] for item in variants]
    base_indices = np.asarray(
        [base_names.index(item["base_profile"]) for item in variants],
        dtype=int,
    )
    multipliers = np.asarray(
        [
            float(item["amplitude_factor"]) * int(item["sign"])
            for item in variants
        ],
        dtype=float,
    )
    base_variant_indices = []
    for index in range(len(base_names)):
        matches = np.flatnonzero(
            (base_indices == index) & (multipliers == 1.0)
        )
        if matches.size != 1:
            raise RuntimeError("frozen base variant changed")
        base_variant_indices.append(int(matches[0]))
    return {
        "variants": variants,
        "packet_ids": packet_ids,
        "base_names": base_names,
        "base_indices": base_indices,
        "multipliers": multipliers,
        "base_variant_indices": np.asarray(
            base_variant_indices,
            dtype=int,
        ),
    }


def _fourth_order(values: dict[float, np.ndarray], step: float) -> np.ndarray:
    return (
        -values[2.0 * step]
        + 8.0 * values[step]
        - 8.0 * values[-step]
        + values[-2.0 * step]
    ) / (12.0 * step)


def _active_directional_audit(
    configuration: dict,
    tangent,
    active_audit,
    name: str,
) -> float:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    columns = np.asarray(
        configuration["primitive_column_scales"],
        dtype=float,
    )
    direction = np.asarray(
        configuration["initial_directions"][name],
        dtype=float,
    ).ravel()
    direction /= max(
        float(np.max(np.abs(direction))),
        np.finfo(float).tiny,
    )
    physical = (columns * direction).reshape(base.shape)
    observations = {}
    step = float(c6base.DIRECTIONAL_STEP)
    for multiplier in (-2.0, -1.0, 1.0, 2.0):
        offset = multiplier * step
        charts = base + offset * physical
        evaluation = evaluate_causal_five_field_monolithic_backward_euler(
            charts,
            charts,
            1.0,
            configuration["context"],
            path_quadrature_order=c6base.PATH_QUADRATURE_ORDER,
        )
        observations[offset] = causal_embedded_active_direct_observables(
            evaluation,
            configuration["active_cells"],
        )
    direct = _fourth_order(observations, step)
    assembled = active_audit.observable_map @ direction
    return _relative_defect(direct, assembled)


def _build_tangents(
    configurations: dict,
) -> tuple[dict, dict, dict, bool]:
    tangents = {}
    active_audits = {}
    reports = {}
    all_passed = True
    for label in LABELS:
        print(f"WP10c9d6c7b: build monolithic tangent {label}", flush=True)
        configuration = configurations[label]
        tangent = causal_five_field_monolithic_frozen_tangent(
            configuration["context"],
            configuration["base_primitives"],
            primitive_column_scales=(
                configuration["primitive_column_scales"]
            ),
            conservation_row_scales=(
                configuration["conservation_row_scales"]
            ),
            path_quadrature_order=c6base.PATH_QUADRATURE_ORDER,
        )
        active = causal_embedded_active_observable_audit(
            tangent,
            configuration["active_cells"],
        )
        inherited = c6base._method_report(configuration, tangent)
        active_directional = {
            name: _active_directional_audit(
                configuration,
                tangent,
                active,
                name,
            )
            for name in ("common_mode", "heldout_near_excision")
        }
        maximum_active_directional = max(active_directional.values())
        coupling_gates = {
            "active_directional_export": bool(
                maximum_active_directional
                <= c6base.MAXIMUM_DIRECTIONAL_EXPORT_DEFECT
            ),
            "shared_flux_telescoping": bool(
                active.conservative_transport_telescoping_defect
                <= manifest_shared_flux_tolerance()
            ),
            "active_prefix_ledger": bool(
                active.active_prefix_ledger_defect
                <= manifest_shared_flux_tolerance()
            ),
        }
        passed = bool(inherited["passed"] and all(coupling_gates.values()))
        reports[label] = {
            "passed": passed,
            "inherited_monolithic_method_report": inherited,
            "active_directional_export_defects": active_directional,
            "maximum_active_directional_export_defect": (
                maximum_active_directional
            ),
            "conservative_transport_telescoping_defect": (
                active.conservative_transport_telescoping_defect
            ),
            "active_prefix_ledger_defect": (
                active.active_prefix_ledger_defect
            ),
            "coupling_gates": coupling_gates,
        }
        tangents[label] = tangent
        active_audits[label] = active
        all_passed = bool(all_passed and passed)
    return tangents, active_audits, reports, all_passed


def manifest_shared_flux_tolerance() -> float:
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    return float(
        manifest["observable_contract"][
            "maximum_shared_flux_telescoping_defect"
        ]
    )


def _variant_directions(
    configurations: dict,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> dict:
    result = {}
    for label in LABELS:
        columns = np.asarray(
            configurations[label]["primitive_column_scales"],
            dtype=float,
        )
        primary_base = np.column_stack(
            [
                np.asarray(
                    arrays[f"{name}__{label}__primary_physical"],
                    dtype=float,
                ).ravel()
                / columns
                for name in metadata["base_names"]
            ]
        )
        secondary_base = np.column_stack(
            [
                np.asarray(
                    arrays[f"{name}__{label}__secondary_physical"],
                    dtype=float,
                ).ravel()
                / columns
                for name in metadata["base_names"]
            ]
        )
        result[label] = {
            "primary_scaled": (
                primary_base[:, metadata["base_indices"]]
                * metadata["multipliers"][None, :]
            ),
            "secondary_base_scaled": secondary_base,
        }
    return result


def _restrict_cell_integrals(
    values: np.ndarray,
    layout,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape[-1] != layout.n_cells:
        raise ValueError("embedded cell-integral history has wrong shape")
    result = np.zeros(
        array.shape[:-1] + (layout.n_parent_cells,),
        dtype=float,
    )
    for cell, parent in enumerate(layout.parent_cell_indices):
        result[..., int(parent)] += array[..., cell]
    return result


def _propagate(
    configurations: dict,
    layouts: dict,
    tangents: dict,
    active_audits: dict,
    directions: dict,
    metadata: dict,
    manifest: dict,
) -> tuple[dict, dict]:
    propagated = {}
    reports = {}
    common_parent_faces = tuple(
        manifest["common_surface_contract"]["parent_face_indices"]
    )
    base_indices = metadata["base_variant_indices"]
    for label in LABELS:
        print(f"WP10c9d6c7b: propagate frozen profiles on {label}", flush=True)
        configuration = configurations[label]
        layout = layouts[label]
        tangent = tangents[label]
        generator = np.asarray(tangent.scaled_generator_per_s, dtype=float)
        primary = np.asarray(
            directions[label]["primary_scaled"],
            dtype=float,
        )
        secondary = np.asarray(
            directions[label]["secondary_base_scaled"],
            dtype=float,
        )
        initial = np.column_stack((primary, secondary))
        times = np.asarray(configuration["times"], dtype=float)
        trace = float(np.trace(generator))
        scaled = np.asarray(
            expm_multiply(
                generator,
                initial,
                start=float(times[0]),
                stop=float(times[-1]),
                num=int(times.size),
                endpoint=True,
                traceA=trace,
            ),
            dtype=float,
        )
        primary_scaled = scaled[:, :, : primary.shape[1]]
        secondary_scaled = scaled[:, :, primary.shape[1] :]
        half = np.asarray(
            expm_multiply(
                0.5 * float(times[-1]) * generator,
                primary,
                traceA=0.5 * float(times[-1]) * trace,
            ),
            dtype=float,
        )
        restarted = np.asarray(
            expm_multiply(
                0.5 * float(times[-1]) * generator,
                half,
                traceA=0.5 * float(times[-1]) * trace,
            ),
            dtype=float,
        )
        exact = causal_exact_semigroup_integral_history(
            generator,
            primary_scaled,
            primary,
        )
        observable = active_audits[label].observable_map
        signals = np.einsum("tnp,on->pto", primary_scaled, observable)
        cumulative = np.einsum(
            "tnp,on->pto",
            exact.integrated_states,
            observable,
        )
        corrections = np.einsum(
            "tnp,on->pto",
            exact.correction_states,
            observable,
        )
        columns = np.asarray(
            configuration["primitive_column_scales"],
            dtype=float,
        )
        cells = int(layout.n_cells)
        primary_physical = np.transpose(
            primary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(primary.shape[1], times.size, cells, 5)
        secondary_physical = np.transpose(
            secondary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(secondary.shape[1], times.size, cells, 5)
        restart_physical = np.transpose(
            restarted * columns[:, None],
            (1, 0),
        ).reshape(primary.shape[1], cells, 5)
        parent_primary = restrict_causal_embedded_patch_cell_averages(
            primary_physical,
            layout,
        )
        parent_secondary = restrict_causal_embedded_patch_cell_averages(
            secondary_physical,
            layout,
        )
        parent_restart = restrict_causal_embedded_patch_cell_averages(
            restart_physical,
            layout,
        )

        base_states = primary_scaled[:, :, base_indices]
        base_integrals = exact.integrated_states[:, :, base_indices]
        cell_map = active_audits[label].lower_height_cell_map
        cell_actions = np.transpose(
            np.einsum("cfn,tnp->tpcf", cell_map, base_states),
            (1, 0, 2, 3),
        )
        cumulative_cell_actions = np.transpose(
            np.einsum("cfn,tnp->tpcf", cell_map, base_integrals),
            (1, 0, 2, 3),
        )
        cell_actions[:, :, configuration["active_cells"] :, :] = 0.0
        cumulative_cell_actions[
            :, :, configuration["active_cells"] :, :
        ] = 0.0
        parent_cell_actions = _restrict_cell_integrals(
            np.moveaxis(cell_actions, -2, -1),
            layout,
        )
        parent_cell_actions = np.moveaxis(parent_cell_actions, -1, -2)
        parent_cumulative_cell_actions = _restrict_cell_integrals(
            np.moveaxis(cumulative_cell_actions, -2, -1),
            layout,
        )
        parent_cumulative_cell_actions = np.moveaxis(
            parent_cumulative_cell_actions,
            -1,
            -2,
        )

        face_indices = np.asarray(
            [
                manifest["layout_reports"][label][
                    "common_parent_to_embedded_face_indices"
                ][str(parent_face)]
                for parent_face in common_parent_faces
            ],
            dtype=int,
        )
        face_maps = (
            tangent.spatial_tangent.shared_face_flux_scaled_jacobians
        )
        selected_face_maps = np.asarray(
            face_maps[face_indices][:, CONSERVATIVE_FIELDS],
            dtype=float,
        )
        common_face_fluxes = np.einsum(
            "fkn,tnp->ptfk",
            selected_face_maps,
            base_states,
        )
        propagated[label] = {
            "times": times,
            "primary_physical": primary_physical,
            "secondary_base_physical": secondary_physical,
            "restart_physical": restart_physical,
            "parent_primary_physical": parent_primary,
            "parent_secondary_base_physical": parent_secondary,
            "parent_restart_physical": parent_restart,
            "signals": signals,
            "cumulative_signals": cumulative,
            "cumulative_corrections": corrections,
            "integral_relative_solve_residuals": (
                exact.relative_solve_residuals.T
            ),
            "base_parent_cell_actions": parent_cell_actions,
            "base_parent_cumulative_cell_actions": (
                parent_cumulative_cell_actions
            ),
            "base_common_face_fluxes": common_face_fluxes,
        }
        reports[label] = {
            "cell_count": cells,
            "active_cell_count": int(configuration["active_cells"]),
            "variant_count": int(primary.shape[1]),
            "base_count": int(base_indices.size),
            "maximum_exact_integral_relative_solve_residual": (
                exact.maximum_relative_solve_residual
            ),
            "restart_relative_defect": _relative_defect(
                restarted,
                primary_scaled[-1],
            ),
        }
    return propagated, reports


def _save_propagation_checkpoint(
    propagated: dict,
    propagation_report: dict,
    method_reports: dict,
) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload = {}
    for label in LABELS:
        for name, values in propagated[label].items():
            payload[f"{label}::{name}"] = np.asarray(values)
    np.savez_compressed(PROPAGATION_CHECKPOINT, **payload)
    _write_json(
        PROPAGATION_CHECKPOINT_REPORT,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "manifest_sha256": FROZEN_MANIFEST_SHA256,
            "parent_decisive_arrays_sha256": _sha256(PARENT_ARRAYS),
            "labels": LABELS,
            "propagation_report": propagation_report,
            "method_reports": method_reports,
            "checkpoint_sha256": _sha256(PROPAGATION_CHECKPOINT),
        },
    )


def _load_propagation_checkpoint() -> tuple[dict, dict, dict]:
    if (
        not PROPAGATION_CHECKPOINT.is_file()
        or not PROPAGATION_CHECKPOINT_REPORT.is_file()
    ):
        raise RuntimeError("WP10c9d6c7b propagation checkpoint is unavailable")
    report = json.loads(
        PROPAGATION_CHECKPOINT_REPORT.read_text(encoding="utf-8")
    )
    if (
        report["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or report["parent_decisive_arrays_sha256"] != _sha256(PARENT_ARRAYS)
        or tuple(report["labels"]) != LABELS
        or report["checkpoint_sha256"] != _sha256(PROPAGATION_CHECKPOINT)
    ):
        raise RuntimeError("WP10c9d6c7b propagation checkpoint changed")
    raw = _load_npz(PROPAGATION_CHECKPOINT)
    propagated = {label: {} for label in LABELS}
    for key, values in raw.items():
        label, name = key.split("::", 1)
        if label not in propagated:
            raise RuntimeError("checkpoint contains an unknown layout")
        propagated[label][name] = values
    return (
        propagated,
        report["propagation_report"],
        report["method_reports"],
    )


def _metric_payload(metrics) -> dict:
    return _named_metric_payload(metrics, OBSERVABLE_NAMES)


def _named_metric_payload(metrics, names: tuple[str, ...]) -> dict:
    indices = np.asarray(metrics.significant_components, dtype=int)
    return {
        "passed": metrics.passed,
        "significant_components": [
            names[index] for index in indices
        ],
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "component_orders": {
            names[index]: float(metrics.component_orders[position])
            for position, index in enumerate(indices)
        },
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "coarse_medium_rms_difference": (
            metrics.coarse_medium_rms_difference
        ),
        "medium_fine_rms_difference": (
            metrics.medium_fine_rms_difference
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _packet_metrics(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    scales: np.ndarray,
    contract: dict,
):
    return causal_packet_history_metrics(
        coarse,
        medium,
        fine,
        physical_scales=scales,
        relative_activity=contract["minimum_relative_activity"],
        minimum_rms_order=contract["minimum_rms_order"],
        minimum_maximum_order=contract["minimum_maximum_order"],
        minimum_significant_component_order=contract[
            "minimum_significant_component_order"
        ],
        maximum_fine_normalized_difference=contract[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=contract["minimum_history_cosine"],
        minimum_refinement_error_cosine=contract[
            "minimum_refinement_error_cosine"
        ],
    )


def _optional_packet_metric(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    scales: np.ndarray,
    contract: dict,
    names: tuple[str, ...],
) -> dict:
    values = tuple(
        np.asarray(history, dtype=float)
        for history in (coarse, medium, fine)
    )
    physical_scales = np.asarray(scales, dtype=float).ravel()
    response = np.max(
        np.abs(
            np.asarray(
                [
                    history / physical_scales[None, :]
                    for history in values
                ]
            )
        ),
        axis=(0, 1),
    )
    if not np.any(response >= contract["minimum_relative_activity"]):
        return {
            "active": False,
            "passed": True,
            "maximum_response_over_scale": float(np.max(response)),
        }
    metrics = _packet_metrics(
        values[0],
        values[1],
        values[2],
        physical_scales,
        contract,
    )
    return {
        "active": True,
        **_named_metric_payload(metrics, names),
    }


def _history_norm(
    values: np.ndarray,
    scales: np.ndarray,
    times: np.ndarray,
) -> float:
    normalized = np.asarray(values, dtype=float) / np.asarray(
        scales,
        dtype=float,
    )[None, :]
    weights = causal_trapezoid_weights(times)
    return float(
        np.sqrt(np.einsum("to,to,t->", normalized, normalized, weights))
    )


def _comparison_report(
    manifest: dict,
    layouts: dict,
    metadata: dict,
    propagated: dict,
    observable_scales: np.ndarray,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = manifest["prospective_propagation_contract"]
    times = np.asarray(propagated[LABELS[0]]["times"], dtype=float)
    parent_grid = layouts[LABELS[0]].parent_grid
    parent_measures = np.asarray(parent_grid.cell_measures, dtype=float)
    coarse = propagated[LABELS[0]]["parent_primary_physical"]
    medium = propagated[LABELS[1]]["parent_primary_physical"]
    fine = propagated[LABELS[2]]["parent_primary_physical"]
    secondary = propagated[LABELS[2]][
        "parent_secondary_base_physical"
    ]
    restart = propagated[LABELS[2]]["parent_restart_physical"]
    state_gates = {
        "maximum_N128_Richardson_error": (
            c6a2.MAXIMUM_COMPLETE_SEMIGROUP_ERROR
        ),
        "maximum_reference_uncertainty_to_fine_difference": (
            contract["maximum_reference_uncertainty_to_fine_difference"]
        ),
        "maximum_projection_uncertainty_to_fine_difference": 0.10,
        "maximum_restart_uncertainty_to_fine_difference": 0.10,
        "maximum_boundary_integral_uncertainty_to_fine_difference": 0.10,
    }
    reports = {}
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "fixed_physical_observable_scales": observable_scales,
        "field_scales": field_scales,
    }
    instantaneous_matrix = np.empty((len(metadata["packet_ids"]), 6))
    cumulative_matrix = np.empty((len(metadata["packet_ids"]), 6))
    state_matrix = np.empty((len(metadata["packet_ids"]), 8))
    maximum_scaling_defect = 0.0
    for packet_index, packet_id in enumerate(metadata["packet_ids"]):
        instantaneous = _packet_metrics(
            propagated[LABELS[0]]["signals"][packet_index],
            propagated[LABELS[1]]["signals"][packet_index],
            propagated[LABELS[2]]["signals"][packet_index],
            observable_scales,
            contract,
        )
        cumulative = _packet_metrics(
            propagated[LABELS[0]]["cumulative_signals"][packet_index],
            propagated[LABELS[1]]["cumulative_signals"][packet_index],
            propagated[LABELS[2]]["cumulative_signals"][packet_index],
            observable_scales * float(times[-1]),
            contract,
        )
        richardson = causal_windowed_richardson_reference(
            coarse[packet_index],
            medium[packet_index],
            fine[packet_index],
            times=times,
            coarse_cell_measures=parent_measures,
            field_scales=field_scales,
        )
        fine_state_difference = max(
            richardson.medium_fine_history_norm,
            np.finfo(float).tiny,
        )
        secondary_history = (
            metadata["multipliers"][packet_index]
            * secondary[metadata["base_indices"][packet_index]]
        )
        projection_ratio = (
            causal_field_history_norm(
                fine[packet_index] - secondary_history,
                cell_measures=parent_measures,
                field_scales=field_scales,
                time_weights=causal_trapezoid_weights(times),
            )
            / fine_state_difference
        )
        restart_ratio = (
            causal_field_history_norm(
                np.stack(
                    (
                        restart[packet_index] - fine[packet_index, -1],
                    )
                    * 2,
                    axis=0,
                ),
                cell_measures=parent_measures,
                field_scales=field_scales,
                time_weights=np.ones(2),
            )
            / fine_state_difference
        )
        medium_boundary = propagated[LABELS[1]][
            "cumulative_signals"
        ][packet_index, :, :6]
        fine_boundary = propagated[LABELS[2]][
            "cumulative_signals"
        ][packet_index, :, :6]
        boundary_difference = max(
            _history_norm(
                medium_boundary - fine_boundary,
                observable_scales[:6],
                times,
            ),
            np.finfo(float).tiny,
        )
        boundary_ratio = (
            _history_norm(
                propagated[LABELS[2]]["cumulative_corrections"][
                    packet_index, :, :6
                ],
                observable_scales[:6],
                times,
            )
            / boundary_difference
        )
        state_parent_replay = bool(
            richardson.observed_order >= c6a2.MINIMUM_CROSS_GRID_ORDER
            and richardson.minimum_significant_component_order
            >= c6a2.MINIMUM_COMPONENT_ORDER
            and richardson.refinement_error_cosine
            >= c6a2.MINIMUM_REFINEMENT_ERROR_COSINE
        )
        state_passed = bool(
            state_parent_replay
            and richardson.maximum_coarse_reference_relative_error
            <= state_gates["maximum_N128_Richardson_error"]
            and richardson.reference_choice_to_fine_difference_ratio
            <= state_gates[
                "maximum_reference_uncertainty_to_fine_difference"
            ]
            and projection_ratio
            <= state_gates[
                "maximum_projection_uncertainty_to_fine_difference"
            ]
            and restart_ratio
            <= state_gates[
                "maximum_restart_uncertainty_to_fine_difference"
            ]
            and boundary_ratio
            <= state_gates[
                "maximum_boundary_integral_uncertainty_to_fine_difference"
            ]
        )
        base_variant = metadata["base_variant_indices"][
            metadata["base_indices"][packet_index]
        ]
        factor = metadata["multipliers"][packet_index]
        scaling_defects = {}
        for label in LABELS:
            for quantity in (
                "primary_physical",
                "signals",
                "cumulative_signals",
            ):
                defect = _relative_defect(
                    propagated[label][quantity][packet_index],
                    factor
                    * propagated[label][quantity][base_variant],
                )
                scaling_defects[f"{label}::{quantity}"] = defect
                maximum_scaling_defect = max(
                    maximum_scaling_defect,
                    defect,
                )
        packet_passed = bool(
            instantaneous.passed
            and cumulative.passed
            and state_passed
            and max(scaling_defects.values())
            <= MAXIMUM_PROPAGATION_SCALING_DEFECT
        )
        reports[packet_id] = {
            "manifest_variant": metadata["variants"][packet_index],
            "instantaneous_exports": _metric_payload(instantaneous),
            "cumulative_exports": _metric_payload(cumulative),
            "state_reference": {
                "parent_state_convergence_replayed": state_parent_replay,
                "observed_order": richardson.observed_order,
                "minimum_significant_component_order": (
                    richardson.minimum_significant_component_order
                ),
                "refinement_error_cosine": (
                    richardson.refinement_error_cosine
                ),
                "maximum_N128_Richardson_error": (
                    richardson.maximum_coarse_reference_relative_error
                ),
                "reference_uncertainty_to_fine_difference": (
                    richardson.reference_choice_to_fine_difference_ratio
                ),
                "projection_uncertainty_to_fine_difference": (
                    projection_ratio
                ),
                "restart_uncertainty_to_fine_difference": restart_ratio,
                "boundary_integral_uncertainty_to_fine_difference": (
                    boundary_ratio
                ),
                "passed": state_passed,
            },
            "propagation_scaling_defects": scaling_defects,
            "passed": packet_passed,
        }
        instantaneous_matrix[packet_index] = (
            instantaneous.observed_rms_order,
            instantaneous.observed_maximum_order,
            instantaneous.minimum_significant_component_order,
            instantaneous.maximum_fine_normalized_difference,
            instantaneous.history_cosine,
            instantaneous.refinement_error_cosine,
        )
        cumulative_matrix[packet_index] = (
            cumulative.observed_rms_order,
            cumulative.observed_maximum_order,
            cumulative.minimum_significant_component_order,
            cumulative.maximum_fine_normalized_difference,
            cumulative.history_cosine,
            cumulative.refinement_error_cosine,
        )
        state_matrix[packet_index] = (
            richardson.observed_order,
            richardson.minimum_significant_component_order,
            richardson.refinement_error_cosine,
            richardson.maximum_coarse_reference_relative_error,
            richardson.reference_choice_to_fine_difference_ratio,
            projection_ratio,
            restart_ratio,
            boundary_ratio,
        )
    base_indices = metadata["base_variant_indices"]
    for label in LABELS:
        arrays[f"{label}__base_instantaneous_exports"] = propagated[
            label
        ]["signals"][base_indices]
        arrays[f"{label}__base_cumulative_exports"] = propagated[label][
            "cumulative_signals"
        ][base_indices]
        arrays[f"{label}__base_parent_state_histories"] = propagated[
            label
        ]["parent_primary_physical"][base_indices]
        arrays[f"{label}__base_common_face_fluxes"] = propagated[label][
            "base_common_face_fluxes"
        ]
    arrays["instantaneous_metric_matrix"] = instantaneous_matrix
    arrays["cumulative_metric_matrix"] = cumulative_matrix
    arrays["state_metric_matrix"] = state_matrix
    arrays["direct_packet_pass_flags"] = np.asarray(
        [reports[name]["passed"] for name in metadata["packet_ids"]],
        dtype=np.int8,
    )
    return {
        "packet_reports": reports,
        "packet_count": len(metadata["packet_ids"]),
        "maximum_propagation_scaling_defect": maximum_scaling_defect,
        "failed_packets": [
            name for name in metadata["packet_ids"] if not reports[name]["passed"]
        ],
        "all_packets_passed": all(
            reports[name]["passed"] for name in metadata["packet_ids"]
        ),
        "passed": bool(
            all(reports[name]["passed"] for name in metadata["packet_ids"])
            and maximum_scaling_defect
            <= MAXIMUM_PROPAGATION_SCALING_DEFECT
        ),
        "state_gates": state_gates,
    }, arrays


def _conditioning_report(
    manifest: dict,
    layouts: dict,
    metadata: dict,
    propagated: dict,
    f1: dict,
    observable_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = manifest["component_route_contract"]
    parent_grid = layouts[LABELS[0]].parent_grid
    band_indices, band_edges = c6d._band_edges(parent_grid)
    times = np.asarray(propagated[LABELS[0]]["times"], dtype=float)
    weights = causal_trapezoid_weights(times)
    physical_scale = float(observable_scales[TARGET_OBSERVABLE_INDEX])
    reports = {}
    arrays: dict[str, np.ndarray] = {
        "conditioning_band_indices": band_indices,
        "conditioning_band_edges_over_rg": band_edges,
    }
    for profile_index, name in enumerate(metadata["base_names"]):
        reports[name] = {}
        continuum_ratio = f1["continuum_reference_report"][name][
            "uncertainty_to_fine_difference"
        ]
        for history_name, source_key in (
            ("instantaneous", "base_parent_cell_actions"),
            ("cumulative", "base_parent_cumulative_cell_actions"),
        ):
            histories = tuple(
                propagated[label][source_key][
                    profile_index, :, :, ANGULAR_FIELD
                ]
                for label in LABELS
            )
            bands = tuple(
                causal_partition_cell_integrals(values, band_indices)
                for values in histories
            )
            global_metrics = c6d._scalar_metrics(
                np.sum(histories[0], axis=-1),
                np.sum(histories[1], axis=-1),
                np.sum(histories[2], axis=-1),
                physical_scale=physical_scale,
            )
            cell_metrics = [
                c6d._scalar_metrics(
                    histories[0][:, index],
                    histories[1][:, index],
                    histories[2][:, index],
                    physical_scale=physical_scale,
                )
                for index in range(histories[0].shape[1])
            ]
            band_metrics = [
                c6d._scalar_metrics(
                    bands[0][:, index],
                    bands[1][:, index],
                    bands[2][:, index],
                    physical_scale=physical_scale,
                )
                for index in range(bands[0].shape[1])
            ]
            cell_response = np.maximum.reduce(
                tuple(np.max(np.abs(values), axis=0) for values in histories)
            ) / physical_scale
            band_response = np.maximum.reduce(
                tuple(np.max(np.abs(values), axis=0) for values in bands)
            ) / physical_scale
            active_cells = cell_response >= contract["minimum_relative_activity"]
            active_bands = band_response >= contract["minimum_relative_activity"]
            first_band_errors = bands[1] - bands[0]
            second_band_errors = bands[2] - bands[1]
            gram = causal_signed_band_gram_matrix(
                second_band_errors,
                physical_scale=physical_scale,
                time_weights=weights,
            )
            signed_norm_squared = float(
                np.sum(
                    weights
                    * (
                        np.sum(second_band_errors, axis=1)
                        / physical_scale
                    )
                    ** 2
                )
            )
            gram_closure = abs(float(np.sum(gram)) - signed_norm_squared) / max(
                abs(float(np.sum(gram))),
                abs(signed_norm_squared),
                np.finfo(float).tiny,
            )
            plus_variant = int(metadata["base_variant_indices"][profile_index])
            signal_key = (
                "signals"
                if history_name == "instantaneous"
                else "cumulative_signals"
            )
            parity = max(
                _relative_defect(
                    np.sum(values, axis=-1),
                    propagated[label][signal_key][
                        plus_variant, :, TARGET_OBSERVABLE_INDEX
                    ],
                )
                for label, values in zip(LABELS, histories, strict=True)
            )
            coarse_cancellation = c6e1.causal_cancellation_ratio(
                first_band_errors,
                time_weights=weights,
            )
            fine_cancellation = c6e1.causal_cancellation_ratio(
                second_band_errors,
                time_weights=weights,
            )
            envelope = c6e1.causal_absolute_band_error_envelope(
                second_band_errors,
                physical_scale=physical_scale,
            )
            if np.any(active_cells) and np.any(active_bands):
                decision = c6e1.causal_integral_conditioning_decision(
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
                    coarse_medium_cancellation_ratio=coarse_cancellation,
                    medium_fine_cancellation_ratio=fine_cancellation,
                    direct_sum_defect=parity,
                    gram_closure_defect=gram_closure,
                    continuum_uncertainty_to_fine=continuum_ratio,
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
                    "active_band_count": int(np.count_nonzero(active_bands)),
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
                "continuum_uncertainty_to_fine_difference": continuum_ratio,
            }
            prefix = f"{name}__{history_name}__conditioning_"
            arrays[prefix + "band_histories_coarse"] = bands[0]
            arrays[prefix + "band_histories_medium"] = bands[1]
            arrays[prefix + "band_histories_fine"] = bands[2]
            arrays[prefix + "medium_fine_band_gram"] = gram
            arrays[prefix + "active_cells"] = active_cells.astype(np.int8)
            arrays[prefix + "active_bands"] = active_bands.astype(np.int8)
    return reports, arrays


def _prospective_decision(
    manifest: dict,
    direct: dict,
    conditioning: dict,
    metadata: dict,
) -> dict:
    propagation = manifest["prospective_propagation_contract"]
    reports = {}
    direct_bases = set()
    alternate_bases = set()
    for packet_id, base_index, multiplier in zip(
        metadata["packet_ids"],
        metadata["base_indices"],
        metadata["multipliers"],
        strict=True,
    ):
        name = metadata["base_names"][int(base_index)]
        parent = direct["packet_reports"][packet_id]
        if parent["passed"]:
            passed = True
            route = "historical_direct_contract"
            direct_bases.add(name)
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
                    if order
                    < propagation["minimum_significant_component_order"]
                }
                if low and low != {TARGET_OBSERVABLE_NAME}:
                    histories_pass = False
                if TARGET_OBSERVABLE_NAME in low:
                    decision = conditioning[name][history_name]["decision"]
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
                        >= propagation[
                            "minimum_significant_component_order"
                        ]
                    )
                histories_pass = bool(
                    histories_pass
                    and c6e1._other_export_gates_pass(metric, propagation)
                )
            passed = bool(
                histories_pass
                and used_alternate
                and parent["state_reference"]["passed"]
                and max(parent["propagation_scaling_defects"].values())
                <= MAXIMUM_PROPAGATION_SCALING_DEFECT
            )
            route = (
                "proof_style_cancellation_conditioned_band_envelope"
                if passed
                else "failed"
            )
            if passed:
                alternate_bases.add(name)
        reports[packet_id] = {
            "base_profile": name,
            "multiplier": float(multiplier),
            "historical_direct_passed": parent["passed"],
            "route": route,
            "passed": passed,
        }
    return {
        "variant_reports": reports,
        "all_variants_passed": all(
            item["passed"] for item in reports.values()
        ),
        "direct_base_profiles": sorted(direct_bases),
        "alternate_base_profiles": sorted(alternate_bases),
        "direct_variant_count": sum(
            item["route"] == "historical_direct_contract"
            for item in reports.values()
        ),
        "alternate_variant_count": sum(
            item["route"]
            == "proof_style_cancellation_conditioned_band_envelope"
            for item in reports.values()
        ),
        "failed_variants": sorted(
            name for name, item in reports.items() if not item["passed"]
        ),
        "passed": all(item["passed"] for item in reports.values()),
    }


def _scalar_energy_metric(
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: float,
    contract: dict,
) -> dict:
    response = max(
        float(np.max(np.abs(values))) for values in histories
    ) / max(float(scale), np.finfo(float).tiny)
    if response < contract["minimum_relative_activity"]:
        return {
            "active": False,
            "passed": True,
            "maximum_response_over_scale": response,
        }
    metrics = _packet_metrics(
        histories[0][:, None],
        histories[1][:, None],
        histories[2][:, None],
        np.asarray([scale], dtype=float),
        contract,
    )
    return {
        "active": True,
        "passed": metrics.passed,
        "maximum_response_over_scale": response,
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "minimum_component_order": (
            metrics.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _coupling_diagnostics(
    manifest: dict,
    layouts: dict,
    configurations: dict,
    metadata: dict,
    propagated: dict,
    observable_scales: np.ndarray,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = manifest["prospective_propagation_contract"]
    coupling = manifest["coupling_diagnostic_contract"]
    parent_grid = layouts[LABELS[0]].parent_grid
    parent_base = restrict_causal_embedded_patch_cell_averages(
        configurations[LABELS[0]]["base_primitives"],
        layouts[LABELS[0]],
    )
    physical_bases = []
    maximum_condition = 0.0
    maximum_eigenpair = 0.0
    for radius, chart in zip(
        parent_grid.centers,
        parent_base,
        strict=True,
    ):
        basis = _analytic_coordinate_principal_basis(
            configurations[LABELS[0]]["context"],
            float(radius),
            chart,
        )
        physical_bases.append(basis.primitive_right_eigenvectors)
        maximum_condition = max(
            maximum_condition,
            float(basis.descriptor_condition_number),
        )
        maximum_eigenpair = max(
            maximum_eigenpair,
            float(basis.maximum_eigenpair_defect),
        )
    inverse = causal_dimensionless_characteristic_inverse(
        np.asarray(physical_bases),
        field_scales,
    )
    base_indices = metadata["base_variant_indices"]
    energy_histories = {}
    arrays: dict[str, np.ndarray] = {
        "parent_characteristic_inverse": inverse,
    }
    inner_left, inner_right = coupling[
        "inner_energy_window_parent_faces"
    ]
    outer_left, outer_right = coupling[
        "outer_energy_window_parent_faces"
    ]
    for label in LABELS:
        parent_states = propagated[label][
            "parent_primary_physical"
        ][base_indices]
        energy = np.asarray(
            [
                causal_characteristic_energy_history(
                    parent_states[index],
                    inverse,
                    field_scales,
                    parent_grid.cell_measures,
                )
                for index in range(len(BASE_PROFILES))
            ]
        )
        energy_histories[label] = energy
        arrays[f"{label}__base_characteristic_cell_energy"] = energy

    energy_reports = {}
    energy_passed = True
    for profile_index, name in enumerate(BASE_PROFILES):
        family = name.split("__", 1)[1]
        selected = FAMILY_INDEX[family]
        opposite = OPPOSITE_INDEX.get(family)
        selected_exclusions = {selected}
        if opposite is not None:
            selected_exclusions.add(opposite)
        channels = {}
        for label in LABELS:
            energy = energy_histories[label][profile_index]
            inner = np.sum(energy[:, inner_left:inner_right], axis=1)
            outer = np.sum(energy[:, outer_left:outer_right], axis=1)
            other_indices = [
                index for index in range(5) if index not in selected_exclusions
            ]
            channels[label] = {
                "incident_inner_selected": inner[:, selected],
                "reflected_inner_opposite": (
                    inner[:, opposite]
                    if opposite is not None
                    else np.zeros(inner.shape[0])
                ),
                "transmitted_outer_selected": outer[:, selected],
                "outer_opposite": (
                    outer[:, opposite]
                    if opposite is not None
                    else np.zeros(outer.shape[0])
                ),
                "inner_other": np.sum(inner[:, other_indices], axis=1),
                "outer_other": np.sum(outer[:, other_indices], axis=1),
            }
        initial_total = float(
            np.sum(energy_histories[LABELS[0]][profile_index, 0])
        )
        profile_report = {
            "selected_family": family,
            "opposite_family": (
                "outward_shear"
                if family == "inward_shear"
                else "inward_shear"
                if family == "outward_shear"
                else None
            ),
            "scattering_interpretation": (
                "outward_selected_family_incident_on_coupling"
                if family == "outward_shear"
                else "kinematic_energy_diagnostic_only"
            ),
            "fixed_energy_scale": initial_total,
            "channels": {},
        }
        for channel in channels[LABELS[0]]:
            histories = tuple(
                channels[label][channel] for label in LABELS
            )
            metric = _scalar_energy_metric(
                histories,
                initial_total,
                contract,
            )
            profile_report["channels"][channel] = metric
            energy_passed = bool(energy_passed and metric["passed"])
            arrays[f"{name}__{channel}__histories"] = np.asarray(histories)
        profile_report["passed"] = all(
            item["passed"] for item in profile_report["channels"].values()
        )
        energy_reports[name] = profile_report

    face_reports = {}
    common_faces = tuple(
        manifest["common_surface_contract"]["parent_face_indices"]
    )
    for profile_index, name in enumerate(BASE_PROFILES):
        face_reports[name] = {}
        for position, parent_face in enumerate(common_faces):
            payload = _optional_packet_metric(
                propagated[LABELS[0]]["base_common_face_fluxes"][
                    profile_index, :, position
                ],
                propagated[LABELS[1]]["base_common_face_fluxes"][
                    profile_index, :, position
                ],
                propagated[LABELS[2]]["base_common_face_fluxes"][
                    profile_index, :, position
                ],
                observable_scales[:3],
                contract,
                ("mass", "angular_momentum", "killing_energy"),
            )
            face_reports[name][str(parent_face)] = payload

    interface_cells = np.asarray(
        [c7a.PARENT_COUPLING_FACE - 1, c7a.PARENT_COUPLING_FACE],
        dtype=int,
    )
    interface_reports = {}
    interface_passed = True
    interface_scales = np.tile(field_scales, 2)
    for profile_index, name in enumerate(BASE_PROFILES):
        histories = tuple(
            propagated[label]["parent_primary_physical"][
                metadata["base_variant_indices"][profile_index],
                :,
                interface_cells,
                :,
            ].reshape(
                propagated[label]["times"].size,
                -1,
            )
            for label in LABELS
        )
        payload = _optional_packet_metric(
            histories[0],
            histories[1],
            histories[2],
            interface_scales,
            contract,
            tuple(
                f"{side}_{field}"
                for side in ("inner", "outer")
                for field in (
                    "surface_density",
                    "radial_velocity",
                    "azimuthal_velocity",
                    "pressure",
                    "stress",
                )
            ),
        )
        interface_reports[name] = payload
        interface_passed = bool(interface_passed and payload["passed"])
        arrays[f"{name}__interface_state_histories"] = np.asarray(histories)

    coupling_face_position = common_faces.index(
        c7a.PARENT_COUPLING_FACE
    )
    coupling_face_passed = all(
        face_reports[name][str(c7a.PARENT_COUPLING_FACE)]["passed"]
        for name in BASE_PROFILES
    )
    del coupling_face_position
    passed = bool(
        energy_passed and interface_passed and coupling_face_passed
    )
    return {
        "passed": passed,
        "characteristic_energy_method": (
            "complete_coordinate_descriptor_pencil_eigenvectors_"
            "normalized_by_fixed_physical_field_scales"
        ),
        "maximum_characteristic_descriptor_condition": maximum_condition,
        "maximum_characteristic_eigenpair_defect": maximum_eigenpair,
        "energy_reports": energy_reports,
        "energy_convergence_passed": energy_passed,
        "common_face_flux_reports": face_reports,
        "coupling_face_flux_convergence_passed": coupling_face_passed,
        "interface_state_reports": interface_reports,
        "interface_state_convergence_passed": interface_passed,
        "absolute_reflection_threshold_applied": False,
    }, arrays


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "labels": LABELS,
        "base_profiles": BASE_PROFILES,
        "profile_variants": manifest["profile_contract"]["profile_variants"],
        "observable_names": OBSERVABLE_NAMES,
        "observable_contract": manifest["observable_contract"],
        "coupling_diagnostic_contract": (
            manifest["coupling_diagnostic_contract"]
        ),
        "component_route_contract": manifest["component_route_contract"],
        "prospective_propagation_contract": (
            manifest["prospective_propagation_contract"]
        ),
    }


def _interface_localization_report(
    arrays: dict[str, np.ndarray],
) -> dict:
    """Condition the embedded refinement error by physical export sector.

    This is a post-result localization diagnostic.  It does not alter the
    frozen propagation gate or provide an alternate route to certification.
    """

    scales = np.asarray(
        arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    groups = {
        "all_active_exports": tuple(range(len(OBSERVABLE_NAMES))),
        "inner_face_and_distributed": (0, 1, 2, 9, 10, 11, 12),
        "coupling_face_only": (3, 4, 5),
        "coupling_face_and_net_drive": (3, 4, 5, 6, 7, 8),
    }

    def cosine(
        coarse_medium: np.ndarray,
        medium_fine: np.ndarray,
        indices: tuple[int, ...],
    ) -> float | None:
        left = np.asarray(
            coarse_medium[:, indices],
            dtype=float,
        ).reshape(-1)
        right = np.asarray(
            medium_fine[:, indices],
            dtype=float,
        ).reshape(-1)
        denominator = float(
            np.linalg.norm(left) * np.linalg.norm(right)
        )
        if denominator == 0.0:
            return None
        return float(np.dot(left, right) / denominator)

    profiles = {}
    for profile_index, profile_name in enumerate(BASE_PROFILES):
        histories = [
            np.asarray(
                arrays[f"{label}__base_instantaneous_exports"][
                    profile_index
                ],
                dtype=float,
            )
            for label in LABELS
        ]
        coarse_medium = (histories[1] - histories[0]) / scales
        medium_fine = (histories[2] - histories[1]) / scales
        profiles[profile_name] = {
            group_name: cosine(
                coarse_medium,
                medium_fine,
                indices,
            )
            for group_name, indices in groups.items()
        }
    return {
        "binding": False,
        "post_result_diagnostic_only": True,
        "physical_scaling": "frozen_13_observable_scales",
        "profiles": profiles,
    }


def _parent_hashes() -> dict[str, str]:
    paths = (
        PARENT_CONFIG,
        PARENT_SUMMARY,
        PARENT_MANIFEST,
        PARENT_ARRAYS,
        PARENT_PROVENANCE,
        C0E_CONTEXTS,
        C0E_INPUTS,
        F1_SUMMARY,
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
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
        "uniform_certification_preserved": True,
        "historical_classifications_preserved": True,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_path": str(DECISIVE_ARRAYS.relative_to(ROOT)),
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
            "PROSPECTIVE EMBEDDED CLASS CERTIFIED"
            if result.get("passed")
            else "REJECTED OR STOPPED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_embedded_validation_wp10c9d6c7b.py"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": _parent_hashes(),
    }
    _write_json(CONFIG_PATH, _config(manifest))
    _write_json(SUMMARY_PATH, result)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return result


def run(*, reuse_propagation_checkpoint: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, manifest, arrays, f1 = _load_frozen_inputs()
    configurations, layouts, metadata = _build_inputs(manifest, arrays)
    result_arrays: dict[str, np.ndarray] = {
        "times": np.asarray(arrays["times"], dtype=float),
        "fixed_physical_observable_scales": np.asarray(
            arrays["fixed_physical_observable_scales"],
            dtype=float,
        ),
        "field_scales": np.asarray(arrays["field_scales"], dtype=float),
        "common_parent_face_indices": np.asarray(
            arrays["common_parent_face_indices"],
            dtype=int,
        ),
        "common_face_radii_over_rg": np.asarray(
            arrays["common_face_radii_over_rg"],
            dtype=float,
        ),
    }
    if reuse_propagation_checkpoint:
        propagated, propagation_report, method_reports = (
            _load_propagation_checkpoint()
        )
        method_passed = all(
            item["passed"] for item in method_reports.values()
        )
    else:
        tangents, active_audits, method_reports, method_passed = (
            _build_tangents(configurations)
        )
        if not method_passed:
            return _finalize(
                identity=identity,
                parent=parent,
                manifest=manifest,
                arrays=result_arrays,
                summary={
                    "classification": (
                        "embedded_monolithic_method_gate_failed"
                    ),
                    "authorized_next": None,
                    "passed": False,
                    "propagation_executed": False,
                    "method_reports": method_reports,
                    "method_passed": False,
                    "embedded_profile_class_certified": False,
                    "bounded_nonlinear_common_mode_authorized": False,
                    "runtime_seconds": float(time.perf_counter() - started),
                },
            )
        directions = _variant_directions(
            configurations,
            arrays,
            metadata,
        )
        propagated, propagation_report = _propagate(
            configurations,
            layouts,
            tangents,
            active_audits,
            directions,
            metadata,
            manifest,
        )
        _save_propagation_checkpoint(
            propagated,
            propagation_report,
            method_reports,
        )
    observable_scales = np.asarray(
        arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    field_scales = np.asarray(arrays["field_scales"], dtype=float)
    direct, direct_arrays = _comparison_report(
        manifest,
        layouts,
        metadata,
        propagated,
        observable_scales,
        field_scales,
    )
    conditioning, conditioning_arrays = _conditioning_report(
        manifest,
        layouts,
        metadata,
        propagated,
        f1,
        observable_scales,
    )
    prospective = _prospective_decision(
        manifest,
        direct,
        conditioning,
        metadata,
    )
    coupling, coupling_arrays = _coupling_diagnostics(
        manifest,
        layouts,
        configurations,
        metadata,
        propagated,
        observable_scales,
        field_scales,
    )
    result_arrays.update(direct_arrays)
    result_arrays.update(conditioning_arrays)
    result_arrays.update(coupling_arrays)
    interface_localization = _interface_localization_report(result_arrays)
    maximum_integral_residual = max(
        item["maximum_exact_integral_relative_solve_residual"]
        for item in propagation_report.values()
    )
    passed = bool(
        method_passed
        and maximum_integral_residual <= MAXIMUM_EXACT_INTEGRAL_RESIDUAL
        and prospective["passed"]
        and coupling["passed"]
    )
    classification = (
        "embedded_operator_certified_for_declared_resolved_profile_class"
        if passed
        else "prospective_embedded_profile_validation_failed"
    )
    authorized_next = (
        "WP10c9d6c8_bounded_nonlinear_common_mode_preflight"
        if passed
        else None
    )
    result = _finalize(
        identity=identity,
        parent=parent,
        manifest=manifest,
        arrays=result_arrays,
        summary={
            "classification": classification,
            "authorized_next": authorized_next,
            "passed": passed,
            "propagation_executed": True,
            "method_reports": method_reports,
            "method_passed": method_passed,
            "propagation_report": propagation_report,
            "maximum_exact_integral_relative_solve_residual": (
                maximum_integral_residual
            ),
            "historical_direct_contract_report": direct,
            "band_envelope_report": conditioning,
            "prospective_decision": prospective,
            "coupling_diagnostic_report": coupling,
            "interface_localization_report": interface_localization,
            "embedded_profile_class_certified": passed,
            "bounded_nonlinear_common_mode_authorized": passed,
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "authorized_next": authorized_next,
                "direct_failed_count": len(direct["failed_packets"]),
                "prospective_failed_count": len(
                    prospective["failed_variants"]
                ),
                "direct_variant_count": prospective[
                    "direct_variant_count"
                ],
                "alternate_variant_count": prospective[
                    "alternate_variant_count"
                ],
                "coupling_diagnostics_passed": coupling["passed"],
                "maximum_exact_integral_residual": (
                    maximum_integral_residual
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def refresh_metadata_only() -> dict:
    if not SUMMARY_PATH.is_file() or not DECISIVE_ARRAYS.is_file():
        raise RuntimeError("WP10c9d6c7b canonical evidence is unavailable")
    identity = _validate_analyzed_git_identity()
    parent, manifest, _parent_arrays, _f1 = _load_frozen_inputs()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    arrays = _load_npz(DECISIVE_ARRAYS)
    immutable = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "implementation_source_hashes",
            "implementation_source_manifest_sha256",
            "decisive_arrays_path",
            "decisive_arrays_sha256",
            "decisive_array_hashes",
        }
    }
    immutable["interface_localization_report"] = (
        _interface_localization_report(arrays)
    )
    return _finalize(
        identity=identity,
        parent=parent,
        manifest=manifest,
        summary=immutable,
        arrays=arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-metadata-only", action="store_true")
    parser.add_argument(
        "--reuse-propagation-checkpoint",
        action="store_true",
    )
    arguments = parser.parse_args()
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    else:
        run(
            reuse_propagation_checkpoint=(
                arguments.reuse_propagation_checkpoint
            )
        )


if __name__ == "__main__":
    main()
