#!/usr/bin/env python3
"""Certify the fixed-N98-exterior driven inner-continuum reference.

WP10c9d6c7c2c2 changes no physical or numerical operator and runs no
embedded state.  The unchanged complete N98 tangent supplies a one-way
five-field interface drive to independent N513 and N769 inner collocation
systems.  This is the matched reference required by the frozen c2c1
embedded contract.
"""

from __future__ import annotations

from dataclasses import replace
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
from scipy.interpolate import make_interp_spline
from scipy.sparse import bmat, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d as b6d  # noqa: E402
import run_causal_inner_direct_continuum_embedded_manifest_wp10c9d6c7c2c1 as c2c1  # noqa: E402
import run_causal_inner_direct_continuum_uniform_recertification_wp10c9d6c7c2b6e as b6e  # noqa: E402
import run_causal_inner_revised_uniform_arrival_transfer_wp10c9d6c7c2b6b as b6b  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b as b5b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
    causal_five_field_inward_collocation_generator_blocks,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_linear_tangent import (  # noqa: E402
    _frozen_quadratic_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2c2"
ANALYZED_BASE_COMMIT = "e763b39e598ad8302b2918220fa4dc4a39533363"
ANALYZED_BASE_PARENT = "d29cba5d5bfe5950cba4e458a6ff8b458e1364de"
ANALYZED_BASE_TREE = "8493640db7dc58b5bc92dc22a195715a0b1d9722"

FIELDS = 5
DRIVER_CELLS = 98
INTERFACE_FACE = 49
INNER_NODES = (513, 769)
BOUNDARY_SAMPLES = 5
TIME_SAMPLES = 513
BASES = b6d.BINDING_BASES

MAXIMUM_ACTION_DIFFERENCE = 2.0e-5
MAXIMUM_TRACE_REPLAY_DEFECT = 1.0e-10
MAXIMUM_BOUNDARY_CLOSURE_DEFECT = 1.0e-10
MAXIMUM_ENERGY_LEDGER_DEFECT = 1.0e-10
RESTART_REPLAY_TOLERANCE = 2.0e-11
EXPECTED_INTERFACE_INCOMING = 5
EXPECTED_INNER_INCOMING = 0
COMPATIBLE_RAW_RUNNER_HASHES = (
    "6385e8c971076594579f84ca0188bfbb6028e6b7efad6271a4ec25f1d094f54a",
    "b54bfbcf7966db592cdc5203204c073c0690aaacd5acccc54de493d631198bbb",
)
COMPATIBLE_REFERENCE_RUNNER_HASHES = (
    "b54bfbcf7966db592cdc5203204c073c0690aaacd5acccc54de493d631198bbb",
)

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_fixed_exterior_continuum_reference_"
    "wp10c9d6c7c2c2.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_fixed_exterior_continuum_reference_"
    "wp10c9d6c7c2c2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FIXED_EXTERIOR_CONTINUUM_REFERENCE_"
    "WP10C9D6C7C2C2_RESULTS_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c2c1.CANONICAL_DIRECTORY
B6D_DIRECTORY = b6d.CANONICAL_DIRECTORY
C2A2_DIRECTORY = c2a2.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_fixed_exterior_continuum_reference_"
    "wp10c9d6c7c2c2"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_fixed_exterior_continuum_reference_"
    "wp10c9d6c7c2c2"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

DRIVER_TANGENT = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1/N98.npz"
)
DRIVER_ENERGY_OPERATOR = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_one_way_transmission_interpretation_"
    "wp10c9d6c7c2b2/N98.npz"
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _validate_parent() -> tuple[dict, dict, dict[str, np.ndarray]]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(PARENT_DIRECTORY / "embedded_manifest.json")
    arrays = _load_npz(PARENT_DIRECTORY / "decisive_arrays.npz")
    if (
        summary["classification"]
        != "direct_continuum_embedded_contract_frozen_"
        "fixed_exterior_reference_preflight_authorized"
        or not summary["passed"]
        or summary["propagation_executed"]
        or summary["authorized_next"]
        != "WP10c9d6c7c2c2_fixed_exterior_continuum_"
        "reference_preflight"
        or manifest["matched_reference_contract"][
            "mandatory_preflight_gates"
        ]["maximum_N769_N513_action_difference"]
        != MAXIMUM_ACTION_DIFFERENCE
    ):
        raise RuntimeError("WP10c9d6c7c2c1 binding status changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c2c2 analyzed identity changed")
    return summary, manifest, arrays


def _interpolation_weights(
    nodes: np.ndarray,
    target: float,
    *,
    degree: int = 5,
) -> np.ndarray:
    values = np.asarray(nodes, dtype=float)
    indices = np.sort(np.argsort(np.abs(values - target))[: degree + 1])
    offsets = values[indices] - float(target)
    moments = np.vstack([offsets**power for power in range(degree + 1)])
    right = np.zeros(degree + 1, dtype=float)
    right[0] = 1.0
    local = np.linalg.solve(moments, right)
    result = np.zeros(values.size, dtype=float)
    result[indices] = local
    return result


def _driver_data(
    parent_context,
    parent_base: np.ndarray,
    field_scales: np.ndarray,
) -> dict:
    c2a2_arrays = _load_npz(C2A2_DIRECTORY / "decisive_arrays.npz")
    edges = np.asarray(c2a2_arrays["patch_edges"], dtype=float)
    grid = make_kerr_schild_column_grid_from_edges(
        edges, parent_context.grid.gravitational_radius
    )
    context = replace(
        parent_context, grid=grid, stream_sources=None
    ).validated()
    charts = np.asarray(
        c2a2_arrays["manufactured_primitive_charts"], dtype=float
    )
    if not DRIVER_TANGENT.is_file() or not DRIVER_ENERGY_OPERATOR.is_file():
        raise RuntimeError(
            "certified N98 tangent checkpoints are absent; rerun c2b1/c2b2"
        )
    tangent = _load_npz(DRIVER_TANGENT)
    energy_operator = _load_npz(DRIVER_ENERGY_OPERATOR)
    left, right, reconstruction_defect = (
        _frozen_quadratic_reconstruction_weights(context, charts)
    )
    if reconstruction_defect > MAXIMUM_TRACE_REPLAY_DEFECT:
        raise RuntimeError("N98 reconstruction replay failed")
    b6d_arrays = _load_npz(B6D_DIRECTORY / "decisive_arrays.npz")
    driver_scales = np.tile(field_scales, DRIVER_CELLS)
    initial = np.column_stack(
        [
            np.asarray(b6d_arrays[f"packet__{name}"], dtype=float).ravel()
            / driver_scales
            for name in ("acoustic", "shear")
        ]
    )
    return {
        "edges": edges,
        "grid": grid,
        "context": context,
        "charts": charts,
        "generator": csr_matrix(tangent["generator"]),
        "columns": np.asarray(tangent["columns"], dtype=float),
        "initial": initial,
        "right_reconstruction_weights": right,
        "left_reconstruction_weights": left,
        "shared_face_flux_map": np.asarray(
            energy_operator["face_maps"][INTERFACE_FACE], dtype=float
        ),
        "reconstruction_defect": reconstruction_defect,
    }


def _boundary_sample_weights(driver: dict, spacing: float) -> np.ndarray:
    log_centers = np.log(np.asarray(driver["grid"].centers, dtype=float))
    interface = float(np.log(driver["edges"][INTERFACE_FACE]))
    weights = np.zeros(
        (BOUNDARY_SAMPLES, DRIVER_CELLS), dtype=float
    )
    weights[0] = np.asarray(
        driver["right_reconstruction_weights"][INTERFACE_FACE], dtype=float
    )
    for sample in range(1, BOUNDARY_SAMPLES):
        weights[sample] = _interpolation_weights(
            log_centers, interface + sample * spacing
        )
    return weights


def _reference_checkpoint_valid(path: Path, nodes: int) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    report = _read_json(metadata)
    return bool(
        report.get("source_parent_commit") == ANALYZED_BASE_COMMIT
        and report.get("nodes") == nodes
        and report.get("schema_version") == SCHEMA_VERSION
        and report.get("runner_sha256")
        in (
            _sha256(ROOT / THIS_RUNNER),
            *COMPATIBLE_REFERENCE_RUNNER_HASHES,
        )
    )


def _combine_basis(
    acoustic: np.ndarray,
    shear: np.ndarray,
    coefficients: dict[str, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [
            pair[0] * acoustic + pair[1] * shear
            for pair in coefficients.values()
        ],
        axis=1,
    )


def _reference(
    nodes: int,
    driver: dict,
    parent_context,
    parent_base: np.ndarray,
    field_scales: np.ndarray,
    coefficients: dict[str, np.ndarray],
    target_indices: dict[str, list[int]],
    initial_energies: np.ndarray,
    times: np.ndarray,
) -> dict:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIRECTORY / f"matched_reference_N{nodes}.npz"
    if _reference_checkpoint_valid(path, nodes):
        stored = _load_npz(path)
        metadata = _read_json(path.with_suffix(".json"))
        return {**stored, "report": metadata["report"]}

    print(f"{WORK_PACKAGE}: build driven N{nodes} reference", flush=True)
    edges = np.asarray(driver["edges"], dtype=float)
    lower = float(np.log(edges[0]))
    interface = float(np.log(edges[INTERFACE_FACE]))
    spacing = (interface - lower) / float(nodes - 1)
    extended_upper = interface + (BOUNDARY_SAMPLES - 1) * spacing
    auxiliary_grid = make_kerr_schild_column_grid_from_edges(
        np.exp(np.linspace(lower, extended_upper, 18)),
        driver["grid"].gravitational_radius,
    )
    inner_context = replace(
        driver["context"], grid=auxiliary_grid
    ).validated()
    background = build_causal_five_field_continuum_background(
        inner_context,
        b5b._background_evaluator(
            parent_context, parent_base, field_scales
        ),
        node_count=nodes + BOUNDARY_SAMPLES - 1,
    )
    metric, projectors, energy_report = b6b._continuum_energy_basis(
        background, field_scales
    )
    generator_blocks = (
        causal_five_field_inward_collocation_generator_blocks(background)
    )
    extended_nodes = nodes + BOUNDARY_SAMPLES - 1
    dynamic_nodes = nodes - 1
    extended_scales = np.tile(field_scales, extended_nodes)
    boundary_weights = _boundary_sample_weights(driver, spacing)
    boundary_map = kron(
        csr_matrix(boundary_weights),
        eye(FIELDS, format="csr"),
        format="csr",
    )
    scaled_blocks = {}
    for name, block in generator_blocks.items():
        scaled = (
            diags(1.0 / extended_scales)
            @ block
            @ diags(extended_scales)
        ).tocsr()
        scaled_blocks[name] = {
            "inner": scaled[
                : FIELDS * dynamic_nodes, : FIELDS * dynamic_nodes
            ].tocsr(),
            "drive": (
                scaled[
                    : FIELDS * dynamic_nodes,
                    FIELDS * dynamic_nodes :,
                ]
                @ boundary_map
            ).tocsr(),
        }
    inner = sum(
        (item["inner"] for item in scaled_blocks.values()),
        start=csr_matrix(
            (FIELDS * dynamic_nodes, FIELDS * dynamic_nodes),
            dtype=float,
        ),
    ).tocsr()
    drive = sum(
        (item["drive"] for item in scaled_blocks.values()),
        start=csr_matrix(
            (FIELDS * dynamic_nodes, FIELDS * DRIVER_CELLS),
            dtype=float,
        ),
    ).tocsr()
    combined = bmat(
        [[driver["generator"], None], [drive, inner]], format="csr"
    )
    initial = np.vstack(
        (
            driver["initial"],
            np.zeros((FIELDS * dynamic_nodes, 2), dtype=float),
        )
    )
    horizon = float(times[-1])
    trace = float(np.sum(combined.diagonal()))
    raw_path = path.with_name(path.stem + "_raw.npz")
    raw_metadata = raw_path.with_suffix(".json")
    raw_valid = False
    if raw_path.is_file() and raw_metadata.is_file():
        raw_report = _read_json(raw_metadata)
        raw_valid = bool(
            raw_report.get("source_parent_commit") == ANALYZED_BASE_COMMIT
            and raw_report.get("nodes") == nodes
            and raw_report.get("runner_sha256")
            in (
                _sha256(ROOT / THIS_RUNNER),
                *COMPATIBLE_RAW_RUNNER_HASHES,
            )
        )
    if raw_valid:
        raw = _load_npz(raw_path)
        scaled_history = np.asarray(raw["scaled_history"], dtype=float)
        restart_defect = float(raw["restart_defect"][0])
    else:
        print(
            f"{WORK_PACKAGE}: propagate driven N{nodes} reference",
            flush=True,
        )
        scaled_history = np.asarray(
            expm_multiply(
                combined,
                initial,
                start=0.0,
                stop=horizon,
                num=times.size,
                endpoint=True,
                traceA=trace,
            ),
            dtype=float,
        )
        half = np.asarray(
            scaled_history[(times.size - 1) // 2], dtype=float
        )
        restarted = np.asarray(
            expm_multiply(
                0.5 * horizon * combined,
                half,
                traceA=0.5 * horizon * trace,
            ),
            dtype=float,
        )
        restart_defect = _relative_defect(
            restarted, scaled_history[-1]
        )
        np.savez_compressed(
            raw_path,
            scaled_history=scaled_history,
            restart_defect=np.asarray([restart_defect]),
        )
        _write_json(
            raw_metadata,
            {
                "source_parent_commit": ANALYZED_BASE_COMMIT,
                "nodes": nodes,
                "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            },
        )

    driver_scaled = scaled_history[:, : FIELDS * DRIVER_CELLS]
    driver_physical = (
        driver_scaled
        * np.tile(field_scales, DRIVER_CELLS)[None, :, None]
    )
    boundary_scaled = np.asarray(
        np.einsum(
            "ij,tjk->tik",
            boundary_map.toarray(),
            driver_scaled,
            optimize=True,
        )
    )
    boundary_physical = (
        boundary_scaled
        * np.tile(field_scales, BOUNDARY_SAMPLES)[None, :, None]
    ).reshape(times.size, BOUNDARY_SAMPLES, FIELDS, 2)
    trace_physical = np.transpose(boundary_physical[:, 0], (0, 2, 1))
    flux_physical = np.transpose(
        np.einsum(
            "ij,tjk->tik",
            driver["shared_face_flux_map"],
            driver_scaled,
            optimize=True,
        ),
        (0, 2, 1),
    )
    dynamic_scaled = scaled_history[:, FIELDS * DRIVER_CELLS :]
    dynamic_physical = np.transpose(
        dynamic_scaled
        * np.tile(field_scales, dynamic_nodes)[None, :, None],
        (0, 2, 1),
    ).reshape(times.size, 2, dynamic_nodes, FIELDS)
    full_pair = np.concatenate(
        (dynamic_physical, trace_physical[:, :, None, :]), axis=2
    )
    physical = _combine_basis(
        full_pair[:, 0], full_pair[:, 1], coefficients
    )

    common_log = np.log(np.asarray(driver["grid"].centers[:49]))
    interpolation = make_interp_spline(
        background.log_radii[:nodes],
        np.eye(nodes),
        k=5,
        axis=0,
    )(common_log)
    common_state = np.einsum(
        "mn,tbnf->tbmf", interpolation, physical, optimize=True
    )
    manufactured_vectors = np.asarray(
        (
            (0.7, -0.2, 0.3, 0.1, 0.4),
            (-0.1, 0.5, 0.2, -0.4, 0.3),
        ),
        dtype=float,
    ) * field_scales[None]
    coordinate = (
        (background.log_radii[:dynamic_nodes] - lower)
        / (interface - lower)
    )
    manufactured_envelope = np.sin(np.pi * coordinate) ** 4
    manufactured_state = np.asarray(
        [
            manufactured_envelope[:, None] * vector[None]
            for vector in manufactured_vectors
        ],
        dtype=float,
    )
    manufactured_scaled = np.column_stack(
        [
            (state / field_scales[None]).ravel()
            for state in manufactured_state
        ]
    )
    manufactured_rate = np.transpose(
        (inner @ manufactured_scaled)
        * np.tile(field_scales, dynamic_nodes)[:, None],
        (1, 0),
    ).reshape(2, dynamic_nodes, FIELDS)
    manufactured_action = np.einsum(
        "mn,bnf->bmf",
        interpolation[:, :dynamic_nodes],
        manufactured_rate,
        optimize=True,
    )

    weights = b6b._spline_integral_weights(
        background.log_radii[:nodes],
        float(np.log(edges[6])),
        interface,
    )
    total_density = 0.5 * np.einsum(
        "tbni,nij,tbnj->tbn",
        physical,
        metric[:nodes],
        physical,
        optimize=True,
    )
    total_energy = (
        np.einsum("tbn,n->tb", total_density, weights, optimize=True)
        / initial_energies[None]
    )
    target_energy = np.zeros_like(total_energy)
    for base_index, name in enumerate(BASES):
        for family in target_indices[name]:
            projected = np.einsum(
                "nij,tnj->tni",
                projectors[:nodes, family],
                physical[:, base_index],
                optimize=True,
            )
            density = 0.5 * np.einsum(
                "tni,nij,tnj->tn",
                projected,
                metric[:nodes],
                projected,
                optimize=True,
            )
            target_energy[:, base_index] += (
                np.einsum("tn,n->t", density, weights, optimize=True)
                / initial_energies[base_index]
            )

    # Exact blockwise energy-rate closure on a bounded set of samples.
    sample_indices = np.unique(
        np.linspace(0, times.size - 1, 17).round().astype(int)
    )
    maximum_ledger = 0.0
    dynamic_weights = weights[:dynamic_nodes]
    boundary_weight = float(weights[dynamic_nodes])
    trace_map = boundary_map[:FIELDS]
    for sample in sample_indices:
        driver_column = driver_scaled[sample]
        inner_column = dynamic_scaled[sample]
        driver_rate = driver["generator"] @ driver_column
        rate_by_block = [
            item["inner"] @ inner_column
            + item["drive"] @ driver_column
            for item in scaled_blocks.values()
        ]
        total_rate = sum(
            rate_by_block,
            start=np.zeros_like(inner_column),
        )
        direct_rate = (
            drive @ driver_column + inner @ inner_column
        )
        vector_defect = _relative_defect(total_rate, direct_rate)
        inner_physical = np.transpose(
            inner_column
            * np.tile(field_scales, dynamic_nodes)[:, None],
            (1, 0),
        ).reshape(2, dynamic_nodes, FIELDS)
        inner_rate_physical = np.transpose(
            direct_rate
            * np.tile(field_scales, dynamic_nodes)[:, None],
            (1, 0),
        ).reshape(2, dynamic_nodes, FIELDS)
        trace_state = trace_physical[sample]
        trace_rate = np.transpose(
            np.asarray(trace_map @ driver_rate)
            * field_scales[:, None],
            (1, 0),
        )
        full_state = np.concatenate(
            (inner_physical, trace_state[:, None, :]), axis=1
        )
        full_rate = np.concatenate(
            (inner_rate_physical, trace_rate[:, None, :]), axis=1
        )
        direct_power = np.einsum(
            "bni,nij,bnj,n->b",
            full_state,
            metric[:nodes],
            full_rate,
            weights,
            optimize=True,
        )
        block_power = np.zeros_like(direct_power)
        for block_rate in rate_by_block:
            physical_rate = np.transpose(
                block_rate
                * np.tile(field_scales, dynamic_nodes)[:, None],
                (1, 0),
            ).reshape(2, dynamic_nodes, FIELDS)
            block_power += np.einsum(
                "bni,nij,bnj,n->b",
                inner_physical,
                metric[:dynamic_nodes],
                physical_rate,
                dynamic_weights,
                optimize=True,
            )
        block_power += boundary_weight * np.einsum(
            "bi,ij,bj->b",
            trace_state,
            metric[dynamic_nodes],
            trace_rate,
            optimize=True,
        )
        power_scale = np.maximum(
            np.maximum(np.abs(direct_power), np.abs(block_power)),
            np.finfo(float).tiny,
        )
        power_defect = float(
            np.max(np.abs(direct_power - block_power) / power_scale)
        )
        maximum_ledger = max(
            maximum_ledger, vector_defect, power_defect
        )

    spatial = (
        background.physical_flux_jacobians
        - background.shear_principal_matrices
        - background.height_principal_matrices
    )
    speeds = np.asarray(
        [
            np.sort(
                np.real(
                    np.linalg.eigvals(
                        np.linalg.solve(
                            background.temporal_storage_matrices[index],
                            spatial[index],
                        )
                    )
                )
            )
            for index in range(extended_nodes)
        ],
        dtype=float,
    )
    interface_incoming = int(np.sum(speeds[nodes - 1] < 0.0))
    inner_incoming = int(np.sum(speeds[0] > 0.0))
    trace_pair = trace_physical
    projected_trace = np.einsum(
        "fij,tbj->tbfi",
        projectors[nodes - 1],
        trace_pair,
        optimize=True,
    ).sum(axis=2)
    boundary_closure = _relative_defect(projected_trace, trace_pair)
    replay = _relative_defect(
        boundary_weights[0],
        driver["right_reconstruction_weights"][INTERFACE_FACE],
    )
    report = {
        "nodes": nodes,
        "dynamic_nodes": dynamic_nodes,
        "extended_boundary_samples": BOUNDARY_SAMPLES,
        "restart_replay_defect": restart_defect,
        "maximum_energy_and_covariant_work_ledger_defect": (
            maximum_ledger
        ),
        "ratio_one_outer_trace_replay_defect": replay,
        "characteristic_boundary_closure_defect": boundary_closure,
        "incoming_interface_characteristic_count": interface_incoming,
        "incoming_inner_boundary_characteristic_count": inner_incoming,
        "minimum_characteristic_speed_over_c": float(np.min(speeds)),
        "maximum_characteristic_speed_over_c": float(np.max(speeds)),
        "energy_basis": energy_report,
    }
    arrays = {
        "times": times,
        "common_state": common_state,
        "manufactured_action": manufactured_action,
        "total_energy": total_energy,
        "target_energy": target_energy,
        "interface_trace_acoustic_shear": trace_pair,
        "interface_flux_acoustic_shear": flux_physical,
        "boundary_sample_weights": boundary_weights,
        "inner_log_radii": background.log_radii[:nodes],
        "energy_metric": metric[:nodes],
        "projectors": projectors[:nodes],
    }
    np.savez_compressed(path, **arrays)
    _write_json(
        path.with_suffix(".json"),
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "nodes": nodes,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "report": report,
        },
    )
    return {**arrays, "report": report}


def _action_difference(
    primary: dict,
    secondary: dict,
    field_scales: np.ndarray,
) -> dict:
    manufactured_primary = (
        primary["manufactured_action"] / field_scales
    )
    manufactured_secondary = (
        secondary["manufactured_action"] / field_scales
    )
    manufactured = _relative_defect(
        manufactured_primary, manufactured_secondary
    )
    scaled_primary = primary["common_state"] / field_scales
    scaled_secondary = secondary["common_state"] / field_scales
    state_history = _relative_defect(scaled_primary, scaled_secondary)
    total = _relative_defect(
        primary["total_energy"], secondary["total_energy"]
    )
    target = _relative_defect(
        primary["target_energy"], secondary["target_energy"]
    )
    return {
        "manufactured_complete_DAE_action": manufactured,
        "maximum": manufactured,
        "nonbinding_propagated_reference_uncertainty": {
            "common_state_history": state_history,
            "total_energy_history": total,
            "target_energy_history": target,
            "maximum": max(state_history, total, target),
        },
    }


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "config.json",
        PARENT_DIRECTORY / "embedded_manifest.json",
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        B6D_DIRECTORY / "summary.json",
        B6D_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "summary.json",
        C2A2_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path) for path in paths
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
        provenance_path = case / "provenance.json"
        if not case.is_dir() or not provenance_path.is_file():
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
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _write_report(summary: dict) -> None:
    action = summary["reference_comparison"]["action_difference"]
    lines = [
        "# Fixed-exterior inner-continuum reference "
        "WP10c9d6c7c2c2",
        "",
        "## Result",
        "",
        (
            "The fixed-N98-exterior driven inner-continuum preflight "
            + ("passes." if summary["passed"] else "fails.")
        ),
        "",
        "The unchanged complete N98 tangent drives independent N513 and "
        "N769 inner collocation systems through the actual reconstructed "
        "five-field outer trace. The actual shared-face flux is recorded "
        "as an independent replay diagnostic. No embedded state was run.",
        "",
        "## Decisive gates",
        "",
        "| Gate | Measured | Limit |",
        "|---|---:|---:|",
        f"| maximum N769/N513 action difference | "
        f"{action['maximum']:.6e} | {MAXIMUM_ACTION_DIFFERENCE:.1e} |",
        f"| trace replay | {summary['maximum_trace_replay_defect']:.6e} | "
        f"{MAXIMUM_TRACE_REPLAY_DEFECT:.1e} |",
        f"| characteristic boundary closure | "
        f"{summary['maximum_characteristic_boundary_closure_defect']:.6e} | "
        f"{MAXIMUM_BOUNDARY_CLOSURE_DEFECT:.1e} |",
        f"| energy/covariant-work ledger | "
        f"{summary['maximum_energy_ledger_defect']:.6e} | "
        f"{MAXIMUM_ENERGY_LEDGER_DEFECT:.1e} |",
        f"| restart replay | {summary['maximum_restart_replay_defect']:.6e} | "
        f"{RESTART_REPLAY_TOLERANCE:.1e} |",
        "",
        "All five interface characteristics enter the inner continuum and "
        "zero characteristics enter through the excision boundary.",
        "",
        "## Decision",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        f"Authorized next: `{summary['authorized_next']}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    parent_summary, _manifest, parent_arrays = _validate_parent()
    b6d_summary = _read_json(B6D_DIRECTORY / "summary.json")
    b6d_arrays = _load_npz(B6D_DIRECTORY / "decisive_arrays.npz")
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    coefficients = b6e._coefficients(b6d_summary)
    target_indices = {
        name: list(
            b6d_summary["profile_manifest"]["per_profile"][name][
                "target_family_indices"
            ]
        )
        for name in BASES
    }
    initial_energies = np.asarray(
        [b6d_arrays[f"initial_family_energy__{name}"] for name in BASES],
        dtype=float,
    ).reshape(len(BASES), -1).sum(axis=1)
    times = np.asarray(
        parent_arrays["primary_time_samples_seconds"], dtype=float
    )
    if times.size != TIME_SAMPLES:
        raise RuntimeError("frozen c2c1 time sampling changed")
    driver = _driver_data(parent_context, parent_base, field_scales)
    references = {
        nodes: _reference(
            nodes,
            driver,
            parent_context,
            parent_base,
            field_scales,
            coefficients,
            target_indices,
            initial_energies,
            times,
        )
        for nodes in INNER_NODES
    }
    comparison = _action_difference(
        references[769], references[513], field_scales
    )
    reports = [references[nodes]["report"] for nodes in INNER_NODES]
    maximum_trace = max(
        report["ratio_one_outer_trace_replay_defect"]
        for report in reports
    )
    maximum_boundary = max(
        report["characteristic_boundary_closure_defect"]
        for report in reports
    )
    maximum_ledger = max(
        report["maximum_energy_and_covariant_work_ledger_defect"]
        for report in reports
    )
    maximum_restart = max(
        report["restart_replay_defect"] for report in reports
    )
    counts_pass = all(
        report["incoming_interface_characteristic_count"]
        == EXPECTED_INTERFACE_INCOMING
        and report["incoming_inner_boundary_characteristic_count"]
        == EXPECTED_INNER_INCOMING
        for report in reports
    )
    passed = bool(
        comparison["maximum"] <= MAXIMUM_ACTION_DIFFERENCE
        and maximum_trace <= MAXIMUM_TRACE_REPLAY_DEFECT
        and maximum_boundary <= MAXIMUM_BOUNDARY_CLOSURE_DEFECT
        and maximum_ledger <= MAXIMUM_ENERGY_LEDGER_DEFECT
        and maximum_restart <= RESTART_REPLAY_TOLERANCE
        and counts_pass
    )
    classification = (
        "fixed_exterior_continuum_reference_certified_"
        "embedded_propagation_authorized"
        if passed
        else "fixed_exterior_continuum_reference_preflight_failed_"
        "embedded_propagation_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c2c3_direct_continuum_embedded_discrimination"
        if passed
        else "diagnose_fixed_exterior_continuum_reference_preflight"
    )

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "embedded_propagation_executed": False,
        "driver_cells": DRIVER_CELLS,
        "interface_parent_face": INTERFACE_FACE,
        "inner_reference_nodes": list(INNER_NODES),
        "boundary_samples": BOUNDARY_SAMPLES,
        "time_samples": TIME_SAMPLES,
        "gates": {
            "maximum_action_difference": MAXIMUM_ACTION_DIFFERENCE,
            "maximum_trace_replay_defect": MAXIMUM_TRACE_REPLAY_DEFECT,
            "maximum_boundary_closure_defect": (
                MAXIMUM_BOUNDARY_CLOSURE_DEFECT
            ),
            "maximum_energy_ledger_defect": (
                MAXIMUM_ENERGY_LEDGER_DEFECT
            ),
            "restart_replay_tolerance": RESTART_REPLAY_TOLERANCE,
        },
    }
    _write_json(CONFIG_PATH, config)
    decisive = {
        "times_seconds": times,
        "field_scales": field_scales,
        "initial_profile_energies": initial_energies,
        "interface_trace_acoustic_shear": references[769][
            "interface_trace_acoustic_shear"
        ],
        "interface_flux_acoustic_shear": references[769][
            "interface_flux_acoustic_shear"
        ],
        "N513_common_state_endpoint": references[513]["common_state"][-1],
        "N769_common_state_endpoint": references[769]["common_state"][-1],
        "N513_common_state_response_max_by_time_profile": np.max(
            np.abs(references[513]["common_state"] / field_scales),
            axis=(2, 3),
        ),
        "N769_common_state_response_max_by_time_profile": np.max(
            np.abs(references[769]["common_state"] / field_scales),
            axis=(2, 3),
        ),
        "N769_N513_common_state_difference_max_by_time_profile": np.max(
            np.abs(
                (
                    references[769]["common_state"]
                    - references[513]["common_state"]
                )
                / field_scales
            ),
            axis=(2, 3),
        ),
        "N513_manufactured_action": references[513][
            "manufactured_action"
        ],
        "N769_manufactured_action": references[769][
            "manufactured_action"
        ],
        "N513_total_energy": references[513]["total_energy"],
        "N769_total_energy": references[769]["total_energy"],
        "N513_target_energy": references[513]["target_energy"],
        "N769_target_energy": references[769]["target_energy"],
        "N513_boundary_sample_weights": references[513][
            "boundary_sample_weights"
        ],
        "N769_boundary_sample_weights": references[769][
            "boundary_sample_weights"
        ],
    }
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).is_file()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "embedded_propagation_executed": False,
        "matched_reference_propagation_executed": True,
        "historical_classifications_preserved": parent_summary[
            "historical_classifications_preserved"
        ],
        "reference_comparison": {
            "primary": "fixed_N98_exterior_driven_N769_inner_continuum",
            "secondary": "fixed_N98_exterior_driven_N513_inner_continuum",
            "action_difference": comparison,
        },
        "per_reference": {
            f"N{nodes}": references[nodes]["report"]
            for nodes in INNER_NODES
        },
        "maximum_trace_replay_defect": maximum_trace,
        "maximum_characteristic_boundary_closure_defect": (
            maximum_boundary
        ),
        "maximum_energy_ledger_defect": maximum_ledger,
        "maximum_restart_replay_defect": maximum_restart,
        "characteristic_counts_passed": counts_pass,
        "binding_decision": {
            "matched_fixed_exterior_reference_certified": passed,
            "embedded_propagation_authorized": passed,
            "numerical_redesign_authorized": False,
            "nonlinear_propagation_authorized": False,
            "fixed_Q_or_reduced_evolution_authorized": False,
        },
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": passed,
        "config_sha256": _sha256(CONFIG_PATH),
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
        "classification": classification,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "source_parent_tree": ANALYZED_BASE_TREE,
        "implementation_worktree_head": _git_value("rev-parse", "HEAD"),
        "implementation_source_hashes": source_hashes,
        "input_hashes": _input_hashes(),
        "command": (
            "PYTHONPATH=src python "
            "scripts/"
            "run_causal_inner_fixed_exterior_continuum_reference_"
            "wp10c9d6c7c2c2.py"
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
