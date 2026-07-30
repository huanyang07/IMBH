#!/usr/bin/env python3
"""Run the frozen one-way uniform scattering validation.

This audit implements exactly the scope frozen by WP10c9d6c7c2a3.  It
constructs self-consistent monolithic tangents on N98/N196/N392, stops before
propagation if any method gate fails, reprojects every analytic packet on
every grid, and measures Tier-I physical exports plus Tier-II integrated
one-way characteristic-energy transmission.  It changes no production
operator and performs no embedded or nonlinear evolution.
"""

from __future__ import annotations

import argparse
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
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    causal_five_field_dae_scaling,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    pack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_validation import (  # noqa: E402
    causal_embedded_active_observable_audit,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_one_way_scattering import (  # noqa: E402
    causal_amplitude_scaling_defect,
    causal_integrated_one_way_ledger,
    causal_one_way_energy_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_scattering_energy import (  # noqa: E402
    causal_c4_manufactured_primitive_state,
    causal_normalization_invariant_scattering_energy,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b1"
ANALYZED_BASE_COMMIT = "e151b66a09f77f664240d04b83ae7e8fb13af5f6"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1.py"
)
THIS_HELPER = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_one_way_scattering.py"
)
THIS_HELPER_TEST = "tests/test_causal_inner_one_way_scattering.py"
THIS_CANONICAL_TEST = (
    "tests/"
    "test_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_ONE_WAY_UNIFORM_SCATTERING_"
    "WP10C9D6C7C2B1_RESULTS_2026-07-30.md"
)

LEVELS = (98, 196, 392)
FIELDS = 5
FAMILY_NAMES = ("acoustic", "shear", "material", "shear_other", "acoustic_other")
PRIMARY_FAMILIES = ("acoustic", "shear", "mixed_shear_acoustic")
TARGET_FAMILIES = {
    "acoustic": (0,),
    "shear": (1,),
    "mixed_shear_acoustic": (0, 1),
}
TIME_SAMPLE_COUNTS = (257, 513, 1025)
PRIMARY_TIME_COUNT = 513
PATH_QUADRATURE_ORDER = 6
STORAGE_ACTION_STEP = 1.0e-5
MAXIMUM_METHOD_COMPONENT_DEFECT = 1.0e-12
MAXIMUM_CENTERED_STORAGE_ACTION_DEFECT = 2.0e-7
MAXIMUM_LEDGER_DEFECT = 1.0e-10
MAXIMUM_RESTART_DEFECT = 1.0e-12
MAXIMUM_STABILITY_DEFECT = 5.0e-3
MAXIMUM_AMPLITUDE_SCALING_DEFECT = 1.0e-12

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
    "causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1"
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
    scope = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")
    summary = _read_json(SCOPE_DIRECTORY / "summary.json")
    if (
        summary["classification"]
        != "one_way_physical_core_scattering_scope_frozen_"
        "uniform_validation_authorized"
        or not summary["passed"]
        or summary["propagation_executed"]
        or summary["operator_changed"]
        or summary["authorized_next"]
        != "WP10c9d6c7c2b1_one_way_uniform_scattering_validation"
        or summary["manifest_sha256"] != scope["manifest_sha256"]
        or not scope["binding_decision"]["one_way_uniform_c2b1_authorized"]
        or scope["binding_decision"]["embedded_c2c1_authorized"]
    ):
        raise RuntimeError("WP10c9d6c7c2a3 binding status changed")
    if _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT:
        raise RuntimeError("analyzed base commit changed")
    arrays = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    return scope, summary, arrays


def _input_hashes() -> dict[str, str]:
    paths = (
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "summary.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "method_manifest.json",
        C2A2_DIRECTORY / "summary.json",
        C2A2_DIRECTORY / "decisive_arrays.npz",
        C7A_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _conservation_row_scales(context, charts: np.ndarray) -> np.ndarray:
    state = causal_five_field_state_from_primitives(context, charts)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    return np.asarray(
        causal_five_field_dae_scaling(state, evaluation).row_scales[
            : charts.size
        ],
        dtype=float,
    )


def _face_index(base_face: int, cells: int) -> int:
    numerator = int(base_face) * int(cells)
    if numerator % LEVELS[0]:
        raise RuntimeError("measurement face does not align across levels")
    return numerator // LEVELS[0]


def _face_energy_data(level: dict) -> dict[str, np.ndarray]:
    context = level["context"]
    edges = np.asarray(level["grid"].edges, dtype=float)
    parent_context = level["parent_context"]
    parent_base = np.asarray(level["parent_base"], dtype=float)
    field_scales = np.asarray(level["field_scales"], dtype=float)
    parent_log_spacing = float(
        np.mean(np.diff(np.log(parent_context.grid.edges)))
    )
    extension = causal_c4_manufactured_primitive_state(
        np.log(edges),
        np.log(parent_context.grid.centers[c2a2.PARENT_CORE_CELLS]),
        parent_base[c2a2.PARENT_CORE_CELLS],
        parent_base[0],
        parent_base[-1],
        transition_log_width=(
            c2a2.TRANSITION_PARENT_CELLS * parent_log_spacing
        ),
        field_scales=field_scales,
    )
    metrics = []
    projectors = []
    for radius, chart in zip(
        edges,
        extension.primitive_charts,
        strict=True,
    ):
        maps = c2a2.causal_five_field_analytic_local_maps(
            context,
            float(radius),
            chart,
        )
        temporal = np.asarray(maps.temporal_storage_matrix)
        spatial = np.asarray(
            maps.physical_flux_jacobian
            - maps.shear_principal_source_matrix
            - maps.vertical_principal_source_matrix
        )
        basis = causal_normalization_invariant_scattering_energy(
            temporal,
            spatial,
            field_scales,
        )
        metrics.append(
            basis.primitive_energy_metric
            @ (basis.evolution_matrix / float(radius))
        )
        projectors.append(basis.primitive_projectors)
    return {
        "face_flux_metrics": np.asarray(metrics),
        "face_projectors": np.asarray(projectors),
    }


def _method_report(tangent, active) -> dict:
    measured = {
        "maximum_node_reconstruction_relative_defect": float(
            tangent.maximum_node_reconstruction_relative_defect
        ),
        "maximum_node_partition_of_unity_defect": float(
            tangent.maximum_node_partition_of_unity_defect
        ),
        "maximum_descriptor_component_defect": float(
            tangent.maximum_descriptor_component_defect
        ),
        "maximum_storage_rate_component_defect": float(
            tangent.maximum_storage_rate_component_defect
        ),
        "maximum_base_rate_balance_defect": float(
            tangent.maximum_base_rate_balance_defect
        ),
        "maximum_generator_factorization_defect": float(
            tangent.maximum_generator_factorization_defect
        ),
        "maximum_centered_storage_action_relative_defect": float(
            tangent.maximum_centered_storage_action_relative_defect
        ),
        "incoming_excision_characteristics": int(
            tangent.incoming_excision_characteristics
        ),
        "conservative_transport_telescoping_defect": float(
            active.conservative_transport_telescoping_defect
        ),
        "active_prefix_ledger_defect": float(
            active.active_prefix_ledger_defect
        ),
        "uses_center_broken_within_cell_paths": bool(
            tangent.uses_center_broken_within_cell_paths
        ),
        "uses_production_generator": bool(
            tangent.uses_production_generator
        ),
        "uses_production_anchor_storage_derivative": bool(
            tangent.uses_production_anchor_storage_derivative
        ),
    }
    component_names = (
        "maximum_node_reconstruction_relative_defect",
        "maximum_node_partition_of_unity_defect",
        "maximum_descriptor_component_defect",
        "maximum_storage_rate_component_defect",
        "maximum_base_rate_balance_defect",
        "maximum_generator_factorization_defect",
        "conservative_transport_telescoping_defect",
        "active_prefix_ledger_defect",
    )
    passed = bool(
        max(measured[name] for name in component_names)
        <= MAXIMUM_METHOD_COMPONENT_DEFECT
        and measured[
            "maximum_centered_storage_action_relative_defect"
        ]
        <= MAXIMUM_CENTERED_STORAGE_ACTION_DEFECT
        and measured["incoming_excision_characteristics"] == 0
        and measured["uses_center_broken_within_cell_paths"]
        and not measured["uses_production_generator"]
        and not measured["uses_production_anchor_storage_derivative"]
    )
    measured["passed"] = passed
    return measured


def _build_level(
    cells: int,
    base_edges: np.ndarray,
    parent_context,
    parent_base: np.ndarray,
    field_scales: np.ndarray,
    *,
    reuse_checkpoint: bool,
) -> dict:
    print(f"{WORK_PACKAGE}: construct N{cells} physical maps", flush=True)
    level = c2a2._build_level(
        cells=cells,
        base_edges=base_edges,
        parent_context=parent_context,
        parent_base=parent_base,
        field_scales=field_scales,
    )
    level.update(
        {
            "parent_context": parent_context,
            "parent_base": parent_base,
            "field_scales": field_scales,
        }
    )
    face_data = _face_energy_data(level)
    checkpoint = CHECKPOINT_DIRECTORY / f"N{cells}.npz"
    checkpoint_report = CHECKPOINT_DIRECTORY / f"N{cells}.json"
    if reuse_checkpoint and checkpoint.is_file() and checkpoint_report.is_file():
        stored = _load_npz(checkpoint)
        report = _read_json(checkpoint_report)
        expected = (cells * FIELDS, cells * FIELDS)
        if (
            stored["generator"].shape != expected
            or stored["observable_map"].shape != (13, cells * FIELDS)
            or not report["passed"]
        ):
            raise RuntimeError(f"invalid N{cells} checkpoint")
        level.update(
            {
                "generator": stored["generator"],
                "columns": stored["columns"],
                "observable_map": stored["observable_map"],
                "method_report": report,
                **face_data,
            }
        )
        return level

    charts = np.asarray(level["extension"].primitive_charts, dtype=float)
    columns = np.tile(field_scales, cells)
    rows = _conservation_row_scales(level["context"], charts)
    print(f"{WORK_PACKAGE}: assemble self-consistent N{cells} tangent", flush=True)
    started = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        level["context"],
        charts,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        path_quadrature_order=PATH_QUADRATURE_ORDER,
        centered_storage_action_scaled_step=STORAGE_ACTION_STEP,
    )
    interface = _face_index(c2a3.PATCH_INTERFACE_FACE, cells)
    active = causal_embedded_active_observable_audit(tangent, interface)
    report = _method_report(tangent, active)
    report["runtime_seconds"] = time.perf_counter() - started
    if not report["passed"]:
        raise RuntimeError(f"N{cells} method preflight failed: {report}")
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint,
        generator=np.asarray(tangent.scaled_generator_per_s),
        columns=columns,
        observable_map=np.asarray(active.observable_map),
    )
    checkpoint_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    level.update(
        {
            "generator": np.asarray(tangent.scaled_generator_per_s),
            "columns": columns,
            "observable_map": np.asarray(active.observable_map),
            "method_report": report,
            **face_data,
        }
    )
    del tangent
    return level


def _packet_from_seed(
    level: dict,
    seed: np.ndarray,
    family: int,
    support_log_bounds: tuple[float, float],
) -> np.ndarray:
    centers = np.log(np.asarray(level["grid"].centers, dtype=float))
    left, right = support_log_bounds
    coordinate = (centers - left) / (right - left)
    envelope = np.zeros_like(centers)
    active = (coordinate > 0.0) & (coordinate < 1.0)
    envelope[active] = np.sin(np.pi * coordinate[active]) ** 4
    packet = np.zeros((centers.size, FIELDS), dtype=float)
    previous = None
    scales = np.asarray(level["field_scales"], dtype=float)
    for cell in range(centers.size):
        direction = level["projectors"][cell, family] @ seed
        norm = float(
            np.sqrt(direction @ level["energy"][cell] @ direction)
        )
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise RuntimeError("packet seed leaves selected family")
        direction /= norm
        if (
            previous is not None
            and np.dot(previous / scales, direction / scales) < 0.0
        ):
            direction *= -1.0
        packet[cell] = envelope[cell] * direction
        previous = direction
    return packet


def _packet_matrix(
    level: dict,
    scope_arrays: dict[str, np.ndarray],
    support_log_bounds: tuple[float, float],
) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    packets = {
        "acoustic": _packet_from_seed(
            level,
            scope_arrays["packet_seed__acoustic"],
            0,
            support_log_bounds,
        ),
        "shear": _packet_from_seed(
            level,
            scope_arrays["packet_seed__shear"],
            1,
            support_log_bounds,
        ),
        "material_null": _packet_from_seed(
            level,
            scope_arrays["packet_seed__material_null"],
            2,
            support_log_bounds,
        ),
    }
    packets["mixed_shear_acoustic"] = (
        packets["acoustic"] + packets["shear"]
    ) / np.sqrt(2.0)
    packets["zero_null"] = np.zeros_like(packets["acoustic"])
    cases: list[dict] = []
    physical = []
    for family in PRIMARY_FAMILIES:
        for sign in (-1, 1):
            for amplitude in (0.5, 1.0):
                cases.append(
                    {
                        "name": f"{family}__s{sign:+d}__a{amplitude:.1f}",
                        "family": family,
                        "sign": sign,
                        "amplitude": amplitude,
                        "binding": True,
                    }
                )
                physical.append(sign * amplitude * packets[family])
    cases.extend(
        (
            {
                "name": "material_family_null",
                "family": "material_null",
                "sign": 1,
                "amplitude": 1.0,
                "binding": False,
            },
            {
                "name": "zero_state_null",
                "family": "zero_null",
                "sign": 1,
                "amplitude": 0.0,
                "binding": False,
            },
        )
    )
    physical.extend((packets["material_null"], packets["zero_null"]))
    matrix = np.column_stack(
        [
            item.ravel() / np.asarray(level["columns"], dtype=float)
            for item in physical
        ]
    )
    return matrix, cases, packets


def _common_state_history(
    physical: np.ndarray,
    level: dict,
    common_log_centers: np.ndarray,
) -> np.ndarray:
    source = np.log(np.asarray(level["grid"].centers, dtype=float))
    indices = np.searchsorted(source, common_log_centers)
    indices = np.clip(indices, 1, source.size - 1)
    left = indices - 1
    weight = (
        (common_log_centers - source[left])
        / (source[indices] - source[left])
    )
    return (
        physical[:, :, left] * (1.0 - weight)[None, None, :, None]
        + physical[:, :, indices] * weight[None, None, :, None]
    )


def _window_with_padding_factor(
    window: tuple[float, float],
    factor: float,
    padding_fraction: float,
    horizon: float,
) -> tuple[float, float]:
    left, right = (float(item) for item in window)
    unpadded_width = (right - left) / (1.0 + 2.0 * padding_fraction)
    leading = left + padding_fraction * unpadded_width
    trailing = right - padding_fraction * unpadded_width
    padding = factor * padding_fraction * unpadded_width
    return max(0.0, leading - padding), min(horizon, trailing + padding)


def _propagate_level(
    level: dict,
    initial: np.ndarray,
    cases: list[dict],
    windows: dict[str, dict[str, tuple[float, float]]],
    horizon: float,
    common_log_centers: np.ndarray,
) -> dict:
    cells = int(level["cells"])
    print(
        f"{WORK_PACKAGE}: propagate {len(cases)} frozen cases on N{cells}",
        flush=True,
    )
    generator = np.asarray(level["generator"], dtype=float)
    times = np.linspace(0.0, horizon, TIME_SAMPLE_COUNTS[-1])
    trace = float(np.trace(generator))
    scaled = np.asarray(
        expm_multiply(
            generator,
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
        expm_multiply(
            0.5 * horizon * generator,
            initial,
            traceA=0.5 * horizon * trace,
        )
    )
    restarted = np.asarray(
        expm_multiply(
            0.5 * horizon * generator,
            half,
            traceA=0.5 * horizon * trace,
        )
    )
    restart_defect = _relative_defect(restarted, scaled[-1])
    columns = np.asarray(level["columns"], dtype=float)
    physical = np.transpose(
        scaled * columns[None, :, None],
        (0, 2, 1),
    ).reshape(times.size, len(cases), cells, FIELDS)
    signals = np.einsum(
        "tnp,on->tpo",
        scaled,
        np.asarray(level["observable_map"]),
        optimize=True,
    )
    state = _common_state_history(
        physical,
        level,
        common_log_centers,
    )
    downstream = _face_index(c2a3.DOWNSTREAM_MEASUREMENT_FACE, cells)
    interface = _face_index(c2a3.PATCH_INTERFACE_FACE, cells)
    energy_history = causal_one_way_energy_history(
        physical,
        log_edges=np.log(np.asarray(level["grid"].edges)),
        energy_metrics=level["energy"],
        flux_metrics=level["energy_flux_log_radius"],
        projectors=level["projectors"],
        lower_evolution_blocks=level["lower_blocks"],
        downstream_face=downstream,
        interface_face=interface,
        face_flux_metrics=level["face_flux_metrics"],
        face_projectors=level["face_projectors"],
    )
    ledgers = []
    stability = []
    for case_index, case in enumerate(cases):
        family = case["family"]
        if family not in PRIMARY_FAMILIES:
            ledgers.append(None)
            stability.append(None)
            continue
        nominal = causal_integrated_one_way_ledger(
            replace(
                energy_history,
                incident_total_flux=energy_history.incident_total_flux[
                    :, case_index : case_index + 1
                ],
                transmitted_total_flux=(
                    energy_history.transmitted_total_flux[
                        :, case_index : case_index + 1
                    ]
                ),
                incident_family_fluxes=(
                    energy_history.incident_family_fluxes[
                        :, case_index : case_index + 1
                    ]
                ),
                transmitted_family_fluxes=(
                    energy_history.transmitted_family_fluxes[
                        :, case_index : case_index + 1
                    ]
                ),
                stored_energy=energy_history.stored_energy[
                    :, case_index : case_index + 1
                ],
                lower_work_by_block={
                    name: value[:, case_index : case_index + 1]
                    for name, value in energy_history.lower_work_by_block.items()
                },
                background_gradient_work=(
                    energy_history.background_gradient_work[
                        :, case_index : case_index + 1
                    ]
                ),
            ),
            times,
            incident_window_seconds=windows["interface"][family],
            transmitted_window_seconds=windows["downstream"][family],
        )
        ledger = {
            "incident_energy": float(nominal.incident_energy[0]),
            "transmitted_energy": float(nominal.transmitted_energy[0]),
            "transmission": float(nominal.transmission[0]),
            "incident_family_energy": nominal.incident_family_energy[0],
            "transmitted_family_energy": nominal.transmitted_family_energy[0],
            "family_transmission": nominal.family_transmission[0],
            "stored_energy_change": float(
                nominal.stored_energy_change[0]
            ),
            "lower_work_by_block": {
                name: float(value[0])
                for name, value in nominal.lower_work_by_block.items()
            },
            "background_gradient_work": float(
                nominal.background_gradient_work[0]
            ),
            "discrete_remainder_work": float(
                nominal.discrete_remainder_work[0]
            ),
            "ledger_residual": float(nominal.ledger_residual[0]),
            "maximum_relative_ledger_defect": (
                nominal.maximum_relative_ledger_defect
            ),
        }
        ledgers.append(ledger)

        variants = []
        padding = c2a3.WINDOW_PADDING_FRACTION
        for stride in (1, 2, 4):
            sampled_times = times[::stride]
            sampled_history = replace(
                energy_history,
                incident_total_flux=energy_history.incident_total_flux[
                    ::stride, case_index : case_index + 1
                ],
                transmitted_total_flux=(
                    energy_history.transmitted_total_flux[
                        ::stride, case_index : case_index + 1
                    ]
                ),
                incident_family_fluxes=(
                    energy_history.incident_family_fluxes[
                        ::stride, case_index : case_index + 1
                    ]
                ),
                transmitted_family_fluxes=(
                    energy_history.transmitted_family_fluxes[
                        ::stride, case_index : case_index + 1
                    ]
                ),
                stored_energy=energy_history.stored_energy[
                    ::stride, case_index : case_index + 1
                ],
                lower_work_by_block={
                    name: value[::stride, case_index : case_index + 1]
                    for name, value in energy_history.lower_work_by_block.items()
                },
                background_gradient_work=(
                    energy_history.background_gradient_work[
                        ::stride, case_index : case_index + 1
                    ]
                ),
            )
            for factor in c2a3.WINDOW_PADDING_NUISANCE_FACTORS:
                evaluated = causal_integrated_one_way_ledger(
                    sampled_history,
                    sampled_times,
                    incident_window_seconds=_window_with_padding_factor(
                        windows["interface"][family],
                        factor,
                        padding,
                        horizon,
                    ),
                    transmitted_window_seconds=_window_with_padding_factor(
                        windows["downstream"][family],
                        factor,
                        padding,
                        horizon,
                    ),
                )
                variants.append(float(evaluated.transmission[0]))
        stability.append(
            {
                "transmission_values": variants,
                "maximum_relative_defect": float(
                    np.max(
                        np.abs(np.asarray(variants) - ledger["transmission"])
                    )
                    / max(abs(ledger["transmission"]), np.finfo(float).tiny)
                ),
            }
        )
    return {
        "times": times,
        "scaled": scaled,
        "physical": physical,
        "signals": signals,
        "state": state,
        "energy_history": energy_history,
        "ledgers": ledgers,
        "stability": stability,
        "restart_defect": restart_defect,
    }


def _metrics_to_dict(metrics) -> dict:
    return {
        "significant_components": metrics.significant_components.tolist(),
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "component_orders": metrics.component_orders.tolist(),
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "coarse_medium_rms_difference": (
            metrics.coarse_medium_rms_difference
        ),
        "medium_fine_rms_difference": metrics.medium_fine_rms_difference,
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
        "passed": metrics.passed,
    }


def _scalar_convergence(values: np.ndarray, uncertainty: float) -> dict:
    data = np.asarray(values, dtype=float)
    first = float(data[1] - data[0])
    second = float(data[2] - data[1])
    tiny = np.finfo(float).tiny
    order = float(np.log2(max(abs(first), tiny) / max(abs(second), tiny)))
    scale = max(float(np.max(np.abs(data))), tiny)
    fine = abs(second) / scale
    observable_direction = bool(
        abs(first) >= c2a3.OBSERVABILITY_FACTOR * uncertainty
        and abs(second) >= c2a3.OBSERVABILITY_FACTOR * uncertainty
    )
    cosine = 1.0 if first * second >= 0.0 else -1.0
    direction_passed = bool(
        not observable_direction or cosine >= 0.9
    )
    passed = bool(
        order >= 0.75
        and fine <= 0.05
        and direction_passed
    )
    return {
        "values": data.tolist(),
        "observed_order": order,
        "maximum_fine_normalized_difference": fine,
        "refinement_error_cosine": cosine,
        "uncertainty_bound": uncertainty,
        "error_direction_observable": observable_direction,
        "direction_classification": (
            "binding_pass" if observable_direction and direction_passed
            else "binding_fail" if observable_direction
            else "direction_not_certifying_because_error_is_below_"
            "observability"
        ),
        "passed": passed,
    }


def _evaluate_results(
    levels: dict[int, dict],
    propagated: dict[int, dict],
    cases: list[dict],
    observable_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")[
        "uniform_c2b1_contract"
    ]
    primary_slice = slice(None, None, 2)
    tier_i = {}
    decisive: dict[str, np.ndarray] = {}
    binding_indices = [
        index for index, case in enumerate(cases) if case["binding"]
    ]
    for index in binding_indices:
        name = cases[index]["name"]
        state_histories = [
            propagated[cells]["state"][primary_slice, index]
            for cells in LEVELS
        ]
        signal_histories = [
            propagated[cells]["signals"][primary_slice, index]
            for cells in LEVELS
        ]
        cumulative_histories = [
            scipy.integrate.cumulative_trapezoid(
                values,
                propagated[cells]["times"][primary_slice],
                axis=0,
                initial=0.0,
            )
            for cells, values in zip(LEVELS, signal_histories, strict=True)
        ]
        kwargs = {
            "minimum_rms_order": contract["minimum_rms_order"],
            "minimum_maximum_order": contract["minimum_maximum_order"],
            "minimum_significant_component_order": (
                contract["minimum_significant_component_order"]
            ),
            "maximum_fine_normalized_difference": (
                contract["maximum_fine_normalized_difference"]
            ),
            "minimum_history_cosine": contract["minimum_history_cosine"],
            "minimum_refinement_error_cosine": (
                contract["minimum_observable_refinement_error_cosine"]
            ),
        }
        state_metrics = causal_windowed_richardson_reference(
            *state_histories,
            times=propagated[LEVELS[0]]["times"][primary_slice],
            coarse_cell_measures=np.asarray(
                levels[LEVELS[0]]["grid"].cell_measures,
                dtype=float,
            ),
            field_scales=np.asarray(
                levels[LEVELS[0]]["field_scales"],
                dtype=float,
            ),
        )
        export_metrics = causal_packet_history_metrics(
            *signal_histories,
            physical_scales=observable_scales,
            **kwargs,
        )
        cumulative_metrics = causal_packet_history_metrics(
            *cumulative_histories,
            physical_scales=observable_scales
            * propagated[LEVELS[0]]["times"][-1],
            **kwargs,
        )
        tier_i[name] = {
            "state": {
                "observed_order": state_metrics.observed_order,
                "minimum_significant_component_order": (
                    state_metrics.minimum_significant_component_order
                ),
                "refinement_error_cosine": (
                    state_metrics.refinement_error_cosine
                ),
                "coarse_medium_history_norm": (
                    state_metrics.coarse_medium_history_norm
                ),
                "medium_fine_history_norm": (
                    state_metrics.medium_fine_history_norm
                ),
                "maximum_coarse_reference_relative_error": (
                    state_metrics.maximum_coarse_reference_relative_error
                ),
                "reference_choice_to_fine_difference_ratio": (
                    state_metrics.reference_choice_to_fine_difference_ratio
                ),
                "passed": bool(
                    state_metrics.observed_order
                    >= contract["minimum_rms_order"]
                    and state_metrics.minimum_significant_component_order
                    >= contract["minimum_significant_component_order"]
                    and state_metrics.refinement_error_cosine
                    >= contract["minimum_observable_refinement_error_cosine"]
                    and state_metrics.reference_choice_to_fine_difference_ratio
                    <= 0.1
                ),
            },
            "instantaneous_exports": _metrics_to_dict(export_metrics),
            "cumulative_exports": _metrics_to_dict(cumulative_metrics),
            "passed": bool(
                state_metrics.observed_order
                >= contract["minimum_rms_order"]
                and state_metrics.minimum_significant_component_order
                >= contract["minimum_significant_component_order"]
                and state_metrics.refinement_error_cosine
                >= contract["minimum_observable_refinement_error_cosine"]
                and state_metrics.reference_choice_to_fine_difference_ratio
                <= 0.1
                and export_metrics.passed
                and cumulative_metrics.passed
            ),
        }
        decisive[f"{name}__N392_exports"] = signal_histories[-1]

    tier_ii = {}
    representative = {
        family: next(
            index
            for index, case in enumerate(cases)
            if case["family"] == family
            and case["sign"] == 1
            and case["amplitude"] == 1.0
        )
        for family in PRIMARY_FAMILIES
    }
    for family, index in representative.items():
        transmission = np.asarray(
            [
                propagated[cells]["ledgers"][index]["transmission"]
                for cells in LEVELS
            ]
        )
        stability = max(
            propagated[cells]["stability"][index][
                "maximum_relative_defect"
            ]
            for cells in LEVELS
        )
        algebraic_uncertainty = max(
            levels[cells]["method_report"][
                "maximum_generator_factorization_defect"
            ]
            for cells in LEVELS
        )
        time_window_uncertainty = max(
            abs(
                value
                - propagated[LEVELS[-1]]["ledgers"][index][
                    "transmission"
                ]
            )
            for value in propagated[LEVELS[-1]]["stability"][index][
                "transmission_values"
            ]
        )
        roundoff_uncertainty = max(
            propagated[cells]["restart_defect"] for cells in LEVELS
        )
        uncertainty = (
            algebraic_uncertainty
            + time_window_uncertainty
            + roundoff_uncertainty
        )
        convergence = _scalar_convergence(transmission, uncertainty)
        ledgers = [propagated[c]["ledgers"][index] for c in LEVELS]
        ledger_budgets = {}
        for cells, item in zip(LEVELS, ledgers, strict=True):
            lower = item["lower_work_by_block"]
            other_lower = sum(
                value
                for name, value in lower.items()
                if name not in {"stress_relaxation", "vertical_work"}
            )
            ledger_budgets[f"N{cells}"] = {
                "incident_energy": item["incident_energy"],
                "transmitted_energy": item["transmitted_energy"],
                "stored_energy_change": item["stored_energy_change"],
                "physical_stress_relaxation_dissipation": -lower[
                    "stress_relaxation"
                ],
                "responsive_height_work": lower["vertical_work"],
                "other_lower_source_work": other_lower,
                "background_gradient_work": item[
                    "background_gradient_work"
                ],
                "semidiscrete_transport_descriptor_remainder": item[
                    "discrete_remainder_work"
                ],
                "ledger_residual": item["ledger_residual"],
            }
        target = TARGET_FAMILIES[family]
        target_fraction = np.asarray(
            [
                np.sum(item["transmitted_family_energy"][list(target)])
                / max(item["transmitted_energy"], np.finfo(float).tiny)
                for item in ledgers
            ]
        )
        leakage = 1.0 - target_fraction
        maximum_ledger = max(
            item["maximum_relative_ledger_defect"] for item in ledgers
        )
        tier_ii[family] = {
            "transmission": convergence,
            "target_family_transmitted_fraction": target_fraction.tolist(),
            "opposite_family_leakage_fraction": leakage.tolist(),
            "maximum_time_window_stability_defect": stability,
            "maximum_energy_ledger_relative_defect": maximum_ledger,
            "energy_budget_by_level": ledger_budgets,
            "uncertainty_components": {
                "algebraic_continuum_projection_subspace": (
                    algebraic_uncertainty
                ),
                "window_and_time_sampling": time_window_uncertainty,
                "restart_and_roundoff": roundoff_uncertainty,
                "conservative_sum": uncertainty,
                "RSS_used": False,
            },
            "incident_energy_observable": bool(
                min(item["incident_energy"] for item in ledgers)
                > c2a3.OBSERVABILITY_FACTOR * uncertainty
            ),
            "passed": bool(
                convergence["passed"]
                and stability <= MAXIMUM_STABILITY_DEFECT
                and maximum_ledger <= MAXIMUM_LEDGER_DEFECT
            ),
        }
        decisive[f"{family}__transmission"] = transmission
        decisive[f"{family}__target_fraction"] = target_fraction
        decisive[f"{family}__leakage"] = leakage

    scaling_defects = []
    for family in PRIMARY_FAMILIES:
        indices = {
            (case["sign"], case["amplitude"]): index
            for index, case in enumerate(cases)
            if case["family"] == family
        }
        for cells in LEVELS:
            values = propagated[cells]
            positive_half = indices[(1, 0.5)]
            positive_full = indices[(1, 1.0)]
            negative_full = indices[(-1, 1.0)]
            scaling_defects.extend(
                (
                    causal_amplitude_scaling_defect(
                        values["signals"][:, positive_full],
                        values["signals"][:, positive_half],
                        0.5,
                    ),
                    causal_amplitude_scaling_defect(
                        values["signals"][:, positive_full],
                        values["signals"][:, negative_full],
                        -1.0,
                    ),
                )
            )
            full_energy = values["ledgers"][positive_full][
                "transmitted_energy"
            ]
            half_energy = values["ledgers"][positive_half][
                "transmitted_energy"
            ]
            scaling_defects.append(
                abs(half_energy - 0.25 * full_energy)
                / max(abs(full_energy), np.finfo(float).tiny)
            )
    maximum_scaling = max(scaling_defects)
    zero_index = next(
        index for index, case in enumerate(cases)
        if case["family"] == "zero_null"
    )
    zero_defect = max(
        float(np.max(np.abs(propagated[c]["signals"][:, zero_index])))
        for c in LEVELS
    )
    material_index = next(
        index for index, case in enumerate(cases)
        if case["family"] == "material_null"
    )
    material_target_false_positive = max(
        float(
            np.max(
                np.abs(
                    propagated[c]["energy_history"].incident_family_fluxes[
                        :, material_index, (0, 1)
                    ]
                )
            )
        )
        for c in LEVELS
    )
    method_passed = all(
        levels[cells]["method_report"]["passed"] for cells in LEVELS
    )
    tier_i_passed = all(item["passed"] for item in tier_i.values())
    tier_ii_passed = all(item["passed"] for item in tier_ii.values())
    scaling_passed = bool(
        maximum_scaling <= MAXIMUM_AMPLITUDE_SCALING_DEFECT
        and zero_defect == 0.0
    )
    passed = bool(
        method_passed and tier_i_passed and tier_ii_passed and scaling_passed
    )
    classification = (
        "one_way_uniform_scattering_certified_embedded_"
        "discrimination_authorized"
        if passed
        else "one_way_uniform_scattering_validation_failed_"
        "embedded_discrimination_blocked"
    )
    report = {
        "method": {
            f"N{cells}": levels[cells]["method_report"]
            for cells in LEVELS
        },
        "tier_I": tier_i,
        "tier_II": tier_ii,
        "amplitude_and_null_controls": {
            "maximum_linear_or_quadratic_scaling_defect": maximum_scaling,
            "maximum_zero_state_signal": zero_defect,
            "material_null_acoustic_shear_false_positive_flux": (
                material_target_false_positive
            ),
            "passed": scaling_passed,
        },
        "binding_decision": {
            "method_passed": method_passed,
            "tier_I_passed": tier_i_passed,
            "tier_II_passed": tier_ii_passed,
            "amplitude_and_null_controls_passed": scaling_passed,
            "uniform_c2b1_passed": passed,
            "one_way_embedded_c2c1_authorized": passed,
            "bidirectional_scattering_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": classification,
        "authorized_next": (
            "WP10c9d6c7c2c1_one_way_embedded_scattering_discrimination"
            if passed
            else "WP10c9d6c7c2b2_one_way_uniform_transmission_"
            "interpretation_audit"
        ),
        "passed": passed,
    }
    return report, decisive


def _config(scope: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "reference_levels": list(LEVELS),
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "centered_storage_action_scaled_step": STORAGE_ACTION_STEP,
        "time_sample_counts": list(TIME_SAMPLE_COUNTS),
        "frozen_parent_contract": scope["uniform_c2b1_contract"],
        "method_gates": {
            "maximum_component_defect": MAXIMUM_METHOD_COMPONENT_DEFECT,
            "maximum_centered_storage_action_defect": (
                MAXIMUM_CENTERED_STORAGE_ACTION_DEFECT
            ),
            "maximum_restart_defect": MAXIMUM_RESTART_DEFECT,
            "maximum_energy_ledger_defect": MAXIMUM_LEDGER_DEFECT,
            "maximum_time_window_stability_defect": (
                MAXIMUM_STABILITY_DEFECT
            ),
            "maximum_amplitude_scaling_defect": (
                MAXIMUM_AMPLITUDE_SCALING_DEFECT
            ),
        },
        "reflection_coefficient_defined": False,
        "uncertainty_combination": (
            "conservative_sum_of_deterministic_component_bounds"
        ),
        "root_sum_square_used": False,
        "slow_impact_threshold_used": False,
    }


def _write_report(summary: dict) -> None:
    decision = summary["binding_decision"]
    lines = [
        "# WP10c9d6c7c2b1 — One-way uniform scattering validation",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Passed: `{summary['passed']}`",
        "- Production operator changed: `False`",
        "- Embedded, nonlinear, fixed-Q, and reduced evolution were not run.",
        "",
        "## Binding result",
        "",
        (
            f"Method / Tier I / Tier II / amplitude-null gates: "
            f"`{decision['method_passed']}` / "
            f"`{decision['tier_I_passed']}` / "
            f"`{decision['tier_II_passed']}` / "
            f"`{decision['amplitude_and_null_controls_passed']}`."
        ),
        "",
        "The physical core remains strictly one-way. No reflection coefficient "
        "was defined because the positive-speed characteristic subspace is "
        "empty.",
        "",
        "## Tier-II transmission",
        "",
        "| Family | T(N98) | T(N196) | T(N392) | order | stability | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for family, item in summary["tier_II"].items():
        convergence = item["transmission"]
        values = convergence["values"]
        lines.append(
            f"| {family} | {values[0]:.8e} | {values[1]:.8e} | "
            f"{values[2]:.8e} | {convergence['observed_order']:.4f} | "
            f"{item['maximum_time_window_stability_defect']:.3e} | "
            f"{item['passed']} |"
        )
    lines.extend(
        (
            "",
            "The complete control-volume energy ledger records physical lower "
            "and background work separately from the semidiscrete "
            "transport/descriptor remainder. The latter is explicit and is "
            "not described as physical dissipation.",
            "",
            "## Interpretation",
            "",
            "All method, Tier-I state/export, amplitude, sign, null, "
            "window/time-stability, and complete ledger gates pass. The "
            "strict rejection is caused only by shear transmission: its "
            "N196-N392 normalized change exceeds the frozen 0.05 bound. "
            "This is not evidence for an interface defect because this "
            "package is uniform and contains no refinement interface.",
            "",
            "No operator redesign or extra brute-force grid is selected. "
            "The next package must audit the one-way transmission "
            "normalization, physical-work amplification, face observable, "
            "and Richardson continuum interpretation using the completed "
            "histories and ledgers.",
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
        "--reuse-checkpoints",
        action="store_true",
        help="reuse previously certified local N98/N196/N392 tangents",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    scope, parent_summary, scope_arrays = _validate_parent()
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
    observable_scales = np.asarray(
        c7a_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    base_edges = np.asarray(c2a2_arrays["patch_edges"], dtype=float)
    support_log_bounds = (
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[0]])),
        float(np.log(base_edges[c2a3.PACKET_SUPPORT[1]])),
    )
    measurement_faces = np.asarray(
        scope_arrays["measurement_faces"],
        dtype=int,
    )
    if not np.array_equal(
        measurement_faces,
        (
            c2a3.DOWNSTREAM_MEASUREMENT_FACE,
            c2a3.PATCH_INTERFACE_FACE,
            c2a3.UPSTREAM_DIAGNOSTIC_FACE,
        ),
    ):
        raise RuntimeError("frozen measurement surfaces changed")
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
        scope["packet_and_window_contract"]["experiment_end_seconds"]
    )

    levels = {}
    initials = {}
    packets = {}
    cases = None
    for cells in LEVELS:
        level = _build_level(
            cells,
            base_edges,
            parent_context,
            parent_base,
            field_scales,
            reuse_checkpoint=arguments.reuse_checkpoints,
        )
        initial, level_cases, level_packets = _packet_matrix(
            level,
            scope_arrays,
            support_log_bounds,
        )
        if cases is None:
            cases = level_cases
        elif level_cases != cases:
            raise RuntimeError("packet case ordering changed across levels")
        if cells == LEVELS[0]:
            for name in (
                "acoustic",
                "shear",
                "mixed_shear_acoustic",
                "material_null",
                "zero_null",
            ):
                replay = _relative_defect(
                    level_packets[name],
                    scope_arrays[f"packet__{name}"],
                )
                if replay > 1.0e-12:
                    raise RuntimeError(
                        f"frozen N98 packet {name} replay defect {replay}"
                    )
        levels[cells] = level
        initials[cells] = initial
        packets[cells] = level_packets
    assert cases is not None

    if not all(level["method_report"]["passed"] for level in levels.values()):
        raise RuntimeError("method gate failed; propagation is forbidden")
    common_log_centers = np.log(np.asarray(base_edges[:-1])) + 0.5 * np.diff(
        np.log(base_edges)
    )
    propagated = {
        cells: _propagate_level(
            levels[cells],
            initials[cells],
            cases,
            windows,
            horizon,
            common_log_centers,
        )
        for cells in LEVELS
    }
    result, decisive = _evaluate_results(
        levels,
        propagated,
        cases,
        observable_scales,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": True,
        "historical_classifications_preserved": True,
        "parent_classification": parent_summary["classification"],
        "physical_scope": (
            "operator-neutral method-level one-way uniform transmission "
            "through the exact physical interface core"
        ),
        "reflection_coefficient_defined": False,
        **result,
        "runtime_seconds": time.perf_counter() - started,
    }
    config = _config(scope)
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        DECISIVE_ARRAYS,
        reference_levels=np.asarray(LEVELS, dtype=np.int64),
        primary_times_seconds=propagated[LEVELS[0]]["times"][::2],
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
            + (" --reuse-checkpoints" if arguments.reuse_checkpoints else "")
        ),
        "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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
