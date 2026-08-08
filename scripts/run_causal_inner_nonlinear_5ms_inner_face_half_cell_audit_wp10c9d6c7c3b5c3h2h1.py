#!/usr/bin/env python3
"""Execute the operator-neutral 5 ms inner-face/half-cell audit."""

from __future__ import annotations

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
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_inner_face_half_cell_manifest_wp10c9d6c7c3b5c3h2h as h2h  # noqa: E402
import run_causal_inner_nonlinear_5ms_spatial_certificate_wp10c9d6c7c3b5c3h2f as h2f  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as h2e1  # noqa: E402
import run_causal_inner_nonlinear_middle_5ms_completion_wp10c9d6c7c3b5c3h2d1 as h2d1  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as b2b  # noqa: E402

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    causal_five_field_reconstruct_face_charts,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_discrete_tangent import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistoryDirection,
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_discrete_step_matrix,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_linear_tangent import (  # noqa: E402
    causal_five_field_analytic_local_maps,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2h1"
ANALYZED_BASE_COMMIT = "78cbe736c79aa44950c6c7266409bea2957a2aaa"
ANALYZED_BASE_PARENT = "f5d743acf516a9f491c978efd5599b8f763d6500"
ANALYZED_BASE_TREE = "b0cffccec3452a72deb12b0da2caa8868801b986"

LAYOUTS = tuple(h2f.c3g.LAYOUTS)
COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT = LAYOUTS
PREFIXES = tuple(h2h.COMMON_PREFIX_COARSE_FACE_INDICES)
MULTIPLIERS = tuple(h2h.COMMON_PREFIX_FACE_MULTIPLIERS)
GATES = dict(h2h.AUDIT_GATES)
PRIMITIVE_NAMES = tuple(h2h.PRIMITIVE_NAMES)
CHANNELS = tuple(h2h.CONSERVATIVE_CHANNELS)
CHANNEL_INDICES = {"mass": 0, "angular_momentum": 2, "killing_energy": 3}
BLOCK_NAMES = (
    "mapped_temporal_storage",
    "responsive_height_temporal_storage",
    "candidate_shear_principal",
    "candidate_height_principal",
    "candidate_local_stress_relaxation",
    "candidate_geometry",
    "candidate_cooling",
    "candidate_stream",
    "candidate_lower_height_work",
)

ARTIFACT = (
    "causal_inner_nonlinear_5ms_inner_face_half_cell_audit_"
    "wp10c9d6c7c3b5c3h2h1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_inner_face_half_cell_audit_"
    "wp10c9d6c7c3b5c3h2h1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_inner_face_half_cell_audit_"
    "wp10c9d6c7c3b5c3h2h1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_INNER_FACE_"
    "HALF_CELL_AUDIT_WP10C9D6C7C3B5C3H2H1_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float).ravel()
    second = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float).ravel()))


def _validate_parent() -> tuple[dict, dict]:
    manifest_summary = _read_json(h2h.SUMMARY_PATH)
    manifest = _read_json(h2h.MANIFEST_PATH)
    if (
        not manifest_summary["passed"]
        or not manifest_summary["definitions_only"]
        or manifest_summary["classification"]
        != "inner_face_half_cell_audit_manifest_frozen_operator_neutral_"
        "control_volume_diagnostics_authorized"
        or manifest_summary["authorized_next"]
        != "WP10c9d6c7c3b5c3h2h1_operator_neutral_inner_face_half_cell_audit"
        or manifest_summary["fourth_duration_rung_manifest_authorized"]
        or manifest_summary["fixed_q_micro_solver_authorized"]
        or manifest_summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("h2h1 authorization changed")
    if (
        tuple(manifest["common_prefix_coarse_face_indices"]) != PREFIXES
        or tuple(manifest["common_prefix_face_multipliers"]) != MULTIPLIERS
        or dict(manifest["audit_gates"]) != GATES
    ):
        raise RuntimeError("h2h1 frozen contract changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2h1 analyzed identity changed")
    return manifest_summary, manifest


def _standardized_inputs():
    certificate = _load_npz(h2f.DECISIVE_ARRAYS)
    coarse = _load_npz(c3d.DECISIVE_ARRAYS)
    middle = _load_npz(h2d1.DECISIVE_ARRAYS)
    fine = _load_npz(h2e1.DECISIVE_ARRAYS)
    fine_indices = np.asarray(certificate["fine_target_indices"], dtype=int)
    times = np.asarray(certificate["times_seconds"], dtype=float)
    _, layouts, configurations = b2b._layouts_and_contexts(b2b._input_arrays())
    inputs = {
        COARSE_LAYOUT: {
            "layout": layouts[COARSE_LAYOUT],
            "configuration": configurations[COARSE_LAYOUT],
            "times": np.asarray(coarse["main_times_seconds"], dtype=float),
            "base": np.asarray(coarse["base__main__output_states"], dtype=float),
            "anchor": np.asarray(coarse["perturbed__main__output_states"], dtype=float),
            "accepted_indices": np.arange(times.size, dtype=int),
            "arrays": coarse,
        },
        MIDDLE_LAYOUT: {
            "layout": layouts[MIDDLE_LAYOUT],
            "configuration": configurations[MIDDLE_LAYOUT],
            "times": np.asarray(middle["base__accepted_times"], dtype=float),
            "base": np.asarray(middle["base__accepted_states"], dtype=float),
            "anchor": np.asarray(middle["anchor__anchor_states"], dtype=float),
            "accepted_indices": np.arange(times.size, dtype=int),
            "arrays": middle,
        },
        FINE_LAYOUT: {
            "layout": layouts[FINE_LAYOUT],
            "configuration": configurations[FINE_LAYOUT],
            "times": np.asarray(fine["base__accepted_times"], dtype=float),
            "base": np.asarray(fine["base__accepted_states"], dtype=float),
            "anchor": np.asarray(fine["anchor__anchor_states"], dtype=float),
            "accepted_indices": fine_indices,
            "arrays": fine,
        },
    }
    for name, payload in inputs.items():
        selected = payload["times"][payload["accepted_indices"]]
        if not np.array_equal(selected, times):
            raise RuntimeError(f"{name} common accepted times changed")
    return certificate, times, inputs


def _face_histories(times: np.ndarray, inputs: dict):
    selected_time_indices = np.arange(2, times.size, dtype=int)
    histories = {}
    inner_histories = {}
    primitive_contributions = {}
    maximum_ledger = 0.0
    maximum_incoming = 0
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for name in LAYOUTS:
        payload = inputs[name]
        context = payload["configuration"]["context"]
        multiplier = int(payload["layout"].refinement_ratio)
        faces = np.asarray(PREFIXES, dtype=int) * multiplier
        base_faces = []
        anchor_faces = []
        base_inner = []
        anchor_inner = []
        field_paths = []
        for common_index in selected_time_indices:
            state_index = int(payload["accepted_indices"][common_index])
            base_state = payload["base"][state_index]
            anchor_state = payload["anchor"][state_index]
            ledgers = []
            for state in (base_state, anchor_state):
                ledger = causal_five_field_radial_candidate_ledger(context, state)
                ledgers.append(ledger)
                maximum_ledger = max(
                    maximum_ledger,
                    float(ledger.local_block_ledger_defect),
                    float(ledger.source_double_count_defect),
                    float(ledger.interfaces.shared_conservative_face_defect),
                    float(ledger.interfaces.maximum_split_closure_defect),
                )
                maximum_incoming = max(
                    maximum_incoming,
                    int(ledger.interfaces.incoming_excision_characteristics),
                )
            base_flux = ledgers[0].interfaces.candidate_shared_face_fluxes_over_c
            anchor_flux = ledgers[1].interfaces.candidate_shared_face_fluxes_over_c
            base_faces.append(base_flux[faces][:, (0, 2, 3)])
            anchor_faces.append(anchor_flux[faces][:, (0, 2, 3)])
            base_inner.append(base_flux[0, (0, 2, 3)])
            anchor_inner.append(anchor_flux[0, (0, 2, 3)])

            base_trace = causal_five_field_reconstruct_face_charts(
                context, base_state, purpose="flux"
            ).right_face_charts[0]
            anchor_trace = causal_five_field_reconstruct_face_charts(
                context, anchor_state, purpose="flux"
            ).right_face_charts[0]
            delta = np.asarray(anchor_trace - base_trace, dtype=float)
            contribution = np.zeros((5, 5), dtype=float)
            radius = float(context.grid.edges[0])
            measure = float(context.grid.face_measures[0])
            for node, weight in zip(nodes, weights, strict=True):
                fraction = 0.5 * (float(node) + 1.0)
                local = causal_five_field_analytic_local_maps(
                    context,
                    radius,
                    base_trace + fraction * delta,
                )
                contribution += (
                    0.5
                    * float(weight)
                    * measure
                    * local.physical_flux_jacobian
                    * delta[None, :]
                )
            field_paths.append(contribution[(0, 2, 3), :])
        histories[name] = np.asarray(anchor_faces) - np.asarray(base_faces)
        inner_histories[name] = np.asarray(anchor_inner) - np.asarray(base_inner)
        primitive_contributions[name] = np.asarray(field_paths)
    return (
        selected_time_indices,
        histories,
        inner_histories,
        primitive_contributions,
        maximum_ledger,
        maximum_incoming,
    )


def _metric(histories, scales: np.ndarray) -> dict:
    arrays = tuple(np.asarray(values, dtype=float) for values in histories)
    reference = np.asarray(scales, dtype=float)
    activity_floor = (
        float(h2f.SPATIAL_GATES["minimum_relative_activity"]) * reference
    )
    maximum_response = np.max(np.abs(np.stack(arrays, axis=0)), axis=(0, 1))
    if not np.any(maximum_response > activity_floor):
        fine_difference = float(
            np.max(np.abs(arrays[2] - arrays[1]) / reference[None, :])
        )
        return {
            "passed": fine_difference
            <= GATES["maximum_fine_normalized_difference"],
            "upper_bound_route_used": True,
            "observed_rms_order": None,
            "observed_maximum_order": None,
            "minimum_significant_component_order": None,
            "maximum_fine_normalized_difference": fine_difference,
            "history_cosine": 1.0,
            "refinement_error_cosine": 1.0,
        }
    metrics = causal_packet_history_metrics(
        *arrays,
        physical_scales=reference,
        minimum_rms_order=GATES["minimum_spatial_order"],
        minimum_maximum_order=GATES["minimum_spatial_order"],
        minimum_significant_component_order=GATES["minimum_spatial_order"],
        maximum_fine_normalized_difference=GATES[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=GATES["minimum_refinement_error_cosine"],
        minimum_refinement_error_cosine=GATES[
            "minimum_refinement_error_cosine"
        ],
        relative_activity=h2f.SPATIAL_GATES["minimum_relative_activity"],
    )
    return {
        "passed": bool(metrics.passed),
        "upper_bound_route_used": False,
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _recovery_report(histories: dict, scales: np.ndarray) -> dict:
    reports = []
    for position, face in enumerate(PREFIXES):
        metric = _metric(
            tuple(histories[name][:, position, :] for name in LAYOUTS),
            scales,
        )
        reports.append({"coarse_face_index": face, **metric})
    required = int(GATES["minimum_consecutive_recovery_faces"])
    recovery = None
    for index in range(len(reports) - required + 1):
        if all(reports[index + offset]["passed"] for offset in range(required)):
            recovery = reports[index]["coarse_face_index"]
            break
    return {
        "faces": reports,
        "recovery_face_index": recovery,
        "compact_recovery_selected": recovery is not None,
    }


def _primitive_report(contributions: dict, inner_histories: dict) -> dict:
    reports = {}
    stable = {}
    for channel_position, channel in enumerate(CHANNELS):
        pair_reports = {}
        dominant_fields = []
        for pair_name, left, right in (
            ("coarse_middle", COARSE_LAYOUT, MIDDLE_LAYOUT),
            ("middle_fine", MIDDLE_LAYOUT, FINE_LAYOUT),
        ):
            target = (
                inner_histories[right][:, channel_position]
                - inner_histories[left][:, channel_position]
            )
            field_errors = (
                contributions[right][:, channel_position, :]
                - contributions[left][:, channel_position, :]
            )
            norms = np.linalg.norm(field_errors, axis=0)
            total = max(float(np.sum(norms)), np.finfo(float).tiny)
            fractions = norms / total
            alignments = np.asarray(
                [_cosine(field_errors[:, index], target) for index in range(5)]
            )
            dominant = int(np.argmax(fractions))
            dominant_fields.append(dominant)
            closure = np.sum(field_errors, axis=1) - target
            pair_reports[pair_name] = {
                "target_error_norm": _norm(target),
                "field_fractions": {
                    name: float(fractions[index])
                    for index, name in enumerate(PRIMITIVE_NAMES)
                },
                "field_alignments": {
                    name: float(alignments[index])
                    for index, name in enumerate(PRIMITIVE_NAMES)
                },
                "dominant_field": PRIMITIVE_NAMES[dominant],
                "dominant_fraction": float(fractions[dominant]),
                "dominant_alignment": float(alignments[dominant]),
                "path_closure_defect": _norm(closure)
                / max(_norm(target), np.finfo(float).tiny),
            }
        same = dominant_fields[0] == dominant_fields[1]
        selected = bool(
            same
            and all(
                payload["dominant_fraction"]
                >= GATES["minimum_error_dominance_fraction"]
                and payload["dominant_alignment"] >= GATES["minimum_error_alignment"]
                and payload["path_closure_defect"]
                <= GATES["maximum_inner_flux_field_path_closure_defect"]
                for payload in pair_reports.values()
            )
        )
        reports[channel] = {
            **pair_reports,
            "stable_field_localization": selected,
            "selected_field": PRIMITIVE_NAMES[dominant_fields[0]] if selected else None,
        }
        if selected:
            stable[channel] = PRIMITIVE_NAMES[dominant_fields[0]]
    maximum_closure = max(
        payload[pair]["path_closure_defect"]
        for payload in reports.values()
        for pair in ("coarse_middle", "middle_fine")
    )
    return {
        "channels": reports,
        "stable_field_localizations": stable,
        "maximum_path_closure_defect": maximum_closure,
        "path_contract_passed": maximum_closure
        <= GATES["maximum_inner_flux_field_path_closure_defect"],
    }


def _history_direction_for_final(name: str, payload: dict, generic_index: int):
    context = payload["configuration"]["context"]
    if name == COARSE_LAYOUT:
        base = payload["base"]
        anchor = payload["anchor"]
        return causal_five_field_monolithic_bdf_history_direction(
            context,
            base[-3],
            base[-2],
            (anchor[-3] - base[-3])[None, ...],
            (anchor[-2] - base[-2])[None, ...],
        )
    arrays = payload["arrays"]
    old_index = int(payload["accepted_indices"][-1]) - 1
    return CausalFiveFieldMonolithicBDFHistoryDirection(
        previous_primitive_increment=np.asarray(
            arrays["tangent__primitive_history_directions"][old_index, generic_index]
        )[None, ...],
        previous_mapped_storage_increment=np.asarray(
            arrays["tangent__mapped_history_directions"][old_index, generic_index]
        )[None, ...],
        previous_responsive_height_storage_increment=np.asarray(
            arrays["tangent__height_history_directions"][old_index, generic_index]
        )[None, ...],
    )


def _final_discrete_blocks(inputs: dict):
    generic_index = int(h2e1.GENERIC_INDEX)
    all_blocks = {}
    diagnostics = {}
    for name in LAYOUTS:
        payload = inputs[name]
        context = payload["configuration"]["context"]
        arrays = payload["arrays"]
        new_index = int(payload["accepted_indices"][-1])
        old_index = new_index - 1
        base_old = payload["base"][old_index]
        base_new = payload["base"][new_index]
        if name == COARSE_LAYOUT:
            direction_old = payload["anchor"][old_index] - base_old
            current_dt = float(payload["times"][new_index] - payload["times"][old_index])
            previous_dt = float(payload["times"][old_index] - payload["times"][old_index - 1])
        else:
            direction_old = np.asarray(
                arrays["tangent__state_directions"][old_index, generic_index]
            )
            current_dt = float(arrays["base__accepted_timesteps"][old_index])
            previous_dt = float(arrays["base__accepted_previous_timesteps"][old_index])
        columns = np.asarray(
            payload["configuration"]["columns"], dtype=float
        ).reshape(base_old.shape)
        rows = np.asarray(
            payload["configuration"]["rows"], dtype=float
        ).reshape(base_old.shape)
        matrix = causal_five_field_monolithic_discrete_step_matrix(
            context,
            base_old,
            base_new,
            current_dt,
            previous_dt,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        history_direction = _history_direction_for_final(name, payload, generic_index)
        old_scaled = (direction_old / columns).ravel()
        history_mapped = (
            history_direction.previous_mapped_storage_increment[0] / (C * rows)
        ).ravel()
        history_height = (
            history_direction.previous_responsive_height_storage_increment[0]
            / (C * rows)
        ).ravel()
        c0 = float(matrix.current_increment_coefficient)
        c1 = float(matrix.previous_increment_coefficient)
        rhs = (
            c0
            / current_dt
            * (
                matrix.old_mapped_storage_scaled_matrix
                + matrix.old_responsive_height_storage_scaled_matrix
            )
            @ old_scaled
            + c1 / current_dt * (history_mapped + history_height)
        )
        new_scaled = splu(csc_matrix(matrix.scaled_matrix)).solve(-rhs)
        mapped = (
            c0
            / current_dt
            * (
                matrix.old_mapped_storage_scaled_matrix @ old_scaled
                + matrix.mapped_storage_scaled_matrix @ new_scaled
            )
            + c1 / current_dt * history_mapped
        ).reshape(rows.shape) * rows
        height = (
            c0
            / current_dt
            * (
                matrix.old_responsive_height_storage_scaled_matrix @ old_scaled
                + matrix.responsive_height_storage_scaled_matrix @ new_scaled
            )
            + c1 / current_dt * history_height
        ).reshape(rows.shape) * rows
        spatial = matrix.spatial_tangent
        block_rows = {
            "mapped_temporal_storage": mapped,
            "responsive_height_temporal_storage": height,
        }
        for block in BLOCK_NAMES[2:]:
            block_rows[block] = (
                spatial.apply(new_scaled, block=block).reshape(rows.shape) * rows
            )
        face_values = np.einsum(
            "fij,j->fi", spatial.shared_face_flux_scaled_jacobians, new_scaled
        )
        multiplier = int(payload["layout"].refinement_ratio)
        prefix_blocks = {}
        for coarse_face in PREFIXES:
            face = coarse_face * multiplier
            terms = {
                "minus_inner_face_flux": -face_values[0],
                "outer_common_face_flux": face_values[face],
            }
            for block, values in block_rows.items():
                if block == "candidate_conservative_transport":
                    continue
                terms[block] = np.sum(values[:face], axis=0)
            prefix_blocks[coarse_face] = terms
        complete = matrix.scaled_matrix @ new_scaled + rhs
        diagnostics[name] = {
            "current_timestep_seconds": current_dt,
            "previous_timestep_seconds": previous_dt,
            "maximum_component_closure_defect": matrix.maximum_component_closure_defect,
            "linear_residual_relative_defect": _norm(complete)
            / max(_norm(rhs), np.finfo(float).tiny),
            "incoming_excision_characteristics": matrix.incoming_excision_characteristics,
        }
        all_blocks[name] = prefix_blocks
    return all_blocks, diagnostics


def _compensator_report(all_blocks: dict) -> dict:
    reports = {}
    selected = {}
    for channel in CHANNELS:
        component = CHANNEL_INDICES[channel]
        prefix_reports = {}
        for coarse_face in PREFIXES:
            pair_reports = {}
            pair_dominants = []
            for pair_name, left, right in (
                ("coarse_middle", COARSE_LAYOUT, MIDDLE_LAYOUT),
                ("middle_fine", MIDDLE_LAYOUT, FINE_LAYOUT),
            ):
                target = np.asarray(
                    all_blocks[right][coarse_face]["minus_inner_face_flux"][component]
                    - all_blocks[left][coarse_face]["minus_inner_face_flux"][component]
                ).reshape(1)
                compensators = {}
                for block in all_blocks[left][coarse_face]:
                    if block == "minus_inner_face_flux":
                        continue
                    compensators[block] = np.asarray(
                        all_blocks[right][coarse_face][block][component]
                        - all_blocks[left][coarse_face][block][component]
                    ).reshape(1)
                norms = {block: _norm(value) for block, value in compensators.items()}
                total = max(sum(norms.values()), np.finfo(float).tiny)
                fractions = {block: value / total for block, value in norms.items()}
                alignments = {
                    block: _cosine(value, -target) for block, value in compensators.items()
                }
                dominant = max(fractions, key=fractions.get)
                pair_dominants.append(dominant)
                closure = target + sum(compensators.values())
                pair_reports[pair_name] = {
                    "target_inner_error": float(target[0]),
                    "dominant_compensator": dominant,
                    "dominant_fraction": fractions[dominant],
                    "dominant_alignment": alignments[dominant],
                    "compensator_fractions": fractions,
                    "compensator_alignments": alignments,
                    "prefix_balance_closure_defect": _norm(closure)
                    / max(_norm(target), np.finfo(float).tiny),
                }
            stable = bool(
                pair_dominants[0] == pair_dominants[1]
                and all(
                    payload["dominant_fraction"]
                    >= GATES["minimum_error_dominance_fraction"]
                    and payload["dominant_alignment"] >= GATES["minimum_error_alignment"]
                    for payload in pair_reports.values()
                )
            )
            prefix_reports[str(coarse_face)] = {
                **pair_reports,
                "stable_compensator": stable,
                "selected_compensator": pair_dominants[0] if stable else None,
            }
            if stable and coarse_face in (1, 2):
                selected.setdefault(channel, []).append(pair_dominants[0])
        reports[channel] = prefix_reports
    stable_near = {
        channel: values[0]
        for channel, values in selected.items()
        if len(values) == 2 and values[0] == values[1]
    }
    return {"channels": reports, "stable_near_inner_compensators": stable_near}


def _decision(recovery: dict, primitive: dict, compensators: dict) -> tuple[str, str]:
    if recovery["compact_recovery_selected"]:
        return (
            "five_ms_inner_export_recovers_on_common_surface_extraction_surface_manifest_authorized",
            "WP10c9d6c7c3b5c3h2i_conservative_extraction_surface_manifest",
        )
    mass_field = primitive["stable_field_localizations"].get("mass")
    near = compensators["stable_near_inner_compensators"].get("mass")
    if mass_field is not None and near is not None:
        return (
            "five_ms_inner_face_mass_error_has_stable_field_and_half_cell_balance_outgoing_candidate_manifest_authorized",
            "WP10c9d6c7c3b5c3h2i_outgoing_half_cell_candidate_manifest",
        )
    if near in ("mapped_temporal_storage", "responsive_height_temporal_storage"):
        return (
            "five_ms_inner_face_error_balanced_by_temporal_storage_consistency_manifest_authorized",
            "WP10c9d6c7c3b5c3h2i_space_storage_consistency_manifest",
        )
    if near is not None:
        return (
            "five_ms_inner_face_error_balanced_by_distributed_block_consistency_manifest_authorized",
            "WP10c9d6c7c3b5c3h2i_targeted_source_consistency_manifest",
        )
    return (
        "five_ms_inner_face_error_has_no_stable_half_cell_compensator_near_horizon_redesign_manifest_authorized",
        "WP10c9d6c7c3b5c3h2i_near_horizon_space_storage_redesign_manifest",
    )


def _reuse_decisive_analysis(times: np.ndarray):
    """Reload unchanged decisive arrays after report/test-only hardening."""

    decisive = _load_npz(DECISIVE_ARRAYS)
    prior = _read_json(SUMMARY_PATH)
    selected_times = np.asarray(decisive["times_seconds"], dtype=float)
    selected_indices = np.asarray(
        [int(np.flatnonzero(times == value)[0]) for value in selected_times],
        dtype=int,
    )
    face_histories = {
        name: np.asarray(decisive[f"{name}__common_face_flux_response"], dtype=float)
        for name in LAYOUTS
    }
    inner_histories = {
        name: np.asarray(decisive[f"{name}__inner_face_flux_response"], dtype=float)
        for name in LAYOUTS
    }
    primitive_contributions = {
        name: np.asarray(
            decisive[f"{name}__inner_flux_primitive_contributions"], dtype=float
        )
        for name in LAYOUTS
    }
    discrete_blocks = {}
    for name in LAYOUTS:
        discrete_blocks[name] = {}
        for face in PREFIXES:
            discrete_blocks[name][face] = {
                block: np.asarray(decisive[f"{name}__face_{face}__{block}"], dtype=float)
                for block in (
                    "minus_inner_face_flux",
                    "outer_common_face_flux",
                    *BLOCK_NAMES,
                )
            }
    audit = prior["audit"]
    return (
        selected_indices,
        face_histories,
        inner_histories,
        primitive_contributions,
        float(audit["maximum_stationary_ledger_defect"]),
        int(audit["maximum_incoming_excision_characteristics"]),
        discrete_blocks,
        dict(audit["final_discrete_step_diagnostics"]),
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    started = time.perf_counter()
    reuse_decisive = "--reuse-decisive" in sys.argv[1:]
    manifest_summary, _manifest = _validate_parent()
    certificate, times, inputs = _standardized_inputs()
    if reuse_decisive:
        print("h2h1: reuse unchanged decisive arrays", flush=True)
        (
            selected_indices,
            face_histories,
            inner_histories,
            primitive_contributions,
            maximum_ledger,
            maximum_incoming,
            discrete_blocks,
            discrete_diagnostics,
        ) = _reuse_decisive_analysis(times)
    else:
        print("h2h1: evaluate common-face histories", flush=True)
        (
            selected_indices,
            face_histories,
            inner_histories,
            primitive_contributions,
            maximum_ledger,
            maximum_incoming,
        ) = _face_histories(times, inputs)
    print("h2h1: classify recovery and primitive contributions", flush=True)
    flux_scales = np.asarray(certificate["export_scales"][:3], dtype=float)
    recovery = _recovery_report(face_histories, flux_scales)
    primitive = _primitive_report(primitive_contributions, inner_histories)
    if not reuse_decisive:
        print("h2h1: assemble three final accepted-BDF tangents", flush=True)
        discrete_blocks, discrete_diagnostics = _final_discrete_blocks(inputs)
    print("h2h1: classify half-cell compensators", flush=True)
    compensators = _compensator_report(discrete_blocks)
    classification, authorized_next = _decision(recovery, primitive, compensators)
    result = {
        "classification": classification,
        "authorized_next": authorized_next,
        "decisive_arrays_reused_for_metadata_refresh": reuse_decisive,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "selected_common_time_indices": selected_indices,
        "selected_common_times_seconds": times[selected_indices],
        "maximum_stationary_ledger_defect": maximum_ledger,
        "maximum_incoming_excision_characteristics": maximum_incoming,
        "common_face_recovery": recovery,
        "inner_flux_primitive_path": primitive,
        "final_discrete_step_diagnostics": discrete_diagnostics,
        "final_prefix_compensators": compensators,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": True,
        "parent_manifest_classification": manifest_summary["classification"],
        "operator_changed": False,
        "production_defaults_changed": False,
        "propagation_executed": False,
        "five_ms_spatial_convergence_certified": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
        "audit": result,
    }
    decisive = {
        "times_seconds": times[selected_indices],
        "common_prefix_coarse_face_indices": np.asarray(PREFIXES, dtype=int),
    }
    for name in LAYOUTS:
        decisive[f"{name}__common_face_flux_response"] = face_histories[name]
        decisive[f"{name}__inner_face_flux_response"] = inner_histories[name]
        decisive[f"{name}__inner_flux_primitive_contributions"] = primitive_contributions[name]
        for face in PREFIXES:
            for block, values in discrete_blocks[name][face].items():
                decisive[f"{name}__face_{face}__{block}"] = values

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "common_prefix_coarse_face_indices": PREFIXES,
            "primitive_names": PRIMITIVE_NAMES,
            "channels": CHANNELS,
            "gates": GATES,
        },
    )
    if not reuse_decisive:
        np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "scientific_status": "CERTIFIED",
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "manifest_summary_sha256": _sha256(h2h.SUMMARY_PATH),
            "input_hashes": {
                "certificate": _sha256(h2f.DECISIVE_ARRAYS),
                "coarse": _sha256(c3d.DECISIVE_ARRAYS),
                "middle": _sha256(h2d1.DECISIVE_ARRAYS),
                "fine": _sha256(h2e1.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            "command": (
                f"PYTHONPATH=src:scripts python {THIS_RUNNER}"
                + (" --reuse-decisive" if reuse_decisive else "")
            ),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 5 ms inner-face/half-cell audit WP10c9d6c7c3b5c3h2h1",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "No state was propagated and no operator was changed. Seven committed late-history targets were evaluated on ten nested common physical faces. The final accepted BDF tangent was decomposed into its exact temporal-storage, shared-face, principal, and lower-source actions.",
                "",
                "## Result",
                "",
                f"- compact common-face recovery selected: `{recovery['compact_recovery_selected']}`",
                f"- recovery coarse face index: `{recovery['recovery_face_index']}`",
                f"- stable inner-flux primitive fields: `{primitive['stable_field_localizations']}`",
                f"- primitive-path contract passed / maximum defect: `{primitive['path_contract_passed']}` / `{primitive['maximum_path_closure_defect']:.6e}`",
                f"- stable near-inner BDF compensators: `{compensators['stable_near_inner_compensators']}`",
                f"- maximum incoming excision characteristics: `{maximum_incoming}`",
                "",
                "## Decision",
                "",
                f"Only `{authorized_next}` is authorized. The rejected 5 ms certificate, fourth duration rung, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
