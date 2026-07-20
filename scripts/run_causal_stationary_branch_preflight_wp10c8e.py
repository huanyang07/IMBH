"""Bound stationary solvability and root-predictor preflight for WP10c8e."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve

import run_causal_characteristic_extension_wp10c7l as wp10c7l
from imri_qpe.layer3_minidisk_1d import (
    KerrSchildCellSourceRates,
    audit_causal_five_field_state_gates,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_regression_seed_parameters,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4247696c2c65039fc4c08d6aaca7cbace8be6636"
WP10C8D_OUTPUT = (
    ROOT / "outputs/tables/causal_mixed_mode_reduction_audit_wp10c8d.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_stationary_branch_preflight_wp10c8e.json"
)
PREFLIGHT_RESOLUTION = 16
CONFIRMATION_RESOLUTION = 32
SOURCE_AMPLITUDES = (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0)
NEWTON_DAMPING_FACTORS = (
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
)
MAXIMUM_STATIONARY_CONDITION_ESTIMATE = 1.0e14
MAXIMUM_PREDICTOR_RESIDUAL_RATIO = 0.5
MAXIMUM_SCALED_PREDICTOR_CHANGE = 1.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate WP10c8d evidence without constructing descriptors.",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_authorization() -> tuple[dict, str]:
    if not WP10C8D_OUTPUT.exists():
        raise RuntimeError("WP10c8e requires canonical WP10c8d evidence")
    evidence = json.loads(WP10C8D_OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / str(evidence.get("artifacts", {}).get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c8d"
        and evidence.get("decision")
        == "wp10c8d_compact_cross_mesh_markovian_basis_not_found"
        and evidence.get("next_authorization")
        == "stationary_branch_preflight_or_narrower_observable_audit"
        and not evidence.get("gates", {}).get(
            "compact_cross_mesh_basis_found",
            True,
        )
        and not evidence.get("gates", {}).get(
            "nonlinear_rom_authorized",
            True,
        )
        and arrays.exists()
        and _sha256(arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c8d did not authorize branch preflight")
    return evidence, _sha256(WP10C8D_OUTPUT)


def _scaled_stream_context(n_cells: int, amplitude: float):
    context = make_causal_five_field_regression_context(
        n_cells,
        **wp10c7l.SPATIAL_OPTIONS,
    )
    source = context.stream_sources
    if source is None:
        raise RuntimeError("WP10c8e requires the exact stream source")
    factor = float(amplitude)
    scaled = KerrSchildCellSourceRates(
        rest_mass=factor * np.asarray(source.rest_mass, dtype=float),
        radial_momentum_over_c=(
            factor * np.asarray(source.radial_momentum_over_c, dtype=float)
        ),
        angular_momentum_over_c=(
            factor * np.asarray(source.angular_momentum_over_c, dtype=float)
        ),
        killing_energy_over_c2=(
            factor * np.asarray(source.killing_energy_over_c2, dtype=float)
        ),
    )
    return replace(context, stream_sources=scaled).validated()


def _stationary_ledger(context, vector: np.ndarray) -> dict:
    state = causal_five_field_state_from_primitives(
        context,
        (
            vector[
                5 * context.grid.centers.size :
                10 * context.grid.centers.size
            ].reshape(context.grid.centers.size, 5)
        ),
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    transport = (
        state.weighted_face_fluxes_over_c[-1]
        - state.weighted_face_fluxes_over_c[0]
    )
    sources = np.sum(evaluation.integrated_sources_per_ct, axis=0)
    defect = transport - sources
    names = {
        0: "rest_mass",
        2: "angular_momentum",
        3: "killing_energy",
    }
    rows = {}
    for component, name in names.items():
        scale = max(
            abs(float(transport[component])),
            abs(float(sources[component])),
            np.finfo(float).tiny,
        )
        rows[name] = {
            "boundary_transport_per_ct": float(transport[component]),
            "integrated_source_per_ct": float(sources[component]),
            "stationary_defect_per_ct": float(defect[component]),
            "relative_defect": float(abs(defect[component]) / scale),
        }
    return {
        "components": rows,
        "maximum_relative_defect": max(
            row["relative_defect"] for row in rows.values()
        ),
    }


def _trial_row(
    context,
    base_primitives: np.ndarray,
    primitive_scales: np.ndarray,
    scaled_delta: np.ndarray,
    row_scales: np.ndarray,
    base_maximum: float,
    base_norm: float,
    damping: float,
) -> dict:
    trial_primitives = (
        base_primitives
        + float(damping) * primitive_scales * scaled_delta
    )
    try:
        state = causal_five_field_state_from_primitives(
            context,
            trial_primitives.reshape(context.grid.centers.size, 5),
        )
        vector = pack_causal_five_field_state(state)
        gates = audit_causal_five_field_state_gates(context, vector)
        residual = (
            causal_five_field_reduced_stationary_residual(
                trial_primitives,
                context,
            )
            / row_scales
        )
        maximum_ratio = float(
            np.max(np.abs(residual)) / max(base_maximum, np.finfo(float).tiny)
        )
        norm_ratio = float(
            np.linalg.norm(residual) / max(base_norm, np.finfo(float).tiny)
        )
        return {
            "damping_factor": float(damping),
            "state_constructed": True,
            "state_gates_passed": bool(gates["passed"]),
            "maximum_residual_ratio": maximum_ratio,
            "l2_residual_ratio": norm_ratio,
            "maximum_scaled_primitive_change": float(
                abs(damping) * np.max(np.abs(scaled_delta))
            ),
            "stationary_ledger": _stationary_ledger(context, vector),
            "error": None,
        }
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {
            "damping_factor": float(damping),
            "state_constructed": False,
            "state_gates_passed": False,
            "maximum_residual_ratio": None,
            "l2_residual_ratio": None,
            "maximum_scaled_primitive_change": float(
                abs(damping) * np.max(np.abs(scaled_delta))
            ),
            "stationary_ledger": None,
            "error": str(exc),
        }


def _amplitude_audit(n_cells: int, amplitude: float) -> dict:
    started = time.perf_counter()
    context = _scaled_stream_context(n_cells, amplitude)
    try:
        seed_parameters = causal_five_field_regression_seed_parameters(
            context
        )
        state = make_causal_five_field_seed(context, **seed_parameters)
        vector = pack_causal_five_field_state(state)
        state_gates = audit_causal_five_field_state_gates(context, vector)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {
            "n_cells": n_cells,
            "source_amplitude": float(amplitude),
            "seed_constructed": False,
            "seed_state_gates_passed": False,
            "error": str(exc),
            "root_predictor_authorized": False,
            "wall_seconds": time.perf_counter() - started,
        }
    if not state_gates["passed"]:
        return {
            "n_cells": n_cells,
            "source_amplitude": float(amplitude),
            "seed_constructed": True,
            "seed_state_gates_passed": False,
            "error": "source-compatible seed failed physical gates",
            "root_predictor_authorized": False,
            "wall_seconds": time.perf_counter() - started,
        }

    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    stationary = np.asarray(
        reduced["stationary_reduced_scaled_jacobian"],
        dtype=float,
    )
    row_scales = np.asarray(
        reduced["conservation_row_scales"],
        dtype=float,
    )
    primitive_scales = np.asarray(
        reduced["primitive_column_scales"],
        dtype=float,
    )
    base_primitives = np.asarray(state.primitives, dtype=float).ravel()
    residual = (
        causal_five_field_reduced_stationary_residual(
            base_primitives,
            context,
        )
        / row_scales
    )
    singular = np.linalg.svd(stationary, compute_uv=False)
    rank_threshold = max(
        1.0e-11 * float(singular[0]),
        np.finfo(float).eps * stationary.shape[0] * float(singular[0]),
    )
    rank = int(np.count_nonzero(singular > rank_threshold))
    condition = float(singular[0] / singular[-1])
    scaled_delta = solve(
        stationary,
        -residual,
        assume_a="gen",
        check_finite=True,
    )
    base_maximum = float(np.max(np.abs(residual)))
    base_norm = float(np.linalg.norm(residual))
    trials = [
        _trial_row(
            context,
            base_primitives,
            primitive_scales,
            scaled_delta,
            row_scales,
            base_maximum,
            base_norm,
            damping,
        )
        for damping in NEWTON_DAMPING_FACTORS
    ]
    valid_trials = [
        row
        for row in trials
        if row["state_constructed"] and row["state_gates_passed"]
    ]
    best = (
        min(
            valid_trials,
            key=lambda row: row["maximum_residual_ratio"],
        )
        if valid_trials
        else None
    )
    authorized = bool(
        rank == stationary.shape[0]
        and condition <= MAXIMUM_STATIONARY_CONDITION_ESTIMATE
        and best is not None
        and best["maximum_residual_ratio"]
        <= MAXIMUM_PREDICTOR_RESIDUAL_RATIO
        and best["maximum_scaled_primitive_change"]
        <= MAXIMUM_SCALED_PREDICTOR_CHANGE
    )
    return {
        "n_cells": n_cells,
        "source_amplitude": float(amplitude),
        "seed_constructed": True,
        "seed_state_gates_passed": True,
        "seed_stationary_ledger": _stationary_ledger(context, vector),
        "stationary_dimensions": reduced["dimensions"],
        "stationary_rank": rank,
        "stationary_condition_estimate": condition,
        "base_maximum_scaled_residual": base_maximum,
        "base_l2_scaled_residual": base_norm,
        "maximum_scaled_newton_correction": float(
            np.max(np.abs(scaled_delta))
        ),
        "l2_scaled_newton_correction": float(np.linalg.norm(scaled_delta)),
        "trials": trials,
        "best_physical_trial": best,
        "root_predictor_authorized": authorized,
        "error": None,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    authorization, authorization_sha256 = _validate_authorization()
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8e",
                    "preflight_passed": True,
                    "wp10c8d_evidence_sha256": authorization_sha256,
                    "wp10c8d_decision": authorization["decision"],
                },
                sort_keys=True,
            )
        )
        return

    coarse = {}
    for amplitude in SOURCE_AMPLITUDES:
        coarse[str(amplitude)] = _amplitude_audit(
            PREFLIGHT_RESOLUTION,
            amplitude,
        )
        print(
            json.dumps(
                {
                    "work_package": "WP10c8e",
                    "n_cells": PREFLIGHT_RESOLUTION,
                    "source_amplitude": amplitude,
                    "seed_constructed": coarse[str(amplitude)][
                        "seed_constructed"
                    ],
                    "root_predictor_authorized": coarse[str(amplitude)][
                        "root_predictor_authorized"
                    ],
                },
                sort_keys=True,
            )
        )
    authorized_amplitudes = [
        amplitude
        for amplitude in SOURCE_AMPLITUDES
        if coarse[str(amplitude)]["root_predictor_authorized"]
    ]
    confirmation = {}
    for amplitude in authorized_amplitudes:
        confirmation[str(amplitude)] = _amplitude_audit(
            CONFIRMATION_RESOLUTION,
            amplitude,
        )
    matched = [
        amplitude
        for amplitude in authorized_amplitudes
        if confirmation[str(amplitude)]["root_predictor_authorized"]
    ]
    branch_anchor_authorized = bool(matched)
    decision = (
        "wp10c8e_matched_stationary_anchor_solve_authorized"
        if branch_anchor_authorized
        else "wp10c8e_stationary_anchor_solve_not_authorized"
    )
    payload = {
        "work_package": "WP10c8e",
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "next_authorization": (
            "bounded_n16_n32_stationary_newton_anchor"
            if branch_anchor_authorized
            else "retain_full_dae_microbursts_and_narrow_scientific_observables"
        ),
        "scope": {
            "description": (
                "Integrated-ledger, rank, and one-step damped stationary "
                "root-predictor preflight"
            ),
            "preflight_resolution": PREFLIGHT_RESOLUTION,
            "conditional_confirmation_resolution": (
                CONFIRMATION_RESOLUTION
            ),
            "source_amplitudes": SOURCE_AMPLITUDES,
            "newton_damping_factors": NEWTON_DAMPING_FACTORS,
            "full_stationary_root_solved": False,
            "pseudo_arclength_continuation_run": False,
            "hot_branch_search_run": False,
        },
        "authorization": {
            "wp10c8d_decision": authorization["decision"],
            "wp10c8d_evidence_sha256": authorization_sha256,
        },
        "n16_amplitude_audits": coarse,
        "n32_confirmation_audits": confirmation,
        "n16_authorized_amplitudes": authorized_amplitudes,
        "matched_authorized_amplitudes": matched,
        "gates": {
            "maximum_stationary_condition_estimate": (
                MAXIMUM_STATIONARY_CONDITION_ESTIMATE
            ),
            "maximum_predictor_residual_ratio": (
                MAXIMUM_PREDICTOR_RESIDUAL_RATIO
            ),
            "maximum_scaled_predictor_change": (
                MAXIMUM_SCALED_PREDICTOR_CHANGE
            ),
            "matched_stationary_anchor_solve_authorized": (
                branch_anchor_authorized
            ),
        },
        "interpretation": {
            "invalid_low_source_seeds": (
                "The current optically thick diffusion closure may reject "
                "zero or weak source-compatible seeds before root solving."
            ),
            "negative_preflight_scope": (
                "A failed local predictor blocks this branch strategy from "
                "the tested seeds; it is not a global nonexistence theorem."
            ),
        },
    }
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "work_package": "WP10c8e",
                "decision": decision,
                "matched_authorized_amplitudes": matched,
                "output": _relative(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
