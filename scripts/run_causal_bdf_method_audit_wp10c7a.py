"""Audit the increment-primary BDF1/BDF2 method contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldBDFRestart,
    audit_causal_five_field_dae_jacobian,
    causal_bdf_coefficients,
    causal_bdf_discrete_ledger,
    causal_bdf_increment_rate,
    causal_five_field_bdf_history,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_dae_scaling,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_state_from_primitives,
    causal_trapezoidal_physical_interval_ledger,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_bdf,
    evaluate_causal_five_field_increment_backward_euler,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_bdf_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "a11cad99f3270f738fde97ec0c74954af45a3914"
N_CELLS = 4
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables"
    / "causal_bdf_method_audit_wp10c7a.json"
)
CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c7a"
    / "causal_wp10c7a_N004_method_restart.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _orders(errors: list[float]) -> list[float]:
    return [
        float(np.log2(errors[index] / errors[index + 1]))
        for index in range(len(errors) - 1)
    ]


def _scalar_relaxation(subdivisions: int, rate: float) -> float:
    timestep = 1.0 / subdivisions
    previous = 1.0
    current = previous / (1.0 + rate * timestep)
    coefficients = causal_bdf_coefficients(2, timestep, timestep)
    a0 = coefficients.current_increment_coefficient
    a_previous = coefficients.previous_increment_coefficient
    for _index in range(1, subdivisions):
        new = (
            (a0 - a_previous) * current
            + a_previous * previous
        ) / (a0 + rate * timestep)
        previous, current = current, new
    return float(current)


def _index_one_dae(subdivisions: int) -> tuple[float, float]:
    timestep = 1.0 / subdivisions
    previous_x = 1.0
    current_x = previous_x / (1.0 + 0.5 * timestep)
    current_z = 0.5 * current_x
    coefficients = causal_bdf_coefficients(2, timestep, timestep)
    a0 = coefficients.current_increment_coefficient
    a_previous = coefficients.previous_increment_coefficient
    for _index in range(1, subdivisions):
        matrix = np.asarray(
            [
                [a0 / timestep + 1.0, -1.0],
                [-0.5, 1.0],
            ]
        )
        right = np.asarray(
            [
                (
                    (a0 - a_previous) * current_x
                    + a_previous * previous_x
                )
                / timestep,
                0.0,
            ]
        )
        new_x, new_z = np.linalg.solve(matrix, right)
        previous_x, current_x = current_x, float(new_x)
        current_z = float(new_z)
    return current_x, current_z


def _three_level_states():
    context = make_causal_five_field_regression_context(N_CELLS)
    current = make_causal_five_field_seed(context)
    previous_primitives = np.array(current.primitives, copy=True)
    new_primitives = np.array(current.primitives, copy=True)
    interior = slice(None, -1)
    previous_primitives[interior, 0] -= 1.0e-4
    previous_primitives[interior, 1] -= 2.0e-5
    previous_primitives[interior, 2] += 1.0e-5
    previous_primitives[interior, 3] -= 1.5e-4
    previous_primitives[interior, 4] *= 0.9998
    new_primitives[interior, 0] += 1.2e-4
    new_primitives[interior, 1] += 1.0e-5
    new_primitives[interior, 2] -= 1.5e-5
    new_primitives[interior, 3] += 2.0e-4
    new_primitives[interior, 4] *= 1.0003
    previous = causal_five_field_state_from_primitives(
        context,
        previous_primitives,
    )
    new = causal_five_field_state_from_primitives(
        context,
        new_primitives,
    )
    return (
        context,
        pack_causal_five_field_state(previous),
        pack_causal_five_field_state(current),
        pack_causal_five_field_state(new),
    )


def _maximum_relative_defect(
    measured: np.ndarray,
    expected: np.ndarray,
) -> float:
    scale = np.maximum(
        np.maximum(np.abs(measured), np.abs(expected)),
        1.0e-30,
    )
    return float(np.max(np.abs(measured - expected) / scale))


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)

    first = causal_bdf_coefficients(1, 0.25)
    second = causal_bdf_coefficients(2, 0.25, 0.25)
    previous_time = 0.3
    old_time = 1.0
    new_time = 1.4
    variable = causal_bdf_coefficients(
        2,
        new_time - old_time,
        old_time - previous_time,
    )
    quadratic_rate = float(
        causal_bdf_increment_rate(
            new_time**2 - old_time**2,
            old_time**2 - previous_time**2,
            variable,
        )
    )
    quadratic_defect = abs(quadratic_rate - 2.0 * new_time)

    scalar_subdivisions = (40, 80, 160)
    scalar_rate = 4.0
    scalar_exact = float(np.exp(-scalar_rate))
    scalar_solutions = [
        _scalar_relaxation(level, scalar_rate)
        for level in scalar_subdivisions
    ]
    scalar_errors = [
        abs(solution - scalar_exact)
        for solution in scalar_solutions
    ]
    scalar_orders = _orders(scalar_errors)

    dae_subdivisions = (20, 40, 80)
    dae_exact = float(np.exp(-0.5))
    dae_solutions = [
        _index_one_dae(level) for level in dae_subdivisions
    ]
    dae_errors = [
        abs(solution[0] - dae_exact)
        for solution in dae_solutions
    ]
    dae_orders = _orders(dae_errors)
    maximum_algebraic_defect = max(
        abs(algebraic - 0.5 * differential)
        for differential, algebraic in dae_solutions
    )

    vertical_timesteps = (0.1, 0.05, 0.025)
    vertical_errors = []
    for timestep in vertical_timesteps:
        coefficients = causal_bdf_coefficients(
            2,
            timestep,
            timestep,
        )
        vertical_rate = float(
            causal_bdf_increment_rate(
                np.sin(1.0) - np.sin(1.0 - timestep),
                (
                    np.sin(1.0 - timestep)
                    - np.sin(1.0 - 2.0 * timestep)
                ),
                coefficients,
            )
        )
        vertical_errors.append(abs(vertical_rate - np.cos(1.0)))
    vertical_orders = _orders(vertical_errors)

    ledger_timestep = 0.1
    ledger_rate = 2.0
    ledger_coefficients = causal_bdf_coefficients(
        2,
        ledger_timestep,
        ledger_timestep,
    )
    previous_previous = 1.0
    previous = np.exp(-ledger_rate * ledger_timestep)
    current = (
        (
            ledger_coefficients.current_increment_coefficient
            - ledger_coefficients.previous_increment_coefficient
        )
        * previous
        + ledger_coefficients.previous_increment_coefficient
        * previous_previous
    ) / (
        ledger_coefficients.current_increment_coefficient
        + ledger_rate * ledger_timestep
    )
    discrete_ledger = causal_bdf_discrete_ledger(
        current - previous,
        previous - previous_previous,
        ledger_rate * current,
        ledger_coefficients,
    )
    physical_defects = []
    for timestep in vertical_timesteps:
        old = 1.0
        new = np.exp(-ledger_rate * timestep)
        physical = causal_trapezoidal_physical_interval_ledger(
            new - old,
            ledger_rate * old,
            ledger_rate * new,
            timestep,
        )
        physical_defects.append(abs(float(physical.closure_defect)))
    physical_orders = _orders(physical_defects)

    context, previous_vector, current_vector, new_vector = (
        _three_level_states()
    )
    previous_increment = current_vector - previous_vector
    current_increment = new_vector - current_vector
    previous_dt = 3.0e-4
    current_dt = 2.0e-4
    history = causal_five_field_bdf_history(
        context,
        current_vector,
        previous_increment,
        previous_dt,
    )
    bdf2 = evaluate_causal_five_field_increment_bdf(
        current_increment,
        context,
        old_vector=current_vector,
        timestep_seconds=current_dt,
        order=2,
        history=history,
    )
    bdf2_coefficients = causal_bdf_coefficients(
        2,
        current_dt,
        previous_dt,
    )
    n_cells = context.grid.centers.size
    current_conserved = current_increment[: 5 * n_cells].reshape(
        n_cells,
        5,
    )
    previous_conserved = previous_increment[: 5 * n_cells].reshape(
        n_cells,
        5,
    )
    expected_conserved = (
        context.grid.cell_measures[:, None]
        * causal_bdf_increment_rate(
            current_conserved,
            previous_conserved,
            bdf2_coefficients,
        )
        / C
    )
    current_state = unpack_causal_five_field_state(
        current_vector,
        n_cells,
    )
    new_state = unpack_causal_five_field_state(new_vector, n_cells)
    current_vertical = causal_five_field_path_temporal_storage_increment(
        context,
        current_state.primitives,
        new_state.primitives,
    ).vertical_killing_increment
    expected_vertical = (
        context.grid.cell_measures[:, None]
        * causal_bdf_increment_rate(
            current_vertical,
            history.previous_vertical_killing_increment,
            bdf2_coefficients,
        )
        / C
    )
    conserved_rate_defect = _maximum_relative_defect(
        bdf2.temporal_conserved_storage,
        expected_conserved,
    )
    vertical_rate_defect = _maximum_relative_defect(
        bdf2.temporal_vertical_storage,
        expected_vertical,
    )

    bdf1 = evaluate_causal_five_field_increment_bdf(
        current_increment,
        context,
        old_vector=current_vector,
        timestep_seconds=current_dt,
        order=1,
    )
    backward_euler = (
        evaluate_causal_five_field_increment_backward_euler(
            current_increment,
            context,
            old_vector=current_vector,
            timestep_seconds=current_dt,
        )
    )
    bdf1_parity_defect = float(
        np.max(np.abs(bdf1.residual - backward_euler.residual))
    )

    rank_dt = 0.1
    rank_history = causal_five_field_bdf_history(
        context,
        current_vector,
        previous_increment,
        rank_dt,
    )
    scaling = causal_five_field_dae_scaling(
        current_state,
        evaluate_causal_five_field_dae(current_vector, context),
    )
    rank_audit = audit_causal_five_field_dae_jacobian(
        lambda increment: evaluate_causal_five_field_increment_bdf(
            increment,
            context,
            old_vector=current_vector,
            timestep_seconds=rank_dt,
            order=2,
            history=rank_history,
        ).residual,
        np.zeros_like(current_vector),
        scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )

    restart = CausalFiveFieldBDFRestart(
        state_vector=current_vector,
        history=history,
        elapsed_time=0.25,
        dt_next=2.5e-4,
        next_order=2,
        accepted_steps=8,
        rejected_attempts=1,
        provenance={
            "work_package": "WP10c7a",
            "base_commit": BASE_COMMIT,
            "role": "method_restart_audit",
        },
    )
    save_causal_five_field_bdf_restart(
        CHECKPOINT,
        context,
        restart,
    )
    restored = load_causal_five_field_bdf_restart(
        CHECKPOINT,
        context,
    )
    restart_bitwise = causal_five_field_bdf_restarts_equal(
        restart,
        restored,
    )

    gates = {
        "quadratic_derivative_defect": quadratic_defect <= 1.0e-14,
        "minimum_scalar_order": min(scalar_orders) >= 1.8,
        "minimum_index_one_order": min(dae_orders) >= 1.8,
        "maximum_algebraic_defect": (
            maximum_algebraic_defect <= 2.0e-14
        ),
        "minimum_vertical_order": min(vertical_orders) >= 1.9,
        "discrete_ledger_defect": (
            abs(float(discrete_ledger.closure_defect)) <= 1.0e-14
        ),
        "minimum_physical_ledger_order": (
            min(physical_orders) >= 2.9
        ),
        "bdf1_backward_euler_parity": bdf1_parity_defect == 0.0,
        "declared_conserved_history_defect": (
            conserved_rate_defect <= 2.0e-13
        ),
        "vertical_history_defect": vertical_rate_defect <= 2.0e-13,
        "bdf2_jacobian_full_rank": rank_audit.full_rank,
        "restart_roundtrip_bitwise": restart_bitwise,
    }
    passed = all(gates.values())
    output = {
        "work_package": "WP10c7a",
        "scope": (
            "method-level increment-primary BDF1/BDF2 audit; "
            "no production disk trajectory"
        ),
        "base_commit": BASE_COMMIT,
        "coefficients": {
            "bdf1": {
                "current_increment": (
                    first.current_increment_coefficient
                ),
                "previous_increment": (
                    first.previous_increment_coefficient
                ),
            },
            "equal_step_bdf2": {
                "current_increment": (
                    second.current_increment_coefficient
                ),
                "previous_increment": (
                    second.previous_increment_coefficient
                ),
            },
            "variable_step_quadratic": {
                "step_ratio": variable.step_ratio,
                "derivative": quadratic_rate,
                "defect": quadratic_defect,
            },
        },
        "scalar_relaxation": {
            "subdivisions": list(scalar_subdivisions),
            "solutions": scalar_solutions,
            "errors": scalar_errors,
            "orders": scalar_orders,
        },
        "index_one_dae": {
            "subdivisions": list(dae_subdivisions),
            "solutions": [
                list(solution) for solution in dae_solutions
            ],
            "errors": dae_errors,
            "orders": dae_orders,
            "maximum_algebraic_defect": maximum_algebraic_defect,
        },
        "manufactured_vertical_storage": {
            "timesteps": list(vertical_timesteps),
            "errors": vertical_errors,
            "orders": vertical_orders,
        },
        "ledgers": {
            "discrete_defect": float(
                discrete_ledger.closure_defect
            ),
            "physical_interval_defects": physical_defects,
            "physical_interval_orders": physical_orders,
        },
        "five_field": {
            "n_cells": N_CELLS,
            "bdf1_backward_euler_parity_defect": (
                bdf1_parity_defect
            ),
            "declared_conserved_history_relative_defect": (
                conserved_rate_defect
            ),
            "vertical_history_relative_defect": (
                vertical_rate_defect
            ),
            "rank_timestep_seconds": rank_dt,
            "jacobian_dimensions": list(rank_audit.dimensions),
            "jacobian_rank": rank_audit.numerical_rank,
            "jacobian_condition_estimate": (
                rank_audit.condition_estimate
            ),
        },
        "restart": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "sha256": _sha256(CHECKPOINT),
            "roundtrip_bitwise": restart_bitwise,
        },
        "gates": gates,
        "authorization": {
            "wp10c7b_fixed_n16_bdf2_authorized": passed,
            "wp10c7c_adaptive_n16_bdf2_authorized": False,
            "wp10c7d_matched_n32_bdf2_authorized": False,
            "n64_n128_production_authorized": False,
            "long_evolution_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_certified": False,
        },
        "decision": (
            "authorize_wp10c7b_fixed_n16_bdf2"
            if passed
            else "stop_wp10c7a_method_gate_failed"
        ),
        "passed": passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "decision": output["decision"],
                "passed": passed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
