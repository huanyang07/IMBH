from __future__ import annotations

import numpy as np

import run_causal_inner_coordinate_hessian_diagnosis_wp10c9d6c7c3b5c4f25au as f25au


def test_frozen_coordinate_hessian_manifest_is_locked():
    frozen = f25au._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25au.WORK_PACKAGE


def test_five_cell_coloring_recovers_a_synthetic_block_banded_hessian():
    cells = 17
    dimension = 5 * cells
    rng = np.random.default_rng(25)
    hessian = np.zeros((dimension, dimension))
    for column in range(dimension):
        cell = column // 5
        rows = f25au._row_indices_for_cell(cell, cells)
        hessian[rows, column] = rng.normal(size=rows.size)
    responses = {}
    for field in range(5):
        for color in range(5):
            direction = np.zeros(dimension)
            direction[f25au._columns_for_color(cells, field, color)] = 1.0
            responses[(field, color)] = hessian @ direction
    recovered, leakage = f25au._recover_from_colored_responses(responses, cells)
    assert np.array_equal(recovered, hessian)
    assert leakage == 0.0


def test_complete_tangent_uses_the_coordinate_hessian_with_negative_sign():
    source = f25au.__file__
    text = open(source, encoding="utf-8").read()
    assert "generator / system[\"rate_scale\"] - hessian" in text


def test_diagnosis_has_no_physical_rate_execution_path():
    contract = f25au.manifest._contract()
    assert contract["sparse_recovery"]["new_fixed_Q_rate_evaluations"] == 0
    assert not contract["claim_boundary"]["physical_conditional_branch_found"]
