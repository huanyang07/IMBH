from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/"
    "run_causal_inner_uncolored_additivity_wp10c9d5c0c.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_uncolored_additivity_wp10c9d5c0c"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5c0c_runner",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_uncolored_additivity_contract_is_predeclared() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "508e5c284c2eaf7305efb45ae30a437b29dabb33"
    )
    assert module.ANALYZED_BASE_PARENT == (
        "5c88fa02f8f25fa62e9e0fdb648e66974bca38d3"
    )
    assert module.ANALYZED_BASE_TREE == (
        "6fd889e2f6acc27304f1b243011c8bf255ae5d0b"
    )
    assert module.LABEL == "N128_exterior_N128_inner_c48"
    assert module.DIRECTION_NAME == "calibration_global_inner_0"
    assert module.DERIVATIVE_ORDERS == (4, 6)
    assert module.FINITE_DIFFERENCE_STEP == 2.0e-4
    assert module.SELECTED_COLUMN_COUNT == 12
    assert module.SELECTION_RADIUS_OVER_RG == 8.0
    assert module.MAXIMUM_ACTION_DEFECT == 5.0e-5
    assert module.FAIL_FAST_AFTER_CELL_ADDITIVITY is True


def test_selected_columns_use_weighted_matrix_contribution(
    monkeypatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        module.wp10c9d5c0b.wp10c9d5c0,
        "_actual_faces",
        lambda _configuration: {8.0: 1},
    )
    monkeypatch.setattr(
        module.wp10c9d5c0b.wp10c9d5c0,
        "_region_rows",
        lambda _configuration, _face, *, halo: np.asarray([0, 1]),
    )
    monkeypatch.setattr(module, "SELECTED_COLUMN_COUNT", 2)
    matrix = csr_matrix(
        np.asarray(
            [
                [1.0, 3.0, 0.0, 2.0],
                [0.0, 4.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    direction = np.asarray([2.0, 1.0, 1.0, 4.0])
    columns, weights = module._selected_columns(
        {},
        direction,
        matrix,
    )
    np.testing.assert_array_equal(columns, np.asarray([3, 1]))
    np.testing.assert_allclose(weights, np.asarray([8.0, 5.0]))


def test_canonical_uncolored_additivity_evidence_is_self_consistent() -> None:
    module = _module()
    required = (
        CANONICAL / "config.json",
        CANONICAL / "decisive_arrays.npz",
        CANONICAL / "provenance.json",
        CANONICAL / "summary.json",
        CANONICAL / "SHA256SUMS.txt",
    )
    assert all(path.exists() for path in required)
    for line in (CANONICAL / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", maxsplit=1)
        assert _sha256(CANONICAL / name) == expected
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["work_package"] == "WP10c9d5c0c"
    assert summary["parent_wp10c9d5c0b_remains_rejected"] is True
    assert summary["parent_wp10c9d5_candidate_remains_rejected"] is True
    assert summary["parent_wp10c9d5b_branch_d_preserved"] is True
    assert summary["explicit_finite_difference_matrix_authorized"] is False
    assert (
        summary["matrix_free_finite_difference_jvp_authorized"]
        is False
    )
    assert (
        summary["analytic_or_ad_linear_tangent_work_authorized"]
        is True
    )
    assert (
        summary["wp10c9d5c1_extended_localization_authorized"]
        is False
    )
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["decisive_arrays_sha256"] == _sha256(
        CANONICAL / "decisive_arrays.npz"
    )
    with np.load(CANONICAL / "decisive_arrays.npz") as archive:
        assert set(archive.files) == set(summary["decisive_array_hashes"])
        for name in archive.files:
            assert (
                module._array_sha256(archive[name])
                == summary["decisive_array_hashes"][name]
            )
