from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/"
    "run_causal_inner_analytic_tangent_wp10c9d5c0d.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_analytic_tangent_wp10c9d5c0d"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5c0d_runner",
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


def test_analytic_tangent_contract_is_predeclared() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "e492299df5668b49412f033e33df3d42e92f512e"
    )
    assert module.ANALYZED_BASE_PARENT == (
        "508e5c284c2eaf7305efb45ae30a437b29dabb33"
    )
    assert module.ANALYZED_BASE_TREE == (
        "8ee7f16a299ef0f1d0a22093cff2a9b4c4d983ec"
    )
    assert module.LABEL == "N128_exterior_N128_inner_c48"
    assert module.PATH_QUADRATURE_ORDER == 6
    assert module.MAXIMUM_LINEARITY_DEFECT == 1.0e-10
    assert module.MAXIMUM_INDEPENDENT_BLOCK_DEFECT == 2.0e-8
    assert module.MAXIMUM_DIRECT_ACTION_DEFECT == 5.0e-5


def test_linearity_report_detects_one_exact_linear_map() -> None:
    module = _module()
    matrix = np.asarray(
        [
            [2.0, -1.0, 0.5],
            [0.25, 3.0, -2.0],
            [1.0, 0.0, 4.0],
        ]
    )
    directions = {
        "a": np.asarray([1.0, -2.0, 3.0]),
        "b": np.asarray([-4.0, 0.5, 2.0]),
        "c": np.asarray([0.25, 1.5, -1.0]),
        "d": np.asarray([2.0, 2.0, 2.0]),
    }
    report, _arrays = module._linearity_report(matrix, directions)
    assert report["passed"] is True
    assert report["maximum_relative_defect"] <= 1.0e-15


def test_canonical_analytic_tangent_evidence_is_self_consistent() -> None:
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
    assert summary["work_package"] == "WP10c9d5c0d"
    assert summary["parent_wp10c9d5c0c_remains_rejected"] is True
    assert summary["parent_wp10c9d5_candidate_remains_rejected"] is True
    assert summary["parent_wp10c9d5b_branch_d_preserved"] is True
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert (
        summary["cross_grid_analytic_tangent_work_authorized"]
        == summary["passed"]
    )
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
