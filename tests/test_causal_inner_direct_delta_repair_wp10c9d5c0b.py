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
    "run_causal_inner_direct_delta_repair_wp10c9d5c0b.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_delta_repair_wp10c9d5c0b"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5c0b_runner",
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


def test_direct_delta_repair_contract_is_predeclared() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "5c88fa02f8f25fa62e9e0fdb648e66974bca38d3"
    )
    assert module.ANALYZED_BASE_PARENT == (
        "f9d21e7bd8ede7c0548c93fc0b18021c30fde7fa"
    )
    assert module.ANALYZED_BASE_TREE == (
        "ddcc241bc0a6bac03ba00c42fbcbb5b2056b3787"
    )
    assert module.HIGH_ORDER_STEP == 2.0e-4
    assert module.DERIVATIVE_ORDERS == (4, 6)
    assert module.MAXIMUM_MATRIX_ACTION_DEFECT == 5.0e-5
    assert module.MAXIMUM_MATRIX_ORDER_DIFFERENCE == 2.0e-5
    assert module.MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE == 5.0e-3
    assert module.MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO == 0.10
    assert module.FAIL_FAST_ON_FIRST_FAILED_GRID is True


def test_stationary_delta_is_formed_before_differentiation(
    monkeypatch,
) -> None:
    module = _module()
    calls = []

    def direct_delta(configuration, increment):
        calls.append((configuration, np.asarray(increment, dtype=float)))
        return np.asarray(increment, dtype=float) * 3.0

    monkeypatch.setattr(
        module.wp10c9d5c0,
        "_scaled_delta",
        direct_delta,
    )
    configuration = {"label": "synthetic"}
    increment = np.asarray([1.0, -2.0, 4.0])
    blocks = module._stationary_delta_blocks(configuration, increment)
    assert set(blocks) == {"stationary_delta"}
    np.testing.assert_array_equal(
        blocks["stationary_delta"],
        3.0 * increment,
    )
    assert calls[0][0] is configuration
    np.testing.assert_array_equal(calls[0][1], increment)


def test_canonical_direct_delta_repair_evidence_is_self_consistent() -> None:
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
    assert summary["work_package"] == "WP10c9d5c0b"
    assert summary["parent_wp10c9d5c0a_remains_rejected"] is True
    assert summary["parent_wp10c9d5_candidate_remains_rejected"] is True
    assert summary["parent_wp10c9d5b_branch_d_preserved"] is True
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert (
        summary["wp10c9d5c1_extended_localization_authorized"]
        == summary["direct_delta_repair_passed"]
    )
    if summary["fail_fast_trigger"] is not None:
        assert summary["direct_delta_matrix_passed"] is False
        assert summary["physical_sensitivity"]["executed"] is False
        assert summary["unattempted_labels"]
        assert summary["linearity_equivalence"]["executed"] is True
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
