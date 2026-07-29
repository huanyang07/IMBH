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
    "run_causal_inner_derivative_repair_wp10c9d5c0a.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_derivative_repair_wp10c9d5c0a"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5c0a_runner",
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


def test_derivative_repair_contract_is_predeclared() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "f9d21e7bd8ede7c0548c93fc0b18021c30fde7fa"
    )
    assert module.HIGH_ORDER_STEP == 2.0e-4
    assert module.COARSE_HIGH_ORDER_STEP == 4.0e-4
    assert module.DERIVATIVE_ORDERS == (4, 6)
    assert module.HELD_OUT_DIRECTION_SEEDS == (93061, 93062)
    assert module.CANCELLATION_ATTRIBUTION_DIRECTION == (
        "heldout_near_excision_0"
    )
    assert module.MAXIMUM_DIRECT_ORDER_DIFFERENCE == 2.0e-5
    assert module.MAXIMUM_DIRECT_SCALE_DIFFERENCE == 2.0e-5
    assert module.MAXIMUM_MATRIX_ACTION_DEFECT == 5.0e-5
    assert module.MAXIMUM_MATRIX_ORDER_DIFFERENCE == 2.0e-5
    assert module.MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE == 5.0e-3
    assert module.MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO == 0.10


def test_recovery_stability_uses_only_repaired_methods() -> None:
    module = _module()
    no_recovery = {
        method: {"recovery_surface_index": None}
        for method in module.METHOD_NAMES
    }
    assert module._recovery_is_stable(no_recovery)
    neighboring = {
        module.METHOD_NAMES[0]: {"recovery_surface_index": 3},
        module.METHOD_NAMES[1]: {"recovery_surface_index": 4},
    }
    assert module._recovery_is_stable(neighboring)
    disagreement = {
        module.METHOD_NAMES[0]: {"recovery_surface_index": 2},
        module.METHOD_NAMES[1]: {"recovery_surface_index": 4},
    }
    assert not module._recovery_is_stable(disagreement)


def test_canonical_derivative_repair_evidence_is_self_consistent() -> None:
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
    assert summary["work_package"] == "WP10c9d5c0a"
    assert summary["parent_wp10c9d5_candidate_remains_rejected"] is True
    assert summary["parent_wp10c9d5b_branch_d_preserved"] is True
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert (
        summary["wp10c9d5c1_extended_localization_authorized"]
        == summary["derivative_repair_passed"]
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
