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
    "run_causal_inner_cross_grid_hardening_wp10c9d5c0.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_hardening_wp10c9d5c0"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5c0_runner",
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


def test_cross_grid_hardening_contract_is_predeclared() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "9c2a4ac6fa464a43fbaed3318cf5e1233a70fe55"
    )
    assert module.TARGET_RADII_OVER_RG == (5.0, 8.0, 12.0)
    assert module.DIRECTIONAL_STEPS == (
        5.0e-6,
        1.0e-5,
        2.0e-5,
        4.0e-5,
        8.0e-5,
    )
    assert module.DIRECTION_SEEDS == (92051, 92052, 92053, 92054)
    assert module.MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE == 5.0e-3
    assert module.MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO == 0.10
    assert module.MINIMUM_HISTORY_COSINE == 0.90
    assert module.MINIMUM_ERROR_COSINE == 0.90


def test_extrapolated_directional_report_uses_independent_step_windows() -> None:
    module = _module()
    assert module.FINE_EXTRAPOLATION_STEP == 1.0e-5
    assert module.COARSE_EXTRAPOLATION_STEP == 2.0e-5
    assert module.STORED_MATRIX_STEP == 4.0e-5
    values = np.asarray([1.0, -2.0])
    perturbed = values + np.asarray([1.0e-6, -2.0e-6])
    defect = module._relative_difference(values, perturbed)
    assert 0.0 < defect < 2.0e-6


def test_canonical_cross_grid_hardening_evidence_is_self_consistent() -> None:
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
    assert summary["work_package"] == "WP10c9d5c0"
    assert summary["parent_wp10c9d5_candidate_remains_rejected"] is True
    assert summary["parent_wp10c9d5b_branch_d_preserved"] is True
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
