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
    "run_causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5c0e_runner",
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


def test_cross_grid_contract_is_predeclared() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "d57bcc3e63bcd778823736a795a9311592173bd9"
    )
    assert module.ANALYZED_BASE_PARENT == (
        "e492299df5668b49412f033e33df3d42e92f512e"
    )
    assert module.ANALYZED_BASE_TREE == (
        "1048352852cb195abb6a99f7b822e6c8a2cab419"
    )
    assert module.GEOMETRY_LOG_RADIUS_STEPS == (
        1.0e-5,
        2.0e-5,
        4.0e-5,
    )
    assert module.MAXIMUM_INDEPENDENT_BLOCK_DEFECT == 2.0e-8
    assert module.MAXIMUM_PRODUCTION_JVP_DEFECT == 2.0e-6
    assert module.MINIMUM_CHARACTERISTIC_SPECTRAL_GAP == 1.0e-6


def test_committed_replay_inputs_reconstruct_all_three_grids() -> None:
    module = _module()
    payload, arrays = module._load_replay_inputs()
    configurations = module._configurations(payload, arrays)
    assert tuple(configurations) == module.LABELS
    assert [
        configuration["base_primitives"].shape[0]
        for configuration in configurations.values()
    ] == [64, 112, 208]
    for configuration in configurations.values():
        dimensions = configuration["base_primitives"].size
        assert configuration["candidate_native"][
            "production_generator"
        ].shape == (dimensions, dimensions)
        assert configuration["anchor_storage_derivative"].shape == (
            dimensions,
            dimensions,
        )
        assert tuple(configuration["directions"]) == (
            module.DIRECTION_NAMES
        )


def test_canonical_cross_grid_evidence_is_self_consistent() -> None:
    module = _module()
    required = (
        CANONICAL / "config.json",
        CANONICAL / "decisive_arrays.npz",
        CANONICAL / "provenance.json",
        CANONICAL / "replay_contexts.json",
        CANONICAL / "replay_inputs.npz",
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
    assert summary["work_package"] == "WP10c9d5c0e"
    assert summary["passed"] is True
    assert summary["cross_grid_analytic_tangent_certified"] is True
    assert (
        summary["derivative_choice_physical_sensitivity_authorized"]
        is True
    )
    assert summary["wp10c9d5c1_extended_localization_authorized"] is False
    assert summary["parent_wp10c9d5_candidate_remains_rejected"] is True
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    for label in module.LABELS:
        grid = summary["grids"][label]
        assert grid["passed"] is True
        assert all(grid["method_gates"].values())
        assert grid["spectral"]["incoming_inner_characteristics"] == 0
        assert grid["spectral"]["minimum_spectral_gap_over_c"] >= (
            module.MINIMUM_CHARACTERISTIC_SPECTRAL_GAP
        )
    assert summary["decisive_arrays_sha256"] == _sha256(
        CANONICAL / "decisive_arrays.npz"
    )
    with np.load(CANONICAL / "decisive_arrays.npz") as archive:
        assert set(archive.files) == set(summary["decisive_array_hashes"])
        for name in archive.files:
            assert module._array_sha256(archive[name]) == (
                summary["decisive_array_hashes"][name]
            )
