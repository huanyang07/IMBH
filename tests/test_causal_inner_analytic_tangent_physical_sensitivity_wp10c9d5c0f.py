from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_causal_inner_analytic_tangent_physical_sensitivity_"
    "wp10c9d5c0f.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_analytic_tangent_physical_sensitivity_wp10c9d5c0f"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location("wp10c9d5c0f_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def test_wp10c9d5c0f_contract_is_predeclared() -> None:
    assert RUNNER.ANALYZED_BASE_COMMIT == (
        "e5fd93352aea3dc920e528bb566b60fa7a3c8b0c"
    )
    assert RUNNER.METHODS == (
        "historical_finite_difference",
        "analytic_frozen_subspace",
    )
    assert RUNNER.PERTURBATIONS == (
        "common_mode",
        "heldout_near_excision",
    )
    assert RUNNER.MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE == 5.0e-3
    assert RUNNER.MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO == 0.10
    assert RUNNER.OUTPUT_REFERENCE_ORDER == 6


def test_wp10c9d5c0f_heldout_initial_maps_to_native_direction() -> None:
    payload, arrays = RUNNER.wp10c9d5c0e._load_replay_inputs()
    configurations = RUNNER.wp10c9d5c0e._configurations(payload, arrays)
    for configuration in configurations.values():
        initial = RUNNER._perturbation_initial(
            configuration,
            "heldout_near_excision",
        )
        columns = np.asarray(
            configuration["candidate_native"]["primitive_column_scales"],
            dtype=float,
        )
        amplitudes = np.asarray(
            configuration["amplitudes"],
            dtype=float,
        ).ravel()
        recovered = initial.ravel() * amplitudes / columns
        assert np.allclose(
            recovered,
            configuration["directions"]["near_excision_0"],
            rtol=2.0e-14,
            atol=0.0,
        )


def test_wp10c9d5c0f_canonical_evidence_is_self_consistent() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d5c0f"
    assert summary["passed"]
    assert summary["derivative_choice_physical_sensitivity_passed"]
    assert summary["wp10c9d5c1_extended_localization_authorized"]
    assert summary["parent_wp10c9d5_candidate_remains_rejected"]
    assert not summary["self_consistent_tangent_authorized"]
    assert not summary["nonlinear_candidate_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
    for report in summary["perturbations"].values():
        assert report["passed"]
        assert (
            report["maximum_derivative_export_difference"]
            <= RUNNER.MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
        )
        assert (
            report["derivative_to_spatial_ratio"]
            <= RUNNER.MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
        )
        assert (
            report["maximum_restart_defect"]
            <= RUNNER.MAXIMUM_RESTART_DEFECT
        )
