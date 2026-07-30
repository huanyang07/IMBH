from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_causal_inner_full_symbol_limiter_wp10c9d6c6a1.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_full_symbol_limiter_wp10c9d6c6a1"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c6a1_runner",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
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


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_wp10c9d6c6a1_freezes_limiter_and_ray_preflight_gates() -> None:
    assert RUNNER.AUDIT_RADII_OVER_RG == (2.2, 3.0, 5.0, 8.0, 11.0)
    assert RUNNER.AUDIT_THETA_VALUES == (0.10, 0.17, 0.18, 0.20, 0.30)
    assert RUNNER.AUDIT_TIMES_S[-1] == 0.125
    assert RUNNER.PACKET_CONTRACT_ERROR == 0.025
    assert RUNNER.MINIMUM_USABLE_THETA == 0.20
    assert RUNNER.MAXIMUM_RAY_PREFLIGHT_ERROR == 0.05
    assert RUNNER.MINIMUM_RAY_BRANCH_OVERLAP == 0.90
    assert RUNNER.MINIMUM_SIGNIFICANT_COMPONENT_ORDER == 1.25
    assert RUNNER.SIGNIFICANT_PROPAGATOR_CONTRIBUTION == 2.5e-4


def test_wp10c9d6c6a1_preserves_parent_and_downstream_stops() -> None:
    summary = _summary()
    assert summary["work_package"] == "WP10c9d6c6a1"
    assert summary["parent_classification_preserved"]
    assert summary["parent_classification"] == (
        "symbol_derived_packet_resolution_contract_failed"
    )
    assert summary["parent_packet_contract_error_preserved"]
    assert summary["parent_minimum_usable_theta_preserved"]
    assert not summary["operator_changed"]
    assert not summary["packet_resolution_contract_certified"]
    assert not summary["prospective_packet_manifest_authorized"]
    assert not summary["uniform_packet_propagation_authorized"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6a1_method_and_decision_follow_frozen_rules() -> None:
    summary = _summary()
    method = summary["method_report"]
    assert (
        method["maximum_component_closure_defect"]
        <= RUNNER.MAXIMUM_COMPONENT_CLOSURE_DEFECT
    )
    assert (
        method["maximum_generator_parity_defect"]
        <= RUNNER.MAXIMUM_GENERATOR_PARITY_DEFECT
    )
    assert (
        method["maximum_shapley_closure_defect"]
        <= RUNNER.MAXIMUM_SHAPLEY_CLOSURE_DEFECT
    )
    ray = summary["ray_preflight_report"]
    if not method["passed"] or not ray["method_passed"]:
        expected = "none"
    elif ray["maximum_theta20_error"] <= RUNNER.MAXIMUM_RAY_PREFLIGHT_ERROR:
        expected = (
            "WP10c9d6c6a2_variable_coefficient_windowed_contract"
        )
    else:
        expected = "none"
    assert summary["authorized_next"] == expected
    assert summary["passed"] == (expected != "none")
    assert summary["classification"] == (
        "full_symbol_limiter_convergent_accumulation_"
        "windowed_contract_audit_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c6a2_variable_coefficient_windowed_contract"
    )


def test_wp10c9d6c6a1_preserves_failed_ray_methods_and_tracks_branches() -> None:
    summary = _summary()
    assert not summary[
        "initial_midpoint_ray_preflight_report"
    ]["method_passed"]
    assert not summary[
        "speed_sorted_rk4_ray_preflight_report"
    ]["method_passed"]
    correction = summary["ray_method_correction"]
    assert correction["branch_association_changed"]
    assert not correction["physical_rays_changed"]
    assert not correction["step_sizes_changed"]
    assert not correction["scientific_gates_changed"]
    ray = summary["ray_preflight_report"]
    assert ray["method_passed"]
    assert (
        ray["minimum_branch_overlap"]
        >= RUNNER.MINIMUM_RAY_BRANCH_OVERLAP
    )
    assert (
        ray["maximum_integrator_to_error_ratio"]
        <= RUNNER.MAXIMUM_RAY_INTEGRATOR_TO_ERROR_RATIO
    )
    assert (
        ray["maximum_continuum_reference_to_error_ratio"]
        <= RUNNER.MAXIMUM_RAY_REFERENCE_TO_ERROR_RATIO
    )


def test_wp10c9d6c6a1_shapley_and_cross_grid_contracts_close() -> None:
    summary = _summary()
    assert (
        summary["method_report"]["maximum_shapley_closure_defect"]
        <= RUNNER.MAXIMUM_SHAPLEY_CLOSURE_DEFECT
    )
    for report in summary["local_attribution_reports"].values():
        assert not report["touches_boundary"]
        assert (
            report["maximum_component_closure_defect"]
            <= RUNNER.MAXIMUM_COMPONENT_CLOSURE_DEFECT
        )
        assert (
            report["maximum_generator_parity_defect"]
            <= RUNNER.MAXIMUM_GENERATOR_PARITY_DEFECT
        )
        assert (
            report["maximum_continuum_generator_parity_defect"]
            <= RUNNER.MAXIMUM_GENERATOR_PARITY_DEFECT
        )


def test_wp10c9d6c6a1_canonical_arrays_and_hashes() -> None:
    summary = _summary()
    assert SUMMARY.exists()
    assert DECISIVE.exists()
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
