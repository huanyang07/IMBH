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
    "run_causal_inner_windowed_contract_wp10c9d6c6a2.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_windowed_contract_wp10c9d6c6a2"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"
TRAPEZOID_PREFLIGHT = CANONICAL / "trapezoid_preflight_summary.json"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c6a2_runner",
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


def test_wp10c9d6c6a2_freezes_window_and_parent_gates() -> None:
    assert RUNNER.WINDOW_POWERS == (2, 4)
    assert RUNNER.BINDING_POWER == 4
    assert RUNNER.TIME_HORIZON_S == 0.125
    assert RUNNER.TIME_SAMPLE_COUNT == 65
    assert RUNNER.SPECTRAL_ENERGY_QUANTILE == 0.99
    assert RUNNER.MAXIMUM_COMPLETE_SEMIGROUP_ERROR == 0.025
    assert RUNNER.MINIMUM_USABLE_THETA == 0.20
    assert RUNNER.MAXIMUM_ALIAS_FRACTION == 1.0e-3
    assert (
        RUNNER.MAXIMUM_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
        == 0.10
    )
    assert (
        RUNNER.MAXIMUM_BOUNDARY_INTEGRAL_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
        == 0.10
    )


def test_wp10c9d6c6a2_preserves_c6a_and_c6a1_classifications() -> None:
    summary = _summary()
    assert summary["work_package"] == "WP10c9d6c6a2"
    assert summary["parent_classification_preserved"]
    assert summary["c6a_rejection_preserved"]
    assert summary["parent_packet_contract_error_preserved"]
    assert summary["parent_minimum_usable_theta_preserved"]
    assert not summary["operator_changed"]
    assert summary["classification"] == (
        "variable_coefficient_windowed_contract_certified_"
        "packet_manifest_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c6b_packet_definition_manifest_only"
    )


def test_wp10c9d6c6a2_certifies_all_frozen_probes() -> None:
    summary = _summary()
    comparison = summary["comparison_report"]
    assert summary["method_passed"]
    assert comparison["passed"]
    assert comparison["all_probes_passed"]
    assert comparison["binding_range_reaches_theta20"]
    assert comparison["minimum_binding_theta_99"] >= 0.20
    for name, report in comparison["probe_reports"].items():
        assert name in RUNNER.PROBE_DEFINITIONS
        assert report["passed"]
        assert (
            report["maximum_coarse_reference_relative_error"]
            <= RUNNER.MAXIMUM_COMPLETE_SEMIGROUP_ERROR
        )
        assert (
            report["observed_order"]
            >= RUNNER.MINIMUM_CROSS_GRID_ORDER
        )
        assert (
            report["refinement_error_cosine"]
            >= RUNNER.MINIMUM_REFINEMENT_ERROR_COSINE
        )
        assert (
            report[
                "boundary_integral_uncertainty_to_fine_difference_ratio"
            ]
            <= (
                RUNNER
                .MAXIMUM_BOUNDARY_INTEGRAL_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
            )
        )


def test_wp10c9d6c6a2_preserves_failed_trapezoid_preflight() -> None:
    summary = _summary()
    initial = summary["initial_trapezoid_preflight_report"]
    assert TRAPEZOID_PREFLIGHT.exists()
    assert initial["classification"] == (
        "variable_coefficient_windowed_contract_failed_"
        "packet_manifest_blocked"
    )
    assert initial["authorized_next"] == "none"
    correction = summary["boundary_integration_method_correction"]
    assert not correction["physical_probes_changed"]
    assert not correction["window_definitions_changed"]
    assert not correction["tangents_changed"]
    assert not correction["scientific_gates_changed"]
    assert any(
        report["diagnostic_trapezoid_to_fine_difference_ratio"]
        > 0.10
        for report in summary["comparison_report"][
            "probe_reports"
        ].values()
    )


def test_wp10c9d6c6a2_authorizes_manifest_only() -> None:
    summary = _summary()
    assert summary["prospective_packet_manifest_authorized"]
    assert not summary["uniform_packet_propagation_authorized"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6a2_canonical_arrays_and_hashes() -> None:
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
