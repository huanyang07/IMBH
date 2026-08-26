from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

import run_causal_inner_entropy_complete_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizek as target


def test_crossing_manifest_is_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["bounded_crossing_execution_authorized"]
    assert validated["summary"]["maximum_new_trajectory_steps"] == 4


def test_checkpoint_roundtrip_is_bitwise() -> None:
    values = np.arange(56, dtype=float).reshape(8, 7)
    assert np.array_equal(values, target._roundtrip(values))


def test_step_metrics_fail_closed_on_physical_violation() -> None:
    operator = SimpleNamespace(
        maximum_imaginary_speed_over_c=0.0,
        maximum_light_cone_excess_over_c=0.0,
        maximum_eigenvector_condition_number=10.0,
        maximum_CFL_for_timestep=0.1,
        minimum_height_over_radius=1.0e-5,
        maximum_height_over_radius=0.1,
        minimum_optical_depth=10.0,
        temporal_solve_relative_residuals=np.asarray([1.0e-15]),
        incoming_inner_characteristics=0,
        incoming_outer_characteristics=1,
    )
    charts = np.zeros((2, 7), dtype=float)
    step = SimpleNamespace(
        initial_operator=operator,
        euler_operator=operator,
        accepted_operator=operator,
        maximum_scaled_chart_change=1.0e-3,
        exact_flux_balance_relative_defect=1.0e-8,
        accepted_charts=charts,
    )
    metrics = target._step_metrics(step, target.parent._contract()["binding_gates"])
    assert not metrics["passed"]
    assert "physical:height_min" in metrics["failure_reasons"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1); assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["accepted_new_steps"] <= 4
    assert not summary["fixed_Q_invariant_object_execution_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
