from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_absolute_baseline_observable_memory_screen_wp10c9d6c7c3b5c4f1 as c4f1


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_is_analysis_only_and_q3_staged():
    contract = c4f1._contract()
    assert not contract["new_nonlinear_trajectory"]
    assert not contract["fixed_Q_dynamics"]
    assert contract["slow_coordinate"]["mapped_storage_rows"] == [0, 2, 3]
    assert contract["lift_ensemble"]["per_step_projection"] is False


def test_absolute_result_stops_before_memory_propagation():
    summary = _read(c4f1.SUMMARY_PATH)
    assert not summary["passed"]
    assert not summary["physical_failure_detected"]
    assert summary["absolute_state_and_Q3_storage_passed"]
    assert not summary["observable_memory_propagation_executed"]
    assert not summary["Q3_screen_completed"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_failure_is_localized_to_coupling_and_net_drive():
    summary = _read(c4f1.SUMMARY_PATH)
    failures = set(summary["failed_instantaneous_components"])
    assert failures == {
        "interface_flux_mass",
        "interface_flux_angular_momentum",
        "interface_flux_killing_energy",
        "net_drive_mass",
        "net_drive_angular_momentum",
        "net_drive_killing_energy",
    }
    metrics = summary["absolute_baseline"][
        "instantaneous_extraction_component_metrics"
    ]
    for name in (
        "inner_flux_mass",
        "inner_flux_angular_momentum",
        "inner_flux_killing_energy",
        "cooling_angular_momentum",
        "cooling_killing_energy",
        "vertical_work_angular_momentum",
        "vertical_work_killing_energy",
    ):
        assert metrics[name]["passed"]
        assert metrics[name]["error_direction_cosine"] > 0.99
    for name in failures:
        assert not metrics[name]["passed"]
        assert metrics[name]["RMS_order"] > 1.5
        assert metrics[name]["error_direction_cosine"] < -0.8


def test_canonical_arrays_contain_no_memory_propagator():
    with np.load(c4f1.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert "coarse__absolute_q3" in arrays.files
        assert "middle__absolute_q3" in arrays.files
        assert "fine__absolute_q3" in arrays.files
        assert not any(name.startswith("middle__memory_") for name in arrays.files)
        assert not any(name.startswith("fine__memory_") for name in arrays.files)


def test_canonical_hashes_close():
    for line in (c4f1.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(c4f1.CANONICAL_DIRECTORY / name) == expected
