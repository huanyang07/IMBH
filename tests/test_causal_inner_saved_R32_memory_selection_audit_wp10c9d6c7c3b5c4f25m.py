from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as f25m


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_locked_and_forbids_truth(monkeypatch):
    for name, value in f25m.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25m._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert budget["allowed_new_truth_anchors"] == 0
    assert budget["allowed_saved_generator_memory_fits"] == 12


def test_heldout_frequency_ladder_is_prospective_and_interior():
    source = np.asarray((0.0, 2.0, 8.0, 32.0))
    heldout = f25m._heldout_frequencies(source)
    assert np.array_equal(heldout, np.asarray((0.0, 1.0, 4.0, 16.0)))


def test_coherent_projection_is_real_and_rank_limited():
    forcing = np.arange(24.0).reshape(4, 6)
    observation = np.arange(20.0).reshape(5, 4)
    output_basis = np.eye(5)
    input_basis = np.eye(6)
    projected_forcing, projected_observation = f25m._project_channels(
        forcing, observation, output_basis, input_basis, 2
    )
    assert np.isrealobj(projected_forcing)
    assert np.isrealobj(projected_observation)
    assert np.linalg.matrix_rank(projected_forcing) <= 2
    assert np.linalg.matrix_rank(projected_observation) <= 2


def test_error_metrics_are_zero_for_exact_approximation():
    direct = np.asarray(((1.0,),))
    reference = np.asarray((direct + 2.0, direct + 1.0), dtype=complex)
    dynamic, total, metrics = f25m._error_metrics(reference, reference, direct)
    assert np.array_equal(dynamic, np.zeros(2))
    assert np.array_equal(total, np.zeros(2))
    assert metrics["maximum_normalized_dynamic_transfer_relative_error"] == 0.0
    assert metrics["maximum_normalized_total_transfer_relative_error"] == 0.0


def test_hankel_cumulative_rank_does_not_square_the_values():
    values = np.asarray((1.0, 0.5, 0.25, 0.25))
    assert f25m._cumulative_rank(values, 0.50) == 1
    assert f25m._cumulative_rank(values, 0.75) == 2
    assert f25m._energy_rank(values, 0.70) == 1


def test_classification_is_fail_closed_and_scoped():
    assert f25m._classification({"full_order_numerical_passed": False}) == (
        f25m.NUMERICAL_FAIL_CLASSIFICATION,
        None,
    )
    assert f25m._classification({
        "full_order_numerical_passed": True,
        "selected_label": None,
    }) == (
        f25m.NO_MODEL_CLASSIFICATION,
        "definitions_only_resolved_variable_reassessment_manifest",
    )
    classification, authorized = f25m._classification({
        "full_order_numerical_passed": True,
        "selected_label": "global_balanced_r96",
        "selected_family": "global_balanced",
        "selected_memory_order": 96,
        "selected_spatial_channel_rank": None,
    })
    assert "global_balanced_order_96" in classification
    assert authorized == "definitions_only_common_resolved_subspace_cross_anchor_preflight_manifest"


def test_canonical_result_when_available():
    summary_path = f25m.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert summary["new_full_560_direction_generator_assemblies"] == 0
    assert summary["new_truth_anchors"] == 0
    assert not summary["physical_failure_detected"]
    for line in (f25m.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25m.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
