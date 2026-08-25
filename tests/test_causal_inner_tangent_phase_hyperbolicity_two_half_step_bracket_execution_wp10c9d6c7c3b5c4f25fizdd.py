from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

import run_causal_inner_tangent_phase_hyperbolicity_two_half_step_bracket_execution_wp10c9d6c7c3b5c4f25fizdd as target


def test_execution_contract_is_exactly_the_two_half_step_scope() -> None:
    contract = target._static_execution_contract()
    assert contract["half_step_seconds"] == 1.25e-4
    assert contract["maximum_attempts"] == 2
    assert contract["maximum_accepted_steps"] == 2
    assert contract["all_face_hyperbolicity_preflight"]
    assert contract["preflight_before_each_exact_field"]
    assert contract["real_characteristic_basis_required"]
    assert not contract["failed_full_step_propagated"]
    assert contract["rejected_half_step_never_propagates"]


def test_initial_progress_keeps_full_step_history_but_proposes_half_step() -> None:
    progress = target._initial_progress()
    assert progress["previous_span"] == 2.5e-4
    assert progress["next_span"] == 1.25e-4
    assert progress["accepted_segments_total"] == 267
    assert progress["accepted_segments_new"] == 0
    assert progress["attempts"] == 0


def test_face_hyperbolicity_classifies_real_and_complex(monkeypatch) -> None:
    class Grid:
        edges = np.asarray([1.0, 2.0])

    class Context:
        grid = Grid()

    monkeypatch.setattr(
        target.boundary_diagnostic,
        "_face_charts",
        lambda _context, _state: np.asarray([[1.0], [2.0]]),
    )
    values = iter((0.0, 2.0e-8))

    def pencil(_context, _radius, _chart):
        imaginary = next(values)
        return {
            "values": np.asarray([1.0 + 1j * imaginary]),
            "vectors": np.asarray([[1.0 + 0j]]),
            "column_scales": np.ones(1),
            "maximum_imaginary_speed": imaginary,
            "maximum_eigenpair_defect": 1.0e-16,
        }

    monkeypatch.setattr(target.boundary_diagnostic, "_analytic_pencil", pencil)
    metrics, arrays = target._face_hyperbolicity(Context(), np.ones((1, 1)))
    assert not metrics["passed"]
    assert metrics["first_complex_face"] == 1
    assert metrics["maximum_imaginary_face"] == 1
    assert arrays["face_eigenvalues"].shape == (2, 1)


def test_guard_stops_before_original_field_on_complex_state(
    tmp_path: Path, monkeypatch
) -> None:
    metrics = {
        "faces_checked": 113,
        "passed": False,
        "maximum_imaginary_coordinate_speed": 2.0e-8,
        "maximum_imaginary_primitive_eigenvector_component": 0.0,
        "maximum_complex_characteristic_component": 2.0e-8,
        "maximum_imaginary_face": 3,
        "first_complex_face": 3,
        "maximum_eigenpair_defect": 1.0e-16,
        "real_spectrum_gate": 1.0e-10,
        "confirmed_complex_threshold": 1.0e-8,
    }
    arrays = {
        "face_charts": np.ones((113, 5)),
        "face_eigenvalues": np.ones((113, 5), dtype=complex),
        "face_maximum_imaginary_speeds": np.zeros(113),
        "face_maximum_imaginary_primitive_eigenvector_components": np.zeros(113),
        "face_eigenpair_defects": np.zeros(113),
    }
    monkeypatch.setattr(
        target, "_face_hyperbolicity", lambda _context, _state: (metrics, arrays)
    )
    monkeypatch.setattr(
        target,
        "_ORIGINAL_METRIC_FIELD",
        lambda **_kwargs: pytest.fail("exact field must not run"),
    )
    inputs = {"base": {"configuration": {"context": object()}}}
    with pytest.raises(target.HyperbolicityBoundary):
        target._guarded_metric_field(
            directory=tmp_path,
            stem="endpoint_field",
            inputs=inputs,
            exact_chart=None,
            state=np.ones((2, 2)),
            coordinate=np.ones(3),
            retraction={},
            anchor_chart=None,
        )


def test_outcome_requires_two_accepted_steps_for_continuation() -> None:
    accepted = {
        "accepted": True,
        "physical_failure": False,
        "stop_reason": None,
        "phase_geometry": {"passed": True},
    }
    boundary = {
        "accepted": False,
        "physical_failure": False,
        "stop_reason": "hyperbolicity_boundary",
        "phase_geometry": None,
        "hyperbolicity_preflight": {
            "maximum_complex_characteristic_component": 2.0e-8
        },
    }
    success = target._classification_from_records(
        [(accepted, {}), (accepted, {})]
    )
    first = target._classification_from_records([(boundary, {})])
    second = target._classification_from_records(
        [(accepted, {}), (boundary, {})]
    )
    assert success["trajectory_continuation_passed"]
    assert success["authorized_next"] == target.AUTHORIZED_NEXT
    assert first["classification"] == target.FIRST_BOUNDARY_CLASSIFICATION
    assert second["classification"] == target.SECOND_BOUNDARY_CLASSIFICATION
    assert first["authorized_next"] is None
    assert second["authorized_next"] is None


def test_manifest_validation_preserves_no_complex_coercion() -> None:
    validated = target._validate_manifest(require_clean=False)
    hyperbolicity = validated["contract"]["binding_hyperbolicity_gate"]
    assert hyperbolicity["no_complex_flux_split"]
    assert hyperbolicity["no_real_part_coercion"]


def test_incomplete_current_attempt_is_not_prior_accepted_history(
    tmp_path: Path, monkeypatch
) -> None:
    attempt = tmp_path / "attempt_0000"
    attempt.mkdir()
    metrics = {
        "accepted": True,
        "phase_geometry": None,
        "recurrence_geometry": None,
    }
    (attempt / "attempt.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (attempt / "attempt.npz").open("wb") as handle:
        np.savez(handle, accepted_coordinate_rate470_per_s=np.ones(470))
    monkeypatch.setattr(target, "SCRATCH_DIRECTORY", tmp_path)
    assert target._accepted_attempts() == []
    metrics["phase_geometry"] = {"passed": True}
    metrics["recurrence_geometry"] = {"passed": True}
    (attempt / "attempt.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert len(target._accepted_attempts()) == 1


def test_v1_partial_step1_field_is_valid_but_not_propagated() -> None:
    snapshot = target._v1_partial_snapshot()
    assert snapshot["strict_retraction_passed"]
    assert snapshot["all_face_hyperbolicity_passed"]
    assert snapshot["exact_field_physical_passed"]
    assert not snapshot["phase_geometry_completed"]
    assert not snapshot["accepted_checkpoint_written"]
    assert not snapshot["candidate_propagated"]


def test_execution_context_installs_and_restores_guarded_adapters() -> None:
    original_attempt = target.engine._attempt
    original_field = target.suffix._metric_field
    original_replay = target.engine._restart_replay
    with target._execution_context():
        assert target.engine._attempt is target._guarded_phase_attempt
        assert target.suffix._metric_field is target._guarded_metric_field
        assert target.engine._restart_replay is target._short_restart_replay
        assert target.engine.manifest is target.sys.modules[target.__name__]
    assert target.engine._attempt is original_attempt
    assert target.suffix._metric_field is original_field
    assert target.engine._restart_replay is original_replay
