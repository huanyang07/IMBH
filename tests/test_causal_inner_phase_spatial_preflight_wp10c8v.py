from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v


def test_local_and_active_cell_counts_are_nested() -> None:
    assert tuple(
        wp10c8v._local_cell_count(mesh)
        for mesh in wp10c8v.GLOBAL_EQUIVALENT_MESHES
    ) == (24, 48, 96)
    assert tuple(
        wp10c8v._active_cell_count(mesh)
        for mesh in wp10c8v.GLOBAL_EQUIVALENT_MESHES
    ) == (16, 32, 64)


def test_shape_preserving_resampling_reproduces_constant_columns() -> None:
    source_radius = np.geomspace(2.0, 20.0, 8)
    target_radius = np.geomspace(1.9, 21.0, 17)
    source = np.column_stack(
        (
            np.full(source_radius.size, 3.0),
            np.full(source_radius.size, -2.0),
        )
    )
    result = wp10c8v._resample_columns(
        source_radius,
        source,
        target_radius,
    )
    np.testing.assert_allclose(
        result,
        np.tile((3.0, -2.0), (target_radius.size, 1)),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_generator_similarity_rescaling_preserves_physical_action() -> None:
    generator = np.asarray(((2.0, -1.0), (0.5, 3.0)))
    source_scales = np.asarray((2.0, 5.0))
    target_scales = np.asarray((7.0, 11.0))
    rescaled = wp10c8v._similarity_rescale_generator(
        generator,
        source_scales,
        target_scales,
    )
    physical_state = np.asarray((13.0, -17.0))
    source_coordinates = physical_state / source_scales
    target_coordinates = physical_state / target_scales
    physical_rate = source_scales * (
        generator @ source_coordinates
    )
    np.testing.assert_allclose(
        target_scales * (rescaled @ target_coordinates),
        physical_rate,
    )


def test_pairwise_restriction_is_conservative() -> None:
    values = np.asarray(
        (
            (
                (1.0, 2.0),
                (3.0, 4.0),
                (5.0, 6.0),
                (7.0, 8.0),
            ),
        )
    )
    measures = np.asarray((1.0, 3.0, 2.0, 2.0))
    restricted = wp10c8v._restrict_pairwise(values, measures)
    np.testing.assert_allclose(
        restricted,
        np.asarray((((2.5, 3.5), (6.0, 7.0)),)),
    )
    np.testing.assert_allclose(
        np.sum(
            restricted[0]
            * np.asarray((4.0, 4.0))[:, None],
            axis=0,
        ),
        np.sum(values[0] * measures[:, None], axis=0),
    )


def test_zero_crossings_are_linearly_interpolated() -> None:
    crossings = wp10c8v._zero_crossings(
        np.asarray((0.0, 1.0, 2.0)),
        np.asarray((1.0, -1.0, 3.0)),
    )
    np.testing.assert_allclose(crossings, (0.5, 1.25))


def test_signal_pair_requires_resolved_frequency_and_damping() -> None:
    unresolved = {
        "zero_crossings_seconds": np.asarray((0.5,)),
        "frequency_hz": None,
        "envelope_log_slope_per_s": None,
    }
    result = wp10c8v._signal_pair_metrics(unresolved, unresolved)
    assert result["frequency_relative_defect"] is None
    assert result["damping_relative_defect"] is None


def test_committed_machine_evidence_keeps_architecture_blocked() -> None:
    if not wp10c8v.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c8v.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    assert payload["work_package"] == "WP10c8v"
    assert payload["classification"] == (
        "inner_fast_phase_spatially_unresolved_local_preflight"
    )
    assert payload["decision_gates"] == {
        "authorized_next": False,
        "buffered_boundary_history_reproduction_passed": True,
        "buffered_boundary_reproduction_passed": True,
        "spatial_phase_refinement_passed": False,
        "temporal_refinement_passed": True,
    }
    assert payload["scope"]["formal_fast_average_certified"] is False
    assert payload["scope"]["reduced_architecture_selected"] is False
