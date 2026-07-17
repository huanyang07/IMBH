from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    KerrSchildCellSourceRates,
    SchwarzschildCurvatureVerticalFrequency,
    causal_five_field_h_over_r_profile,
    causal_five_field_loading_time,
    causal_five_field_physical_step_ledger,
    causal_five_field_state_summary,
    evaluate_causal_five_field_increment_backward_euler,
    fiducial_hill_roche_nozzle_geometry,
    load_causal_five_field_adaptive_restart,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
    save_causal_five_field_adaptive_restart,
)
from imri_qpe.parameters import FiducialParams


def _context(
    n_cells: int = 4,
    *,
    stream: bool = False,
) -> CausalFiveFieldDAEContext:
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        n_cells,
        gravitational_radius,
    )
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    source = None
    if stream:
        rates = np.zeros(n_cells)
        rates[-1] = 1.0e20
        source = KerrSchildCellSourceRates(
            rest_mass=rates,
            radial_momentum_over_c=0.1 * rates,
            angular_momentum_over_c=1.0e9 * rates,
            killing_energy_over_c2=0.99 * rates,
        )
    return CausalFiveFieldDAEContext(
        grid=grid,
        vertical_frequency=SchwarzschildCurvatureVerticalFrequency(
            gravitational_radius
        ),
        outer_boundary_provider=GasRadiationHillRocheNozzleProvider(
            geometry,
            transverse_quadrature_zones=24,
        ),
        stream_sources=source,
        include_radiative_cooling=True,
    ).validated()


def test_causal_adaptive_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        CausalFiveFieldAdaptiveStepConfig(
            minimum_dt=2.0,
            maximum_dt=1.0,
        ).validated()
    with pytest.raises(ValueError):
        CausalFiveFieldAdaptiveStepConfig(
            minimum_dt=1.0,
            maximum_dt=2.0,
            shrink_factor=1.0,
        ).validated()


def test_causal_state_summary_and_loading_time_are_finite() -> None:
    context = _context(stream=True)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    summary = causal_five_field_state_summary(context, vector)
    loading_time = causal_five_field_loading_time(context, vector)

    assert summary["integrated_conserved"][0] > 0.0
    assert summary["maximum_h_over_r"] > 0.0
    assert causal_five_field_h_over_r_profile(
        context,
        vector,
    ).shape == (4,)
    assert np.all(np.isfinite(summary["inner_face_rates"]))
    assert np.all(np.isfinite(summary["outer_face_rates"]))
    assert np.isfinite(loading_time)
    assert loading_time > 0.0


def test_causal_physical_step_ledger_separates_exact_stream() -> None:
    context = _context(stream=True)
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    increment = np.zeros_like(vector)
    timestep = 2.5e-8

    ledger = causal_five_field_physical_step_ledger(
        context,
        vector,
        increment,
        timestep,
    )
    evaluation = evaluate_causal_five_field_increment_backward_euler(
        increment,
        context,
        old_vector=vector,
        timestep_seconds=timestep,
        temporal_height_scheme="path_integrated",
    )
    expected_stream = np.zeros(5)
    assert context.stream_sources is not None
    expected_stream[:4] = (
        timestep * np.sum(context.stream_sources.matrix, axis=0)
    )

    np.testing.assert_array_equal(
        ledger.prescribed_stream_source,
        expected_stream,
    )
    np.testing.assert_allclose(
        ledger.closure_defect,
        C * timestep * np.sum(evaluation.conservation_rows, axis=0),
        rtol=2.0e-14,
        atol=1.0e-5,
    )


def test_causal_adaptive_restart_round_trips_bitwise(tmp_path) -> None:
    context = _context()
    vector = pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )
    increment = np.linspace(-1.0, 1.0, vector.size) * 1.0e-12
    restart = CausalFiveFieldAdaptiveRestart(
        state_vector=vector,
        previous_physical_increment=increment,
        elapsed_time=3.0e-8,
        dt_next=2.0e-8,
        previous_dt=1.5e-8,
        accepted_steps=4,
        rejected_attempts=1,
        provenance={"case": "causal-restart-test"},
    )
    path = tmp_path / "causal_restart.npz"
    save_causal_five_field_adaptive_restart(
        path,
        context,
        restart,
    )
    restored = load_causal_five_field_adaptive_restart(
        path,
        context,
    )

    np.testing.assert_array_equal(
        restored.state_vector,
        restart.state_vector,
    )
    np.testing.assert_array_equal(
        restored.previous_physical_increment,
        restart.previous_physical_increment,
    )
    assert restored.elapsed_time == restart.elapsed_time
    assert restored.dt_next == restart.dt_next
    assert restored.previous_dt == restart.previous_dt
    assert restored.accepted_steps == restart.accepted_steps
    assert restored.rejected_attempts == restart.rejected_attempts
    assert restored.provenance == restart.provenance
