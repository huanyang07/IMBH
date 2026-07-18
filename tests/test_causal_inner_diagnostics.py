from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    KerrSchildCellSourceRates,
    SchwarzschildCurvatureVerticalFrequency,
    audit_causal_five_field_state_gates,
    causal_backward_euler_step_doubling_factor,
    causal_five_field_cell_states,
    causal_five_field_local_timescale_audit,
    causal_five_field_observable_snapshot,
    causal_five_field_temporal_error_ratio,
    compare_causal_five_field_observables,
    fiducial_hill_roche_nozzle_geometry,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams


def _context(*, stream: bool = True) -> CausalFiveFieldDAEContext:
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        4,
        gravitational_radius,
    )
    source = None
    if stream:
        rates = np.zeros(4)
        rates[-1] = 1.0e20
        source = KerrSchildCellSourceRates(
            rest_mass=rates,
            radial_momentum_over_c=0.1 * rates,
            angular_momentum_over_c=1.0e9 * rates,
            killing_energy_over_c2=0.99 * rates,
        )
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
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


def _vector(context: CausalFiveFieldDAEContext) -> np.ndarray:
    return pack_causal_five_field_state(
        make_causal_five_field_seed(context)
    )


def test_causal_cell_states_recover_every_cell() -> None:
    context = _context()
    cells = causal_five_field_cell_states(context, _vector(context))

    assert len(cells) == 4
    assert all(cell.primitive.surface_density > 0.0 for cell in cells)
    assert all(cell.thermodynamics.temperature > 0.0 for cell in cells)


def test_causal_observables_are_versioned_and_compare_identically() -> None:
    context = _context()
    vector = _vector(context)
    snapshot = causal_five_field_observable_snapshot(
        context,
        vector,
        cooling_inner_cutoff=6.0 * context.grid.gravitational_radius,
    )
    errors = compare_causal_five_field_observables(snapshot, snapshot)

    assert (
        snapshot.schema_version
        == CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
    )
    assert snapshot.cooling_power_proxy_erg_s > 0.0
    assert snapshot.cooling_power_proxy_outside_cutoff_erg_s > 0.0
    assert snapshot.inner_accretion_rate_g_s > 0.0
    assert errors["maximum_log_h_over_r_profile"] == 0.0
    assert errors["maximum_integrated_conserved_relative"] == 0.0


def test_causal_local_timescales_are_positive_and_source_aware() -> None:
    context = _context()
    clocks = causal_five_field_local_timescale_audit(
        context,
        _vector(context),
    )

    assert np.all(clocks.characteristic_crossing_seconds > 0.0)
    assert np.all(clocks.stress_relaxation_seconds > 0.0)
    assert np.all(clocks.thermal_response_seconds > 0.0)
    assert np.all(clocks.luminosity_response_seconds > 0.0)
    assert np.all(clocks.radial_advection_seconds > 0.0)
    assert np.count_nonzero(np.isfinite(clocks.local_loading_seconds)) == 1
    assert clocks.global_loading_seconds > 0.0
    assert np.array_equal(
        clocks.cooling_log_temperature_derivative,
        np.full(4, 4.0),
    )


def test_causal_state_gate_audit_preserves_declared_contract() -> None:
    context = _context()
    audit = audit_causal_five_field_state_gates(
        context,
        _vector(context),
    )

    assert audit["schema_version"] == "causal-five-field-state-gates-v1"
    assert audit["gates"]["outer_incoming_characteristics"] == 2
    assert audit["gates"]["outer_boundary_choked"] is False
    assert audit["measured"]["maximum_h_over_r"] > 0.0
    assert audit["passed"]


def test_causal_temporal_error_ratio_and_controller_factor() -> None:
    audit = causal_five_field_temporal_error_ratio(
        {"cooling": 2.0e-4, "thickness": 3.0e-3},
        {"cooling": 1.0e-3, "thickness": 2.0e-3},
    )

    assert audit["maximum_normalized_error"] == 1.5
    assert audit["controlling_observables"] == ["thickness"]
    assert audit["violated_observables"] == ["thickness"]
    assert not audit["passed"]
    assert causal_backward_euler_step_doubling_factor(0.0) == 2.0
    assert np.isclose(
        causal_backward_euler_step_doubling_factor(0.25),
        1.6,
    )
    assert (
        causal_backward_euler_step_doubling_factor(1.0e4)
        == 0.25
    )
