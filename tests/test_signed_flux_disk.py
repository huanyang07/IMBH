from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    StreamInjectionState,
    advance_signed_flux_explicit,
    advance_signed_flux_implicit,
    make_log_grid,
    normalized_stream_cell_rates,
    normalized_stream_injection_state,
    signed_flux_transport,
    solve_signed_flux_steady,
)
from imri_qpe.units import solar_masses_to_g


def _ring(n: int = 128):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(6.1 * potential.r_g, 300.0 * potential.r_g, n)
    sigma = 1.0 + 100.0 * np.exp(
        -0.5 * (np.log(grid.centers / (80.0 * potential.r_g)) / 0.15) ** 2
    )
    return mass, grid, sigma


def test_signed_ring_has_finite_density_accretion_and_decretion() -> None:
    mass, grid, sigma = _ring()
    transport = signed_flux_transport(
        grid,
        sigma,
        1.0e14,
        mass,
        boundary=SignedFluxBoundary(outer_mode="tidal_wall"),
    )

    assert np.any(transport.mdot_faces[1:-1] > 0.0)
    assert np.any(transport.mdot_faces[1:-1] < 0.0)
    crossings = np.flatnonzero(
        transport.mdot_faces[1:-2] * transport.mdot_faces[2:-1] < 0.0
    )
    assert crossings.size > 0
    crossing = int(crossings[np.argmin(np.abs(crossings - 80))])
    assert sigma[crossing] > 0.0
    assert sigma[crossing + 1] > 0.0


def test_signed_flux_global_mass_and_angular_ledgers_close() -> None:
    mass, grid, sigma = _ring()
    transport = signed_flux_transport(grid, sigma, 1.0e14, mass)

    assert np.sum(transport.mass_rate_cells) == pytest.approx(
        transport.mass_budget_rate, rel=2.0e-15
    )
    angular_scale = max(
        abs(transport.angular_momentum_budget_rate),
        abs(transport.angular_momentum_rate_from_state),
    )
    assert abs(transport.angular_momentum_budget_defect) / angular_scale < 1.0e-12


def test_absolute_stream_source_is_exactly_normalized() -> None:
    mass, grid, sigma = _ring()
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 3.0e22
    stream_l = float(potential.l_k(100.0 * potential.r_g))
    source_mass, source_angular = normalized_stream_cell_rates(
        grid,
        stream_rate,
        center=100.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
    )
    transport = signed_flux_transport(
        grid,
        sigma,
        0.0,
        mass,
        source_mass_rate_cells=source_mass,
        source_specific_angular_momentum=np.full_like(sigma, stream_l),
    )

    assert np.sum(source_mass) == pytest.approx(stream_rate, rel=2.0e-15)
    assert np.sum(source_angular) == pytest.approx(stream_rate * stream_l, rel=2.0e-15)
    assert transport.mass_budget_rate == pytest.approx(stream_rate, rel=2.0e-15)
    assert np.sum(transport.source_angular_rate_cells) == pytest.approx(
        stream_rate * stream_l, rel=2.0e-15
    )


def test_unified_stream_state_normalizes_all_moments_and_is_immutable() -> None:
    mass, grid, _sigma = _ring()
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 3.0e22
    stream_l = float(potential.l_k(100.0 * potential.r_g))
    stream_B = -2.0e18
    source = normalized_stream_injection_state(
        grid,
        stream_rate,
        center=100.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
        specific_total_energy=stream_B,
    )

    assert isinstance(source, StreamInjectionState)
    assert np.sum(source.mass_rate_cells) == pytest.approx(stream_rate, rel=2.0e-15)
    assert np.sum(source.angular_momentum_rate_cells) == pytest.approx(
        stream_rate * stream_l, rel=2.0e-15
    )
    assert np.sum(source.total_energy_rate_cells) == pytest.approx(
        stream_rate * stream_B, rel=2.0e-15
    )
    with pytest.raises(ValueError):
        source.mass_rate_cells[0] = 0.0


def test_explicit_step_preserves_budget_and_rejects_negative_mass() -> None:
    mass, grid, sigma = _ring()
    transport = signed_flux_transport(grid, sigma, 1.0e14, mass)
    annular_mass = sigma * grid.area
    draining = transport.mass_rate_cells < 0.0
    dt_limit = float(np.min(annular_mass[draining] / -transport.mass_rate_cells[draining]))
    dt = 0.01 * dt_limit
    result = advance_signed_flux_explicit(grid, sigma, 1.0e14, mass, dt)

    before = float(np.sum(annular_mass))
    after = float(np.sum(result.surface_density * grid.area))
    assert after - before == pytest.approx(dt * transport.mass_budget_rate, rel=5.0e-9)
    assert np.all(result.surface_density > 0.0)

    with pytest.raises(ValueError, match="non-positive annular mass"):
        advance_signed_flux_explicit(grid, sigma, 1.0e14, mass, 1.01 * dt_limit)


def test_implicit_step_crosses_explicit_limit_and_satisfies_backward_euler() -> None:
    mass, grid, sigma = _ring(256)
    transport = signed_flux_transport(grid, sigma, 1.0e14, mass)
    annular_mass = sigma * grid.area
    draining = transport.mass_rate_cells < 0.0
    explicit_limit = float(
        np.min(annular_mass[draining] / -transport.mass_rate_cells[draining])
    )
    dt = 10.0 * explicit_limit
    result = advance_signed_flux_implicit(grid, sigma, 1.0e14, mass, dt)

    assert np.all(result.surface_density > 0.0)
    backward_euler = (
        (result.surface_density - sigma) * grid.area / dt
        - result.transport.mass_rate_cells
    )
    scale = max(float(np.max(np.abs(result.transport.mass_rate_cells))), 1.0)
    assert np.max(np.abs(backward_euler)) / scale < 2.0e-11


def test_time_step_rejects_unclosed_nonlocal_stream_angular_momentum() -> None:
    mass, grid, sigma = _ring(64)
    potential = PaczynskiWiitaPotential(mass)
    source_mass, _ = normalized_stream_cell_rates(
        grid,
        1.0e20,
        center=100.0 * potential.r_g,
        log_width=0.08,
    )
    nonlocal_l = np.full(grid.centers.size, potential.l_k(150.0 * potential.r_g))

    with pytest.raises(ValueError, match="coupled angular IMEX operator"):
        advance_signed_flux_implicit(
            grid,
            sigma,
            1.0e14,
            mass,
            1.0,
            source_mass_rate_cells=source_mass,
            source_specific_angular_momentum=nonlocal_l,
        )


@pytest.mark.parametrize("outer_mode", ["tidal_wall", "zero_torque"])
def test_absolute_stream_supply_sets_emergent_boundary_flux(outer_mode: str) -> None:
    mass, grid, _sigma = _ring(192)
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0e22
    source_mass, _source_angular = normalized_stream_cell_rates(
        grid,
        stream_rate,
        center=100.0 * potential.r_g,
        log_width=0.08,
    )
    result = solve_signed_flux_steady(
        grid,
        1.0e14,
        mass,
        boundary=SignedFluxBoundary(outer_mode=outer_mode),
        source_mass_rate_cells=source_mass,
    )

    assert np.all(result.surface_density > 0.0)
    assert np.max(np.abs(result.mass_rate_cells)) / stream_rate < 2.0e-12
    assert result.mdot_faces[0] == pytest.approx(
        stream_rate + result.mdot_faces[-1], rel=2.0e-12
    )
    if outer_mode == "tidal_wall":
        assert result.mdot_faces[-1] == pytest.approx(0.0, abs=1.0e-20 * stream_rate)
        assert result.mdot_faces[0] == pytest.approx(stream_rate, rel=2.0e-12)
    else:
        assert result.mdot_faces[-1] < 0.0
        assert 0.0 < result.mdot_faces[0] < stream_rate


def test_physical_stream_angular_momentum_sets_open_split_and_wall_torque() -> None:
    mass, grid, _sigma = _ring(256)
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 5.0e22
    stream_l = float(potential.l_k(100.0 * potential.r_g))
    source = normalized_stream_injection_state(
        grid,
        stream_rate,
        center=92.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
        specific_total_energy=0.0,
    )
    edge_l = np.asarray(potential.l_k(grid.edges), dtype=float)
    expected_open = (edge_l[-1] - stream_l) / (edge_l[-1] - edge_l[0])
    expected_wall_torque = (stream_l - edge_l[0]) / stream_l

    opened = solve_signed_flux_steady(
        grid,
        1.0e14,
        mass,
        boundary=SignedFluxBoundary(outer_mode="zero_torque"),
        stream_state=source,
    )
    wall = solve_signed_flux_steady(
        grid,
        1.0e14,
        mass,
        boundary=SignedFluxBoundary(outer_mode="tidal_wall"),
        stream_state=source,
    )

    angular_scale = stream_rate * stream_l
    assert opened.mdot_faces[0] / stream_rate == pytest.approx(
        expected_open, rel=2.0e-14
    )
    assert opened.viscous_torque_faces[-1] == pytest.approx(0.0, abs=1.0e-12 * angular_scale)
    assert wall.viscous_torque_faces[-1] / angular_scale == pytest.approx(
        expected_wall_torque, rel=2.0e-14
    )
    assert abs(opened.angular_momentum_budget_defect) < 1.0e-12 * angular_scale
    assert abs(wall.angular_momentum_budget_defect) < 1.0e-12 * angular_scale


def test_named_external_torque_changes_open_split_without_unmodeled_defect() -> None:
    mass, grid, _sigma = _ring(128)
    potential = PaczynskiWiitaPotential(mass)
    stream_rate = 2.0e22
    stream_l = float(potential.l_k(100.0 * potential.r_g))
    source = normalized_stream_injection_state(
        grid,
        stream_rate,
        center=100.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
        specific_total_energy=0.0,
    )
    external = np.zeros(grid.centers.size)
    external[-4:] = 0.01 * stream_rate * stream_l / 4.0
    result = solve_signed_flux_steady(
        grid,
        1.0e14,
        mass,
        boundary=SignedFluxBoundary(outer_mode="zero_torque"),
        stream_state=source,
        external_angular_rate_cells=external,
    )
    edge_l = np.asarray(potential.l_k(grid.edges), dtype=float)
    expected = (
        stream_rate * edge_l[-1]
        - stream_rate * stream_l
        - np.sum(external)
    ) / (stream_rate * (edge_l[-1] - edge_l[0]))

    assert result.mdot_faces[0] / stream_rate == pytest.approx(expected, rel=2.0e-14)
    assert abs(result.angular_momentum_budget_defect) < 1.0e-12 * stream_rate * stream_l
