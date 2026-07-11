from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    PhysicalTransportClosure,
    TransonicSlimParams,
    conservative_block_jacobian,
    conservative_eta_bordered_jacobian,
    conservative_jacobian_directional_audit,
    conservative_jacobian_sparsity,
    conservative_local_dae_matrix,
    conservative_local_dae_residual,
    conservative_residual,
    conservative_residual_audit,
    conservative_seed_from_legacy,
    conservative_sonic_diagnostics,
    conservative_transport_quadrature_profile,
    conservative_wind_escape_profile,
    multidomain_conservative_grid,
    nested_refined_conservative_grid,
    pack_conservative_state,
    remap_conservative_state,
    residual_adapted_conservative_grid,
    source_block_refined_conservative_grid,
    unpack_conservative_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]


def _canonical_mdot5() -> tuple[np.ndarray, TransonicSlimParams]:
    fiducial = FiducialParams()
    with np.load(ROOT / "results/canonical/no_wind_mdot5/state.npz") as data:
        grid = np.asarray(data["custom_grid_xi"], dtype=float)
        slopes = np.asarray(data["outer_match_log_slopes"], dtype=float)
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=float(data["ratio"]) * eddington_mdot(fiducial.M2_g),
            alpha=0.01,
            mu_stress=0.0,
            stress_factor=1.0,
            R_out_rg=float(data["R_out_rg"]),
            n_nodes=int(data["n_nodes"]),
            grid_power=float(data["grid_power"]),
            custom_grid_xi=tuple(float(value) for value in grid),
            outer_closure=str(np.asarray(data["outer_closure"]).item()),
            outer_match_log_slopes=(float(slopes[0]), float(slopes[1])),
            residual_tol=1.0e-8,
            max_nfev=1,
        )
        z = np.asarray(data["z"], dtype=float)
    return z, params


def test_pack_unpack_round_trip_on_small_view() -> None:
    legacy, disk_full = _canonical_mdot5()
    n = 8
    disk = replace(
        disk_full,
        n_nodes=n,
        custom_grid_xi=tuple(np.linspace(0.0, 1.0, n)),
    )
    closure = PhysicalTransportClosure(
        stream_circularization_radius=0.8 * disk.R_out,
    )
    # Only shapes are relevant here; the legacy state is not remapped.
    logu = np.linspace(20.0, 18.0, n)
    logT = np.linspace(17.0, 15.0, n)
    F = np.ones(n)
    j = np.ones(n)
    epsilon = np.linspace(-0.1, 0.1, n)
    x = pack_conservative_state(logu, logT, F, j, epsilon, np.log(5.0 * disk.r_g))
    from imri_qpe.layer3_minidisk_1d import ConservativeBoundary, ConservativeSolverParams

    params = ConservativeSolverParams(
        disk=disk,
        closure=closure,
        boundary=ConservativeBoundary(logT[-1], 0.0),
    )
    unpacked = unpack_conservative_state(x, params)
    np.testing.assert_allclose(unpacked[0], logu)
    np.testing.assert_allclose(unpacked[1], logT)
    np.testing.assert_allclose(unpacked[2], F)
    np.testing.assert_allclose(unpacked[3], j)
    np.testing.assert_allclose(unpacked[4], epsilon)
    assert unpacked[-1].shape == (n,)
    assert legacy.size > 0


def test_canonical_legacy_mapping_is_finite_and_sparse() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(
        stream_circularization_radius=0.8 * disk.R_out,
    )
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    residual = conservative_residual(seed, params)
    audit = conservative_residual_audit(seed, params)
    sparsity = conservative_jacobian_sparsity(params)

    assert residual.shape == seed.shape
    assert np.all(np.isfinite(residual))
    assert np.isfinite(audit.maximum)
    assert audit.mass < 1.0e-12
    assert audit.angular_momentum < 1.0e-12
    assert audit.energy_compatibility < 5.0e-4
    assert sparsity.shape == (seed.size, seed.size)
    assert sparsity.nnz < 30 * seed.size


def test_conservative_remap_preserves_constant_dimensional_fluxes() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    target_disk = replace(disk, n_nodes=97, custom_grid_xi=None, grid_power=0.6)
    remapped, target = remap_conservative_state(seed, params, target_disk)
    old = unpack_conservative_state(seed, params)
    new = unpack_conservative_state(remapped, target)

    assert np.ptp(old[2]) < 1.0e-12
    assert np.ptp(old[3]) < 1.0e-12
    assert np.ptp(new[2]) < 1.0e-12
    assert np.ptp(new[3]) < 1.0e-12
    assert new[2][0] * target.flux_scales.mdot == pytest.approx(
        old[2][0] * params.flux_scales.mdot
    )


def test_conservative_local_dae_matrix_is_finite_and_matches_direction() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    logu, logT, F, j, epsilon, _logR_son, logR = unpack_conservative_state(seed, params)
    state = np.asarray([logu[0], logT[0], F[0], j[0], epsilon[0]], dtype=float)
    gradient = np.asarray([0.1, -0.2, 0.0, 0.0, 0.01], dtype=float)
    matrix, affine = conservative_local_dae_matrix(logR[0], state, params)
    direct = conservative_local_dae_residual(logR[0], state, gradient, params)

    assert matrix.shape == (5, 5)
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(affine))
    np.testing.assert_allclose(direct, matrix @ gradient + affine, rtol=2.0e-5, atol=2.0e-7)

    diagnostics = conservative_sonic_diagnostics(logR[0], state, params)
    assert diagnostics.singular_values.shape == (5,)
    assert np.isfinite(diagnostics.determinant)
    assert np.isfinite(diagnostics.compatibility)


def test_optimizer_weights_leave_raw_conservative_audit_unchanged() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    base = conservative_residual_audit(seed, params)
    weighted = replace(
        params,
        mass_weight=7.0,
        angular_momentum_weight=5.0,
        energy_flux_weight=3.0,
        energy_balance_weight=11.0,
        inner_mass_weight=13.0,
    )
    audit = conservative_residual_audit(seed, weighted)

    assert audit.mass == pytest.approx(base.mass)
    assert audit.angular_momentum == pytest.approx(base.angular_momentum)
    assert audit.energy == pytest.approx(base.energy)
    assert audit.energy_compatibility == pytest.approx(base.energy_compatibility)
    assert audit.inner_mass == pytest.approx(base.inner_mass)


def test_power_and_carried_wind_modes_have_equivalent_production_residuals() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(
        stream_circularization_radius=0.8 * disk.R_out,
        wind_launch_energy_multiplier=8.0,
    )
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    wind_disk = replace(
        disk,
        wind_energy_limited_epsilon=0.2,
        wind_activation_width_fraction=0.05,
    )
    power = replace(params, disk=wind_disk, wind_energy_transport_mode="power")
    carried = replace(power, wind_energy_transport_mode="carried")

    residual_power = conservative_residual(seed, power)
    residual_carried = conservative_residual(seed, carried)
    np.testing.assert_allclose(residual_power, residual_carried, rtol=0.0, atol=2.0e-14)

    escape = conservative_wind_escape_profile(seed, power)
    assert escape["R_mid_rg"].shape == (disk.n_nodes - 1,)
    assert np.all(np.isfinite(escape["wind_bernoulli"]))
    assert np.all(
        escape["escaping"]
        == (escape["terminal_margin"] >= 0.0)
    )

    capped = replace(
        power,
        closure=replace(
            power.closure,
            wind_mass_loading_cap_per_log_radius=1.0e-12,
        ),
    )
    capped_profile = conservative_wind_escape_profile(seed, capped)
    assert np.any(capped_profile["wind_cap_active"])
    assert np.all(capped_profile["wind_prime"] <= capped_profile["wind_raw_prime"])
    assert np.any(capped_profile["wind_prime"] < capped_profile["wind_raw_prime"])
    np.testing.assert_allclose(
        capped_profile["wind_launch_power_prime"],
        capped_profile["wind_prime"] * capped_profile["prescribed_launch_energy"],
        rtol=2.0e-15,
        atol=0.0,
    )

    target_disk = replace(disk, n_nodes=97, custom_grid_xi=None, grid_power=0.6)
    _remapped, remapped_params = remap_conservative_state(seed, carried, target_disk)
    assert remapped_params.wind_energy_transport_mode == "carried"

    from imri_qpe.layer3_minidisk_1d import ConservativeBoundary, ConservativeSolverParams

    with pytest.raises(ValueError, match="wind_energy_transport_mode"):
        ConservativeSolverParams(
            disk=disk,
            closure=closure,
            boundary=ConservativeBoundary(0.0, 0.0),
            wind_energy_transport_mode="invalid",
        )


def test_residual_adapted_grid_is_monotone_and_has_fixed_endpoints() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    grid = np.asarray(
        residual_adapted_conservative_grid(seed, params, target_n=73), dtype=float
    )

    assert grid.shape == (73,)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0.0)


def test_source_block_grid_preserves_outside_nodes_and_exact_landmarks() -> None:
    legacy, disk = _canonical_mdot5()
    disk = replace(
        disk,
        R_out_rg=335.0,
        stream_source_fraction=0.3,
        stream_source_center_fraction=240.0 / 335.0,
        stream_source_log_width=0.08,
        stream_source_shape="compact_c2",
    )
    closure = PhysicalTransportClosure(stream_circularization_radius=240.0 * disk.r_g)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    *_fields, logR_son, old_logR = unpack_conservative_state(seed, params)
    grid = np.asarray(
        source_block_refined_conservative_grid(seed, params, source_nodes=32),
        dtype=float,
    )
    new_logR = logR_son + grid * (old_logR[-1] - logR_son)
    center = np.log(240.0 * disk.r_g)
    left = center - 0.08
    right = center + 0.08

    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0.0)
    assert np.count_nonzero((new_logR >= left) & (new_logR <= right)) == 32
    for landmark in (left, center, right):
        assert np.min(np.abs(new_logR - landmark)) < 1.0e-13
    inherited = old_logR[(old_logR < left) | (old_logR > right)]
    for value in inherited:
        assert np.min(np.abs(new_logR - value)) < 1.0e-13


def test_high_order_transport_audit_returns_finite_interval_profiles() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    profile8 = conservative_transport_quadrature_profile(seed, params, order=8)
    profile16 = conservative_transport_quadrature_profile(seed, params, order=16)

    for name in ("mass_rhs", "angular_rhs", "energy_rhs"):
        assert profile8[name].shape == (disk.n_nodes - 1,)
        assert profile16[name].shape == (disk.n_nodes - 1,)
        assert np.all(np.isfinite(profile8[name]))
        assert np.all(np.isfinite(profile16[name]))
    np.testing.assert_allclose(profile8["mass_rhs"], profile16["mass_rhs"], atol=0.0)


def test_multidomain_grid_freezes_inner_nodes_and_resolves_source() -> None:
    legacy, disk = _canonical_mdot5()
    disk = replace(
        disk,
        R_out_rg=335.0,
        stream_source_fraction=0.3,
        stream_source_center_fraction=240.0 / 335.0,
        stream_source_log_width=0.08,
        stream_source_shape="compact_c2",
    )
    closure = PhysicalTransportClosure(stream_circularization_radius=240.0 * disk.r_g)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    *_fields, logR_son, old_logR = unpack_conservative_state(seed, params)
    grid = np.asarray(
        multidomain_conservative_grid(
            seed,
            params,
            target_n=900,
            source_nodes=64,
            frozen_inner_nodes=12,
        )
    )
    new_logR = logR_son + grid * (old_logR[-1] - logR_son)
    center = np.log(240.0 * disk.r_g)

    assert grid.shape == (900,)
    assert np.all(np.diff(grid) > 0.0)
    np.testing.assert_allclose(new_logR[:12], old_logR[:12], atol=1.0e-13)
    for landmark in (center - 0.08, center, center + 0.08):
        assert np.min(np.abs(new_logR - landmark)) < 1.0e-13


def test_nested_grid_preserves_every_existing_node() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    *_fields, logR_son, old_logR = unpack_conservative_state(seed, params)
    grid = np.asarray(
        nested_refined_conservative_grid(seed, params, target_n=disk.n_nodes + 37)
    )
    new_logR = logR_son + grid * (old_logR[-1] - logR_son)

    assert new_logR.size == old_logR.size + 37
    assert np.all(np.diff(new_logR) > 0.0)
    for value in old_logR:
        assert np.min(np.abs(new_logR - value)) < 1.0e-13


def test_block_jacobian_matches_production_direction_on_small_grid() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    target_disk = replace(
        disk,
        n_nodes=9,
        custom_grid_xi=tuple(np.linspace(0.0, 1.0, 9)),
    )
    state, target = remap_conservative_state(seed, params, target_disk)
    jacobian = conservative_block_jacobian(state, target, rel_step=3.0e-6)
    audit = conservative_jacobian_directional_audit(
        state,
        target,
        steps=(1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5),
        jacobian_rel_step=3.0e-6,
    )

    assert jacobian.shape == (state.size, state.size)
    assert np.all(np.isfinite(jacobian.data))
    # This synthetic nine-node state is not a resolved sonic anchor; the
    # nested determinant/compatibility diagnostic limits its full-row audit.
    assert audit.best_relative_error < 3.0e-2
    n = target.disk.n_nodes
    assert jacobian[3, 4 * n] == pytest.approx(-target.energy_flux_weight)
    assert jacobian[3, 4 * n + 1] == pytest.approx(target.energy_flux_weight)


def test_eta_bordered_jacobian_adds_parameter_and_arc_rows() -> None:
    legacy, disk = _canonical_mdot5()
    closure = PhysicalTransportClosure(stream_circularization_radius=0.8 * disk.R_out)
    seed, params = conservative_seed_from_legacy(legacy, disk, closure)
    target_disk = replace(
        disk,
        n_nodes=7,
        custom_grid_xi=tuple(np.linspace(0.0, 1.0, 7)),
    )
    state, target = remap_conservative_state(seed, params, target_disk)
    tangent = np.ones(state.size + 1, dtype=float)
    tangent /= np.linalg.norm(tangent)
    scales = np.ones_like(tangent)
    jacobian = conservative_eta_bordered_jacobian(
        state,
        0.1,
        target,
        tangent=tangent,
        scales=scales,
    )

    assert jacobian.shape == (state.size + 1, state.size + 1)
    assert np.all(np.isfinite(jacobian.data))
    np.testing.assert_allclose(jacobian[-1].toarray().ravel(), tangent)
