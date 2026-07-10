from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    PhysicalTransportClosure,
    TransonicSlimParams,
    conservative_jacobian_sparsity,
    conservative_local_dae_matrix,
    conservative_local_dae_residual,
    conservative_residual,
    conservative_residual_audit,
    conservative_seed_from_legacy,
    conservative_sonic_diagnostics,
    pack_conservative_state,
    remap_conservative_state,
    residual_adapted_conservative_grid,
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
