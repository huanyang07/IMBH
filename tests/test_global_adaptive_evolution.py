from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from imri_qpe.constants import M_SUN
from imri_qpe.layer3_minidisk_1d.global_adaptive_evolution import (
    GlobalAdaptiveRestart,
    GlobalAdaptiveStepConfig,
    advance_global_adaptive_backward_euler,
    load_global_adaptive_restart,
    save_global_adaptive_milestone,
    save_global_adaptive_restart,
)
from imri_qpe.layer3_minidisk_1d.global_signed_evolution import (
    make_global_mechanical_energy_reference,
    state_from_thermodynamic_primitives,
)
from imri_qpe.layer3_minidisk_1d.grid import make_log_grid
from imri_qpe.layer3_minidisk_1d.transonic_potential import (
    PaczynskiWiitaPotential,
)


def _state(n_cells: int = 6):
    mass = 1.0e4 * M_SUN
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(20.0 * potential.r_g, 80.0 * potential.r_g, n_cells)
    state = state_from_thermodynamic_primitives(
        grid,
        np.full(n_cells, 2.0e4),
        np.zeros(n_cells),
        np.asarray(potential.omega_k(grid.centers), dtype=float),
        np.full(n_cells, 8.0e5),
        mass,
    )
    return mass, grid, state


def test_adaptive_config_rejects_invalid_controller_values() -> None:
    with pytest.raises(ValueError, match="minimum_dt"):
        GlobalAdaptiveStepConfig(2.0, 1.0).validated()
    with pytest.raises(ValueError, match="shrink_factor"):
        GlobalAdaptiveStepConfig(1.0, 2.0, shrink_factor=1.0).validated()
    assert GlobalAdaptiveStepConfig(1.0, 2.0).validated().growth_factor == 1.5


def test_adaptive_step_rejects_halves_and_then_grows(monkeypatch) -> None:
    mass, grid, state = _state()
    accepted_state = state_from_thermodynamic_primitives(
        grid,
        np.full(grid.centers.size, 2.001e4),
        np.zeros(grid.centers.size),
        np.asarray(PaczynskiWiitaPotential(mass).omega_k(grid.centers)),
        np.full(grid.centers.size, 8.001e5),
        mass,
    )
    calls = []

    def fake_step(_grid, original, _mass, dt, **_options):
        calls.append(dt)
        accepted = len(calls) > 1
        return SimpleNamespace(
            state=accepted_state if accepted else original,
            accepted=accepted,
            nfev=8 if accepted else 100,
            maximum_scaled_residual=1.0e-12 if accepted else 1.0e-3,
            message="accepted" if accepted else "rejected",
        )

    monkeypatch.setattr(
        "imri_qpe.layer3_minidisk_1d.global_adaptive_evolution."
        "advance_global_backward_euler",
        fake_step,
    )
    result = advance_global_adaptive_backward_euler(
        grid,
        state,
        mass,
        1.0,
        GlobalAdaptiveStepConfig(0.1, 2.0),
    )
    assert result.accepted
    assert calls == [1.0, 0.5]
    assert result.dt_used == 0.5
    assert result.dt_next == 0.75
    assert len(result.attempts) == 2
    assert not result.attempts[0].nonlinear_accepted
    assert result.attempts[1].physical_change_accepted
    controller = result.attempts[1].controller
    assert controller is not None
    assert controller.variable in {
        "log_surface_density",
        "log_temperature",
        "relative_thickness",
    }
    assert 0 <= controller.cell_index < grid.centers.size
    assert controller.radius == grid.centers[controller.cell_index]
    assert controller.change_metric >= 0.0
    assert controller.fraction_of_limit == pytest.approx(
        controller.change_metric / controller.limit
    )
    assert len(controller.characteristic_speeds) == 4
    assert result.attempts[1].nonlinear_solve_audit is None


def test_global_adaptive_restart_round_trips_bitwise(tmp_path) -> None:
    _mass, grid, state = _state()
    correction = np.linspace(-2.0e14, 3.0e14, grid.centers.size)
    mechanical = make_global_mechanical_energy_reference(
        grid,
        correction,
        state,
        provenance={"case": "adaptive-restart-test"},
    )
    restart = GlobalAdaptiveRestart(
        state=state,
        reference_state=state,
        mechanical_reference=mechanical,
        elapsed_time=12.5,
        dt_next=0.25,
        accepted_steps=7,
        rejected_attempts=2,
        provenance={"solver": "test", "version": 1},
    )
    path = tmp_path / "restart.npz"
    save_global_adaptive_restart(path, grid, restart)
    restored_grid, restored = load_global_adaptive_restart(path, grid=grid)
    assert np.array_equal(restored_grid.edges, grid.edges)
    for name in (
        "mass",
        "radial_momentum",
        "angular_momentum",
        "total_energy",
    ):
        assert np.array_equal(getattr(restored.state, name), getattr(state, name))
        assert np.array_equal(
            getattr(restored.reference_state, name), getattr(state, name)
        )
    assert np.array_equal(
        restored.mechanical_reference.specific_offset, correction
    )
    assert restored.elapsed_time == 12.5
    assert restored.dt_next == 0.25
    assert restored.accepted_steps == 7
    assert restored.rejected_attempts == 2
    assert restored.provenance == restart.provenance

    incompatible = make_log_grid(
        grid.edges[0], 1.01 * grid.edges[-1], grid.centers.size
    )
    with pytest.raises(ValueError, match="grid does not match"):
        load_global_adaptive_restart(path, grid=incompatible)


def test_global_adaptive_milestone_is_immutable_and_manifested(tmp_path) -> None:
    _mass, grid, state = _state()
    mechanical = make_global_mechanical_energy_reference(
        grid,
        np.zeros(grid.centers.size),
        state,
        provenance={"case": "milestone-test"},
    )
    restart = GlobalAdaptiveRestart(
        state=state,
        reference_state=state,
        mechanical_reference=mechanical,
        elapsed_time=3.25,
        dt_next=0.125,
        accepted_steps=11,
        rejected_attempts=4,
        provenance={"git": {"full_sha": "a" * 40}},
    )
    first = save_global_adaptive_milestone(
        tmp_path,
        "global test",
        grid,
        restart,
        metadata={"target_reached": True},
    )
    second = save_global_adaptive_milestone(
        tmp_path,
        "global test",
        grid,
        restart,
        metadata={"target_reached": True},
    )
    assert first == second
    assert len(first["checkpoint_sha256"]) == 64
    assert len(first["state_sha256"]) == 64
    assert (tmp_path / first["path"]).exists()
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["checkpoints"] == [first]
