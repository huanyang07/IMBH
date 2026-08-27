import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_equilibrium_core_trajectory import (
    advance_equilibrium_core_trajectory,
    audit_equilibrium_core_trajectory,
    initialize_equilibrium_core_trajectory,
    load_equilibrium_core_trajectory_checkpoint,
    save_equilibrium_core_trajectory_checkpoint,
    trajectory_primitive_array,
)
from tests.test_causal_inner_conservative_entropy_projection_microstep import (
    _three_cell_patch,
)


def test_trajectory_advances_accepted_state_and_roundtrips(tmp_path):
    geometry, height, points, seeds = _three_cell_patch()
    state = initialize_equilibrium_core_trajectory(
        geometry=geometry,
        proper_half_thickness=height,
        points=points,
        seeds=seeds,
    )
    for _ in range(3):
        advance = advance_equilibrium_core_trajectory(
            state, courant_factor=0.01
        )
        assert advance.accepted
        state = advance.state
    diagnostics = audit_equilibrium_core_trajectory(state)
    assert diagnostics.cumulative_conservation_relative_defect <= 2.0e-12
    assert diagnostics.cumulative_entropy_relative_defect <= 2.0e-11
    checkpoint = tmp_path / "trajectory.npz"
    save_equilibrium_core_trajectory_checkpoint(checkpoint, state)
    loaded = load_equilibrium_core_trajectory_checkpoint(checkpoint)
    assert loaded.accepted_steps == state.accepted_steps
    assert loaded.accumulated_courant_time == state.accumulated_courant_time
    assert np.array_equal(
        trajectory_primitive_array(loaded), trajectory_primitive_array(state)
    )
    original_next = advance_equilibrium_core_trajectory(
        state, courant_factor=0.01
    )
    replay_next = advance_equilibrium_core_trajectory(
        loaded, courant_factor=0.01
    )
    assert original_next.accepted and replay_next.accepted
    assert np.array_equal(
        trajectory_primitive_array(original_next.state),
        trajectory_primitive_array(replay_next.state),
    )
