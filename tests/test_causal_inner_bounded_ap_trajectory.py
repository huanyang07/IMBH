import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_bounded_ap_trajectory import (
    APAtlasPath,
    APTrajectoryCheckpoint,
    deterministic_initial_state,
    deterministic_slow_forcing,
    fast_slaving_defect,
    integrate_ap_trajectory,
    load_ap_checkpoint,
    save_ap_checkpoint,
    source_nullity,
)


def _fixture():
    radial0 = np.diag(np.linspace(-0.7, 0.7, 11))
    rotation = np.eye(11)
    angle = 0.11
    rotation[0, 1] = np.sin(angle)
    rotation[1, 0] = -np.sin(angle)
    rotation[0, 0] = rotation[1, 1] = np.cos(angle)
    radial1 = rotation @ radial0 @ rotation.T
    source0 = np.diag([0.0] * 4 + list(-np.linspace(0.4, 1.0, 7)))
    source1 = np.diag([0.0] * 4 + list(-np.linspace(0.5, 1.1, 7)))
    return APAtlasPath(radial0, source0, radial1, source1)


def _functions(horizon):
    wave = lambda time: 0.3 + 0.05 * np.sin(2.0 * np.pi * time / horizon)
    forcing = lambda time: deterministic_slow_forcing(time, horizon)
    return wave, forcing


def test_online_ap_trajectory_is_bounded_contracting_and_restartable():
    path = _fixture(); horizon = 2.0; wave, forcing = _functions(horizon)
    initial = deterministic_initial_state()
    full = integrate_ap_trajectory(path, initial, start_time=0.0, end_time=horizon, atlas_horizon=horizon, step_count=16, stiffness=100.0, wave_number=wave, forcing=forcing)
    first = integrate_ap_trajectory(path, initial, start_time=0.0, end_time=horizon / 2.0, atlas_horizon=horizon, step_count=8, stiffness=100.0, wave_number=wave, forcing=forcing)
    # Replay the suffix with the global endpoint while preserving its step size.
    second = integrate_ap_trajectory(path, first.final_state, start_time=horizon / 2.0, end_time=horizon, atlas_horizon=horizon, step_count=8, stiffness=100.0, wave_number=wave, forcing=forcing)
    assert np.array_equal(full.final_state, second.final_state)
    assert full.maximum_state_norm < 0.5
    assert full.maximum_homogeneous_step_expansivity < 2.0e-12
    assert source_nullity(path.source_start) == 4


def test_stiff_fast_coordinates_are_slaved():
    path = _fixture(); horizon = 2.0; wave, forcing = _functions(horizon)
    result = integrate_ap_trajectory(path, deterministic_initial_state(), start_time=0.0, end_time=horizon, atlas_horizon=horizon, step_count=32, stiffness=1000.0, wave_number=wave, forcing=forcing)
    defect = fast_slaving_defect(path, result.final_state, time=horizon, horizon=horizon, stiffness=1000.0, wave_number=wave, forcing=forcing)
    assert defect < 2.0e-2


def test_arbitrary_step_checkpoint_is_lossless(tmp_path):
    path = _fixture()
    state = deterministic_initial_state()
    checkpoint = APTrajectoryCheckpoint(path, state, 1.0, 2.0, 100.0, 8)
    filename = tmp_path / "checkpoint.npz"
    save_ap_checkpoint(checkpoint, filename)
    loaded = load_ap_checkpoint(filename)
    assert np.array_equal(loaded.state, checkpoint.state)
    assert np.array_equal(loaded.path.radial_start, path.radial_start)
    assert np.array_equal(loaded.path.source_end, path.source_end)
    assert loaded.time == checkpoint.time
    assert loaded.completed_steps == checkpoint.completed_steps
