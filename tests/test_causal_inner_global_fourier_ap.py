import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_bounded_ap_trajectory import APAtlasPath
from imri_qpe.layer3_minidisk_1d.causal_inner_global_fourier_ap import (
    deterministic_global_forcing,
    deterministic_global_initial_state,
    integrate_global_fourier_ap,
)


def _path():
    radial0 = np.diag(np.linspace(-0.7, 0.7, 11))
    radial1 = np.diag(np.linspace(-0.65, 0.72, 11))
    source0 = np.diag([0.0] * 4 + list(-np.linspace(0.4, 1.0, 7)))
    source1 = np.diag([0.0] * 4 + list(-np.linspace(0.5, 1.1, 7)))
    return APAtlasPath(radial0, source0, radial1, source1)


def test_global_fourier_ap_preserves_core_totals_and_is_contracting():
    count = 12; horizon = 2.0
    forcing = lambda time: deterministic_global_forcing(time, horizon, count)
    result = integrate_global_fourier_ap(
        _path(),
        deterministic_global_initial_state(count),
        start_time=0.0,
        end_time=horizon,
        atlas_horizon=horizon,
        step_count=8,
        stiffness=100.0,
        forcing=forcing,
    )
    assert result.maximum_core_total_conservation_defect < 2.0e-12
    assert result.maximum_homogeneous_mode_expansivity < 2.0e-12
    assert result.maximum_state_norm < 1.0


def test_global_fourier_ap_suffix_replay_is_bitwise():
    count = 12; horizon = 2.0; path = _path()
    forcing = lambda time: deterministic_global_forcing(time, horizon, count)
    initial = deterministic_global_initial_state(count)
    full = integrate_global_fourier_ap(path, initial, start_time=0.0, end_time=horizon, atlas_horizon=horizon, step_count=8, stiffness=1000.0, forcing=forcing)
    first = integrate_global_fourier_ap(path, initial, start_time=0.0, end_time=1.0, atlas_horizon=horizon, step_count=4, stiffness=1000.0, forcing=forcing)
    suffix = integrate_global_fourier_ap(path, first.final_state, start_time=1.0, end_time=horizon, atlas_horizon=horizon, step_count=4, stiffness=1000.0, forcing=forcing)
    assert np.array_equal(full.final_state, suffix.final_state)
