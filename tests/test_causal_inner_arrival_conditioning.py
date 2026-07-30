import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_arrival_conditioning import (
    causal_arrival_history_conditioning,
    causal_history_uncertainty_envelope,
    causal_horizon_completeness,
    causal_quadratic_peak,
)


def test_quadratic_peak_recovers_between_samples():
    times = np.linspace(0.0, 1.0, 6)
    values = 3.0 - (times - 0.47) ** 2
    peak = causal_quadratic_peak(times, values)
    assert peak.interpolation_used
    assert np.isclose(peak.interpolated_time_seconds, 0.47)
    assert np.isclose(peak.interpolated_value, 3.0)


def test_history_conditioning_separates_gain_amplitude_and_shape():
    times = np.linspace(0.0, 1.0, 129)
    shape = np.sin(np.pi * times) ** 2
    continuum = 5000.0 * shape
    coarse = continuum + 400.0 * shape + 20.0 * np.sin(2.0 * np.pi * times)
    medium = continuum + 100.0 * shape + 5.0 * np.sin(2.0 * np.pi * times)
    fine = continuum + 25.0 * shape + 1.25 * np.sin(2.0 * np.pi * times)
    audit = causal_arrival_history_conditioning(
        coarse,
        medium,
        fine,
        times_seconds=times,
    )
    assert audit.absolute_fine_maximum_difference > 70.0
    assert audit.response_relative_fine_maximum_difference < 0.02
    assert np.isclose(audit.weighted_rms_order, 2.0)
    assert audit.amplitude_relative_fine_difference < 0.02
    assert audit.shape_fine_maximum_difference < 1.0e-3
    assert audit.fixed_second_order_reference_difference > 0.0


def test_uncertainty_envelope_uses_conservative_sum_not_rss():
    times = np.linspace(0.0, 1.0, 17)
    continuum = np.sin(np.pi * times)
    histories = (
        continuum + 0.16,
        continuum + 0.04,
        continuum + 0.01,
    )
    variations = {
        "band": np.asarray(
            [
                histories,
                tuple(item + 0.002 * times for item in histories),
            ]
        ),
        "time": np.asarray(
            [
                histories,
                tuple(
                    item + scale * 0.001 * times
                    for scale, item in zip((1.0, 0.5, 0.25), histories)
                ),
            ]
        ),
    }
    envelope = causal_history_uncertainty_envelope(
        *histories,
        times_seconds=times,
        variations=variations,
        observability_factor=5.0,
    )
    assert np.isclose(
        envelope.coarse_medium_conservative_l2,
        sum(envelope.coarse_medium_components_l2.values()),
    )
    assert np.isclose(
        envelope.medium_fine_conservative_linf,
        sum(envelope.medium_fine_components_linf.values()),
    )
    assert envelope.coarse_medium_observable
    assert envelope.medium_fine_observable


def test_horizon_completeness_distinguishes_cleared_and_live_tails():
    times = np.linspace(0.0, 10.0, 201)
    cleared = np.exp(-((times - 4.0) / 0.8) ** 2)
    live = 1.0 - np.exp(-times)
    gates = {
        "final_window_fraction": 0.1,
        "maximum_terminal_to_peak": 0.01,
        "maximum_final_window_range_to_peak": 0.02,
        "maximum_terminal_slope_horizon_to_peak": 0.05,
    }
    assert causal_horizon_completeness(times, cleared, **gates).complete
    assert not causal_horizon_completeness(times, live, **gates).complete
