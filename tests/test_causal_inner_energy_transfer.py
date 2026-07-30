import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_energy_transfer import (
    causal_normalized_arrival_energy,
    causal_positive_band_energy_history,
)


def test_positive_band_energy_and_invariant_family_partition():
    times = np.linspace(0.0, 1.0, 9)
    cells = 4
    fields = 2
    history = np.zeros((times.size, 2, cells, fields))
    history[:, 0, 1:3, 0] = times[:, None]
    history[:, 1, 1:3, 1] = 2.0 * times[:, None]
    edges = np.linspace(0.0, 1.0, cells + 1)
    metric = np.repeat(np.eye(fields)[None], cells, axis=0)
    projectors = np.zeros((cells, fields, fields, fields))
    projectors[:, 0, 0, 0] = 1.0
    projectors[:, 1, 1, 1] = 1.0

    measured = causal_positive_band_energy_history(
        history,
        log_edges=edges,
        energy_metrics=metric,
        projectors=projectors,
        lower_face=1,
        upper_face=3,
    )

    expected_first = 0.25 * times**2
    expected_second = times**2
    assert np.allclose(measured.total_energy[:, 0], expected_first)
    assert np.allclose(measured.total_energy[:, 1], expected_second)
    assert np.allclose(measured.family_energy[:, 0, 0], expected_first)
    assert np.all(measured.family_energy[:, 0, 1] == 0.0)
    assert np.all(measured.family_energy[:, 1, 0] == 0.0)
    assert np.allclose(measured.family_energy[:, 1, 1], expected_second)
    assert measured.maximum_family_partition_relative_defect == 0.0


def test_normalized_arrival_is_quadratic_and_positive():
    times = np.linspace(0.0, 1.0, 129)
    history = np.zeros((times.size, 2, 2, 2))
    history[:, 0, :, 0] = times[:, None]
    history[:, 1, :, 0] = 0.5 * times[:, None]
    edges = np.asarray((0.0, 0.5, 1.0))
    metric = np.repeat(np.eye(2)[None], 2, axis=0)
    projectors = np.zeros((2, 2, 2, 2))
    projectors[:, 0, 0, 0] = 1.0
    projectors[:, 1, 1, 1] = 1.0
    measured = causal_positive_band_energy_history(
        history,
        log_edges=edges,
        energy_metrics=metric,
        projectors=projectors,
        lower_face=0,
        upper_face=2,
    )
    arrival = causal_normalized_arrival_energy(
        times,
        measured,
        initial_source_energy=np.asarray((0.5, 0.125)),
        window_seconds=(0.0, 1.0),
    )
    expected = np.trapezoid(times**2, times)
    assert np.allclose(arrival.total_time_average, expected)
    assert np.allclose(arrival.family_time_average[:, 0], expected)
    assert np.all(arrival.family_time_average[:, 1] == 0.0)
    assert np.allclose(arrival.peak_total, 1.0)
    assert arrival.maximum_integrated_partition_relative_defect == 0.0


def test_positive_band_energy_rejects_invalid_geometry_and_normalization():
    history = np.zeros((3, 1, 2, 1))
    metric = np.ones((2, 1, 1))
    projectors = np.ones((2, 1, 1, 1))
    with pytest.raises(ValueError, match="band-energy"):
        causal_positive_band_energy_history(
            history,
            log_edges=np.asarray((0.0, 1.0, 0.5)),
            energy_metrics=metric,
            projectors=projectors,
            lower_face=0,
            upper_face=2,
        )
    measured = causal_positive_band_energy_history(
        history,
        log_edges=np.asarray((0.0, 0.5, 1.0)),
        energy_metrics=metric,
        projectors=projectors,
        lower_face=0,
        upper_face=2,
    )
    with pytest.raises(ValueError, match="arrival-energy"):
        causal_normalized_arrival_energy(
            np.asarray((0.0, 0.5, 1.0)),
            measured,
            initial_source_energy=np.asarray((0.0,)),
            window_seconds=(0.0, 1.0),
        )
