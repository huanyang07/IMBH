import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_semidiscrete_energy import (
    causal_scaled_control_energy_metric,
    causal_semidiscrete_control_energy_history,
    causal_semidiscrete_generator_components,
)


def _toy_problem():
    cells = 3
    fields = 5
    dimensions = cells * fields
    descriptor = np.eye(dimensions)
    face = np.zeros((cells + 1, fields, dimensions))
    identity = np.eye(fields)
    for selected in range(cells + 1):
        if selected == 0:
            face[selected, :, :fields] = identity
        elif selected == cells:
            face[selected, :, -fields:] = identity
        else:
            left = slice(fields * (selected - 1), fields * selected)
            right = slice(fields * selected, fields * (selected + 1))
            face[selected, :, left] = 0.5 * identity
            face[selected, :, right] = 0.5 * identity
    conservative = np.zeros((dimensions, dimensions))
    for cell in range(cells):
        selected = slice(fields * cell, fields * (cell + 1))
        conservative[selected] = face[cell + 1] - face[cell]
    damping = 0.05 * np.eye(dimensions)
    stationary = {
        "candidate_conservative_transport": conservative,
        "candidate_local_stress_relaxation": damping,
    }
    mapped = 0.02 * np.eye(dimensions)
    height = 0.01 * np.eye(dimensions)
    components, defect = causal_semidiscrete_generator_components(
        descriptor,
        stationary_scaled_blocks=stationary,
        mapped_storage_rate_scaled_matrix=mapped,
        responsive_height_storage_rate_scaled_matrix=height,
    )
    generator = sum(
        components.values(),
        start=np.zeros_like(descriptor),
    )
    return descriptor, face, stationary, mapped, height, generator


def test_scaled_control_energy_metric_selects_declared_band():
    energy = np.repeat(np.eye(5)[None, :, :], 3, axis=0)
    metric = causal_scaled_control_energy_metric(
        energy,
        np.asarray((0.0, 0.5, 1.5, 3.0)),
        np.ones(15),
        1,
        3,
    )
    assert np.array_equal(metric[:5, :5], np.zeros((5, 5)))
    assert np.allclose(metric[5:10, 5:10], np.eye(5))
    assert np.allclose(metric[10:, 10:], 1.5 * np.eye(5))


def test_generator_blocks_and_face_powers_close_exactly():
    descriptor, face, stationary, mapped, height, generator = _toy_problem()
    rng = np.random.default_rng(1207)
    states = rng.normal(size=(7, 2, 15))
    metric = causal_scaled_control_energy_metric(
        np.repeat(np.eye(5)[None, :, :], 3, axis=0),
        np.arange(4, dtype=float),
        np.ones(15),
        0,
        3,
    )
    history = causal_semidiscrete_control_energy_history(
        states,
        scaled_energy_metric=metric,
        descriptor_scaled_matrix=descriptor,
        scaled_generator_per_s=generator,
        stationary_scaled_blocks=stationary,
        mapped_storage_rate_scaled_matrix=mapped,
        responsive_height_storage_rate_scaled_matrix=height,
        conservation_row_scales=np.ones(15),
        shared_face_flux_scaled_jacobians=face,
    )
    assert history.maximum_generator_power_defect <= 1.0e-14
    assert history.maximum_block_power_defect <= 1.0e-14
    assert history.maximum_face_power_defect <= 1.0e-14
    assert np.allclose(
        np.sum(history.conservative_face_powers, axis=-1),
        history.block_powers["candidate_conservative_transport"],
    )
    assert np.allclose(
        np.sum(
            np.asarray(tuple(history.block_powers.values())),
            axis=0,
        ),
        history.direct_generator_power,
    )
