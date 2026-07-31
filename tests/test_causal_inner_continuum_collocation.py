import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (
    CausalFiveFieldContinuumBackground,
    causal_five_field_inward_collocation_generator,
    causal_five_field_inward_collocation_generator_blocks,
    causal_sixth_order_inward_collocation_derivative,
)


def test_sixth_order_inward_derivative_for_outer_buffered_wave():
    errors = []
    for count in (65, 129, 257):
        nodes = np.linspace(0.0, 1.0, count)
        values = np.sin(np.pi * nodes) ** 8
        exact = (
            8.0
            * np.pi
            * np.sin(np.pi * nodes) ** 7
            * np.cos(np.pi * nodes)
        )
        derivative = causal_sixth_order_inward_collocation_derivative(nodes)
        errors.append(
            float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(derivative @ values).ravel()
                            - exact
                        )
                        ** 2
                    )
                )
            )
        )
    assert np.log2(errors[0] / errors[1]) >= 5.5
    assert np.log2(errors[1] / errors[2]) >= 5.5


def test_continuum_generator_blocks_sum_to_complete_generator():
    count = 9
    radii = np.exp(np.linspace(np.log(1.8), np.log(2.8), count))
    identity = np.eye(5)
    zeros_3 = np.zeros((count, 5, 5))
    zeros_4 = np.zeros((count, 5, 5, 5))
    temporal = np.repeat(identity[None], count, axis=0)
    flux = np.repeat((-0.5 * identity)[None], count, axis=0)
    background = CausalFiveFieldContinuumBackground(
        context=None,
        radii=radii,
        log_radii=np.log(radii),
        primitive_charts=np.zeros((count, 5)),
        primitive_radius_derivative=np.zeros((count, 5)),
        base_rate_per_s=np.zeros((count, 5)),
        face_measures=np.ones(count),
        mapped_conserved_jacobians=temporal,
        mapped_conserved_hessians=zeros_4,
        vertical_storage_matrices=zeros_3,
        vertical_storage_derivatives=zeros_4,
        temporal_storage_matrices=temporal,
        physical_flux_jacobians=flux,
        shear_principal_matrices=zeros_3,
        height_principal_matrices=zeros_3,
        shear_principal_derivatives=zeros_4,
        height_principal_derivatives=zeros_4,
        lower_source_values={
            "stress_relaxation": np.zeros((count, 5)),
            "perfect_fluid_geometry": np.zeros((count, 5)),
            "stress_geometry": np.zeros((count, 5)),
            "radiative_cooling": np.zeros((count, 5)),
            "vertical_work": np.zeros((count, 5)),
        },
        lower_source_jacobians={
            "stress_relaxation": zeros_3,
            "perfect_fluid_geometry": zeros_3,
            "stress_geometry": zeros_3,
            "radiative_cooling": zeros_3,
            "vertical_work": zeros_3,
        },
        base_stationary_block_densities={},
        base_stationary_density=np.zeros((count, 5)),
    )
    blocks = causal_five_field_inward_collocation_generator_blocks(
        background
    )
    complete = causal_five_field_inward_collocation_generator(background)
    assembled = sum(blocks.values()).tocsr()
    assert tuple(blocks) == (
        "candidate_conservative_transport",
        "candidate_shear_principal",
        "candidate_height_principal",
        "candidate_local_stress_relaxation",
        "candidate_geometry",
        "candidate_cooling",
        "candidate_stream",
        "candidate_lower_height_work",
        "mapped_storage_rate_derivative",
        "responsive_height_storage_rate_derivative",
    )
    assert np.max(np.abs((assembled - complete).toarray())) == 0.0
