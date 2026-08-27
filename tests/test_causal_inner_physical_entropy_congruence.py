import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import (
    build_full_port_atlas_anchor,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_physical_entropy_congruence import (
    audit_ap_fast_propagator,
    audit_corrected_physical_port_atlas,
    audit_physical_entropy_congruence,
    build_corrected_physical_port_atlas,
    build_physical_entropy_congruence,
    exponential_affine_step,
)


def _fixture():
    geometry = kerr_schild_column_geometry(3.2e9, 8.0e8)
    height = 2.4e8
    congruence = build_physical_entropy_congruence(
        geometry,
        proper_half_thickness=height,
        density=2.1e-7,
        temperature=4.3e6,
        radial_velocity_over_c=-0.21,
        azimuthal_velocity_over_c=0.31,
    )
    anchor = build_full_port_atlas_anchor(
        sound_speed=0.07 * 29979245800.0,
        temperature=4.3e6,
        proper_half_thickness=height,
        proper_vertical_frequency=2.0,
        alpha=0.1,
        shear_relaxation_time=0.2,
        transport_speed_over_c=-0.21,
    )
    atlas = build_corrected_physical_port_atlas(anchor, congruence, geometry)
    return geometry, congruence, anchor, atlas


def test_physical_entropy_congruence_and_corrected_port_are_causal():
    geometry, congruence, anchor, atlas = _fixture()
    assert audit_physical_entropy_congruence(congruence).passed
    assert audit_corrected_physical_port_atlas(
        atlas, anchor, congruence, geometry
    ).passed


def test_ap_propagator_is_contracting_and_composes():
    _, _, _, atlas = _fixture()
    assert audit_ap_fast_propagator(atlas).passed


def test_exponential_affine_step_matches_two_half_steps():
    generator = np.asarray(((-2.0, 0.5), (-0.5, -1.0)))
    state = np.asarray((0.2, -0.3))
    forcing = np.asarray((0.1, 0.05))
    full = exponential_affine_step(generator, state, forcing, 0.4)
    half = exponential_affine_step(generator, state, forcing, 0.2)
    twice = exponential_affine_step(generator, half, forcing, 0.2)
    assert np.allclose(full, twice, rtol=2e-14, atol=2e-14)
