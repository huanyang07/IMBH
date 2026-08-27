import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import full_shear_rest_frame
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import kerr_schild_column_geometry
from imri_qpe.layer3_minidisk_1d.causal_inner_nonlinear_port_atlas import (
    audit_conditioned_discrete_gradient_radial_flux,
    audit_equilibrium_entropy_path_flux,
    audit_stf_polar_connection,
    equilibrium_temporal_conserved,
    equilibrium_entropy_point_from_primitive,
    equilibrium_mathematical_entropy_decimal,
    recover_equilibrium_point_from_temporal_conserved,
)


def _geometry():
    return kerr_schild_column_geometry(radius=3.2e9, gravitational_radius=8.0e8)


def test_entropy_path_flux_satisfies_tadmor_identity():
    geometry = _geometry(); height = 2.4e8
    left = equilibrium_entropy_point_from_primitive(geometry, density=2.1e-7, temperature=4.3e6, proper_half_thickness=height, radial_velocity_over_c=-0.21, azimuthal_velocity_over_c=0.31)
    right = equilibrium_entropy_point_from_primitive(geometry, density=2.12e-7, temperature=4.28e6, proper_half_thickness=height, radial_velocity_over_c=-0.209, azimuthal_velocity_over_c=0.309)
    audit = audit_equilibrium_entropy_path_flux(left, right)
    assert audit.passed


def test_moving_STF_polar_connection_is_an_isometric_roundtrip():
    geometry = _geometry()
    left = full_shear_rest_frame(geometry, radial_velocity_over_c=-0.21, azimuthal_velocity_over_c=0.31, vertical_velocity_over_c=0.0)
    right = full_shear_rest_frame(geometry, radial_velocity_over_c=-0.209, azimuthal_velocity_over_c=0.309, vertical_velocity_over_c=0.0)
    audit = audit_stf_polar_connection(left, right)
    assert audit.passed
    assert audit.orthogonality_defect <= 2e-12
    assert audit.reverse_roundtrip_defect <= 2e-12


def test_conditioned_discrete_gradient_is_symmetric_consistent_and_conservative():
    geometry = _geometry(); height = 2.4e8
    left = equilibrium_entropy_point_from_primitive(geometry, density=2.1e-7, temperature=4.3e6, proper_half_thickness=height, radial_velocity_over_c=-0.21, azimuthal_velocity_over_c=0.31)
    right = equilibrium_entropy_point_from_primitive(geometry, density=2.12e-7, temperature=4.28e6, proper_half_thickness=height, radial_velocity_over_c=-0.209, azimuthal_velocity_over_c=0.309)
    audit = audit_conditioned_discrete_gradient_radial_flux(left, right)
    assert audit.passed


def test_temporal_current_recovery_roundtrips_physical_primitive():
    geometry = _geometry(); height = 2.4e8
    point = equilibrium_entropy_point_from_primitive(geometry, density=2.1e-7, temperature=4.3e6, proper_half_thickness=height, radial_velocity_over_c=-0.21, azimuthal_velocity_over_c=0.31)
    recovered = recover_equilibrium_point_from_temporal_conserved(geometry, proper_half_thickness=height, target_conserved=equilibrium_temporal_conserved(point), initial_density=2.08e-7, initial_temperature=4.28e6, initial_radial_velocity_over_c=-0.209, initial_azimuthal_velocity_over_c=0.309)
    assert recovered.converged
    assert recovered.scaled_residual_norm <= 1e-11
    assert abs(recovered.density / 2.1e-7 - 1) <= 2e-10
    assert equilibrium_mathematical_entropy_decimal(recovered.point).is_finite()
