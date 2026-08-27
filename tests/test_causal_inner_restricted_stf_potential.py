import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import (
    full_shear_rest_frame,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_restricted_stf_potential import (
    audit_restricted_five_stf_scalar_potential,
)


def _frame():
    geometry = kerr_schild_column_geometry(3.0351964349786267e9, 8.0e8)
    return full_shear_rest_frame(
        geometry,
        radial_velocity_over_c=0.059228133078858144,
        azimuthal_velocity_over_c=0.8785842814260398,
        vertical_velocity_over_c=0.0,
    )


def test_restricted_moving_stf_invariants_cannot_generate_linear_stress():
    audit = audit_restricted_five_stf_scalar_potential(
        _frame(), temperature=4.4e6
    )
    assert audit.maximum_beta_transversality_defect <= 2.0e-13
    assert audit.maximum_nu_basis_derivative <= 2.0e-13
    assert audit.maximum_first_invariant_derivative_at_origin <= 2.0e-13
    assert audit.desired_linear_stress_map_norm > 0.0
    assert audit.candidate_linear_stress_map_norm == 0.0
    assert audit.linear_stress_map_relative_defect == pytest.approx(1.0)
    assert not audit.candidate_viable


def test_invalid_temperature_fails_closed():
    with pytest.raises(ValueError, match="positive"):
        audit_restricted_five_stf_scalar_potential(_frame(), temperature=0.0)
