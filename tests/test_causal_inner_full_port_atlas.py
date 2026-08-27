import numpy as np,pytest
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import *
def _a(v=-.27):return build_full_port_atlas_anchor(sound_speed=2.287047318996709e9,temperature=4.436398409641123e6,proper_half_thickness=2.45860382301911e8,proper_vertical_frequency=8.279018646718441,alpha=.1,shear_relaxation_time=.06262958217858178,transport_speed_over_c=v)
def test_field_count_and_full_stf_incidence():
 a=_a();assert len(FULL_PORT_FIELD_NAMES)==11;assert a.radial_stf_incidence.shape==(3,5);assert np.linalg.matrix_rank(a.radial_stf_incidence)==3;assert a.radial_stf_incidence[2,3]!=0
@pytest.mark.parametrize("v",(-.4,-.2,0,.2))
def test_anchor_is_symmetric_causal_and_dissipative(v):
 q=audit_full_port_atlas_anchor(_a(v));assert q.passed;assert q.coordinate_maximum_absolute_speed_over_c<.999;assert q.source_entropy_positive_part<=1e-12
def test_relativistic_spectral_map_not_galilean_shift():
 a=_a(.3);r=np.linalg.eigvalsh(a.rest_radial_matrix);mapped=np.linalg.eigvalsh(a.coordinate_radial_matrix);np.testing.assert_allclose(mapped,(.3+r)/(1+.3*r),atol=2e-15);assert np.max(np.abs(mapped-(.3+r)))>1e-5
def test_invalid_anchor_fails_closed():
 with pytest.raises(ValueError,match="physical"):build_full_port_atlas_anchor(sound_speed=0,temperature=1,proper_half_thickness=1,proper_vertical_frequency=1,alpha=.1,shear_relaxation_time=1,transport_speed_over_c=0)
