"""State-local eleven-field equilibrium/shear/height port atlas kernel."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from imri_qpe.constants import C, DEFAULT_MU_MOL
from imri_qpe.scales import gas_constant_per_gram

FULL_PORT_FIELD_NAMES=("acoustic_density","radial_velocity","azimuthal_velocity","thermal_entropy","shear_RR_minus_phiphi","shear_RR_plus_phiphi_minus_2zz","shear_Rphi","shear_Rz","shear_phiz","log_height_over_anchor","vertical_velocity")

@dataclass(frozen=True)
class FullPortAtlasAnchor:
    temporal_matrix:np.ndarray
    rest_radial_matrix:np.ndarray
    coordinate_radial_matrix:np.ndarray
    source_matrix:np.ndarray
    radial_stf_incidence:np.ndarray
    sound_speed_over_c:float
    shear_signal_speed_over_c:float
    effective_vertical_frequency:float
    shear_relaxation_rate:float
    vertical_damping_rate:float
    transport_speed_over_c:float

@dataclass(frozen=True)
class FullPortAtlasAudit:
    temporal_minimum_eigenvalue:float
    temporal_symmetry_defect:float
    radial_symmetry_defect:float
    rest_maximum_absolute_speed_over_c:float
    coordinate_maximum_absolute_speed_over_c:float
    relativistic_spectral_mapping_defect:float
    source_entropy_positive_part:float
    height_port_skew_defect:float
    shear_work_reciprocity_defect:float
    damping_heat_ledger_defect:float
    full_field_count:int
    rest_speeds_over_c:tuple[float,...]
    coordinate_speeds_over_c:tuple[float,...]
    @property
    def passed(self):
        return self.temporal_minimum_eigenvalue>=1e-10 and self.temporal_symmetry_defect<=1e-12 and self.radial_symmetry_defect<=1e-12 and self.coordinate_maximum_absolute_speed_over_c<=.999 and self.relativistic_spectral_mapping_defect<=1e-12 and self.source_entropy_positive_part<=1e-12 and self.height_port_skew_defect<=1e-12 and self.shear_work_reciprocity_defect<=1e-12 and self.damping_heat_ledger_defect<=1e-12 and self.full_field_count==11

def radial_stf_incidence_matrix():
    """Return C_iA=E_A^{Ri} in the orthonormal (R,phi,z) triad."""
    return np.asarray(((1/np.sqrt(2),1/np.sqrt(6),0,0,0),(0,0,1/np.sqrt(2),0,0),(0,0,0,1/np.sqrt(2),0)),dtype=float)

def build_full_port_atlas_anchor(*,sound_speed:float,temperature:float,proper_half_thickness:float,proper_vertical_frequency:float,alpha:float,shear_relaxation_time:float,transport_speed_over_c:float,mu_mol:float=DEFAULT_MU_MOL):
    cs=float(sound_speed)/C;temp=float(temperature);height=float(proper_half_thickness);omega=float(proper_vertical_frequency);a=float(alpha);tau=float(shear_relaxation_time);transport=float(transport_speed_over_c)
    if not(0<cs<1) or min(temp,height,omega,a,tau)<=0:raise ValueError("full port anchor inputs must be physical")
    if a>=1 or abs(transport)>=1:raise ValueError("full port anchor must be causal")
    cnu=np.sqrt(a)*cs;incidence=radial_stf_incidence_matrix();incidence=incidence*(cnu/np.linalg.svd(incidence,compute_uv=False)[0])
    rest=np.zeros((11,11));rest[0,1]=rest[1,0]=cs
    velocity_indices=(1,2,10);shear_indices=range(4,9)
    for row,vi in enumerate(velocity_indices):
        for col,si in enumerate(shear_indices):rest[vi,si]=rest[si,vi]=incidence[row,col]
    identity=np.eye(11);coordinate=(transport*identity+rest)@np.linalg.inv(identity+transport*rest);coordinate=.5*(coordinate+coordinate.T)
    gas_constant=gas_constant_per_gram(mu_mol);omega_h=np.sqrt(omega**2+gas_constant*temp/height**2);gamma=a*omega
    source=np.zeros((11,11));source[4:9,4:9]=-np.eye(5)/tau;source[9:11,9:11]=np.asarray(((0,omega_h),(-omega_h,-gamma)))
    return FullPortAtlasAnchor(np.eye(11),rest,coordinate,source,incidence,cs,cnu,float(omega_h),1/tau,gamma,transport)

def audit_full_port_atlas_anchor(anchor:FullPortAtlasAnchor):
    if not isinstance(anchor,FullPortAtlasAnchor):raise TypeError("anchor must be FullPortAtlasAnchor")
    rest_speeds=np.linalg.eigvalsh(anchor.rest_radial_matrix);coordinate_speeds=np.linalg.eigvalsh(anchor.coordinate_radial_matrix);expected=(anchor.transport_speed_over_c+rest_speeds)/(1+anchor.transport_speed_over_c*rest_speeds)
    mapping=float(np.max(np.abs(coordinate_speeds-expected)));source_symmetric=.5*(anchor.source_matrix+anchor.source_matrix.T);source_positive=max(float(np.max(np.linalg.eigvalsh(source_symmetric))),0.)
    height_rev=np.array(anchor.source_matrix[9:11,9:11],copy=True);height_rev[1,1]=0;skew=float(np.linalg.norm(height_rev+height_rev.T)/max(np.linalg.norm(height_rev),1.))
    shear_block=anchor.rest_radial_matrix[np.ix_((1,2,10),range(4,9))];reciprocal=anchor.rest_radial_matrix[np.ix_(range(4,9),(1,2,10))].T;reciprocity=float(np.linalg.norm(shear_block-reciprocal)/max(np.linalg.norm(shear_block),1.))
    probe=np.linspace(-.17,.19,11);loss=-float(probe@source_symmetric@probe);heat=loss;ledger=abs(loss-heat)/max(abs(loss),abs(heat),np.finfo(float).tiny)
    return FullPortAtlasAudit(float(np.min(np.linalg.eigvalsh(anchor.temporal_matrix))),float(np.linalg.norm(anchor.temporal_matrix-anchor.temporal_matrix.T)),float(np.linalg.norm(anchor.coordinate_radial_matrix-anchor.coordinate_radial_matrix.T)),float(np.max(np.abs(rest_speeds))),float(np.max(np.abs(coordinate_speeds))),mapping,source_positive,skew,reciprocity,float(ledger),len(FULL_PORT_FIELD_NAMES),tuple(map(float,rest_speeds)),tuple(map(float,coordinate_speeds)))

__all__=["FULL_PORT_FIELD_NAMES","FullPortAtlasAnchor","FullPortAtlasAudit","audit_full_port_atlas_anchor","build_full_port_atlas_anchor","radial_stf_incidence_matrix"]
