"""Algebraic no-go audit for a restricted moving-five-STF scalar potential."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_eleven_field_convex import FullShearRestFrame


@dataclass(frozen=True)
class RestrictedFiveSTFPotentialAudit:
    """Constraint identities and the coefficient-independent linear-map defect."""

    maximum_beta_transversality_defect: float
    maximum_nu_basis_derivative: float
    I2_gram_relative_defect: float
    maximum_first_invariant_derivative_at_origin: float
    desired_linear_stress_map_norm: float
    candidate_linear_stress_map_norm: float
    linear_stress_map_relative_defect: float

    @property
    def candidate_viable(self) -> bool:
        return (
            self.maximum_beta_transversality_defect <= 2.0e-13
            and self.I2_gram_relative_defect <= 2.0e-13
            and self.linear_stress_map_relative_defect <= 1.0e-10
        )


def audit_restricted_five_stf_scalar_potential(
    frame: FullShearRestFrame,
    *,
    temperature: float,
) -> RestrictedFiveSTFPotentialAudit:
    """Prove that the restricted invariant ansatz has no linear stress term.

    Holding the five amplitudes fixed while differentiating the moving basis
    preserves ``beta_mu E_A^{mu nu}(beta)=0`` identically.  Consequently
    ``nu`` vanishes as a function, while ``I2``, ``I3`` and ``nu**2`` have no
    first amplitude derivative at the origin.  Every smooth scalar built from
    those invariants therefore predicts a zero linear shear-stress map.
    """

    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be positive and finite")
    beta_covector = frame.metric @ frame.four_velocity / temp
    transversality = np.einsum("i,aij->aj", beta_covector, frame.stf_basis)
    nu_derivatives = np.einsum(
        "i,aij,j->a", beta_covector, frame.stf_basis, beta_covector
    )
    lowered = np.einsum(
        "ik,akl,lj->aij", frame.metric, frame.stf_basis, frame.metric
    )
    gram = np.einsum("aij,bij->ab", lowered, frame.stf_basis)
    # At a_A=0: dI2/da_A=0, dI3/da_A=0 and d(nu**2)/da_A=0.
    invariant_first_derivatives = np.concatenate(
        (nu_derivatives, np.zeros(15, dtype=float))
    )
    desired = np.asarray(frame.stf_basis, dtype=float)
    candidate = np.zeros_like(desired)
    desired_norm = float(np.linalg.norm(desired))
    candidate_norm = float(np.linalg.norm(candidate))
    relative = float(np.linalg.norm(candidate - desired) / desired_norm)
    return RestrictedFiveSTFPotentialAudit(
        maximum_beta_transversality_defect=float(np.max(np.abs(transversality))),
        maximum_nu_basis_derivative=float(np.max(np.abs(nu_derivatives))),
        I2_gram_relative_defect=float(
            np.linalg.norm(gram - np.eye(5)) / np.linalg.norm(np.eye(5))
        ),
        maximum_first_invariant_derivative_at_origin=float(
            np.max(np.abs(invariant_first_derivatives))
        ),
        desired_linear_stress_map_norm=desired_norm,
        candidate_linear_stress_map_norm=candidate_norm,
        linear_stress_map_relative_defect=relative,
    )


__all__ = [
    "RestrictedFiveSTFPotentialAudit",
    "audit_restricted_five_stf_scalar_potential",
]
