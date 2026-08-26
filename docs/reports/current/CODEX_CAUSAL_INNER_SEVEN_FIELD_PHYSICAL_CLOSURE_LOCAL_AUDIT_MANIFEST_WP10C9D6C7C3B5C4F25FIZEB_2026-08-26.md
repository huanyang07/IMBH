# Seven-field physical closure and local-audit manifest

Classification: `seven_field_physical_closure_local_structural_audit_manifest_frozen`.

This package is definitions-only. It freezes the physical Kerr--Schild closure obligations and the exact local audit envelope. It executes no eigenvalue campaign, nonlinear root, spatial step, or trajectory.

## Covariant state architecture

The seven primitives are `(ln Sigma, beta_R, beta_phi, ln T, chi, ln H, w_H/c)`. The rest-frame height contents `Sigma H` and `Sigma w_H` are densitized consistently with the mass current: the coordinate conserved variables are `Z_H=D H` and `P_H=D w_H`, where `D=Sigma W` is the Valencia rest-mass storage. Thus `H=Z_H/D` and `w_H=P_H/D`; height is not a redundant algebraic unknown.

The local rest-frame total energy contains gas+radiation internal energy, vertical kinetic and gravitational reservoirs, and a positive shear-relaxation reservoir. These reservoirs enter the relativistic inertia before the Kerr--Schild Killing projections are formed. The same stress-energy tensor supplies state, flux, and geometric source terms.

## Entropy/Godunov construction

Internal energy is recovered from total energy after subtracting all mechanical and relaxation reservoirs. The binding mathematical entropy is minus the recovered column entropy. Its exact Legendre state and flux potentials must generate both `U7` and `F7`; post-hoc symmetrization is forbidden. Shear and vertical damping transfer reservoir energy to heat, so internal relaxation has zero total-energy defect and non-positive mathematical-entropy production.

## Frozen evidence envelope

The package contains `8401` canonical five-field base charts and `47` deterministic witnesses. Inputs are the 20 ms primary profile, 16 ms held-out profile, all 72 accepted pre-boundary profiles, the rejected full-step profile, and the independently diagnosed failed face. Mutable scratch files are not used.

Empirical chart minimum: `[ 4.63798607e+00 -3.65584421e-01  4.43640328e-01  1.43210943e+01
  3.36305897e-06]`.

Empirical chart maximum: `[6.73918992e+00 1.34317092e-01 9.20497426e-01 1.54257571e+01
 5.12884997e-04]`.

Every base chart is audited at vertical equilibrium. Deterministic witnesses additionally use `ln(H/H_eq) in {-0.10,0,0.10}`, `w_H/c in {-0.03,0,0.03}`, sign-reflected shear amplitudes through 1.25 times the physical amplitude, and prospectively frozen axis perturbations. This is a discrete local stencil, not a claim over a continuous hyperrectangle.

## Shear and vertical closure

The alpha amplitude, reference shear, viscosity, and relaxation time retain the existing state-local calibration. The exact nonlinear one-shear Israel--Stewart causality and strong-hyperbolicity inequalities are binding in addition to the full seven-field spectrum. No coefficient or floor may be tuned to the failed face. The vertical balance is a covariant advected oscillator with `gamma_H=alpha Omega_perp`; damping is lower order and heats the internal reservoir.

## Compatibility boundary

On every certified pre-boundary equilibrium state, state/flux/source parity, compressed principal parity, and relaxation subcharacteristic interlacing are binding. At the failed face, exact principal parity is forbidden because it would force the repaired model to inherit the old complex spectrum. The old face is retained only as a negative control; the finite-inertia seven-field spectrum must be real and causal there.

## Decision

Authorized next: `WP10c9d6c7c3b5c4f25fizec_seven_field_physical_closure_local_structural_audit` only. It may implement the local physical closure and structural audit under the frozen gates. It may not construct a spatial discretization or advance a trajectory. The stopped five-field trajectory remains stopped, and fixed-Q evolution, slow-flux mapping, a complete cycle, and reduced slow evolution remain unauthorized.
