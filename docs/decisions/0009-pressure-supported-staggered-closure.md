# ADR 0009: Do not promote the staggered pressure-supported closure

## Status

Accepted on 2026-07-11.

## Context

The Keplerian reservoir matches conserved fluxes but retains a mesh-stable
integrated-pressure discontinuity at the transonic interface. A pressure-
supported rotation update was added consistently to angular momentum, shear,
torque work, orbital energy, and alpha viscosity.

Raw pressure updates can reverse shear or angular-momentum gradients in the
interface cells. Trial profiles are therefore projected onto decreasing,
Rayleigh-stable rotation before entering the viscous solve, with the force-
balance mismatch retained as a diagnostic.

## Decision

The projected staggered closure remains an experimental diagnostic. It is not
the production reservoir model.

## Consequences

At `N=64`, continuation to full pressure support converges and reduces the
angular-frequency mismatch near `40 r_g` from about five percent to `0.37%`.
However, the integrated-pressure mismatch worsens to `0.356`, and the full
radial-force mismatch is `1.38%`. None of the `N=128` damping/smoothing cases
passes the existing fixed-point gate.

The next implementation must solve radial momentum, angular transport, and
total energy simultaneously, with rotation as a nonlinear unknown. More
staggered damping or a looser fixed-point tolerance is not an acceptable
production path.
