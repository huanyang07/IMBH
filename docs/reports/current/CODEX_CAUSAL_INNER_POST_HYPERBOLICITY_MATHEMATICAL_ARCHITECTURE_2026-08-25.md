# Post-hyperbolicity mathematical architecture

Date: 2026-08-25

Status: analysis and model-design recommendation only. This document is not
an execution manifest and does not authorize a new trajectory.

## Decision

The current five-field causal inner system must not be used as the truth
generator for a complete phase cycle or for fixed-\(Q\) averaging.

The authentic two-half-step experiment accepted the first 0.125 ms step and
rejected the second before exact free-field evaluation. The rejected endpoint
lost strong hyperbolicity at face 3. The accepted chain therefore stops at
72 endpoints and time

\[
t=0.18587500000000012\ {\rm s}.
\]

This is not evidence that the physical disk becomes complex. It is evidence
that the present five-field closure ceases to define a robust hyperbolic
initial-value problem in the state region reached by the numerical path.
Consequently, smaller timesteps, real-part projection of eigenvalues, added
numerical diffusion, or continued phase marching cannot certify the desired
slow model.

The replacement truth model should be an entropy-compatible,
symmetrizable-hyperbolic relaxation system in which vertical height and
vertical momentum are dynamical fields. The reduced slow solver should then
average a steady state or periodic orbit of that repaired fast system through
a fixed-\(Q\) boundary-value problem, rather than resolving every fast cycle
inside every slow step.

The existing machine-readable decision remains binding:

```text
classification = hyperbolicity_boundary_bracketed_after_first_half_step
authorized_next = null
complete_cycle_execution_authorized = false
reduced_slow_evolution_authorized = false
```

## Evidence fixed by the completed execution

The decisive result is in commit `3db75a53`; the post-run lifecycle validation
repair is in `a92fc70f`.

The first half-step passed every propagation gate:

- all 113 face pencils had real spectra;
- the raw characteristic condition number was `9273.243683269748`;
- the equilibrated metric condition number was `4.491395328234775`;
- the endpoint defect was `3.181631566130393e-4`;
- the reconstruction limiter was inactive;
- the maximum height ratio was `0.21346613166048903`;
- the minimum optical depth was `17.58552536931864`;
- checkpoint roundtrip and suffix-history replay were exact.

The second half-step passed retraction but failed the all-face principal gate:

- first failing face: `3`;
- maximum imaginary characteristic speed: `1.0169588429410766e-4`;
- maximum imaginary primitive-eigenvector component: `0.30711083786241344`;
- generalized eigenpair defect: `5.394521466065854e-16`;
- exact free-field calls after the failure: `0`;
- rejected candidates propagated: `0`.

The independent nonpropagating diagnostic also found the same complex pair
with finite-difference pencils as the perturbation was refined. The smallest
finite-difference-to-analytic imaginary-speed defect was about `6.08e-5`.
This rules out ordinary eigensolver noise and a simple analytic-Jacobian
implementation error.

## Mathematical diagnosis

For the exact five-field local maps, the principal pencil is

\[
\bigl(A(U)-\lambda T(U)\bigr)v=0,
\]

where

\[
T=U_q+T_H,
\qquad
A=F_q-S_{\rm shear}-S_H.
\]

Here `q` denotes the five primitive charts, `T_H` is the responsive-height
storage contribution, `S_shear` is the Maxwell--Cattaneo shear-gradient
contribution, and `S_H` is vertical work created by differentiating the
algebraic quasi-hydrostatic height closure.

The exploratory block decomposition at the failing face gives the important
structural fact:

- the physical flux pencil alone is real at both neighboring states;
- the shear contribution alone can create a complex pair;
- the responsive-height contribution alone can create a complex pair;
- at the accepted state their combined effects cancel enough to recover a
  real basis;
- at the rejected state that cancellation no longer closes and the full
  pencil is complex.

Along the straight accepted-to-rejected state chord, an exploratory bisection
put the real/complex transition near fraction `0.95157036`. On the real side,
the primitive eigenvector condition number grew from about `2.18e3` at the
accepted endpoint to about `7.35e6` at the boundary, while the minimum speed
gap fell to about `2.08e-7`. These chord values are diagnostic rather than a
canonical trajectory certificate, but they identify an exceptional point:
two modes coalesce and their eigenvectors lose a uniformly bounded basis
before becoming complex.

Percent-level rescaling of the shear and vertical contributions did not
restore a real pencil at the failed state. Arbitrary coefficient tuning is
therefore neither a mathematical repair nor a useful immediate diagnostic.

The simplified local-rest audit remains real because it does not contain the
full relativistic responsive-height and shear principal coupling. Its real
sound and viscous signal speeds do not certify the exact five-field pencil.

### Consequence

The failure is stronger than a CFL or accuracy problem. A smaller timestep
might generate a curved numerical path that delays or avoids this particular
candidate, but the unbounded eigenbasis conditioning already removes the
uniform strong-hyperbolicity margin needed for a reliable cycle calculation.
The current five-field system cannot be the foundation of an expensive
reduced-order atlas unless its principal structure is repaired first.

## Replacement fast truth model

### State

Use a seven-field state, schematically

\[
U=(\Sigma,m_r,L,E_{\rm tot},\Pi_{r\phi},H,P_H),
\]

or the corresponding primitive charts

\[
q=(\ln\Sigma,\beta_r,\beta_\phi,\ln T,\chi,\ln H,w_H).
\]

`H` is the column height and `P_H` (or `w_H`) is its conjugate vertical
momentum. `E_tot` must include the vertical kinetic and potential energy and
the thermodynamically required relaxation energy. The stress remains a causal
relaxation field; it is not returned to an instantaneous Navier--Stokes
closure.

### Balance-law form

The target system has the form

\[
A^0(U)\,\partial_t U+A^1(U)\,\partial_r U=S(U),
\]

with mass, radial momentum, angular momentum, and total energy written as
conservative balances. The new vertical pair should have a conservative or
entropy-compatible material form such as

\[
\partial_t(\Sigma H)+\partial_r(\Sigma u^r H)=\Sigma w_H,
\]

\[
\partial_t(\Sigma w_H)+\partial_r(\Sigma u^r w_H)
=\mathcal P_H(U)-\Sigma\Omega_H^2H-\gamma_H\Sigma w_H.
\]

The precise relativistic factors must be derived from the same column energy
and geometry as the other balances. These equations illustrate the required
architecture; they are not yet accepted production equations.

The shear relaxation equation should remain causal,

\[
\tau_\pi D_t\Pi_{r\phi}+\Pi_{r\phi}
=-2\eta\,\sigma_{r\phi},
\]

but its derivative coupling must be placed consistently in the full
principal matrix and derived together with the relaxation contribution to
the entropy. Algebraic elimination of `H(U)` and folding
`D log(H)[D_r U]` into the four energy-momentum rows is prohibited in the
new truth model.

### Structural certificate

The model is acceptable only if there is a convex entropy (or mathematical
energy) `eta(U)` and a positive symmetrizer `G(U)` such that, throughout a
prospectively frozen physical envelope,

\[
G A^0=(G A^0)^T>0,
\qquad
G A^1=(G A^1)^T,
\]

and the relaxation sources satisfy

\[
\nabla\eta(U)\cdot S_{\rm relax}(U)\le 0.
\]

This makes the characteristic problem self-adjoint in the energy metric and
keeps it real and diagonalizable even when physical characteristic speeds
coincide. A numerical scan of eigenvalues is an audit of this proof, not a
replacement for it.

The equilibrium limit must obey a full coupled subcharacteristic condition:
the characteristic speeds of the five-field hydrostatic/stress-equilibrium
limit must lie within the relaxation system's characteristic cone. The
vertical and stress relaxation times cannot be selected independently of
that condition.

## Reduced slow architecture after the truth model is repaired

Let `Q` denote the selected slow conserved quantities, initially expected to
include integrated mass, angular momentum, and energy. Let `z` contain the
remaining fast spatial state. At fixed `Q`, the repaired model defines

\[
\partial_t z=\mathcal G(Q,z),
\qquad C(z)=Q.
\]

The first task is to determine whether the attracting object is an
equilibrium or a limit cycle.

For an equilibrium, solve the constrained steady KKT system directly. For a
cycle, solve a periodic boundary-value problem

\[
\partial_\theta z_Q=T_Q\mathcal G(Q,z_Q),
\qquad z_Q(0)=z_Q(1),
\]

together with `C(z_Q)=Q` and one phase condition. Multiple shooting or a
collocation method should be used, with the certified causal time integrator
only as a segment propagator. Long transient marching is a fallback
diagnostic, not the production cell solver.

The averaged slow vector field is then

\[
\overline{F}(Q)=
\int_0^1 F_Q\bigl(Q,z_Q(\theta)\bigr)\,d\theta,
\]

and slow evolution is

\[
\frac{dQ}{dT}=\epsilon\,\overline F(Q).
\]

Construct an adaptive continuation atlas of `z_Q`, `T_Q`, and
`overline F(Q)`. Use tangent/adjoint sensitivities to predict neighboring
anchors and a held-out residual estimator to decide when another expensive
cell solve is needed. The complete slow cycle should evaluate the atlas and
occasionally correct it; it must not replay tens of thousands of fast steps
at every slow macro step.

### Cost architecture

The cost goal is separated into offline and online work:

- offline: construct and validate a sparse atlas of fixed-`Q` equilibria or
  cycles; target at most 72 wall hours on the declared hardware, with anchors
  independently parallelizable;
- online: integrate one full slow cycle from the validated atlas in at most
  24 wall hours, with a hard project requirement below 72 wall hours;
- correction: no more than a prospectively frozen small number of new cell
  solves during the online cycle.

This is the essential speedup. Cross-step Broyden reuse can make an individual
fast solve cheaper, but only invariant-object continuation plus an averaged
flux atlas removes the factor between a microsecond fast timestep and a
cycle-scale slow evolution.

## Prospective work packages

No package below is authorized by the current result. The next action is to
freeze a separate model-design manifest for Stage 1.

### Stage 1 -- derive the seven-field entropy system

1. Choose the total column energy including vertical and relaxation energy.
2. Derive all seven balances from that energy and the Kerr--Schild column
   geometry.
3. Derive `A0`, `A1`, the entropy variables, and the candidate symmetrizer.
4. Derive the equilibrium five-field limit and the coupled
   subcharacteristic inequalities.
5. Prohibit parameter fitting to the observed failing face.

Pass only if positive definiteness, symmetric principal form, entropy
dissipation, and the equilibrium limit are established symbolically or by a
computer-assisted identity with independently audited derivatives.

### Stage 2 -- local numerical structural audit

Implement only the local seven-field maps. Audit the primary and held-out
states, every committed trajectory face, the accepted boundary state, the
rejected face-3 candidate, and a prospectively frozen neighborhood around
them.

Require positive symmetrizer margin, real energy-metric spectra, bounded
metric conditioning, a positive subcharacteristic margin, independent
finite-difference pencils, and agreement of analytic and numerical entropy
identities. Stop without a spatial trajectory on any failure.

### Stage 3 -- relaxation-limit and short-trajectory equivalence

Before the old boundary, compare the seven-field model with the certified
five-field model at matched endpoints. Verify convergence toward
quasi-hydrostatic height and equilibrium stress as the relaxation parameters
are reduced while retaining the subcharacteristic margin. Preserve all
physical, storage, reaction, restart, and ledger gates.

Then execute only a bounded seven-field trajectory through the old face-3
region with all-face symmetrizer and hyperbolicity checks.

### Stage 4 -- identify the fixed-`Q` invariant object

At the primary and held-out `Q` anchors, determine whether the repaired fast
system approaches a steady state or a periodic orbit. Certify the selected
steady KKT or periodic multiple-shooting solve, its Floquet/stability margin,
restart, and spatial/temporal refinement. Do not construct an average from an
unattracting or nonunique orbit.

### Stage 5 -- build and validate the slow atlas

Construct two to three initial `Q` anchors, compute averaged physical fluxes,
derive tangent or adjoint sensitivities, and test interpolation on held-out
anchors. Compare the reduced prediction with matched segments of the repaired
truth model. Freeze an error and cost budget before adding anchors.

### Stage 6 -- complete-cycle authorization

Only after Stages 1--5 pass may a definitions-only complete-cycle manifest be
created. It must freeze the atlas, macro integrator, online correction rule,
cycle closure tolerance, conserved-ledger budget, held-out checks, and the
72-hour hard runtime ceiling.

## Prohibited shortcuts

- Do not propagate a candidate with complex characteristic speeds.
- Do not replace a complex pair by its real part.
- Do not infer hyperbolicity from the simplified local-rest audit.
- Do not treat smaller timesteps as a repair of a vanishing eigenbasis
  margin.
- Do not add diffusion merely to hide the principal defect.
- Do not tune shear or vertical coefficients to this one failing face.
- Do not average the current five-field trajectory or train a surrogate on
  it past the accepted boundary.
- Do not authorize a complete cycle until the repaired fast invariant object
  and its averaged flux map are independently held out and costed.

## Bottom line

The reduced-slow objective remains mathematically reasonable, but the current
five-field truth architecture is not. The durable path is:

\[
\boxed{\text{seven-field entropy/symmetric-hyperbolic relaxation truth}}
\;\longrightarrow\;
\boxed{\text{fixed-}Q\text{ steady/periodic boundary-value solve}}
\;\longrightarrow\;
\boxed{\text{averaged-flux continuation atlas}}
\;\longrightarrow\;
\boxed{\text{slow cycle}}.
\]

This architecture addresses both observed blockers: it removes the fragile
principal cancellation that caused the hyperbolicity boundary, and it avoids
paying the fast-step cost throughout an entire slow cycle.
