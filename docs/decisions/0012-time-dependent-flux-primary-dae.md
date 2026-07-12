# ADR 0012: Flux-primary time-dependent DAE

## Status

Accepted for the first no-tide production implementation.

## Context

The coupled open-overflow root is supported on `96/64` and `144/96`, but its
controlled `168/112` refinement fails at the outer endpoint. A single
zero-torque asymptotic remap reduced the seed and solved endpoint defects but
did not recover an accepted root. Further steady open-boundary tuning is
closed.

The existing non-Keplerian reservoir reconstructs torque from transported
mass and angular-momentum fluxes,

\[
G_f=\dot M_f l_f-{\cal J}_f,
\]

and closes this reconstructed torque against the common alpha stress at cell
centers. The open edge imposes `G_out=0` on the same mixed-flux operator.

## Decision

Use a semi-explicit, low-radial-Mach, flux-primary DAE.

Differential outer variables are

```text
ln Sigma, ln T, ln Omega                         3 No
```

Algebraic variables are

```text
inner transonic state                           2 Ni + 2
signed face Mdot                                No + 1
signed face angular flux J                      No + 1
interface total-energy flux                     1
```

The complete coupled state and residual size is

\[
\boxed{2N_i+5N_o+5}.
\]

The outer differential rows are finite-volume mass, angular-momentum, and
column-total-energy conservation. The algebraic rows are the inner transonic
core, common stress, quasi-static radial force, interface `Sigma/T`
continuity, interface angular/energy extraction, and the open-edge condition

\[
\dot M_{\rm out}l_{\rm out}-{\cal J}_{\rm out}=0.
\]

The interface mass flux and angular flux are shared variables. No duplicate
continuity rows are permitted.

Temporal radial kinetic-energy storage is omitted in the first architecture.
The architecture is valid only while the evolved outer domain satisfies

```text
maximum radial Mach number <= 0.1
```

The energy derivative includes temporal one-zone vertical work. Accepted
steps must satisfy independent global ledgers, not only scaled cell residuals.

## Evidence

Small physical prototypes at `No=8,12,16` give:

```text
storage rank                                  3 No
equilibrated algebraic rank                   2 No + 2
equilibrated full descriptor rank             5 No + 2
maximum radial Mach number                    0.0068-0.0091
maximum initialized common-stress defect      <= 1.5e-14
relative outer-torque defect                  <= 3.1e-15
```

The equilibrated full-system condition estimate is approximately
`6.4e4-3.0e5`, materially better than the thermodynamic boundary-eliminated
fallback.

Downsampled canonical open states at `No=16,32` complete accepted
backward-Euler steps at `dt/t_load=1e-6` with maximum residuals below
`3e-10` and relative mass, angular, and energy ledger defects below `9e-11`.

The first direct inner-transonic coupling uses the same count and adds the
checked-in absolute stream moments plus radiative cooling. At `Ni/No=16/8`
and `24/16`, `dt/t_load=1e-9` steps are accepted with maximum residual below
`3.2e-8`; interface continuity/extraction is below `5.8e-9`, and all three
independent global ledger defects are below `9.0e-9`.

The production prototype uses a structurally audited colored sparsity pattern
and regularized LSMR trust-region subproblems. A three-level
full-step/two-half-step comparison resolves an outer-state change of `7.67e-8`;
its outer-state differences decrease from `2.32e-12` to `8.14e-13` and
`2.36e-13`. Timestep convergence is supported over this range, while an exact
formal order is not claimed at the nonlinear floor.

Eight `16/8` repeated steps pass at `dt/t_load=1.25e-8`, and a step-four
restart is bitwise reproducible. A steady homotopy reaches the full
cross-interface first radial stencil with a residual of `3.33e-9`. The
`24/16` mesh still fails on its third subcycled step at `1.0466e-7`, now
primarily in interface continuity and flux extraction despite closed global
ledgers. ADR acceptance therefore covers the architecture and coarse repeated
control, not fine-mesh long evolution or physical tide.

The final boundary-eliminated variant made interface `Sigma,T` continuous to
roundoff but failed its first `24/16` subcycled step at `1.271e-7` in the inner
transonic core. It was reverted. No additional splice preconditioner or
interface variable elimination is authorized under this ADR.

## Rejected alternatives

### Thermodynamic face torque

Treating `G_out` as a constraint only on differential thermodynamic variables
requires boundary elimination or an index-two constrained DAE. Both variants
remain tested fallbacks but change the repository's certified mixed-flux
operator and are not selected initially.

### Steady angular-flux template

A fixed or shifted steady angular-flux profile cannot represent
time-dependent cell angular-momentum storage and is prohibited.

### Hidden boundary clipping

Unconfigured outer inflow, negative inner accretion, or non-positive states
must reject a step or activate a separately counted physical boundary model.

## Consequences

1. The production Jacobian must include both face-flux blocks.
2. Dense finite differences are prototype-only; production requires sparse
   analytic, automatic, or colored derivatives.
3. Backward Euler is the first integrator. BDF2 or IMEX follows only after
   mesh and timestep convergence.
4. Wind and physical tide remain off until the no-tide open control passes.
5. Failure of the low-Mach or algebraic-rank gate promotes radial dynamics;
   it does not authorize tolerance relaxation.
