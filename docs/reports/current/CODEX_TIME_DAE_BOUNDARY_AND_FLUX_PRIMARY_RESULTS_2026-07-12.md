# Time-dependent DAE boundary and flux-primary results

## Scope

This work closes the open-edge preflight, selects the first time-dependent
outer architecture, and implements its first direct inner-transonic coupling.
The coupled prototype includes the absolute stream moments and radiative
cooling. It certifies a bounded timestep-convergence sequence and coarse-mesh
restart, but not evolved-mesh convergence, long evolution, tide, or wind.

## Bounded steady endpoint audit

The authorized zero-torque remap transported

\[
G/(R_{\rm out}-R)
\]

plus gas-pressure fraction, then recovered `Sigma,T` through the vertical
closure.

At `168/112`:

| Quantity | Old primitive seed | Asymptotic seed | Final Newton state |
|---|---:|---:|---:|
| Maximum residual | 1.5324 | 1.1652 | 0.3620 |
| Outer stress maximum | 1.5324 | 1.1652 | 0.3620 |
| Outer energy maximum | 0.7485 | 0.7745 | 0.2263 |
| Outer radial maximum | 0.01264 | 0.00515 | 0.00159 |

Newton ended with a line-search failure. The endpoint-aware retry therefore
does not mesh-certify the steady open branch. No second steady remedy is
authorized.

## Boundary architecture comparison

The thermodynamic boundary-eliminated and constrained prototypes become full
rank only after declared two-sided equilibration. The eliminated candidate is
regular, but its equilibrated full-system condition estimate is approximately
`1.4e6-5.6e6` over `No=8-16`.

The repository-compatible flux-primary operator promotes both

\[
\dot M_f,\qquad {\cal J}_f
\]

to algebraic face variables and reconstructs

\[
G_f=\dot M_fl_f-{\cal J}_f.
\]

Its small-mesh results are:

| `No` | Algebraic rank | Full rank | Equilibrated condition | Max Mach |
|---:|---:|---:|---:|---:|
| 8 | 18/18 | 42/42 | `6.35e4` | 0.00680 |
| 12 | 26/26 | 62/62 | `3.04e5` | 0.00828 |
| 16 | 34/34 | 82/82 | `1.47e5` | 0.00907 |

This is the selected index-one candidate.

## Backward-Euler prototype

Downsampled canonical open states were tested at `No=16,32`. At
`dt/t_load=1e-6` both meshes are accepted:

| `No` | Maximum residual | Mass defect | Angular defect | Energy defect |
|---:|---:|---:|---:|---:|
| 16 | `2.81e-10` | `5.03e-11` | `4.80e-11` | `5.57e-11` |
| 32 | `2.11e-10` | `8.00e-11` | `5.03e-11` | `8.66e-11` |

Very small trial steps are less reliable because full-storage scaling can
dominate the transported increment. Conservation rows now use per-step flux
scales with a declared storage floor, and independent global ledgers are part
of acceptance.

## Direct inner-transonic coupling

The production prototype uses exactly

\[
2N_i+5N_o+5
\]

unknowns and rows. The shared first face mass flux is passed directly to the
inner transonic parameters, so interface mass continuity is a variable
identity rather than an extra row. The complete face mass and angular-flux
profiles remain algebraic unknowns, and the interface total-energy flux is an
explicit signed scalar.

The outer backward-Euler ledger now includes the checked-in cell-integrated
stream mass, angular-momentum, and total-energy moments plus state-dependent
radiative cooling. The inner and outer domains impose two primitive interface
rows and extract angular and total-energy fluxes from the re-solved inner
profile.

At `dt/t_load=1e-9`:

| `Ni/No` | Unknowns | Max residual | Mass defect | Angular defect | Energy defect |
|---:|---:|---:|---:|---:|---:|
| 16/8 | 77 | `1.20e-8` | `2.81e-11` | `1.53e-10` | `8.90e-9` |
| 24/16 | 133 | `3.13e-8` | `8.89e-10` | `3.62e-10` | `7.12e-9` |

Both steps retain inward inner flow and outward open-edge flow. Interface,
flux-extraction, and edge residuals remain below `5.8e-9`.

The first sparse pattern omitted four derivatives from the radial endpoint
stencils. The production model now evaluates the first outer radial row with
the inner endpoint as a physical ghost state. A steady homotopy in the stencil
fraction

```text
0 -> 0.01 -> 0.02 -> 0.05 -> 0.1 -> 0.2 -> 0.4 -> 0.7 -> 1
```

reaches the full cross-interface stencil at `24/16` with a maximum steady
residual of `3.33e-9`. Numerical Jacobian audits at stencil fractions zero and
one find no derivative larger than `1e-10` of its row scale outside the sparse
pattern. The selected nonlinear solver is a colored central sparse
trust-region method with tightly solved regularized LSMR subproblems.

A resolved three-level temporal comparison uses
`dt/t_load=1e-7,5e-8,2.5e-8,1.25e-8`. All seven full and chained half-steps are
accepted. At the coarsest level the outer differential state changes by
`7.67e-8`. Full-step/two-half-step outer-state differences decrease through

```text
2.32e-12 -> 8.14e-13 -> 2.36e-13
```

with ratios `2.85` and `3.44`. The complete-state differences decrease by
factors `1.42` and `3.71`. This certifies timestep convergence over the tested
range, although no exact asymptotic order is claimed because the finest
differences approach the nonlinear floor. Long-duration tests remain required.

## Repeated steps and restart

The `16/8` control accepts eight consecutive steps at
`dt/t_load=1.25e-8`, reaching `t/t_load=1e-7` or `0.1421 s`. A complete restart
after step four reproduces both the stored state and the next implicit step
bit for bit.

Over the bounded run:

```text
max outer-q change                7.67e-8
max H/R                           0.12468007 -> 0.12468004
Lrad/LEdd                         0.159305104 -> 0.159305100
Mdot_inner/Mdot_supply            0.167466048 -> 0.167466029
```

All eight steps pass the fixed residual and ledger gates. The interval is a
restart/repeated-step certification, not a thermal-timescale stability run.

## Evolved-mesh stop gate

The `24/16` mesh does not pass the same evolved interval. Even after
subcycling at `dt/t_load=6.25e-9`, two steps are accepted and the third stops at
a maximum residual of `1.0466e-7`, only `4.7%` above the fixed gate. The
largest rows are interface continuity (`1.0466e-7`) and flux extraction
(`9.12e-8`). The cross-interface radial row is reduced to `4.88e-8`, so the
former one-sided radial boundary defect is no longer controlling. Mass,
angular-momentum, and energy ledger defects remain below `7.5e-9`.

One bounded interface-only seed correction did not reduce the residual and
was removed. No residual weighting, tolerance relaxation, timestep scan, or
accepted-state projection is retained.

## Boundary-eliminated stop test

The final authorized interface variant removed the two primitive continuity
rows and reconstructed the inner endpoint `Sigma,T` directly from the outer
edge. The square count changed from

\[
2N_i+5N_o+5
\]

to

\[
2N_i+5N_o+3.
\]

Continuity then closed to `7.8e-16`, but the first `24/16` subcycled step
failed at `1.271e-7` in the inner transonic core. Flux extraction was
`2.80e-8`, the first radial row was `3.24e-8`, and all global ledgers remained
below `4.4e-9`. The reduction therefore moved the conditioning defect into
the inner core and was worse than the retained square formulation. It was
reverted.

This is not evidence for physical instability. It is an evolved-interface
conditioning failure. The physical-tide stage remains blocked until a
fine-mesh repeated sequence passes without residual weighting, tolerance
relaxation, or another solver-parameter scan.

## Scientific status

The project now has:

```text
supported open steady reference, not mesh certified
closed steady endpoint investigation
full-rank flux-primary DAE architecture at small N
conservative outer-evolving, inner-quasi-steady backward-Euler prototype
accepted directly coupled stream-fed steps at Ni/No=16/8 and 24/16
```

It does not yet have a fine-mesh or long-duration physical evolution. The
declared interface remedies are now exhausted: the cross-interface radial
stencil works, while both an interface-only correction and algebraic primitive
elimination fail the evolved-mesh gate. Further splice conditioning is closed.

The next implementation must promote the inner response to a time-dependent
conservative domain, or move directly to one global signed conservative
transonic system evolving mass, angular momentum, total energy, and radial
momentum. Physical distributed tide remains blocked until that no-tide model
passes mesh and timestep convergence.
