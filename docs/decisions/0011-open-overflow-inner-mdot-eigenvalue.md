# ADR 0011: Open Overflow With An Inner-Mdot Eigenvalue

- Status: accepted
- Date: 2026-07-11

## Context

The corrected coupled minidisk ends at `335 rg`, inside the fiducial Hill
radius `746.90 rg`. Its ideal zero-mass-flux wall is numerically mesh supported.
When the wall torque is paired with binary pattern-speed power, however, the
tidal band exceeds `H/R=0.3` at the first nonzero power stage. Perfect
confinement is therefore outside the declared one-zone validity regime.

The next solve must permit mass to leave the outer edge. Holding
`Mdot_inner=5 Mdot_Edd` while injecting an absolute `5 Mdot_Edd` stream would
force zero outer mass flux, so the inner rate must become an outcome.

## Decision

Add one signed physical scalar to the coupled state:

```text
m = log(Mdot_inner/Mdot_Edd).
```

The first implementation keeps `Mdot_inner>0` but permits either sign of the
outer face flux. The stream supply and its `(S_M,S_J,S_E)` moments remain
absolute and fixed.

For every nonlinear trial, integrate candidate face fluxes outward from the
inner interface without enforcing either outer wall condition:

```text
Mdot_out = Mdot_inner - sum(S_M)
J_out    = J_inner - sum(S_J + T_ext)
G_out    = Mdot_out l_out - J_out.
```

The existing inner transonic equations are evaluated with the trial
`Mdot_inner`. The interface definitions remain

```text
J_I   = Mdot_inner l - G
F_E,I = Mdot_inner B - Omega G.
```

Add exactly one outer boundary row. Continue a boundary parameter `chi` from
the certified mass wall to the open zero-torque edge:

```text
R_edge(chi) = (1-chi) Mdot_out/Mdot_stream
            + chi G_out/G_scale,
0 <= chi <= 1.
```

At `chi=0`, the augmented system recovers the ideal-wall root and determines
`Mdot_inner=Mdot_stream`. At `chi=1`, it imposes `G_out=0`; the inner rate and
signed overflow are solved rather than prescribed.

## Degree Of Freedom Count

The current coupled system has

```text
2 N_inner + 3 N_outer + 4
```

unknowns and residuals. Adding `m` and `R_edge` gives

```text
2 N_inner + 3 N_outer + 5
```

of each. No other boundary or primitive row is added. Interface continuity
remains hard in `log Sigma` and `log T`; pressure, rotation, scale height, and
radial velocity remain audits.

## Locked Numerical Sequence

1. Reproduce the corrected `Rout=335 rg`, `192/128`, `chi=0` root with the
   augmented state.
2. Verify the augmented scaled Jacobian is full rank and the new weakest mode
   is not an outer-stencil artifact.
3. Continue `chi=0 -> 1` at `96/64`, then prolong the full open root through
   `144/96 -> 192/128`.
4. Repeat only at `R_I=35,40,50 rg` after the `40 rg` mesh gate passes.
5. Add a distributed tidal torque and paired pattern power only by continuing
   from the open root.

No wind, stability calculation, time evolution, fitted stress factor,
projection, clipping, or damping grid is part of this work package.

## Acceptance Gates

```text
maximum normalized residual             <= 1e-7
scaled Jacobian                          full column rank
interface and sonic responses            rank 2
mass, angular, and total-energy defects  <= 1e-9 relative
outer viscous torque at chi=1            <= 1e-8 relative
interface primitive audit                <= 0.01
N144/96 to N192/128 luminosity shift     <= 1%
N144/96 to N192/128 H/R shift            <= 2%
Hill-band H/R                            < 0.3 for physical promotion
```

## Stop Conditions

Move to coupled time evolution, rather than another steady boundary variant,
if the open root cannot be prolonged to `192/128`, if the inner rate tends to
zero, if no positive-Sigma state exists, or if the Hill band remains outside
the thickness/optical-depth validity gates. Wind remains last.
