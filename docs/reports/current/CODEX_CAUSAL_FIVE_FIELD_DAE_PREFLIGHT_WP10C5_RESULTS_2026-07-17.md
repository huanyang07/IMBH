# Causal Five-Field DAE Preflight WP10c5 Results

Date: 2026-07-17

## Verdict

The bounded WP10c5 count and local characteristic preflight passes, but the
production stationary-root gate remains closed.

The causal stress adds one evolved field to the four Killing conservation
laws. With independent conserved states, primitive states, and face fluxes,
the exact flux-primary count is

```text
conserved unknowns       5 N
primitive unknowns       5 N
face-flux unknowns       5 (N + 1)
total unknowns           15 N + 5

conservation rows        5 N
primitive-map rows       5 N
interior face rows       5 (N - 1)
inner face rows          5
outer face rows          5
total rows               15 N + 5
```

The resolved shear-gradient term belongs inside the fifth conservation row.
It adds neither an unknown nor a separate algebraic row.

| Mesh | Unknowns | Rows | Square |
|---:|---:|---:|---:|
| 16 | 245 | 245 | yes |
| 64 | 965 | 965 | yes |
| 96 | 1445 | 1445 | yes |

This count supersedes the four-field `12N+4` base count reported in WP10c4.

## Resolved Covariant Shear

The accepted Maxwell-Cattaneo closure cannot be assembled by merely advecting
`D chi`. WP10c3a already showed that this pressure-amplitude-only control has
a finite-difference-stable complex characteristic pair.

WP10c5 now defines the signed rest-frame shear rate

```text
q = -2 c e_(R)^mu e_(phi)^nu sigma_mu_nu,
```

assuming a stationary, axisymmetric radial profile. The implementation uses
the full Kerr-Schild connection and the supplied radial derivative
`d u_mu/dR`.

In the Newtonian circular limit this becomes

```text
q -> -R dOmega/dR.
```

The measured relative defects against `3 Omega/2` are:

| Radius | Relative defect |
|---:|---:|
| `1000 rg` | `1.0030e-3` |
| `10000 rg` | `1.0003e-4` |
| `100000 rg` | `1.0000e-5` |

The first-order weak-field convergence is the expected Schwarzschild
correction, not a fitted normalization.

## Five-Field Principal Matrix

The local primitive order is

```text
(ln Sigma, beta_R, beta_phi, ln T, chi).
```

The responsive-height acoustic block uses the physical adiabat already
certified in WP10c3b. The shear block is

```text
M_shear = [h 0; 0 1]
K_shear = [0 1; h c_nu^2/c^2 0],
```

so the local-rest shear modes are exactly `+/-c_nu`.

The full frozen local-rest spectrum is

```text
(-a, -c_nu, 0, +c_nu, +a).
```

Across the bounded inner, representative, and outer fixtures:

```text
maximum eigenvalue defect     2.68e-17
maximum imaginary component   0
maximum light-cone excess     0
```

This is a responsive-height five-field principal audit. It is not yet the
Jacobian of the complete transformed finite-volume residual.

## Boundary Rank

At the inner edge `R=1.8 rg`, all five coordinate characteristics point out
of the numerical domain:

```text
incoming inner modes = 0
```

No physical inner boundary row is allowed.

At the coordinate-stationary outer reservoir edge `R=335 rg`, the mode count
is:

```text
incoming acoustic modes = 1
incoming shear modes    = 1
zero/contact modes      = 1
outgoing modes          = 2
```

The outer physical contract therefore requires exactly two independent
conditions:

1. the Hill/Roche acoustic/nozzle condition;
2. zero outer shear stress.

After scaling the acoustic response by `a` and the shear response by the
characteristic impedance `h c_nu`, the `2 x 2` incoming response has:

```text
rank                    2
smallest singular value 1.0
```

The five algebraic outer face rows retain rank five with respect to the five
face-flux unknowns. The physical boundary-condition count is two; it must not
be confused with the algebraic face-row count.

The coordinate-stationary contact speed is zero at the outer fixture, so the
local stationary coordinate-flux rank is four. This is expected and must be
handled by the lower-order source and boundary structure in the assembled
stationary residual.

## Temporal Height Work

The trapezoidal responsive-height work

```text
Delta W_H = 0.5 (Pi_old + Pi_new) Delta ln H
```

is now transformed into all four Killing storage components. For a moving
column, the finite increment is the negative of the integrated isotropic
comoving four-force. At rest in flat spacetime it reduces to

```text
Delta E_K = Delta W_H / c^2.
```

The moving-fixture componentwise identity closes at

```text
2.74e-16.
```

This is the temporal mass-matrix contract required by backward Euler.

## Why Roots Were Not Run

The predeclared rule was to attempt N64/N96 stationary roots only after the
complete nonlinear count and rank audit passed. Three assembly items remain:

1. insert the covariant shear gradient into a declared path-conservative
   fifth finite-volume row;
2. insert the temporal Killing-storage increment into the complete
   backward-Euler primitive map;
3. extend the migrated four-component Roche provider to a five-component
   face contract with zero shear stress.

The present local count and mode audit does not prove that the Jacobian of
that assembled residual has full rank. Running roots now would silently
revert to the rejected advected-stress system or omit responsive vertical
work.

Therefore:

```text
N64 stationary root       not authorized
N96 stationary root       not authorized
tiny implicit step        not authorized
```

No residual tolerance, damping scan, or alternate splice was attempted.

## Locked Next Step

Proceed once to WP10c5b:

1. define one path-conservative face/cell discretization for the covariant
   shear-gradient term;
2. assemble the complete five-field primitive map, Killing sources, cooling,
   radial and temporal height work, exact stream moments, and five-component
   Roche face;
3. differentiate that exact residual at N16 and report full rank, smallest
   singular values, and inner/outer characteristic response;
4. if and only if the assembled N16 rank gate passes, chain N64 then N96
   low-throughput stationary roots;
5. attempt one tiny backward-Euler step only after both roots pass.

If the assembled residual is rank deficient because of the zero outer contact
mode or the nonconservative shear path, stop and repair that declared operator.
Do not replace it with an instantaneous `alpha Pi` stress, another inner/outer
splice, distributed tide, wind, or long evolution.

## Classification

```text
numerical status:
    supported local/count preflight

physical status:
    diagnostic only

production status:
    blocked before roots
```

## Verification

```text
focused causal DAE/stress/thermal tests   30 passed
complete repository suite                459 passed, 4 subtests passed
repository hygiene                       passed at 605 tracked files
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_dae_preflight_wp10c5.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_causal_inner_dae_preflight_wp10c5.py
```
