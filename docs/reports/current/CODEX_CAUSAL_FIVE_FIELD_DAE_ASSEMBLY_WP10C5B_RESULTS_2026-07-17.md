# Causal Five-Field DAE Assembly WP10c5b Results

Date: 2026-07-17

## Verdict

The complete five-field Kerr-Schild finite-volume residual is now assembled.
Its time-dependent descriptor structure passes, but its stationary `N=16`
Jacobian misses the locked rank gate by one direction. The required
`N=64/N=96` roots and tiny implicit step are therefore not authorized.

This is a bounded stop, not a nonlinear-solver failure:

```text
assembled unknowns/rows          245 / 245
descriptor mass rank             80 / 80
backward-Euler rank, 0.1 s      245 / 245
backward-Euler rank, 1.0 s      245 / 245
backward-Euler rank, 10 s       245 / 245
stationary rank                 244 / 245
```

No root solve, damping scan, tolerance relaxation, distributed tide, wind, or
long evolution was attempted.

## Assembled State

The five primitives in each cell are

```text
(ln Sigma, beta_R, beta_phi, ln T, chi),
```

where `chi` is the signed rest-frame specific shear stress. The five
conserved fields are

```text
(D, S_R, S_phi, E_K, D chi),
```

including the stress-tensor increments in the four Killing fields.

The flux-primary state contains

```text
5N cell conserved states
5N cell primitive states
5(N+1) proper-measure weighted face fluxes
```

and the residual contains the matching conservation, primitive-map,
interior-face, excision-face, and Roche-face rows. The exact count remains

```text
15N + 5.
```

## Discrete Contract

### Interior faces

Each interior face uses one five-field local-Lax-Friedrichs/Rusanov flux. Its
maximum coordinate speed includes both the responsive acoustic cone and the
causal shear cone.

### Nonconservative shear path

The accepted path is declared explicitly:

1. reconstruct the lower four-velocity to faces by the straight arithmetic
   path between adjacent cell states;
2. differentiate that face path across each cell;
3. evaluate the full Kerr-Schild covariant rest-frame shear at the cell
   state;
4. insert the resulting Maxwell-Cattaneo target in the fifth conservation
   source.

The same straight face path is used for `ln H` in radial vertical work.

### Sources

The first four rows contain:

- perfect-fluid and viscous-tensor radial geometric sources;
- comoving diffusion cooling;
- radial responsive-height work;
- optional exact cell-integrated stream mass, radial momentum, angular
  momentum, and Killing energy.

The fifth row contains the resolved stress-relaxation source. Stress work
remains in the stress tensor flux and is not added again as heat.

### Time storage

Backward Euler uses the full Killing storage map. The finite

```text
0.5 (Pi_old + Pi_new) Delta ln H
```

work is added to radial momentum, angular momentum, and Killing energy
storage through the comoving four-force transform. It is not included again
as radial work.

### Boundaries

The inner edge is at `1.8 rg` and uses a one-sided physical flux with no
physical boundary condition. The outer face uses the existing Hill/Roche
acoustic provider for the first four fluxes and imposes zero specific shear
stress as the fifth physical response.

The rank seed is displaced by `1e-5 c` into the outward-contact regime. This
keeps central finite differences on one side of the zero-speed contact
without changing the boundary contract.

## Ledger Tests

The flux-consistent seed closes exactly:

```text
maximum primitive-map residual       0
maximum face-map residual            0
conservation telescoping defect       1.78e-16
minimum scattering optical depth      1.70e4
outer incoming characteristics        2
```

Unit tests also verify that exact stream moments enter only the four Killing
rows and that temporal height work enters all non-mass Killing storage
components.

## Stationary Rank

The same physical scaling and a relative singular-value threshold of `1e-11`
were used for three central-difference steps:

| FD step | Rank | Smallest singular value | Condition estimate |
|---:|---:|---:|---:|
| `1e-6` | `244/245` | `6.0891e-10` | `3.6676e11` |
| `2e-6` | `244/245` | `6.0909e-10` | `3.6666e11` |
| `5e-6` | `244/245` | `6.0909e-10` | `3.6666e11` |

The weak response is therefore finite-difference stable.

Its right singular vector is concentrated in the outermost cell:

```text
outermost primitive-cell fraction       0.997424
lnT primitive norm                       0.196137
specific-stress primitive norm           0.072283
```

The outer physical face row has negligible weight in the corresponding left
singular vector. The mode instead couples conservation, interior transport,
and the excision face. This is not an omitted fifth Roche row; it is a global
stationary thermal/stress response anchored near the closed outer cell.

## Time-Dependent Rank

The transformed storage contribution has exactly the expected rank:

```text
descriptor rank = 5N = 80.
```

The full backward-Euler Jacobian is rank `245/245` at all tested timesteps:

| Timestep | Smallest singular value | Condition estimate |
|---:|---:|---:|
| `0.1 s` | `3.4429e-7` | `1.2247e10` |
| `1.0 s` | `4.4341e-8` | `9.5093e9` |
| `10 s` | `5.0800e-8` | `4.3989e9` |

This supports the descriptor/storage architecture. It does not override the
locked rule that a timestep follows only mesh-supported stationary roots.

## Gate Decision

```text
stationary N16 rank             FAIL
descriptor storage rank        PASS
backward-Euler rank             PASS
N64 stationary root            NOT AUTHORIZED
N96 stationary root            NOT AUTHORIZED
tiny implicit step              NOT AUTHORIZED
```

## Locked Next Step

Perform one bounded stationary null-mode audit:

1. analytically eliminate the independent conserved and face-flux identity
   blocks to form the reduced `5N x 5N` primitive stationary response;
2. report the outer thermal/stress Schur response and its dependence on the
   closed-versus-open Roche gate without changing the provider;
3. distinguish an actual duplicate/constraint omission from a physically
   weak but nonzero thermal relaxation direction;
4. if and only if an exact redundant row or missing physical condition is
   demonstrated, repair that declared operator and repeat `N=16`;
5. otherwise retain the assembled DAE and move to a time-dependent
   consistent-initial-data strategy under a separately approved gate.

Do not promote the rank from `244` to `245` by lowering the threshold, adding
an arbitrary temperature condition, or relaxing the Roche characteristic
contract.

## Verification

```text
PYTHONPATH=src python3 -m pytest -q \
  tests/test_causal_inner_dae.py \
  tests/test_causal_inner_dae_system.py

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_dae_assembly_wp10c5b.py
```

Machine-readable output:

```text
outputs/tables/causal_five_field_dae_assembly_wp10c5b.json
```
