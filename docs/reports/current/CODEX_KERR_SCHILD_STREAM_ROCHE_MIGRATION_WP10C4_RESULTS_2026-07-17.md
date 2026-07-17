# Kerr-Schild stream/Roche migration WP10c4 results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** explicit vertical-frequency provider, one-state relativistic stream
moments, exact compact source cells, and closed/choked Hill/Roche face
migration. No stationary root, implicit timestep, long evolution, distributed
tide, or wind was run.

## Verdict

WP10c4 passes its bounded migration gate:

```text
maximum vertical-frequency slope defect          1.2378e-11
maximum covariant stream-moment defect            0
maximum exact source-moment defect                 2.0551e-16
maximum source-per-ct conversion defect            1.7931e-16
weak-field angular-momentum defect at 10000 rg     1.50034e-4
weak-field binding-energy defect at 10000 rg      -7.50338e-5
fiducial Roche opening temperature                 988232.828 K
maximum Roche J/E/pattern-power ledger defect       0
incoming outer characteristics                     1
outer face-row rank                                4 of 4
base DAE counts at N16/N64/N128                    square
```

The source and physical outer boundary now use the same Kerr-Schild Killing
energy convention as the one-domain finite-volume core. This is an adapter
and rank result, not a production disk solution.

## Explicit vertical-frequency provider

The responsive WP10c3b column now receives

```text
Omega_perp(R) = c sqrt(rg/R^3),
dlnOmega_perp/dlnR = -3/2
```

from an explicit provider. It is finite at and inside the Schwarzschild
horizon in the selected chart and recovers the weak-field orbital scale. It
is still a quasi-hydrostatic curvature prescription, not an evolved vertical
degree of freedom.

The audit samples `1.5`, `2`, `20`, `240`, and `335 rg`. The largest centered
logarithmic-slope defect is `1.24e-11`.

## One-state stream ledger

One local Eulerian stream four-state supplies

```text
p_R/c   = hbar u_R,
l/c     = hbar u_phi,
E_K/c^2 = -hbar u_t,
hbar    = 1 + e/c^2 + Pi/(Sigma c^2).
```

The absolute stream rate and these moments are immutable. C2 and C4 cell
weights are exact differences of their analytic cumulative profiles. At
N32/N64/N128 the source occupies 2/3/5 cells while total mass, radial
momentum, angular momentum, and Killing energy remain exact below
`2.06e-16`.

The DAE coordinate is `x^0=ct`. The source API therefore exposes both
physical mass-equivalent rates per second and the values divided by `c`
consumed by the finite-volume residual. That conversion closes below
`1.80e-16`.

The evidence runner uses a circular column at `240 rg` only as a bounded
four-state fixture. It does not claim that the real captured stream is already
circularized or thermally assimilated. A future Layer-1 ballistic state must
enter through the same one-state contract.

## Weak-field regression

A cold circular state at `10000 rg` gives

```text
v_transport/c                         5.42e-20
(l_KS/l_Newton)-1                     1.50034e-4
[(E_K-c^2)/E_bind,Newton]-1          -7.50338e-5
```

This is the expected first-order Schwarzschild correction and satisfies the
WP10b requirement that the migrated physical moments recover the Newtonian
benchmark.

## Roche energy-zero handshake

The reduced nozzle still uses its local PW-secondary plus Hill tidal force.
WP10c4 does not relabel that force as a full relativistic binary potential.
Instead it supplies the edge values

```text
v_R,nozzle = v_transport,
l_kin      = c u_phi,
l_flux     = c hbar u_phi,
B_inertial = E_K - c^2.
```

A constant potential shift makes the nozzle edge Bernoulli equal
`B_inertial`. The edge-to-saddle potential difference and force are unchanged.
The nozzle's angular and energy ledgers use `l_flux` and the full Killing
energy including rest mass when converted back to the finite-volume face.

The physical face rates enter as

```text
[Mdot/c, Pdot_R/c^2, Jdot/c^2, Edot_K/c^3].
```

An old PW disk-energy label, nonzero outer shear stress, inward nozzle mass,
or a failed angular/Killing/pattern-power identity is rejected.

## Closed/choked and characteristic gate

For the bounded `Sigma=1e4 g cm^-2` edge fixture:

| State | T [K] | H/R | Available energy [erg/g] | Mdot overflow [g/s] |
|---|---:|---:|---:|---:|
| Closed | `8.0e5` | `0.03911` | `-6.9031e16` | `0` |
| Threshold below | `9.882318e5` | `0.08925` | `-6.7634e11` | `0` |
| Threshold above | `9.882338e5` | `0.08925` | `+6.7634e11` | `27.92` |
| Choked | `1.0e6` | `0.09353` | `+8.3967e15` | `4.8644e19` |

Closed states retain pressure traction and zero advective mass/J/E. Choked
states add only outward nozzle flux. Every state has exactly one incoming
outer acoustic characteristic.

The gas-radiation sonic solve becomes numerically ill-conditioned extremely
close to zero available energy; the independent energetic gate is therefore
audited separately. At `1e-6` relative temperature on either side, both the
closed and regular choked contracts solve.

## Rank and scope

Exact sources add no unknowns and no algebraic rows. Relative to the four-field
WP10b flux-primary base, the outer provider supplies four face rows whose
Jacobian with respect to the four face-flux unknowns is the identity. At
N16/N64/N128:

```text
total unknowns = total rows = 12N+4
outer face-row rank = 4
physical outer boundary conditions = 1
```

This is deliberately a base-count audit. The evolved causal shear variable
and responsive-height time-derivative coupling have not yet been assembled
into the final nonlinear DAE. Their exact count and complete characteristic
rank must be frozen before accepting a stationary root.

## Classification

```text
numerical status:
    supported but not fully certified for the source/boundary adapter

physical status:
    diagnostic only

production status:
    blocked
```

WP10c4 does not establish:

1. a calibrated ballistic Layer-1 injection state;
2. a relativistic multidimensional L1/L2 force geometry;
3. the final stress-augmented nonlinear eigensystem;
4. a stationary N64/N96 disk;
5. a stable tiny implicit timestep;
6. a hot branch, limit cycle, tide, or wind.

## Locked next step

Proceed to WP10c5, with a count/rank preflight before any nonlinear solve:

1. assemble the responsive thermal mass matrix, causal shear variable,
   Kerr-Schild geometric source, comoving cooling/height work, exact stream
   source, and Roche face in one finite-volume DAE;
2. document the exact unknown and residual count, including the evolved shear
   field and all face variables;
3. repeat the full nonlinear characteristic and inner/outer boundary-rank
   audit;
4. solve a chained N64 then N96 low-throughput stationary root only if that
   audit passes;
5. place the inner face inside `2 rg` and require zero incoming inner modes;
6. attempt one tiny implicit step only after both stationary meshes pass.

Do not add another inner/outer splice, distributed tide, wind, or long loading
in WP10c5.

## Verification

```text
focused migration/nozzle/thermal tests   33 passed
complete repository suite                450 passed, 4 subtests passed
repository hygiene                       passed at 601 tracked files
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_migration_wp10c4.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_causal_inner_migration_wp10c4.py
```
