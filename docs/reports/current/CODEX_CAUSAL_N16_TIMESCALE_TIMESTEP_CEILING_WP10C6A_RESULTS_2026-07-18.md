# Causal N16 Timescale and Timestep-Ceiling WP10c6a Results

Date: 2026-07-18

## Verdict

The bounded N16 audit identifies a finite observable-accuracy ceiling for the
current backward-Euler causal five-field DAE:

```text
largest passing timestep                 1.921821997458634e-3 s
first failing timestep                   3.843643994917267e-3 s
failure class                            temporal accuracy only
inherited controller timestep            7.507117177572788e-6 s
certified timestep gain                  256
shortest measured physical clock         4.791648827929410e-2 s
```

Every full step and half step in the bracket converges and passes the
unchanged nonlinear, algebraic, conservation, causal, optical-depth, and
Roche-boundary contracts. The upper rung fails only the declared temporal
observable gate.

The inherited microsecond timestep is therefore not a demonstrated physical
causal limit. The N16 temporal ceiling is set by backward-Euler accuracy well
before the shortest physical clock.

This result authorizes one N32 temporal audit. It does not authorize N64/N128
production ladders, long evolution, tide, wind, stability, a hot-state
search, or a cycle search.

## Fixed Datum

Every rung starts independently from the same accepted restart:

```text
checkpoint       causal_wp10c5q_N016_final.npz
checkpoint SHA   9b8247536daf2ddd2868f571d911751062a90d08690499ffd69794bff9046e7e
work package     WP10c5q
elapsed time     8.484232672865630e-4 s
accepted steps   71
rejected trials  0
stream           exact circularized C2 regression stream
```

The restart passes:

```text
maximum H/R                            9.9932071e-2
minimum scattering optical depth      18.5218106
inner incoming characteristics        0
inner light-cone excess               0
outer incoming characteristics        2
outer Roche channel                   closed, nonchoked
scaled algebraic residual             1.41757e-15
```

No trajectory history is accumulated across ladder rungs. Each comparison is
one backward-Euler step of `dt` against two backward-Euler steps of `dt/2`
from this identical datum and controller history.

## Versioned Observables

WP10c6a adds the immutable schema:

```text
causal-five-field-observables-v1
```

It contains:

- the integrated two-face comoving diffusion-cooling power proxy;
- the same proxy using cells centered outside `6 rg`;
- positive inward rest-mass flux at the inner face;
- maximum and cellwise `H/R`;
- all five integrated conserved fields.

The cooling quantities are finite-volume integrals of the local comoving
cooling law. They are temporal-control observables, not luminosities at
infinity.

The accuracy gates are:

```text
total cooling proxy relative error             1e-3
cooling proxy outside 6 rg relative error      1e-3
inner mass-flux relative error                  1e-3
maximum Delta ln(H/R)                           2e-3
maximum integrated-conserved relative error    1e-3
maximum baseline-scaled full-state difference  2e-3
```

The last metric is an additional whole-DAE diagnostic. It is not the
controlling error in the final bracket.

## Local Clock Audit

All clocks are coordinate-time quantities evaluated on the accepted restart.

| Clock | Definition | Minimum | Radius |
|---|---|---:|---:|
| Characteristic cell crossing | `Delta R/(c max abs lambda)` | `4.79165e-2 s` | `2.11935 rg` |
| Causal-stress relaxation | `gamma tau_relax/lapse` | `1.58779e-1 s` | `2.11935 rg` |
| Radial advection | `R/abs(c u^R/u^0)` | `1.51254e-1 s` | `2.11935 rg` |
| Cooling-luminosity response | `C_lnT/(4 Qrad)` | `1.35250 s` | `2.11935 rg` |
| Thermal response | `C_lnT/Qrad` | `5.41001 s` | `2.11935 rg` |
| Local stream loading | cell rest mass / cell source | `3.96155e5 s` | `205.236 rg` |
| Global loading | disk rest mass / total source | `8.48423e5 s` | global |

Here

```text
C_lnT = Sigma (de/dlnT)_Sigma + Pi (dlnH/dlnT)_Sigma .
```

The current diffusion closure has

```text
(d ln Qrad/d ln T)_Sigma = 4 .
```

The inherited controller step is only `1.5667e-4` of the shortest physical
clock.

## Step-Doubling Ladder

The ladder doubles the accepted restart timestep and stops at the first
failure.

| `dt` (s) | Total cooling | Cooling outside `6 rg` | Inner `Mdot` | Max `Delta ln(H/R)` | Scaled state | Status |
|---:|---:|---:|---:|---:|---:|---|
| `7.507117178e-6` | `7.250e-9` | `2.807e-9` | `5.200e-10` | `1.796e-8` | `4.518e-9` | pass |
| `1.501423436e-5` | `2.900e-8` | `1.123e-8` | `2.080e-9` | `7.183e-8` | `1.807e-8` | pass |
| `3.002846871e-5` | `1.160e-7` | `4.495e-8` | `8.316e-9` | `2.873e-7` | `7.228e-8` | pass |
| `6.005693742e-5` | `4.642e-7` | `1.799e-7` | `3.324e-8` | `1.149e-6` | `2.891e-7` | pass |
| `1.201138748e-4` | `1.857e-6` | `7.210e-7` | `1.328e-7` | `4.595e-6` | `1.156e-6` | pass |
| `2.402277497e-4` | `7.435e-6` | `2.893e-6` | `5.297e-7` | `1.837e-5` | `4.621e-6` | pass |
| `4.804554994e-4` | `2.978e-5` | `1.165e-5` | `2.108e-6` | `7.336e-5` | `1.846e-5` | pass |
| `9.609109987e-4` | `1.194e-4` | `4.722e-5` | `8.340e-6` | `2.925e-4` | `7.360e-5` | pass |
| `1.921821997e-3` | `4.798e-4` | `1.941e-4` | `3.266e-5` | `1.160e-3` | `2.921e-4` | pass |
| `3.843643995e-3` | `1.932e-3` | `8.225e-4` | `1.253e-4` | `4.538e-3` | `1.143e-3` | fail |

The differences grow approximately by a factor of four per timestep
doubling, as expected for the local step-doubling difference of a first-order
method.

At the failing rung:

- all three nonlinear solves converge in six iterations or fewer;
- the maximum scaled nonlinear residual is below `5.3e-13`;
- the maximum algebraic residual is below `3.1e-15`;
- all causal, optical, and Roche state gates pass;
- the maximum physical five-field ledger defect is below `3.4e-13`;
- the audit-only primitive/total change bounds are inactive.

The failure is triggered by the total cooling and `H/R` profile observables,
not the solver, conservation, boundary contract, or emergency change bounds.

## Classification

The N16 timestep is:

```text
temporal-accuracy limited below the shortest physical clock
```

The bracket occupies only:

```text
passing dt / shortest clock       0.04011
failing dt / shortest clock       0.08022
```

This is useful but not yet a production timestep law. The audit measures one
local state on N16. It does not establish that the same ceiling applies after
substantial loading, near a thermal transition, on a finer mesh, or after
tide/wind terms are introduced.

## Locked Next Work

WP10c6b should repeat this exact observable and state-gate contract at N32:

1. consume the accepted source-compatible N32 WP10c5q checkpoint;
2. measure the same local clocks and use the same versioned observables;
3. run one full versus two half backward-Euler steps on a factor-two ladder;
4. bracket the first temporal-accuracy failure without changing tolerances;
5. compare the N16 and N32 ceilings and controlling observables;
6. define a conservative temporal-controller target only if the ceiling is
   mesh supported.

Do not run N64/N128 production ladders or a long physical trajectory in
WP10c6b. Generalized eigenanalysis, BDF2, Jacobian reuse, reduced-stress
limits, and multirate integration remain later optimization packages.

## Verification

```text
focused causal diagnostics/evolution/DAE tests   34 passed
full repository suite                            493 passed, 4 subtests
production ladder                               passed
repository hygiene                              passed
Python compilation                              passed
git diff --check                                passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_timescale_timestep_audit_wp10c6a.py
```

Machine-readable output:

```text
outputs/tables/causal_timescale_timestep_audit_wp10c6a.json
```

The generated output remains ignored under the repository artifact policy.
