# Causal N32 Temporal-Controller WP10c6b Results

Date: 2026-07-18

## Verdict

The bounded N32 audit exactly reproduces the WP10c6a N16 temporal bracket:

```text
largest passing timestep                 1.921821997458634e-3 s
first failing timestep                   3.843643994917267e-3 s
failure class                            temporal accuracy only
inherited controller timestep            7.507117177572788e-6 s
certified timestep gain                  256
```

Both meshes first fail the same two observables:

```text
total diffusion-cooling power proxy
maximum Delta ln(H/R) profile
```

Every full step and half step in the N32 bracket converges and passes the
unchanged nonlinear, algebraic, conservation, causal, optical-depth, and
Roche-boundary contracts. The N16/N32 ceiling ratio and first-failure ratio
are both exactly one.

This result certifies a mesh-supported local temporal-controller contract for
the accepted source-compatible state. It does not certify the controller
implementation, a long trajectory, stability, a hot state, a limit cycle,
tide, or wind.

## Fixed N32 Datum

Every rung starts independently from:

```text
checkpoint       causal_wp10c5q_N032_final.npz
checkpoint SHA   1799f2917f25a55c4175e78f17cd9eb6fe9e80009fae76e989e2483c526d6a64
work package     WP10c5q
elapsed time     8.484232672865630e-4 s
accepted steps   73
rejected trials  0
stream           exact circularized C2 regression stream
```

The restart has `H/R=0.0999582`, minimum scattering optical depth `18.5205`,
zero inner incoming characteristics, two outer incoming characteristics, and
a closed, nonchoked Roche channel.

The versioned observable schema and all accuracy gates are identical to
WP10c6a:

```text
causal-five-field-observables-v1
```

No trajectory history is shared between ladder rungs.

## Clock Comparison

All clocks are coordinate-time quantities evaluated on the accepted restart.

| Minimum clock | N16 | N32 |
|---|---:|---:|
| Characteristic cell crossing | `4.79165e-2 s` | `2.16380e-2 s` |
| Causal-stress relaxation | `1.58779e-1 s` | `1.43370e-1 s` |
| Radial advection | `1.51254e-1 s` | `1.36972e-1 s` |
| Cooling-luminosity response | `1.35250 s` | `1.24627 s` |
| Thermal response | `5.41001 s` | `4.98508 s` |
| Local stream loading | `3.96155e5 s` | `2.31062e5 s` |
| Global loading | `8.48423e5 s` | `8.49611e5 s` |

The finer cells shorten the characteristic crossing clock by a factor of
`2.214`, but the measured observable-accuracy ceiling is unchanged. At N32:

```text
passing dt / shortest physical clock       0.08882
failing dt / shortest physical clock       0.17763
inherited dt / shortest physical clock     3.46941e-4
```

The local backward-Euler accuracy ceiling is therefore not behaving like a
cell-crossing CFL limit over N16-N32.

## Bracket Comparison

| Mesh | Last passing `dt` | First failing `dt` | Failing observables |
|---:|---:|---:|---|
| N16 | `1.921821997e-3 s` | `3.843643995e-3 s` | cooling, `H/R` |
| N32 | `1.921821997e-3 s` | `3.843643995e-3 s` | cooling, `H/R` |

At the N32 last passing rung:

```text
total cooling relative error                 4.40079e-4
cooling outside 6 rg relative error          1.72503e-4
inner accretion-rate relative error           3.87502e-5
maximum Delta ln(H/R)                         1.20285e-3
maximum baseline-scaled state difference      3.01861e-4
maximum integrated-conserved relative error   5.09233e-7
maximum normalized temporal error             0.601426
controlling observable                        Delta ln(H/R)
```

At the N32 first failing rung:

```text
total cooling relative error                 1.74819e-3
maximum Delta ln(H/R)                         4.67770e-3
maximum baseline-scaled state difference      1.17501e-3
maximum integrated-conserved relative error   2.06866e-6
maximum normalized temporal error             2.33885
```

The failing full step still has:

```text
maximum scaled nonlinear residual             7.40909e-12
maximum scaled algebraic state-gate residual  1.91373e-13
maximum physical five-field ledger defect     7.89157e-12
nonlinear iterations                          5
maximum scaled primitive/total change         4.45326e-2
```

Thus the failure is neither nonlinear nor physical-contract failure.

## Authorized Controller Contract

The mesh comparison authorizes this first-order backward-Euler
step-doubling contract:

```text
trial                    one full step and two half steps
accepted state           two-half-step state
error                    full-step observables minus half-step observables
normalized error         max_i(error_i / declared_gate_i)
accept                    normalized error <= 1 and every existing gate passes
factor                    clip(0.8/sqrt(normalized_error), 0.25, 2.0)
initial timestep          9.609109987293168e-4 s
```

The initial timestep is one half of the shared conservative ceiling. The N32
last-passing state would propose a factor `1.03157`.

All existing requirements remain mandatory on the full step and both half
steps:

- nonlinear and algebraic residual gates;
- independent five-field physical ledgers;
- causal characteristic counts and light-cone bound;
- scattering optical depth and `H/R`;
- Roche active-set consistency;
- primitive and total-change emergency bounds.

The controller may reduce its timestep when a state evolves into a faster
thermal response. It must not treat the fixed local ceiling as a permanent
global maximum justified by this one audit.

## Classification

The N16/N32 result is:

```text
mesh-supported local temporal-controller contract
```

It is not:

```text
a long-duration physics result
a stability result
a hot-state or cycle result
a tide/wind authorization
a claim that N64/N128 have the same ceiling
```

The shortest cell-crossing clock changes under refinement while the
observable bracket does not. That supports observable-driven adaptivity for
the current implicit method and state, but only a short matched-duration
controller trial can establish practical efficiency and error accumulation.

## Locked Next Work

WP10c6c should:

1. implement the authorized step-doubling controller in the production causal
   evolution path;
2. preserve the exact accepted-state, gate, ledger, and checkpoint contract;
3. begin at `9.609109987293168e-4 s`;
4. run N16 and N32 to one short exact common time selected before execution;
5. compare against a tighter fixed-step reference at each mesh;
6. report accepted/rejected trials, nonlinear work, achieved errors, and
   restart identity;
7. stop after the bounded matched-duration efficiency/accuracy gate.

Do not run N64/N128, extend to a physical loading or thermal time, introduce
tide or wind, or interpret the trajectory as hot, stable, or cyclic.

## Verification

```text
focused causal diagnostics tests                35 passed
full repository suite                            494 passed, 4 subtests
N32 production ladder                           passed
N16/N32 mesh comparison                         passed
repository hygiene                              passed
Python compilation                              passed
git diff --check                                passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_timescale_timestep_audit_wp10c6a.py \
  --n-cells 32
```

Machine-readable output:

```text
outputs/tables/causal_timescale_timestep_audit_wp10c6b.json
```

The generated output remains ignored under the repository artifact policy.
