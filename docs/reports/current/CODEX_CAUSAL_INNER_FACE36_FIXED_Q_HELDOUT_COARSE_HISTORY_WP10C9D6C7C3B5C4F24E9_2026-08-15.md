# Fixed-Q Held-Out Coarse History Stage WP10c9d6c7c3b5c4f24e9

## Classification

`fixed_Q_remaining_history_stage_heldout_coarse_failed`

The held-out middle-layout 16 ms state passes its authentic constrained BDF1
startup at `h=1e-7 s`, but the following exact-history constrained BDF2 root
does not converge under the prospectively frozen one-exact-Jacobian plus
Broyden solver budget.

No refined timestep, fixed-`Q` microburst, or reduced slow evolution is
authorized. No physical failure is detected.

## Passing BDF1 startup

The held-out startup begins without the historical primary-state root seed:

```text
initial maximum scaled residual              9.051540987472656e-1
accepted maximum scaled residual             6.105620675602080e-11
iterations                                   6
function evaluations                         13
exact Jacobian assemblies                    1
Schur identity closure                       7.286365431463959e-14
```

Exact Q3, exact-increment binding storage, direct-rate parity, inactive
reconstruction, reaction and action ledgers, physical guards, primitive
change, and outgoing excision all pass. The accepted BDF1 history survives
the serialized restart roundtrip bitwise.

## Rejected BDF2 continuation

The BDF2 root follows this residual history:

```text
2.692789561671447e+0
1.479516867342656e-3
1.618567590094822e-6
4.437709200111328e-9
1.562552753853197e-9
```

The next Broyden direction fails to provide descent over all 12 frozen line
search lengths. The step is rejected at

```text
maximum scaled residual                      1.562552753853197e-9
required residual                            1.0e-10
iterations                                   4
function evaluations                         17
exact Jacobian assemblies                    1
Schur identity closure                       6.569955038071508e-14
```

Every non-root acceptance gate passes:

- Q3 defect `2.35e-16`;
- storage parity `2.81e-14`;
- mapped endpoint/path closure `2.61e-10`;
- reconstruction factors exactly one;
- reaction/action ledgers `1.90e-16/1.09e-23`;
- `H/R=0.09802`, optical depth `19.19`;
- primitive change `0.004670 < 0.005`;
- zero incoming excision characteristics.

The rejection is therefore a held-out BDF2 nonlinear-solver failure, not a
Schur, constraint, storage, admissibility, or physical-equation failure.

## Next diagnostic

Freeze one analysis-only exact-refresh diagnostic at the saved rejected BDF2
endpoint:

1. replay the endpoint residual bitwise;
2. assemble one fresh complete bordered Jacobian;
3. solve one correction and audit its linear residual;
4. evaluate the frozen line-search sequence;
5. compare analytic action with a direct residual JVP on that correction;
6. classify whether the failure is a stale-Broyden problem or a local
   residual/linearization floor.

The diagnostic may not change the historical stage classification, relax
`1e-10`, or authorize the refined ladder. If a fresh exact correction reaches
the root, a new prospective solver policy must be frozen and retested at the
already certified primary coarse case before the ladder can resume.
