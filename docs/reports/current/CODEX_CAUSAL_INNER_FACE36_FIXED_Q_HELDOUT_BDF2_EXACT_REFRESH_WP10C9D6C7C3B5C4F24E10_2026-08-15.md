# Fixed-Q Held-Out BDF2 Exact-Refresh Diagnostic WP10c9d6c7c3b5c4f24e10

## Classification

`heldout_BDF2_exact_refresh_reached_root_adaptive_policy_manifest_authorized`

The rejected 16 ms, `h=1e-7 s` BDF2 endpoint is not at a local residual or
linearization floor. Its failure under the original one-exact-Jacobian plus
Broyden budget is caused by the stale Broyden linearization.

The historical held-out stage remains rejected. The remaining timestep
ladder, fixed-`Q` microbursts, and reduced slow evolution remain blocked.

## Exact source reproduction

The diagnostic loads the canonical accepted BDF1 restart and the saved
rejected BDF2 primitive increment, multiplier, and residual. Before applying
any correction, it reproduces

```text
saved endpoint maximum scaled residual       1.562552753853197e-9
fresh endpoint maximum scaled residual       1.562552753853197e-9
source endpoint reproduction                  bitwise
```

Thus the diagnostic differs from the rejected run only in its fresh complete
bordered Jacobian.

## One exact correction

The prospectively frozen diagnostic permits one Newton iteration, one exact
Jacobian assembly, and the unchanged 12-step line search. The full correction
is accepted immediately:

```text
line-search alpha                             1.0
corrected maximum scaled residual             3.475167854795006e-13
required maximum scaled residual              1.0e-10
function evaluations                          2
linear solves                                 1
exact Jacobian assemblies                     1
wall time                                     528.75 s
```

The solver records one terminal secant update after accepting the correction,
but no Broyden direction is subsequently used.

## Non-root gates

Every independent acceptance condition passes:

- Q3 relative defect `1.44e-16`;
- raw Schur identity closure `8.99e-15`, rank three, condition `3.48e4`;
- storage parity `2.51e-14`;
- exact-increment binding path active and direct-rate path audit-only;
- reconstruction factors exactly one;
- reaction and constraint-action ledgers `1.90e-16` and `1.33e-16`;
- `H/R=0.09802`, scattering optical depth `19.19`;
- maximum scaled primitive change `0.004670 < 0.005`;
- zero incoming excision characteristics.

No physical, storage, reaction, constraint, Schur, or boundary failure is
detected.

## Binding interpretation

This result proves that the rejected held-out BDF2 endpoint is recoverable by
a fresh exact linearization. It does not retroactively pass the historical
stage and does not authorize simply increasing the exact-assembly budget in
the running ladder.

The next package must prospectively define an adaptive policy with these
properties:

1. begin each root with the existing exact bordered Jacobian;
2. continue with Broyden updates while they produce descent;
3. after a Broyden line-search failure, assemble at most one additional exact
   Jacobian at the unchanged current iterate;
4. retry the correction without changing the residual, row scales, merit,
   line-search sequence, or `1e-10` gate;
5. serialize the refresh reason and all solver counters;
6. first revalidate the already certified primary coarse case;
7. retry held-out coarse only after primary non-regression passes.

The refined `5e-8` and `2.5e-8 s` rungs remain unauthorized until both coarse
states pass under the prospective policy with bitwise restart/replay.
