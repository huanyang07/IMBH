# Fixed-Q Authentic History Ladder WP10c9d6c7c3b5c4f24e1

## Classification

`authentic_fixed_Q_history_ladder_rejected_at_primary_BDF1_solver_budget`

The authentic fixed-`Q` history ladder stopped at its first fail-fast case:
the 20 ms middle-layout state with `h=1e-7 s`. The increment-primary BDF1
root did not reach the unchanged `1e-10` complete-residual gate under the
frozen policy of one exact bordered Jacobian assembly followed by Broyden
updates.

This is a solver-policy rejection, not a physical fixed-`Q` rejection. The
continuous KKT construction, conservative reaction map, and previously
certified complete Jacobian are preserved.

## Decisive result

The old direct-rate root supplied a strong initial seed with complete scaled
residual `6.8220e-10`. One exact Jacobian correction and five accepted
Broyden updates reduced the saved residual only to

```text
4.718758821187219e-10
```

after 36 complete residual evaluations, six linear solves, and 4681.56 s.
The solve terminated with `fixed-Q bound-aware line search failed`.

Only these two binding gates failed:

- nonlinear root;
- complete residual.

Every independent non-root gate passed:

| Diagnostic | Result | Gate |
|---|---:|---:|
| `Q3` relative defect | `1.17e-16` | `<=1e-12` |
| storage parity defect | `5.39e-10` | `<=1e-9` |
| minimum reconstruction factor | `1.0` | `>=1-1e-12` |
| reaction-channel ledger defect | `1.89e-16` | `<=1e-12` |
| reaction-action ledger defect | `1.33e-16` | `<=1e-12` |
| raw Schur rank | `3` | `3` |
| raw Schur condition number | `3.385e4` | `<=1e8` |
| incoming excision modes | `0` | `0` |
| maximum `H/R` | `0.09784` | `<=0.12` |
| minimum scattering depth | `19.25` | `>=1` |
| maximum scaled primitive change | `0.004698` | `<=0.005` |

The reconstruction limiter remained inactive. The state stayed physically
admissible and causally outgoing. The reaction and constraint ledgers close
near roundoff.

## Interpretation

The one-refresh Broyden model lost useful descent. Later full Broyden trials
raised the residual to `1.07e-8`; many increasingly small line-search trials
also failed the full merit test. The binding run is therefore rejected and
cannot be converted into a pass by a diagnostic rerun.

The remaining five state/timestep cases were not launched. This preserves the
predeclared fail-fast cost contract.

## Authorized next action

Run one diagnostic-only exact-Jacobian refresh from the saved rejected BDF1
endpoint. Keep the same equation, timestep, residual gate, constraint target,
reaction representation, and physical guards.

- If that exact refresh reaches `1e-10`, the equation root remains available
  and an adaptive exact-refresh solver policy may be frozen prospectively.
- If it does not, localize the exact matrix-action and residual blocks at the
  rejected endpoint before changing the nonlinear solver.

The diagnostic may not amend this rejection. A repaired binding ladder must
start again from the original committed 20 ms state under a new prospective
solver contract.

One-`Q` execution, fixed-`Q` microbursts, 50 ms propagation, and reduced slow
evolution remain blocked.
