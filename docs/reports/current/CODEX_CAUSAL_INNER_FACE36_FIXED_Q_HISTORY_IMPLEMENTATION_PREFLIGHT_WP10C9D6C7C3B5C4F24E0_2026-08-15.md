# Fixed-Q History Implementation Preflight WP10c9d6c7c3b5c4f24e0

## Classification

`fixed_Q_history_implementation_preflight_certified_physical_history_ladder_authorized`

The corrected fixed-`Q` constrained-history implementation is certified for
the next bounded physical preflight. This package advances no trajectory and
changes no physical operator. It authorizes only the authentic
BDF1-start/BDF2-history ladder frozen by WP10c9d6c7c3b5c4f24d1.

## Certified implementation changes

- The binding nonlinear root uses the complete increment-primary mapped and
  responsive-height storage residual. The direct-rate form is evaluated only
  after the root as an independent parity diagnostic.
- A single fail-closed acceptance record binds the nonlinear residual, exact
  `Q3`, outgoing excision, storage parity, reconstruction, reaction ledgers,
  physical guards, primitive change, and reaction-Schur conditioning.
- State-normalized reaction channels are rejected by the binding solver.
  Frozen-normalized channels remain binding and raw channels diagnostic.
- The reaction Schur system uses a stable solve with explicit singular values,
  rank, condition number, and solve defect.
- Rejected steps cannot create BDF history. An accepted BDF1 endpoint stores
  its literal primitive difference and complete mapped/height path history.
- The fixed-`Q` restart records the `Q3` target, constraint scales, multiplier
  predictor, timestep/order history, reaction policy, and provenance. The
  endpoint reaction normalization is recomputed at the accepted state.
- Exact-Jacobian, Broyden-update, and linear-solve counts are explicit.

## Verification

The exact committed execution source was
`66a3039e202d5162a4b9222ad1bd76e1e477d03b`, tree
`277dd56d2c94826f263c5aa402431ae223cdd41f`. The tracked worktree was clean
at execution start. Thirteen unrelated untracked files under
`docs/reports/gpt/` were recorded and left untouched.

The focused suite completed in `216.57 s`:

```text
18 passed in 216.25s
```

The tests cover the fixed-`Q` residual/reaction implementation and the shared
monolithic storage, BDF, and discrete-tangent contracts. In particular they
exercise binding temporal dispatch, stable Schur failure, rejection of the
state-normalized kernel, fail-closed synthetic-history handling, literal
accepted history, and restart roundtrip.

## Scope and next action

This is an implementation certificate, not a physical fixed-`Q` result. It
does not show that the one-complete-Jacobian-plus-Broyden budget reaches the
`1e-10` root on the committed 20 ms and 16 ms states, nor that the authentic
BDF2 histories converge.

Run the frozen fail-fast ladder next:

1. 20 ms and 16 ms states at `h=1e-7 s`;
2. both states at `h=5e-8 s`;
3. both states at `h=2.5e-8 s`;
4. serialized BDF2 replay;
5. adjacent-pair convergence of state-space BDF rate and physical reaction
   action.

Fixed-`Q` microbursts, 50 ms evolution, one-`Q` closure fitting, and reduced
slow evolution remain blocked.
