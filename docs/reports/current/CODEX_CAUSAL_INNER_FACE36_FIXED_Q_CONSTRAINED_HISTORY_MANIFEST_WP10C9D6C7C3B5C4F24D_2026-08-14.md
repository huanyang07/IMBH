# Fixed-Q constrained startup/history manifest

Classification:
`fixed_Q_constrained_BDF1_startup_BDF2_history_manifest_frozen_execution_preflight_authorized`.

This package is definitions-only. It advances no state and changes no physical
operator. It preserves the c4f24c result that the complete Jacobian and all
exact equal-Q BDF2 roots pass, while the synthetic projected-history limit
does not converge.

The next preflight must use the same history sequence as an actual execution:

1. one exact constrained BDF1 startup;
2. the accepted primitive, mapped-storage, responsive-height, and timestep
   history from that root;
3. one exact constrained equal-step BDF2 continuation;
4. a serialized restart and bitwise BDF2 replay.

No backward continuous-tangent history or reaction-coordinate projection may
be used. The primary state is the committed middle `20 ms` endpoint and the
held-out state is the committed middle `16 ms` endpoint. The fail-fast
timestep ladder is `1e-7`, `5e-8`, and `2.5e-8 s`.

The binding residual is the increment-primary complete BDF residual. Direct
storage-rate evaluation remains an independent parity audit with a `1e-9`
gate. The unchanged residual and Q3 gates remain `1e-10` and `1e-12`.
Convergence is measured in the state rate and physical reaction action;
multiplier coordinates remain a conditioning diagnostic.

Passing the execution-shaped chain may authorize only a fresh definitions-only
one-Q execution manifest. It does not authorize a fixed-Q microburst, a 50 ms
trajectory, or reduced slow evolution.
