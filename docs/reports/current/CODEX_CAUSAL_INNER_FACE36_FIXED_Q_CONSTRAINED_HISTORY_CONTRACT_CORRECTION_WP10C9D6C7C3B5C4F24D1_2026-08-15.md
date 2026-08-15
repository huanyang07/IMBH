# Fixed-Q constrained-history contract correction

Classification:
`fixed_Q_constrained_history_contract_corrected_implementation_preflight_authorized`.

This definitions-only package supersedes only the execution contract frozen
by WP10c9d6c7c3b5c4f24d. It preserves the repaired fixed-Q Jacobian, exact
direct-rate roots, continuous KKT construction, and rejection of the
synthetic projected history. It executes no state and changes no physical
operator.

The correction is required because the frozen manifest named the
increment-primary complete BDF residual as binding, while the current solver
passes the primitive interval rate into the direct-rate storage path during
the nonlinear root. The next implementation package must make the
increment-primary residual the only binding root and evaluate the direct-rate
form independently after convergence.

The corrected contract also:

- replaces the maximum-only reconstruction check with a binding minimum path
  factor of `1 - 1e-12`;
- requires one fail-closed acceptance record before BDF history construction;
- defines a fixed-Q restart carrying the complete monolithic history, Q3
  target, constraint scales, multiplier predictor, and reaction policy;
- replaces the raw 3x3 Schur inverse with a stable solve and records rank,
  singular values, conditioning, normalization closure, and action
  sensitivity;
- restricts nonlinear roots to the certified frozen-normalized reaction
  basis, with raw channels diagnostic and state-normalized channels
  audit-only;
- describes the actual solver as one complete bordered Jacobian followed by
  dense rank-one Broyden updates, not matrix-free corrections;
- separates the reaction-channel ledger, multiplier-weighted physical action
  ledger, endpoint Q3 constraint, and continuous tangency diagnostic;
- requires binding evidence to run from a clean worktree at an exact committed
  SHA without modifying unrelated user-owned untracked reports.

The old direct-rate roots may seed the corrected solves but cannot certify
them. The next authorized work is implementation and inexpensive testing only.
Physical history execution, a fixed-Q microburst, 50 ms propagation, and
reduced slow evolution remain blocked.
