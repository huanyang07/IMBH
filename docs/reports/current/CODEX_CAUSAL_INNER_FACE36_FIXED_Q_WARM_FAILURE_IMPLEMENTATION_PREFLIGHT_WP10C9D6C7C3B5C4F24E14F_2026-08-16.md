# Fixed-Q warm-failure implementation preflight

Work package: `WP10c9d6c7c3b5c4f24e14f`

## Classification

`warm_failure_accounting_and_endpoint_replay_certified_endpoint_diagnostic_manifest_authorized`

The accounting and replay repair passed. This package did not solve a
nonlinear root, assemble an exact Jacobian, construct a continuation state, or
advance the accepted trajectory. The parent classification
`bounded_continuation_failed` is unchanged.

## Binding results

- The rejected `warm_1` residual replayed bitwise. Its maximum scaled residual
  is exactly `5.708109263036221e-9`, and the maximum array difference from the
  committed residual is zero.
- The failure-aware canonical accounting contains one accepted root
  (`cold_1`), one rejected root (`warm_1`), and an accepted trajectory horizon
  of `1e-7 s`. The rejected candidate ledger is kept diagnostic-only.
- The historical cold checkpoint loads explicitly as solver-state schema 1
  with `legacy_untrusted_aggregate` counter semantics.
- Reconstruction from the committed cold event trace gives six total Broyden
  updates but only one update since the last of two exact assemblies.
- New solver-state schema 2 records total updates separately from updates
  since the last exact assembly, resets the age counter at every exact
  assembly, and records call counts plus exclusive profiling categories.

## Authorization

The sole next authorized action is a separate definitions-only manifest for
one exact-Jacobian diagnostic at the saved rejected endpoint. That diagnostic
must be nonpropagating and may use at most one exact complete Jacobian and one
exact Newton correction.

No warm-policy execution, full primary retry, held-out continuation,
operational-timestep search, physical microburst, fast averaging, or reduced
slow evolution is authorized.
