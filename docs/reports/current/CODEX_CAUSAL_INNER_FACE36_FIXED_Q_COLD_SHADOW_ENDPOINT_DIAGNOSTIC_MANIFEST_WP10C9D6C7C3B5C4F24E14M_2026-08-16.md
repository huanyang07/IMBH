# Fixed-Q Cold-Shadow Endpoint Diagnostic Manifest WP10c9d6c7c3b5c4f24e14m

## Classification

`cold_shadow_endpoint_diagnostic_manifest_frozen_one_nonpropagating_exact_correction_authorized`

This definitions-only package preserves the binding
`bounded_continuation_failed` classification from WP10c9d6c7c3b5c4f24e14l.
It authorizes one exact-Jacobian correction at the saved accepted cold-shadow
endpoint. It authorizes no continuation state, trajectory advance, retry,
held-out run, timestep study, microburst, averaging, or reduced evolution.

## Locked evidence

The manifest checksum-locks:

- the common pre-`warm_2` checkpoint;
- the accepted `warm_2` result;
- the accepted same-history cold-shadow result;
- the complete parent metrics, summary, and provenance;
- every diagnostic source and focused test.

The committed parent comparison is frozen as:

```text
warm residual                              5.048217216618925e-13
cold-shadow residual                       6.398284679853816e-11
scaled state absolute difference           7.859135564558528e-11
physical reaction-action relative defect   2.8666087608919947e-8
state/action equivalence tolerance          1.0e-8 / 1.0e-8
```

Both saved residual vectors and both saved physical reaction actions must be
reproduced bitwise before the diagnostic may assemble a matrix.

## Authorized diagnostic

At the saved cold-shadow endpoint, the runner may:

1. assemble one exact complete bordered Jacobian;
2. solve one exact Newton correction;
3. evaluate the prospectively frozen full/halved line-search factors;
4. audit the corrected nonpropagating candidate with every unchanged physical,
   storage, reaction, conditioning, and ledger gate;
5. compare corrected state and physical reaction action with the committed
   warm endpoint.

The binding residual remains `1e-10`; the state and physical-action endpoint
equivalence gates remain `1e-8`. Multiplier-coordinate agreement is not used
as an invariant.

## Prospective classifications

`cold_shadow_residual_limited_action_equivalence_diagnosed`

: The corrected candidate passes every audit and closes both state and action
  equivalence at the unchanged tolerances. This may authorize only a
  definitions-only same-history equivalence-policy manifest.

`cold_shadow_exact_endpoint_diagnostic_inconclusive`

: The exact correction improves residual and action agreement but does not
  close every frozen gate.

`cold_shadow_exact_endpoint_diagnostic_failed`

: The exact correction does not produce the required improvement or audit
  closure.

No outcome retroactively changes WP10c9d6c7c3b5c4f24e14l.

## Verification

The pre-freeze focused suite passes `7/7` with two prospective skips before
canonical artifacts exist. The diagnostic and manifest scripts compile. The
manifest is frozen from definition commit `79a3c2d` with the single-thread
BLAS/OpenMP environment recorded.

## Next action

Execute only the frozen nonpropagating diagnostic. Stop and classify from its
result. Do not rerun any BDF2 trajectory root.
