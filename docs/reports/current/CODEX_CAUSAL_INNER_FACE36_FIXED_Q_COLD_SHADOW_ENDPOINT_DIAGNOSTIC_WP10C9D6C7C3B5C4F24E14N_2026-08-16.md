# Fixed-Q Cold-Shadow Endpoint Diagnostic WP10c9d6c7c3b5c4f24e14n

## Classification

`cold_shadow_residual_limited_action_equivalence_diagnosed`

The one-correction nonpropagating diagnostic passes. The saved warm and cold
residual vectors and physical reaction actions reproduce bitwise. One exact
complete bordered Jacobian correction at the accepted cold-shadow endpoint
reduces the residual and closes the unchanged state/action equivalence gates
while passing every inherited physical, storage, reaction, conditioning, and
ledger audit.

The parent WP10c9d6c7c3b5c4f24e14l classification remains
`bounded_continuation_failed`. This diagnostic does not retroactively convert
that execution into a pass and adds no accepted trajectory time.

## Saved endpoint reproduction

Before assembling a matrix, the runner reconstructed both endpoints from the
hash-locked common start checkpoint and required bitwise equality for:

- warm complete residual;
- cold complete residual;
- warm physical reaction action;
- cold physical reaction action.

All four bitwise gates pass. The committed initial comparison is reproduced:

```text
warm maximum scaled residual                 5.048217216618925e-13
cold maximum scaled residual                 6.398284679853816e-11
scaled state absolute defect                 7.859135564558528e-11
reaction-action relative defect              2.866608760891995e-8
```

## Exact correction

The diagnostic assembled one exact complete bordered Jacobian at the saved
cold endpoint. The equilibrated linear solve has relative residual
`8.60170454638767e-12`. The full correction is not bound-limited and is
accepted on the first frozen trial.

```text
corrected maximum scaled residual            5.108629683951753e-13
corrected-to-warm state defect               5.064393349130114e-12
corrected-to-warm action defect              1.875569179324639e-9
action-defect reduction factor               15.283940429881735
```

The unchanged gates are:

```text
maximum scaled residual                      1e-10
maximum scaled state equivalence defect      1e-8
maximum reaction-action equivalence defect   1e-8
```

All three pass after the correction.

The cold-to-corrected change is small in state space but amplified in the
reaction action:

```text
scaled state correction                      7.352696229645517e-11
reaction-action relative change              2.679131902000647e-8
action change / scaled state correction      364.3740769812561
```

This quantitatively explains why a cold root that is valid under the common
`1e-10` residual gate can still fail a `1e-8` invariant-action equivalence
test. The root tolerance is adequate for per-step physical acceptance but is
not sufficient by itself for this more sensitive cross-algorithm comparison.

## Corrected physical audit

The corrected candidate passes every inherited gate:

```text
maximum Q3 relative defect                   2.794308181737699e-16
maximum storage-parity defect                3.727712428543362e-14
maximum reaction-ledger defect               2.989737646549545e-22
maximum constraint-action defect             7.824791862720501e-24
raw Schur rank / condition number             3 / 3.452378615412227e4
maximum raw Schur solve defect                3.257565761503338e-14
minimum / maximum reconstruction factor      1.0 / 1.0
maximum H/R                                  0.0978374774166655
minimum scattering optical depth             19.254319053888015
maximum scaled primitive change              0.004158089147894511
incoming excision characteristics            0
```

No continuation state is constructed, the corrected candidate does not enter
history, and accepted trajectory time added is exactly zero.

## Scientific interpretation

The parent rejection is now localized to a mismatch between two tolerances:

1. the common per-root acceptance gate permits the cold shadow to stop at
   `6.40e-11`;
2. the physical reaction action near this endpoint changes by roughly 364
   relative units per unit of maximum scaled state correction;
3. the resulting action uncertainty exceeds the tighter `1e-8` endpoint
   equivalence gate;
4. one exact endpoint polish removes that uncertainty and selects the same
   physical action as the warm root.

This is not evidence of multiple physical roots, reaction-basis inconsistency,
or Schur failure. The corrected cold and committed warm endpoints agree well
inside the frozen thresholds.

## Next plan

The result authorizes only a definitions-only same-history equivalence-policy
manifest. That prospective policy should:

1. leave the production `1e-10` step-acceptance gate unchanged;
2. apply only to nonpropagating same-history equivalence controls;
3. require both compared roots to reach a tighter diagnostic residual, proposed
   as `1e-12`, before applying the `1e-8` state/action comparison;
4. permit at most one exact endpoint-polish assembly/correction when an
   already accepted control root fails action equivalence;
5. rerun every physical and ledger audit after polishing;
6. forbid the polished control from entering trajectory history;
7. preserve the historical parent failure;
8. authorize no held-out continuation or timestep study until the policy is
   prospectively certified and the primary evidence is re-aggregated.

## Canonical evidence

```text
results/canonical/
causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_
wp10c9d6c7c3b5c4f24e14n/
```

The package contains the replayed endpoints, exact matrix and correction,
corrected residual and reaction action, complete audit, provenance, summary,
metrics, catalog records, and closing SHA256 checksums. Focused tests pass
`9/9`.
