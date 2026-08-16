# Fixed-Q exact endpoint diagnostic

Work package: `WP10c9d6c7c3b5c4f24e14h`

## Classification

`stale_carried_matrix_refresh_trigger_diagnosed`

The one-correction, nonpropagating endpoint diagnostic passed. The parent
`bounded_continuation_failed` classification is preserved; this result
diagnoses that failure and does not retroactively convert it into a pass.

## Result

The committed rejected residual reproduced at
`5.708109263036221e-9`. One exact complete bordered Jacobian was assembled at
that endpoint. Its full Newton correction was admissible with alpha 1 and
reduced the maximum scaled residual to `6.919126059694603e-13`.

Every unchanged audit passed:

- fixed-Q relative defect: `2.86049e-16`;
- storage parity: `2.36675e-14`;
- reaction ledger: `1.88944e-16`;
- constraint-action ledger: `1.33229e-16`;
- Schur rank 3 and condition number `3.43514e4`;
- reconstruction factors exactly 1;
- maximum H/R `0.0978375`;
- minimum scattering optical depth `19.2543`;
- zero incoming excision characteristics.

The exact and carried corrections had cosine `0.884714` (angle
`0.484915 rad`), and the exact correction norm was `2.05175` times the carried
correction norm. The carried and exact matrix actions differed by order unity
on the decisive correction directions even though the full-matrix relative
Frobenius defect was only `2.77e-5`. This confirms that a global matrix norm
would have hidden the root-relevant staleness.

No continuation state was constructed, no candidate entered history, and no
physical trajectory time was added.

## Authorization

The result authorizes only a definitions-only prospective warm-policy
manifest. The policy may retain the carried matrix at iteration zero and use
at most one exact assembly, triggered primarily at iteration reserve
`maximum_iterations - 2` and secondarily after four failed relative
backtracks. The root tolerance, eight-iteration budget, and all physical gates
remain unchanged.

No warm-policy root has yet been executed by this package. No full primary
retry, held-out continuation, operational-timestep search, microburst, fast
averaging, or reduced slow evolution is authorized.
