# Fixed-Q exact endpoint diagnostic manifest

Work package: `WP10c9d6c7c3b5c4f24e14g`

This definitions-only package freezes one nonpropagating diagnosis at the
saved rejected `warm_1` endpoint. It preserves the parent classification
`bounded_continuation_failed` and authorizes at most one exact complete
bordered Jacobian assembly and one exact Newton correction.

The residual must first reproduce bitwise at
`5.708109263036221e-9`. The corrected candidate must retain the unchanged
`1e-10` root tolerance and every fixed-Q, storage, reconstruction, reaction,
conditioning, physical, primitive-change, and excision gate. Neither the
rejected endpoint nor the corrected diagnostic candidate may enter BDF
history or define a continuation state.

Only a positive classification
`stale_carried_matrix_refresh_trigger_diagnosed` may authorize a subsequent
definitions-only warm-policy manifest. No warm-policy execution, full primary
retry, held-out continuation, operational-timestep search, microburst, fast
averaging, or reduced evolution is authorized here.
