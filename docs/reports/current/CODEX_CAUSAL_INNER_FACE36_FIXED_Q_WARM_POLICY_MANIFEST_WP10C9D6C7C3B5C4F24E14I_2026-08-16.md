# Fixed-Q warm-policy manifest

Work package: `WP10c9d6c7c3b5c4f24e14i`

This definitions-only package freezes one `warm_1` solver-policy certificate
from the hash-locked accepted `cold_1` checkpoint. The carried matrix is used
at iteration zero. No exact matrix is forced initially, and at most one exact
assembly is allowed. The primary trigger is iteration 6 of the unchanged
eight-iteration budget; four failed relative backtracks are the secondary
trigger.

If the warm root passes every unchanged gate, one same-history cold control
must be solved from the identical state, BDF history, target, scales, and
predictor. Endpoint state and physical reaction action must agree within
`1e-8`, and the warm checkpoint must roundtrip bitwise.

The cost gate is a warm/cold wall-time ratio no greater than 0.75. A zero-
refresh warm root is not required. This package authorizes only one warm root
and its nonpropagating cold control; it does not authorize a full continuation
retry, held-out continuation, timestep search, microburst, fast averaging, or
reduced slow evolution.
